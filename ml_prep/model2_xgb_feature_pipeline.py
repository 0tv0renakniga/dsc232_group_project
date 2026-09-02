"""
Antarctic Ice Mass Loss — Model 2 XGBoost Feature Engineering Pipeline
======================================================================

Distinct from Model 1's pipeline in three key ways:
  1. Bucketizer for dist_to_grounding_line → categorical proximity bands
  2. 6-month rolling temporal memory features (t_star, lwe, delta_h)
  3. Physics-interaction hand-crafted columns instead of PolynomialExpansion

Reads from the base ``ml_ready/`` Parquet output of the shared
``feature_engineering_pipeline.py`` and produces ``ml_ready_xgb/``.

Execution
---------
Local:
    python model2_xgb_feature_pipeline.py --mode local

SDSC Expanse:
    spark-submit model2_xgb_feature_pipeline.py --mode sdsc \
        --input-path /expanse/lustre/.../ml_ready \
        --output-path /expanse/lustre/.../ml_ready_xgb
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType

# ── Constants ────────────────────────────────────────────────────────
TRAIN_MAX_MONTH_IDX = 24264   # Dec 2021
VAL_MAX_MONTH_IDX = 24276     # Dec 2022

LABEL_COL = "basal_loss_agreement"
WEIGHT_COL = "weightCol"


# =====================================================================
# Spark Session Factory
# =====================================================================

def get_spark(mode: str) -> SparkSession:
    """Return a SparkSession configured for *mode* in {"local", "hpc", "sdsc"}."""

    scratch = os.environ.get("TMPDIR", os.path.join(os.getcwd(), "spark_scratch"))

    shared = {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.parquet.filterPushdown": "true",
        "spark.sql.parquet.mergeSchema": "false",
        "spark.network.timeout": "1200s",
        "spark.local.dir": scratch,
    }

    builder = SparkSession.builder.appName("AntarcticModel2_XGB_Features")

    if mode == "local":
        builder = (
            builder
            .master("local[4]")
            .config("spark.driver.memory", "8g")
            .config("spark.sql.shuffle.partitions", "8")
        )
    else:
        builder = (
            builder
            .config("spark.executor.instances", "6")
            .config("spark.executor.cores", "5")
            .config("spark.executor.memory", "19g")
            .config("spark.driver.memory", "10g")
            .config("spark.driver.maxResultSize", "4g")
            .config("spark.sql.shuffle.partitions", "300")
        )

    for k, v in shared.items():
        builder = builder.config(k, v)

    return builder.getOrCreate()


# =====================================================================
# Step 1 — Extended Temporal Memory Features
# =====================================================================

def add_temporal_memory(df: DataFrame) -> DataFrame:
    """
    Compute 6-month rolling averages and rate-of-change for key signals.

    These features give the model temporal trajectory information that
    the quarterly rolling stats (3-month) in the base pipeline do not
    capture.  A 6-month window catches sustained forcing episodes —
    e.g. a multi-month warm water intrusion — rather than only the
    current season's state.

    Window: partitionBy(x, y).orderBy(month_idx) with 5 preceding rows
    (= 6 months inclusive).  Uses unboundedPreceding fallback for pixels
    with fewer than 6 observations so no rows are dropped.
    """

    w6 = (
        Window.partitionBy("x", "y")
        .orderBy("month_idx")
        .rowsBetween(-5, 0)
    )

    # Lag window for rate-of-change
    w_lag = Window.partitionBy("x", "y").orderBy("month_idx")

    df = (
        df
        # --- 6-month rolling averages ---
        .withColumn("t_star_6mo_avg", F.avg("t_star_mo").over(w6))
        .withColumn("lwe_6mo_avg", F.avg("lwe_mo").over(w6))
        .withColumn("delta_h_6mo_avg", F.avg("delta_h").over(w6))
        # --- Rate of change (current vs 6-month avg) ---
        .withColumn(
            "t_star_rate",
            F.col("t_star_mo") - F.col("t_star_6mo_avg"),
        )
        .withColumn(
            "lwe_acceleration",
            F.col("lwe_mo") - F.col("lwe_6mo_avg"),
        )
        .withColumn(
            "delta_h_rate",
            F.col("delta_h") - F.col("delta_h_6mo_avg"),
        )
        # --- Lagged values for month-over-month change detection ---
        .withColumn("t_star_prev", F.lag("t_star_mo", 1).over(w_lag))
        .withColumn(
            "t_star_mom_change",
            F.col("t_star_mo") - F.col("t_star_prev"),
        )
        .withColumn("delta_h_prev", F.lag("delta_h", 1).over(w_lag))
        .withColumn(
            "delta_h_mom_change",
            F.col("delta_h") - F.col("delta_h_prev"),
        )
    )

    # Clean up intermediate lag columns
    df = df.drop("t_star_prev", "delta_h_prev")

    print("[add_temporal_memory] 8 temporal memory features added.")
    return df


# =====================================================================
# Step 2 — Physics Interaction Features
# =====================================================================

def add_physics_interactions(df: DataFrame) -> DataFrame:
    """
    Hand-crafted interaction terms encoding known glaciological physics.

    These replace Model 1's PolynomialExpansion with targeted,
    physically-motivated combinations.
    """

    df = (
        df
        # Ocean heat content proxy: temperature × depth × proximity
        .withColumn(
            "ocean_heat_content_proxy",
            F.col("thetao_mo") * F.abs(F.col("ice_draft"))
            / (F.col("dist_to_ocean") + F.lit(1.0)),
        )
        # Draft ratio: how much of the ice column is below sea level
        .withColumn(
            "draft_ratio",
            F.when(F.col("thickness") > 0,
                   F.abs(F.col("ice_draft")) / F.col("thickness"))
            .otherwise(F.lit(0.0)),
        )
        # Thermal driving × distance to grounding line interaction
        # High thermal driving near grounding line = extreme risk
        .withColumn(
            "thermal_x_gl_proximity",
            F.col("t_star_mo") / (F.col("dist_to_grounding_line") + F.lit(1.0)),
        )
        # Freezing point departure: how far above freezing
        .withColumn(
            "freezing_departure",
            F.col("thetao_mo") - F.col("t_f_mo"),
        )
        # Bed geometry risk: slope × depth below sea level interaction
        .withColumn(
            "bed_geometry_risk",
            F.col("bed_slope") * F.least(F.col("bed"), F.lit(0.0)),
        )
        # Mass flux proxy: elevation change × ice area
        .withColumn(
            "mass_flux_proxy",
            F.col("delta_h") * F.col("ice_area"),
        )
    )

    print("[add_physics_interactions] 6 physics interaction features added.")
    return df


# =====================================================================
# Step 3 — Grounding Line Proximity Buckets
# =====================================================================

def add_grounding_line_buckets(df: DataFrame) -> DataFrame:
    """
    Discretize dist_to_grounding_line into categorical proximity bands
    via boundary thresholds (analogous to Bucketizer).

    Bands:
        0: < 5 km     (critical zone)
        1: 5-20 km    (near zone)
        2: 20-50 km   (transition zone)
        3: 50-100 km  (moderate distance)
        4: > 100 km   (far zone)

    The Bucketizer transformer is applied in the training pipeline;
    here we just prepare the raw distances.  But we also create the
    bucketed column directly for use in feature engineering.
    """

    df = df.withColumn(
        "gl_proximity_bucket",
        F.when(F.col("dist_to_grounding_line") < 5000.0, F.lit(0))
        .when(F.col("dist_to_grounding_line") < 20000.0, F.lit(1))
        .when(F.col("dist_to_grounding_line") < 50000.0, F.lit(2))
        .when(F.col("dist_to_grounding_line") < 100000.0, F.lit(3))
        .otherwise(F.lit(4))
        .cast(IntegerType()),
    )

    print("[add_grounding_line_buckets] Grounding line proximity bucket added.")
    return df


# =====================================================================
# Step 4 — Final Leakage Audit and Write
# =====================================================================

def write_output(df: DataFrame, output_path: str) -> None:
    """
    Validate no leakage, repartition, and write model-2-ready Parquet.
    """

    # ── Leakage firewall ─────────────────────────────────────────────
    assert "lwe_fused" not in df.columns, (
        "LEAKAGE BUG: lwe_fused found in output DataFrame."
    )

    row_count = df.count()
    bytes_per_row = 140  # slightly larger than model 1 due to extra features
    target_partition_bytes = 128 * 1024 * 1024
    n_parts = max(4, min(2000, int(row_count * bytes_per_row / target_partition_bytes)))
    print(f"[write_output] {row_count:,} rows -> {n_parts} output partitions")

    df = df.repartition(n_parts, "month_idx", "mascon_id")

    print(f"[write_output] Writing to {output_path} ...")
    df.write.mode("overwrite").parquet(output_path)

    # ── Validation ───────────────────────────────────────────────────
    written = df.sparkSession.read.parquet(output_path)
    cols = sorted(written.columns)
    n_cols = len(cols)
    n_rows = written.count()

    print("\n" + "=" * 72)
    print("  MODEL 2 FEATURE OUTPUT SUMMARY")
    print("=" * 72)
    print(f"  Total rows    : {n_rows:,}")
    print(f"  Total columns : {n_cols}")
    print(f"  New features  : t_star_6mo_avg, lwe_6mo_avg, delta_h_6mo_avg,")
    print(f"                  t_star_rate, lwe_acceleration, delta_h_rate,")
    print(f"                  t_star_mom_change, delta_h_mom_change,")
    print(f"                  ocean_heat_content_proxy, draft_ratio,")
    print(f"                  thermal_x_gl_proximity, freezing_departure,")
    print(f"                  bed_geometry_risk, mass_flux_proxy,")
    print(f"                  gl_proximity_bucket")
    print("=" * 72)

    assert "lwe_fused" not in cols, "LEAKAGE: lwe_fused in output!"
    assert LABEL_COL in cols, f"Missing label column: {LABEL_COL}"
    assert WEIGHT_COL in cols, f"Missing weight column: {WEIGHT_COL}"

    print("[write_output] All assertions passed. Model 2 feature pipeline complete.")


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Antarctic ice mass loss — Model 2 XGB feature pipeline"
    )
    parser.add_argument(
        "--input-path",
        default=os.path.join(os.getcwd(), "ml_ready"),
        help="Path to base feature-engineered Parquet (from feature_engineering_pipeline.py).",
    )
    parser.add_argument(
        "--output-path",
        default=os.path.join(os.getcwd(), "ml_ready_xgb"),
        help="Output directory for Model 2 feature-engineered Parquet.",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "hpc", "sdsc"],
        default="local",
        help="Spark configuration profile.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    spark = get_spark(args.mode)

    try:
        print(f"[main] Reading base features from: {args.input_path}")
        df = spark.read.parquet(args.input_path)
        print(f"[main] Loaded {len(df.columns)} columns.")

        # Filter rows where label and weight are present
        df = df.filter(
            F.col(LABEL_COL).isNotNull() & F.col(WEIGHT_COL).isNotNull()
        )

        df = add_temporal_memory(df)        # Step 1
        df = add_physics_interactions(df)   # Step 2
        df = add_grounding_line_buckets(df) # Step 3

        write_output(df, args.output_path)  # Step 4
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
