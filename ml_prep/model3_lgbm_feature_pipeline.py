"""
Antarctic Ice Mass Loss — Model 3 LightGBM Feature Engineering Pipeline
=======================================================================

Distinct from Models 1 and 2 in three key ways:
  1. Normalizer (L2 norm) on ocean feature block
  2. PCA on correlated ocean variables to reduce multicollinearity
  3. Extended temporal trajectory features (momentum, acceleration)
  4. Regional_subset_id as integer-coded (LightGBM native categorical)

Reads from the base ``ml_ready/`` Parquet output of the shared
``feature_engineering_pipeline.py`` and produces ``ml_ready_lgbm/``.

Execution
---------
Local:
    python model3_lgbm_feature_pipeline.py --mode local

SDSC Expanse:
    spark-submit model3_lgbm_feature_pipeline.py --mode sdsc \
        --input-path /expanse/lustre/.../ml_ready \
        --output-path /expanse/lustre/.../ml_ready_lgbm
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Imputer,
    Normalizer,
    PCA,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType

# ── Constants ────────────────────────────────────────────────────────
TRAIN_MAX_MONTH_IDX = 24264   # Dec 2021
VAL_MAX_MONTH_IDX = 24276     # Dec 2022

LABEL_COL = "basal_loss_agreement"
WEIGHT_COL = "weightCol"

# Ocean-domain columns subject to PCA dimensionality reduction
OCEAN_COLS = [
    "thetao_mo", "t_star_mo", "so_mo", "t_f_mo",
    "t_star_quarterly_avg", "t_star_quarterly_std",
    "thetao_quarterly_avg", "thetao_quarterly_std",
]

# PCA output dimensionality (8 -> 4 components)
PCA_K = 4


# =====================================================================
# Spark Session Factory
# =====================================================================

def get_spark(mode: str) -> SparkSession:
    """Return a SparkSession configured for *mode*."""

    scratch = os.environ.get("TMPDIR", os.path.join(os.getcwd(), "spark_scratch"))

    shared = {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.parquet.filterPushdown": "true",
        "spark.sql.parquet.mergeSchema": "false",
        "spark.network.timeout": "1200s",
        "spark.local.dir": scratch,
    }

    builder = SparkSession.builder.appName("AntarcticModel3_LGBM_Features")

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
# Step 1 — Temporal Trajectory Features
# =====================================================================

def add_temporal_trajectories(df: DataFrame) -> DataFrame:
    """
    Compute temporal trajectory features that capture the *direction*
    and *acceleration* of change, not just the instantaneous state.

    This is the core distinction of Model 3's feature philosophy:
    - Model 1 uses static engineered features
    - Model 2 uses 6-month rolling averages
    - Model 3 uses momentum (1st derivative) and acceleration (2nd derivative)
    """

    w_lag = Window.partitionBy("x", "y").orderBy("month_idx")

    # 3-month momentum: weighted recent change (more weight to recent)
    w3 = (
        Window.partitionBy("x", "y")
        .orderBy("month_idx")
        .rowsBetween(-2, 0)
    )

    # 6-month window for trend context
    w6 = (
        Window.partitionBy("x", "y")
        .orderBy("month_idx")
        .rowsBetween(-5, 0)
    )

    # 12-month window for annual cycle removal
    w12 = (
        Window.partitionBy("x", "y")
        .orderBy("month_idx")
        .rowsBetween(-11, 0)
    )

    df = (
        df
        # ── Delta-h trajectory ─────────────────────────────────────
        .withColumn("delta_h_lag1", F.lag("delta_h", 1).over(w_lag))
        .withColumn("delta_h_lag2", F.lag("delta_h", 2).over(w_lag))
        # Momentum: first-difference
        .withColumn(
            "delta_h_momentum",
            F.col("delta_h") - F.coalesce(F.col("delta_h_lag1"), F.col("delta_h")),
        )
        # Acceleration: second-difference
        .withColumn(
            "delta_h_acceleration",
            F.col("delta_h")
            - F.lit(2.0) * F.coalesce(F.col("delta_h_lag1"), F.col("delta_h"))
            + F.coalesce(F.col("delta_h_lag2"), F.col("delta_h")),
        )
        # 3-month trend slope (linear-fit approx via weighted mean)
        .withColumn("delta_h_3mo_trend", F.avg("delta_h").over(w3) - F.col("delta_h"))
        # Annual-deseasonalized anomaly
        .withColumn("delta_h_12mo_avg", F.avg("delta_h").over(w12))
        .withColumn(
            "delta_h_deseason",
            F.col("delta_h") - F.col("delta_h_12mo_avg"),
        )
        # ── Thermal driving trajectory ─────────────────────────────
        .withColumn("t_star_lag1", F.lag("t_star_mo", 1).over(w_lag))
        .withColumn(
            "t_star_momentum",
            F.col("t_star_mo") - F.coalesce(F.col("t_star_lag1"), F.col("t_star_mo")),
        )
        .withColumn("t_star_6mo_avg", F.avg("t_star_mo").over(w6))
        .withColumn(
            "t_star_sustained_anomaly",
            F.col("t_star_6mo_avg") - F.avg("t_star_mo").over(w12),
        )
        # ── LWE trajectory ────────────────────────────────────────
        .withColumn("lwe_lag1", F.lag("lwe_mo", 1).over(w_lag))
        .withColumn(
            "lwe_momentum",
            F.col("lwe_mo") - F.coalesce(F.col("lwe_lag1"), F.col("lwe_mo")),
        )
        .withColumn("lwe_6mo_avg", F.avg("lwe_mo").over(w6))
        .withColumn(
            "lwe_sustained_trend",
            F.col("lwe_6mo_avg") - F.avg("lwe_mo").over(w12),
        )
    )

    # Clean up lag columns
    df = df.drop(
        "delta_h_lag1", "delta_h_lag2", "delta_h_12mo_avg",
        "t_star_lag1", "lwe_lag1",
    )

    print("[add_temporal_trajectories] 10 trajectory features added.")
    return df


# =====================================================================
# Step 2 — Region Integer Encoding (for LightGBM Categoricals)
# =====================================================================

def add_region_integer_encoding(df: DataFrame) -> DataFrame:
    """
    Encode regional_subset_id as a contiguous integer for LightGBM's
    native categorical handling.  Unlike Model 1/2 which use OHE,
    LightGBM learns per-category split thresholds directly.
    """

    region_map = {
        "amundsen_sea": 0,
        "antarctic_peninsula": 1,
        "lambert_amery": 2,
        "ronne": 3,
        "ross": 4,
        "totten_and_aurora": 5,
        "sample": 6,  # for local mode sparse sample
    }

    mapping_expr = F.lit(6)  # default for unknown
    for name, idx in region_map.items():
        mapping_expr = F.when(
            F.col("regional_subset_id") == name, F.lit(idx),
        ).otherwise(mapping_expr)

    df = df.withColumn("region_cat_idx", mapping_expr.cast(IntegerType()))

    print("[add_region_integer_encoding] region_cat_idx added.")
    return df


# =====================================================================
# Step 3 — Ocean PCA and Normalizer (MLlib Pipeline)
# =====================================================================

def build_ocean_pca_pipeline(available_cols: List[str]) -> Pipeline:
    """
    Build an MLlib pipeline that:
      1. Imputes ocean columns
      2. Assembles them into a vector
      3. Normalizes using L2 norm (distinct from Model 1/2)
      4. Applies PCA to reduce multicollinearity

    Returns the pipeline (must be fit on training data only).
    """

    ocean_available = [c for c in OCEAN_COLS if c in available_cols]
    imputed_ocean = [f"{c}_oc_imp" for c in ocean_available]

    imputer = Imputer(
        strategy="median",
        inputCols=ocean_available,
        outputCols=imputed_ocean,
    )

    assembler = VectorAssembler(
        inputCols=imputed_ocean,
        outputCol="ocean_vec_raw",
        handleInvalid="skip",
    )

    normalizer = Normalizer(
        inputCol="ocean_vec_raw",
        outputCol="ocean_vec_norm",
        p=2.0,  # L2 norm
    )

    k = min(PCA_K, len(ocean_available))
    pca = PCA(
        k=k,
        inputCol="ocean_vec_norm",
        outputCol="ocean_pca",
    )

    return Pipeline(stages=[imputer, assembler, normalizer, pca])


# =====================================================================
# Step 4 — Write Output
# =====================================================================

def write_output(df: DataFrame, output_path: str) -> None:
    """Validate no leakage, repartition, and write model-3-ready Parquet."""

    assert "lwe_fused" not in df.columns, (
        "LEAKAGE BUG: lwe_fused found in output DataFrame."
    )

    row_count = df.count()
    bytes_per_row = 150
    target_partition_bytes = 128 * 1024 * 1024
    n_parts = max(4, min(2000, int(row_count * bytes_per_row / target_partition_bytes)))
    print(f"[write_output] {row_count:,} rows -> {n_parts} output partitions")

    df = df.repartition(n_parts, "month_idx", "mascon_id")

    print(f"[write_output] Writing to {output_path} ...")
    df.write.mode("overwrite").parquet(output_path)

    # Validation
    written = df.sparkSession.read.parquet(output_path)
    cols = sorted(written.columns)

    print("\n" + "=" * 72)
    print("  MODEL 3 FEATURE OUTPUT SUMMARY")
    print("=" * 72)
    print(f"  Total rows    : {written.count():,}")
    print(f"  Total columns : {len(cols)}")
    print(f"  New features  : delta_h_momentum, delta_h_acceleration,")
    print(f"                  delta_h_3mo_trend, delta_h_deseason,")
    print(f"                  t_star_momentum, t_star_6mo_avg,")
    print(f"                  t_star_sustained_anomaly, lwe_momentum,")
    print(f"                  lwe_6mo_avg, lwe_sustained_trend,")
    print(f"                  region_cat_idx, ocean_pca[0..{PCA_K-1}]")
    print("=" * 72)

    assert "lwe_fused" not in cols, "LEAKAGE: lwe_fused in output!"
    assert LABEL_COL in cols, f"Missing label column: {LABEL_COL}"
    assert WEIGHT_COL in cols, f"Missing weight column: {WEIGHT_COL}"

    print("[write_output] All assertions passed. Model 3 feature pipeline complete.")


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Antarctic ice mass loss — Model 3 LightGBM feature pipeline"
    )
    parser.add_argument(
        "--input-path",
        default=os.path.join(os.getcwd(), "ml_ready"),
        help="Path to base feature-engineered Parquet.",
    )
    parser.add_argument(
        "--output-path",
        default=os.path.join(os.getcwd(), "ml_ready_lgbm"),
        help="Output directory for Model 3 feature-engineered Parquet.",
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

        # Step 1: Temporal trajectory features
        df = add_temporal_trajectories(df)

        # Step 2: Region integer encoding for LightGBM categoricals
        df = add_region_integer_encoding(df)

        # Step 3: Ocean PCA pipeline (fit on training slice only)
        print("[main] Fitting Ocean PCA pipeline on training data...")
        train_slice = df.filter(F.col("month_idx") <= TRAIN_MAX_MONTH_IDX)
        ocean_pipeline = build_ocean_pca_pipeline(df.columns)
        ocean_model = ocean_pipeline.fit(train_slice)

        # Transform all data using the training-fitted pipeline
        df = ocean_model.transform(df)

        # Convert PCA vector (ML UDT) to native array, then extract components
        df = df.withColumn("ocean_pca_arr", vector_to_array("ocean_pca"))
        for i in range(PCA_K):
            df = df.withColumn(
                f"ocean_pca_{i}",
                F.col("ocean_pca_arr").getItem(i).cast(FloatType()),
            )

        # Drop intermediate vector columns to save space
        df = df.drop(
            "ocean_vec_raw", "ocean_vec_norm", "ocean_pca", "ocean_pca_arr",
            *[f"{c}_oc_imp" for c in OCEAN_COLS if f"{c}_oc_imp" in df.columns],
        )

        # Step 4: Write
        write_output(df, args.output_path)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
