"""
Antarctic Ice Mass Loss — PySpark Feature Engineering Pipeline
==============================================================

Transforms regional Parquet subsets (or a single full-continent
Hive-partitioned Parquet) into a model-ready dataset with:

  • Binary label ``basal_loss_agreement`` with strict leakage prevention.
    Supports two labelling strategies via ``--label-mode``:
      - ``dual_sensor``   GRACE + ICESat-2 agreement (default)
      - ``grace_anomaly``  GRACE-only 25th-percentile flag (backup)
  • 20+ physically-motivated features spanning static geometry,
    dynamic ice state, ocean thermal forcing, and spatio-temporal
    context.
  • Two-component sample weights (regional importance x class balance)
    normalised on the training partition.

Execution
---------
Local smoke-test (sparse sample):
    python feature_engineering_pipeline.py --mode local --label-mode grace_anomaly

HPC with pre-split regional files:
    spark-submit feature_engineering_pipeline.py --mode hpc \
        --data-root /path/to/regional_files \
        --output-path /path/to/ml_ready

SDSC Expanse with full-continent Parquet:
    spark-submit feature_engineering_pipeline.py --mode sdsc \
        --data-root /expanse/lustre/.../antarctica_sparse_features.parquet \
        --output-path /expanse/lustre/.../ml_ready
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from functools import reduce
from typing import Dict

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType

# ── Split constants ──────────────────────────────────────────────────
TRAIN_MAX_MONTH_IDX = 24264   # Dec 2021: 2021*12 + 12
VAL_MAX_MONTH_IDX = 24276     # Dec 2022

REGION_FILES: Dict[str, str] = {
    "amundsen_sea":        "ml_subset_amundsen_sea.parquet",
    "antarctic_peninsula": "ml_subset_antarctic_peninsula.parquet",
    "lambert_amery":       "ml_subset_lambert_amery.parquet",
    "ronne":               "ml_subset_ronne.parquet",
    "ross":                "ml_subset_ross.parquet",
    "totten_and_aurora":   "ml_subset_totten_and_aurora.parquet",
}

SPARSE_SAMPLE = "antarctica_sparse_features_sample.parquet"

# EPSG:3031 bounding boxes for assigning regions from the full-continent
# Parquet when pre-split regional files are not available (sdsc mode).
REGION_BOUNDS: Dict[str, Dict[str, float]] = {
    "amundsen_sea": {
        "x_min": -1800000.0, "x_max": -1100000.0,
        "y_min": -800000.0,  "y_max": -100000.0,
    },
    "antarctic_peninsula": {
        "x_min": -2500000.0, "x_max": -1000000.0,
        "y_min": 500000.0,   "y_max": 2000000.0,
    },
    "lambert_amery": {
        "x_min": 1500000.0,  "x_max": 2500000.0,
        "y_min": 0.0,        "y_max": 1200000.0,
    },
    "ronne": {
        "x_min": -1500000.0, "x_max": -500000.0,
        "y_min": 0.0,        "y_max": 1500000.0,
    },
    "ross": {
        "x_min": -500000.0,  "x_max": 1000000.0,
        "y_min": -1500000.0, "y_max": -500000.0,
    },
    "totten_and_aurora": {
        "x_min": 1800000.0,  "x_max": 2600000.0,
        "y_min": -1500000.0, "y_max": -500000.0,
    },
}

REGION_WEIGHTS: Dict[str, float] = {
    "amundsen_sea":        2.0,
    "totten_and_aurora":   2.0,
    "antarctic_peninsula": 1.5,
    "lambert_amery":       1.0,
    "ross":                0.7,
    "ronne":               0.7,
}


# =====================================================================
# Spark Session Factory
# =====================================================================

def get_spark(mode: str) -> SparkSession:
    """Return a SparkSession configured for *mode* in {"local", "hpc", "sdsc"}."""

    scratch = os.environ.get("TMPDIR", os.path.join(os.getcwd(), "spark_scratch"))

    shared = {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.advisoryPartitionSizeInBytes": "128m",
        "spark.sql.parquet.filterPushdown": "true",
        "spark.sql.parquet.mergeSchema": "false",
        "spark.sql.sources.parallelPartitionDiscovery.parallelism": "64",
        "spark.network.timeout": "1200s",
        "spark.local.dir": scratch,
    }

    builder = SparkSession.builder.appName("AntarcticFeatureEngineering")

    if mode == "local":
        builder = (
            builder
            .master("local[4]")
            .config("spark.driver.memory", "8g")
            .config("spark.sql.shuffle.partitions", "8")
        )
    elif mode == "hpc":
        builder = (
            builder
            .config("spark.executor.instances", "6")
            .config("spark.executor.cores", "5")
            .config("spark.executor.memory", "19g")
            .config("spark.driver.memory", "10g")
            .config("spark.driver.maxResultSize", "4g")
            .config("spark.sql.shuffle.partitions", "300")
        )
    elif mode == "sdsc":
        # Full 1.38B-row continent file needs more shuffle partitions:
        # ~166 GB uncompressed / 128 MB target = ~1300; 2000 for headroom.
        builder = (
            builder
            .config("spark.executor.instances", "6")
            .config("spark.executor.cores", "5")
            .config("spark.executor.memory", "19g")
            .config("spark.driver.memory", "10g")
            .config("spark.driver.maxResultSize", "4g")
            .config("spark.sql.shuffle.partitions", "2000")
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'local', 'hpc', or 'sdsc'.")

    for k, v in shared.items():
        builder = builder.config(k, v)

    return builder.getOrCreate()


# =====================================================================
# Step 0 — Data Ingestion
# =====================================================================

def _assign_region_from_bounds(df: DataFrame) -> DataFrame:
    """Assign ``regional_subset_id`` from EPSG:3031 bounding boxes.

    Pixels outside all six boxes are tagged ``"other"`` and dropped —
    they sit on the interior plateau with no ocean interaction signal.
    """
    region_expr = F.lit("other")
    for name, b in reversed(list(REGION_BOUNDS.items())):
        region_expr = F.when(
            (F.col("x") >= b["x_min"]) & (F.col("x") <= b["x_max"])
            & (F.col("y") >= b["y_min"]) & (F.col("y") <= b["y_max"]),
            F.lit(name),
        ).otherwise(region_expr)

    df = df.withColumn("regional_subset_id", region_expr)

    n_before = df.count()
    df = df.filter(F.col("regional_subset_id") != "other")
    n_after = df.count()
    print(f"[load_data] Region assignment: kept {n_after:,} of {n_before:,} rows "
          f"({n_before - n_after:,} outside all bounding boxes dropped).")
    return df


def _derive_month_idx(df: DataFrame) -> DataFrame:
    """
    Derive ``month_idx = year * 12 + month`` from ``exact_time`` when
    the parquet files carry a timestamp instead of the pre-computed key.
    If ``month_idx`` already exists, this is a no-op.
    """
    if "month_idx" in df.columns:
        return df

    if "exact_time" not in df.columns:
        raise ValueError(
            "Neither 'month_idx' nor 'exact_time' found in schema — "
            f"available columns: {df.columns}"
        )

    df = df.withColumn(
        "month_idx",
        (F.year("exact_time") * 12 + F.month("exact_time")).cast(IntegerType()),
    )
    print("[load_data] Derived month_idx from exact_time.")
    return df


def load_data(spark: SparkSession, data_root: str, mode: str) -> DataFrame:
    """Read Parquet data and tag each row with its region.

    Modes
    -----
    local : reads the small sparse sample; tags all rows ``"sample"``.
    hpc   : reads 6 pre-split regional files; tags from filename.
    sdsc  : reads the single full-continent Hive-partitioned Parquet;
            assigns regions from EPSG:3031 bounding boxes and drops
            pixels that fall outside all six boxes.
    """

    if mode == "local":
        path = os.path.join(data_root, SPARSE_SAMPLE)
        print(f"[load_data] LOCAL mode — reading sparse sample: {path}")
        df = spark.read.parquet(path).withColumn(
            "regional_subset_id", F.lit("sample")
        )
        return _derive_month_idx(df)

    if mode == "sdsc":
        print(f"[load_data] SDSC mode — reading full continent: {data_root}")
        df = spark.read.parquet(data_root)
        df = _derive_month_idx(df)
        df = _assign_region_from_bounds(df)
        return df

    # hpc — six pre-split regional files
    frames = []
    for region_name, filename in REGION_FILES.items():
        path = os.path.join(data_root, filename)
        print(f"[load_data] Reading {region_name}: {path}")
        region_df = spark.read.parquet(path).withColumn(
            "regional_subset_id", F.lit(region_name)
        )
        frames.append(region_df)

    df_raw = reduce(DataFrame.unionByName, frames)
    df_raw = _derive_month_idx(df_raw)
    print(f"[load_data] Union complete — {len(frames)} regions loaded.")
    return df_raw


# =====================================================================
# Step 1 — Label Construction
# =====================================================================

def build_label(df: DataFrame, label_mode: str = "dual_sensor") -> DataFrame:
    """
    Construct ``basal_loss_agreement`` and immediately purge ``lwe_fused``.

    Parameters
    ----------
    label_mode : {"dual_sensor", "grace_anomaly"}
        dual_sensor   — GRACE 25-pctl flag AND ICESat-2 deviation flag.
        grace_anomaly — GRACE 25-pctl flag only (~25 % positive by
                        construction).  Use as backup when the dual-
                        sensor intersection is too sparse.
    """

    mascon_window = Window.partitionBy("mascon_id", "month_idx")
    grace_p25 = F.percentile_approx("lwe_fused", 0.25).over(mascon_window)
    # coalesce handles NULLs from rows where lwe_fused is missing
    grace_flag = F.coalesce(F.col("lwe_fused") < grace_p25, F.lit(False))

    if label_mode == "grace_anomaly":
        df = df.withColumn(
            "basal_loss_agreement", grace_flag.cast(IntegerType()),
        )
    else:
        pixel_window = Window.partitionBy("x", "y")
        pixel_mean_dh = F.avg("delta_h").over(pixel_window)
        pixel_std_dh = F.stddev("delta_h").over(pixel_window)
        icesat_flag = F.coalesce(
            F.col("delta_h") < (pixel_mean_dh - pixel_std_dh), F.lit(False),
        )

        df = df.withColumn(
            "basal_loss_agreement",
            (grace_flag & icesat_flag).cast(IntegerType()),
        )

    # ── LEAKAGE FIREWALL ────────────────────────────────────────────
    df = df.drop("lwe_fused")
    assert "lwe_fused" not in df.columns, (
        "LEAKAGE BUG: lwe_fused survived the drop — aborting."
    )

    print(f"[build_label] label_mode={label_mode!r}. lwe_fused confirmed absent.")
    return df


# =====================================================================
# Step 2 — Regional Structure
# =====================================================================

def assign_regions(df: DataFrame) -> DataFrame:
    """
    Validate regional_subset_id completeness and derive
    `bed_below_sea_level`.

    Target-encoding of regional_subset_id is deferred to the model
    pipeline to avoid leaking validation/test label statistics into
    training features.
    """

    null_count = df.filter(F.col("regional_subset_id").isNull()).count()
    if null_count > 0:
        raise ValueError(
            f"regional_subset_id has {null_count} NULL rows — "
            "data loading is broken."
        )

    df = df.withColumn(
        "bed_below_sea_level",
        (F.col("bed") < 0).cast(IntegerType()),
    )

    print("[assign_regions] Regional structure validated; bed_below_sea_level added.")
    return df


# =====================================================================
# Step 3 — Static Geometry Features
# =====================================================================

def add_static_features(df: DataFrame) -> DataFrame:
    """Pure column arithmetic — no shuffles."""

    df = (
        df
        .withColumn(
            "draft_x_thermal_access",
            F.col("ice_draft") / (F.col("dist_to_ocean") + F.lit(1.0)),
        )
        .withColumn(
            "grounding_line_vulnerability",
            F.col("thickness") / (F.col("dist_to_grounding_line") + F.lit(1.0)),
        )
        .withColumn(
            "retrograde_flag",
            (F.col("bed_slope") < 0).cast(IntegerType()),
        )
    )

    print("[add_static_features] 3 static interaction features added.")
    return df


# =====================================================================
# Step 4 — Dynamic State Features
# =====================================================================

def add_dynamic_features(df: DataFrame) -> DataFrame:
    """
    Expanding-window pixel statistics and lagged surface slope.

    pixel_time_window uses unboundedPreceding..currentRow so each row
    sees only past-and-present observations — no future leakage.
    """

    # Expanding window: all months up to the current row for each pixel
    pixel_time_window = (
        Window.partitionBy("x", "y")
        .orderBy("month_idx")
        .rowsBetween(Window.unboundedPreceding, 0)
    )

    # Ordered window for lag (defaults to unbounded frame which is fine
    # for lag/lead — Spark ignores the frame spec for those functions)
    pixel_lag_window = Window.partitionBy("x", "y").orderBy("month_idx")

    df = (
        df
        .withColumn(
            "pixel_mean_delta_h",
            F.avg("delta_h").over(pixel_time_window),
        )
        .withColumn(
            "delta_h_deviation",
            F.col("delta_h") - F.col("pixel_mean_delta_h"),
        )
        .withColumn(
            "surface_slope_prev",
            F.lag("surface_slope", 1).over(pixel_lag_window),
        )
        .withColumn(
            "surface_slope_change",
            F.col("surface_slope") - F.col("surface_slope_prev"),
        )
    )

    df = df.drop("surface_slope_prev")

    print("[add_dynamic_features] 3 dynamic features added.")
    return df


# =====================================================================
# Step 5 — Ocean Interaction Features
# =====================================================================

def add_ocean_features(df: DataFrame) -> DataFrame:
    """
    Column arithmetic for ocean–ice interactions plus a regional
    monthly climatology window for thermal anomaly decomposition.
    """

    # Regional monthly window: one partition per (region, month) so the
    # climatology reflects the spatial average across all pixels in that
    # region for a given month — required for anomaly decomposition.
    region_month_window = Window.partitionBy("regional_subset_id", "month_idx")

    df = (
        df
        .withColumn(
            "thermal_driving_x_draft",
            F.col("t_star_mo") * F.col("ice_draft"),
        )
        .withColumn(
            "thermal_anomaly",
            F.col("t_star_mo") - F.col("t_star_quarterly_avg"),
        )
        .withColumn(
            "salinity_stratification_proxy",
            F.col("so_mo") * F.col("clamped_depth"),
        )
        .withColumn(
            "lwe_trend",
            F.col("lwe_mo") - F.col("lwe_quarterly_avg"),
        )
        .withColumn(
            "regional_t_star_climatology",
            F.avg("t_star_mo").over(region_month_window),
        )
        .withColumn(
            "regional_t_star_anomaly",
            F.col("t_star_mo") - F.col("regional_t_star_climatology"),
        )
    )

    print("[add_ocean_features] 6 ocean interaction features added.")
    return df


# =====================================================================
# Step 6 — Temporal and Spatial Context
# =====================================================================

def add_context_features(df: DataFrame) -> DataFrame:
    """
    Cyclical month encoding (no ordinal leakage between Dec/Jan),
    mascon-level aggregates, and regional percentile ranking.
    """

    two_pi_over_12 = 2.0 * math.pi / 12.0

    # Mascon × month window: aggregates all pixels sharing a mascon in
    # the same month — summarises the local neighbourhood's state.
    mascon_window_time = Window.partitionBy("mascon_id", "month_idx")

    # Reuse the same (region, month) window from Step 5 for percentile
    region_month_window = Window.partitionBy("regional_subset_id", "month_idx")

    df = (
        df
        # Seasonal encoding
        .withColumn("month_of_year", F.col("month_idx") % 12)
        .withColumn(
            "sin_month",
            F.sin(F.col("month_of_year").cast(FloatType()) * F.lit(two_pi_over_12)),
        )
        .withColumn(
            "cos_month",
            F.cos(F.col("month_of_year").cast(FloatType()) * F.lit(two_pi_over_12)),
        )
        .drop("month_of_year")
        # Mascon context
        .withColumn(
            "mascon_mean_delta_h",
            F.avg("delta_h").over(mascon_window_time),
        )
        .withColumn(
            "mascon_mean_t_star",
            F.avg("t_star_mo").over(mascon_window_time),
        )
        # Regional context
        .withColumn(
            "regional_delta_h_percentile",
            F.percent_rank().over(
                region_month_window.orderBy("delta_h")
            ),
        )
        .withColumn(
            "regional_lwe_mean",
            F.avg("lwe_mo").over(region_month_window),
        )
    )

    print("[add_context_features] 6 context features added.")
    return df


# =====================================================================
# Step 7 — Sample Weights
# =====================================================================

def add_sample_weights(df: DataFrame) -> DataFrame:
    """
    Two-component weight: regional importance × within-region class
    balance, normalised so the mean weight on training rows ≈ 1.0.

    The class-balance ratio is computed on training rows only
    (month_idx <= TRAIN_MAX_MONTH_IDX) then broadcast-joined back
    to all rows, keeping the test partition uncontaminated.
    """

    # ── Component 1: regional importance (hard-coded domain priors) ──
    region_weight_expr = F.lit(1.0)
    for region, weight in REGION_WEIGHTS.items():
        region_weight_expr = F.when(
            F.col("regional_subset_id") == region, F.lit(weight)
        ).otherwise(region_weight_expr)

    df = df.withColumn("regional_weight", region_weight_expr)

    # ── Component 2: within-region class balance ─────────────────────
    train_slice = df.filter(F.col("month_idx") <= TRAIN_MAX_MONTH_IDX)

    class_counts = (
        train_slice
        .groupBy("regional_subset_id")
        .agg(
            F.sum(
                F.when(F.col("basal_loss_agreement") == 0, 1).otherwise(0)
            ).alias("neg_count"),
            F.sum(
                F.when(F.col("basal_loss_agreement") == 1, 1).otherwise(0)
            ).alias("pos_count"),
        )
        .withColumn(
            "class_ratio",
            F.when(F.col("pos_count") > 0, F.col("neg_count") / F.col("pos_count"))
            .otherwise(F.lit(1.0)),
        )
        .select("regional_subset_id", "class_ratio")
    )

    df = df.join(
        F.broadcast(class_counts),
        on="regional_subset_id",
        how="left",
    )

    df = df.withColumn(
        "class_balance_weight",
        F.when(F.col("basal_loss_agreement") == 1, F.col("class_ratio"))
        .otherwise(F.lit(1.0)),
    )

    # ── Raw weight and normalisation ─────────────────────────────────
    df = df.withColumn(
        "raw_weight",
        F.col("regional_weight") * F.col("class_balance_weight"),
    )

    train_mean_weight: float = (
        df.filter(F.col("month_idx") <= TRAIN_MAX_MONTH_IDX)
        .agg(F.avg("raw_weight"))
        .collect()[0][0]
    )

    print(f"[add_sample_weights] Training mean raw weight = {train_mean_weight:.6f}")

    df = df.withColumn(
        "weightCol",
        F.col("raw_weight") / F.lit(train_mean_weight),
    )

    df = df.drop("regional_weight", "class_ratio", "class_balance_weight", "raw_weight")

    print("[add_sample_weights] Sample weights computed and normalised.")
    return df


# =====================================================================
# Step 8 — I/O Optimisation & Write
# =====================================================================

def write_output(df: DataFrame, output_path: str) -> None:
    """
    Repartition by (month_idx, mascon_id) for downstream windowed reads,
    then write Parquet partitioned by month_idx.

    Partition count is derived from the data: target ~128 MB per
    partition assuming ~120 bytes/row.  Clamped to [4, 2000].
    """

    # ── Final leakage assertion ──────────────────────────────────────
    assert "lwe_fused" not in df.columns, (
        "LEAKAGE BUG: lwe_fused found in output DataFrame."
    )

    row_count = df.count()
    bytes_per_row = 120
    target_partition_bytes = 128 * 1024 * 1024  # 128 MB
    n_parts = max(4, min(2000, int(row_count * bytes_per_row / target_partition_bytes)))
    print(f"[write_output] {row_count:,} rows -> {n_parts} output partitions")

    df = df.repartition(n_parts, "month_idx", "mascon_id")

    print(f"[write_output] Writing to {output_path} ...")
    df.write.mode("overwrite").partitionBy("month_idx").parquet(output_path)

    # ── Summary statistics (single aggregation pass) ─────────────────
    written = spark_read = df.sparkSession.read.parquet(output_path)
    summary = (
        written
        .groupBy("regional_subset_id")
        .agg(
            F.count("*").alias("n_rows"),
            F.avg(F.col("basal_loss_agreement").cast(FloatType())).alias("pos_rate"),
        )
        .collect()
    )

    total_rows = sum(r["n_rows"] for r in summary)
    total_pos = sum(r["n_rows"] * r["pos_rate"] for r in summary)
    global_pos_rate = total_pos / total_rows if total_rows > 0 else 0.0

    train_mean_w = (
        written
        .filter(F.col("month_idx") <= TRAIN_MAX_MONTH_IDX)
        .agg(F.avg("weightCol"))
        .collect()[0][0]
    )

    columns_present = sorted(written.columns)

    print("\n" + "=" * 72)
    print("  OUTPUT SUMMARY")
    print("=" * 72)
    print(f"  Total rows       : {total_rows:,}")
    print(f"  Global pos rate  : {global_pos_rate:.4f}")
    print(f"  Training mean wt : {train_mean_w:.6f}")
    print()
    print("  Regional breakdown:")
    for row in summary:
        print(f"    {row['regional_subset_id']:25s}  "
              f"pos_rate={row['pos_rate']:.4f}  "
              f"n={row['n_rows']:>12,}")
    print()
    print(f"  Columns ({len(columns_present)}):")
    for c in columns_present:
        print(f"    {c}")
    print("=" * 72)

    assert "lwe_fused" not in columns_present, "LEAKAGE: lwe_fused in output!"
    assert abs(train_mean_w - 1.0) < 0.01, (
        f"Weight normalisation failed: mean = {train_mean_w:.6f}, expected ≈ 1.0"
    )

    print("[write_output] All assertions passed. Pipeline complete.")


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Antarctic ice mass loss — feature engineering pipeline"
    )
    parser.add_argument(
        "--data-root",
        default=os.path.join(os.getcwd(), "data"),
        help="Root directory containing Parquet files.  For --mode sdsc "
             "this should be the path to the single Hive-partitioned "
             "antarctica_sparse_features.parquet directory.",
    )
    parser.add_argument(
        "--output-path",
        default=os.path.join(os.getcwd(), "ml_ready"),
        help="Output directory for model-ready Parquet.",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "hpc", "sdsc"],
        default="local",
        help="Spark configuration profile.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["dual_sensor", "grace_anomaly"],
        default="dual_sensor",
        help="Label strategy.  'grace_anomaly' guarantees ~25%% positive "
             "rate and works even on sparse data.",
    )
    args = parser.parse_args()

    spark = get_spark(args.mode)

    try:
        df = load_data(spark, args.data_root, args.mode)

        df = build_label(df, args.label_mode)  # Step 1
        df = assign_regions(df)                # Step 2
        df = add_static_features(df)           # Step 3
        df = add_dynamic_features(df)          # Step 4
        df = add_ocean_features(df)            # Step 5
        df = add_context_features(df)          # Step 6
        df = add_sample_weights(df)            # Step 7

        write_output(df, args.output_path)  # Step 8
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
