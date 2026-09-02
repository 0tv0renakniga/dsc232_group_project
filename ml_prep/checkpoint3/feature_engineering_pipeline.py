# =====================================================================
# feature_engineering_pipeline.py: Feature engineering ONLY.
#
# Reads raw Parquet data, constructs labels, engineers all features,
# computes sample weights, and writes the unified dataset to
# data/ml_ready_unified/.  Model training is handled by separate
# scripts (ml_pipeline_xgb.py, ml_pipeline_classic.py, etc.).
# =====================================================================

from __future__ import annotations

import os
import math
from functools import reduce

# ---------------------------------------------------------------------
# PySpark Core
# ---------------------------------------------------------------------
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------
from pipeline_config import (
    DATA_ROOT, INPUT_PATH, OUTPUT_PATH, OUTPUT_DIR, scratch_dir,
    MODE, FLUSH_PARTITIONS,
    LABEL_COL, WEIGHT_COL,
    TRAIN_MIN_MONTH_IDX, TRAIN_MAX_MONTH_IDX,
    REGION_FILES, SPARSE_SAMPLE, REGION_BOUNDS, REGION_WEIGHTS,
    get_spark,
)
# ---------------------------------------------------------------------


# =====================================================================
# 1. CONFIG (FE-specific)
# =====================================================================

# ---------------------------------------------------------------------
# Feature engineering source
# "from_raw"      = build from scratch starting from DATA_ROOT
# "from_ml_ready" = read from INPUT_PATH (base features already built)
# ---------------------------------------------------------------------
FEATURE_SOURCE = "from_raw"
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Label construction mode
# "dual_sensor"   = GRACE + ICESat-2 agreement (strict, very sparse)
# "grace_anomaly" = GRACE-only 25th-pctl flag (~25% positive, backup)
# ---------------------------------------------------------------------
LABEL_MODE = "dual_sensor"
# ---------------------------------------------------------------------
# =====================================================================


# =====================================================================
# 2. RAW DATA LOADING
# =====================================================================

def _assign_region_from_bounds(df):
    """
    Assign regional_subset_id from EPSG:3031 bounding boxes.

        note: pixels outside all six boxes are tagged 'other'
              and dropped.
    """

    region_expr = F.lit("other")
    for name, b in reversed(list(REGION_BOUNDS.items())):
        region_expr = F.when(
            (F.col("x") >= b["x_min"]) & (F.col("x") <= b["x_max"])
            & (F.col("y") >= b["y_min"]) & (F.col("y") <= b["y_max"]),
            F.lit(name),
        ).otherwise(region_expr)

    df = df.withColumn("regional_subset_id", region_expr)
    df = df.filter(F.col("regional_subset_id") != "other")
    print("[load] Region assignment applied; rows outside bounding boxes dropped (lazy).")

    return df


def _derive_month_idx(df):
    """Derive month_idx from exact_time if not already present."""

    if "month_idx" in df.columns:
        return df
    if "exact_time" not in df.columns:
        raise ValueError("Neither 'month_idx' nor 'exact_time' found.")

    df = df.withColumn(
        "month_idx",
        (F.year("exact_time") * 12 + F.month("exact_time")).cast(IntegerType()),
    )
    print("[load] Derived month_idx from exact_time.")

    return df


def load_raw_data(spark):
    """Read raw Parquet data and tag each row with its region.

    local: reads sparse sample, tags all rows 'sample'.
    hpc:   reads 6 pre-split regional files.
    sdsc:  reads full-continent, assigns regions from bounding boxes.
    """

    if MODE == "local":
        path = os.path.join(DATA_ROOT, SPARSE_SAMPLE)
        print(f"[load] LOCAL: sparse sample: {path}")
        df = spark.read.parquet(path).withColumn(
            "regional_subset_id", F.lit("sample")
        )
        return _derive_month_idx(df)

    if MODE == "hpc":
        frames = []
        for region_name, filename in REGION_FILES.items():
            path = os.path.join(DATA_ROOT, filename)
            print(f"[load] Reading {region_name}: {path}")
            rdf = spark.read.parquet(path).withColumn(
                "regional_subset_id", F.lit(region_name)
            )
            frames.append(rdf)
        df = reduce(DataFrame.unionByName, frames)
        df = _derive_month_idx(df)
        print(f"[load] Union complete: {len(frames)} regions.")
        return df

    # sdsc: full continent, assign regions from bounding boxes
    print(f"[load] SDSC: full continent: {DATA_ROOT}")
    df = spark.read.parquet(DATA_ROOT)
    df = _derive_month_idx(df)

    return _assign_region_from_bounds(df)

# =====================================================================


# =====================================================================
# 3. LABEL CONSTRUCTION
# =====================================================================

def build_label(df):
    """
    Construct basal_loss_agreement and immediately purge lwe_fused.

        note: uses groupBy + broadcast-join instead of window aggs to
              avoid OOM-inducing shuffle-sorts on large mascon
              partitions.
    """

    mascon_p25 = (
        df.groupBy("mascon_id", "month_idx")
        .agg(F.percentile_approx("lwe_fused", 0.25).alias("_grace_p25"))
    )
    df = df.join(F.broadcast(mascon_p25), on=["mascon_id", "month_idx"], how="left")
    grace_flag = F.coalesce(F.col("lwe_fused") < F.col("_grace_p25"), F.lit(False))

    if LABEL_MODE == "grace_anomaly":
        df = df.withColumn(LABEL_COL, grace_flag.cast(IntegerType()))

    else:  # dual_sensor
        pixel_stats = (
            df.groupBy("x", "y")
            .agg(
                F.avg("delta_h").alias("_px_mean_dh"),
                F.stddev("delta_h").alias("_px_std_dh"),
            )
        )
        df = df.join(pixel_stats, on=["x", "y"], how="left")
        icesat_flag = F.coalesce(
            F.col("delta_h") < (F.col("_px_mean_dh") - F.col("_px_std_dh")),
            F.lit(False),
        )
        df = df.withColumn(LABEL_COL, (grace_flag & icesat_flag).cast(IntegerType()))
        df = df.drop("_px_mean_dh", "_px_std_dh")

    # LEAKAGE FIREWALL: ensure targets are not used as features
    df = df.drop("_grace_p25", "lwe_fused")
    assert "lwe_fused" not in df.columns, "LEAKAGE BUG: lwe_fused survived!"
    print(f"[label] label_mode={LABEL_MODE!r}. lwe_fused confirmed absent.")

    return df

# =====================================================================


# =====================================================================
# 4. BASE FEATURE ENGINEERING
# =====================================================================

def assign_regions(df):
    """
    Lazy null-filter + bed_below_sea_level.

        note: no eager .count() action.
    """

    df = df.filter(F.col("regional_subset_id").isNotNull())
    df = df.withColumn("bed_below_sea_level", (F.col("bed") < 0).cast(IntegerType()))
    print("[features] bed_below_sea_level added (region nulls filtered lazily).")

    return df


def add_static_features(df):
    """Static geometry interactions: no shuffles."""

    df = (
        df
        .withColumn("draft_x_thermal_access",
                     F.col("ice_draft") / (F.col("dist_to_ocean") + F.lit(1.0)))
        .withColumn("grounding_line_vulnerability",
                     F.col("thickness") / (F.col("dist_to_grounding_line") + F.lit(1.0)))
        .withColumn("retrograde_flag",
                     (F.col("bed_slope") < 0).cast(IntegerType()))
    )
    print("[features] 3 static interaction features added.")

    return df


def add_dynamic_features(df):
    """Expanding-window pixel statistics and lagged surface slope."""

    pixel_time_w = (
        Window.partitionBy("x", "y")
        .orderBy("month_idx")
        .rowsBetween(Window.unboundedPreceding, 0)
    )
    pixel_lag_w = Window.partitionBy("x", "y").orderBy("month_idx")

    df = (
        df
        .withColumn("pixel_mean_delta_h", F.avg("delta_h").over(pixel_time_w))
        .withColumn("delta_h_deviation",
                     F.col("delta_h") - F.col("pixel_mean_delta_h"))
        .withColumn("surface_slope_prev",
                     F.lag("surface_slope", 1).over(pixel_lag_w))
        .withColumn("surface_slope_change",
                     F.col("surface_slope") - F.col("surface_slope_prev"))
    ).drop("surface_slope_prev")
    print("[features] 3 dynamic features added.")

    return df


def add_ocean_features(df):
    """Ocean-ice interactions + regional thermal anomaly.

        note: regional aggregate uses groupBy + broadcast-join (~216
              rows) instead of a Window that would shuffle millions
              of rows per partition.
    """

    df = (
        df
        .withColumn("thermal_driving_x_draft",
                     F.col("t_star_mo") * F.col("ice_draft"))
        .withColumn("thermal_anomaly",
                     F.col("t_star_mo") - F.col("t_star_quarterly_avg"))
        .withColumn("salinity_stratification_proxy",
                     F.col("so_mo") * F.col("clamped_depth"))
        .withColumn("lwe_trend",
                     F.col("lwe_mo") - F.col("lwe_quarterly_avg"))
    )

    region_month_agg = (
        df.groupBy("regional_subset_id", "month_idx")
        .agg(F.avg("t_star_mo").alias("regional_t_star_climatology"))
    )
    df = df.join(
        F.broadcast(region_month_agg),
        on=["regional_subset_id", "month_idx"],
        how="left",
    )
    df = df.withColumn(
        "regional_t_star_anomaly",
        F.col("t_star_mo") - F.col("regional_t_star_climatology"),
    )

    print("[features] 6 ocean interaction features added.")

    return df


def add_context_features(df):
    """
    Cyclical month encoding, mascon aggregates, regional percentile.

        note: all group-level aggregations use groupBy + broadcast-join
              instead of Window functions to avoid shuffle-sort
              memory pressure.
    """

    two_pi_12 = 2.0 * math.pi / 12.0

    df = (
        df
        .withColumn("month_of_year", F.col("month_idx") % 12)
        .withColumn("sin_month",
                     F.sin(F.col("month_of_year").cast(FloatType()) * F.lit(two_pi_12)))
        .withColumn("cos_month",
                     F.cos(F.col("month_of_year").cast(FloatType()) * F.lit(two_pi_12)))
        .drop("month_of_year")
    )

    mascon_agg = (
        df.groupBy("mascon_id", "month_idx")
        .agg(
            F.avg("delta_h").alias("mascon_mean_delta_h"),
            F.avg("t_star_mo").alias("mascon_mean_t_star"),
        )
    )
    df = df.join(F.broadcast(mascon_agg), on=["mascon_id", "month_idx"], how="left")

    region_agg = (
        df.groupBy("regional_subset_id", "month_idx")
        .agg(
            F.avg("delta_h").alias("_reg_avg_dh"),
            F.stddev("delta_h").alias("_reg_std_dh"),
            F.avg("lwe_mo").alias("regional_lwe_mean"),
        )
    )
    df = df.join(
        F.broadcast(region_agg),
        on=["regional_subset_id", "month_idx"],
        how="left",
    )
    df = df.withColumn(
        "regional_delta_h_percentile",
        (F.col("delta_h") - F.col("_reg_avg_dh"))
        / (F.col("_reg_std_dh") + F.lit(1e-8)),
    )
    df = df.drop("_reg_avg_dh", "_reg_std_dh")

    print("[features] 6 context features added.")

    return df


def add_sample_weights(df):
    """
    Regional importance x class balance weights, normalised on
    training partition.
    """

    # Regional importance
    rw_expr = F.lit(1.0)
    for region, weight in REGION_WEIGHTS.items():
        rw_expr = F.when(
            F.col("regional_subset_id") == region, F.lit(weight)
        ).otherwise(rw_expr)
    df = df.withColumn("regional_weight", rw_expr)

    # Class balance (training-only stats)
    train_slice = df.filter(
        F.col("month_idx").between(TRAIN_MIN_MONTH_IDX, TRAIN_MAX_MONTH_IDX)
    )
    class_counts = (
        train_slice.groupBy("regional_subset_id")
        .agg(
            F.sum(F.when(F.col(LABEL_COL) == 0, 1).otherwise(0)).alias("neg_count"),
            F.sum(F.when(F.col(LABEL_COL) == 1, 1).otherwise(0)).alias("pos_count"),
        )
        .withColumn(
            "class_ratio",
            F.when(F.col("pos_count") > 0, F.col("neg_count") / F.col("pos_count"))
            .otherwise(F.lit(1.0)),
        )
        .select("regional_subset_id", "class_ratio")
    )
    df = df.join(F.broadcast(class_counts), on="regional_subset_id", how="left")
    df = df.withColumn(
        "class_balance_weight",
        F.when(F.col(LABEL_COL) == 1, F.col("class_ratio")).otherwise(F.lit(1.0)),
    )
    df = df.withColumn("raw_weight", F.col("regional_weight") * F.col("class_balance_weight"))

    train_mean = (
        df.filter(F.col("month_idx").between(TRAIN_MIN_MONTH_IDX, TRAIN_MAX_MONTH_IDX))
        .agg(F.avg("raw_weight"))
        .collect()[0][0]
    )
    print(f"[weights] Training mean raw weight = {train_mean:.6f}")

    df = df.withColumn(WEIGHT_COL, F.col("raw_weight") / F.lit(train_mean))
    df = df.drop("regional_weight", "class_ratio", "class_balance_weight", "raw_weight")
    print("[weights] Sample weights computed and normalised.")

    return df

# =====================================================================


# =====================================================================
# 5. ADDITIONAL FEATURES: Model 2/3 extras, applied to all models
# =====================================================================

def add_temporal_memory_features(df):
    """
    Six-month rolling averages and rate-of-change (Model 2 features).
    """

    w6 = (
        Window.partitionBy("x", "y")
        .orderBy("month_idx")
        .rowsBetween(-5, 0)
    )
    w_lag = Window.partitionBy("x", "y").orderBy("month_idx")

    for col_name in ["t_star_mo", "lwe_mo", "delta_h"]:
        if col_name not in df.columns:
            continue
        avg_col  = f"{col_name.replace('_mo', '')}_6mo_avg"
        rate_col = f"{col_name.replace('_mo', '')}_rate"
        df = df.withColumn(avg_col, F.avg(col_name).over(w6))
        df = df.withColumn(
            rate_col,
            F.col(col_name) - F.coalesce(F.col(avg_col), F.col(col_name)),
        )

    for col_name, out_name in [("t_star_mo", "t_star_mom_change"),
                                ("delta_h", "delta_h_mom_change")]:
        if col_name in df.columns:
            lag_col = F.lag(col_name, 1).over(w_lag)
            df = df.withColumn(
                out_name,
                F.col(col_name) - F.coalesce(lag_col, F.col(col_name)),
            )

    print("[features] Temporal memory features added.")

    return df


def add_trajectory_features(df):
    """Momentum and acceleration features (Model 3 features)."""

    w_lag = Window.partitionBy("x", "y").orderBy("month_idx")
    w12   = Window.partitionBy("x", "y").orderBy("month_idx").rowsBetween(-11, 0)
    w6    = Window.partitionBy("x", "y").orderBy("month_idx").rowsBetween(-5, 0)

    if "delta_h" in df.columns:
        lag1 = F.lag("delta_h", 1).over(w_lag)
        lag2 = F.lag("delta_h", 2).over(w_lag)
        df = df.withColumn(
            "delta_h_momentum",
            F.col("delta_h") - F.coalesce(lag1, F.col("delta_h")),
        )
        df = df.withColumn(
            "delta_h_acceleration",
            F.col("delta_h") - F.lit(2.0) * F.coalesce(lag1, F.col("delta_h"))
            + F.coalesce(lag2, F.col("delta_h")),
        )
        df = df.withColumn(
            "delta_h_deseason",
            F.col("delta_h") - F.avg("delta_h").over(w12),
        )

    if "t_star_mo" in df.columns:
        t_lag = F.lag("t_star_mo", 1).over(w_lag)
        df = df.withColumn(
            "t_star_momentum",
            F.col("t_star_mo") - F.coalesce(t_lag, F.col("t_star_mo")),
        )
        df = df.withColumn(
            "t_star_sustained_anomaly",
            F.avg("t_star_mo").over(w6) - F.avg("t_star_mo").over(w12),
        )

    if "lwe_mo" in df.columns:
        l_lag = F.lag("lwe_mo", 1).over(w_lag)
        df = df.withColumn(
            "lwe_momentum",
            F.col("lwe_mo") - F.coalesce(l_lag, F.col("lwe_mo")),
        )
        df = df.withColumn(
            "lwe_sustained_trend",
            F.avg("lwe_mo").over(w6) - F.avg("lwe_mo").over(w12),
        )

    print("[features] Trajectory features added.")

    return df


def add_physics_interactions(df):
    """Physics interaction features (Model 2 features)."""

    cols = set(df.columns)

    if {"thetao_mo", "ice_draft", "dist_to_ocean"} <= cols:
        df = df.withColumn(
            "ocean_heat_content_proxy",
            F.col("thetao_mo") * F.abs(F.col("ice_draft"))
            / (F.col("dist_to_ocean") + F.lit(1.0)),
        )
    if {"ice_draft", "thickness"} <= cols:
        df = df.withColumn(
            "draft_ratio",
            F.abs(F.col("ice_draft")) / (F.col("thickness") + F.lit(1.0)),
        )
    if {"t_star_mo", "dist_to_grounding_line"} <= cols:
        df = df.withColumn(
            "thermal_x_gl_proximity",
            F.col("t_star_mo") / (F.col("dist_to_grounding_line") + F.lit(1.0)),
        )
    if {"thetao_mo", "t_f_mo"} <= cols:
        df = df.withColumn(
            "freezing_departure",
            F.col("thetao_mo") - F.col("t_f_mo"),
        )
    if {"bed_slope", "bed"} <= cols:
        df = df.withColumn(
            "bed_geometry_risk",
            F.col("bed_slope") * F.least(F.col("bed"), F.lit(0.0)),
        )
    if {"delta_h", "ice_area"} <= cols:
        df = df.withColumn(
            "mass_flux_proxy",
            F.col("delta_h") * F.col("ice_area"),
        )

    print("[features] Physics interaction features added.")

    return df


def add_region_integer_encoding(df):
    """Integer-encode region for native categorical support."""

    region_map = {
        "amundsen_sea": 0, "antarctic_peninsula": 1, "lambert_amery": 2,
        "ronne": 3, "ross": 4, "totten_and_aurora": 5, "sample": 6,
    }
    expr = F.lit(6)
    for name, idx in region_map.items():
        expr = F.when(
            F.col("regional_subset_id") == name, F.lit(idx)
        ).otherwise(expr)
    df = df.withColumn("region_cat_idx", expr.cast(IntegerType()))
    print("[features] Region integer encoding added.")

    return df

# =====================================================================


# =====================================================================
# 6. UNIFIED FEATURE ENGINEERING RUNNER
# =====================================================================

def _flush(df, spark, tag):
    """
    Write to Parquet and read back to truncate Spark's DAG lineage.

        note: forces materialisation of all pending transformations,
              releases the intermediate shuffle state held by the
              ShuffleExternalSorter, and gives the next stage a clean,
              shallow execution plan.
    """

    path = os.path.join(scratch_dir, f"_flush_{tag}")
    df.write.mode("overwrite").parquet(path)
    print(f"[flush] Lineage truncated -> {tag}")

    return spark.read.parquet(path)


def _build_region_features(rdf, spark, tag="all"):
    """Run all feature engineering on a single-region DataFrame."""

    rdf = build_label(rdf)
    rdf = assign_regions(rdf)
    rdf = add_static_features(rdf)

    rdf = rdf.repartition(FLUSH_PARTITIONS, "x", "y")
    rdf = add_dynamic_features(rdf)
    rdf = add_ocean_features(rdf)
    rdf = add_context_features(rdf)

    rdf = _flush(rdf, spark, f"phase1_{tag}")

    rdf = rdf.repartition(FLUSH_PARTITIONS, "x", "y")
    rdf = add_temporal_memory_features(rdf)
    rdf = add_trajectory_features(rdf)
    rdf = add_physics_interactions(rdf)
    rdf = add_region_integer_encoding(rdf)

    return rdf


def run_feature_engineering(spark):
    """
    Build all features.

        note: for large datasets (HPC/SDSC), processes each region
              independently so window-operation shuffles never exceed
              memory.

        note: sample weights are computed globally after reuniting
              the regions.
    """

    is_large = MODE in ("hpc", "sdsc")

    if FEATURE_SOURCE == "from_raw":
        print(f"\n[features] ===== BUILDING FROM RAW DATA =====")
        raw = load_raw_data(spark)

        if is_large:
            regions = list(REGION_BOUNDS.keys())

            # Stage A: save raw data split by region (narrow ops only)
            raw_path = os.path.join(scratch_dir, "raw_by_region")
            print(f"[features] Saving raw data by region -> {raw_path}")
            (raw.write.mode("overwrite")
                .partitionBy("regional_subset_id")
                .parquet(raw_path))
            print("[features] Raw partition write complete.")

            # Stage B: feature-engineer each region separately
            for region in regions:
                print(f"\n[features] ═══ {region.upper()} ═══")
                rdf = spark.read.parquet(
                    os.path.join(raw_path, f"regional_subset_id={region}"))
                rdf = rdf.withColumn("regional_subset_id", F.lit(region))

                rdf = _build_region_features(rdf, spark, tag=region)

                out = os.path.join(scratch_dir, f"features_{region}")
                rdf.write.mode("overwrite").parquet(out)
                print(f"[features] {region} complete -> {out}")

            # Stage C: union regions + global sample weights
            print(f"\n[features] ═══ UNION + SAMPLE WEIGHTS ═══")
            dfs = [spark.read.parquet(
                       os.path.join(scratch_dir, f"features_{r}"))
                   for r in regions]
            df = reduce(DataFrame.unionByName, dfs)
            df = add_sample_weights(df)

        else:
            df = _build_region_features(raw, spark)
            df = add_sample_weights(df)

    else:
        print(f"[features] Reading pre-built features from: {INPUT_PATH}")
        df = spark.read.parquet(INPUT_PATH)
        print(f"[features] Loaded {len(df.columns)} columns.")
        df = df.filter(F.col(LABEL_COL).isNotNull() & F.col(WEIGHT_COL).isNotNull())

    assert "lwe_fused" not in df.columns, "LEAKAGE: lwe_fused in output!"

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    df.write.mode("overwrite").parquet(OUTPUT_PATH)
    print(f"[features] Done -> {OUTPUT_PATH}")

# =====================================================================


# =====================================================================
# 7. Feature Engineering Pipeline 
# =====================================================================

def fe_pipeline():
    """Feature engineering pipeline entry point."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    spark = get_spark()

    try:
        if not os.path.exists(OUTPUT_PATH) or not os.listdir(OUTPUT_PATH):
            print(f"\n{'='*60}\n  FEATURE ENGINEERING\n{'='*60}")
            run_feature_engineering(spark)
        else:
            print(f"[skip] Features exist at {OUTPUT_PATH}")

    finally:
        spark.stop()

# =====================================================================


if __name__ == "__main__":
    fe_pipeline()
