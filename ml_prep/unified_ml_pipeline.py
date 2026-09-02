"""
Antarctic Ice Mass Loss: Unified ML Pipeline
=============================================

Single consolidated script for all models, designed for Jupyter notebook use.
Paste sections into cells or run as a script.

Models available:
    1. XGB_Baseline / XGB_Tuned          (SparkXGBClassifier)
    2. Stack_Baseline / Stack_Tuned       (RF+GBT -> XGBoost meta-learner)
    3. CorrStack_LR_Baseline / _Tuned    (RF+GBT -> LogisticRegression, OOF, pruned)
    4. Classic_DT / Classic_RF / Classic_GBT  (DT -> RF -> GBT progression
       with PolynomialExpansion + StandardScaler + Ocean PCA)

Key improvements over individual scripts:
    - Undersampling: keeps all positives, samples negatives to target ratio
    - PR-AUC metric alongside ROC-AUC (critical for rare positive class)
    - Single feature pipeline for all models
    - No argparse: configure via constants at top of file
"""

# =====================================================================
# 1. LIBRARIES
# =====================================================================

from __future__ import annotations

import os
import math
from functools import reduce
from typing import Dict, List, Tuple, Optional

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import (
    Bucketizer,
    Imputer,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    PCA,
    PolynomialExpansion,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType


# =====================================================================
# 2. CONFIG  (edit these for your environment)
# =====================================================================

# ── Paths ────────────────────────────────────────────────────────────
DATA_ROOT   = os.path.join(os.getcwd(), "data")            # raw Parquet root
INPUT_PATH  = os.path.join(os.getcwd(), "ml_ready")        # base features (if already built)
OUTPUT_PATH = os.path.join(os.getcwd(), "ml_ready_unified") # unified features
OUTPUT_DIR  = os.path.join(os.getcwd(), "unified_output")  # results

# ── Execution mode ───────────────────────────────────────────────────
MODE = "local"  # "local" | "hpc" | "sdsc"

# ── Feature engineering source ───────────────────────────────────────
# "from_ml_ready" = read from INPUT_PATH (base features already built)
# "from_raw"      = build from scratch starting from DATA_ROOT
FEATURE_SOURCE = "from_ml_ready"

# ── Label construction mode ──────────────────────────────────────────
# "dual_sensor"   = GRACE + ICESat-2 agreement (strict, very sparse)
# "grace_anomaly" = GRACE-only 25th-pctl flag (~25% positive, backup)
LABEL_MODE = "dual_sensor"

# ── Which model to train (change this to switch models) ──────────────
# Options: "xgb", "stack", "corrected_stack", "classic"
ACTIVE_MODEL = "xgb"

# ── Temporal split boundaries ────────────────────────────────────────
TRAIN_MAX_MONTH_IDX = 24264   # end Dec 2021
VAL_MAX_MONTH_IDX   = 24276   # end Dec 2022

# ── Class imbalance: undersampling config ────────────────────────────
UNDERSAMPLE_ENABLED  = True
UNDERSAMPLE_NEG_RATIO = 10   # keep 1:10 pos:neg ratio (set higher for less aggressive)

# ── Column names ─────────────────────────────────────────────────────
LABEL_COL      = "basal_loss_agreement"
WEIGHT_COL     = "weightCol"
PREDICTION_COL = "prediction"

KEY_COLS = ["x", "y", "month_idx", "mascon_id", "regional_subset_id"]

# ── Data loading constants ───────────────────────────────────────────
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

# ── XGBoost hyperparameters ──────────────────────────────────────────
XGB_CONFIGS = {
    "XGB_Baseline": dict(
        max_depth=4,
        n_estimators=50 if MODE == "local" else 100,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
    ),
    "XGB_Tuned": dict(
        max_depth=6 if MODE == "local" else 8,
        n_estimators=100 if MODE == "local" else 400,
        learning_rate=0.05 if MODE == "local" else 0.02,
        subsample=0.75,
        colsample_bytree=0.7,
        min_child_weight=15 if MODE == "local" else 20,
        reg_alpha=0.1,
        reg_lambda=1.0,
    ),
}

# ── Base learner hyperparameters (for stacking models) ───────────────
LOCAL = MODE == "local"
RF_CONFIG = dict(numTrees=20 if LOCAL else 100, maxDepth=6 if LOCAL else 10)
GBT_CONFIG = dict(maxIter=30 if LOCAL else 150, maxDepth=4 if LOCAL else 6,
                  stepSize=0.1 if LOCAL else 0.05)

# ── Corrected stack: meta-learner context features (pruned) ──────────
META_CONTEXT_COLS = [
    "region_cat_idx", "sin_month", "cos_month",
    "dist_to_grounding_line", "delta_h",
]

# ── Grounding line bucket splits (for XGBoost model) ─────────────────
GL_BUCKET_SPLITS = [float("-inf"), 5000.0, 20000.0, 50000.0, 100000.0, float("inf")]

# ── Ocean PCA config ─────────────────────────────────────────────────
OCEAN_COLS = [
    "thetao_mo", "t_star_mo", "so_mo", "t_f_mo",
    "t_star_quarterly_avg", "t_star_quarterly_std",
    "thetao_quarterly_avg", "thetao_quarterly_std",
]
PCA_K = 4

# ── PolynomialExpansion config (classic model) ──────────────────────
POLY_INPUT_COLS = ["t_star_mo", "ice_draft", "dist_to_grounding_line"]


# =====================================================================
# 3. SPARK SESSION
# =====================================================================

def get_spark() -> SparkSession:
    scratch = os.environ.get("TMPDIR", os.path.join(os.getcwd(), "spark_scratch"))

    shared = {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.parquet.filterPushdown": "true",
        "spark.sql.parquet.mergeSchema": "false",
        "spark.network.timeout": "1200s",
        "spark.local.dir": scratch,
        "spark.sql.debug.maxToStringFields": "2000",
    }

    builder = SparkSession.builder.appName("AntarcticUnifiedML")

    if MODE == "local":
        builder = (
            builder.master("local[4]")
            .config("spark.driver.memory", "8g")
            .config("spark.sql.shuffle.partitions", "8")
        )
    else:
        builder = (
            builder
            .config("spark.executor.instances", "6")
            .config("spark.executor.cores", "5")
            # Heap memory for Spark/JVM data processing
            .config("spark.executor.memory", "14g") 
            # Buffer for Python workers, GBT histograms, and XGBoost (Off-Heap)
            .config("spark.yarn.executor.memoryOverhead", "5g") 
            .config("spark.driver.memory", "10g")
            .config("spark.driver.maxResultSize", "4g")
            # Optimized for 30 total cores (6 instances * 5 cores)
            .config("spark.sql.shuffle.partitions", "300")
            .config("spark.network.timeout", "1200s")
        )
        """
        builder = (
            builder
            .config("spark.executor.instances", "6")
            .config("spark.executor.cores", "5")
            .config("spark.executor.memory", "19g")
            .config("spark.driver.memory", "10g")
            .config("spark.driver.maxResultSize", "4g")
            .config("spark.sql.shuffle.partitions", "300")
        )
        """

    for k, v in shared.items():
        builder = builder.config(k, v)

    return builder.getOrCreate()


# =====================================================================
# 4A. RAW DATA LOADING  (used when FEATURE_SOURCE == "from_raw")
# =====================================================================

def _assign_region_from_bounds(df: DataFrame) -> DataFrame:
    """Assign regional_subset_id from EPSG:3031 bounding boxes.
    Pixels outside all six boxes are tagged 'other' and dropped."""

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
    print(f"[load] Region assignment: kept {n_after:,}/{n_before:,} "
          f"({n_before - n_after:,} outside bounding boxes dropped).")
    return df


def _derive_month_idx(df: DataFrame) -> DataFrame:
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


def load_raw_data(spark: SparkSession) -> DataFrame:
    """Read raw Parquet data and tag each row with its region.

    local: reads sparse sample, tags all rows 'sample'.
    hpc:   reads 6 pre-split regional files.
    sdsc:  reads full-continent, assigns regions from bounding boxes.
    """

    if MODE == "local":
        path = os.path.join(DATA_ROOT, SPARSE_SAMPLE)
        print(f"[load] LOCAL — sparse sample: {path}")
        df = spark.read.parquet(path).withColumn(
            "regional_subset_id", F.lit("sample")
        )
        return _derive_month_idx(df)

    if MODE == "sdsc":
        print(f"[load] SDSC — full continent: {DATA_ROOT}")
        df = spark.read.parquet(DATA_ROOT)
        df = _derive_month_idx(df)
        return _assign_region_from_bounds(df)

    # hpc — six pre-split regional files
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
    print(f"[load] Union complete — {len(frames)} regions.")
    return df


# =====================================================================
# 4B. LABEL CONSTRUCTION  (used when FEATURE_SOURCE == "from_raw")
# =====================================================================

def build_label(df: DataFrame) -> DataFrame:
    """Construct basal_loss_agreement and immediately purge lwe_fused."""

    mascon_w = Window.partitionBy("mascon_id", "month_idx")
    grace_p25 = F.percentile_approx("lwe_fused", 0.25).over(mascon_w)
    grace_flag = F.coalesce(F.col("lwe_fused") < grace_p25, F.lit(False))

    if LABEL_MODE == "grace_anomaly":
        df = df.withColumn(LABEL_COL, grace_flag.cast(IntegerType()))
    else:  # dual_sensor
        pixel_w = Window.partitionBy("x", "y")
        pixel_mean_dh = F.avg("delta_h").over(pixel_w)
        pixel_std_dh  = F.stddev("delta_h").over(pixel_w)
        icesat_flag = F.coalesce(
            F.col("delta_h") < (pixel_mean_dh - pixel_std_dh), F.lit(False),
        )
        df = df.withColumn(LABEL_COL, (grace_flag & icesat_flag).cast(IntegerType()))

    # LEAKAGE FIREWALL
    df = df.drop("lwe_fused")
    assert "lwe_fused" not in df.columns, "LEAKAGE BUG: lwe_fused survived!"
    print(f"[label] label_mode={LABEL_MODE!r}. lwe_fused confirmed absent.")
    return df


# =====================================================================
# 4C. BASE FEATURE ENGINEERING  (used when FEATURE_SOURCE == "from_raw")
# =====================================================================

def assign_regions(df: DataFrame) -> DataFrame:
    """Validate regional_subset_id and add bed_below_sea_level."""
    null_count = df.filter(F.col("regional_subset_id").isNull()).count()
    if null_count > 0:
        raise ValueError(f"regional_subset_id has {null_count} NULLs.")
    df = df.withColumn("bed_below_sea_level", (F.col("bed") < 0).cast(IntegerType()))
    print("[features] Regional structure validated; bed_below_sea_level added.")
    return df


def add_static_features(df: DataFrame) -> DataFrame:
    """Static geometry interactions — no shuffles."""
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


def add_dynamic_features(df: DataFrame) -> DataFrame:
    """Expanding-window pixel statistics and lagged surface slope."""
    pixel_time_w = (
        Window.partitionBy("x", "y").orderBy("month_idx")
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


def add_ocean_features(df: DataFrame) -> DataFrame:
    """Ocean-ice interactions + regional thermal anomaly."""
    region_month_w = Window.partitionBy("regional_subset_id", "month_idx")
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
        .withColumn("regional_t_star_climatology",
                    F.avg("t_star_mo").over(region_month_w))
        .withColumn("regional_t_star_anomaly",
                    F.col("t_star_mo") - F.col("regional_t_star_climatology"))
    )
    print("[features] 6 ocean interaction features added.")
    return df


def add_context_features(df: DataFrame) -> DataFrame:
    """Cyclical month encoding, mascon aggregates, regional percentile."""
    two_pi_12 = 2.0 * math.pi / 12.0
    mascon_w = Window.partitionBy("mascon_id", "month_idx")
    region_month_w = Window.partitionBy("regional_subset_id", "month_idx")

    df = (
        df
        .withColumn("month_of_year", F.col("month_idx") % 12)
        .withColumn("sin_month",
                    F.sin(F.col("month_of_year").cast(FloatType()) * F.lit(two_pi_12)))
        .withColumn("cos_month",
                    F.cos(F.col("month_of_year").cast(FloatType()) * F.lit(two_pi_12)))
        .drop("month_of_year")
        .withColumn("mascon_mean_delta_h", F.avg("delta_h").over(mascon_w))
        .withColumn("mascon_mean_t_star", F.avg("t_star_mo").over(mascon_w))
        .withColumn("regional_delta_h_percentile",
                    F.percent_rank().over(region_month_w.orderBy("delta_h")))
        .withColumn("regional_lwe_mean", F.avg("lwe_mo").over(region_month_w))
    )
    print("[features] 6 context features added.")
    return df


def add_sample_weights(df: DataFrame) -> DataFrame:
    """Regional importance × class balance weights, normalised on training partition."""

    # Regional importance
    rw_expr = F.lit(1.0)
    for region, weight in REGION_WEIGHTS.items():
        rw_expr = F.when(F.col("regional_subset_id") == region, F.lit(weight)).otherwise(rw_expr)
    df = df.withColumn("regional_weight", rw_expr)

    # Class balance (training-only stats)
    train_slice = df.filter(F.col("month_idx") <= TRAIN_MAX_MONTH_IDX)
    class_counts = (
        train_slice.groupBy("regional_subset_id")
        .agg(
            F.sum(F.when(F.col(LABEL_COL) == 0, 1).otherwise(0)).alias("neg_count"),
            F.sum(F.when(F.col(LABEL_COL) == 1, 1).otherwise(0)).alias("pos_count"),
        )
        .withColumn("class_ratio",
                    F.when(F.col("pos_count") > 0, F.col("neg_count") / F.col("pos_count"))
                    .otherwise(F.lit(1.0)))
        .select("regional_subset_id", "class_ratio")
    )
    df = df.join(F.broadcast(class_counts), on="regional_subset_id", how="left")
    df = df.withColumn("class_balance_weight",
                       F.when(F.col(LABEL_COL) == 1, F.col("class_ratio")).otherwise(F.lit(1.0)))
    df = df.withColumn("raw_weight", F.col("regional_weight") * F.col("class_balance_weight"))

    train_mean = (
        df.filter(F.col("month_idx") <= TRAIN_MAX_MONTH_IDX)
        .agg(F.avg("raw_weight")).collect()[0][0]
    )
    print(f"[weights] Training mean raw weight = {train_mean:.6f}")

    df = df.withColumn(WEIGHT_COL, F.col("raw_weight") / F.lit(train_mean))
    df = df.drop("regional_weight", "class_ratio", "class_balance_weight", "raw_weight")
    print("[weights] Sample weights computed and normalised.")
    return df


# =====================================================================
# 4D. ADDITIONAL FEATURES  (Model 2/3 extras, applied to all models)
# =====================================================================

def add_temporal_memory_features(df: DataFrame) -> DataFrame:
    """6-month rolling averages and rate-of-change (Model 2 features)."""

    w6 = (
        Window.partitionBy("x", "y")
        .orderBy("month_idx")
        .rowsBetween(-5, 0)
    )
    w_lag = Window.partitionBy("x", "y").orderBy("month_idx")

    for col_name in ["t_star_mo", "lwe_mo", "delta_h"]:
        if col_name not in df.columns:
            continue
        avg_col = f"{col_name.replace('_mo', '')}_6mo_avg"
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


def add_trajectory_features(df: DataFrame) -> DataFrame:
    """Momentum and acceleration features (Model 3 features)."""

    w_lag = Window.partitionBy("x", "y").orderBy("month_idx")
    w12 = Window.partitionBy("x", "y").orderBy("month_idx").rowsBetween(-11, 0)
    w6  = Window.partitionBy("x", "y").orderBy("month_idx").rowsBetween(-5, 0)

    if "delta_h" in df.columns:
        lag1 = F.lag("delta_h", 1).over(w_lag)
        lag2 = F.lag("delta_h", 2).over(w_lag)
        df = df.withColumn("delta_h_momentum",
                           F.col("delta_h") - F.coalesce(lag1, F.col("delta_h")))
        df = df.withColumn("delta_h_acceleration",
                           F.col("delta_h") - F.lit(2.0) * F.coalesce(lag1, F.col("delta_h"))
                           + F.coalesce(lag2, F.col("delta_h")))
        df = df.withColumn("delta_h_deseason",
                           F.col("delta_h") - F.avg("delta_h").over(w12))

    if "t_star_mo" in df.columns:
        t_lag = F.lag("t_star_mo", 1).over(w_lag)
        df = df.withColumn("t_star_momentum",
                           F.col("t_star_mo") - F.coalesce(t_lag, F.col("t_star_mo")))
        df = df.withColumn("t_star_sustained_anomaly",
                           F.avg("t_star_mo").over(w6) - F.avg("t_star_mo").over(w12))

    if "lwe_mo" in df.columns:
        l_lag = F.lag("lwe_mo", 1).over(w_lag)
        df = df.withColumn("lwe_momentum",
                           F.col("lwe_mo") - F.coalesce(l_lag, F.col("lwe_mo")))
        df = df.withColumn("lwe_sustained_trend",
                           F.avg("lwe_mo").over(w6) - F.avg("lwe_mo").over(w12))

    print("[features] Trajectory features added.")
    return df


def add_physics_interactions(df: DataFrame) -> DataFrame:
    """Hand-crafted physics interaction features (Model 2 features)."""

    cols = set(df.columns)
    if {"thetao_mo", "ice_draft", "dist_to_ocean"} <= cols:
        df = df.withColumn("ocean_heat_content_proxy",
                           F.col("thetao_mo") * F.abs(F.col("ice_draft"))
                           / (F.col("dist_to_ocean") + F.lit(1.0)))
    if {"ice_draft", "thickness"} <= cols:
        df = df.withColumn("draft_ratio",
                           F.abs(F.col("ice_draft")) / (F.col("thickness") + F.lit(1.0)))
    if {"t_star_mo", "dist_to_grounding_line"} <= cols:
        df = df.withColumn("thermal_x_gl_proximity",
                           F.col("t_star_mo") / (F.col("dist_to_grounding_line") + F.lit(1.0)))
    if {"thetao_mo", "t_f_mo"} <= cols:
        df = df.withColumn("freezing_departure",
                           F.col("thetao_mo") - F.col("t_f_mo"))
    if {"bed_slope", "bed"} <= cols:
        df = df.withColumn("bed_geometry_risk",
                           F.col("bed_slope") * F.least(F.col("bed"), F.lit(0.0)))
    if {"delta_h", "ice_area"} <= cols:
        df = df.withColumn("mass_flux_proxy",
                           F.col("delta_h") * F.col("ice_area"))
    print("[features] Physics interaction features added.")
    return df


def add_region_integer_encoding(df: DataFrame) -> DataFrame:
    """Integer-encode region for native categorical support."""
    region_map = {
        "amundsen_sea": 0, "antarctic_peninsula": 1, "lambert_amery": 2,
        "ronne": 3, "ross": 4, "totten_and_aurora": 5, "sample": 6,
    }
    expr = F.lit(6)
    for name, idx in region_map.items():
        expr = F.when(F.col("regional_subset_id") == name, F.lit(idx)).otherwise(expr)
    df = df.withColumn("region_cat_idx", expr.cast(IntegerType()))
    print("[features] Region integer encoding added.")
    return df


# =====================================================================
# 4E. UNIFIED FEATURE ENGINEERING RUNNER
# =====================================================================

def run_feature_engineering(spark: SparkSession) -> None:
    """Build all features.  Reads from ml_ready or raw data per FEATURE_SOURCE.

    For SDSC-scale data (>100M rows), this function breaks the DAG
    lineage with a checkpoint after the heavy window operations to
    prevent OOM.  Without this, Spark tries to execute ~20 chained
    window ops in a single task, which exceeds executor memory.
    """

    # Checkpoint dir for breaking DAG lineage on large datasets
    scratch = os.environ.get("TMPDIR", os.path.join(os.getcwd(), "spark_scratch"))
    ckpt_dir = os.path.join(scratch, "unified_checkpoints")
    spark.sparkContext.setCheckpointDir(ckpt_dir)

    is_large = MODE in ("hpc", "sdsc")
    n_shuffle = 2000 if is_large else 8

    if FEATURE_SOURCE == "from_raw":
        print(f"\n[features] ===== BUILDING FROM RAW DATA =====")
        df = load_raw_data(spark)
        df = build_label(df)
        df = assign_regions(df)
        df = add_static_features(df)
        df = add_dynamic_features(df)
        df = add_ocean_features(df)
        df = add_context_features(df)
        df = add_sample_weights(df)

        # ── LINEAGE BREAK ─────────────────────────────────────────
        # The base features chain ~12 window operations.  Checkpoint
        # forces materialisation so the next round of windows starts
        # with a clean, flat DAG.
        if is_large:
            print("[features] Breaking DAG lineage (checkpoint)...")
            df = df.repartition(n_shuffle, "month_idx", "mascon_id")
            df = df.checkpoint(eager=True)
            print("[features] Checkpoint complete — DAG reset.")
    else:
        print(f"[features] Reading pre-built features from: {INPUT_PATH}")
        df = spark.read.parquet(INPUT_PATH)
        print(f"[features] Loaded {len(df.columns)} columns.")
        df = df.filter(F.col(LABEL_COL).isNotNull() & F.col(WEIGHT_COL).isNotNull())

    # Add Model 2/3 extras on top (another ~10 window ops)
    df = add_temporal_memory_features(df)
    df = add_trajectory_features(df)
    df = add_physics_interactions(df)
    df = add_region_integer_encoding(df)

    # Leakage check
    assert "lwe_fused" not in df.columns, "LEAKAGE: lwe_fused in output!"

    # Write — use more partitions for large datasets
    row_count = df.count()
    if is_large:
        n_parts = n_shuffle  # 2000 partitions for 474M rows
    else:
        n_parts = max(4, min(500, int(row_count * 150 / (128 * 1024 * 1024))))
    df = df.repartition(n_parts, "month_idx", "mascon_id")

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    df.write.mode("overwrite").parquet(OUTPUT_PATH)
    print(f"[features] Wrote {row_count:,} rows -> {OUTPUT_PATH} ({n_parts} partitions)")


# =====================================================================
# 5. DATA LOADING, SPLITTING, AND UNDERSAMPLING
# =====================================================================

def load_and_split(spark: SparkSession) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Load unified features and split temporally."""

    df = spark.read.parquet(OUTPUT_PATH)
    print(f"[load] {len(df.columns)} columns from {OUTPUT_PATH}")

    df = df.filter(F.col(LABEL_COL).isNotNull() & F.col(WEIGHT_COL).isNotNull())

    train = df.filter(F.col("month_idx") <= TRAIN_MAX_MONTH_IDX)
    val = df.filter(
        (F.col("month_idx") > TRAIN_MAX_MONTH_IDX)
        & (F.col("month_idx") <= VAL_MAX_MONTH_IDX)
    )
    test = df.filter(F.col("month_idx") > VAL_MAX_MONTH_IDX)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        n = split.count()
        pos = split.filter(F.col(LABEL_COL) == 1).count()
        rate = pos / max(1, n)
        print(f"  {name:5s}: {n:>12,} rows, pos={pos:,}, rate={rate:.6f}")

    return train, val, test


def undersample(train: DataFrame) -> DataFrame:
    """
    Undersample majority class to handle extreme imbalance.

    Keeps 100% of positive cases.
    Samples negative cases to UNDERSAMPLE_NEG_RATIO * num_positives.
    Validates/tests on original distribution.
    """

    if not UNDERSAMPLE_ENABLED:
        print("[undersample] Disabled, using full training set.")
        return train

    pos = train.filter(F.col(LABEL_COL) == 1)
    neg = train.filter(F.col(LABEL_COL) == 0)

    n_pos = pos.count()
    n_neg = neg.count()

    if n_pos == 0:
        print("[undersample] WARNING: zero positive cases, returning full set.")
        return train

    target_neg = n_pos * UNDERSAMPLE_NEG_RATIO
    frac = min(1.0, target_neg / max(1, n_neg))

    neg_sampled = neg.sample(fraction=frac, seed=42)
    n_neg_sampled = neg_sampled.count()

    balanced = pos.unionByName(neg_sampled)

    print(f"[undersample] Positives: {n_pos:,} (kept 100%)")
    print(f"[undersample] Negatives: {n_neg:,} -> {n_neg_sampled:,} "
          f"(ratio 1:{n_neg_sampled // max(1, n_pos)})")

    return balanced


# =====================================================================
# 6. PREPROCESSING PIPELINES
# =====================================================================

def get_all_numeric_cols(available: List[str]) -> List[str]:
    """Return all numeric feature columns available in the DataFrame."""

    candidates = [
        # Geometry + ice
        "surface", "bed", "thickness", "bed_slope",
        "dist_to_grounding_line", "clamped_depth", "dist_to_ocean", "ice_draft",
        "delta_h", "ice_area", "surface_slope", "h_surface_dynamic",
        "bed_below_sea_level",
        "draft_x_thermal_access", "grounding_line_vulnerability", "retrograde_flag",
        "pixel_mean_delta_h", "delta_h_deviation", "surface_slope_change",
        "thermal_driving_x_draft", "thermal_anomaly",
        "salinity_stratification_proxy", "lwe_trend",
        "sin_month", "cos_month",
        "mascon_mean_delta_h", "mascon_mean_t_star",
        "regional_delta_h_percentile", "regional_lwe_mean",
        # GRACE
        "lwe_mo", "lwe_quarterly_avg", "lwe_quarterly_std",
        # Ocean
        "thetao_mo", "t_star_mo", "so_mo", "t_f_mo",
        "t_star_quarterly_avg", "t_star_quarterly_std",
        "thetao_quarterly_avg", "thetao_quarterly_std",
        "regional_t_star_climatology", "regional_t_star_anomaly",
        # Temporal memory (model 2)
        "t_star_6mo_avg", "lwe_6mo_avg", "delta_h_6mo_avg",
        "t_star_rate", "lwe_rate", "delta_h_rate",
        "t_star_mom_change", "delta_h_mom_change",
        # Physics interactions (model 2)
        "ocean_heat_content_proxy", "draft_ratio",
        "thermal_x_gl_proximity", "freezing_departure",
        "bed_geometry_risk", "mass_flux_proxy",
        # Trajectory (model 3)
        "delta_h_momentum", "delta_h_acceleration", "delta_h_deseason",
        "t_star_momentum", "t_star_sustained_anomaly",
        "lwe_momentum", "lwe_sustained_trend",
        # Region categorical
        "region_cat_idx",
    ]
    return [c for c in candidates if c in available]


def build_xgb_preprocessing(available: List[str]) -> Pipeline:
    """XGBoost model: Imputer -> Bucketizer -> OHE -> Assembler -> MinMaxScaler."""

    numeric = get_all_numeric_cols(available)
    imputed = [f"{c}_imp" for c in numeric]

    stages = [
        Imputer(strategy="median", inputCols=numeric, outputCols=imputed),
        Bucketizer(splits=GL_BUCKET_SPLITS,
                   inputCol="dist_to_grounding_line",
                   outputCol="gl_bucket_idx", handleInvalid="keep"),
        StringIndexer(inputCol="regional_subset_id",
                      outputCol="region_index", handleInvalid="keep"),
        OneHotEncoder(inputCol="gl_bucket_idx", outputCol="gl_bucket_ohe"),
        OneHotEncoder(inputCol="region_index", outputCol="region_ohe"),
        VectorAssembler(inputCols=imputed + ["gl_bucket_ohe", "region_ohe"],
                        outputCol="raw_features", handleInvalid="skip"),
        MinMaxScaler(inputCol="raw_features", outputCol="features"),
    ]
    print(f"[preprocess:xgb] {len(numeric)} numeric + Bucketizer + OHE + MinMaxScaler")
    return Pipeline(stages=stages)


def build_stack_preprocessing(available: List[str]) -> Pipeline:
    """Stacking models: Imputer -> Assembler -> Normalizer (L2)."""

    numeric = get_all_numeric_cols(available)
    imputed = [f"{c}_imp" for c in numeric]

    stages = [
        Imputer(strategy="median", inputCols=numeric, outputCols=imputed),
        VectorAssembler(inputCols=imputed, outputCol="raw_features",
                        handleInvalid="skip"),
        Normalizer(inputCol="raw_features", outputCol="features", p=2.0),
    ]
    print(f"[preprocess:stack] {len(numeric)} numeric + Normalizer(L2)")
    return Pipeline(stages=stages)


def build_classic_preprocessing(available: List[str]) -> Pipeline:
    """Classic model: Imputer -> StringIndexer -> OHE -> PCA(ocean) ->
    PolynomialExpansion -> VectorAssembler -> StandardScaler.

    Demonstrates every required MLlib transformer in one pipeline.
    """

    numeric = get_all_numeric_cols(available)
    imputed = [f"{c}_imp" for c in numeric]

    # Ocean columns for PCA (use imputed names)
    ocean_imp = [f"{c}_imp" for c in OCEAN_COLS if c in available]
    non_ocean_imp = [c for c in imputed if c not in ocean_imp]

    # Polynomial expansion inputs (use imputed names)
    poly_imp = [f"{c}_imp" for c in POLY_INPUT_COLS if c in available]

    stages = [
        Imputer(strategy="median", inputCols=numeric, outputCols=imputed),
        StringIndexer(inputCol="regional_subset_id",
                      outputCol="region_index", handleInvalid="keep"),
        OneHotEncoder(inputCol="region_index", outputCol="region_ohe"),
    ]

    # PCA on correlated ocean variables
    if ocean_imp:
        stages += [
            VectorAssembler(inputCols=ocean_imp, outputCol="ocean_vec",
                            handleInvalid="skip"),
            PCA(k=min(PCA_K, len(ocean_imp)),
                inputCol="ocean_vec", outputCol="ocean_pca"),
        ]

    # PolynomialExpansion on key physics triple
    if poly_imp:
        stages += [
            VectorAssembler(inputCols=poly_imp, outputCol="poly_input",
                            handleInvalid="skip"),
            PolynomialExpansion(degree=2, inputCol="poly_input",
                                outputCol="poly_features"),
        ]

    # Final assembly
    final_inputs = non_ocean_imp + ["region_ohe"]
    if ocean_imp:
        final_inputs.append("ocean_pca")
    if poly_imp:
        final_inputs.append("poly_features")

    stages += [
        VectorAssembler(inputCols=final_inputs, outputCol="raw_features",
                        handleInvalid="skip"),
        StandardScaler(inputCol="raw_features", outputCol="features",
                       withMean=True, withStd=True),
    ]

    print(f"[preprocess:classic] {len(numeric)} numeric + OHE + "
          f"PCA(k={PCA_K}) + PolyExpansion(deg=2) + StandardScaler")
    return Pipeline(stages=stages)


PREPROCESS_MAP = {
    "xgb": build_xgb_preprocessing,
    "stack": build_stack_preprocessing,
    "corrected_stack": build_stack_preprocessing,
    "classic": build_classic_preprocessing,
}


# =====================================================================
# 7. MODEL INITIALISATION
# =====================================================================

def init_xgb_models() -> List[Tuple[str, object]]:
    """SparkXGBClassifier baseline + tuned."""

    try:
        from xgboost.spark import SparkXGBClassifier
    except ImportError:
        print("WARNING: xgboost.spark unavailable, using GBT proxy.")
        return [
            (name, GBTClassifier(
                labelCol=LABEL_COL, featuresCol="features", weightCol=WEIGHT_COL,
                maxIter=cfg.get("n_estimators", 50), maxDepth=cfg["max_depth"],
                stepSize=cfg["learning_rate"], seed=42))
            for name, cfg in XGB_CONFIGS.items()
        ]

    models = []
    for name, cfg in XGB_CONFIGS.items():
        m = SparkXGBClassifier(
            features_col="features", label_col=LABEL_COL, weight_col=WEIGHT_COL,
            eval_metric="logloss", use_gpu=False, missing=0.0,
            num_workers=2 if LOCAL else 6, **cfg,
        )
        models.append((name, m))
    return models


def init_base_learners() -> List[Tuple[str, object]]:
    """RF + GBT base learners for stacking."""

    return [
        ("Base_RF", RandomForestClassifier(
            labelCol=LABEL_COL, featuresCol="features", weightCol=WEIGHT_COL,
            predictionCol="rf_prediction", probabilityCol="rf_probability",
            rawPredictionCol="rf_rawPrediction",
            featureSubsetStrategy="sqrt", seed=42, **RF_CONFIG)),
        ("Base_GBT", GBTClassifier(
            labelCol=LABEL_COL, featuresCol="features", weightCol=WEIGHT_COL,
            predictionCol="gbt_prediction",
            seed=42, **GBT_CONFIG)),
    ]


def init_stack_meta_learners() -> List[Tuple[str, object]]:
    """XGBoost meta-learners for uncorrected stacking."""

    try:
        from xgboost.spark import SparkXGBClassifier
        return [
            ("Stack_Baseline", SparkXGBClassifier(
                features_col="meta_features", label_col=LABEL_COL,
                weight_col=WEIGHT_COL, max_depth=3,
                n_estimators=30 if LOCAL else 100, learning_rate=0.1,
                subsample=0.8, min_child_weight=10, eval_metric="logloss",
                use_gpu=False, num_workers=2 if LOCAL else 6, missing=0.0)),
            ("Stack_Tuned", SparkXGBClassifier(
                features_col="meta_features", label_col=LABEL_COL,
                weight_col=WEIGHT_COL, max_depth=4 if LOCAL else 6,
                n_estimators=50 if LOCAL else 300,
                learning_rate=0.05 if LOCAL else 0.02,
                subsample=0.75, min_child_weight=15 if LOCAL else 20,
                reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
                use_gpu=False, num_workers=2 if LOCAL else 6, missing=0.0)),
        ]
    except ImportError:
        return [
            ("Stack_GBT_proxy", GBTClassifier(
                labelCol=LABEL_COL, featuresCol="meta_features",
                weightCol=WEIGHT_COL, maxIter=50, maxDepth=3, stepSize=0.1, seed=99)),
        ]


def init_corrected_meta_learners() -> List[Tuple[str, object]]:
    """LogisticRegression meta-learners for corrected stacking."""

    return [
        ("CorrStack_LR_Baseline", LogisticRegression(
            labelCol=LABEL_COL, featuresCol="meta_features", weightCol=WEIGHT_COL,
            predictionCol=PREDICTION_COL, probabilityCol="probability",
            rawPredictionCol="rawPrediction",
            maxIter=50 if LOCAL else 200, regParam=0.01, elasticNetParam=0.0)),
        ("CorrStack_LR_Tuned", LogisticRegression(
            labelCol=LABEL_COL, featuresCol="meta_features", weightCol=WEIGHT_COL,
            predictionCol=PREDICTION_COL, probabilityCol="probability",
            rawPredictionCol="rawPrediction",
            maxIter=100 if LOCAL else 500, regParam=0.1, elasticNetParam=0.5)),
    ]


def init_classic_models() -> List[Tuple[str, object]]:
    """DecisionTree -> RandomForest -> GBT progression."""

    return [
        ("Classic_DT", DecisionTreeClassifier(
            labelCol=LABEL_COL, featuresCol="features", weightCol=WEIGHT_COL,
            maxDepth=6 if LOCAL else 8, seed=42)),
        ("Classic_RF", RandomForestClassifier(
            labelCol=LABEL_COL, featuresCol="features", weightCol=WEIGHT_COL,
            numTrees=20 if LOCAL else 100, maxDepth=6 if LOCAL else 10,
            featureSubsetStrategy="sqrt", seed=42)),
        ("Classic_GBT", GBTClassifier(
            labelCol=LABEL_COL, featuresCol="features", weightCol=WEIGHT_COL,
            maxIter=30 if LOCAL else 200, maxDepth=4 if LOCAL else 6,
            stepSize=0.1 if LOCAL else 0.05, seed=42)),
    ]


# =====================================================================
# 8. STACKING HELPERS
# =====================================================================

def build_full_meta_features(df: DataFrame, imputed_cols: List[str]) -> DataFrame:
    """Build meta-features for uncorrected stacking (all features + base preds)."""

    if "rf_probability" in df.columns:
        df = df.withColumn("rf_prob_arr", vector_to_array("rf_probability"))
        df = df.withColumn("rf_pos_prob", F.col("rf_prob_arr").getItem(1).cast(FloatType()))
        df = df.drop("rf_prob_arr")
    else:
        df = df.withColumn("rf_pos_prob", F.lit(0.5).cast(FloatType()))

    if "gbt_prediction" in df.columns:
        df = df.withColumn("gbt_score", F.col("gbt_prediction").cast(FloatType()))
    else:
        df = df.withColumn("gbt_score", F.lit(0.0).cast(FloatType()))

    if "rf_prediction" in df.columns and "gbt_prediction" in df.columns:
        df = df.withColumn("base_agreement",
                           (F.col("rf_prediction") == F.col("gbt_prediction")).cast(FloatType()))
    else:
        df = df.withColumn("base_agreement", F.lit(1.0).cast(FloatType()))

    meta_inputs = ["rf_pos_prob", "gbt_score", "base_agreement"]
    available_imp = [c for c in imputed_cols if c in df.columns]
    meta_inputs.extend(available_imp)

    assembler = VectorAssembler(inputCols=meta_inputs, outputCol="meta_features",
                                 handleInvalid="skip")
    df = assembler.transform(df)

    # Drop colliding columns
    for c in ["rawPrediction", "prediction", "probability"]:
        if c in df.columns:
            df = df.drop(c)

    return df


def build_pruned_meta_features(df: DataFrame, available_cols: List[str]) -> DataFrame:
    """Build PRUNED meta-features for corrected stacking (base preds + 5 context only)."""

    if "rf_probability" in df.columns:
        df = df.withColumn("rf_prob_arr", vector_to_array("rf_probability"))
        df = df.withColumn("rf_pos_prob", F.col("rf_prob_arr").getItem(1).cast(FloatType()))
        df = df.drop("rf_prob_arr")
    else:
        df = df.withColumn("rf_pos_prob", F.lit(0.5).cast(FloatType()))

    if "gbt_prediction" in df.columns:
        df = df.withColumn("gbt_score", F.col("gbt_prediction").cast(FloatType()))
    else:
        df = df.withColumn("gbt_score", F.lit(0.0).cast(FloatType()))

    if "rf_prediction" in df.columns and "gbt_prediction" in df.columns:
        df = df.withColumn("base_agreement",
                           (F.col("rf_prediction") == F.col("gbt_prediction")).cast(FloatType()))
    else:
        df = df.withColumn("base_agreement", F.lit(1.0).cast(FloatType()))

    meta_inputs = ["rf_pos_prob", "gbt_score", "base_agreement"]
    for col in META_CONTEXT_COLS:
        if col in available_cols:
            imp_name = f"{col}_meta"
            df = df.withColumn(imp_name,
                               F.coalesce(F.col(col), F.lit(0.0)).cast(FloatType()))
            meta_inputs.append(imp_name)

    assembler = VectorAssembler(inputCols=meta_inputs, outputCol="meta_features",
                                 handleInvalid="skip")
    df = assembler.transform(df)

    for c in ["rawPrediction", "prediction", "probability"]:
        if c in df.columns:
            df = df.drop(c)

    return df


def oof_split(train: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """Split training data temporally for OOF stacking."""

    stats = train.agg(F.min("month_idx").alias("mn"),
                      F.max("month_idx").alias("mx")).collect()[0]
    mid = (stats["mn"] + stats["mx"]) // 2

    fold_a = train.filter(F.col("month_idx") <= mid)
    fold_b = train.filter(F.col("month_idx") > mid)

    for name, fold in [("A", fold_a), ("B", fold_b)]:
        n = fold.count()
        pos = fold.filter(F.col(LABEL_COL) == 1).count()
        print(f"  [OOF] Fold {name}: {n:,} rows, pos={pos:,}")

    return fold_a, fold_b


# =====================================================================
# 9. EVALUATION
# =====================================================================

def evaluate(model_name: str, preds: DataFrame, split_name: str) -> Dict:
    """Compute ROC-AUC, PR-AUC, F1, Precision, Recall."""

    label_stats = preds.agg(
        F.min(LABEL_COL).alias("mn"), F.max(LABEL_COL).alias("mx"),
    ).collect()[0]

    nan = float("nan")
    if label_stats["mn"] == label_stats["mx"]:
        print(f"  [{model_name}] {split_name:7s}  single class={label_stats['mn']}")
        return {"model": model_name, "split": split_name,
                "roc_auc": nan, "pr_auc": nan, "f1": nan,
                "precision": nan, "recall": nan}

    f1  = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol=PREDICTION_COL, metricName="f1"
    ).evaluate(preds)
    prec = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol=PREDICTION_COL, metricName="weightedPrecision"
    ).evaluate(preds)
    rec = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol=PREDICTION_COL, metricName="weightedRecall"
    ).evaluate(preds)

    roc_auc = nan
    pr_auc  = nan
    if "rawPrediction" in preds.columns:
        try:
            roc_auc = BinaryClassificationEvaluator(
                labelCol=LABEL_COL, rawPredictionCol="rawPrediction",
                metricName="areaUnderROC",
            ).evaluate(preds)
            pr_auc = BinaryClassificationEvaluator(
                labelCol=LABEL_COL, rawPredictionCol="rawPrediction",
                metricName="areaUnderPR",
            ).evaluate(preds)
        except Exception:
            pass

    print(f"  [{model_name}] {split_name:7s}  "
          f"ROC={roc_auc:.4f}  PR={pr_auc:.4f}  "
          f"F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")

    return {"model": model_name, "split": split_name,
            "roc_auc": roc_auc, "pr_auc": pr_auc,
            "f1": f1, "precision": prec, "recall": rec}


def regional_summary(preds: DataFrame, model_name: str) -> None:
    """Print per-region prediction stats."""

    rows = (
        preds.groupBy("regional_subset_id")
        .agg(
            F.avg(F.col(PREDICTION_COL).cast("float")).alias("pred_rate"),
            F.avg(F.col(LABEL_COL).cast("float")).alias("true_rate"),
            F.count("*").alias("n"),
        )
        .orderBy("regional_subset_id")
        .collect()
    )
    print(f"\n  [{model_name}] Regional breakdown:")
    for r in rows:
        print(f"    {r['regional_subset_id']:25s}  "
              f"pred={r['pred_rate']:.4f}  true={r['true_rate']:.4f}  n={r['n']:>10,}")


def save_predictions(preds: DataFrame, model_name: str, split_name: str) -> None:
    """Save 500-row prediction sample."""

    cols = [c for c in KEY_COLS if c in preds.columns]
    cols += [LABEL_COL, PREDICTION_COL, WEIGHT_COL]
    if "probability" in preds.columns:
        cols.append("probability")

    path = os.path.join(OUTPUT_DIR, f"preds_{model_name}_{split_name}")
    preds.select(*cols).limit(500).write.mode("overwrite").parquet(path)
    print(f"  [{model_name}] Saved -> {path}")


# =====================================================================
# 10. TRAIN AND TEST
# =====================================================================

def train_xgb(train: DataFrame, val: DataFrame, test: DataFrame) -> List[Dict]:
    """Train SparkXGBClassifier models."""

    preprocess = build_xgb_preprocessing(train.columns)
    pp_model = preprocess.fit(train)
    train_p = pp_model.transform(undersample(train)).cache()
    val_p   = pp_model.transform(val).cache()
    test_p  = pp_model.transform(test).cache()

    # Also transform full train for train-error evaluation
    train_full_p = pp_model.transform(train).cache()

    results = []
    for name, clf in init_xgb_models():
        print(f"\n{'='*60}\n  TRAINING: {name}\n{'='*60}")
        fitted = Pipeline(stages=[clf]).fit(train_p)

        for sname, sdf in [("train", train_full_p), ("val", val_p), ("test", test_p)]:
            p = fitted.transform(sdf)
            results.append(evaluate(name, p, sname))
            save_predictions(p, name, sname)
            if sname == "test":
                regional_summary(p, name)

    for df in [train_p, val_p, test_p, train_full_p]:
        df.unpersist()
    return results


def train_stack(train: DataFrame, val: DataFrame, test: DataFrame) -> List[Dict]:
    """Train uncorrected stacking ensemble."""

    preprocess = build_stack_preprocessing(train.columns)
    pp_model = preprocess.fit(train)
    train_p = pp_model.transform(undersample(train)).cache()
    val_p   = pp_model.transform(val).cache()
    test_p  = pp_model.transform(test).cache()
    train_full_p = pp_model.transform(train).cache()

    numeric = get_all_numeric_cols(train.columns)
    imputed = [f"{c}_imp" for c in numeric]

    results = []
    base_models = {}

    # Layer 1: base learners
    print(f"\n{'='*60}\n  LAYER 1: BASE LEARNERS\n{'='*60}")
    for name, clf in init_base_learners():
        print(f"\n  Training {name}...")
        fitted = Pipeline(stages=[clf]).fit(train_p)
        base_models[name] = fitted

        for sname, sdf in [("train", train_full_p), ("val", val_p), ("test", test_p)]:
            p = fitted.transform(sdf)
            pred_col = "rf_prediction" if "RF" in name else "gbt_prediction"
            ev = p.withColumn(PREDICTION_COL, F.col(pred_col))
            if "rf_rawPrediction" in ev.columns:
                ev = ev.withColumn("rawPrediction", F.col("rf_rawPrediction"))
            results.append(evaluate(name, ev, sname))

    # Layer 2: stack
    print(f"\n{'='*60}\n  LAYER 2: META-LEARNER\n{'='*60}")

    def add_base(df):
        for _, m in base_models.items():
            df = m.transform(df)
        return df

    train_s = build_full_meta_features(add_base(train_full_p), imputed)
    val_s   = build_full_meta_features(add_base(val_p), imputed)
    test_s  = build_full_meta_features(add_base(test_p), imputed)
    # For fitting: use undersampled training set
    train_fit_s = build_full_meta_features(add_base(train_p), imputed)

    for name, clf in init_stack_meta_learners():
        print(f"\n  Training {name}...")
        fitted = Pipeline(stages=[clf]).fit(train_fit_s)
        for sname, sdf in [("train", train_s), ("val", val_s), ("test", test_s)]:
            p = fitted.transform(sdf)
            results.append(evaluate(name, p, sname))
            save_predictions(p, name, sname)
            if sname == "test":
                regional_summary(p, name)

    for df in [train_p, val_p, test_p, train_full_p]:
        df.unpersist()
    return results


def train_corrected_stack(train: DataFrame, val: DataFrame, test: DataFrame) -> List[Dict]:
    """Train corrected stacking ensemble (OOF + pruned + LR)."""

    preprocess = build_stack_preprocessing(train.columns)
    pp_model = preprocess.fit(train)

    train_p = pp_model.transform(train).cache()
    val_p   = pp_model.transform(val).cache()
    test_p  = pp_model.transform(test).cache()

    available_cols = train.columns
    results = []
    base_models = {}

    # OOF split
    fold_a_raw, fold_b_raw = oof_split(train_p)
    fold_a = undersample(fold_a_raw).cache()
    fold_b = fold_b_raw.cache()

    # Layer 1: base learners on fold A
    print(f"\n{'='*60}\n  LAYER 1: BASE LEARNERS ON FOLD A\n{'='*60}")
    for name, clf in init_base_learners():
        print(f"\n  Training {name}...")
        fitted = Pipeline(stages=[clf]).fit(fold_a)
        base_models[name] = fitted

        for sname, sdf in [("fold_a", fold_a), ("fold_b", fold_b),
                           ("val", val_p), ("test", test_p)]:
            p = fitted.transform(sdf)
            pred_col = "rf_prediction" if "RF" in name else "gbt_prediction"
            ev = p.withColumn(PREDICTION_COL, F.col(pred_col))
            if "rf_rawPrediction" in ev.columns:
                ev = ev.withColumn("rawPrediction", F.col("rf_rawPrediction"))
            results.append(evaluate(name, ev, sname))

    # Layer 2: corrected meta-learner on fold B
    print(f"\n{'='*60}\n  LAYER 2: LR META-LEARNER ON FOLD B\n{'='*60}")

    def add_base(df):
        for _, m in base_models.items():
            df = m.transform(df)
        return df

    fold_b_m = build_pruned_meta_features(add_base(fold_b), available_cols)
    val_m    = build_pruned_meta_features(add_base(val_p), available_cols)
    test_m   = build_pruned_meta_features(add_base(test_p), available_cols)
    train_m  = build_pruned_meta_features(add_base(train_p), available_cols)

    n_meta = fold_b_m.select("meta_features").head(1)[0]["meta_features"].size
    print(f"  Meta-features: {n_meta} dimensions (pruned)")

    for name, clf in init_corrected_meta_learners():
        print(f"\n  Training {name}...")
        fitted = Pipeline(stages=[clf]).fit(fold_b_m)

        # Print LR coefficients
        lr = fitted.stages[-1]
        if hasattr(lr, "coefficients"):
            coeffs = lr.coefficients.toArray()
            print(f"    Intercept: {lr.intercept:.4f}")
            meta_names = ["rf_pos_prob", "gbt_score", "base_agreement"] + \
                         [f"{c}_meta" for c in META_CONTEXT_COLS]
            for n, c in zip(meta_names[:len(coeffs)], coeffs):
                sign = "+" if c >= 0 else "-"
                print(f"    {n:30s}  {sign}{abs(c):.4f}")

        for sname, sdf in [("train", train_m), ("val", val_m), ("test", test_m)]:
            p = fitted.transform(sdf)
            results.append(evaluate(name, p, sname))
            save_predictions(p, name, sname)
            if sname == "test":
                regional_summary(p, name)

    for df in [train_p, val_p, test_p, fold_a, fold_b]:
        df.unpersist()
    return results


def train_classic(train: DataFrame, val: DataFrame, test: DataFrame) -> List[Dict]:
    """Train the DecisionTree -> RandomForest -> GBT progression.

    Uses PolynomialExpansion + StandardScaler + Ocean PCA preprocessing.
    """

    preprocess = build_classic_preprocessing(train.columns)
    pp_model = preprocess.fit(train)
    train_p = pp_model.transform(undersample(train)).cache()
    val_p   = pp_model.transform(val).cache()
    test_p  = pp_model.transform(test).cache()
    train_full_p = pp_model.transform(train).cache()

    n_features = train_p.select("features").head(1)[0]["features"].size
    print(f"  Feature vector dimension: {n_features}")

    results = []
    for name, clf in init_classic_models():
        print(f"\n{'='*60}\n  TRAINING: {name}\n{'='*60}")
        fitted = Pipeline(stages=[clf]).fit(train_p)

        for sname, sdf in [("train", train_full_p), ("val", val_p), ("test", test_p)]:
            p = fitted.transform(sdf)
            results.append(evaluate(name, p, sname))
            save_predictions(p, name, sname)
            if sname == "test":
                regional_summary(p, name)

        tree_model = fitted.stages[-1]
        if hasattr(tree_model, "featureImportances"):
            importances = tree_model.featureImportances.toArray()
            pairs = sorted(enumerate(importances), key=lambda p: p[1], reverse=True)
            print(f"\n  [{name}] Top 10 features:")
            for idx, imp in pairs[:10]:
                bar = "#" * int(imp * 100)
                print(f"    feat[{idx:3d}]  {imp:.4f}  {bar}")

    for df in [train_p, val_p, test_p, train_full_p]:
        df.unpersist()
    return results


TRAIN_MAP = {
    "xgb": train_xgb,
    "stack": train_stack,
    "corrected_stack": train_corrected_stack,
    "classic": train_classic,
}


# =====================================================================
# 11. ANALYSIS
# =====================================================================

def print_results_table(results: List[Dict]) -> None:
    """Print formatted results summary."""

    print(f"\n{'='*72}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Model':<25s} {'Split':<7s} {'ROC-AUC':>8s} {'PR-AUC':>8s} "
          f"{'F1':>8s} {'Prec':>8s} {'Rec':>8s}")
    print(f"  {'-'*70}")

    for r in results:
        def fmt(v):
            return f"{v:>8.4f}" if v == v else "     N/A"
        print(f"  {r['model']:<25s} {r['split']:<7s} "
              f"{fmt(r['roc_auc'])} {fmt(r['pr_auc'])} "
              f"{fmt(r['f1'])} {fmt(r['precision'])} {fmt(r['recall'])}")
    print(f"{'='*72}")


def fitting_analysis(results: List[Dict]) -> None:
    """Diagnose overfitting/underfitting."""

    print(f"\n{'='*72}")
    print(f"  FITTING ANALYSIS")
    print(f"{'='*72}")

    models = set(r["model"] for r in results if r["split"] in ("train", "test"))

    for name in sorted(models):
        mr = {r["split"]: r for r in results if r["model"] == name}
        tr = mr.get("train", {}).get("roc_auc", float("nan"))
        te = mr.get("test", {}).get("roc_auc", float("nan"))
        pr = mr.get("test", {}).get("pr_auc", float("nan"))

        if tr != tr or te != te:
            print(f"\n  {name}: insufficient data.")
            continue

        gap = tr - te
        print(f"\n  {name}:")
        print(f"    Train ROC-AUC: {tr:.4f}")
        print(f"    Test  ROC-AUC: {te:.4f}  (gap: {gap:.4f})")
        print(f"    Test  PR-AUC:  {pr:.4f}")

        if tr < 0.60 and te < 0.60:
            print(f"    -> UNDERFITTING")
        elif gap > 0.10:
            print(f"    -> OVERFITTING")
        elif gap > 0.05:
            print(f"    -> MILD OVERFITTING")
        else:
            print(f"    -> GOOD FIT")

    print(f"{'='*72}")


def plot_geographic_errors(preds: DataFrame, model_name: str) -> None:
    """Generate Plotly geographic error visualizations."""

    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd
    except ImportError:
        print(f"  Plotly not available, skipping plots.")
        return

    total = preds.count()
    cols = [c for c in ["x", "y", "regional_subset_id", LABEL_COL, PREDICTION_COL]
            if c in preds.columns]

    pdf = (preds.select(*cols)
           .sample(fraction=min(1.0, 10000 / max(total, 1)))
           .toPandas())

    if pdf.empty:
        return

    pdf["error_type"] = "True Negative"
    pdf.loc[(pdf[LABEL_COL] == 1) & (pdf[PREDICTION_COL] == 1), "error_type"] = "True Positive"
    pdf.loc[(pdf[LABEL_COL] == 0) & (pdf[PREDICTION_COL] == 1), "error_type"] = "False Positive"
    pdf.loc[(pdf[LABEL_COL] == 1) & (pdf[PREDICTION_COL] == 0), "error_type"] = "False Negative"

    cmap = {"True Positive": "#2ecc71", "True Negative": "#95a5a6",
            "False Positive": "#e67e22", "False Negative": "#e74c3c"}

    # Plot 1: All predictions
    fig = px.scatter(pdf, x="x", y="y", color="error_type", color_discrete_map=cmap,
                     title=f"{model_name}: Geographic Errors (EPSG:3031)",
                     opacity=0.6, category_orders={"error_type": [
                         "False Negative", "False Positive", "True Positive", "True Negative"]})
    fig.update_layout(template="plotly_dark", width=1000, height=800)
    fig.update_traces(marker_size=3)
    p1 = os.path.join(OUTPUT_DIR, f"{model_name}_geo_errors.html")
    fig.write_html(p1)
    print(f"  [{model_name}] -> {p1}")

    # Plot 2: Regional error rates
    if "regional_subset_id" in pdf.columns:
        rs = pdf.groupby("regional_subset_id").apply(
            lambda g: pd.Series({
                "FNR": (g["error_type"] == "False Negative").sum() / max(1, (g[LABEL_COL] == 1).sum()),
                "FPR": (g["error_type"] == "False Positive").sum() / max(1, (g[LABEL_COL] == 0).sum()),
            })).reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="FNR", x=rs["regional_subset_id"], y=rs["FNR"],
                              marker_color="#e74c3c"))
        fig2.add_trace(go.Bar(name="FPR", x=rs["regional_subset_id"], y=rs["FPR"],
                              marker_color="#e67e22"))
        fig2.update_layout(title=f"{model_name}: Regional Error Rates",
                           template="plotly_dark", barmode="group", width=900, height=500)
        p2 = os.path.join(OUTPUT_DIR, f"{model_name}_regional_errors.html")
        fig2.write_html(p2)
        print(f"  [{model_name}] -> {p2}")

    # Plot 3: Errors only
    errs = pdf[pdf["error_type"].isin(["False Positive", "False Negative"])]
    if not errs.empty:
        fig3 = px.scatter(errs, x="x", y="y", color="error_type", color_discrete_map=cmap,
                          title=f"{model_name}: Misclassified Only", opacity=0.8)
        fig3.update_layout(template="plotly_dark", width=1000, height=800)
        fig3.update_traces(marker_size=4)
        p3 = os.path.join(OUTPUT_DIR, f"{model_name}_errors_only.html")
        fig3.write_html(p3)
        print(f"  [{model_name}] -> {p3}")


def plot_temporal_residuals(preds: DataFrame, model_name: str) -> None:
    """Generate Plotly temporal error-rate plots (monthly FNR / FPR over time)."""

    try:
        import plotly.graph_objects as go
        import pandas as pd
    except ImportError:
        print("  Plotly not available, skipping temporal plots.")
        return

    cols = ["month_idx", LABEL_COL, PREDICTION_COL]
    if not all(c in preds.columns for c in cols):
        print(f"  [{model_name}] Missing columns for temporal plot, skipping.")
        return

    pdf = (preds
           .select(*cols)
           .groupBy("month_idx")
           .agg(
               F.count("*").alias("n"),
               F.sum(F.when(F.col(LABEL_COL) == 1, 1).otherwise(0)).alias("pos"),
               F.sum(F.when(F.col(LABEL_COL) == 0, 1).otherwise(0)).alias("neg"),
               F.sum(F.when((F.col(LABEL_COL) == 1) & (F.col(PREDICTION_COL) == 0), 1)
                      .otherwise(0)).alias("fn"),
               F.sum(F.when((F.col(LABEL_COL) == 0) & (F.col(PREDICTION_COL) == 1), 1)
                      .otherwise(0)).alias("fp"),
           )
           .orderBy("month_idx")
           .toPandas())

    if pdf.empty:
        return

    pdf["fnr"] = pdf["fn"] / pdf["pos"].clip(lower=1)
    pdf["fpr"] = pdf["fp"] / pdf["neg"].clip(lower=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pdf["month_idx"], y=pdf["fnr"], mode="lines+markers",
        name="FNR (miss rate)", line=dict(color="#e74c3c", width=2),
        marker=dict(size=4)))
    fig.add_trace(go.Scatter(
        x=pdf["month_idx"], y=pdf["fpr"], mode="lines+markers",
        name="FPR (false alarm)", line=dict(color="#e67e22", width=2),
        marker=dict(size=4)))
    fig.update_layout(
        title=f"{model_name}: Temporal Error Rates by Month",
        xaxis_title="month_idx", yaxis_title="Error Rate",
        template="plotly_dark", width=1000, height=500,
        yaxis=dict(range=[0, 1]))

    path = os.path.join(OUTPUT_DIR, f"{model_name}_temporal_residuals.html")
    fig.write_html(path)
    print(f"  [{model_name}] -> {path}")


def print_conclusion() -> None:
    """Print conclusion for rubric."""

    print(f"""
{'='*72}
  CONCLUSION
{'='*72}

  1. MODEL SUMMARY:
     - Classic (Model 1): DT -> RF -> GBT progression with
       PolynomialExpansion, Ocean PCA, and StandardScaler. Demonstrates
       MLlib transformer breadth and natural complexity progression.
     - XGBoost (Model 2): SparkXGBClassifier with MinMaxScaler,
       Bucketizer, hand-crafted physics interactions, 6-month temporal
       memory.  Best for capturing threshold-like physics.
     - Stacking (Model 3): RF+GBT base learners, XGBoost meta-learner.
       Learns WHERE each base model is reliable.
     - Corrected Stacking (Model 4): OOF training, pruned meta-features,
       LogisticRegression.  Fixes severe overfitting from Model 3.

  2. UNDERSAMPLING IMPACT:
     Training on balanced data (1:{UNDERSAMPLE_NEG_RATIO} ratio) forces
     the model to attend to rare positive events rather than achieving
     high accuracy by always predicting negative.

  3. PR-AUC AS PRIMARY METRIC:
     With <1% positive rate, ROC-AUC can be misleadingly high.
     PR-AUC directly measures how well the model finds true positives
     without drowning in false alarms.

  4. HOW DISTRIBUTED COMPUTING HELPED:
     - Spark distributes histogram construction for XGBoost/GBT
     - RF tree building parallelised across executors
     - Feature engineering (window functions) partitioned by pixel
     - 40 GB dataset impossible on single machine; 6 executors
       reduce training from hours to minutes
{'='*72}
""")


# =====================================================================
# 12. MAIN EXECUTION
# =====================================================================

def run_all():
    """Main entry point.  Call from notebook or as script."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    spark = get_spark()

    try:
        # Step 1: Feature engineering (run once, then comment out)
        if not os.path.exists(OUTPUT_PATH) or not os.listdir(OUTPUT_PATH):
            print(f"\n{'='*60}\n  FEATURE ENGINEERING\n{'='*60}")
            run_feature_engineering(spark)
        else:
            print(f"[skip] Features exist at {OUTPUT_PATH}")

        # Step 2: Load and split
        print(f"\n{'='*60}\n  DATA LOADING\n{'='*60}")
        train, val, test = load_and_split(spark)

        # Step 3: Train active model
        print(f"\n{'='*60}\n  TRAINING: {ACTIVE_MODEL.upper()}\n{'='*60}")
        train_fn = TRAIN_MAP[ACTIVE_MODEL]
        results = train_fn(train, val, test)

        # Step 4: Results
        print_results_table(results)
        fitting_analysis(results)

        # Step 5: Plotly error-analysis plots (read back saved test predictions)
        print(f"\n{'='*60}\n  ERROR ANALYSIS PLOTS\n{'='*60}")
        model_names = sorted(set(r["model"] for r in results if r["split"] == "test"))
        for mname in model_names:
            pred_path = os.path.join(OUTPUT_DIR, f"preds_{mname}_test")
            if os.path.exists(pred_path):
                test_preds = spark.read.parquet(pred_path)
                plot_geographic_errors(test_preds, mname)
                plot_temporal_residuals(test_preds, mname)
            else:
                print(f"  [{mname}] No test predictions found at {pred_path}")

        print_conclusion()

        return results

    finally:
        spark.stop()


# ── Run when executed as script ──────────────────────────────────────
if __name__ == "__main__":
    run_all()
