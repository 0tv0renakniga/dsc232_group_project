# =====================================================================
# 1. IMPORT LIBRARIES
# =====================================================================

from __future__ import annotations

import os
import math
from functools import reduce
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------
# PySpark ML
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# PySpark Core
# ---------------------------------------------------------------------
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType
from pyspark import StorageLevel
# ---------------------------------------------------------------------


# In[39]:


# =====================================================================
# 2. CONFIG  (edit these for your environment)
# =====================================================================

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
# raw Parquet root
DATA_ROOT   = os.path.join(os.getcwd(), "data/fused_data")
# base features (if already built)
INPUT_PATH  = os.path.join(os.getcwd(), "data/ml_ready")
# unified features
OUTPUT_PATH = os.path.join(os.getcwd(), "data/ml_ready_unified")
# results
OUTPUT_DIR  = os.path.join(os.getcwd(), "dataunified_output")
# SDSC fix: /tmp is not large enough for intermediate data
scratch_dir = os.environ.get("LOCAL_SCRATCH", "/expanse/lustre/projects/uci157/rrogers/temp")
# initialize scratch dir for Spark temp data
os.makedirs(scratch_dir, exist_ok=True)
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Execution mode
# note: "local" for local testing, "sdsc" for running on SDSC
# ---------------------------------------------------------------------
MODE = "hpc"
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# SDSC fix: clear intermediate data from memory/disk after each stage
# ---------------------------------------------------------------------
FLUSH_PARTITIONS = 8 if MODE == "local" else 400
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Feature engineering
# note: switch between building from raw vs. reading pre-built features
# "from_ml_ready" = read from INPUT_PATH (base features already built)
# "from_raw"      = build from scratch starting from DATA_ROOT
# ---------------------------------------------------------------------
FEATURE_SOURCE = "from_ml_ready"
# ---------------------------------------------------------------------



# ---------------------------------------------------------------------
# Label construction: target variable definition for training
# "dual_sensor"   = GRACE + ICESat-2 agreement (strict, very sparse)
# "grace_anomaly" = GRACE-only 25th-pctl flag (~25% positive, backup)
LABEL_MODE = "dual_sensor"
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Model
# note: "xgb", "stack", "corrected_stack", "classic"
# note: baseline xgboost or classic
# note: "stack" = stacking ensemble of XGBoost + RF + GBT + LR
# ---------------------------------------------------------------------
ACTIVE_MODEL = "corrected_stack"
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Temporal split boundaries
# note: using April 2020 becuse of 2019 spike
# ---------------------------------------------------------------------
# Apr 2020
TRAIN_MIN_MONTH_IDX = 24244
# end Dec 2022
TRAIN_MAX_MONTH_IDX = 24276
# end Oct 2023
VAL_MAX_MONTH_IDX   = 24286
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Class imbalance
# note: undersampling config because target is very sparse
# ---------------------------------------------------------------------
UNDERSAMPLE_ENABLED  = True
# keep 1:10 pos:neg ratio
UNDERSAMPLE_NEG_RATIO = 10
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------
LABEL_COL      = "basal_loss_agreement"
WEIGHT_COL     = "weightCol"
PREDICTION_COL = "prediction"

KEY_COLS = ["x", "y", "month_idx", "mascon_id", "regional_subset_id"]
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Data loading constants
# ---------------------------------------------------------------------
REGION_FILES: Dict= {
    "amundsen_sea": "ml_subset_amundsen_sea.parquet",
    "antarctic_peninsula": "ml_subset_antarctic_peninsula.parquet",
    "lambert_amery": "ml_subset_lambert_amery.parquet",
    "ronne": "ml_subset_ronne.parquet",
    "ross":"ml_subset_ross.parquet",
    "totten_and_aurora": "ml_subset_totten_and_aurora.parquet",
}


SPARSE_SAMPLE = "antarctica_sparse_features_sample.parquet"


# EPSG:3031 bounding boxes for assigning regions from the full-continent
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
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# XGBoost hyperparameters
# ---------------------------------------------------------------------
XGB_CONFIGS = {

    "XGB_Tuned": dict(
        max_depth=4 if MODE == "local" else 5,
        n_estimators=100 if MODE == "local" else 150,
        learning_rate=0.05,
        subsample=0.75,
        colsample_bytree=0.7,
        min_child_weight=15 if MODE == "local" else 20,
        reg_alpha=0.1,
        reg_lambda=1.0,
)
}
# ---------------------------------------------------------------------
"""
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

    "XGB_Baseline": dict(
        max_depth=4,
        n_estimators=50 if MODE == "local" else 100,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
)
"""


# ---------------------------------------------------------------------
# Base learner hyperparameters for stacking models
# ---------------------------------------------------------------------
LOCAL = MODE == "local"
RF_CONFIG = dict(numTrees=20 if LOCAL else 100, maxDepth=6 if LOCAL else 10)
GBT_CONFIG = dict(maxIter=30 if LOCAL else 150, maxDepth=4 if LOCAL else 6,
                  stepSize=0.1 if LOCAL else 0.05)
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Corrected stack: meta-learner context features (pruned)
# ---------------------------------------------------------------------
META_CONTEXT_COLS = [
    "region_cat_idx", "sin_month", "cos_month",
    "dist_to_grounding_line", "delta_h",
]
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Grounding line bucket splits (for XGBoost model)
# ---------------------------------------------------------------------
GL_BUCKET_SPLITS = [
    float("-inf"),
    5000.0,
    20000.0,
    50000.0,
    100000.0,
    float("inf")
]
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Ocean PCA config
# ---------------------------------------------------------------------
OCEAN_COLS = [
    "thetao_mo", "t_star_mo", "so_mo", "t_f_mo",
    "t_star_quarterly_avg", "t_star_quarterly_std",
    "thetao_quarterly_avg", "thetao_quarterly_std",
]
PCA_K = 4
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# PolynomialExpansion config (classic model)
# ---------------------------------------------------------------------
POLY_INPUT_COLS = ["t_star_mo", "ice_draft", "dist_to_grounding_line"]
# ---------------------------------------------------------------------


# In[40]:


# =====================================================================
# 3. SPARK SESSION
# =====================================================================


# ---------------------------------------------------------------------
# get_spark
# ---------------------------------------------------------------------
def get_spark() -> SparkSession:
    scratch = os.environ.get("TMPDIR", os.path.join(os.getcwd(), "spark_scratch"))

    shared = {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.sql.parquet.filterPushdown": "true",
        "spark.sql.parquet.mergeSchema": "false",
        "spark.network.timeout": "7200s",
        "spark.executor.heartbeatInterval": "300s",
        "spark.local.dir": scratch,
        "spark.sql.debug.maxToStringFields": "2000",
        "spark.shuffle.compress": "true",
        "spark.shuffle.spill.compress": "true",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
    }

    builder = SparkSession.builder.appName("AntarcticUnifiedML")

    if MODE == "local":
        builder = (
            builder.master("local[4]")
            .config("spark.driver.memory", "8g")
            .config("spark.sql.shuffle.partitions", "8")
        )
    elif MODE == "sdsc":
        # local[12]: half the parallelism → double the per-task memory.
        # Off-heap REMOVED: it competed with XGBoost's native DMatrix
        # allocation for physical RAM, tipping the OS OOM reaper.
        # storageFraction lowered so execution memory can evict cached
        # blocks freely during XGBoost barrier stages.
        builder = (
            builder.master("local[12]")
            .config("spark.driver.memory", "80g")
            .config("spark.driver.maxResultSize", "4g")
            .config("spark.sql.shuffle.partitions", "1200")
            .config("spark.local.dir", scratch_dir)
            .config("spark.memory.fraction", "0.6")
            .config("spark.memory.storageFraction", "0.2")
            .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128m")
        )
    else:  # hpc
        # To use YARN:  export SPARK_MASTER_URL=yarn  before launching.
        # Without that env-var this falls back to local[12] so the
        # notebook stays runnable on a single Slurm node.
        master = os.environ.get("SPARK_MASTER_URL", "local[24]")
        builder = (
            builder
            .master(master)
            .config("spark.executor.instances", "6")
            .config("spark.executor.cores", "4")
            .config("spark.executor.memory", "20g")
            .config("spark.yarn.executor.memoryOverhead", "10g")
            .config("spark.driver.memory", "80g")
            .config("spark.driver.maxResultSize", "4g")
            .config("spark.sql.shuffle.partitions", "800")
            .config("spark.local.dir", scratch_dir)
            .config("spark.memory.fraction", "0.5")
            .config("spark.memory.storageFraction", "0.1")
            .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true")
        )

    for k, v in shared.items():
        builder = builder.config(k, v)

    spark = builder.getOrCreate()
    jvm_max_gb = spark._jvm.Runtime.getRuntime().maxMemory() / (1024**3)
    cores = spark.sparkContext.defaultParallelism
    print(f"[spark] JVM max heap: {jvm_max_gb:.1f} GB | parallelism: {cores}")
    return spark
# ---------------------------------------------------------------------


# In[41]:


# =====================================================================
# 4A. RAW DATA LOADING  (used when FEATURE_SOURCE == "from_raw")
# =====================================================================


# ---------------------------------------------------------------------
# _assign_region_from_bounds: Assign regional_subset_id from EPSG:3031 bounding boxes.
# ---------------------------------------------------------------------
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
    df = df.filter(F.col("regional_subset_id") != "other")
    print("[load] Region assignment applied; rows outside bounding boxes dropped (lazy).")
    return df
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# _derive_month_idx: Derive month_idx from exact_time if not already present.
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# load_raw_data: Read raw Parquet data and tag each row with its region.
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------


# In[42]:


# =====================================================================
# 4B. LABEL CONSTRUCTION  (used when FEATURE_SOURCE == "from_raw")
# =====================================================================


# ---------------------------------------------------------------------
# build_label: Construct basal_loss_agreement and immediately purge lwe_fused.
# ---------------------------------------------------------------------
def build_label(df: DataFrame) -> DataFrame:
    """Construct basal_loss_agreement and immediately purge lwe_fused.

    Uses groupBy + broadcast-join instead of window aggregations to
    avoid OOM-inducing shuffle-sorts on large mascon partitions.
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

    # LEAKAGE FIREWALL
    df = df.drop("_grace_p25", "lwe_fused")
    assert "lwe_fused" not in df.columns, "LEAKAGE BUG: lwe_fused survived!"
    print(f"[label] label_mode={LABEL_MODE!r}. lwe_fused confirmed absent.")
    return df
# ---------------------------------------------------------------------


# In[43]:


# =====================================================================
# 4C. BASE FEATURE ENGINEERING  (used when FEATURE_SOURCE == "from_raw")
# =====================================================================


# ---------------------------------------------------------------------
# assign_regions: Lazy null-filter + bed_below_sea_level (no eager .count() action).
# ---------------------------------------------------------------------
def assign_regions(df: DataFrame) -> DataFrame:
    """Lazy null-filter + bed_below_sea_level (no eager .count() action)."""
    df = df.filter(F.col("regional_subset_id").isNotNull())
    df = df.withColumn("bed_below_sea_level", (F.col("bed") < 0).cast(IntegerType()))
    print("[features] bed_below_sea_level added (region nulls filtered lazily).")
    return df
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# add_static_features: Static geometry interactions — no shuffles.
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# add_dynamic_features: Expanding-window pixel statistics and lagged surface slope.
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# add_ocean_features: Ocean-ice interactions + regional thermal anomaly.
# ---------------------------------------------------------------------
def add_ocean_features(df: DataFrame) -> DataFrame:
    """Ocean-ice interactions + regional thermal anomaly.

    Regional aggregate uses groupBy + broadcast-join (~216 rows) instead
    of a Window that would shuffle millions of rows per partition.
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
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# add_context_features: Cyclical month encoding, mascon aggregates, regional percentile.
# ---------------------------------------------------------------------
def add_context_features(df: DataFrame) -> DataFrame:
    """Cyclical month encoding, mascon aggregates, regional percentile.

    All group-level aggregations use groupBy + broadcast-join instead
    of Window functions to avoid shuffle-sort memory pressure.
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
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# add_sample_weights: Regional importance × class balance weights, normalised on training partition.
# ---------------------------------------------------------------------
def add_sample_weights(df: DataFrame) -> DataFrame:
    """Regional importance × class balance weights, normalised on training partition."""

    # Regional importance
    rw_expr = F.lit(1.0)
    for region, weight in REGION_WEIGHTS.items():
        rw_expr = F.when(F.col("regional_subset_id") == region, F.lit(weight)).otherwise(rw_expr)
    df = df.withColumn("regional_weight", rw_expr)

    # Class balance (training-only stats)
    # sdsc fix
    #train_slice = df.filter(F.col("month_idx") <= TRAIN_MAX_MONTH_IDX)
    train_slice = df.filter(F.col("month_idx").between(TRAIN_MIN_MONTH_IDX, TRAIN_MAX_MONTH_IDX))
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
        #df.filter(F.col("month_idx") <= TRAIN_MAX_MONTH_IDX)
        df.filter(F.col("month_idx").between(TRAIN_MIN_MONTH_IDX, TRAIN_MAX_MONTH_IDX))
        .agg(F.avg("raw_weight")).collect()[0][0]
    )
    print(f"[weights] Training mean raw weight = {train_mean:.6f}")

    df = df.withColumn(WEIGHT_COL, F.col("raw_weight") / F.lit(train_mean))
    df = df.drop("regional_weight", "class_ratio", "class_balance_weight", "raw_weight")
    print("[weights] Sample weights computed and normalised.")
    return df
# ---------------------------------------------------------------------


# In[44]:


# =====================================================================
# 4D. ADDITIONAL FEATURES  (Model 2/3 extras, applied to all models)
# =====================================================================


# ---------------------------------------------------------------------
# add_temporal_memory_features: 6-month rolling averages and rate-of-change (Model 2 features).
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# add_trajectory_features: Momentum and acceleration features (Model 3 features).
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# add_physics_interactions: Hand-crafted physics interaction features (Model 2 features).
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# add_region_integer_encoding: Integer-encode region for native categorical support.
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------


# In[45]:


# =====================================================================
# 4E. UNIFIED FEATURE ENGINEERING RUNNER
# =====================================================================


# ---------------------------------------------------------------------
# _flush: Write to Parquet and read back to truncate Spark's DAG lineage.
# ---------------------------------------------------------------------
def _flush(df: DataFrame, spark: SparkSession, tag: str) -> DataFrame:
    """Write to Parquet and read back to truncate Spark's DAG lineage.

    This forces materialisation of all pending transformations, releases
    the intermediate shuffle state held by the ShuffleExternalSorter,
    and gives the next stage a clean, shallow execution plan.
    """
    path = os.path.join(scratch_dir, f"_flush_{tag}")
    df.write.mode("overwrite").parquet(path)
    print(f"[flush] Lineage truncated -> {tag}")
    return spark.read.parquet(path)
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# _build_region_features
# ---------------------------------------------------------------------
def _build_region_features(
    rdf: DataFrame, spark: SparkSession, tag: str = "all",
) -> DataFrame:
    """Run all feature engineering on a single-region DataFrame.

    Two key structural changes vs. the original:
    1. An explicit repartition by (x, y) co-locates pixel data so that
       all pixel-level windows reuse a single shuffle exchange.
    2. A Parquet flush between the two window batches caps DAG depth
       and releases executor memory held by prior shuffle sorters.
    """
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
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# run_feature_engineering: Build all features.
# ---------------------------------------------------------------------
def run_feature_engineering(spark: SparkSession) -> None:
    """Build all features.

    For large datasets (SDSC/HPC), processes each region independently
    (~80 M rows each) so window-operation shuffles never exceed memory.
    Sample weights are computed globally after reuniting the regions.
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
# ---------------------------------------------------------------------


# In[46]:


# =====================================================================
# 5. DATA LOADING, SPLITTING, AND UNDERSAMPLING
# =====================================================================


# ---------------------------------------------------------------------
# load_and_split: Load unified features and split temporally.
# ---------------------------------------------------------------------
def load_and_split(spark: SparkSession) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Load unified features and split temporally."""

    df = spark.read.parquet(OUTPUT_PATH)
    print(f"[load] {len(df.columns)} columns from {OUTPUT_PATH}")

    df = df.filter(F.col(LABEL_COL).isNotNull() & F.col(WEIGHT_COL).isNotNull())

    train = df.filter(F.col("month_idx").between(TRAIN_MIN_MONTH_IDX, TRAIN_MAX_MONTH_IDX))
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
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# undersample: Undersample majority class to handle extreme imbalance.
# ---------------------------------------------------------------------
def undersample(train: DataFrame, checkpoint_dir: str = None) -> DataFrame:
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

    # CRITICAL: break DAG lineage before any downstream fit/transform.
    # Without this, every pipeline stage re-scans 199M rows from disk.
    ckpt_path = checkpoint_dir or os.path.join(scratch_dir, "_undersample_ckpt")
    spark = train.sparkSession
    spark.sparkContext.setCheckpointDir(ckpt_path)
    print("[undersample] Checkpointing balanced set (breaking lineage)...")
    balanced = balanced.checkpoint(eager=True)
    print(f"[undersample] Checkpoint complete: {balanced.count():,} rows")
    return balanced
# ---------------------------------------------------------------------


# In[47]:


# =====================================================================
# 6. PREPROCESSING PIPELINES
# =====================================================================


# ---------------------------------------------------------------------
# get_all_numeric_cols: Return all numeric feature columns available in the DataFrame.
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------


# In[48]:


# ---------------------------------------------------------------------
# build_xgb_preprocessing: XGBoost model: Imputer -> Bucketizer -> OHE -> Assembler -> MinMaxScaler.
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# build_stack_preprocessing: Stacking models: Imputer -> Assembler -> Normalizer (L2).
# ---------------------------------------------------------------------
def build_stack_preprocessing(available: List[str]) -> Pipeline:
    """Stacking models: Imputer -> grouped PCA -> Assembler -> Normalizer (L2).

    PCA applied to three correlated feature groups:
    - Ocean variables (physically correlated)
    - Temporal memory (linear combinations of same signals)
    - Trajectory (momentum/acceleration pairs)
    Remaining features pass through directly.
    """

    numeric = get_all_numeric_cols(available)
    imputed = [f"{c}_imp" for c in numeric]

    # Correlated groups — PCA reduces these
    ocean_imp      = [f"{c}_imp" for c in OCEAN_COLS if c in available]
    temporal_imp   = [f"{c}_imp" for c in [
                        "t_star_6mo_avg", "lwe_6mo_avg", "delta_h_6mo_avg",
                        "t_star_rate", "lwe_rate", "delta_h_rate",
                        "t_star_mom_change", "delta_h_mom_change",
                      ] if c in available]
    trajectory_imp = [f"{c}_imp" for c in [
                        "delta_h_momentum", "delta_h_acceleration", "delta_h_deseason",
                        "t_star_momentum", "t_star_sustained_anomaly",
                        "lwe_momentum", "lwe_sustained_trend",
                      ] if c in available]

    pca_cols   = set(ocean_imp + temporal_imp + trajectory_imp)
    remaining  = [c for c in imputed if c not in pca_cols]

    stages = [
        Imputer(strategy="median", inputCols=numeric, outputCols=imputed),
    ]

    final_inputs = list(remaining)

    # Ocean PCA
    if ocean_imp:
        stages += [
            VectorAssembler(inputCols=ocean_imp, outputCol="ocean_vec",
                            handleInvalid="skip"),
            PCA(k=min(4, len(ocean_imp)), inputCol="ocean_vec",
                outputCol="ocean_pca"),
        ]
        final_inputs.append("ocean_pca")

    # Temporal PCA
    if temporal_imp:
        stages += [
            VectorAssembler(inputCols=temporal_imp, outputCol="temporal_vec",
                            handleInvalid="skip"),
            PCA(k=min(3, len(temporal_imp)), inputCol="temporal_vec",
                outputCol="temporal_pca"),
        ]
        final_inputs.append("temporal_pca")

    # Trajectory PCA
    if trajectory_imp:
        stages += [
            VectorAssembler(inputCols=trajectory_imp, outputCol="trajectory_vec",
                            handleInvalid="skip"),
            PCA(k=min(3, len(trajectory_imp)), inputCol="trajectory_vec",
                outputCol="trajectory_pca"),
        ]
        final_inputs.append("trajectory_pca")

    # Final assembly + normalize
    total = len(remaining) + (4 if ocean_imp else 0) + \
            (3 if temporal_imp else 0) + (3 if trajectory_imp else 0)

    stages += [
        VectorAssembler(inputCols=final_inputs, outputCol="raw_features",
                        handleInvalid="skip"),
        Normalizer(inputCol="raw_features", outputCol="features", p=2.0),
    ]

    print(f"[preprocess:stack] {len(numeric)} numeric → PCA groups → "
          f"{total} features + Normalizer(L2)")
    return Pipeline(stages=stages)
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# build_classic_preprocessing: Classic model: Imputer -> StringIndexer -> OHE -> PCA(ocean) ->
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------


# In[49]:


# =====================================================================
# 7. MODEL INITIALISATION
# =====================================================================


# ---------------------------------------------------------------------
# init_xgb_models: SparkXGBClassifier baseline + tuned.
# ---------------------------------------------------------------------
def init_xgb_models() -> List[Tuple[str, object]]:
    """SparkXGBClassifier baseline only (tuned commented out to save time)."""

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

    #is_local = MODE in ("local", "sdsc")
    num_workers = 1 #if is_local else 6

    models = []
    for name, cfg in XGB_CONFIGS.items():
        m = SparkXGBClassifier(
            features_col="features",
            label_col=LABEL_COL,
            weight_col=WEIGHT_COL,
            eval_metric="logloss",
            use_gpu=False,
            missing=0.0,
            num_workers=num_workers,
            **cfg,
        )
        models.append((name, m))
    return models
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# init_base_learners: RF + GBT base learners for stacking.
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# init_stack_meta_learners: XGBoost meta-learners for uncorrected stacking.
# ---------------------------------------------------------------------
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
                use_gpu=False, num_workers=2 if LOCAL else 4, missing=0.0)),
            ("Stack_Tuned", SparkXGBClassifier(
                features_col="meta_features", label_col=LABEL_COL,
                weight_col=WEIGHT_COL, max_depth=4 if LOCAL else 6,
                n_estimators=50 if LOCAL else 300,
                learning_rate=0.05 if LOCAL else 0.02,
                subsample=0.75, min_child_weight=15 if LOCAL else 20,
                reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
                use_gpu=False, num_workers=2 if LOCAL else 4, missing=0.0)),
        ]
    except ImportError:
        return [
            ("Stack_GBT_proxy", GBTClassifier(
                labelCol=LABEL_COL, featuresCol="meta_features",
                weightCol=WEIGHT_COL, maxIter=50, maxDepth=3, stepSize=0.1, seed=99)),
        ]
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# init_corrected_meta_learners: LogisticRegression meta-learners for corrected stacking.
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# init_classic_models: DecisionTree -> RandomForest -> GBT progression.
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------


# In[50]:


# =====================================================================
# 8. STACKING HELPERS
# =====================================================================


# ---------------------------------------------------------------------
# build_full_meta_features: Build meta-features for uncorrected stacking (all features + base preds).
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# build_pruned_meta_features: Build PRUNED meta-features for corrected stacking (base preds + 5 context only).
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# oof_split: Split training data temporally for OOF stacking.
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------


# In[51]:


# =====================================================================
# 9. EVALUATION
# =====================================================================


# ---------------------------------------------------------------------
# evaluate: Compute ROC-AUC, PR-AUC, F1, Precision, Recall.
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# regional_summary: Print per-region prediction stats.
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# save_predictions: Save 500-row prediction sample.
# ---------------------------------------------------------------------
def save_predictions(preds: DataFrame, model_name: str, split_name: str) -> None:
    """Save 500-row prediction sample."""

    cols = [c for c in KEY_COLS if c in preds.columns]
    cols += [LABEL_COL, PREDICTION_COL, WEIGHT_COL]
    if "probability" in preds.columns:
        cols.append("probability")

    path = os.path.join(OUTPUT_DIR, f"preds_{model_name}_{split_name}")
    preds.select(*cols).limit(500).write.mode("overwrite").parquet(path)
    print(f"  [{model_name}] Saved -> {path}")
# ---------------------------------------------------------------------


# In[52]:


# =====================================================================
# 10. TRAIN AND TEST
# =====================================================================


# ---------------------------------------------------------------------
# train_xgb: Train SparkXGBClassifier models.
# ---------------------------------------------------------------------
def train_xgb(train: DataFrame, val: DataFrame, test: DataFrame) -> List[Dict]:
    """Train SparkXGBClassifier models."""

    spark = train.sparkSession
    train_pp_path = os.path.join(scratch_dir, "_xgb_train_pp")

    # ------------------------------------------------------------------
    # PREPROCESSING — skip if already on disk from a previous run
    # ------------------------------------------------------------------
    if os.path.exists(train_pp_path) and os.listdir(train_pp_path):
        print(f"[xgb] Preprocessed data found, skipping to training.")
        # Refit preprocessor on a sample of train for val/test inference.
        # We cannot use train_pp_path (it only has features/label/weight).
        # 2M rows is enough for stable Imputer medians + MinMaxScaler range.
        print("[xgb] Refitting preprocessor on sample for val/test inference...")
        pp_model = build_xgb_preprocessing(train.columns).fit(
            train.sample(fraction=0.01, seed=42)
        )
        print("[xgb] Preprocessor refit complete.")

    else:
        # Step 1: checkpoint raw train to break lineage from 199M-row parquet
        train_raw_path = os.path.join(scratch_dir, "_xgb_train_raw")
        print("[xgb] Checkpointing raw train split to disk...")
        train.write.mode("overwrite").parquet(train_raw_path)
        train_ckpt = spark.read.parquet(train_raw_path)
        print("[xgb] Raw train checkpoint complete.")

        # Step 2: undersample -> checkpoint -> preprocess -> write
        balanced = undersample(
            train_ckpt,
            checkpoint_dir=os.path.join(scratch_dir, "_us_ckpt")
        )

        preprocess = build_xgb_preprocessing(balanced.columns)
        print("[xgb] Fitting preprocessor on undersampled data...")
        pp_model = preprocess.fit(balanced)
        print("[xgb] Preprocessor fit complete.")

        # Step 3: transform then select — features only exists AFTER transform
        print("[xgb] Transforming and writing preprocessed data...")
        transformed = pp_model.transform(balanced)
        (transformed
         .select("features", LABEL_COL, WEIGHT_COL)   # only 3 cols XGBoost needs
         .repartition(600)
         .write.mode("overwrite")
         .parquet(train_pp_path))
        print("[xgb] Preprocessed training data written.")

    # ------------------------------------------------------------------
    # LOAD — always reload from disk, DISK_ONLY leaves RAM for XGBoost DMatrix
    # ------------------------------------------------------------------
    train_p = spark.read.parquet(train_pp_path).persist(StorageLevel.DISK_ONLY)
    n_train = train_p.count()
    print(f"[xgb] Training data ready: {n_train:,} rows | "
          f"columns: {train_p.columns}")  # sanity check — features must be here

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------
    results = []
    for name, clf in init_xgb_models():
        print(f"\n{'='*60}\n  TRAINING: {name}\n{'='*60}")
        fitted = Pipeline(stages=[clf]).fit(train_p)

        # Train metrics on cached train_p
        p_train = fitted.transform(train_p).persist(StorageLevel.DISK_ONLY)
        results.append(evaluate(name, p_train, "train"))
        save_predictions(p_train, name, "train")
        p_train.unpersist()

        # Val / test: preprocess with pp_model then transform with fitted XGB
        for sname, raw_df in [("val", val), ("test", test)]:
            p = fitted.transform(pp_model.transform(raw_df))
            p = p.persist(StorageLevel.DISK_ONLY)
            results.append(evaluate(name, p, sname))
            save_predictions(p, name, sname)
            if sname == "test":
                regional_summary(p, name)
            p.unpersist()

    train_p.unpersist()
    return results
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# train_stack: Train uncorrected stacking ensemble.
# ---------------------------------------------------------------------
def train_stack(train: DataFrame, val: DataFrame, test: DataFrame) -> List[Dict]:
    """Train uncorrected stacking ensemble."""

    spark = train.sparkSession
    preprocess = build_stack_preprocessing(train.columns)
    pp_model = preprocess.fit(train)

    numeric = get_all_numeric_cols(train.columns)
    imputed = [f"{c}_imp" for c in numeric]

    train_pp_path = os.path.join(scratch_dir, "_stack_train_pp")
    pp_model.transform(undersample(train)).write.mode("overwrite").parquet(train_pp_path)
    train_p = spark.read.parquet(train_pp_path).persist(StorageLevel.MEMORY_AND_DISK)
    train_p.count()
    print("[stack] Preprocessed training data flushed and reloaded.")

    results = []
    base_models = {}

    # Layer 1: base learners
    print(f"\n{'='*60}\n  LAYER 1: BASE LEARNERS\n{'='*60}")
    for name, clf in init_base_learners():
        print(f"\n  Training {name}...")
        fitted = Pipeline(stages=[clf]).fit(train_p)
        base_models[name] = fitted

        for sname, raw_df in [("train", train), ("val", val), ("test", test)]:
            p = fitted.transform(pp_model.transform(raw_df))
            pred_col = "rf_prediction" if "RF" in name else "gbt_prediction"
            ev = p.withColumn(PREDICTION_COL, F.col(pred_col))
            if "rf_rawPrediction" in ev.columns:
                ev = ev.withColumn("rawPrediction", F.col("rf_rawPrediction"))
            ev = ev.persist(StorageLevel.DISK_ONLY)
            results.append(evaluate(name, ev, sname))
            ev.unpersist()

    # Layer 2: stack
    print(f"\n{'='*60}\n  LAYER 2: META-LEARNER\n{'='*60}")

    def add_base(df):
        for _, m in base_models.items():
            df = m.transform(df)
        return df

    train_fit_s = build_full_meta_features(add_base(train_p), imputed)

    for name, clf in init_stack_meta_learners():
        print(f"\n  Training {name}...")
        fitted = Pipeline(stages=[clf]).fit(train_fit_s)
        for sname, raw_df in [("train", train), ("val", val), ("test", test)]:
            sdf_m = build_full_meta_features(
                add_base(pp_model.transform(raw_df)), imputed)
            p = fitted.transform(sdf_m)
            p = p.persist(StorageLevel.DISK_ONLY)
            results.append(evaluate(name, p, sname))
            save_predictions(p, name, sname)
            if sname == "test":
                regional_summary(p, name)
            p.unpersist()

    train_p.unpersist()
    return results
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# train_corrected_stack: Train corrected stacking ensemble (OOF + pruned + LR).
# ---------------------------------------------------------------------
def train_corrected_stack(train: DataFrame, val: DataFrame, test: DataFrame) -> List[Dict]:
    """Train corrected stacking ensemble (OOF + pruned + LR)."""

    spark = train.sparkSession
    preprocess = build_stack_preprocessing(train.columns)
    pp_model = preprocess.fit(train)

    available_cols = train.columns
    results = []
    base_models = {}

    # Flush preprocessed full train (not undersampled) for OOF split
    train_pp_path = os.path.join(scratch_dir, "_cstack_train_pp")
    pp_model.transform(train).write.mode("overwrite").parquet(train_pp_path)
    train_p = spark.read.parquet(train_pp_path)

    # OOF split on flushed data
    fold_a_raw, fold_b_raw = oof_split(train_p)
    fold_a = undersample(fold_a_raw).persist(StorageLevel.MEMORY_AND_DISK)
    fold_a.count()

    # Layer 1: base learners on fold A
    print(f"\n{'='*60}\n  LAYER 1: BASE LEARNERS ON FOLD A\n{'='*60}")
    for name, clf in init_base_learners():
        print(f"\n  Training {name}...")
        fitted = Pipeline(stages=[clf]).fit(fold_a)
        base_models[name] = fitted

        eval_splits = [
            ("fold_a", fold_a_raw),
            ("fold_b", fold_b_raw),
            ("val", pp_model.transform(val)),
            ("test", pp_model.transform(test)),
        ]
        for sname, sdf in eval_splits:
            p = fitted.transform(sdf)
            pred_col = "rf_prediction" if "RF" in name else "gbt_prediction"
            ev = p.withColumn(PREDICTION_COL, F.col(pred_col))
            if "rf_rawPrediction" in ev.columns:
                ev = ev.withColumn("rawPrediction", F.col("rf_rawPrediction"))
            ev = ev.persist(StorageLevel.DISK_ONLY)
            results.append(evaluate(name, ev, sname))
            ev.unpersist()

    fold_a.unpersist()

    # Layer 2: corrected meta-learner on fold B
    print(f"\n{'='*60}\n  LAYER 2: LR META-LEARNER ON FOLD B\n{'='*60}")

    def add_base(df):
        for _, m in base_models.items():
            df = m.transform(df)
        return df

    # Persist fold_b meta-features: LR iterates over this many times.
    # Only ~8 meta dimensions so the footprint is small.
    fold_b_m = build_pruned_meta_features(add_base(fold_b_raw), available_cols)
    fold_b_m = fold_b_m.persist(StorageLevel.MEMORY_AND_DISK)

    n_meta = fold_b_m.select("meta_features").head(1)[0]["meta_features"].size
    print(f"  Meta-features: {n_meta} dimensions (pruned)")

    for name, clf in init_corrected_meta_learners():
        print(f"\n  Training {name}...")
        fitted = Pipeline(stages=[clf]).fit(fold_b_m)

        lr = fitted.stages[-1]
        if hasattr(lr, "coefficients"):
            coeffs = lr.coefficients.toArray()
            print(f"    Intercept: {lr.intercept:.4f}")
            meta_names = ["rf_pos_prob", "gbt_score", "base_agreement"] + \
                         [f"{c}_meta" for c in META_CONTEXT_COLS]
            for n, c in zip(meta_names[:len(coeffs)], coeffs):
                sign = "+" if c >= 0 else "-"
                print(f"    {n:30s}  {sign}{abs(c):.4f}")

        for sname, raw_df in [("train", train), ("val", val), ("test", test)]:
            sdf = pp_model.transform(raw_df)
            p = fitted.transform(
                build_pruned_meta_features(add_base(sdf), available_cols))
            p = p.persist(StorageLevel.DISK_ONLY)
            results.append(evaluate(name, p, sname))
            save_predictions(p, name, sname)
            if sname == "test":
                regional_summary(p, name)
            p.unpersist()

    fold_b_m.unpersist()
    return results
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# train_classic: Train the DecisionTree -> RandomForest -> GBT progression.
# ---------------------------------------------------------------------
def train_classic(train: DataFrame, val: DataFrame, test: DataFrame) -> List[Dict]:
    """Train the DecisionTree -> RandomForest -> GBT progression.

    Uses PolynomialExpansion + StandardScaler + Ocean PCA preprocessing.
    """

    spark = train.sparkSession
    preprocess = build_classic_preprocessing(train.columns)
    pp_model = preprocess.fit(train)

    train_pp_path = os.path.join(scratch_dir, "_classic_train_pp")
    pp_model.transform(undersample(train)).write.mode("overwrite").parquet(train_pp_path)
    train_p = spark.read.parquet(train_pp_path).persist(StorageLevel.MEMORY_AND_DISK)
    train_p.count()
    print("[classic] Preprocessed training data flushed and reloaded.")

    n_features = train_p.select("features").head(1)[0]["features"].size
    print(f"  Feature vector dimension: {n_features}")

    results = []
    for name, clf in init_classic_models():
        print(f"\n{'='*60}\n  TRAINING: {name}\n{'='*60}")
        fitted = Pipeline(stages=[clf]).fit(train_p)

        for sname, raw_df in [("train", train), ("val", val), ("test", test)]:
            p = fitted.transform(pp_model.transform(raw_df))
            p = p.persist(StorageLevel.DISK_ONLY)
            results.append(evaluate(name, p, sname))
            save_predictions(p, name, sname)
            if sname == "test":
                regional_summary(p, name)
            p.unpersist()

        tree_model = fitted.stages[-1]
        if hasattr(tree_model, "featureImportances"):
            importances = tree_model.featureImportances.toArray()
            pairs = sorted(enumerate(importances), key=lambda p: p[1], reverse=True)
            print(f"\n  [{name}] Top 10 features:")
            for idx, imp in pairs[:10]:
                bar = "#" * int(imp * 100)
                print(f"    feat[{idx:3d}]  {imp:.4f}  {bar}")

    train_p.unpersist()
    return results


TRAIN_MAP = {
    "xgb": train_xgb,
    "stack": train_stack,
    "corrected_stack": train_corrected_stack,
    "classic": train_classic,
}
# ---------------------------------------------------------------------


# In[53]:


# =====================================================================
# 11. ANALYSIS
# =====================================================================


# ---------------------------------------------------------------------
# print_results_table: Print formatted results summary.
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# fitting_analysis: Diagnose overfitting/underfitting.
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# plot_geographic_errors: Generate Plotly geographic error visualizations.
# ---------------------------------------------------------------------
def plot_geographic_errors(preds: DataFrame, model_name: str) -> None:
    """Generate Plotly geographic error visualizations."""
    # note: maybe conda install
    #import subprocess, sys
    #subprocess.run([sys.executable, "-m", "pip", "install", "plotly", "pandas", "--quiet"], check=False)

    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd
    except ImportError:
        print(f"  [{model_name}] Plotly unavailable even after install attempt, skipping.")
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
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# plot_temporal_residuals: Generate Plotly temporal error-rate plots (monthly FNR / FPR over time).
# ---------------------------------------------------------------------
def plot_temporal_residuals(preds: DataFrame, model_name: str) -> None:
    """Generate Plotly temporal error-rate plots (monthly FNR / FPR over time)."""
    # note: i think it need conda install or conda add idk
    #import subprocess, sys
    #subprocess.run([sys.executable, "-m", "pip", "install", "plotly", "pandas", "--quiet"], check=False)

    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd
    except ImportError:
        print(f"  [{model_name}] Plotly unavailable even after install attempt, skipping.")
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
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# print_conclusion: Print conclusion for rubric.
# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------


# In[54]:


# =====================================================================
# 12. MAIN EXECUTION
# =====================================================================


# ---------------------------------------------------------------------
# run_all: Main entry point.  Call from notebook or as script.
# ---------------------------------------------------------------------
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
                try:
                    plot_geographic_errors(test_preds, mname)
                    plot_temporal_residuals(test_preds, mname)
                except:
                    print(f"  [{mname}] Plot failed: {e}, continuing.")
            else:
                print(f"  [{mname}] No test predictions found at {pred_path}")

        print_conclusion()

        return results

    finally:
        spark.stop()
# ---------------------------------------------------------------------


# In[55]:


run_all()
