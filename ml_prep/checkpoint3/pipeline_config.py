# =====================================================================
# pipeline_config.py — Shared configuration, Spark session, evaluation,
# plotting, and data loading utilities for modular ML pipeline scripts.
# =====================================================================

from __future__ import annotations

import os
import math
from typing import Dict, List

# ---------------------------------------------------------------------
# PySpark
# ---------------------------------------------------------------------
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType
from pyspark import StorageLevel
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Matplotlib (headless-safe for HPC)
# ---------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------


# =====================================================================
# 1. CONSTANTS
# =====================================================================

# ---------------------------------------------------------------------
# Set User: just username on SDSC Expanse
# ---------------------------------------------------------------------
USER = "rrogers"
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DATA_ROOT   = os.path.join(os.getcwd(), "data/fused_data")
INPUT_PATH  = os.path.join(os.getcwd(), "data/ml_ready")
OUTPUT_PATH = os.path.join(os.getcwd(), "data/ml_ready_unified")
OUTPUT_DIR  = os.path.join(os.getcwd(), "dataunified_output")
scratch_dir = os.environ.get(
    "LOCAL_SCRATCH",
    "/expanse/lustre/projects/uci157/rrogers/temp",
)
os.makedirs(scratch_dir, exist_ok=True)
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Execution mode
# note: "local" for local testing, "hpc" for running on SDSC Expanse
# ---------------------------------------------------------------------
MODE = "hpc"
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Flush partitions (lineage truncation)
# ---------------------------------------------------------------------
FLUSH_PARTITIONS = 8 if MODE == "local" else 400
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------
LABEL_COL      = "basal_loss_agreement"
WEIGHT_COL     = "weightCol"
PREDICTION_COL = "prediction"
KEY_COLS       = ["x", "y", "month_idx", "mascon_id", "regional_subset_id"]
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Temporal split boundaries
# note: using April 2020 because of 2019 spike
# ---------------------------------------------------------------------
TRAIN_MIN_MONTH_IDX = 24244   # Apr 2020
TRAIN_MAX_MONTH_IDX = 24276   # end Dec 2022
VAL_MAX_MONTH_IDX   = 24286   # end Oct 2023
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Class imbalance
# ---------------------------------------------------------------------
UNDERSAMPLE_ENABLED   = True
UNDERSAMPLE_NEG_RATIO = 10    # keep 1:10 pos:neg ratio
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Region files (HPC pre-split parquets)
# ---------------------------------------------------------------------
REGION_FILES = {
    "amundsen_sea":        "ml_subset_amundsen_sea.parquet",
    "antarctic_peninsula": "ml_subset_antarctic_peninsula.parquet",
    "lambert_amery":       "ml_subset_lambert_amery.parquet",
    "ronne":               "ml_subset_ronne.parquet",
    "ross":                "ml_subset_ross.parquet",
    "totten_and_aurora":   "ml_subset_totten_and_aurora.parquet",
}
SPARSE_SAMPLE = "antarctica_sparse_features_sample.parquet"
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# EPSG:3031 bounding boxes for region assignment
# ---------------------------------------------------------------------
REGION_BOUNDS = {
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
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Region weights (upweight scientifically critical basins)
# ---------------------------------------------------------------------
REGION_WEIGHTS = {
    "amundsen_sea":        2.0,
    "totten_and_aurora":   2.0,
    "antarctic_peninsula": 1.5,
    "lambert_amery":       1.0,
    "ross":                0.7,
    "ronne":               0.7,
}
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Ocean PCA config (shared by stacking + classic preprocessing)
# ---------------------------------------------------------------------
OCEAN_COLS = [
    "thetao_mo", "t_star_mo", "so_mo", "t_f_mo",
    "t_star_quarterly_avg", "t_star_quarterly_std",
    "thetao_quarterly_avg", "thetao_quarterly_std",
]
PCA_K = 4
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Convenience flag
# ---------------------------------------------------------------------
LOCAL = MODE == "local"
# ---------------------------------------------------------------------
# =====================================================================


# =====================================================================
# 2. SPARK SESSION
# =====================================================================

def get_spark():
    """Create or retrieve a SparkSession for local or HPC."""

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
    # hpc: assumes 128GB RAM per node and 32 cores per node
    else:  
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

# =====================================================================


# =====================================================================
# 3. NUMERIC FEATURE REGISTRY
# =====================================================================

def get_all_numeric_cols(available):
    """Return all numeric feature columns available in the DataFrame.

        note: this is the single source of truth for which columns
              are treated as numeric features by the preprocessing
              pipelines.
    """

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

# =====================================================================


# =====================================================================
# 4. DATA LOADING, SPLITTING, AND UNDERSAMPLING
# =====================================================================

def load_and_split(spark):
    """Load unified features and split temporally."""

    df = spark.read.parquet(OUTPUT_PATH)
    print(f"[load] {len(df.columns)} columns from {OUTPUT_PATH}")

    df = df.filter(F.col(LABEL_COL).isNotNull() & F.col(WEIGHT_COL).isNotNull())

    train = df.filter(
        F.col("month_idx").between(TRAIN_MIN_MONTH_IDX, TRAIN_MAX_MONTH_IDX)
    )
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


def undersample(train, checkpoint_dir=None):
    """Undersample majority class to handle extreme imbalance.

        note: uses .checkpoint(eager=True) to break DAG lineage,
              preventing re-scanning of the full dataset during
              downstream pipeline stages.
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

    # CRITICAL: break DAG lineage before any downstream fit/transform.
    ckpt_path = checkpoint_dir or os.path.join(scratch_dir, "_undersample_ckpt")
    spark = train.sparkSession
    spark.sparkContext.setCheckpointDir(ckpt_path)
    print("[undersample] Checkpointing balanced set (breaking lineage)...")
    balanced = balanced.checkpoint(eager=True)
    print(f"[undersample] Checkpoint complete: {balanced.count():,} rows")

    return balanced

# =====================================================================


# =====================================================================
# 5. EVALUATION
# =====================================================================

def evaluate(model_name, preds, split_name):
    """Compute ROC-AUC, PR-AUC, F1, Precision, Recall."""

    label_stats = preds.agg(
        F.min(LABEL_COL).alias("mn"),
        F.max(LABEL_COL).alias("mx"),
    ).collect()[0]

    nan = float("nan")
    if label_stats["mn"] == label_stats["mx"]:
        print(f"  [{model_name}] {split_name:7s}  single class={label_stats['mn']}")
        return {
            "model": model_name, "split": split_name,
            "roc_auc": nan, "pr_auc": nan, "f1": nan,
            "precision": nan, "recall": nan,
        }

    f1 = MulticlassClassificationEvaluator(
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
        except Exception as e:
            print(f"  [{model_name}] AUC computation failed: {e}")

    print(f"  [{model_name}] {split_name:7s}  "
          f"ROC={roc_auc:.4f}  PR={pr_auc:.4f}  "
          f"F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")

    return {
        "model": model_name, "split": split_name,
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "f1": f1, "precision": prec, "recall": rec,
    }


def regional_summary(preds, model_name):
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
              f"pred={r['pred_rate']:.4f}  true={r['true_rate']:.4f}  "
              f"n={r['n']:>10,}")


def save_predictions(preds, model_name, split_name):
    """Save 500-row prediction sample."""

    cols = [c for c in KEY_COLS if c in preds.columns]
    cols += [LABEL_COL, PREDICTION_COL, WEIGHT_COL]
    if "probability" in preds.columns:
        cols.append("probability")

    path = os.path.join(OUTPUT_DIR, f"preds_{model_name}_{split_name}")
    preds.select(*cols).limit(500).write.mode("overwrite").parquet(path)
    print(f"  [{model_name}] Saved -> {path}")

# =====================================================================


# =====================================================================
# 6. ANALYSIS
# =====================================================================

def print_results_table(results):
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


def fitting_analysis(results):
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

# =====================================================================


# =====================================================================
# 7. PLOTTING (matplotlib, headless-safe)
#
# note: replaces Plotly which was unavailable on HPC Singularity.
#       uses Agg backend so no display server is needed.
#       all figures saved as PNG to OUTPUT_DIR.
# =====================================================================

def plot_geographic_errors(preds, model_name):
    """Generate matplotlib geographic error scatter plots."""

    try:
        import pandas as pd
    except ImportError:
        print(f"  [{model_name}] pandas unavailable, skipping geographic plots.")
        return

    total = preds.count()
    cols = [c for c in ["x", "y", "regional_subset_id", LABEL_COL, PREDICTION_COL]
            if c in preds.columns]

    pdf = (
        preds.select(*cols)
        .sample(fraction=min(1.0, 10000 / max(total, 1)))
        .toPandas()
    )

    if pdf.empty:
        return

    # Classify error types
    pdf["error_type"] = "TN"
    pdf.loc[(pdf[LABEL_COL] == 1) & (pdf[PREDICTION_COL] == 1), "error_type"] = "TP"
    pdf.loc[(pdf[LABEL_COL] == 0) & (pdf[PREDICTION_COL] == 1), "error_type"] = "FP"
    pdf.loc[(pdf[LABEL_COL] == 1) & (pdf[PREDICTION_COL] == 0), "error_type"] = "FN"

    cmap = {"TP": "#2ecc71", "TN": "#95a5a6", "FP": "#e67e22", "FN": "#e74c3c"}

    # Plot 1: All predictions 
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#2d2d2d")
    ax.set_facecolor("#1e1e1e")
    for etype in ["FN", "FP", "TP", "TN"]:
        subset = pdf[pdf["error_type"] == etype]
        if not subset.empty:
            ax.scatter(subset["x"], subset["y"], c=cmap[etype],
                       label=etype, alpha=0.6, s=3)
    ax.set_title(f"{model_name}: Geographic Errors (EPSG:3031)", color="white")
    ax.set_xlabel("x", color="white")
    ax.set_ylabel("y", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#3d3d3d", edgecolor="white", labelcolor="white")
    p1 = os.path.join(OUTPUT_DIR, f"{model_name}_geo_errors.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [{model_name}] -> {p1}")

    # Plot 2: Regional error rates
    if "regional_subset_id" in pdf.columns:
        regions = sorted(pdf["regional_subset_id"].unique())
        fnrs, fprs = [], []
        for r in regions:
            g = pdf[pdf["regional_subset_id"] == r]
            fnrs.append(
                (g["error_type"] == "FN").sum() / max(1, (g[LABEL_COL] == 1).sum())
            )
            fprs.append(
                (g["error_type"] == "FP").sum() / max(1, (g[LABEL_COL] == 0).sum())
            )

        x_pos = list(range(len(regions)))
        width = 0.35
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.bar([p - width / 2 for p in x_pos], fnrs, width,
                label="FNR", color="#e74c3c")
        ax2.bar([p + width / 2 for p in x_pos], fprs, width,
                label="FPR", color="#e67e22")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(regions, rotation=45, ha="right")
        ax2.set_title(f"{model_name}: Regional Error Rates")
        ax2.set_ylabel("Rate")
        ax2.legend()
        p2 = os.path.join(OUTPUT_DIR, f"{model_name}_regional_errors.png")
        fig2.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  [{model_name}] -> {p2}")

    # Plot 3: Errors only
    errs = pdf[pdf["error_type"].isin(["FP", "FN"])]
    if not errs.empty:
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        fig3.patch.set_facecolor("#2d2d2d")
        ax3.set_facecolor("#1e1e1e")
        for etype in ["FN", "FP"]:
            subset = errs[errs["error_type"] == etype]
            if not subset.empty:
                ax3.scatter(subset["x"], subset["y"], c=cmap[etype],
                            label=etype, alpha=0.8, s=4)
        ax3.set_title(f"{model_name}: Misclassified Only", color="white")
        ax3.tick_params(colors="white")
        ax3.legend(facecolor="#3d3d3d", edgecolor="white", labelcolor="white")
        p3 = os.path.join(OUTPUT_DIR, f"{model_name}_errors_only.png")
        fig3.savefig(p3, dpi=150, bbox_inches="tight", facecolor=fig3.get_facecolor())
        plt.close(fig3)
        print(f"  [{model_name}] -> {p3}")


def plot_temporal_residuals(preds, model_name):
    """Generate matplotlib temporal error-rate line plots."""

    try:
        import pandas as pd
    except ImportError:
        print(f"  [{model_name}] pandas unavailable, skipping temporal plots.")
        return

    cols = ["month_idx", LABEL_COL, PREDICTION_COL]
    if not all(c in preds.columns for c in cols):
        print(f"  [{model_name}] Missing columns for temporal plot, skipping.")
        return

    pdf = (
        preds.select(*cols)
        .groupBy("month_idx")
        .agg(
            F.count("*").alias("n"),
            F.sum(F.when(F.col(LABEL_COL) == 1, 1).otherwise(0)).alias("pos"),
            F.sum(F.when(F.col(LABEL_COL) == 0, 1).otherwise(0)).alias("neg"),
            F.sum(
                F.when(
                    (F.col(LABEL_COL) == 1) & (F.col(PREDICTION_COL) == 0), 1
                ).otherwise(0)
            ).alias("fn"),
            F.sum(
                F.when(
                    (F.col(LABEL_COL) == 0) & (F.col(PREDICTION_COL) == 1), 1
                ).otherwise(0)
            ).alias("fp"),
        )
        .orderBy("month_idx")
        .toPandas()
    )

    if pdf.empty:
        return

    pdf["fnr"] = pdf["fn"] / pdf["pos"].clip(lower=1)
    pdf["fpr"] = pdf["fp"] / pdf["neg"].clip(lower=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#2d2d2d")
    ax.set_facecolor("#1e1e1e")
    ax.plot(pdf["month_idx"], pdf["fnr"], "o-", color="#e74c3c",
            markersize=4, linewidth=2, label="FNR (miss rate)")
    ax.plot(pdf["month_idx"], pdf["fpr"], "o-", color="#e67e22",
            markersize=4, linewidth=2, label="FPR (false alarm)")
    ax.set_title(f"{model_name}: Temporal Error Rates by Month", color="white")
    ax.set_xlabel("month_idx", color="white")
    ax.set_ylabel("Error Rate", color="white")
    ax.set_ylim(0, 1)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#3d3d3d", edgecolor="white", labelcolor="white")

    path = os.path.join(OUTPUT_DIR, f"{model_name}_temporal_residuals.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [{model_name}] -> {path}")

# =====================================================================


# =====================================================================
# 8. CONCLUSION
# =====================================================================

def print_conclusion():
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
