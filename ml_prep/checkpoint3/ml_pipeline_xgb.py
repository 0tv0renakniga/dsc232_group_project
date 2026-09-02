# =====================================================================
# ml_pipeline_xgb.py: XGBoost model training.
#
# Loads unified features from data/ml_ready_unified/, applies
# XGBoost-specific preprocessing, trains SparkXGBClassifier models,
# evaluates, and generates error-analysis plots.
# =====================================================================

from __future__ import annotations

import os

# ---------------------------------------------------------------------
# PySpark
# ---------------------------------------------------------------------
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import (
    Bucketizer,
    Imputer,
    MinMaxScaler,
    OneHotEncoder,
    StringIndexer,
    VectorAssembler,
)
from pyspark.sql import functions as F
from pyspark import StorageLevel
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------
from pipeline_config import (
    MODE, LOCAL, scratch_dir, OUTPUT_DIR,
    LABEL_COL, WEIGHT_COL, PREDICTION_COL,
    get_spark, get_all_numeric_cols, load_and_split, undersample,
    evaluate, regional_summary, save_predictions,
    print_results_table, fitting_analysis,
    plot_geographic_errors, plot_temporal_residuals, print_conclusion,
)
# ---------------------------------------------------------------------


# =====================================================================
# 1. XGB-SPECIFIC CONSTANTS
# =====================================================================

# ---------------------------------------------------------------------
# Grounding line bucket splits (for Bucketizer)
# ---------------------------------------------------------------------
GL_BUCKET_SPLITS = [
    float("-inf"),
    5000.0,
    20000.0,
    50000.0,
    100000.0,
    float("inf"),
]
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# XGBoost hyperparameters and model selection
# ---------------------------------------------------------------------
# Select model to train: "XGB_Baseline" or "XGB_Tuned"
XGB_MODEL = "XGB_Baseline"
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
        max_depth=4 if LOCAL else 5,
        n_estimators=100 if LOCAL else 150,
        learning_rate=0.05,
        subsample=0.75,
        colsample_bytree=0.7,
        min_child_weight=15 if LOCAL else 20,
        reg_alpha=0.1,
        reg_lambda=1.0,
    ),
}
# ---------------------------------------------------------------------
# =====================================================================


# =====================================================================
# 2. PREPROCESSING
# =====================================================================

def build_xgb_preprocessing(available):
    """XGBoost: Imputer -> Bucketizer -> OHE -> Assembler -> MinMaxScaler."""

    numeric = get_all_numeric_cols(available)
    imputed = [f"{c}_imp" for c in numeric]

    stages = [
        Imputer(strategy="median", inputCols=numeric, outputCols=imputed),
        Bucketizer(
            splits=GL_BUCKET_SPLITS,
            inputCol="dist_to_grounding_line",
            outputCol="gl_bucket_idx",
            handleInvalid="keep",
        ),
        StringIndexer(
            inputCol="regional_subset_id",
            outputCol="region_index",
            handleInvalid="keep",
        ),
        OneHotEncoder(inputCol="gl_bucket_idx", outputCol="gl_bucket_ohe"),
        OneHotEncoder(inputCol="region_index", outputCol="region_ohe"),
        VectorAssembler(
            inputCols=imputed + ["gl_bucket_ohe", "region_ohe"],
            outputCol="raw_features",
            handleInvalid="skip",
        ),
        MinMaxScaler(inputCol="raw_features", outputCol="features"),
    ]
    print(f"[preprocess:xgb] {len(numeric)} numeric + Bucketizer + OHE + MinMaxScaler")

    return Pipeline(stages=stages)

# =====================================================================


# =====================================================================
# 3. MODEL INITIALISATION
# =====================================================================

def init_xgb_models():
    """SparkXGBClassifier models from XGB_CONFIGS."""

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

    # note: num_workers=2 avoids OOM on barrier-mode XGBoost
    #       when all executor RAM is consumed by DMatrix allocation.
    # note: ensure num_workers <= num_executors
    # note: if num_workers > num_executors, Spark will throw an error
    # note: for baseline, we might get away with 4-6 workers
    # note: for tuned, we might need 2-4 workers
    # note: if we get OOM, reduce num_workers
    num_workers = 2

    models = []
    selected = {XGB_MODEL: XGB_CONFIGS[XGB_MODEL]} if XGB_MODEL else XGB_CONFIGS
    for name, cfg in selected.items():
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

# =====================================================================


# =====================================================================
# 4. TRAINING
# =====================================================================

def train_xgb(train, val, test):
    """Train SparkXGBClassifier models."""

    spark = train.sparkSession
    train_pp_path = os.path.join(scratch_dir, "_xgb_train_pp")

    # ------------------------------------------------------------------
    # PREPROCESSING: skip if already on disk from a previous run
    # ------------------------------------------------------------------
    if os.path.exists(train_pp_path) and os.listdir(train_pp_path):
        print(f"[xgb] Preprocessed data found, skipping to training.")
        print("[xgb] Refitting preprocessor on sample for val/test inference...")
        pp_model = build_xgb_preprocessing(train.columns).fit(
            train.sample(fraction=0.01, seed=42)
        )
        print("[xgb] Preprocessor refit complete.")

    else:
        # Step 1: checkpoint raw train to break lineage
        train_raw_path = os.path.join(scratch_dir, "_xgb_train_raw")
        print("[xgb] Checkpointing raw train split to disk...")
        train.write.mode("overwrite").parquet(train_raw_path)
        train_ckpt = spark.read.parquet(train_raw_path)
        print("[xgb] Raw train checkpoint complete.")

        # Step 2: undersample -> preprocess -> write
        balanced = undersample(
            train_ckpt,
            checkpoint_dir=os.path.join(scratch_dir, "_us_ckpt"),
        )

        preprocess = build_xgb_preprocessing(balanced.columns)
        print("[xgb] Fitting preprocessor on undersampled data...")
        pp_model = preprocess.fit(balanced)
        print("[xgb] Preprocessor fit complete.")

        # Step 3: transform then select only what XGBoost needs
        print("[xgb] Transforming and writing preprocessed data...")
        transformed = pp_model.transform(balanced)
        (transformed
         .select("features", LABEL_COL, WEIGHT_COL)
         .repartition(600)
         .write.mode("overwrite")
         .parquet(train_pp_path))
        print("[xgb] Preprocessed training data written.")

    # ------------------------------------------------------------------
    # LOAD: DISK_ONLY leaves RAM for XGBoost DMatrix
    # ------------------------------------------------------------------
    train_p = spark.read.parquet(train_pp_path).persist(StorageLevel.DISK_ONLY)
    n_train = train_p.count()
    print(f"[xgb] Training data ready: {n_train:,} rows | "
          f"columns: {train_p.columns}")

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------
    results = []
    for name, clf in init_xgb_models():
        print(f"\n{'='*60}\n  TRAINING: {name}\n{'='*60}")
        fitted = Pipeline(stages=[clf]).fit(train_p)

        # Train metrics
        p_train = fitted.transform(train_p).persist(StorageLevel.DISK_ONLY)
        results.append(evaluate(name, p_train, "train"))
        save_predictions(p_train, name, "train")
        p_train.unpersist()

        # Val / test
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

# =====================================================================


# =====================================================================
# 5. ENTRY POINT
# =====================================================================

def run_xgb():
    """XGBoost training entry point."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    spark = get_spark()

    try:
        print(f"\n{'='*60}\n  DATA LOADING\n{'='*60}")
        train, val, test = load_and_split(spark)

        print(f"\n{'='*60}\n  TRAINING: XGBOOST\n{'='*60}")
        results = train_xgb(train, val, test)

        print_results_table(results)
        fitting_analysis(results)

        # Error-analysis plots
        print(f"\n{'='*60}\n  ERROR ANALYSIS PLOTS\n{'='*60}")
        model_names = sorted(set(r["model"] for r in results if r["split"] == "test"))
        for mname in model_names:
            pred_path = os.path.join(OUTPUT_DIR, f"preds_{mname}_test")
            if os.path.exists(pred_path):
                test_preds = spark.read.parquet(pred_path)
                try:
                    plot_geographic_errors(test_preds, mname)
                    plot_temporal_residuals(test_preds, mname)
                except Exception as e:
                    print(f"  [{mname}] Plot failed: {e}, continuing.")
            else:
                print(f"  [{mname}] No test predictions found at {pred_path}")

        print_conclusion()

        return results

    finally:
        spark.stop()

# =====================================================================


if __name__ == "__main__":
    run_xgb()
