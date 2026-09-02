# =====================================================================
# ml_pipeline_corrected_stack.py — Corrected stacking ensemble.
#
# Uses out-of-fold (OOF) training for base learners, pruned
# meta-features, and LogisticRegression meta-learner.  Fixes
# the overfitting from uncorrected stacking (Model 3).
# =====================================================================

from __future__ import annotations

import os

# ---------------------------------------------------------------------
# PySpark
# ---------------------------------------------------------------------
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.feature import (
    Imputer,
    Normalizer,
    PCA,
    VectorAssembler,
)
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from pyspark import StorageLevel
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------
from pipeline_config import (
    MODE, LOCAL, scratch_dir, OUTPUT_DIR,
    LABEL_COL, WEIGHT_COL, PREDICTION_COL,
    OCEAN_COLS,
    get_spark, get_all_numeric_cols, load_and_split, undersample,
    evaluate, regional_summary, save_predictions,
    print_results_table, fitting_analysis,
    plot_geographic_errors, plot_temporal_residuals, print_conclusion,
)
# ---------------------------------------------------------------------


# =====================================================================
# 1. STACK-SPECIFIC CONSTANTS
# =====================================================================

# ---------------------------------------------------------------------
# Base learner hyperparameters
# ---------------------------------------------------------------------
RF_CONFIG = dict(
    numTrees=20 if LOCAL else 100,
    maxDepth=6 if LOCAL else 10,
)
GBT_CONFIG = dict(
    maxIter=30 if LOCAL else 150,
    maxDepth=4 if LOCAL else 6,
    stepSize=0.1 if LOCAL else 0.05,
)
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Corrected stack: meta-learner context features (pruned)
# ---------------------------------------------------------------------
META_CONTEXT_COLS = [
    "region_cat_idx", "sin_month", "cos_month",
    "dist_to_grounding_line", "delta_h",
]
# ---------------------------------------------------------------------
# =====================================================================


# =====================================================================
# 2. PREPROCESSING
# =====================================================================

def build_stack_preprocessing(available):
    """
    Stacking: Imputer -> grouped PCA -> Assembler -> Normalizer (L2).

        note: PCA applied to three correlated feature groups:
            - Ocean variables (physically correlated)
            - Temporal memory (linear combinations of same signals)
            - Trajectory (momentum/acceleration pairs)

        note: remaining features pass through directly.
    """

    numeric = get_all_numeric_cols(available)
    imputed = [f"{c}_imp" for c in numeric]

    # Correlated groups: PCA reduces these
    ocean_imp = [f"{c}_imp" for c in OCEAN_COLS if c in available]
    temporal_imp = [f"{c}_imp" for c in [
        "t_star_6mo_avg", "lwe_6mo_avg", "delta_h_6mo_avg",
        "t_star_rate", "lwe_rate", "delta_h_rate",
        "t_star_mom_change", "delta_h_mom_change",
    ] if c in available]
    trajectory_imp = [f"{c}_imp" for c in [
        "delta_h_momentum", "delta_h_acceleration", "delta_h_deseason",
        "t_star_momentum", "t_star_sustained_anomaly",
        "lwe_momentum", "lwe_sustained_trend",
    ] if c in available]

    pca_cols  = set(ocean_imp + temporal_imp + trajectory_imp)
    remaining = [c for c in imputed if c not in pca_cols]

    stages = [
        Imputer(strategy="median", inputCols=numeric, outputCols=imputed),
    ]

    final_inputs = list(remaining)

    # Ocean PCA
    if ocean_imp:
        stages += [
            VectorAssembler(
                inputCols=ocean_imp, outputCol="ocean_vec",
                handleInvalid="skip",
            ),
            PCA(k=min(4, len(ocean_imp)), inputCol="ocean_vec",
                outputCol="ocean_pca"),
        ]
        final_inputs.append("ocean_pca")

    # Temporal PCA
    if temporal_imp:
        stages += [
            VectorAssembler(
                inputCols=temporal_imp, outputCol="temporal_vec",
                handleInvalid="skip",
            ),
            PCA(k=min(3, len(temporal_imp)), inputCol="temporal_vec",
                outputCol="temporal_pca"),
        ]
        final_inputs.append("temporal_pca")

    # Trajectory PCA
    if trajectory_imp:
        stages += [
            VectorAssembler(
                inputCols=trajectory_imp, outputCol="trajectory_vec",
                handleInvalid="skip",
            ),
            PCA(k=min(3, len(trajectory_imp)), inputCol="trajectory_vec",
                outputCol="trajectory_pca"),
        ]
        final_inputs.append("trajectory_pca")

    total = (
        len(remaining)
        + (4 if ocean_imp else 0)
        + (3 if temporal_imp else 0)
        + (3 if trajectory_imp else 0)
    )

    stages += [
        VectorAssembler(
            inputCols=final_inputs, outputCol="raw_features",
            handleInvalid="skip",
        ),
        Normalizer(inputCol="raw_features", outputCol="features", p=2.0),
    ]

    print(f"[preprocess:stack] {len(numeric)} numeric → PCA groups → "
          f"{total} features + Normalizer(L2)")

    return Pipeline(stages=stages)

# =====================================================================


# =====================================================================
# 3. MODEL INITIALISATION
# =====================================================================

def init_base_learners():
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


def init_corrected_meta_learners():
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

# =====================================================================


# =====================================================================
# 4. STACKING HELPERS
# =====================================================================

def build_pruned_meta_features(df, available_cols):
    """Build PRUNED meta-features for corrected stacking.

        note: uses only base predictions + 5 context features to
              prevent overfitting from full-feature meta-learner.
    """

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
        df = df.withColumn(
            "base_agreement",
            (F.col("rf_prediction") == F.col("gbt_prediction")).cast(FloatType()),
        )
    else:
        df = df.withColumn("base_agreement", F.lit(1.0).cast(FloatType()))

    meta_inputs = ["rf_pos_prob", "gbt_score", "base_agreement"]
    for col in META_CONTEXT_COLS:
        if col in available_cols:
            imp_name = f"{col}_meta"
            df = df.withColumn(
                imp_name,
                F.coalesce(F.col(col), F.lit(0.0)).cast(FloatType()),
            )
            meta_inputs.append(imp_name)

    assembler = VectorAssembler(
        inputCols=meta_inputs, outputCol="meta_features",
        handleInvalid="skip",
    )
    df = assembler.transform(df)

    for c in ["rawPrediction", "prediction", "probability"]:
        if c in df.columns:
            df = df.drop(c)

    return df


def oof_split(train):
    """Split training data temporally for OOF stacking."""

    stats = train.agg(
        F.min("month_idx").alias("mn"),
        F.max("month_idx").alias("mx"),
    ).collect()[0]
    mid = (stats["mn"] + stats["mx"]) // 2

    fold_a = train.filter(F.col("month_idx") <= mid)
    fold_b = train.filter(F.col("month_idx") > mid)

    for name, fold in [("A", fold_a), ("B", fold_b)]:
        n = fold.count()
        pos = fold.filter(F.col(LABEL_COL) == 1).count()
        print(f"  [OOF] Fold {name}: {n:,} rows, pos={pos:,}")

    return fold_a, fold_b

# =====================================================================


# =====================================================================
# 5. TRAINING
# =====================================================================

def train_corrected_stack(train, val, test):
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

    # ------------------------------------------------------------------
    # Layer 1: base learners on fold A
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Layer 2: corrected meta-learner on fold B
    # ------------------------------------------------------------------
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
            meta_names = (
                ["rf_pos_prob", "gbt_score", "base_agreement"]
                + [f"{c}_meta" for c in META_CONTEXT_COLS]
            )
            for n, c in zip(meta_names[:len(coeffs)], coeffs):
                sign = "+" if c >= 0 else "-"
                print(f"    {n:30s}  {sign}{abs(c):.4f}")

        for sname, raw_df in [("train", train), ("val", val), ("test", test)]:
            sdf = pp_model.transform(raw_df)
            p = fitted.transform(
                build_pruned_meta_features(add_base(sdf), available_cols)
            )
            p = p.persist(StorageLevel.DISK_ONLY)
            results.append(evaluate(name, p, sname))
            save_predictions(p, name, sname)
            if sname == "test":
                regional_summary(p, name)
            p.unpersist()

    fold_b_m.unpersist()

    return results

# =====================================================================


# =====================================================================
# 6. ENTRY POINT
# =====================================================================

def run_corrected_stack():
    """Corrected stacking ensemble entry point."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    spark = get_spark()

    try:
        print(f"\n{'='*60}\n  DATA LOADING\n{'='*60}")
        train, val, test = load_and_split(spark)

        print(f"\n{'='*60}\n  TRAINING: CORRECTED STACKING\n{'='*60}")
        results = train_corrected_stack(train, val, test)

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
    run_corrected_stack()
