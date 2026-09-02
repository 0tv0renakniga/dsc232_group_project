# =====================================================================
# ml_pipeline_classic.py — Classic Spark ML model training.
#
# Trains DecisionTree -> RandomForest -> GBT progression with
# PolynomialExpansion, Ocean PCA, and StandardScaler preprocessing.
# Demonstrates MLlib transformer breadth.
# =====================================================================

from __future__ import annotations

import os

# ---------------------------------------------------------------------
# PySpark
# ---------------------------------------------------------------------
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    GBTClassifier,
    RandomForestClassifier,
)
from pyspark.ml.feature import (
    Imputer,
    OneHotEncoder,
    PCA,
    PolynomialExpansion,
    StandardScaler,
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
    OCEAN_COLS, PCA_K,
    get_spark, get_all_numeric_cols, load_and_split, undersample,
    evaluate, regional_summary, save_predictions,
    print_results_table, fitting_analysis,
    plot_geographic_errors, plot_temporal_residuals, print_conclusion,
)
# ---------------------------------------------------------------------


# =====================================================================
# 1. CLASSIC-SPECIFIC CONSTANTS
# =====================================================================

# ---------------------------------------------------------------------
# PolynomialExpansion input columns
# ---------------------------------------------------------------------
POLY_INPUT_COLS = ["t_star_mo", "ice_draft", "dist_to_grounding_line"]
# ---------------------------------------------------------------------
# =====================================================================


# =====================================================================
# 2. PREPROCESSING
# =====================================================================

def build_classic_preprocessing(available):
    """Classic: Imputer -> OHE -> PCA(ocean) -> PolyExpansion -> StandardScaler.

        note: demonstrates every required MLlib transformer in one
              pipeline.
    """

    numeric = get_all_numeric_cols(available)
    imputed = [f"{c}_imp" for c in numeric]

    # Ocean columns for PCA (use imputed names)
    ocean_imp     = [f"{c}_imp" for c in OCEAN_COLS if c in available]
    non_ocean_imp = [c for c in imputed if c not in ocean_imp]

    # Polynomial expansion inputs (use imputed names)
    poly_imp = [f"{c}_imp" for c in POLY_INPUT_COLS if c in available]

    stages = [
        Imputer(strategy="median", inputCols=numeric, outputCols=imputed),
        StringIndexer(
            inputCol="regional_subset_id",
            outputCol="region_index",
            handleInvalid="keep",
        ),
        OneHotEncoder(inputCol="region_index", outputCol="region_ohe"),
    ]

    # PCA on correlated ocean variables
    if ocean_imp:
        stages += [
            VectorAssembler(
                inputCols=ocean_imp, outputCol="ocean_vec",
                handleInvalid="skip",
            ),
            PCA(
                k=min(PCA_K, len(ocean_imp)),
                inputCol="ocean_vec",
                outputCol="ocean_pca",
            ),
        ]

    # PolynomialExpansion on key physics triple
    if poly_imp:
        stages += [
            VectorAssembler(
                inputCols=poly_imp, outputCol="poly_input",
                handleInvalid="skip",
            ),
            PolynomialExpansion(
                degree=2,
                inputCol="poly_input",
                outputCol="poly_features",
            ),
        ]

    # Final assembly
    final_inputs = non_ocean_imp + ["region_ohe"]
    if ocean_imp:
        final_inputs.append("ocean_pca")
    if poly_imp:
        final_inputs.append("poly_features")

    stages += [
        VectorAssembler(
            inputCols=final_inputs, outputCol="raw_features",
            handleInvalid="skip",
        ),
        StandardScaler(
            inputCol="raw_features", outputCol="features",
            withMean=True, withStd=True,
        ),
    ]

    print(f"[preprocess:classic] {len(numeric)} numeric + OHE + "
          f"PCA(k={PCA_K}) + PolyExpansion(deg=2) + StandardScaler")

    return Pipeline(stages=stages)

# =====================================================================


# =====================================================================
# 3. MODEL INITIALISATION
# =====================================================================

def init_classic_models():
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


# =====================================================================
# 4. TRAINING
# =====================================================================

def train_classic(train, val, test):
    """
    Train the DecisionTree -> RandomForest -> GBT progression.

        note: uses PolynomialExpansion + StandardScaler + Ocean PCA
              preprocessing.
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

        # Feature importance
        tree_model = fitted.stages[-1]
        if hasattr(tree_model, "featureImportances"):
            importances = tree_model.featureImportances.toArray()
            pairs = sorted(enumerate(importances), key=lambda x: x[1], reverse=True)
            print(f"\n  [{name}] Top 10 features:")
            for idx, imp in pairs[:10]:
                bar = "#" * int(imp * 100)
                print(f"    feat[{idx:3d}]  {imp:.4f}  {bar}")

    train_p.unpersist()

    return results

# =====================================================================


# =====================================================================
# 5. ENTRY POINT
# =====================================================================

def run_classic():
    """Classic models training entry point."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    spark = get_spark()

    try:
        print(f"\n{'='*60}\n  DATA LOADING\n{'='*60}")
        train, val, test = load_and_split(spark)

        print(f"\n{'='*60}\n  TRAINING: CLASSIC (DT -> RF -> GBT)\n{'='*60}")
        results = train_classic(train, val, test)

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
    run_classic()
