"""
Antarctic Ice Mass Loss — Model Training Pipeline
==================================================

Reads the feature-engineered Parquet output from
``feature_engineering_pipeline.py`` and trains a progression of
tree-based classifiers:

    1. DecisionTreeClassifier   — interpretable baseline
    2. RandomForestClassifier   — bagged ensemble (variance reduction)
    3. GBTClassifier            — boosted ensemble (bias reduction)

Each stage uses a shared Spark ML ``Pipeline`` that demonstrates
the required MLlib preprocessing transformers:

    Imputer, StringIndexer, OneHotEncoder,
    VectorAssembler, StandardScaler, PolynomialExpansion

Execution
---------
Local (on the ml_ready output from the feature pipeline):
    python model_training_pipeline.py --mode local

SDSC Expanse:
    spark-submit model_training_pipeline.py --mode sdsc \
        --input-path /expanse/lustre/.../ml_ready
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    GBTClassifier,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import (
    Imputer,
    OneHotEncoder,
    PolynomialExpansion,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# ── Constants ────────────────────────────────────────────────────────
TRAIN_MAX_MONTH_IDX = 24264   # end Dec 2021
VAL_MAX_MONTH_IDX = 24276     # end Dec 2022

LABEL_COL = "basal_loss_agreement"
WEIGHT_COL = "weightCol"
PREDICTION_COL = "prediction"
PROBABILITY_COL = "probability"

KEY_COLS = ["x", "y", "month_idx", "mascon_id", "regional_subset_id"]

NUMERIC_FEATURE_COLS = [
    "surface", "bed", "thickness", "bed_slope",
    "dist_to_grounding_line", "clamped_depth", "dist_to_ocean", "ice_draft",
    "delta_h", "ice_area", "surface_slope", "h_surface_dynamic",
    "thetao_mo", "t_star_mo", "so_mo", "t_f_mo",
    "t_star_quarterly_avg", "t_star_quarterly_std",
    "thetao_quarterly_avg", "thetao_quarterly_std",
    "lwe_mo", "lwe_quarterly_avg", "lwe_quarterly_std",
    "bed_below_sea_level",
    "draft_x_thermal_access", "grounding_line_vulnerability", "retrograde_flag",
    "pixel_mean_delta_h", "delta_h_deviation", "surface_slope_change",
    "thermal_driving_x_draft", "thermal_anomaly",
    "salinity_stratification_proxy", "lwe_trend",
    "regional_t_star_climatology", "regional_t_star_anomaly",
    "sin_month", "cos_month",
    "mascon_mean_delta_h", "mascon_mean_t_star",
    "regional_delta_h_percentile", "regional_lwe_mean",
]

POLY_INPUT_COLS = ["t_star_mo", "ice_draft", "dist_to_grounding_line"]


# =====================================================================
# Spark Session Factory
# =====================================================================

def get_spark(mode: str) -> SparkSession:
    scratch = os.environ.get("TMPDIR", os.path.join(os.getcwd(), "spark_scratch"))

    shared = {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.parquet.filterPushdown": "true",
        "spark.sql.parquet.mergeSchema": "false",
        "spark.network.timeout": "1200s",
        "spark.local.dir": scratch,
    }

    builder = SparkSession.builder.appName("AntarcticModelTraining")

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
# Data Loading and Splitting
# =====================================================================

def load_and_split(
    spark: SparkSession, input_path: str
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Load feature-engineered Parquet and split temporally."""

    df = spark.read.parquet(input_path)
    print(f"[load] Read {input_path}")
    print(f"[load] Columns: {len(df.columns)}, schema verified.")

    df = df.filter(
        F.col(LABEL_COL).isNotNull() & F.col(WEIGHT_COL).isNotNull()
    )

    split_col = (
        F.when(F.col("month_idx") <= TRAIN_MAX_MONTH_IDX, F.lit("train"))
        .when(F.col("month_idx") <= VAL_MAX_MONTH_IDX, F.lit("val"))
        .otherwise(F.lit("test"))
    )

    stats = (
        df.withColumn("_split", split_col)
        .groupBy("_split")
        .agg(
            F.count("*").alias("n"),
            F.sum(F.col(LABEL_COL).cast("int")).alias("pos"),
        )
        .collect()
    )
    stats_map = {r["_split"]: r for r in stats}

    for name in ["train", "val", "test"]:
        r = stats_map.get(name)
        if r:
            rate = r["pos"] / r["n"] if r["n"] > 0 else 0.0
            print(f"  {name:5s}: {r['n']:>12,} rows, pos_rate={rate:.4f}")
        else:
            print(f"  {name:5s}:            0 rows")

    train = df.filter(F.col("month_idx") <= TRAIN_MAX_MONTH_IDX)
    val = df.filter(
        (F.col("month_idx") > TRAIN_MAX_MONTH_IDX)
        & (F.col("month_idx") <= VAL_MAX_MONTH_IDX)
    )
    test = df.filter(F.col("month_idx") > VAL_MAX_MONTH_IDX)

    return train, val, test


# =====================================================================
# MLlib Preprocessing Pipeline
# =====================================================================

def build_preprocessing_pipeline() -> Pipeline:
    """
    Assemble the MLlib preprocessing pipeline:
        Imputer -> StringIndexer -> OneHotEncoder ->
        PolynomialExpansion -> VectorAssembler -> StandardScaler
    """

    imputed_cols = [f"{c}_imp" for c in NUMERIC_FEATURE_COLS]

    imputer = Imputer(
        strategy="median",
        inputCols=list(NUMERIC_FEATURE_COLS),
        outputCols=imputed_cols,
    )

    indexer = StringIndexer(
        inputCol="regional_subset_id",
        outputCol="region_index",
        handleInvalid="keep",
    )

    encoder = OneHotEncoder(
        inputCol="region_index",
        outputCol="region_ohe",
    )

    poly_input_imputed = [f"{c}_imp" for c in POLY_INPUT_COLS]
    poly_assembler = VectorAssembler(
        inputCols=poly_input_imputed,
        outputCol="poly_input",
        handleInvalid="skip",
    )
    poly = PolynomialExpansion(
        degree=2,
        inputCol="poly_input",
        outputCol="poly_features",
    )

    final_assembler = VectorAssembler(
        inputCols=imputed_cols + ["region_ohe", "poly_features"],
        outputCol="raw_features",
        handleInvalid="skip",
    )

    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withMean=True,
        withStd=True,
    )

    return Pipeline(stages=[
        imputer, indexer, encoder,
        poly_assembler, poly,
        final_assembler, scaler,
    ])


# =====================================================================
# Model Definitions
# =====================================================================

def get_model_configs(mode: str) -> List[Tuple[str, object]]:
    """Return (name, classifier) pairs scaled to the execution mode.

    Local mode uses lighter hyperparameters so a smoke-test completes
    in minutes rather than an hour.  HPC/SDSC use production values.
    """

    local = mode == "local"

    return [
        (
            "DecisionTree",
            DecisionTreeClassifier(
                labelCol=LABEL_COL,
                featuresCol="features",
                weightCol=WEIGHT_COL,
                maxDepth=6 if local else 8,
                seed=42,
            ),
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                labelCol=LABEL_COL,
                featuresCol="features",
                weightCol=WEIGHT_COL,
                numTrees=20 if local else 100,
                maxDepth=6 if local else 10,
                featureSubsetStrategy="sqrt",
                seed=42,
            ),
        ),
        (
            "GBT",
            GBTClassifier(
                labelCol=LABEL_COL,
                featuresCol="features",
                weightCol=WEIGHT_COL,
                maxIter=30 if local else 200,
                maxDepth=4 if local else 6,
                stepSize=0.1 if local else 0.05,
                seed=42,
            ),
        ),
    ]


# =====================================================================
# Evaluation
# =====================================================================

def evaluate(
    model_name: str,
    predictions: DataFrame,
    split_name: str,
) -> Dict[str, float]:
    """Compute AUC-ROC and F1 for a single (model, split) pair.

    Returns NaN for AUC when only one class is present in the split
    (e.g. sparse local smoke-test data).
    """

    label_stats = predictions.agg(
        F.min(LABEL_COL).alias("mn"), F.max(LABEL_COL).alias("mx"),
    ).collect()[0]

    if label_stats["mn"] == label_stats["mx"]:
        print(f"  [{model_name}] {split_name:5s}  AUC=  N/A   F1=  N/A   "
              f"(single class: {label_stats['mn']})")
        return {"model": model_name, "split": split_name,
                "auc": float("nan"), "f1": float("nan")}

    auc_eval = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    f1_eval = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL,
        predictionCol=PREDICTION_COL,
        metricName="f1",
    )

    auc = auc_eval.evaluate(predictions)
    f1 = f1_eval.evaluate(predictions)

    print(f"  [{model_name}] {split_name:5s}  AUC={auc:.4f}  F1={f1:.4f}")
    return {"model": model_name, "split": split_name, "auc": auc, "f1": f1}


def regional_summary(predictions: DataFrame, model_name: str) -> None:
    """Print per-region positive prediction rate."""

    summary = (
        predictions
        .groupBy("regional_subset_id")
        .agg(
            F.avg(F.col(PREDICTION_COL).cast("float")).alias("pred_pos_rate"),
            F.avg(F.col(LABEL_COL).cast("float")).alias("true_pos_rate"),
            F.count("*").alias("n"),
        )
        .orderBy("regional_subset_id")
        .collect()
    )

    print(f"\n  [{model_name}] Regional breakdown:")
    for row in summary:
        print(f"    {row['regional_subset_id']:25s}  "
              f"pred={row['pred_pos_rate']:.4f}  "
              f"true={row['true_pos_rate']:.4f}  "
              f"n={row['n']:>10,}")


def extract_feature_importance(
    model_name: str,
    fitted_pipeline: PipelineModel,
) -> None:
    """Print top-10 feature importances from the tree model."""

    tree_model = fitted_pipeline.stages[-1]

    if not hasattr(tree_model, "featureImportances"):
        return

    importances = tree_model.featureImportances.toArray()

    imputed_names = [f"{c}_imp" for c in NUMERIC_FEATURE_COLS]
    poly_count = len(POLY_INPUT_COLS) + len(POLY_INPUT_COLS) * (len(POLY_INPUT_COLS) + 1) // 2
    feature_names = (
        imputed_names
        + [f"region_ohe_{i}" for i in range(6)]
        + [f"poly_{i}" for i in range(poly_count)]
    )

    if len(feature_names) < len(importances):
        feature_names += [
            f"feat_{i}" for i in range(len(importances) - len(feature_names))
        ]

    pairs = sorted(
        zip(feature_names, importances), key=lambda p: p[1], reverse=True
    )

    print(f"\n  [{model_name}] Top 10 features:")
    for name, imp in pairs[:10]:
        bar = "#" * int(imp * 100)
        print(f"    {name:40s}  {imp:.4f}  {bar}")


def save_sample_predictions(
    predictions: DataFrame,
    model_name: str,
    split_name: str,
    output_dir: str,
) -> None:
    """Persist a 500-row sample with key columns + predictions."""

    sample = (
        predictions
        .select(
            *KEY_COLS,
            LABEL_COL,
            PREDICTION_COL,
            PROBABILITY_COL,
            WEIGHT_COL,
        )
        .limit(500)
    )

    path = os.path.join(output_dir, f"predictions_{model_name}_{split_name}")
    sample.write.mode("overwrite").parquet(path)
    print(f"  [{model_name}] Saved {split_name} predictions -> {path}")


# =====================================================================
# Main Training Loop
# =====================================================================

def train_and_evaluate(
    train: DataFrame,
    val: DataFrame,
    test: DataFrame,
    output_dir: str,
    mode: str = "local",
) -> None:
    """Fit preprocessing once, then train each classifier in sequence."""

    print("\n" + "=" * 72)
    print("  FITTING PREPROCESSING PIPELINE ON TRAINING DATA")
    print("=" * 72)

    preprocess = build_preprocessing_pipeline()
    preprocess_model = preprocess.fit(train)

    train_prep = preprocess_model.transform(train).cache()
    val_prep = preprocess_model.transform(val).cache()
    test_prep = preprocess_model.transform(test).cache()

    n_features = train_prep.select("features").head(1)[0]["features"].size
    print(f"  Feature vector dimension: {n_features}")

    all_results = []

    for model_name, classifier in get_model_configs(mode):
        print("\n" + "-" * 72)
        print(f"  TRAINING: {model_name}")
        print("-" * 72)

        full_pipeline = Pipeline(stages=[classifier])
        fitted = full_pipeline.fit(train_prep)

        for split_name, split_df in [("train", train_prep), ("val", val_prep), ("test", test_prep)]:
            preds = fitted.transform(split_df)
            metrics = evaluate(model_name, preds, split_name)
            all_results.append(metrics)

            if split_name == "test":
                regional_summary(preds, model_name)
                save_sample_predictions(preds, model_name, split_name, output_dir)

        extract_feature_importance(model_name, fitted)

    train_prep.unpersist()
    val_prep.unpersist()
    test_prep.unpersist()

    # ── Summary table ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)
    print(f"  {'Model':<18s} {'Split':<7s} {'AUC':>8s} {'F1':>8s}")
    print("  " + "-" * 43)
    for r in all_results:
        auc_str = f"{r['auc']:>8.4f}" if r["auc"] == r["auc"] else "     N/A"
        f1_str = f"{r['f1']:>8.4f}" if r["f1"] == r["f1"] else "     N/A"
        print(f"  {r['model']:<18s} {r['split']:<7s} {auc_str} {f1_str}")
    print("=" * 72)


# =====================================================================
# Entry Point
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Antarctic ice mass loss — model training pipeline"
    )
    parser.add_argument(
        "--input-path",
        default=os.path.join(os.getcwd(), "ml_ready"),
        help="Path to feature-engineered Parquet output.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.getcwd(), "model_output"),
        help="Directory for prediction samples and model artefacts.",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "hpc", "sdsc"],
        default="local",
        help="Spark configuration profile.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    spark = get_spark(args.mode)

    try:
        train, val, test = load_and_split(spark, args.input_path)
        train_and_evaluate(train, val, test, args.output_dir, args.mode)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
