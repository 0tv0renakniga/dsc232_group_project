# Milestone 3 Summary

---

## Important Details

| Constraint | Status | Evidence |
|---|---|---|
| Distributed implementation | **Done** | `SparkXGBClassifier` from `xgboost.spark`, Spark MLlib `Pipeline`, all preprocessing via PySpark transformers. No scikit-learn. |
| Not simple linear regression | **Done** | Gradient-boosted trees with 64+ engineered features, PCA, PolynomialExpansion, temporal windows. |
| No driver-only models | **Done** | Training runs inside Spark executors. No `.collect()`-then-sklearn pattern. |
| Meaningful parallelization | **Done** | HPC config: 6 executors x 4 cores, 20 GB each (L358-373 in ml_pipeline.py). 800 shuffle partitions. |

* Note: `num_workers = 1` is hard-coded on L1337 in ml_pipeline.py. This is just from running the ensemble(stacking) pipeline. You can change this to 2 -   4 workers for XGBoost training.

---

## 1. Features Engineering

### 1.1 What Features Were Engineered and Why

The pipeline engineers **64+ features** across five categories, all motivated by glaciological physics:

| Category | Example Features | Rationale |
|---|---|---|
| Static geometry | `draft_x_thermal_access`, `grounding_line_vulnerability`, `retrograde_flag` | Encode ice-ocean exposure and marine instability geometry |
| Dynamic pixel-level | `pixel_mean_delta_h`, `delta_h_deviation`, `surface_slope_change` | Capture per-pixel thinning trends via expanding windows |
| Ocean interactions | `thermal_driving_x_draft`, `thermal_anomaly`, `salinity_stratification_proxy`, `lwe_trend` | Encode warm-water intrusion and gravitational mass signals |
| Context / cyclical | `sin_month`, `cos_month`, `mascon_mean_delta_h`, `regional_delta_h_percentile` | Seasonal cycles + neighborhood context without leakage |
| Temporal memory | `t_star_6mo_avg`, `delta_h_rate`, `delta_h_momentum`, `delta_h_acceleration` | 6-month rolling averages, momentum, and acceleration for trend detection |
| Physics interactions | `ocean_heat_content_proxy`, `draft_ratio`, `thermal_x_gl_proximity`, `freezing_departure`, `bed_geometry_risk`, `mass_flux_proxy` | Hand-crafted domain features combining ice draft, ocean temperature, and bed topography |

### 1.2 Scaling / Transforming

Three distinct scalers are used across the three preprocessing pipelines:

```python
# XGBoost pipeline: MinMaxScaler (L1146)
MinMaxScaler(inputCol="raw_features", outputCol="features")

# Classic pipeline: StandardScaler with mean centering (L1292-1293)
StandardScaler(inputCol="raw_features", outputCol="features",
               withMean=True, withStd=True)

# Stack pipeline: L2 Normalizer (L1228)
Normalizer(inputCol="raw_features", outputCol="features", p=2.0)
```

### 1.3 Imputing

Every pipeline uses `Imputer` with median strategy as its first stage:

```python
# L1136  (identical pattern in all three pipelines)
Imputer(strategy="median", inputCols=numeric, outputCols=imputed)
```

This is critical because the Antarctic dataset has significant sparsity(i.e., ocean variables are `null` for inland pixels, and temporal features have nulls at series boundaries).

### 1.4 Encoding

```python
# StringIndexer: region name -> integer index (L1140-1141)
StringIndexer(inputCol="regional_subset_id",
              outputCol="region_index", handleInvalid="keep")

# OneHotEncoder: grounding-line distance buckets + region (L1142-1143)
OneHotEncoder(inputCol="gl_bucket_idx", outputCol="gl_bucket_ohe")
OneHotEncoder(inputCol="region_index", outputCol="region_ohe")

# Bucketizer: continuous distance -> categorical bins (L1137-1139)
Bucketizer(splits=[-inf, 5000, 20000, 50000, 100000, inf],
           inputCol="dist_to_grounding_line",
           outputCol="gl_bucket_idx", handleInvalid="keep")

# VectorAssembler: combine all features into a single vector (L1144-1145)
VectorAssembler(inputCols=imputed + ["gl_bucket_ohe", "region_ohe"],
                outputCol="raw_features", handleInvalid="skip")
```

### 1.5 Feature Engineering (Spark SQL + Advanced Transformers)

```python
# Window functions: pixel-level expanding mean (L570-584)
pixel_time_w = Window.partitionBy("x", "y").orderBy("month_idx")
    .rowsBetween(Window.unboundedPreceding, 0)
df = df.withColumn("pixel_mean_delta_h", F.avg("delta_h").over(pixel_time_w))

# PolynomialExpansion on physics triple (L1276-1279, classic pipeline)
PolynomialExpansion(degree=2, inputCol="poly_input",
                    outputCol="poly_features")
# Inputs: t_star_mo x ice_draft x dist_to_grounding_line

# PCA on correlated ocean variables (L1194-1198)
PCA(k=4, inputCol="ocean_vec", outputCol="ocean_pca")
# Groups: thetao_mo, t_star_mo, so_mo, t_f_mo + quarterly aggregates

# Cyclic encoding via Spark SQL (L640-649)
df = df.withColumn("sin_month", F.sin(F.col("month_of_year") * (2*pi/12)))
       .withColumn("cos_month", F.cos(F.col("month_of_year") * (2*pi/12)))
```

### 1.6 Feature Table (for Milestone write-up)

| Feature | Description | Why | Spark MLlib Transformers | Spark DataFrame Ops |
|---|---|---|---|---|
| `gl_bucket_ohe` | Grounding-line distance bins | Captures proximity thresholds for warm-water access | `Bucketizer` $\rightarrow$ `OneHotEncoder` | n/a |
| `region_ohe` | One-hot region encoding | Regional ice dynamics vary fundamentally | `StringIndexer` $\rightarrow$ `OneHotEncoder` | n/a |
| features (scaled) | Min-max scaled feature vector | Normalises heterogeneous scales for XGBoost | `Imputer` $\rightarrow$ `VectorAssembler` $\rightarrow$ `MinMaxScaler` | n/a |
| `ocean_pca` | PCA(k=4) of 8 ocean variables | Decorrelates collinear ocean measurements | `VectorAssembler` $\rightarrow$ `PCA` | n/a |
| `poly_features` | Degree-2 expansion of physics triple | Captures nonlinear interactions | `VectorAssembler` $\rightarrow$ `PolynomialExpansion` | n/a |
| `pixel_mean_delta_h` | Expanding-window mean of ice-height change | Tracks cumulative thinning per pixel | n/a | `F.avg().over(Window)` |
| `thermal_driving_x_draft` | `t_star_mo x ice_draft` | Ocean thermal forcing x ice exposure | n/a | `F.col() * F.col()` |
| `sin_month` / `cos_month` | Cyclic month encoding | Preserves seasonal continuity | n/a | `F.sin()`, `F.cos()` |
| `delta_h_momentum` | 1-month lag difference | Rate of thinning acceleration | n/a | `F.lag().over(Window)` |
| `regional_delta_h_percentile` | Z-score within region-month | Where this pixel ranks relative to regional peers | n/a | `groupBy` + `broadcast join` |

---

## 2. Distributed Model Training

### 2.1 Model Selection Rationale

**`SparkXGBClassifier`** was selected because:

1. **Native Spark integration:** `xgboost.spark.SparkXGBClassifier` runs inside the Spark execution engine, distributing histogram construction across workers.
2. **Threshold-like physics:** XGBoost's gradient-boosted decision trees excel at learning threshold-based decision boundaries (e.g., "if thermal forcing > X AND grounding-line distance < Y $\rightarrow$ basal melt"). This matches the physics of ice-sheet basal melt.
3. **Handles imbalanced data:** Via `weightCol` support and `logloss` evaluation metric.
4. **Scale:** 400M rows, 72 columns, 40+ GB: only a distributed framework is feasible.

### 2.2 Spark Configuration (HPC Mode)

```python
# L354-373 : HPC config
builder = (
    builder
    .master(master)                               # local[24] or YARN
    .config("spark.executor.instances", "6")       # 6 executor JVMs
    .config("spark.executor.cores", "4")           # 4 cores each = 24 total
    .config("spark.executor.memory", "20g")         # 20 GB per executor
    .config("spark.yarn.executor.memoryOverhead", "10g")
    .config("spark.driver.memory", "80g")
    .config("spark.sql.shuffle.partitions", "800")  # high parallelism
    .config("spark.memory.fraction", "0.5")
    .config("spark.memory.storageFraction", "0.1")  # favour execution over caching
)
```

**Verification via Spark UI:** The Spark UI (port 4040) shows:
- **Executors tab:** 6 active executors, each with 4 cores and 20 GB
- **Stages tab:** Shuffle read/write across multiple executors during `repartition`, `groupBy`, and `Window` operations
- **SQL tab:** Distributed query plans for each feature engineering stage
- # **TODO:** Add Spark UI screenshots
### 2.3 Training vs. Test Error Comparison

| Model | Split | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| **XGB_Baseline** | train | 0.9304 | 0.5103 | 0.7512 | 0.9279 | 0.6841 |
| **XGB_Baseline** | val | 0.6969 | 0.0475 | 0.9700 | 0.9636 | 0.9776 |
| **XGB_Baseline** | test | 0.7403 | 0.0914 | 0.9440 | 0.9346 | 0.9558 |
| **XGB_Tuned** | train | 0.9422 | 0.5655 | 0.7667 | 0.9292 | 0.7037 |
| **XGB_Tuned** | val | 0.6957 | 0.0430 | 0.9705 | 0.9630 | 0.9794 |
| **XGB_Tuned** | test | 0.7490 | 0.0900 | 0.9451 | 0.9343 | 0.9608 |

### 2.4 Metric Selection Justification

- **PR-AUC (primary):** With a ~3% positive rate (`rate=0.030064` in train), ROC-AUC is **inflated by the massive true-negative count**. PR-AUC directly measures the precision-recall trade-off for the rare positive class. A PR-AUC of 0.09 is more honest than a ROC-AUC of 0.74: it says the model finds very few true positives without false alarms.
- **ROC-AUC:** Used for overfitting diagnosis (train-test gap) but misleading as a standalone quality metric for imbalanced problems.
- **F1, Precision, Recall:** Complement threshold-dependent analysis. The high F1/Precision/Recall on val/test are artifacts of the majority-class dominance at the default 0.5 threshold.

### 2.5 Ground Truth vs. Predictions (Regional Breakdown)

**XGB_Baseline: Test Set:**

| Region | Predicted Rate | True Rate | n |
|---|---|---|---|
| amundsen_sea | 0.0029 | 0.1155 | 10,869,309 |
| antarctic_peninsula | 0.0870 | 0.0396 | 14,578,163 |
| lambert_amery | 0.0003 | 0.0099 | 27,227,872 |
| ronne | 0.0003 | 0.0224 | 23,682,149 |
| ross | 0.0011 | 0.0440 | 34,371,219 |
| totten_and_aurora | 0.0010 | 0.0312 | 15,988,044 |

> [!CAUTION]
> **Critical failure in Amundsen Sea:** The model predicts a 0.29% positive rate against a true rate of 11.55%: a **40x underestimation** of the most rapidly melting region in Antarctica. The model appears to have learned that "most pixels are negative" and fails to flag the high-loss region. This is the most scientifically important region and must be addressed in Milestone 4.

---

## 3. Fitting Analysis

### 3.1 Where Does the Model Sit on the Fitting Graph?

**Both models are OVERFITTING:**

```
XGB_Baseline:
  Train ROC-AUC: 0.9304
  Test  ROC-AUC: 0.7403  (gap: 0.1902)
  Test  PR-AUC:  0.0914
  $\rightarrow$ OVERFITTING

XGB_Tuned:
  Train ROC-AUC: 0.9422
  Test  ROC-AUC: 0.7490  (gap: 0.1932)
  Test  PR-AUC:  0.0900
  $\rightarrow$ OVERFITTING
```

The ~19% ROC-AUC gap between train and test is a clear overfitting signal. The model memorises training patterns (especially after undersampling alters the class distribution) that do not generalise to the temporal hold-out.

**Why overfitting occurs:**
1. **Temporal distribution shift:** Training data (Apr 2020 – Dec 2022) does not contain the climatic patterns present in the test period (post-Oct 2023). Ice-sheet dynamics are non-stationary.
2. **Undersampling artefact:** The 1:10 ratio amplifies positive examples, which the model can memorise since many are spatially clustered in a few glaciers.
3. **High model capacity:** `max_depth=4-8`, 100–400 trees is sufficient capacity to overfit spatial patterns.

### 3.2 Model Comparison (Baseline vs. Tuned)

| Hyperparameter | XGB_Baseline | XGB_Tuned |
|---|---|---|
| `max_depth` | 4 | 8 |
| `n_estimators` | 100 | 400 |
| `learning_rate` | 0.1 | 0.02 |
| `subsample` | 0.8 | 0.75 |
| `colsample_bytree` | 0.8 | 0.7 |
| `min_child_weight` | 10 | 20 |
| `reg_alpha` | n/a | 0.1 |
| `reg_lambda` | n/a | 1.0 |

**Key differences in tuning rationale:**
- **Lower learning rate (0.02) + more trees (400):** Slower, more incremental gradient steps for finer convergence.
- **Higher `min_child_weight` (20):** Requires more samples per leaf $\rightarrow$ reduces overfitting to small clusters.
- **L1/L2 regularisation added:** `reg_alpha=0.1` (L1) and `reg_lambda=1.0` (L2) explicitly penalise complexity.
- **Lower `subsample` and `colsample_bytree`:** More aggressive bagging $\rightarrow$ reduces variance.

**Result:** XGB_Tuned achieves marginally better test ROC-AUC (0.749 vs 0.740) but **worse test PR-AUC** (0.090 vs 0.091). The overfitting gap actually *increased* slightly (0.193 vs 0.190). The tuned hyperparameters successfully increased train performance but the regularisation was insufficient to close the generalisation gap.


* **Note on the current `XGB_CONFIGS`:** Both produced similar overfitting patterns.

### 3.3 Which Model Performs Best?

**XGB_Baseline is marginally better** on the metric that matters (PR-AUC: 0.091 vs 0.090), despite worse ROC-AUC. The tuned model's deeper trees and more iterations increased memorisation without improving rare-event detection. For a geoscience application where false negatives in Amundsen Sea cost real predictive value, neither model is adequate.

### 3.4 Next Models for Milestone 4

The pipeline already implements the architecture for three additional model families. The recommended progression:

1. **Corrected Stacking (Model 4):** Uses **out-of-fold (OOF) training** with pruned meta-features and `LogisticRegression` as the meta-learner. This directly attacks the overfitting problem by:
   - Training base learners on fold A, predicting on fold B
   - Using only 8 meta-features (3 base predictions + 5 context cols) instead of all 64+
   - Using a low-capacity linear meta-learner (LR) instead of XGBoost

2. **Region-aware resampling:** The regional breakdown shows the model completely ignores Amundsen Sea. A **region-stratified undersampling** strategy that ensures proportional representation of each region's positive class would help.

3. **Temporal cross-validation:** Instead of a single temporal split, use **expanding-window CV** (train on months 1–N, validate on N+1–N+3) to capture seasonal and multi-year patterns.

4. **Threshold tuning:** The default 0.5 classification threshold is suboptimal for 3% positive rate. A **PR-curve-optimised threshold** (e.g., maximising F1 on the PR curve) would significantly improve operational predictions.

---

## 4. Conclusion

### 4.1 First Model Performance Summary

The first model (`SparkXGBClassifier`) demonstrates:
- **Strong discriminative ability on training data** (ROC-AUC 0.93, PR-AUC 0.51)
- **Significant generalisation failure** (test PR-AUC 0.09, ROC-AUC gap ~0.19)
- **Regional bias:** overpredicts Antarctic Peninsula (pred=0.087 vs true=0.040), severely underpredicts Amundsen Sea (pred=0.003 vs true=0.116)
- The model has **learned to be a conservative predictor** on unseen data, predicting "no basal loss" almost everywhere

**Why XGBoost overtrained:** XGBoost is a high-capacity learner (deep trees, aggressive boosting). When combined with undersampling that overrepresents the positive class, the model memorises spatial patterns of known melt zones. These patterns are non-stationary across the temporal split and Antarctic melt dynamics shift year to year based on ocean circulation changes.

### 4.2 Actionable Improvement Strategies

1. **Fix regional imbalance:** Apply region-stratified sampling or increase `REGION_WEIGHTS` for Amundsen Sea (currently 2.0, consider 5.0+)
2. **Reduce overtraining:** Switch to the corrected stacking ensemble (Model 4) which uses OOF training + linear meta-learner to cap capacity
3. **Threshold calibration:** Use Platt scaling or isotonic regression to calibrate predicted probabilities, then select threshold maximising PR-AUC
4. **Feature selection:** The 64+ features likely contain redundancies. Use SHAP or feature importance to prune to the top 20-30 most informative features
5. **Temporal augmentation:** Add year-over-year lagged features (12-month lag of `delta_h`, `lwe_mo`) to help the model learn inter-annual patterns

### 4.3 How Distributed Computing Helped

| Challenge | Spark Solution |
|---|---|
| **40 GB dataset (400M rows)** | Partitioned across 6 executors x 20 GB RAM; impossible on a single machine |
| **Pixel-level window functions** | `repartition("x", "y")` co-locates pixel time-series, enabling `Window.partitionBy("x","y")` without cross-node shuffles on 800 partitions |
| **Feature engineering (30+ transforms)** | MLlib `Pipeline` chains `Imputer` $\rightarrow$ `Bucketizer` $\rightarrow$ `OneHotEncoder` $\rightarrow$ `VectorAssembler` $\rightarrow$ `MinMaxScaler` into a single distributed execution plan |
| **XGBoost histogram construction** | Spark distributes histogram building across cores; the `SparkXGBClassifier` interface handles barrier-mode coordination |
| **Lineage truncation** | DAG lineage from 30+ chained transforms would OOM the driver; `flush()` writes intermediate results to Parquet and reads back, capping DAG depth |
| **Memory management** | `StorageLevel.DISK_ONLY` for XGBoost training data, `checkpoint()` after undersampling, and `unpersist()` after each evaluation stage |

---

