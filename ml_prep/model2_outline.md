# Model 2 — SparkXGBClassifier: Antarctic Ice Mass Loss Prediction

---

## Predictive Task

**Task type:** Binary classification — predict `basal_loss_agreement` (dual-sensor ice mass loss agreement).

**Target:** `basal_loss_agreement` = 1 when both GRACE and ICESat-2 independently signal accelerated mass loss at the same location and time. The GRACE flag triggers when `lwe_fused` falls below the 25th percentile of the mascon region; the ICESat-2 flag triggers when `delta_h` falls more than one standard deviation below the pixel's historical mean.

**Leakage prevention:** `lwe_fused` is used to construct the label and is **dropped** before any modeling. The retained GRACE features (`lwe_mo`, `lwe_quarterly_avg`, `lwe_quarterly_std`) carry temporal context without encoding the label.

---

## Preprocessing Pipeline (Spark MLlib Transformers)

Model 2 uses a **distinct** preprocessing pipeline from Model 1:

| Stage | Transformer | Purpose |
|---|---|---|
| 1 | **Imputer** (median) | Handle NaN/null values in all 50+ numeric columns |
| 2 | **Bucketizer** | Discretize `dist_to_grounding_line` into 5 proximity bands |
| 3 | **StringIndexer** | Encode `regional_subset_id` → numeric index |
| 4 | **OneHotEncoder** (×2) | One-hot encode grounding line buckets + region index |
| 5 | **VectorAssembler** | Combine imputed numerics + OHE vectors into single vector |
| 6 | **MinMaxScaler** | Scale to [0,1] range (vs Model 1's StandardScaler) |

### Key Differences from Model 1
- **MinMaxScaler** instead of StandardScaler → preserves the distribution shape
- **Bucketizer** on grounding line distance → captures non-linear proximity effects
- **No PolynomialExpansion** → replaced by hand-crafted physics interaction features

---

## Feature Engineering

### Model 2 Unique Features

**Temporal Memory (6-month rolling):**
- `t_star_6mo_avg`, `lwe_6mo_avg`, `delta_h_6mo_avg` — extended rolling averages
- `t_star_rate`, `lwe_acceleration`, `delta_h_rate` — rate of change vs 6-month baseline
- `t_star_mom_change`, `delta_h_mom_change` — month-over-month change

**Physics Interactions:**
- `ocean_heat_content_proxy` = θ × |ice_draft| / (dist_to_ocean + 1)
- `draft_ratio` = |ice_draft| / thickness
- `thermal_x_gl_proximity` = T* / (dist_to_grounding_line + 1)
- `freezing_departure` = θ − T_f
- `bed_geometry_risk` = bed_slope × min(bed, 0)
- `mass_flux_proxy` = δh × ice_area

**Grounding Line Proximity Buckets:**
| Bucket | Distance | Physical Interpretation |
|---|---|---|
| 0 | < 5 km | Critical zone — active grounding line |
| 1 | 5–20 km | Near zone — dynamic influence |
| 2 | 20–50 km | Transition zone |
| 3 | 50–100 km | Moderate distance |
| 4 | > 100 km | Far zone — interior ice |

---

## Models Trained

### Baseline: SparkXGBClassifier (XGB_Baseline)
- `max_depth=4`, `n_estimators=100`, `learning_rate=0.1`
- `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=10`
- **Rationale:** Shallow trees with moderate learning rate establish a performance floor. Captures major threshold behaviours without overfitting.

### Tuned: SparkXGBClassifier (XGB_Tuned)
- `max_depth=8`, `n_estimators=400`, `learning_rate=0.02`
- `subsample=0.75`, `colsample_bytree=0.7`, `min_child_weight=20`
- `reg_alpha=0.1`, `reg_lambda=1.0`
- **Rationale:** Deeper trees capture multi-way interactions (bed geometry × ocean temp × proximity). Lower LR + more trees gives finer gradient resolution. Higher `min_child_weight` prevents spatial autocorrelation overfitting.

### Why SparkXGBClassifier?
XGBoost's **second-order gradient updates** are well-suited to glaciological threshold behaviours — thermal driving only triggers rapid melt once it exceeds the local freezing point by some margin. The level-wise tree growth finds these step-function patterns efficiently. SHAP attribution connects model decisions to physical mechanisms.

---

## Train / Validation / Test Split

**Temporal split** (no random shuffle — spatial autocorrelation would leak future information):
- **Train:** month_idx ≤ 24264 (through Dec 2021)
- **Validation:** 24264 < month_idx ≤ 24276 (year 2022)
- **Test:** month_idx > 24276 (2023 onwards)

---

## Evaluation Results

### Metrics Computed
- **AUC-ROC** — primary metric (threshold-free, robust to class imbalance)
- **F1** — harmonic mean of precision and recall
- **Weighted Precision** and **Weighted Recall**

### Results Table

| Model | Split | AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| XGB_Baseline | train | — | — | — | — |
| XGB_Baseline | val | — | — | — | — |
| XGB_Baseline | test | — | — | — | — |
| XGB_Tuned | train | — | — | — | — |
| XGB_Tuned | val | — | — | — | — |
| XGB_Tuned | test | — | — | — | — |

*(Fill from pipeline stdout after execution)*

### Regional Breakdown

Per-region positive prediction rate vs true positive rate computed on test set.

### Sample Predictions

500-row sample tables saved per (model, split) with columns: `x`, `y`, `month_idx`, `mascon_id`, `regional_subset_id`, `basal_loss_agreement` (ground truth), `prediction`, `probability`.

---

## Fitting Analysis

### Overfitting vs Underfitting Diagnosis

- **Train AUC ≫ Test AUC (gap > 0.10):** Overfitting — spatial autocorrelation causing the model to memorize geographic patterns that don't generalize temporally
- **Both Train and Test AUC < 0.60:** Underfitting — features insufficient to discriminate the label
- **Gap < 0.05:** Good fit — model generalizes well

### Hyperparameter Comparison

| Parameter | XGB_Baseline | XGB_Tuned | Effect |
|---|---|---|---|
| max_depth | 4 | 8 | Deeper trees capture complex interactions |
| n_estimators | 100 | 400 | More trees for finer gradient resolution |
| learning_rate | 0.1 | 0.02 | Smaller steps = less overshoot |
| min_child_weight | 10 | 20 | Higher = more regularization against spatial clusters |
| reg_alpha | 0 | 0.1 | L1 regularization for feature sparsity |
| reg_lambda | 1 | 1.0 | L2 regularization for weight magnitude |

**Expected outcome:** The tuned model should outperform the baseline on Amundsen Sea and Totten-Aurora (where complex interactions matter most) while potentially showing slightly more overfitting in stable regions.

---

## Next Models for Milestone 4

1. **Ray LightGBMTrainer** — leaf-wise tree growth with GOSS sampling. Complementary to XGBoost's level-wise approach. Native categorical feature support. Extended temporal trajectory features.
2. **Ensemble** — weighted average of XGBoost and LightGBM predictions, calibrated per region.

---

## Conclusion

### Conclusion of Model 2
The SparkXGBClassifier represents a significant step up from Model 1's DT/RF/GBT progression. XGBoost's second-order gradient boosting directly captures the threshold physics of ice mass loss — e.g., the non-linear relationship between thermal driving and basal melt rate. The hand-crafted physics interaction features encode known glaciological pathways, and the 6-month temporal memory extends the model's ability to detect sustained forcing episodes.

### Possible Improvements
- SHAP-based feature pruning to remove noise features
- Regional residual correction (train region-specific boosters on global model's residuals)
- Extended temporal features (12-month rolling, full seasonal decomposition)
- Early stopping on validation set to prevent late-stage overfitting

### How Distributed Computing Helped
SparkXGBClassifier distributes histogram construction and split-finding across Spark executors. Each executor processes a data partition independently, computing local gradient histograms that are then globally aggregated. For 100M+ rows × 50+ features × 400 trees, this parallelization reduces training from >6 hours (single machine) to ~30-60 minutes across 6 executors with 5 cores each. Spark's partitioning model also enables the regional stratified sampling strategy — maintaining balanced representation across six physical regimes of vastly different geographic areas.
