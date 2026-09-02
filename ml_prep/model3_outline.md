# Model 3: Stacking Ensemble for Antarctic Ice Mass Loss Prediction

---

## Predictive Task

**Task type:** Binary classification of `basal_loss_agreement` (dual-sensor ice mass loss agreement).

**Leakage prevention:** `lwe_fused` dropped; retained GRACE features carry temporal context only.

---

## Architecture: Two-Layer Stacking Ensemble

```
┌─────────────────────────────────────────────────────┐
│  Preprocessing (Spark MLlib)                        │
│  Imputer -> VectorAssembler -> Normalizer (L2)      │
└───────┬────────────────────────────┬────────────────┘
        │                            │
    ┌───▼──────────┐         ┌───────▼────────┐
    │ Base Learner 1│         │ Base Learner 2 │
    │ RandomForest  │         │ GBTClassifier  │
    │ (100 trees,   │         │ (150 iters,    │
    │  depth 10)    │         │  depth 6)      │
    └───────┬───────┘         └───────┬────────┘
            │ rf_probability          │ gbt_score
            │                         │
    ┌───────▼─────────────────────────▼────────┐
    │  Meta-Feature Construction               │
    │  rf_pos_prob + gbt_score + base_agreement│
    │  + ALL original numeric features         │
    └───────────────────┬──────────────────────┘
                        │
              ┌─────────▼──────────┐
              │ Meta-Learner       │
              │ SparkXGBClassifier │
              │ (on meta_features) │
              └────────────────────┘
```

This is fundamentally different from Models 1 and 2 (single-model approaches).

---

## Preprocessing Pipeline (Spark MLlib Transformers)

| Stage | Transformer | Purpose |
|---|---|---|
| 1 | **Imputer** (median) | Handle NaN/null values |
| 2 | **VectorAssembler** | Combine all numeric features into single vector |
| 3 | **Normalizer** (L2) | L2-normalize feature vector |

### Distinctiveness

| Aspect | Model 1 | Model 2 | Model 3 |
|---|---|---|---|
| **Scaler** | StandardScaler | MinMaxScaler | Normalizer (L2) |
| **Expansion** | PolynomialExpansion | None (hand-crafted) | None |
| **Encoding** | StringIndexer+OHE | Bucketizer+OHE | Integer (native categorical) |
| **Architecture** | Single model pipeline | Single model pipeline | **Stacking ensemble** |

---

## Feature Engineering (from model3_lgbm_feature_pipeline.py)

**Temporal Trajectory Features (unique to Model 3):**
- `delta_h_momentum`, `delta_h_acceleration`: velocity and acceleration of elevation change
- `delta_h_3mo_trend`, `delta_h_deseason`: trend and annual-cycle-removed anomaly
- `t_star_momentum`, `t_star_sustained_anomaly`: thermal driving dynamics
- `lwe_momentum`, `lwe_sustained_trend`: mass change direction

**Ocean PCA (unique to Model 3):**
- 4 principal components from 8 correlated ocean variables

---

## Models Trained

### Layer 1: Base Learners
Evaluated independently on train/val/test to establish single-model baselines.

### Layer 2: Meta-Learners

| Model | max_depth | n_estimators | learning_rate | Rationale |
|---|---|---|---|---|
| Stack_Baseline | 3 | 100 | 0.1 | Shallow meta-learner prevents overfitting to base predictions |
| Stack_Tuned | 6 | 300 | 0.02 | Deeper trees learn region-specific base learner reliability |

---

## Geographic Error Visualizations

Three Plotly interactive plots generated per model on the test set:

1. **Geographic Error Distribution**: EPSG:3031 scatter map colored by error type (TP/TN/FP/FN)
2. **Regional Error Rates**: Bar chart comparing false negative/positive rates per region
3. **Errors-Only Map**: Misclassified pixels highlighted to show spatial failure patterns

---

## Evaluation Results

| Model | Split | AUC | F1 |
|---|---|---|---|
| Base_RF | train | — | — |
| Base_RF | test | — | — |
| Base_GBT | train | — | — |
| Base_GBT | test | — | — |
| Stack_Baseline | train | — | — |
| Stack_Baseline | test | — | — |
| Stack_Tuned | train | — | — |
| Stack_Tuned | test | — | — |

*(Fill from pipeline stdout after execution)*

---

## Fitting Analysis

- Train-test gap > 0.10 → OVERFITTING (meta-learner memorising base predictions)
- Both < 0.60 → UNDERFITTING (base learners too weak)
- Gap < 0.05 → GOOD FIT

**Expected:** Stacking should outperform individual base learners because the meta-learner learns WHERE each base model is reliable: RF handles high-variance regions (Amundsen), GBT handles bias-dominated regions (Totten).

---

## Conclusion

The stacking ensemble combines complementary learners (variance-reducing RF + bias-reducing GBT) through a meta-learner that sees base predictions alongside original features. This enables region-aware model combination without explicitly training separate regional models.

**Improvements:** Out-of-fold base predictions, additional base learners, region-specific meta-learners.

**Distributed computing:** Training 3 models sequentially on 40 GB data requires Spark's distributed histogram construction and partition-level gradient aggregation. Single-machine training would take 10+ hours; Spark reduces this to ~1-2 hours across 6 executors.
