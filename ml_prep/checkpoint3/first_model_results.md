# First Model Result
* [INSERT COMMENT ON WHY this MODEL]
    * `SparkXGBClassifier`

## XGB Baseline Results
* The args used:

    ```python
        "XGB_Baseline": dict(
            max_depth=4,
            n_estimators=50 if MODE == "local" else 100,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
        )
    ```

* The results:

```python
============================================================
  DATA LOADING
============================================================
[load] 72 columns from /expanse/lustre/projects/uci157/rrogers/data/ml_ready_unified
  train:  199,191,383 rows, pos=5,988,412, rate=0.030064
  val  :   72,431,730 rows, pos=1,411,047, rate=0.019481
  test :  126,716,756 rows, pos=4,644,918, rate=0.036656

============================================================
  TRAINING: XGB
============================================================
[xgb] Preprocessed data found, skipping to training.
[xgb] Refitting preprocessor on sample for val/test inference...
[preprocess:xgb] 64 numeric + Bucketizer + OHE + MinMaxScaler
[xgb] Preprocessor refit complete.
[xgb] Training data ready: 65,865,547 rows | columns: ['features', 'basal_loss_agreement', 'weightCol']

============================================================
  TRAINING: XGB_Baseline
============================================================
  [XGB_Baseline] train    ROC=0.9304  PR=0.5103  F1=0.7512  Prec=0.9279  Rec=0.6841
  [XGB_Baseline] Saved -> /expanse/lustre/projects/uci157/rrogers/dataunified_output/preds_XGB_Baseline_train
  [XGB_Baseline] val      ROC=0.6969  PR=0.0475  F1=0.9700  Prec=0.9636  Rec=0.9776
  [XGB_Baseline] Saved -> /expanse/lustre/projects/uci157/rrogers/dataunified_output/preds_XGB_Baseline_val
  [XGB_Baseline] test     ROC=0.7403  PR=0.0914  F1=0.9440  Prec=0.9346  Rec=0.9558
  [XGB_Baseline] Saved -> /expanse/lustre/projects/uci157/rrogers/dataunified_output/preds_XGB_Baseline_test

  [XGB_Baseline] Regional breakdown:
    amundsen_sea               pred=0.0029  true=0.1155  n=10,869,309
    antarctic_peninsula        pred=0.0870  true=0.0396  n=14,578,163
    lambert_amery              pred=0.0003  true=0.0099  n=27,227,872
    ronne                      pred=0.0003  true=0.0224  n=23,682,149
    ross                       pred=0.0011  true=0.0440  n=34,371,219
    totten_and_aurora          pred=0.0010  true=0.0312  n=15,988,044

========================================================================
  RESULTS SUMMARY
========================================================================
  Model                     Split    ROC-AUC   PR-AUC       F1     Prec      Rec
  ----------------------------------------------------------------------
  XGB_Baseline              train     0.9304   0.5103   0.7512   0.9279   0.6841
  XGB_Baseline              val       0.6969   0.0475   0.9700   0.9636   0.9776
  XGB_Baseline              test      0.7403   0.0914   0.9440   0.9346   0.9558
========================================================================

========================================================================
  FITTING ANALYSIS
========================================================================

  XGB_Baseline:
    Train ROC-AUC: 0.9304
    Test  ROC-AUC: 0.7403  (gap: 0.1902)
    Test  PR-AUC:  0.0914
    -> OVERFITTING
========================================================================

============================================================
  ERROR ANALYSIS PLOTS
============================================================
  Plotly not available, skipping plots.
  Plotly not available, skipping temporal plots.

========================================================================
  CONCLUSION
========================================================================

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
     Training on balanced data (1:10 ratio) forces
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
========================================================================

```

## XGB Tuned Results
* The args used:

    ```python
        "XGB_Tuned": dict(
            max_depth=6 if MODE == "local" else 8,
            n_estimators=100 if MODE == "local" else 400,
            learning_rate=0.05 if MODE == "local" else 0.02,
            subsample=0.75,
            colsample_bytree=0.7,
            min_child_weight=15 if MODE == "local" else 20,
            reg_alpha=0.1,
            reg_lambda=1.0,
        )
    ```

* The results:

```python
[spark] JVM max heap: 80.0 GB | parallelism: 24
[skip] Features exist at /expanse/lustre/projects/uci157/rrogers/data/ml_ready_unified

============================================================
  DATA LOADING
============================================================
[load] 72 columns from /expanse/lustre/projects/uci157/rrogers/data/ml_ready_unified
  train:  199,191,383 rows, pos=5,988,412, rate=0.030064
  val  :   72,431,730 rows, pos=1,411,047, rate=0.019481
  test :  126,716,756 rows, pos=4,644,918, rate=0.036656

============================================================
  TRAINING: XGB
============================================================
[xgb] Preprocessed data found, skipping to training.
[xgb] Refitting preprocessor on sample for val/test inference...
[preprocess:xgb] 64 numeric + Bucketizer + OHE + MinMaxScaler
[xgb] Preprocessor refit complete.
[xgb] Training data ready: 65,865,547 rows | columns: ['features', 'basal_loss_agreement', 'weightCol']

============================================================
  TRAINING: XGB_Tuned
============================================================
  [XGB_Tuned] train    ROC=0.9422  PR=0.5655  F1=0.7667  Prec=0.9292  Rec=0.7037
  [XGB_Tuned] Saved -> /expanse/lustre/projects/uci157/rrogers/dataunified_output/preds_XGB_Tuned_train
  [XGB_Tuned] val      ROC=0.6957  PR=0.0430  F1=0.9705  Prec=0.9630  Rec=0.9794
  [XGB_Tuned] Saved -> /expanse/lustre/projects/uci157/rrogers/dataunified_output/preds_XGB_Tuned_val
  [XGB_Tuned] test     ROC=0.7490  PR=0.0900  F1=0.9451  Prec=0.9343  Rec=0.9608
  [XGB_Tuned] Saved -> /expanse/lustre/projects/uci157/rrogers/dataunified_output/preds_XGB_Tuned_test

  [XGB_Tuned] Regional breakdown:
    amundsen_sea               pred=0.0001  true=0.1155  n=10,869,309
    antarctic_peninsula        pred=0.0283  true=0.0396  n=14,578,163
    lambert_amery              pred=0.0002  true=0.0099  n=27,227,872
    ronne                      pred=0.0003  true=0.0224  n=23,682,149
    ross                       pred=0.0011  true=0.0440  n=34,371,219
    totten_and_aurora          pred=0.0007  true=0.0312  n=15,988,044

========================================================================
  RESULTS SUMMARY
========================================================================
  Model                     Split    ROC-AUC   PR-AUC       F1     Prec      Rec
  ----------------------------------------------------------------------
  XGB_Tuned                 train     0.9422   0.5655   0.7667   0.9292   0.7037
  XGB_Tuned                 val       0.6957   0.0430   0.9705   0.9630   0.9794
  XGB_Tuned                 test      0.7490   0.0900   0.9451   0.9343   0.9608
========================================================================

========================================================================
  FITTING ANALYSIS
========================================================================

  XGB_Tuned:
    Train ROC-AUC: 0.9422
    Test  ROC-AUC: 0.7490  (gap: 0.1932)
    Test  PR-AUC:  0.0900
    -> OVERFITTING
========================================================================

============================================================
  ERROR ANALYSIS PLOTS
============================================================
  [XGB_Tuned] Plotly unavailable even after install attempt, skipping.
  [XGB_Tuned] Plotly unavailable even after install attempt, skipping.

========================================================================
  CONCLUSION
========================================================================

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
     Training on balanced data (1:10 ratio) forces
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
========================================================================
```



