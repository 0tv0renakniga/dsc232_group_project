# Milestone 3: Pre-Processing
* This milestone covers:
    * completing preprocessing using Spark and building your first distributed model
    * evaluate the model and analyze where it fits on the underfitting/overfitting spectrum
* **IMPORTANT:** 
    * must use a distributed implementation 
    * simple linear regression is not acceptable unless combined with substantial feature engineering using Spark
    * models that only run on the driver (e.g., scikit-learn on collected data) are not acceptable
    * training should demonstrate meaningful parallelization across multiple executors

## 1. Compelete Preprocessing
* what features were engineered and why
* give examples how the following was used:
    * Scaling/Transforming: Use StandardScaler, MinMaxScaler, Normalizer from pyspark.ml.feature
    * Imputing: Use Imputer for handling missing values
    * Encoding: Use StringIndexer, OneHotEncoder, or VectorAssembler
    * Feature Engineering: Create new features using Spark SQL functions, PolynomialExpansion, or custom transformations

[INSERT table HERE like example below]

| Feature | Description | Why | Spark MLlib transformers | Spark DataFrame operations |
|---|---|---|---|---|
| Row 1 Col 1 | Row 1 Col 2 | Row 1 Col 3 | Row 1 Col 4 | Row 1 Col 5 |
| Row 2 Col 1 | Row 2 Col 2 | Row 2 Col 3 | Row 2 Col 4 | Row 2 Col 5 |



## 2. Train First Model 
* which distributed model was trained
    * [INSERT RESPONSE HERE] 

* why select xgboost.spark.SparkXGBClassifier
    * [INSERT RESPONSE HERE]  

* why were hyperparameter selected for xgb tuned given xgb baseline results
    * [INSERT RESPONSE HERE]  

* Training must use multiple executors/workers(verify via Spark UI)
    * Show spark config used
        
        ```python

        #[SPARK CONFIG HERE]  
        ```
    
    * Show screenshot of spark UI(note: need to grab)
        * [INSERT SCREEN SHOT HERE]  

* Evaluate your model: compare training vs. test error 
* why did you select the test metrics you did
* For supervised learning: include example ground truth and predictions for train, validation, and test sets

[INSERT into multiple TABLEs like using below]
[============================================================
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
========================================================================]


[INSERT PLOTS]

## 3. Fitting Analysis
* Where does your model fit in the fitting graph (underfitting vs. overfitting)?
    * [INSERT COMMENT ON PLOT AND TABLE ABOVE]
* Build at least one model with different hyperparameters and compare results
    * [INSERT SIDE BY SIDE SPARK CONFIG CONPARISON]
* Which model performs best and why?
    * [INSERT COMMENT HERE]
* What are the next models you are thinking of for Milestone 4 and why?
    * [EXPLAIN PCA/METALEARNER/ENSEMBE/FIXED STACK]

## 4. Conclusion section
* What is the conclusion of your 1st model?
    * [INSERT COMMENT HERE]: 
        * over trained
        * why xgboost overtrain
        * differences between xgb baseline and xgb tuned: why hyper parameter was tweaked
* What can be done to possibly improve it?
    * [INSERT COMMENT ON NEXT MODEL]: how it fixes region imbalance and over training 
* How did distributed computing help with this task?
    * [INSERT COMMENT HERE]
 

