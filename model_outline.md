# What We're Predicting and Why It Matters

## The Prediction in Plain Language

At any given location on the Antarctic Ice Sheet, at any given month, we want to know: **is this patch of ice losing mass faster than normal, and do two independent satellites agree that it is?**

That's it. Every pixel in your dataset gets a yes or no answer to that question each month.

"Faster than normal" is defined regionally. fast for the Amundsen Sea is different from fast for the Ross Ice Shelf. And "two satellites agree" means GRACE (which weighs the ice from space by measuring gravitational pull) and ICESat-2 (which measures the surface height of the ice with a laser) are both flagging the same location at the same time. When they agree, you can be confident something real is happening rather than instrument noise or measurement artifact from a single sensor.

## Why This Matters

The Antarctic Ice Sheet contains enough ice to raise global sea levels by roughly 58 meters if it melted entirely. It won't , but even partial, accelerated loss from specific vulnerable glaciers (Thwaites, Pine Island, Totten) could contribute meters of sea level rise over the coming centuries, affecting hundreds of millions of people in coastal regions.

The problem is that we don't fully understand *where* loss is accelerating, *when* it begins, or *what drives the onset*. The two satellites measure different physical quantities , GRACE measures mass directly but at coarse resolution (~27 km), while ICESat-2 measures surface elevation at fine resolution (1 km) but elevation change is not the same as mass change without knowing ice density and dynamics. Neither sensor alone is sufficient. When they agree, something physically unambiguous is happening.

Your model learns to predict that agreement signal , and more importantly, learns *which physical conditions precede it*. Ocean temperature at the ice base, proximity to the grounding line, bed geometry below sea level, how much the thermal forcing has changed in recent months. If the model is working correctly, it has implicitly learned the physical pathway from ocean forcing to ice mass loss, expressed as a predictive function over your feature set.

This matters beyond the predictions themselves. A model that correctly identifies Amundsen Sea pixels as high-risk because of warm ocean water at depth, and correctly identifies Totten Glacier pixels as high-risk because of retrograde bed slope, is telling you something about the *mechanism* of loss in each region. That mechanistic attribution is what connects this work to the scientific objectives of satellite Earth observation programs , not just detecting loss after it happens, but understanding which physical configurations make a location vulnerable before the loss accelerates.

---

# Final Master Plan: Antarctic Ice Mass Loss Prediction

---

## The Predictive Task

**Task type:** Binary classification, one prediction per pixel per month.

**Label:** `basal_loss_agreement` , equals 1 when both GRACE and ICESat-2 independently signal accelerated mass loss at the same location and time. Equals 0 otherwise.

**Label construction:**

The GRACE flag triggers when `lwe_fused` falls below the 25th percentile of that pixel's mascon region for that specific month. This is a regionally-relative threshold , what counts as anomalous mass loss in the Amundsen Sea is calibrated against Amundsen Sea baselines, not a global cutoff.

The ICESat-2 flag triggers when `delta_h` falls more than one standard deviation below that pixel's own historical mean elevation change. This is a pixel-relative threshold , each location is compared against its own baseline behavior rather than a global average.

The label equals 1 only when both flags are true simultaneously. This conservative dual-sensor agreement requirement is the scientific core of the task. Single-sensor flags are noisier and harder to interpret physically. Agreement events are rarer but far more trustworthy as signals of genuine ice mass loss.

**Critical leakage prevention:** `lwe_fused` is used to construct the label and must be excluded from all model features without exception. Use `lwe_mo`, `lwe_quarterly_avg`, and `lwe_quarterly_std` as GRACE-derived features instead , these carry predictive signal through temporal context without encoding the label directly.

**Expected class distribution:** Positive events (agreement = 1) will likely represent 15–30% of records globally, with strong regional variation. Amundsen Sea and Antarctic Peninsula will have substantially higher positive rates than Ross and Ronne. Document the exact distribution before any modeling begins , it governs the weighting strategy and sets realistic expectations for what AUC values are achievable in each region.

---

## Regional Structure

The Antarctic Ice Sheet is not uniform. Six physically distinct regions are present in your data, each with a different dominant mechanism of ice loss. Treating these regions identically in a global model would average out the very signals you are trying to capture. The regional structure instead serves three functions: it informs sample weighting during training, it provides stratified evaluation that tests physical plausibility, and it generates scientifically interpretable outputs beyond standard accuracy metrics.

**Amundsen Sea** is the highest-priority region. Home to Thwaites and Pine Island glaciers, it is driven by warm Circumpolar Deep Water intruding beneath floating ice shelves and melting them from below. The expected dominant features here are ocean thermal driving at draft depth and proximity to an actively retreating grounding line.

**Antarctic Peninsula** is the most atmospherically sensitive region. Surface melting and episodic ice shelf collapse , rather than basal ocean forcing , are the primary drivers. Expected dominant features are surface slope change, fractional ice area loss, and the seasonal cycle. This region tests whether the model can distinguish atmospheric-driven loss from ocean-driven loss.

**Ross and Ronne Ice Shelves** are the two largest cold-cavity ice shelves in Antarctica. They act as buttresses restraining the flow of inland ice. Loss events here are rare and represent large-scale structural change rather than continuous thinning. The model should predict low positive rates here; systematic false positives in these regions indicate the model is overfitting to features that happen to correlate with mass loss elsewhere but are not mechanistically relevant in cold-cavity shelves.

**Lambert-Amery** is the primary drainage basin for East Antarctica. It is generally stable and serves as a scientific control , a region where the model should achieve moderate accuracy without being challenged by extreme dynamics. Poor performance here indicates a fundamental modeling problem.

**Totten and Aurora** represent the most scientifically consequential East Antarctic sector. Totten Glacier is grounded well below sea level on a retrograde bed slope, making it susceptible to Marine Ice Sheet Instability , a self-reinforcing retreat mechanism that, once triggered, is difficult to stop. The expected dominant features are bed geometry, bed below sea level, and grounding line vulnerability rather than ocean temperature, distinguishing this region mechanistically from the Amundsen Sea.

---

## Feature Engineering

### Static Geometry , Structural Vulnerability Fingerprint

These features are time-invariant per pixel and establish the inherent physical susceptibility of each location. They are derived from Bedmap3 and spatial calculations.

`bed`, `surface`, `thickness` are the raw subglacial geometry from Bedmap3. `bed_slope` captures the steepness of the bedrock beneath the ice , retrograde slopes (deepening inland) are the defining geometric signature of Marine Ice Sheet Instability. `ice_draft` measures how deep the ice base sits below sea level, controlling how much of the ice column is exposed to warm ocean water. `clamped_depth` constrains the effective ocean access depth. `dist_to_grounding_line` is the distance to the boundary where grounded ice transitions to floating ice shelf , the single most important spatial predictor of dynamic vulnerability. `dist_to_ocean` measures insulation from direct ocean contact. `bed_below_sea_level` is a binary flag derived as `bed < 0`, categorically marking marine-based ice.

**Engineered static interactions:**
- `draft_x_thermal_access = ice_draft / (dist_to_ocean + 1)` , combines draft depth with ocean proximity; deep draft near the ocean is the highest-risk physical configuration for basal melt
- `grounding_line_vulnerability = thickness / (dist_to_grounding_line + 1)` , thin ice close to the grounding line is structurally precarious
- `retrograde_flag` , binary indicator of retrograde bed slope in the inland direction, the geometric precondition for marine instability runaway

### Dynamic Ice State , ICESat-2 Observable Signals

These features vary by month and capture the evolving state of the ice surface as measured by ICESat-2.

`delta_h` is the raw elevation anomaly , the primary ICESat-2 observable. `ice_area` is fractional ice coverage, where decline precedes and accompanies mass loss. `h_surface_dynamic` is the total dynamic surface elevation combining static surface with elevation anomaly. `surface_slope` captures the local gradient of the dynamic surface.

**Engineered dynamic features:**
- `delta_h_deviation = delta_h – pixel_mean_delta_h` , deviation of the current month's elevation change from that pixel's own temporal baseline; more informative than the raw value because it captures *change in behavior* rather than absolute level
- `surface_slope_change` , month-over-month difference in `surface_slope`; a proxy for flow acceleration derived from your available data, approximating the kinematic signal the decadal survey prioritizes
- `regional_delta_h_percentile` , `delta_h` ranked within its regional subset for that month; transforms the raw signal into a regionally-relative measure, preventing stable shelf regions from suppressing the dynamic signal of outlet glaciers in a shared feature space

### Ocean Thermal Forcing , GLORYS12 Signal

This tier connects ocean state to ice loss and is the primary physical mechanism for basal melt in the Amundsen Sea and Totten regions.

`t_star_mo` is monthly thermal driving , the temperature above the local freezing point at the ice base depth, the most direct measure of the ocean's capacity to melt ice. `thetao_mo` is absolute ocean temperature. `so_mo` is salinity. `t_f_mo` is the freezing point temperature. All quarterly rolling statistics , `t_star_quarterly_avg`, `t_star_quarterly_std`, `thetao_quarterly_avg`, `thetao_quarterly_std` , are included. The standard deviation terms capture episodic warm water intrusion events, which are physically distinct from a sustained warm baseline. `lwe_mo`, `lwe_quarterly_avg`, `lwe_quarterly_std` provide GRACE-derived mass signal as features (not the fused label).

**Engineered ocean interactions:**
- `thermal_driving_x_draft = t_star_mo × ice_draft` , the physical melt rate scales with both thermal driving and draft depth; this interaction encodes that relationship directly and is expected to be the dominant feature in Amundsen Sea SHAP profiles
- `thermal_anomaly = t_star_mo – t_star_quarterly_avg` , deviation from recent rolling baseline; captures warm pulses rather than mean state, which is more relevant to episodic melt events
- `salinity_stratification_proxy = so_mo × clamped_depth` , approximates the halocline effect on basal melt; deeper saltier water has a lower freezing point
- `lwe_trend = lwe_mo – lwe_quarterly_avg` , is mass loss accelerating or consistent with the recent average; a leading indicator of regime change
- `regional_t_star_anomaly` , deviation of `t_star_mo` from that regional subset's monthly climatological mean; separates the Amundsen's structurally warm baseline from episodic anomalies, which are physically distinct drivers

### Temporal and Spatial Context

`sin_month` and `cos_month` derived from `month_idx mod 12` encode the seasonal cycle continuously without the ordinal artifact of using month as a raw integer. `mascon_id` provides fine-grained spatial grouping from GRACE. `mascon_mean_delta_h` and `mascon_mean_t_star` capture the regional coherence of signals , whether a pixel is experiencing isolated anomalous behavior or whether its entire mascon region is in a loss regime. `regional_subset_id` encodes which of the six physical regions each pixel belongs to. `regional_lwe_mean` is the mean GRACE signal across the entire regional subset for that month, capturing basin-scale mass budget state. For Model 2 specifically, 6-month rolling averages of `t_star_mo`, `lwe_mo`, and `delta_h` extend the temporal memory of the model.

---

## Regional Sample Weighting

The most important structural design decision in this plan is that sample weights are assigned per pixel before training, reflecting both regional scientific priority and local class rarity. This prevents the largest, most stable ice shelves from numerically dominating gradient updates while the physically critical Amundsen and Totten signals get averaged away.

**Component 1 , Regional importance weight:** Assigned based on scientific priority and expected class scarcity relative to importance. Amundsen Sea and Totten-Aurora receive a weight of 2.0 , highest scientific stakes, highest-risk dynamics, underrepresented in area relative to their importance. Antarctic Peninsula receives 1.5 , rapid change, atmospheric forcing distinct from other regions. Lambert-Amery receives 1.0 , stable baseline, neutral weight. Ross and Ronne receive 0.6–0.8 , numerically large, low event rate, scientifically important as a control but should not dominate training.

**Component 2 , Within-region class balance correction:** Multiply each positive-class pixel's weight by the negative-to-positive ratio within its own regional subset. This ensures positive events in every region receive adequate gradient signal even when the local positive rate is low.

**Final weight:** The product of both components, normalized so the mean weight across the full training set equals 1.0. This preserves effective sample size for learning rate calibration while redirecting gradient attention to scientifically priority regions.

In PySpark pass this as a `weightCol` to `SparkXGBClassifier`. In Ray LightGBM pass it via the `weight` parameter in the dataset constructor.

---

## Train / Validation / Test Split

Split on time. Never shuffle randomly , spatial autocorrelation means random splits leak future information through spatially correlated neighbors and will dramatically overestimate generalization performance.

**Training set:** All `month_idx` values through end of 2021. **Validation set:** Full year 2022, used for hyperparameter tuning and early stopping only. **Test set:** 2023 onward, held out entirely until final evaluation.

Every `mascon_id` and every regional subset appears in all three sets , you are testing temporal generalization, not spatial holdout. The scientific question is whether patterns learned from past months predict future mass loss, which is exactly the operational use case.

**Batch stratification:** Construct training batches with stratified sampling by `regional_subset_id` to ensure every batch contains pixels from all six regions. Without this, stochastic batch construction will produce batches dominated by Ross and Ronne pixels by geographic area, undermining the weighting scheme.

**Regional validation files:** Maintain a separate held-out validation slice per regional subset. All hyperparameter decisions should be evaluated on the global validation set, but regional validation files are used for the diagnostic evaluation framework.

---

## Model 1: SparkXGBClassifier

**Algorithm rationale:** XGBoost's depth-wise tree growth with second-order gradient updates is well-suited to the nonlinear threshold behaviors in glaciological systems , the fact that thermal driving only triggers rapid melt once it exceeds the local freezing point by some margin is exactly the kind of interaction XGBoost captures efficiently. SHAP values are a native output, enabling the physical plausibility evaluation that connects model outputs to decadal survey objectives.

**Feature philosophy:** Emphasize physically-motivated interaction terms , `thermal_driving_x_draft`, `grounding_line_vulnerability`, `draft_x_thermal_access`. These terms encode domain knowledge about how ocean heat reaches the ice base, and XGBoost will exploit them directly rather than needing to reconstruct the interaction from raw variables.

**`regional_subset_id` encoding:** Target-encode as the mean positive class rate per region computed on the training set only. This gives XGBoost a numeric prior probability of mass loss by regime without exposing categorical structure it cannot natively exploit.

**Imbalance handling:** The two-component regional weighting scheme passed via `weightCol` replaces the simpler `scale_pos_weight` approach. Each pixel's weight reflects both its regional scientific importance and its local class rarity.

**Key hyperparameters:** `max_depth` 6–10, `n_estimators` 300–600 with early stopping monitored on validation AUC, `learning_rate` 0.01–0.05, `subsample` and `colsample_bytree` 0.7–0.9, `min_child_weight` 10–20 to prevent overfitting on spatially autocorrelated pixel clusters.

**Primary scientific output:** SHAP value profiles computed per regional subset. The expected pattern , `thermal_driving_x_draft` dominating Amundsen Sea, bed geometry features dominating Totten-Aurora, surface dynamic features dominating Antarctic Peninsula , constitutes the physical plausibility test for the model.

---

## Model 2: Ray LightGBMTrainer

**Algorithm rationale:** LightGBM uses leaf-wise tree growth with Gradient-based One-Side Sampling, finding different decision boundaries than XGBoost's level-wise approach. This is not an incremental variation , it is a fundamentally different optimization path through the same feature space. More importantly, the feature philosophy for Model 2 is explicitly temporal: the question being asked is whether the *trajectory* of ocean forcing and ice state over recent months predicts the agreement label, rather than the instantaneous physical state at observation time. This is a genuinely complementary inductive bias.

**Feature philosophy:** In addition to all shared features, Model 2 uses 6-month rolling averages of `t_star_mo`, `lwe_mo`, and `delta_h`. These extended temporal windows give the model memory of sustained forcing episodes rather than just the current month's state. `regional_subset_id` is passed as a native LightGBM categorical , no encoding needed , allowing the model to learn separate split thresholds per region on all other features when conditioning on the regional identifier, effectively approximating region-conditioned behavior within a single global model.

**Regional residual correction:** After training the global LightGBM model, perform a second focused training pass on each regional subset using the global model's leaf predictions as an `init_score` offset. The regional model learns residuals on top of the global model's predictions rather than starting from scratch. This is the tree-based analog of fine-tuning and is most valuable for Amundsen Sea and Totten-Aurora, where the global model is most likely to underfit due to their physically distinct dynamics. The output is a global model plus six lightweight regional correction models.

**Key hyperparameters:** `num_leaves` 63–255, `min_data_in_leaf` 50–200 (set high given spatial autocorrelation to prevent memorization of spatially coherent clusters), `feature_fraction` and `bagging_fraction` 0.7–0.85, `lambda_l1` and `lambda_l2` tuned on validation AUC, `is_unbalance=False` with explicit sample weights instead.

---

## Evaluation Framework

### Global Metrics

AUC-ROC as the primary metric , threshold-free and robust to class imbalance. Precision-recall AUC as the secondary metric , more informative than ROC when the positive class is rare and costly to miss. F1 at an operationally chosen threshold. Cohen's Kappa to account for agreement by chance.

### Regional Diagnostic Matrix

Compute AUC-ROC, F1, and positive prediction rate for each model on each of the six regional held-out subsets. The result is a 2×6 matrix that is the primary scientific deliverable of the evaluation.

The interpretation standard is deliberately asymmetric. A model that achieves high AUC on Amundsen Sea and Totten-Aurora while achieving moderate AUC on Ross and Ronne is performing correctly , it is discriminating where discrimination is scientifically meaningful and appropriately treating stable shelves as the low-variability baselines they are. A model with uniformly high global AUC driven by Ross and Ronne performance but poor Amundsen discrimination has learned the wrong thing, regardless of how its headline metric looks.

### Physical Plausibility Check

For each region, extract the top 5 SHAP features from Model 1 and compare against the expected physical drivers established in the Regional Structure section. Document agreements and contradictions explicitly. This is not a pass/fail test , contradictions are findings worth investigating, not errors to suppress. A model that finds `lwe_trend` dominant in the Ross Ice Shelf might be detecting precursors to buttressing change that are not obvious from physical first principles.

### Model Disagreement Analysis

Compute the per-pixel disagreement rate between Model 1 and Model 2, stratified by regional subset. High disagreement in Totten-Aurora specifically would mean that instantaneous physical state (Model 1's strength) and temporal trajectory (Model 2's strength) are telling genuinely different stories about East Antarctic marine instability , a result worth reporting as a scientific finding rather than treating as a modeling inconsistency to resolve.

### Temporal Residual Plots

Plot test-set prediction error by `month_idx` separately for Amundsen Sea and Ross/Ronne. Error spikes in the Amundsen series that coincide with known Circumpolar Deep Water intrusion events, with the Ross series remaining flat, constitute evidence that the model is capturing physically real signals rather than spurious statistical correlations. This is the closest thing to external scientific validation available with your dataset.

---

## Summary Table

| Component | Design Decision | Scientific Justification |
|---|---|---|
| Label | Dual-sensor agreement (GRACE + ICESat-2) | Single-sensor flags are noisy; agreement events are physically unambiguous |
| Label thresholds | Regionally and pixel-relative | Prevents global baselines from masking regional signals |
| Feature set | Physical interactions + temporal rolling + regional anomalies | Encodes known physical pathways from ocean forcing to ice loss |
| Sample weights | Regional importance × class balance, normalized | Prevents stable shelf pixels from drowning out high-dynamic signals |
| Batch stratification | Stratified by `regional_subset_id` | Ensures all physical regimes receive gradient signal every batch |
| Temporal split | Train ≤2021, val 2022, test ≥2023 | Tests temporal generalization; avoids spatial autocorrelation leakage |
| Model 1 (XGBoost) | Physical interaction terms + SHAP attribution | Captures nonlinear thresholds; produces mechanistic feature importance |
| Model 2 (LightGBM) | Temporal trajectory + regional residual correction | Captures forcing history; region-conditioned via native categoricals and init_score fine-tuning |
| Evaluation | 2×6 regional diagnostic matrix + plausibility check | Connects ML performance to physical expectations and decadal survey objectives |

---

## Milestone 3 Requirements

Yes to all three, and I can tell you exactly how each one is satisfied or needs a small addition. Let me go through them honestly.

---

## 1. Training Requirements

**Multiple executors/workers** , satisfied by design. SparkXGBClassifier distributes across Spark executors natively; Ray LightGBMTrainer distributes across Ray workers via `ScalingConfig`. Both models require a cluster to run as specified. No changes needed.

**Training vs. test error comparison** , satisfied. The temporal split (train ≤2021, val 2022, test ≥2023) produces three distinct error measurements. You compute AUC-ROC and F1 on each split separately and plot them side by side. The gap between training and test error is your bias-variance diagnostic.

# **IMPORTANT: minor change to master plan needed**
**Ground truth and predictions for train, validation, and test sets** , this requires a small explicit addition to the plan. After training each model, save a sample table of roughly 500–1000 rows per split containing the pixel identifier (`x`, `y`, `month_idx`), the regional subset, the ground truth label, the predicted probability, and the predicted binary class at your chosen threshold. This is straightforward to produce from both Spark and Ray and gives you the concrete example table the requirement asks for. Include at least one example from each regional subset in your sample to make it scientifically illustrative.

---

## 2. Questions the Plan Can Answer

**Where does the model fit on the fitting graph?**

The plan answers this directly through the training vs. validation vs. test AUC comparison. The expected result given your setup is mild overfitting , tree-based models on spatially autocorrelated data tend to memorize local geographic patterns that don't fully generalize temporally. The `min_child_weight` (XGBoost) and `min_data_in_leaf` (LightGBM) hyperparameters are specifically tuned to push back against this. If training AUC is substantially higher than test AUC the model is overfitting; if both are low and close together it is underfitting, likely meaning the features are insufficient to discriminate the label. The regional diagnostic matrix adds nuance , you may find the model overfits in Lambert-Amery (stable, learnable) and underfits in Antarctic Peninsula (high variance, atmospheric drivers not fully captured), which is itself a scientifically meaningful finding.

**Different hyperparameters comparison**

The plan as written tunes hyperparameters on the validation set but doesn't explicitly frame a comparison table. Add one deliberate comparison run for each model: train a "default" version with no tuning and a "tuned" version with your optimized hyperparameters, and report both on the regional diagnostic matrix. For XGBoost the most impactful axis to vary is `max_depth` (shallow at 4 vs. deep at 10) paired with `min_child_weight`. For LightGBM vary `num_leaves` (31 vs. 127 vs. 255). This gives you a concrete hyperparameter sensitivity story , deeper trees improve Amundsen Sea discrimination but increase overfitting in stable regions, which connects back to the physical interpretation.

**Which model performs best and why**

The plan produces a direct answer through the 2×6 regional diagnostic matrix. The expected outcome worth reasoning about in advance: Model 1 (XGBoost) will likely outperform on Amundsen Sea and Totten-Aurora where instantaneous physical state features like `thermal_driving_x_draft` are mechanistically decisive. Model 2 (LightGBM) will likely outperform on Antarctic Peninsula where the temporal trajectory of forcing , sustained warming over months rather than a single month's ocean state , better captures the episodic collapse dynamics. If this pattern holds, neither model is globally superior, which is itself the correct scientific answer and motivates the ensemble framing.

**Reasoning from SparkXGBClassifier to Ray LightGBMTrainer**

This is actually one of the strongest parts of the plan and the reasoning is layered. The first layer is algorithmic , XGBoost builds trees level-by-level with exact second-order gradients, making it precise but computationally heavier; LightGBM builds leaf-wise with gradient-based sampling, making it faster and often better at capturing asymmetric loss distributions like yours. The second layer is scientific , after examining XGBoost's SHAP profiles you have evidence about which features matter by region, and you specifically design LightGBM's feature set to extend temporal memory in regions where XGBoost's instantaneous features showed residual error. The third layer is infrastructure , moving from Spark to Ray tests a different distributed execution model, which is relevant to the distributed computing question. The progression is therefore not arbitrary; each choice is motivated by what you learned from the previous model.

**How distributed computing helped**

Your dataset spans Antarctica at 1 km resolution across multiple years of monthly observations. A rough lower bound on dataset size: Antarctica is approximately 14 million square kilometers, giving ~14 billion 1km pixels, though your actual coverage after masking ocean and no-data regions will be far smaller. Even at a few hundred million rows after temporal expansion, single-machine training of a gradient boosted model with 30+ features and 300–600 trees is impractical within reasonable time constraints. Spark parallelizes the histogram construction and split-finding across executors, each processing a partition of the data independently. Ray parallelizes the training across workers with shared memory for gradient aggregation. The specific answer to include is the comparison between estimated single-machine training time and actual distributed training time, which you can measure directly and report. Beyond speed, distributed computing enables the regional stratified sampling strategy , maintaining balanced batches across six geographic subsets of varying size is a data pipeline problem that Spark handles naturally through its partitioning model.

---

## 3. Existing Academic Work for Comparison

Yes, this is possible and there is relevant literature. Your task is novel enough that no paper will match it exactly, but three bodies of work provide legitimate comparison anchors.

The most directly comparable work is the line of research using machine learning to predict ice shelf basal melt rates from ocean and geometric features. Nakayama et al. and subsequent work using random forests and gradient boosting to predict melt from ocean temperature observations at the ice-ocean interface are methodologically close. Their reported skill metrics on held-out time periods give you a baseline for what AUC and correlation values are achievable on similar prediction targets.

The second comparison anchor is the IMBIE (Ice Sheet Mass Balance Inter-comparison Exercise) consensus estimates. IMBIE doesn't use ML but it provides the ground truth benchmark for where mass loss is occurring and at what rate across Antarctica's drainage basins. If your model's regional positive prediction rates agree qualitatively with IMBIE's reported loss patterns by basin , high loss in Amundsen and Totten sectors, low loss in Ross and Ronne , that constitutes external scientific validation even without a direct metric comparison.

The third anchor is the growing literature on ML emulators for ice sheet models, particularly work using neural networks and gradient boosting to emulate BISICLES and Elmer/Ice outputs. Bett et al. and related papers report skill scores for predicting ice dynamical quantities from geometric and ocean forcing inputs. Your features overlap substantially and their reported error ranges give you a defensible comparison even though the exact prediction target differs.

The honest framing for your write-up is to position your work as complementary to physics-based models rather than competing with them , you are predicting observational agreement between two satellite systems, not simulating ice dynamics, which is a distinct and valid contribution that existing emulator papers do not address.

---

## Bonus ML emulators (*qad derivation on single ply TP*) 
### 1. Emulator
* Possible to build a partial emulator that predicts delta_h as a continuous regression target given the ocean and geometric inputs, framed as emulating the surface mass balance component of an ice sheet model. 
    * This is scientifically legitimate but narrower than a full dynamic emulator. 
    * The comparison anchor here is Jouvet et al. and Bett et al., who build emulators for specific ice sheet model outputs using similar geometric and forcing inputs.
### 2. GP Emulator
* Gaussian Process emulator
    * This does not scale well.. expensive AF. 
    * **However**,at mascon level (`mascon_id`) inputs would be regional averages of ocean forcing, geometric properties, and ice state. 
        * The output would be regional delta_h or lwe_mo. 
        * A GP emulator at this scale would give you a probabilistic prediction of regional mass change as a function of ocean forcing and geometry, with calibrated uncertainty bounds. 
        * This is directly comparable to the GP emulators used in climate sensitivity analysis
        * Tools to handle high resolution:
            * Sparse GP methods (inducing point approximations like FITC or SVGP) to handle sub-mascon resolution.
            * GPyTorch or GPflow support these methods and scale to similar sized datasets
### 3. ML Data Assimilation Surrogate
* **def'n,**: Data assimilation combines a physics model with observations to produce an optimal estimate of the true system state 
    * think of it as a Bayesian update that corrects the model trajectory using real measurements. 
    * The standard algorithm in geosciences is the Ensemble Kalman Filter (EnKF) or 4D-Var. 
        * EnKF and 4D-Var are expensive: they require running the physics model many times. 
            * An ML surrogate replaces the physics model inside the assimilation loop, making the state estimation fast enough to run operationally.
* Possible Now:
    * A one-step-ahead prediction of delta_h and lwe_mo given the current month's full feature vector is directly constructable from what you have. 
    * the temporal structure needed exists in the dataset
        * `month_idx` partitioning (*hive partioned parquet*) enables construction of state at time t and state at time t+1 pairs. 
            * *note: This is simpler than a full assimilation system but represents the core ML component that would go inside one.*
### 4. Autoencoder for Anomaly Detection
* Rather than predicting a binary label, train an autoencoder on pixels in stable ice sheet regions (Ross, Ronne interior, Lambert-Amery) to learn what "normal" ice behavior looks like in your feature space. 
* Apply the trained autoencoder to the full dataset and flag high reconstruction error as anomalous.* Pixels near active grounding lines and in the Amundsen sector should show consistently high reconstruction error. 
* This is an unsupervised approach that does not require the dual-sensor label construction and could identify anomalous behavior in regions where your labeling threshold might miss early-stage changes

---

## Future Work
* add Ice velocity fields from MEaSUREs or ITS_LIVE (annual or monthly surface velocity at 1 km resolution) 
    * With velocity we can emulate the kinematic output of models like BISICLES directly. 
    * Without velocity, we can emulate the mass balance output (thickness change, surface elevation) but not the dynamic output (flow acceleration, grounding line migration).
    * **absence of ice velocity**: adding MEaSUREs or ITS_LIVE velocity fields would unlock the full dynamic emulator and PINN directions and substantially increase the scientific value of everything else.
* add atmospheric data
    * look up why but i think atmosphere/ocean surface flux coupling very important

