# Preprocessing Plan

## Antarctica Sparse Feature Store — Data Preparation Game Plan

> **Context**: This plan describes the theoretical preprocessing strategy for
> `antarctica_sparse_features.parquet`, a 28-column fused geospatial dataset
> containing ICESat-2 elevation anomalies, GRACE mass changes, Bedmap3
> topography, and GLORYS ocean reanalysis data across the Antarctic ice sheet.

> **Target**: PySpark ML pipeline on SDSC (32 cores / 128 GB).

> **Important**: This document is a conceptual roadmap — it contains NO
> implementation code. PySpark operation names are referenced as function
> signatures for clarity, not as runnable code.

---

## 1. Temporal Feature Engineering: Cyclic Encoding of `exact_time`

### 1.1 The Problem with Raw Timestamps

The `exact_time` column is a `timestamp_ntz` (timestamp without timezone). Using it directly
as a numeric feature (e.g. Unix epoch seconds) would impose a false linear ordering: the model
would learn that December 2025 is "farther" from January 2024 than November 2025, when in
reality December and January are seasonally adjacent. Geophysical processes in Antarctica —
ice dynamics, ocean circulation, ice-shelf basal melting — are driven by **cyclical seasonal
forcing**, not the raw date.

### 1.2 The Solution: Sine/Cosine Cyclical Encoding

Extract the **month-of-year** (1–12) from `exact_time`, then project it onto a unit circle
using sine and cosine transforms:

- $\text{month}_{sin} = sin(\frac{2\pi \cdot month_{i}}{12})$
- $\text{month}_{cos} = cos(\frac{2\pi \cdot month_{i}}{12})$

This encoding has three critical properties:

1. **Continuity at the year boundary**: December (month=12) and January (month=1) are
   adjacent on the unit circle, with a smooth gradient between them. A raw integer month
   would create a discontinuous jump from 12 → 1.

2. **Equal distance for equal time gaps**: Months 3 months apart always have the same
   Euclidean distance in (sin, cos) space, regardless of which months they are.

3. **Two-component uniqueness**: A single sine function has an ambiguity (sin maps both
   month 2 and month 10 to the same value). The cosine component resolves this, making
   every month a unique (sin, cos) pair.

### 1.3 Why Not Other Approaches?

- **One-hot encoding of month** would create 12 binary columns, inflating dimensionality
  and losing the notion of temporal proximity (January and February would be as "far apart"
  as January and July).
- **Integer month (1–12)** imposes a false ordinal scale (December=12 appears maximally
  far from January=1).
- **Unix epoch** captures absolute time but not cyclical seasonality. If absolute temporal
  trend is also needed, it can be included as a separate feature.

### 1.4 PySpark MLlib Approach

The transformation requires extracting the month from the timestamp using
`F.month(F.col("exact_time"))`, then applying `F.sin()` and `F.cos()` with the
appropriate scaling factor (2π / 12). This produces two new continuous columns
(`month_sin`, `month_cos`) that replace `exact_time` in the feature vector. The
original `exact_time` column is retained for metadata/indexing but excluded from the
ML feature assembler.

---

## 2. Handling Missing Values in Sparse Data

### 2.1 Why Missing Values Exist

Missing data in this dataset is **not random** — it is **structurally determined
by physics and sensor geometry**:

| Feature Category | Why Null | Expected Null Rate |
|---|---|---|
| **Coordinates** (y, x, exact_time) | Never null — spatiotemporal backbone | 0% |
| **Identifiers** (mascon_id) | Always populated — spatial index | 0% |
| **Bedmap3 Static** (surface, bed, thickness, bed_slope, dist_to_grounding_line) | Null at grid cells outside Bedmap3 coverage (rare in Antarctic interior) | <5% |
| **Ocean / Coastal** (clamped_depth, dist_to_ocean, ice_draft) | Null at grounded-ice pixels far from the coast | 10–40% |
| **Ocean Reanalysis** (thetao_mo, so_mo, t_f_mo, t_star_mo, quarterly stats) | Null at grounded-ice pixels with no ocean interface — physically meaningful absence | 50–90% |
| **ICESat-2 Dynamic** (delta_h, ice_area, surface_slope, h_surface_dynamic) | Null where ICESat-2 had no valid return signal (cloud cover, off-track) | 10–30% |
| **GRACE Mass** (lwe_mo, lwe_quarterly_avg/std, lwe_fused) | Null outside GRACE mascon footprint or where fused signal could not be computed | 30–60% |

### 2.2 Strategy Per Category

**Coordinates + Identifiers** — No action needed. Always populated.

**Bedmap3 Static** — Forward-fill using nearest spatial neighbour within the same mascon.
Bedmap3 covers the entire continent; these nulls are edge-of-grid artefacts. PySpark approach:
use a Window function partitioned by `mascon_id`, ordered by distance, with `F.last(ignorenulls=True)`.
Alternatively, since these are static features, a simple `fillna()` with the per-mascon median
is acceptable.

**Ocean / Coastal Features** — Two-level strategy:
1. **Create a binary indicator**: `has_ocean_data` = 1 if any ocean feature is non-null, 0 otherwise.
   This encodes the physical meaning of absence ("no ocean interface here") as a learnable feature.
2. **Zero-fill all ocean nulls**: After creating the indicator, fill `thetao_mo`, `so_mo`, `t_f_mo`,
   `t_star_mo`, `clamped_depth`, `dist_to_ocean`, `ice_draft`, and all quarterly statistics with 0.0.
   Zero is a physically neutral value (no ocean thermal forcing, no salinity signal).

**ICESat-2 Dynamic** — Two-tier approach:
1. **Per-pixel temporal forward-fill**: For each (y, x) pixel, carry the last valid observation
   forward through time. Ice elevation changes slowly (metres per year), so the most recent
   observation is a reasonable estimate. Use a Window partitioned by `(y, x)`, ordered by
   `exact_time`, with `F.last("delta_h", ignorenulls=True)`.
2. **Drop remaining nulls**: After forward-fill, any pixel with no valid observation at any
   time step is genuinely unobserved. Drop these rows rather than imputing — fabricating
   elevation data would be scientifically unsound.

**GRACE Mass** — Zero-fill for `lwe_mo`, `lwe_quarterly_avg`, and `lwe_quarterly_std`.
GRACE mascons that do not overlap a pixel contribute zero mass signal. For `lwe_fused`,
zero-fill only if all upstream inputs are also zero-filled; otherwise drop the row.

### 2.3 What NOT To Do

- **Do not mean-impute ocean features**. The mean of ocean temperature across coastal pixels
  has no physical meaning at a grounded-ice pixel 500 km inland.
- **Do not drop all rows with any null**. This would eliminate >90% of the dataset because
  ocean and GRACE features are structurally sparse.
- **Do not use global medians** for ICESat-2 features. The spatial structure of elevation
  change is highly heterogeneous — a global median would wash out the signal.

---

## 3. Feature Scaling and Transformations

### 3.1 Scaling Strategy

The 28 columns span vastly different physical units and magnitudes:
- Spatial coordinates: ±2M metres (EPSG:3031)
- Elevations: -2000 to +4000 metres
- Temperatures: -2 to +5 °C
- Mass anomalies: ~±0.01 m LWE
- Slopes: ~0.0 to 0.5 (dimensionless)

Without scaling, distance-based algorithms (k-NN, SVM, neural networks) would be
dominated by the high-magnitude features (coordinates, elevations) while ignoring
physically critical low-magnitude features (temperature, mass anomaly).

| Feature Group | Recommended Scaler | Rationale |
|---|---|---|
| **All continuous features** (default) | StandardScaler (zero-mean, unit-variance) | Required for distance-based models. Tree-based models (GBT, RF) are invariant to scaling but tolerate it. |
| **delta_h, lwe_fused** | RobustScaler (median / IQR-based) | Heavy outlier tails from satellite artefacts and extreme calving events. Median/IQR is resistant to outliers. |
| **dist_to_grounding_line, dist_to_ocean** | Log(1+x) followed by StandardScaler | Severely right-skewed distributions. Log-transform first produces a more Gaussian input for linear models. |
| **mascon_id** | No scaling (categorical) | Index variable. OneHotEncode or embed as a group-level feature. |
| **Cyclical features** (month_sin, month_cos) | No scaling needed | Already bounded to [-1, 1] by construction. |
| **has_ocean_data** (binary indicator) | No scaling needed | Binary 0/1 feature. |

### 3.2 Transformation Order

The order of transformations matters to avoid data leakage and ensure correctness:

1. **Train/test split FIRST** — Before any scaling or imputation statistics are computed.
2. **Imputation** — Forward-fill and zero-fill using training-set rules only.
3. **Log-transform** — Apply `log(1 + x)` to skewed distance features.
4. **Cyclical encoding** — Compute `month_sin` and `month_cos` from `exact_time`.
5. **VectorAssembler** — Combine all numeric features into a single vector column.
6. **StandardScaler / RobustScaler** — Fit on training data, transform both train and test.

### 3.3 What NOT To Do

- **Do not fit scalers on the full dataset** before splitting. This leaks test-set statistics
  into the training pipeline.
- **Do not one-hot encode `mascon_id`** into ~200 binary columns without first assessing
  whether a group-level encoding (e.g. target encoding or embedding) is more appropriate.
- **Do not apply PCA as a preprocessing step** without understanding the physical meaning
  of each component. PCA on geographic coordinates + physical features produces
  uninterpretable axes.

---

## 4. PySpark MLlib Operations Reference

This section maps each preprocessing step to the specific PySpark MLlib function or
operation that will be used in the implementation phase.

### 4.1 Missing Value Handling

| Operation | PySpark Approach |
|---|---|
| Per-pixel temporal forward-fill | `F.last("delta_h", ignorenulls=True).over(Window.partitionBy("y", "x").orderBy("exact_time"))` |
| Zero-fill GRACE features | `df.fillna({"lwe_mo": 0, "lwe_quarterly_avg": 0, "lwe_quarterly_std": 0})` |
| Create binary ocean indicator | `F.when(F.col("thetao_mo").isNotNull(), 1).otherwise(0)` |
| Drop rows still null after fill | `df.dropna(subset=["delta_h", "surface", "bed", "thickness"])` |
| Count nulls per column (diagnostic) | `df.agg(*[F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns])` |

### 4.2 Temporal Feature Engineering

| Operation | PySpark Approach |
|---|---|
| Extract month from timestamp | `F.month(F.col("exact_time"))` |
| Cyclical sine encoding | `F.sin(2 * math.pi * F.col("month_of_year") / 12)` |
| Cyclical cosine encoding | `F.cos(2 * math.pi * F.col("month_of_year") / 12)` |

### 4.3 Scaling and Encoding

| Operation | PySpark Approach |
|---|---|
| StandardScaler | `pyspark.ml.feature.StandardScaler(inputCol="features", outputCol="scaled", withMean=True, withStd=True)` |
| RobustScaler | `pyspark.ml.feature.RobustScaler(inputCol="features", outputCol="robust_scaled")` |
| VectorAssembler | `pyspark.ml.feature.VectorAssembler(inputCols=[...], outputCol="features")` |
| StringIndexer (mascon_id) | `pyspark.ml.feature.StringIndexer(inputCol="mascon_id", outputCol="mascon_idx")` |
| OneHotEncoder (mascon_id) | `pyspark.ml.feature.OneHotEncoder(inputCols=["mascon_idx"], outputCols=["mascon_vec"])` |
| Log transform (distances) | `F.log1p(F.col("dist_to_grounding_line"))` |

### 4.4 Feature Engineering

| Operation | PySpark Approach |
|---|---|
| Temporal lag (delta_h, 1 step) | `F.lag("delta_h", 1).over(Window.partitionBy("y", "x").orderBy("exact_time"))` |
| Rolling mean (3-step) | `F.avg("delta_h").over(Window.partitionBy("y", "x").orderBy("exact_time").rowsBetween(-2, 0))` |
| Interaction column | `F.col("surface_slope") * F.col("t_star_mo")` |
| Outlier clipping (±3σ) | `F.when(F.col("delta_h") > upper, upper).when(F.col("delta_h") < lower, lower).otherwise(F.col("delta_h"))` |

### 4.5 ML Pipeline Assembly

The complete preprocessing pipeline is assembled using PySpark ML `Pipeline`, which chains
all stages into a single, reproducible fit-transform workflow:

1. **Imputer stage** (forward-fill via Window, then dropna)
2. **Log-transform stage** (withColumn for skewed features)
3. **Cyclical encoding stage** (withColumn for sin/cos month)
4. **VectorAssembler** (combine all numeric features into a single vector)
5. **StandardScaler / RobustScaler** (normalise the assembled vector)
6. **Model** (GBTRegressor, RandomForestRegressor, etc.)

PySpark ML's `Pipeline.fit()` ensures that scaler statistics (mean, std) are computed on the
training set and applied consistently to the test set, preventing data leakage.

---

## Summary

| Preprocessing Step | Strategy | Key PySpark Tool |
|---|---|---|
| **Time features** | Extract month from `exact_time`, encode as sin/cos pair | `F.month()`, `F.sin()`, `F.cos()` |
| **Missing values** | Domain-aware: forward-fill dynamic, zero-fill ocean/GRACE, drop remaining | `F.last().over(Window)`, `fillna()`, `dropna()` |
| **Scaling** | StandardScaler (default), RobustScaler (outlier-heavy), Log1p (skewed distances) | `StandardScaler`, `RobustScaler`, `F.log1p()` |
| **Encoding** | Cyclical (month), OneHot (mascon_id), binary indicator (ocean presence) | `F.sin()`, `OneHotEncoder`, `F.when()` |
| **Feature engineering** | Lag features, rolling stats, interaction terms, outlier clipping | `F.lag().over(Window)`, `F.avg().over(Window)` |
