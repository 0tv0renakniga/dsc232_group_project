"""
sdsc_eda_notebook.py

-------------------------------------------------------------------------------
COMPUTATIONAL GLACIOLOGY : HPC-SCALE EDA + PREPROCESSING PIPELINE (REVAMPED)
-------------------------------------------------------------------------------
DATE:   2026-02-22
STATUS: PRODUCTION

Unified pipeline for the 1.38B-row Antarctica Digital Twin dataset.
Generates 4 high-impact aggregated visualisations and a complete PySpark ML
preprocessing pipeline.  ALL plotting follows the 'summarize-then-visualize'
pattern: Spark executors compute aggregates, only O(bins) or O(n²) scalars
transfer to the driver.  Raw data is NEVER collected.

Target Datasets
---------------
  fused_data/
    antarctica_sparse_features.parquet  (1.38B rows, 30 cols, Hive-partitioned)

  indiv_data/
    bedmap3_static.parquet, grace.parquet, icesat2_dynamic.parquet,
    ocean_dynamic.parquet

HPC Runtime Context
-------------------
  Cluster:           SDSC Expanse (32 cores / 128 GB shared partition)
  Executor Model:    5 cores × 6 instances (~19 GB each)
  Driver Memory:     10 GB
  Shuffle Strategy:  96 partitions (3× cores), AQE auto-coalesces

Shuffle Cost Audit
------------------
  * Geospatial density:   1 shuffle (GROUP BY x_bin, y_bin)  → 250K rows
  * delta_h histogram:    0 shuffles (RDD.histogram)
  * Correlation heatmap:  1 shuffle (VectorAssembler + Correlation.corr)
  * Temporal flux:        1 shuffle (GROUP BY month_idx)     → ~84 rows
  * Preprocessing:        2 shuffles (mascon GROUP BY + Window partitionBy)
  Total: 5 shuffles.  At 96 partitions with AQE, this is well within budget.

Figure Catalogue
----------------
  data/eda_plots/fig_geo_density_heatmap.png
  data/eda_plots/fig_delta_h_distribution.png
  data/eda_plots/fig_correlation_heatmap.png
  data/eda_plots/fig_temporal_flux.png
  data/eda_plots/preprocessing_report.txt

Usage
-----
  spark-submit --master local[*] utils/sdsc_eda_notebook.py
  spark-submit --master local[*] utils/sdsc_eda_notebook.py --dry-run

-------------------------------------------------------------------------------
"""

import os
import sys
import math
import time as _time
import traceback

import matplotlib
matplotlib.use("Agg")  # headless backend : MUST precede pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.types import NumericType, DoubleType
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ═══════════════════════════════════════════════════════════════════════════
# ██  MODULE 1 : HPC CONFIGURATION & SPARK SESSION
# ═══════════════════════════════════════════════════════════════════════════

# --- HPC Resource Allocation ---
TOTAL_CORES      = 32
TOTAL_MEMORY_GB  = 128

# --- Executor Geometry ---
# 5 cores per executor: Spark/HDFS best-practice ceiling.
EXECUTOR_CORES     = 5
EXECUTOR_INSTANCES = max((TOTAL_CORES - 1) // EXECUTOR_CORES, 1)  # -> 6

# --- Memory Budget ---
DRIVER_MEMORY_GB   = 10
EXECUTOR_MEMORY_GB = max(
    math.floor((TOTAL_MEMORY_GB - DRIVER_MEMORY_GB) / EXECUTOR_INSTANCES),
    1,
)

# --- Shuffle Tuning ---
# 3× total cores balances task overhead with throughput for GROUP BY ops.
# AQE auto-coalesces further if partitions are too small post-shuffle.
SHUFFLE_PARTITIONS = 3 * TOTAL_CORES  # -> 96

# --- Data Paths ---
FUSED_DATA_PATH = os.path.join("data", "fused_data",
                               "antarctica_sparse_features.parquet")
INDIV_DATA_DIR  = os.path.join("data", "indiv_data")
OUTPUT_DIR      = os.path.join("data", "eda_plots")

# --- Plot Config ---
FIG_DPI         = 200
GRID_RESOLUTION = 500   # 500×500 spatial heatmap grid
HISTOGRAM_BINS  = 100

# --- Dry-run Config ---
DRY_RUN         = "--dry-run" in sys.argv
DRY_RUN_LIMIT   = 100_000

# --- Feature Configuration ---
# Continuous features for correlation / scaling (excludes categoricals).
CONTINUOUS_FEATURES = [
    "surface", "bed", "thickness", "bed_slope",
    "dist_to_grounding_line", "clamped_depth", "ice_draft",
    "delta_h", "ice_area", "h_surface_dynamic", "surface_slope",
    "thetao_mo", "t_star_mo", "so_mo", "t_f_mo",
    "t_star_quarterly_avg", "t_star_quarterly_std",
    "thetao_quarterly_avg", "thetao_quarterly_std",
    "lwe_mo", "lwe_quarterly_avg", "lwe_quarterly_std", "lwe_fused",
]

# Features that need outlier clipping (heavy-tailed or satellite artifacts).
CLIP_FEATURES = ["delta_h", "lwe_fused", "thetao_mo", "so_mo", "t_star_mo"]

# Features for mascon-grouped median imputation (static geometry).
IMPUTE_FEATURES = [
    "bed", "thickness", "surface", "bed_slope", "dist_to_grounding_line",
]


def build_spark_session() -> SparkSession:
    """
    Construct an HPC-optimised SparkSession for 32 cores / 128 GB.

    Resource Allocation
    -------------------
    driver.memory         = 10 GB
    executor.instances    = 6     (5 cores each, 1 core for driver)
    executor.cores        = 5
    executor.memory       = 19 GB
    shuffle.partitions    = 96    (3 × 32 cores, AQE-coalesced)
    memory.fraction       = 0.7   (increased from 0.6 for heavy aggregation)

    Returns
    -------
    SparkSession

    Raises
    ------
    RuntimeError
        If derived executor memory < 1 GB.

    Complexity
    ----------
    O(1) : configuration only.
    """
    if EXECUTOR_MEMORY_GB < 1:
        raise RuntimeError(
            f"Derived executor memory is {EXECUTOR_MEMORY_GB} GB. "
            f"Allocation too small for {EXECUTOR_INSTANCES} executors."
        )

    driver_mem = f"{DRIVER_MEMORY_GB}g"
    exec_mem   = f"{EXECUTOR_MEMORY_GB}g"

    print("=" * 72)
    print("  HPC SparkSession Configuration (Revamped)")
    print("=" * 72)
    print(f"  TOTAL_CORES ............. {TOTAL_CORES}")
    print(f"  TOTAL_MEMORY_GB ......... {TOTAL_MEMORY_GB}")
    print(f"  DRIVER_MEMORY ........... {driver_mem}")
    print(f"  EXECUTOR_CORES .......... {EXECUTOR_CORES}")
    print(f"  EXECUTOR_INSTANCES ...... {EXECUTOR_INSTANCES}")
    print(f"  EXECUTOR_MEMORY ......... {exec_mem}")
    print(f"  SHUFFLE_PARTITIONS ...... {SHUFFLE_PARTITIONS}")
    print(f"  MEMORY_FRACTION ......... 0.7")
    print(f"  DRY_RUN ................. {DRY_RUN}")
    print("=" * 72)

    scratch_dir = os.environ.get(
        "TMPDIR", os.path.join(os.getcwd(), "spark_scratch")
    )

    spark = (
        SparkSession.builder
        .appName("HPC_Antarctic_Revamped_EDA")
        .config("spark.driver.memory",            driver_mem)
        .config("spark.executor.instances",        str(EXECUTOR_INSTANCES))
        .config("spark.executor.cores",            str(EXECUTOR_CORES))
        .config("spark.executor.memory",           exec_mem)
        .config("spark.sql.shuffle.partitions",    str(SHUFFLE_PARTITIONS))
        # --- Memory tuning ---
        .config("spark.driver.maxResultSize",      "4g")
        .config("spark.memory.fraction",           "0.7")
        # --- Network stability ---
        .config("spark.network.timeout",           "1200s")
        # --- Parallel partition discovery ---
        .config("spark.sql.sources.parallelPartitionDiscovery.threshold", "32")
        .config("spark.sql.sources.parallelPartitionDiscovery.parallelism", "64")
        # --- AQE ---
        .config("spark.sql.adaptive.enabled",                      "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled",   "true")
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128m")
        # --- Parquet ---
        .config("spark.sql.parquet.filterPushdown", "true")
        .config("spark.sql.parquet.mergeSchema",    "false")
        # --- Broadcast threshold (100 MB for target encoding join) ---
        .config("spark.sql.autoBroadcastJoinThreshold", str(100 * 1024 * 1024))
        # --- Disk spill ---
        .config("spark.local.dir", scratch_dir)
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    os.makedirs(scratch_dir, exist_ok=True)
    return spark


def load_fused_dataset(spark: SparkSession):
    """
    Load the fused Antarctica Parquet dataset.

    Uses recursiveFileLookup=true to bypass Hive partition discovery
    (avoids AnalysisException when partition cols duplicate data cols).

    In DRY_RUN mode, limits to DRY_RUN_LIMIT rows.

    Returns
    -------
    pyspark.sql.DataFrame
    """
    abs_path = os.path.abspath(FUSED_DATA_PATH)
    print(f"\n  Loading fused dataset: {abs_path}")

    df = (
        spark.read
        .option("recursiveFileLookup", "true")
        .option("mergeSchema", "true")
        .parquet(abs_path)
    )

    if DRY_RUN:
        print(f"  [DRY-RUN] Limiting to {DRY_RUN_LIMIT:,} rows.")
        df = df.limit(DRY_RUN_LIMIT)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# ██  STYLE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _apply_dark_style():
    """Apply a dark, modern matplotlib style globally."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         10,
        "axes.titlesize":    14,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11,
        "figure.facecolor":  "#1a1a2e",
        "axes.facecolor":    "#16213e",
        "axes.edgecolor":    "#e0e0e0",
        "text.color":        "#e0e0e0",
        "xtick.color":       "#b0b0b0",
        "ytick.color":       "#b0b0b0",
        "grid.color":        "#2a2a4a",
        "grid.alpha":        0.5,
    })


def _save_fig(fig, name: str):
    """Save figure to OUTPUT_DIR and close it."""
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"    → Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# ██  MODULE 2 : VISUAL EDA — 4 AGGREGATED PLOTS
# ═══════════════════════════════════════════════════════════════════════════

# ── FIGURE 1 : GEOSPATIAL DENSITY HEATMAP ─────────────────────────────────

def fig_geospatial_density(df):
    """
    500×500 binned geospatial heatmap of mean ice thickness.

    Strategy
    --------
    1. Compute global x/y bounds via a single .agg() call.
    2. Bin x,y into GRID_RESOLUTION equal-width buckets using F.floor().
    3. GROUP BY (x_bin, y_bin) → mean(thickness), count(*).
    4. Collect the ~250K-row result to driver (safe: 250K × 3 cols ≈ 6 MB).
    5. Render with matplotlib imshow().

    Shuffle Cost
    ------------
    1 shuffle for the GROUP BY.  Output is 250K rows max.
    At 96 partitions, each task processes ~15M rows → well-balanced.

    Scientific Interpretation
    -------------------------
    Reveals spatial coverage gaps (NaN cells in the heatmap) and the
    continental-scale thickness gradient from coast (thin) to interior
    (thick, up to ~4500m in East Antarctica).

    Complexity
    ----------
    O(N) scan + 1 shuffle (GROUP BY).

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Must contain columns: x, y, thickness.
    """
    print("\n  [FIG 1] Geospatial Density Heatmap (500×500 grid) ...")
    t0 = _time.perf_counter()

    # Step 1: Global bounds (single agg, no shuffle).
    bounds = df.agg(
        F.min("x").alias("x_min"), F.max("x").alias("x_max"),
        F.min("y").alias("y_min"), F.max("y").alias("y_max"),
    ).head()

    x_min, x_max = float(bounds["x_min"]), float(bounds["x_max"])
    y_min, y_max = float(bounds["y_min"]), float(bounds["y_max"])

    # Guard against degenerate ranges.
    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range <= 0 or y_range <= 0:
        print("    [SKIP] Degenerate x/y range — cannot bin.")
        return

    x_bin_width = x_range / GRID_RESOLUTION
    y_bin_width = y_range / GRID_RESOLUTION

    print(f"    x range: [{x_min:.0f}, {x_max:.0f}]  bin_width={x_bin_width:.1f}")
    print(f"    y range: [{y_min:.0f}, {y_max:.0f}]  bin_width={y_bin_width:.1f}")

    # Step 2: Bin and aggregate on executors.
    df_binned = (
        df.select("x", "y", "thickness")
        .where(F.col("thickness").isNotNull())
        .withColumn("x_bin",
                    F.least(
                        F.floor((F.col("x") - F.lit(x_min)) / F.lit(x_bin_width)).cast("int"),
                        F.lit(GRID_RESOLUTION - 1)
                    ))
        .withColumn("y_bin",
                    F.least(
                        F.floor((F.col("y") - F.lit(y_min)) / F.lit(y_bin_width)).cast("int"),
                        F.lit(GRID_RESOLUTION - 1)
                    ))
        .groupBy("x_bin", "y_bin")
        .agg(
            F.mean("thickness").alias("mean_thickness"),
            F.count("*").alias("point_count"),
        )
    )

    # Step 3: Collect to driver (~250K rows, ~6 MB).
    print("    Collecting binned grid to driver ...")
    rows = df_binned.collect()
    print(f"    Collected {len(rows):,} grid cells.")

    if not rows:
        print("    [SKIP] No binned data returned.")
        return

    # Step 4: Assemble into numpy array.
    grid = np.full((GRID_RESOLUTION, GRID_RESOLUTION), np.nan)
    for row in rows:
        xi = int(row["x_bin"])
        yi = int(row["y_bin"])
        if 0 <= xi < GRID_RESOLUTION and 0 <= yi < GRID_RESOLUTION:
            grid[yi, xi] = float(row["mean_thickness"])

    # Step 5: Plot.
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(
        grid, origin="lower", aspect="equal",
        cmap="viridis",
        extent=[x_min, x_max, y_min, y_max],
    )
    ax.set_xlabel("Easting [m] (EPSG:3031)")
    ax.set_ylabel("Northing [m] (EPSG:3031)")
    ax.set_title("Mean Ice Thickness — 500×500 Binned Grid")
    ax.xaxis.set_major_formatter(mticker.EngFormatter())
    ax.yaxis.set_major_formatter(mticker.EngFormatter())

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Mean Thickness [m]", fontsize=10)

    fig.tight_layout()
    _save_fig(fig, "fig_geo_density_heatmap.png")
    print(f"    [{_time.perf_counter() - t0:.1f}s]")


# ── FIGURE 2 : TARGET DISTRIBUTION HISTOGRAM (delta_h) ────────────────────

def fig_delta_h_distribution(df):
    """
    True histogram of delta_h with percentile-based outlier clipping.

    Strategy
    --------
    1. Compute 1st and 99th percentiles via approxQuantile (no shuffle).
    2. Filter to clipped range.
    3. Compute histogram via RDD.histogram() (executor-side, no shuffle).
    4. Plot bar chart with statistics overlay.

    Why Clip?
    ---------
    delta_h has extreme satellite-artifact tails (±100m) that compress the
    interesting near-zero distribution into an invisible sliver.  Clipping
    to [P01, P99] reveals the true glaciological signal.

    Complexity
    ----------
    O(N) for quantile + O(N) for histogram = O(N) total.  No shuffles.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Must contain column: delta_h.
    """
    print("\n  [FIG 2] Target Distribution (delta_h) ...")
    t0 = _time.perf_counter()

    # Step 1: Percentile clipping bounds.
    print("    Computing P01/P99 for delta_h ...")
    quantiles = df.stat.approxQuantile("delta_h", [0.01, 0.5, 0.99], 0.001)

    if len(quantiles) < 3 or quantiles[0] is None:
        print("    [SKIP] approxQuantile returned nulls — insufficient data.")
        return

    p01, median_val, p99 = quantiles
    print(f"    P01={p01:.4f}  Median={median_val:.4f}  P99={p99:.4f}")

    # Step 2: Clip and histogram.
    df_clipped = df.select("delta_h").where(
        (F.col("delta_h") >= p01) & (F.col("delta_h") <= p99)
    )

    rdd = df_clipped.rdd.map(lambda row: float(row[0]))
    if rdd.isEmpty():
        print("    [SKIP] Clipped RDD is empty.")
        return

    edges, counts = rdd.histogram(HISTOGRAM_BINS)

    # Step 3: Compute global stats for annotation (single agg, no shuffle).
    stats_row = df.agg(
        F.mean("delta_h").alias("mean"),
        F.stddev("delta_h").alias("std"),
        F.count("delta_h").alias("n"),
    ).head()
    mean_val = float(stats_row["mean"]) if stats_row["mean"] else 0.0
    std_val  = float(stats_row["std"])  if stats_row["std"]  else 0.0
    n_val    = int(stats_row["n"])      if stats_row["n"]    else 0

    # Step 4: Plot.
    centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
    widths  = [edges[i + 1] - edges[i] for i in range(len(counts))]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(centres, counts, width=widths, color="#35b779",
           edgecolor="#ffffff22", alpha=0.85)

    # Median + mean lines.
    ax.axvline(median_val, color="#ff6b6b", linestyle="--", linewidth=1.5,
               label=f"Median = {median_val:.4f} m")
    ax.axvline(mean_val, color="#ffd93d", linestyle="-.", linewidth=1.5,
               label=f"Mean = {mean_val:.4f} m")

    ax.set_xlabel("delta_h [m] (clipped to P01–P99)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Ice Elevation Change (delta_h)")
    ax.yaxis.set_major_formatter(mticker.EngFormatter())
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Statistics annotation box.
    textstr = (f"N = {n_val:,}\n"
               f"Mean = {mean_val:.4f} m\n"
               f"Std = {std_val:.4f} m\n"
               f"Clip: [{p01:.3f}, {p99:.3f}]")
    props = dict(boxstyle="round,pad=0.5", facecolor="#16213e",
                 edgecolor="#e0e0e0", alpha=0.9)
    ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=props)

    fig.tight_layout()
    _save_fig(fig, "fig_delta_h_distribution.png")
    print(f"    [{_time.perf_counter() - t0:.1f}s]")


# ── FIGURE 3 : PEARSON CORRELATION HEATMAP ────────────────────────────────

def fig_correlation_heatmap(df):
    """
    Full Pearson correlation matrix for all continuous features.

    Strategy (Primary) — pyspark.ml.stat.Correlation
    -------------------------------------------------
    1. Filter to available numeric columns.
    2. VectorAssembler → dense feature vector.
    3. Correlation.corr() → single DenseMatrix (1 Spark job + 1 shuffle).
    4. Extract to numpy array on driver (~784 floats for 28 features).

    Fallback — pairwise df.stat.corr()
    -----------------------------------
    If VectorAssembler OOMs (e.g. too many null-heavy columns), falls
    back to n*(n-1)/2 individual Spark jobs.  Slower but safer.

    Complexity
    ----------
    Primary:  O(N × C) + 1 shuffle.
    Fallback: O(N × C²) total, no shuffles per pair.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    """
    print("\n  [FIG 3] Pearson Correlation Heatmap ...")
    t0 = _time.perf_counter()

    # Filter to columns that actually exist in the DataFrame.
    available_cols = [c for c in CONTINUOUS_FEATURES if c in df.columns]

    if len(available_cols) < 2:
        print(f"    [SKIP] Need >= 2 numeric columns, found {len(available_cols)}.")
        return

    print(f"    Using {len(available_cols)} features for correlation.")

    # Attempt ML-based correlation (fast, 1 job).
    corr_matrix = _try_ml_correlation(df, available_cols)

    if corr_matrix is None:
        # Fallback to pairwise.
        print("    [FALLBACK] Using pairwise stat.corr() ...")
        corr_matrix = _pairwise_correlation(df, available_cols)

    # Plot.
    n = len(available_cols)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.55), max(9, n * 0.5)))
    ax.set_title("Pearson Correlation Matrix — All Continuous Features",
                 fontsize=13, fontweight="bold")

    # Upper-triangle mask.
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    masked = np.ma.array(corr_matrix, mask=mask)

    im = ax.imshow(masked, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_xticklabels(available_cols, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(available_cols, fontsize=7)

    # Annotate cells (skip masked upper triangle).
    for i in range(n):
        for j in range(n):
            if not mask[i, j]:
                val = corr_matrix[i, j]
                text_color = "#000000" if abs(val) < 0.6 else "#ffffff"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=5, color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson r", fontsize=9)

    fig.tight_layout()
    _save_fig(fig, "fig_correlation_heatmap.png")
    print(f"    [{_time.perf_counter() - t0:.1f}s]")


def _try_ml_correlation(df, cols):
    """
    Attempt correlation via pyspark.ml.stat.Correlation.

    Returns numpy array or None on failure.
    """
    try:
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml.stat import Correlation

        # Drop rows with ANY null in the selected columns to avoid
        # VectorAssembler failures.
        df_clean = df.select(cols).dropna()

        assembler = VectorAssembler(
            inputCols=cols, outputCol="features",
            handleInvalid="skip",
        )
        df_vec = assembler.transform(df_clean).select("features")

        print(f"    Computing ML-based correlation ({len(cols)}×{len(cols)}) ...")
        corr_row = Correlation.corr(df_vec, "features", "pearson").head()
        corr_dense = corr_row[0]
        return np.array(corr_dense.toArray())

    except Exception as exc:
        print(f"    [WARN] ML correlation failed: {exc}")
        return None


def _pairwise_correlation(df, cols):
    """
    Compute pairwise Pearson correlations via df.stat.corr().

    Total calls = n*(n-1)/2.  Each is a single-pass aggregate (no shuffle).

    Returns numpy array of shape (n, n).
    """
    n = len(cols)
    corr = np.eye(n)
    total_pairs = n * (n - 1) // 2
    done = 0

    for i in range(n):
        for j in range(i + 1, n):
            try:
                r = df.stat.corr(cols[i], cols[j])
                if r is None or math.isnan(r):
                    r = 0.0
            except Exception:
                r = 0.0
            corr[i, j] = r
            corr[j, i] = r
            done += 1
            if done % 50 == 0:
                print(f"      Correlation pairs: {done}/{total_pairs}")

    return corr


# ── FIGURE 4 : TEMPORAL FLUX TIME-SERIES ──────────────────────────────────

def fig_temporal_flux(df):
    """
    Monthly mean delta_h and LWE time-series with trend overlays.

    Strategy
    --------
    1. GROUP BY month_idx → mean(delta_h), mean(lwe_fused), count(*).
    2. Collect ~84 rows to driver.
    3. Convert month_idx to datetime for plotting.
    4. Dual-axis line chart with linear trend + Antarctic winter shading.

    Shuffle Cost
    ------------
    1 shuffle for GROUP BY.  Output is ~84 rows.

    Scientific Interpretation
    -------------------------
    Multi-year downward trend in delta_h = accelerating ice mass loss.
    Seasonal oscillation = summer melt / winter accumulation cycle.
    LWE co-variation confirms GRACE measures the same mass signal.

    Complexity
    ----------
    O(N) + 1 shuffle.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Must contain: month_idx, delta_h, lwe_fused.
    """
    print("\n  [FIG 4] Temporal Flux Time-Series ...")
    t0 = _time.perf_counter()

    # Step 1: Monthly aggregation on executors.
    df_monthly = (
        df.groupBy("month_idx")
        .agg(
            F.mean("delta_h").alias("mean_delta_h"),
            F.mean("lwe_fused").alias("mean_lwe"),
            F.count("*").alias("obs_count"),
        )
        .orderBy("month_idx")
    )

    # Step 2: Collect (~84 rows).
    rows = df_monthly.collect()
    if not rows:
        print("    [SKIP] No monthly data.")
        return

    print(f"    Collected {len(rows)} monthly bins.")

    # Step 3: Convert to numpy arrays.
    import datetime
    month_indices = np.array([int(r["month_idx"]) for r in rows])
    mean_dh       = np.array([float(r["mean_delta_h"]) if r["mean_delta_h"] else np.nan
                              for r in rows])
    mean_lwe      = np.array([float(r["mean_lwe"]) if r["mean_lwe"] else np.nan
                              for r in rows])

    # month_idx = year * 12 + month  →  datetime.
    years  = month_indices // 12
    months = month_indices % 12 + 1  # 0-indexed month → 1-indexed
    dates  = np.array([datetime.datetime(int(y), int(m), 15)
                       for y, m in zip(years, months)])

    # Step 4: Plot.
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # delta_h on left axis.
    color_dh = "#35b779"
    ax1.plot(dates, mean_dh, color=color_dh, marker="o", markersize=3,
             linewidth=1.2, label="Mean Δh [m]")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Mean delta_h [m]", color=color_dh)
    ax1.tick_params(axis="y", labelcolor=color_dh)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # Linear trend for delta_h.
    valid_dh = ~np.isnan(mean_dh)
    if np.sum(valid_dh) >= 2:
        x_numeric = np.arange(len(dates))[valid_dh].astype(float)
        z = np.polyfit(x_numeric, mean_dh[valid_dh], 1)
        trend_line = np.polyval(z, np.arange(len(dates)).astype(float))
        ax1.plot(dates, trend_line, color="#ff6b6b", linestyle="--",
                 linewidth=1.5, alpha=0.8,
                 label=f"Δh trend: {z[0]:.4f} m/month")

    # LWE on right axis.
    ax2 = ax1.twinx()
    color_lwe = "#ffd93d"
    ax2.plot(dates, mean_lwe, color=color_lwe, marker="s", markersize=2,
             linewidth=1.0, alpha=0.7, label="Mean LWE [m]")
    ax2.set_ylabel("Mean LWE (fused) [m]", color=color_lwe)
    ax2.tick_params(axis="y", labelcolor=color_lwe)

    # Antarctic winter shading (March–August each year).
    unique_years = sorted(set(years))
    for yr in unique_years:
        winter_start = datetime.datetime(int(yr), 3, 1)
        winter_end   = datetime.datetime(int(yr), 8, 31)
        ax1.axvspan(winter_start, winter_end, alpha=0.08,
                    color="#4fc3f7", zorder=0)

    ax1.set_title("Antarctic Ice Mass Temporal Flux — Monthly Aggregates")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate(rotation=30)

    # Combined legend.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               fontsize=8)

    fig.tight_layout()
    _save_fig(fig, "fig_temporal_flux.png")
    print(f"    [{_time.perf_counter() - t0:.1f}s]")


# ═══════════════════════════════════════════════════════════════════════════
# ██  MODULE 3 : PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def assess_missing_values(df):
    """
    Compute per-column null percentages in a single Spark pass.

    Strategy
    --------
    Single df.agg() with count(when(isNull(c), c)) for each column.
    This compiles to a SINGLE HashAggregate: O(N × C), no shuffle.

    Returns
    -------
    dict : {col_name: {"null_count": int, "null_pct": float, "total": int}}

    Complexity
    ----------
    O(N × C) single-pass.  No shuffles.  Only C scalars returned to driver.
    """
    print("\n  [PREPROC 1] Assessing missing values ...")
    t0 = _time.perf_counter()

    columns = df.columns
    row_count = df.count()
    print(f"    Total rows: {row_count:,}")

    if row_count == 0:
        print("    [SKIP] Empty dataset.")
        return {}

    agg_exprs = [
        F.count(F.when(F.col(c).isNull(), F.lit(1))).alias(f"{c}__null")
        for c in columns
    ]
    null_row = df.agg(*agg_exprs).head()

    results = {}
    for c in columns:
        nc = int(null_row[f"{c}__null"]) if null_row[f"{c}__null"] is not None else 0
        results[c] = {
            "null_count": nc,
            "null_pct":   (nc / row_count * 100) if row_count > 0 else 0.0,
            "total":      row_count,
        }

    # Print summary sorted by null %.
    sorted_cols = sorted(results.items(), key=lambda kv: kv[1]["null_pct"],
                         reverse=True)
    print(f"\n    {'Column':<30s}  {'Null %':>8s}  {'Null Count':>14s}")
    print(f"    {'─' * 30}  {'─' * 8}  {'─' * 14}")
    for col_name, info in sorted_cols:
        print(f"    {col_name:<30s}  {info['null_pct']:>7.2f}%  "
              f"{info['null_count']:>14,}")

    print(f"    [{_time.perf_counter() - t0:.1f}s]")
    return results


def clip_outliers(df):
    """
    Clip extreme outliers in heavy-tailed features to [P01, P99].

    For each column in CLIP_FEATURES:
      1. Compute approxQuantile at 0.01 and 0.99 (O(N), no shuffle).
      2. Clip via F.when().

    Why This Matters
    ----------------
    Without clipping, satellite artifacts in delta_h (±100m) and extreme
    LWE values dominate variance and skew StandardScaler output.  Clipping
    to [P01, P99] preserves 98% of data while removing artifacts.

    Parameters
    ----------
    df : pyspark.sql.DataFrame

    Returns
    -------
    pyspark.sql.DataFrame with clipped columns.

    Complexity
    ----------
    O(N × len(CLIP_FEATURES)) total.  No shuffles.
    """
    print("\n  [PREPROC 2] Clipping outliers ...")
    t0 = _time.perf_counter()

    available = [c for c in CLIP_FEATURES if c in df.columns]
    if not available:
        print("    [SKIP] No clip features found in schema.")
        return df

    for col_name in available:
        try:
            bounds = df.stat.approxQuantile(col_name, [0.01, 0.99], 0.001)
            if len(bounds) < 2 or bounds[0] is None or bounds[1] is None:
                print(f"    [SKIP] {col_name}: quantile returned nulls.")
                continue

            p01, p99 = bounds[0], bounds[1]
            print(f"    Clipping {col_name}: [{p01:.4f}, {p99:.4f}]")

            df = df.withColumn(
                col_name,
                F.when(F.col(col_name) < p01, F.lit(p01))
                .when(F.col(col_name) > p99, F.lit(p99))
                .otherwise(F.col(col_name))
            )
        except Exception as exc:
            print(f"    [WARN] Clip failed for {col_name}: {exc}")

    print(f"    [{_time.perf_counter() - t0:.1f}s]")
    return df


def impute_median_by_mascon(df):
    """
    Median imputation for static geometry features, grouped by mascon_id.

    Strategy
    --------
    For each feature in IMPUTE_FEATURES:
      1. Compute per-mascon approximate median via percentile_approx(col, 0.5)
         over Window.partitionBy("mascon_id").
      2. Use F.coalesce(original_col, median_col) to fill nulls.
      3. Drop the temporary median column.

    Why Group by mascon_id?
    -----------------------
    Bedmap3 geometry (bed, thickness, surface) varies spatially.  A global
    median would introduce non-physical values in coastal vs. interior
    regions.  Mascon-level medians preserve local spatial consistency.

    Shuffle Cost
    ------------
    1 shuffle for the Window partitionBy("mascon_id").

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Must contain: mascon_id + IMPUTE_FEATURES.

    Returns
    -------
    pyspark.sql.DataFrame with nulls filled.

    Complexity
    ----------
    O(N × len(IMPUTE_FEATURES)) + 1 shuffle.
    """
    print("\n  [PREPROC 3] Median imputation by mascon_id ...")
    t0 = _time.perf_counter()

    if "mascon_id" not in df.columns:
        print("    [SKIP] mascon_id not in schema — cannot group.")
        return df

    available = [c for c in IMPUTE_FEATURES if c in df.columns]
    if not available:
        print("    [SKIP] No impute features found in schema.")
        return df

    # Window for per-mascon aggregation (unbounded = full mascon partition).
    w_mascon = Window.partitionBy("mascon_id")

    for col_name in available:
        median_col = f"__{col_name}_mascon_median"
        df = df.withColumn(
            median_col,
            F.percentile_approx(F.col(col_name), 0.5).over(w_mascon)
        )
        df = df.withColumn(
            col_name,
            F.coalesce(F.col(col_name), F.col(median_col))
        )
        df = df.drop(median_col)
        print(f"    Imputed: {col_name}")

    print(f"    [{_time.perf_counter() - t0:.1f}s]")
    return df


def engineer_features(df):
    """
    Create derived features for ML.

    New Features
    ------------
    1. bed_to_surface_ratio = bed / surface  (with div-by-zero guard)
       Physical meaning: ratio of bedrock depth to surface elevation.
       Values near 0 = deep bed under tall ice. Values > 1 = exposed bedrock.

    2. sin_month, cos_month = cyclical encoding of month_idx
       Captures seasonal signal without ordinal discontinuity (Dec→Jan).

    3. ice_thickness_anomaly = thickness - avg(thickness) per mascon
       Highlights pixels that are anomalously thick/thin within their mascon.

    Shuffle Cost
    ------------
    1 shuffle for the Window partitionBy in ice_thickness_anomaly.
    (Can piggyback on the same mascon shuffle from imputation if cached.)

    Parameters
    ----------
    df : pyspark.sql.DataFrame

    Returns
    -------
    pyspark.sql.DataFrame with new columns appended.
    """
    print("\n  [PREPROC 4] Feature engineering ...")
    t0 = _time.perf_counter()

    # 1. bed_to_surface_ratio (guard: surface == 0 → null).
    if "bed" in df.columns and "surface" in df.columns:
        df = df.withColumn(
            "bed_to_surface_ratio",
            F.when(F.abs(F.col("surface")) > 1e-6,
                   F.col("bed") / F.col("surface"))
            .otherwise(F.lit(None).cast(DoubleType()))
        )
        print("    Created: bed_to_surface_ratio")

    # 2. Seasonal cyclical encoding.
    if "month_idx" in df.columns:
        # month_idx = year*12 + month(0-indexed)
        # Extract month-of-year (0..11), encode as sin/cos.
        month_of_year = (F.col("month_idx") % 12).cast(DoubleType())
        df = df.withColumn(
            "sin_month",
            F.sin(2.0 * math.pi * month_of_year / 12.0)
        )
        df = df.withColumn(
            "cos_month",
            F.cos(2.0 * math.pi * month_of_year / 12.0)
        )
        print("    Created: sin_month, cos_month")

    # 3. Ice thickness anomaly per mascon.
    if "thickness" in df.columns and "mascon_id" in df.columns:
        w_mascon = Window.partitionBy("mascon_id")
        df = df.withColumn(
            "ice_thickness_anomaly",
            F.col("thickness") - F.avg("thickness").over(w_mascon)
        )
        print("    Created: ice_thickness_anomaly")

    print(f"    [{_time.perf_counter() - t0:.1f}s]")
    return df


def target_encode_mascon(df):
    """
    Target-encode mascon_id using mean delta_h per mascon.

    Strategy
    --------
    1. GROUP BY mascon_id → mean(delta_h) as mascon_encoded.
    2. Join back to main DataFrame (broadcast if small enough).

    Why Target Encoding?
    --------------------
    mascon_id is a high-cardinality categorical (~300+ unique values).
    One-hot encoding would add ~300 sparse columns.  Target encoding
    replaces the categorical with a single continuous variable that
    captures the spatial bias in ice mass change.

    Shuffle Cost
    ------------
    1 shuffle for GROUP BY.  The join is a broadcast join if the
    mascon aggregate fits in spark.sql.autoBroadcastJoinThreshold (100MB).

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Must contain: mascon_id, delta_h.

    Returns
    -------
    pyspark.sql.DataFrame with mascon_encoded column.
    """
    print("\n  [PREPROC 5] Target encoding mascon_id ...")
    t0 = _time.perf_counter()

    if "mascon_id" not in df.columns or "delta_h" not in df.columns:
        print("    [SKIP] Required columns missing.")
        return df

    mascon_means = (
        df.groupBy("mascon_id")
        .agg(F.mean("delta_h").alias("mascon_encoded"))
    )

    n_mascons = mascon_means.count()
    print(f"    Unique mascons: {n_mascons:,}")

    # Join: Spark auto-broadcasts if small enough (100MB threshold).
    df = df.join(mascon_means, on="mascon_id", how="left")
    print(f"    Created: mascon_encoded")
    print(f"    [{_time.perf_counter() - t0:.1f}s]")
    return df


def build_preprocessing_pipeline(df):
    """
    Build and fit a PySpark ML Pipeline with VectorAssembler + StandardScaler.

    Pipeline Stages
    ---------------
    1. VectorAssembler: Combines all continuous features + engineered features
       into a single 'features' vector column.
    2. StandardScaler(withMean=True, withStd=True): Zero-mean, unit-variance
       normalisation.  Essential for distance-based and gradient-based models.

    Why StandardScaler (not RobustScaler)?
    --------------------------------------
    PySpark ML's StandardScaler is well-tested at billion-row scale.
    Outliers are already clipped (PREPROC 2), so the mean/std are robust.
    RobustScaler is not natively available in PySpark ML.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Must have all continuous features present (after imputation/clipping).

    Returns
    -------
    (pipeline_model, df_transformed)
        The fitted PipelineModel and the transformed DataFrame.

    Complexity
    ----------
    O(N × C) for VectorAssembler pass + O(N × C) for StandardScaler.
    No additional shuffles (both are narrow transforms).
    """
    print("\n  [PREPROC 6] Building ML Pipeline (VectorAssembler + StandardScaler) ...")
    t0 = _time.perf_counter()

    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from pyspark.ml import Pipeline

    # Determine which features are available.
    engineered = ["bed_to_surface_ratio", "sin_month", "cos_month",
                  "ice_thickness_anomaly", "mascon_encoded"]
    all_features = CONTINUOUS_FEATURES + engineered
    available = [c for c in all_features if c in df.columns]

    if not available:
        print("    [SKIP] No features available for pipeline.")
        return None, df

    print(f"    Assembling {len(available)} features ...")

    assembler = VectorAssembler(
        inputCols=available,
        outputCol="raw_features",
        handleInvalid="skip",
    )
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="scaled_features",
        withMean=True,
        withStd=True,
    )

    pipeline = Pipeline(stages=[assembler, scaler])

    print("    Fitting pipeline ...")
    pipeline_model = pipeline.fit(df)
    df_transformed = pipeline_model.transform(df)

    print(f"    Pipeline fitted.  Output has column: 'scaled_features'")
    print(f"    [{_time.perf_counter() - t0:.1f}s]")
    return pipeline_model, df_transformed


def write_preprocessing_report(null_info: dict):
    """
    Write a human-readable preprocessing report to disk.

    Parameters
    ----------
    null_info : dict
        Output of assess_missing_values().
    """
    report_path = os.path.join(OUTPUT_DIR, "preprocessing_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 72 + "\n")
        f.write("  PREPROCESSING REPORT — Antarctica Sparse Features\n")
        f.write("=" * 72 + "\n\n")

        f.write("1. MISSING VALUES\n")
        f.write(f"   {'Column':<30s}  {'Null %':>8s}  {'Null Count':>14s}\n")
        f.write(f"   {'─' * 30}  {'─' * 8}  {'─' * 14}\n")
        for col_name, info in sorted(null_info.items(),
                                     key=lambda kv: kv[1]["null_pct"],
                                     reverse=True):
            f.write(f"   {col_name:<30s}  {info['null_pct']:>7.2f}%  "
                    f"{info['null_count']:>14,}\n")

        f.write(f"\n2. OUTLIER CLIPPING\n")
        f.write(f"   Features clipped to [P01, P99]: {CLIP_FEATURES}\n")

        f.write(f"\n3. IMPUTATION\n")
        f.write(f"   Median by mascon_id: {IMPUTE_FEATURES}\n")

        f.write(f"\n4. FEATURE ENGINEERING\n")
        f.write(f"   bed_to_surface_ratio = bed / surface\n")
        f.write(f"   sin_month, cos_month = cyclical month encoding\n")
        f.write(f"   ice_thickness_anomaly = thickness - mascon_avg\n")
        f.write(f"   mascon_encoded = target encoding (mean delta_h)\n")

        f.write(f"\n5. SCALING\n")
        f.write(f"   StandardScaler(withMean=True, withStd=True)\n")
        f.write(f"   Applied to all continuous + engineered features.\n")

    print(f"    → Report saved: {report_path}")


# ═══════════════════════════════════════════════════════════════════════════
# ██  MODULE 4 : ENTRYPOINT & ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    """
    Orchestrate the full HPC EDA + Preprocessing pipeline.

    Pipeline
    --------
    Phase A : Build HPC SparkSession.
    Phase B : Load fused dataset.
    Phase C : Visual EDA (4 aggregated figures).
    Phase D : Preprocessing pipeline (assess → clip → impute → engineer →
              encode → scale).
    Phase E : Write report + teardown.

    Returns
    -------
    int
        Exit code: 0 = success, 1 = partial failure, 2 = fatal error.
    """
    wall_start = _time.perf_counter()
    n_failures = 0

    # ── Initialise ─────────────────────────────────────────────────────
    _apply_dark_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n  Output directory: {os.path.abspath(OUTPUT_DIR)}")

    # ── Phase A : SparkSession ─────────────────────────────────────────
    try:
        spark = build_spark_session()
    except Exception as exc:
        print(f"[FATAL] SparkSession failed: {exc}")
        return 2

    # ── Phase B : Load Dataset ─────────────────────────────────────────
    try:
        df = load_fused_dataset(spark)
        row_count = df.count()
        print(f"  Dataset loaded: {row_count:,} rows × {len(df.columns)} cols")
    except Exception as exc:
        print(f"[FATAL] Dataset load failed: {exc}")
        spark.stop()
        return 2

    if row_count == 0:
        print("[FATAL] Dataset is empty.")
        spark.stop()
        return 2

    # ── Phase C : Visual EDA ───────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  PHASE C : Visual EDA (4 Aggregated Figures)")
    print(f"{'═' * 72}")

    for fig_fn, fig_name in [
        (fig_geospatial_density,  "Geospatial Density"),
        (fig_delta_h_distribution, "delta_h Distribution"),
        (fig_correlation_heatmap,  "Correlation Heatmap"),
        (fig_temporal_flux,        "Temporal Flux"),
    ]:
        try:
            fig_fn(df)
        except Exception as exc:
            print(f"  [ERROR] {fig_name} failed: {exc}")
            traceback.print_exc()
            n_failures += 1

    # ── Phase D : Preprocessing Pipeline ───────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  PHASE D : Preprocessing Pipeline")
    print(f"{'═' * 72}")

    null_info = {}
    try:
        null_info = assess_missing_values(df)
    except Exception as exc:
        print(f"  [ERROR] Missing value assessment failed: {exc}")
        n_failures += 1

    try:
        df = clip_outliers(df)
    except Exception as exc:
        print(f"  [ERROR] Outlier clipping failed: {exc}")
        n_failures += 1

    try:
        df = impute_median_by_mascon(df)
    except Exception as exc:
        print(f"  [ERROR] Imputation failed: {exc}")
        n_failures += 1

    try:
        df = engineer_features(df)
    except Exception as exc:
        print(f"  [ERROR] Feature engineering failed: {exc}")
        n_failures += 1

    try:
        df = target_encode_mascon(df)
    except Exception as exc:
        print(f"  [ERROR] Target encoding failed: {exc}")
        n_failures += 1

    try:
        pipeline_model, df_transformed = build_preprocessing_pipeline(df)
        if pipeline_model is not None:
            print("  ✓ Pipeline fitted successfully.")
    except Exception as exc:
        print(f"  [ERROR] ML Pipeline failed: {exc}")
        n_failures += 1

    # ── Phase E : Report & Teardown ────────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  PHASE E : Report & Teardown")
    print(f"{'═' * 72}")

    if null_info:
        write_preprocessing_report(null_info)

    total_wall = _time.perf_counter() - wall_start
    spark.stop()

    if n_failures > 0:
        print(f"\n[WARNING] {n_failures} phase(s) encountered errors.")

    print(f"\n  ┌──────────────────────────────────────────────────────────┐")
    print(f"  │  ✓ REVAMPED EDA + PREPROCESSING PIPELINE COMPLETE       │")
    print(f"  │    Total time: {total_wall:.1f}s  ({total_wall / 60:.1f} min){' ':>18s}│")
    print(f"  │    Figures:  {os.path.abspath(OUTPUT_DIR):<42s}│")
    print(f"  │    Failures: {n_failures:<44d}│")
    print(f"  └──────────────────────────────────────────────────────────┘")

    return 1 if n_failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
