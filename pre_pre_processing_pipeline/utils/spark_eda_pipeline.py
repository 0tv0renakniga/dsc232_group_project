"""
spark_eda_pipeline.py

-------------------------------------------------------------------------------
COMPUTATIONAL GLACIOLOGY : UNIFIED HPC EDA + VISUALISATION + SAMPLING PIPELINE
-------------------------------------------------------------------------------
DATE:   2026-02-19
STATUS: PRODUCTION

Performs iterative, per-dataset EDA on BOTH individual and fused Parquet
datasets, generating descriptive statistics, publication-quality
visualisations, and representative sample extracts to disk.

Target Datasets
---------------
  indiv_data/
    1. bedmap3_static.parquet       (2D static topography + mascon + ocean 2D)
    2. grace.parquet                (mascon mass anomaly time series)
    3. icesat2_dynamic.parquet      (3D elevation-change kinematics, multi-step)
    4. ocean_dynamic.parquet        (3D ocean T, S, T_f, T*)

  fused_data/
    5. antarctica_sparse_features.parquet  (all sources fused on master grid)

HPC Runtime Context
-------------------
  Cluster:           SDSC Expanse (or equivalent SLURM cluster)
  Resource Template: 32 cores / 128 GB RAM (shared partition, fast scheduling)
  Executor Model:    5 cores per executor (HDFS/Spark best-practice)
                     -> 6 executor instances
                     -> ~19 GB per executor
  Driver Memory:     10 GB (handles aggregated plotting payloads safely)

Shuffle Minimisation
--------------------
  * .summary() → Catalyst single-pass HashAggregates (no shuffle).
  * spark.sql.shuffle.partitions = 2 x TOTAL_CORES (narrow default).
  * AQE enabled → auto-coalesce of post-shuffle partitions.
  * coalesce preferred over repartition wherever applicable.
  * No .collect() / .toPandas() on raw DataFrames : HPC MANDATORY.

Figure Catalogue
----------------
  Per-dataset:
    fig_03_histograms_<name>.png   Distribution histograms
    fig_04_correlation_<name>.png  Correlation heatmap

  Cross-dataset:
    fig_01_dataset_overview.png    Row counts + column counts
    fig_02_data_completeness.png   Per-column non-null % heatmap
    fig_05_physical_ranges.png     Min/Max/Mean +/- StdDev range bars
    fig_06_null_structure.png      Per-column null counts

  Output dir: data/eda_plots/

Sample Catalogue
----------------
  data/sample_data/<dataset_name>_sample.parquet

Usage
-----
  spark-submit --master local[*] utils/spark_eda_pipeline.py
  spark-submit --master yarn     utils/spark_eda_pipeline.py

-------------------------------------------------------------------------------
"""

import os
import sys
import math
import time as _time
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")  # headless backend : MUST precede pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.types import NumericType
from pyspark.sql import functions as F


# ═══════════════════════════════════════════════════════════════════════════
# ██  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# --- HPC Resource Allocation ---
# Adjust these two constants to match your SLURM job allocation.
# Everything else is derived programmatically.
# 32c / 128G fits comfortably in Expanse's shared partition and schedules
# within minutes.  For dedicated-node runs, scale up to 128c / 256G.
TOTAL_CORES      = 32     # --ntasks or total cores in the allocation
TOTAL_MEMORY_GB  = 128    # Total RAM in GB across the allocation

# --- Executor Geometry ---
# 5 cores per executor is the Spark/HDFS best-practice for maximising
# HDFS throughput without overwhelming the OS thread scheduler.
EXECUTOR_CORES = 5

# Reserve 1 core for the driver process.
EXECUTOR_INSTANCES = max((TOTAL_CORES - 1) // EXECUTOR_CORES, 1)  # -> 6

# --- Driver Memory ---
# Must be large enough to hold aggregated plotting payloads
# (bin arrays, correlation matrices, completeness vectors) without OOM.
# Raw data is NEVER collected to the driver.
DRIVER_MEMORY_GB = 10

# --- Executor Memory ---
# Divide remaining memory evenly across executors.
EXECUTOR_MEMORY_GB = max(
    math.floor((TOTAL_MEMORY_GB - DRIVER_MEMORY_GB) / EXECUTOR_INSTANCES),
    1,  # safety floor: at least 1 GB per executor
)

# --- Shuffle Tuning ---
# 2 x total cores is a conservative default.  AQE coalesces further.
SHUFFLE_PARTITIONS = 2 * TOTAL_CORES

# --- Data Paths ---
DATA_DIRS = [
    os.path.join("data", "indiv_data"),
    os.path.join("data", "fused_data"),
]
OUTPUT_DIR     = os.path.join("data", "eda_plots")
SAMPLE_DIR     = os.path.join("data", "sample_data")

# --- Histogram / Plot Config ---
HISTOGRAM_BINS = 50
FIG_DPI        = 150

# --- Sampling Config ---
# Target sample size in rows.  The fractional sampler will dynamically
# adjust the sampling fraction so that datasets of any size produce
# a sample in this range.  A 1.2x oversample factor compensates for
# Bernoulli variance (Spark's df.sample uses per-row coin flips, so
# the actual count is stochastic around fraction x N).
TARGET_SAMPLE_ROWS = 250_000
OVERSAMPLE_FACTOR  = 1.2

# --- Resume-After-Crash Config ---
# If the kernel dies mid-run, set these to skip already-completed work.
#   RESUME_DATASET_IDX : 0-based index into the discovered dataset list.
#                        Datasets BEFORE this index are skipped entirely.
#                        Set to 0 to process all datasets (default).
#   RESUME_PHASE       : Phase number (1-8) to start from for the FIRST
#                        resumed dataset.  Subsequent datasets always start
#                        from Phase 1.  Set to 1 to run all phases (default).
#
# Example: kernel died during Phase 8 of dataset index 2 (icesat2).
#          Set RESUME_DATASET_IDX = 2, RESUME_PHASE = 8 to re-run only
#          the sampling phase for icesat2, then continue with remaining
#          datasets normally.
#
# NOTE: Phases 1-2 (ingest + row count) always run regardless of
#       RESUME_PHASE because later phases depend on df and row_count.
#       These are cheap (Parquet footer reads), so no real time lost.
RESTART_EDA_FROM_SCRATCH = True
if RESTART_EDA_FROM_SCRATCH:
    RESUME_DATASET_IDX = 0
    RESUME_PHASE       = 1
else:
    # Resume from Phase 8 of dataset index 2 (icesat2)
    RESUME_DATASET_IDX = 2
    RESUME_PHASE       = 8


# ── Physical variables to visualise (exclude sentinels/categoricals) ────
# Maps dataset basename → list of interesting continuous columns.
PHYS_COLUMNS = {
    "bedmap3_static.parquet": [
        "surface", "bed", "thickness", "bed_slope",
        "dist_to_grounding_line", "clamped_depth", "ice_draft",
    ],
    "grace.parquet": [
        "lwe_length",
    ],
    "icesat2_dynamic.parquet": [
        "delta_h", "ice_area", "h_surface_dynamic", "surface_slope",
    ],
    "ocean_dynamic.parquet": [
        "thetao", "so", "T_f", "T_star",
    ],
    # Fused data : union of the most informative physical columns.
    "antarctica_sparse_features.parquet": [
        "surface", "bed", "thickness", "lwe_length",
        "delta_h", "thetao", "so", "T_f", "T_star",
    ],
}

# Short display names for datasets.
DATASET_LABELS = {
    "bedmap3_static.parquet":                "Bedmap3 Static",
    "grace.parquet":                         "GRACE",
    "icesat2_dynamic.parquet":               "ICESat-2 Dynamic",
    "ocean_dynamic.parquet":                 "Ocean Dynamic",
    "antarctica_sparse_features.parquet":    "Fused (Sparse)",
}

# Colour palette per dataset (viridis-derived).
DATASET_COLOURS = {
    "bedmap3_static.parquet":                "#440154",
    "grace.parquet":                         "#31688e",
    "icesat2_dynamic.parquet":               "#35b779",
    "ocean_dynamic.parquet":                 "#fde725",
    "antarctica_sparse_features.parquet":    "#e76f51",
}


# ═══════════════════════════════════════════════════════════════════════════
# ██  SPARK SESSION BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_spark_session() -> SparkSession:
    """
    Construct an HPC-optimised SparkSession for 32 cores / 128 GB.

    Resource Allocation
    -------------------
    driver.memory         = 10 GB
    executor.instances    = 6     (5 cores each, 1 core reserved for driver)
    executor.cores        = 5
    executor.memory       = floor((128 - 10) / 6) = 19 GB

    Shuffle Strategy
    ----------------
    shuffle.partitions    = 64    (2 x 32 cores)
    AQE enabled           → runtime coalesce of post-shuffle partitions.

    Returns
    -------
    SparkSession
        Configured and ready for distributed EDA.

    Raises
    ------
    RuntimeError
        If derived executor memory is less than 1 GB.

    Complexity
    ----------
    O(1) : configuration only, no data scanned.
    """
    if EXECUTOR_MEMORY_GB < 1:
        raise RuntimeError(
            f"Derived executor memory is {EXECUTOR_MEMORY_GB} GB. "
            f"The HPC allocation is too small for {EXECUTOR_INSTANCES} "
            f"executors.  Reduce TOTAL_CORES or increase TOTAL_MEMORY_GB."
        )

    driver_mem = f"{DRIVER_MEMORY_GB}g"
    exec_mem   = f"{EXECUTOR_MEMORY_GB}g"

    print("=" * 72)
    print("  HPC SparkSession Configuration")
    print("=" * 72)
    print(f"  TOTAL_CORES ............. {TOTAL_CORES}")
    print(f"  TOTAL_MEMORY_GB ......... {TOTAL_MEMORY_GB}")
    print(f"  DRIVER_MEMORY ........... {driver_mem}")
    print(f"  EXECUTOR_CORES .......... {EXECUTOR_CORES}")
    print(f"  EXECUTOR_INSTANCES ...... {EXECUTOR_INSTANCES}")
    print(f"  EXECUTOR_MEMORY ......... {exec_mem}")
    print(f"  SHUFFLE_PARTITIONS ...... {SHUFFLE_PARTITIONS}")
    print("=" * 72)

    spark = (
        SparkSession.builder
        .appName("HPC_Antarctic_Unified_EDA_Pipeline")
        .config("spark.driver.memory",            driver_mem)
        .config("spark.executor.instances",        str(EXECUTOR_INSTANCES))
        .config("spark.executor.cores",            str(EXECUTOR_CORES))
        .config("spark.executor.memory",           exec_mem)
        .config("spark.sql.shuffle.partitions",    str(SHUFFLE_PARTITIONS))
        # --- Driver result budget ---
        .config("spark.driver.maxResultSize",      "4g")
        # --- Network stability for large scans ---
        .config("spark.network.timeout",           "1200s")
        # --- Parallel partition discovery for deep dirs ---
        .config("spark.sql.sources.parallelPartitionDiscovery.threshold", "32")
        .config("spark.sql.sources.parallelPartitionDiscovery.parallelism", "64")
        # --- Adaptive Query Execution ---
        .config("spark.sql.adaptive.enabled",                      "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled",   "true")
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128m")
        # --- Parquet pushdown & vectorisation ---
        .config("spark.sql.parquet.filterPushdown",                "true")
        .config("spark.sql.parquet.mergeSchema",                   "false")
        # --- Disk spill safety : use Lustre scratch, NOT /tmp ---
        .config("spark.local.dir",
                os.environ.get("TMPDIR",
                               os.path.join(os.getcwd(), "spark_scratch")))
        .getOrCreate()
    )

    # Suppress noisy INFO-level Spark logs during EDA.
    spark.sparkContext.setLogLevel("WARN")

    # Ensure scratch dir exists.
    scratch = spark.conf.get("spark.local.dir")
    os.makedirs(scratch, exist_ok=True)

    return spark


# ═══════════════════════════════════════════════════════════════════════════
# ██  DATASET DISCOVERY & INGESTION
# ═══════════════════════════════════════════════════════════════════════════

def discover_parquet_datasets(data_dirs: list[str]) -> list[str]:
    """
    Scan multiple base directories for top-level .parquet entries.

    Parameters
    ----------
    data_dirs : list[str]
        List of directories to scan (e.g. ['data/indiv_data', 'data/fused_data']).

    Returns
    -------
    list[str]
        Sorted list of absolute paths to .parquet entries.

    Edge Cases
    ----------
    - Missing directories → logged and skipped (not fatal).
    - Empty directories → silently pass through.
    - Symlinks are followed (os.listdir behaviour).

    Complexity
    ----------
    O(D x N) where D = number of directories, N = entries per directory.
    """
    datasets = []
    for base_dir in data_dirs:
        abs_dir = os.path.abspath(base_dir)
        if not os.path.isdir(abs_dir):
            print(f"  [WARN] Directory not found, skipping: {abs_dir}")
            continue
        for entry in sorted(os.listdir(abs_dir)):
            if entry.endswith(".parquet"):
                datasets.append(os.path.join(abs_dir, entry))

    if not datasets:
        print("[WARNING] No .parquet entries found across any data directory.")

    return datasets


def load_dataset(spark: SparkSession, path: str):
    """
    Read a Parquet dataset with recursiveFileLookup to bypass Hive
    partition discovery.

    Parameters
    ----------
    spark : SparkSession
    path : str
        Absolute path to the .parquet file or directory.

    Returns
    -------
    pyspark.sql.DataFrame

    Notes
    -----
    recursiveFileLookup=true is CRITICAL for:
      - icesat2_dynamic.parquet/ which contains sub-directories
        (step_015.parquet, step_016.parquet, ...)
      - Any Hive-partitioned layout where partition columns (month_idx)
        duplicate data columns → AnalysisException without this flag.
    mergeSchema=true ensures columns from different sub-files are unified.
    """
    return (
        spark.read
        .option("recursiveFileLookup", "true")
        .option("mergeSchema", "true")
        .parquet(path)
    )


# ═══════════════════════════════════════════════════════════════════════════
# ██  STYLE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _apply_dark_style():
    """Apply a dark, modern matplotlib style globally."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         10,
        "axes.titlesize":    13,
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
# ██  DISTRIBUTED METRIC COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_histogram_spark(df, col_name: str, n_bins: int = HISTOGRAM_BINS):
    """
    Compute histogram bin edges and counts using Spark RDD.histogram().

    This is EXECUTOR-side computation.  Only the (edges, counts) arrays
    are returned to the driver: O(n_bins) floats, <10 KB.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    col_name : str
    n_bins : int

    Returns
    -------
    (edges, counts) : (list[float], list[int]) or (None, None) on failure.

    Complexity
    ----------
    O(N) single-pass per-partition aggregate: no shuffle.
    """
    try:
        rdd = df.select(col_name).na.drop().rdd.map(lambda row: float(row[0]))
        if rdd.isEmpty():
            return None, None
        edges, counts = rdd.histogram(n_bins)
        return edges, counts
    except Exception as exc:
        print(f"      [WARN] Histogram failed for '{col_name}': {exc}")
        return None, None


def compute_correlation_matrix_spark(df, cols: list[str]) -> np.ndarray:
    """
    Compute pairwise Pearson correlations using df.stat.corr().

    Each call is a single-pass distributed aggregate.
    Total calls = n*(n-1)/2 (symmetric matrix).
    For n <= 13 columns this is at most 78 Spark jobs: fast.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    cols : list[str]

    Returns
    -------
    np.ndarray of shape (n, n).

    Complexity
    ----------
    O(n² x N) total over all pairs.  Each pair is O(N) single-pass.
    """
    n = len(cols)
    corr_matrix = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            try:
                r = df.stat.corr(cols[i], cols[j])
                if r is None or math.isnan(r):
                    r = 0.0
            except Exception:
                r = 0.0
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r

    return corr_matrix


def compute_range_stats_spark(df, cols: list[str]) -> dict:
    """
    Compute min, max, mean, stddev for each column via a single .agg() call.

    This compiles to a SINGLE-PASS HashAggregate on executors.
    Only one Spark job; O(cols) scalars returned to driver.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    cols : list[str]

    Returns
    -------
    dict : {col: {"min": ..., "max": ..., "mean": ..., "std": ...}}
    """
    agg_exprs = []
    for c in cols:
        agg_exprs.extend([
            F.min(c).alias(f"{c}__min"),
            F.max(c).alias(f"{c}__max"),
            F.mean(c).alias(f"{c}__mean"),
            F.stddev(c).alias(f"{c}__std"),
        ])

    # Single distributed aggregate : 1 Spark job.
    row = df.agg(*agg_exprs).head()
    result = {}
    for c in cols:
        result[c] = {
            "min":  float(row[f"{c}__min"])  if row[f"{c}__min"]  is not None else float("nan"),
            "max":  float(row[f"{c}__max"])  if row[f"{c}__max"]  is not None else float("nan"),
            "mean": float(row[f"{c}__mean"]) if row[f"{c}__mean"] is not None else float("nan"),
            "std":  float(row[f"{c}__std"])  if row[f"{c}__std"]  is not None else float("nan"),
        }
    return result


def collect_dataset_metadata(df, dataset_name: str) -> dict:
    """
    Hardened metadata collection for extreme-scale datasets (1B+ rows).

    Splits work into TWO separate Spark actions to reduce peak driver
    heap pressure:
      1. df.count()         : fast Parquet-footer-optimised count.
      2. df.agg(...).head() : per-column non-null counts.

    If the completeness aggregate fails (OOM on massive nested dirs),
    partial metadata is returned so the script does not abort.

    Returns
    -------
    dict with keys:
      row_count, col_count, columns, completeness, null_counts

    Complexity
    ----------
    - Count: O(N) single-pass (Parquet footer shortcut where available).
    - Completeness: O(N x C) single-pass HashAggregate.
    No shuffles.
    """
    columns = df.columns

    # Step 1/2 : Global row count.
    print(f"      [Step 1/2] Global row count ...")
    row_count = df.count()
    print(f"      → {row_count:,} rows")

    # Step 2/2 : Per-column completeness.
    print(f"      [Step 2/2] Column completeness ...")
    agg_exprs = [
        F.count(F.when(F.col(c).isNotNull(), c)).alias(c)
        for c in columns
    ]

    completeness = {}
    null_counts  = {}

    try:
        non_null_row = df.agg(*agg_exprs).head()
        for c in columns:
            nn = int(non_null_row[c]) if non_null_row[c] is not None else 0
            completeness[c] = (nn / row_count * 100) if row_count > 0 else 0.0
            null_counts[c]  = row_count - nn
    except Exception as exc:
        print(f"      [WARNING] Completeness check failed: {exc}")
        for c in columns:
            completeness[c] = 0.0
            null_counts[c]  = 0

    return {
        "row_count":    row_count,
        "col_count":    len(columns),
        "columns":      columns,
        "completeness": completeness,
        "null_counts":  null_counts,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ██  PER-DATASET VISUALISATION : HISTOGRAMS
# ═══════════════════════════════════════════════════════════════════════════

def fig_histograms_for_dataset(
    df,
    dataset_name: str,
    phys_cols: list[str],
    colour: str,
):
    """
    Create a multi-panel histogram figure for one dataset.

    All histograms are computed via Spark RDD.histogram() (executor-side).
    Only O(n_bins) floats per column transfer to the driver.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    dataset_name : str
    phys_cols : list[str]
        Physical columns to histogram.
    colour : str
        Hex colour for bars.
    """
    available = set(df.columns)
    cols_to_plot = [c for c in phys_cols if c in available]

    if not cols_to_plot:
        print(f"      [SKIP] No plottable columns for {dataset_name}")
        return

    n = len(cols_to_plot)
    ncols_grid = min(n, 3)
    nrows_grid = math.ceil(n / ncols_grid)

    fig, axes = plt.subplots(
        nrows_grid, ncols_grid,
        figsize=(5 * ncols_grid, 4 * nrows_grid),
    )
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    label = DATASET_LABELS.get(dataset_name, dataset_name)
    fig.suptitle(f"Distributions : {label}", fontsize=15,
                 fontweight="bold", y=1.02)

    for idx, col_name in enumerate(cols_to_plot):
        ax = axes[idx]
        print(f"      Computing histogram: {col_name} ...")
        edges, counts = compute_histogram_spark(df, col_name, HISTOGRAM_BINS)

        if edges is None:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="#888888")
            ax.set_title(col_name)
            continue

        centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
        widths  = [edges[i + 1] - edges[i] for i in range(len(counts))]

        ax.bar(centres, counts, width=widths, color=colour,
               edgecolor="#ffffff22", alpha=0.85)
        ax.set_title(col_name, fontsize=11)
        ax.set_ylabel("Count")
        ax.yaxis.set_major_formatter(mticker.EngFormatter())
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Hide unused axes.
    for idx in range(len(cols_to_plot), len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    safe_name = dataset_name.replace(".parquet", "")
    _save_fig(fig, f"fig_03_histograms_{safe_name}.png")


# ═══════════════════════════════════════════════════════════════════════════
# ██  PER-DATASET VISUALISATION : CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════════════════

def fig_correlation_for_dataset(
    df,
    dataset_name: str,
    phys_cols: list[str],
):
    """
    Render a correlation heatmap for one dataset's physical variables.

    Uses df.stat.corr(): executor-side computation.
    Only O(n²) scalars transfer to the driver.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    dataset_name : str
    phys_cols : list[str]
    """
    available = set(df.columns)
    cols_to_use = [c for c in phys_cols if c in available]

    if len(cols_to_use) < 2:
        print(f"      [SKIP] Need >= 2 numeric columns for correlation "
              f"({dataset_name})")
        return

    print(f"      Computing {len(cols_to_use)}x{len(cols_to_use)} "
          f"correlation matrix ...")
    corr = compute_correlation_matrix_spark(df, cols_to_use)

    fig, ax = plt.subplots(figsize=(max(7, len(cols_to_use) * 0.9),
                                    max(6, len(cols_to_use) * 0.8)))
    label = DATASET_LABELS.get(dataset_name, dataset_name)
    ax.set_title(f"Correlation Matrix : {label}", fontsize=13,
                 fontweight="bold")

    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")

    ax.set_xticks(range(len(cols_to_use)))
    ax.set_xticklabels(cols_to_use, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(cols_to_use)))
    ax.set_yticklabels(cols_to_use, fontsize=9)

    for i in range(len(cols_to_use)):
        for j in range(len(cols_to_use)):
            val = corr[i, j]
            text_color = "#000000" if abs(val) < 0.6 else "#ffffff"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson r", fontsize=9)

    fig.tight_layout()
    safe_name = dataset_name.replace(".parquet", "")
    _save_fig(fig, f"fig_04_correlation_{safe_name}.png")


# ═══════════════════════════════════════════════════════════════════════════
# ██  CROSS-DATASET VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════

def fig_dataset_overview(dataset_meta: dict[str, dict]):
    """
    Bar chart of row counts and column counts per dataset.

    Parameters
    ----------
    dataset_meta : dict
        {dataset_name: {"row_count": int, "col_count": int, ...}}
    """
    names  = list(dataset_meta.keys())
    labels = [DATASET_LABELS.get(n, n) for n in names]
    rows   = [dataset_meta[n]["row_count"] for n in names]
    cols   = [dataset_meta[n]["col_count"] for n in names]
    colors = [DATASET_COLOURS.get(n, "#888888") for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dataset Overview", fontsize=16, fontweight="bold", y=1.02)

    # Row counts (log scale due to ICESat-2 dominance).
    bars1 = ax1.barh(labels, rows, color=colors, edgecolor="#ffffff33")
    ax1.set_xscale("log")
    ax1.set_xlabel("Row Count (log scale)")
    ax1.set_title("Rows per Dataset")
    ax1.grid(axis="x", linestyle="--", alpha=0.3)
    for bar, val in zip(bars1, rows):
        ax1.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height() / 2,
                 f"{val:,.0f}", va="center", fontsize=9, color="#e0e0e0")

    # Column counts.
    bars2 = ax2.barh(labels, cols, color=colors, edgecolor="#ffffff33")
    ax2.set_xlabel("Column Count")
    ax2.set_title("Columns per Dataset")
    ax2.grid(axis="x", linestyle="--", alpha=0.3)
    for bar, val in zip(bars2, cols):
        ax2.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                 str(val), va="center", fontsize=9, color="#e0e0e0")

    fig.tight_layout()
    _save_fig(fig, "fig_01_dataset_overview.png")


def fig_data_completeness(dataset_meta: dict[str, dict]):
    """
    Heatmap showing % non-null per column per dataset.

    All computation was done on executors during metadata collection.
    Only the small completeness percentages are used here.
    """
    # Build a union of all column names across datasets.
    all_cols_ordered = []
    seen = set()
    for name in dataset_meta:
        for c in dataset_meta[name]["columns"]:
            if c not in seen:
                all_cols_ordered.append(c)
                seen.add(c)

    names  = list(dataset_meta.keys())
    labels = [DATASET_LABELS.get(n, n) for n in names]

    matrix = np.full((len(names), len(all_cols_ordered)), np.nan)
    for i, name in enumerate(names):
        completeness = dataset_meta[name].get("completeness", {})
        for j, col_name in enumerate(all_cols_ordered):
            if col_name in completeness:
                matrix[i, j] = completeness[col_name]

    fig, ax = plt.subplots(figsize=(max(14, len(all_cols_ordered) * 0.9), 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)

    ax.set_xticks(range(len(all_cols_ordered)))
    ax.set_xticklabels(all_cols_ordered, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title("Data Completeness (% Non-Null)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=7, color="#888888")
            else:
                text_color = "#000000" if val > 50 else "#ffffff"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=7, fontweight="bold", color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("% Non-Null", fontsize=9)

    fig.tight_layout()
    _save_fig(fig, "fig_02_data_completeness.png")


def fig_physical_ranges(all_range_stats: dict[str, dict]):
    """
    Horizontal range bar chart showing [min, max] with mean +/- std
    for each physical variable, grouped by dataset.

    Parameters
    ----------
    all_range_stats : dict
        {dataset_name: {col: {"min", "max", "mean", "std"}}}
    """
    entries = []
    for ds_name, stats in all_range_stats.items():
        colour = DATASET_COLOURS.get(ds_name, "#888888")
        label  = DATASET_LABELS.get(ds_name, ds_name)
        for col_name, vals in stats.items():
            entries.append({
                "label":   f"{col_name}\n({label})",
                "min":     vals["min"],
                "max":     vals["max"],
                "mean":    vals["mean"],
                "std":     vals["std"],
                "colour":  colour,
            })

    if not entries:
        print("    [SKIP] No range stats to plot.")
        return

    n = len(entries)
    fig, ax = plt.subplots(figsize=(14, max(6, n * 0.45)))
    ax.set_title("Physical Variable Ranges (Min / Max / Mean ± Std)",
                 fontsize=14, fontweight="bold")

    for i, e in enumerate(entries):
        ax.barh(i, e["max"] - e["min"], left=e["min"], height=0.6,
                color=e["colour"], alpha=0.35, edgecolor=e["colour"])
        ax.plot(e["mean"], i, "D", color=e["colour"],
                markersize=6, markeredgecolor="#ffffff", markeredgewidth=0.5)
        if not np.isnan(e["std"]):
            ax.plot(
                [e["mean"] - e["std"], e["mean"] + e["std"]],
                [i, i],
                color=e["colour"], linewidth=2, solid_capstyle="round",
            )

    ax.set_yticks(range(n))
    ax.set_yticklabels([e["label"] for e in entries], fontsize=8)
    ax.set_xlabel("Value")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.invert_yaxis()

    fig.tight_layout()
    _save_fig(fig, "fig_05_physical_ranges.png")


def fig_null_structure(dataset_meta: dict[str, dict]):
    """
    Stacked horizontal bar chart showing null counts per column per dataset.
    """
    all_cols_ordered = []
    seen = set()
    for name in dataset_meta:
        for c in dataset_meta[name]["columns"]:
            if c not in seen:
                all_cols_ordered.append(c)
                seen.add(c)

    names = list(dataset_meta.keys())

    fig, ax = plt.subplots(figsize=(14, max(5, len(all_cols_ordered) * 0.35)))
    ax.set_title("Null Counts per Column", fontsize=14, fontweight="bold")

    y_pos = np.arange(len(all_cols_ordered))
    bar_height = 0.7 / max(len(names), 1)

    for ds_idx, ds_name in enumerate(names):
        nulls_dict = dataset_meta[ds_name].get("null_counts", {})
        label = DATASET_LABELS.get(ds_name, ds_name)
        colour = DATASET_COLOURS.get(ds_name, "#888888")

        null_vals = [nulls_dict.get(c, 0) for c in all_cols_ordered]

        ax.barh(
            y_pos + ds_idx * bar_height,
            null_vals,
            height=bar_height,
            color=colour,
            alpha=0.8,
            label=label,
            edgecolor="#ffffff22",
        )

    ax.set_yticks(y_pos + bar_height * (len(names) - 1) / 2)
    ax.set_yticklabels(all_cols_ordered, fontsize=8)
    ax.set_xlabel("Null Count")
    ax.xaxis.set_major_formatter(mticker.EngFormatter())
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.invert_yaxis()

    fig.tight_layout()
    _save_fig(fig, "fig_06_null_structure.png")


# ═══════════════════════════════════════════════════════════════════════════
# ██  REPRESENTATIVE SAMPLING
# ═══════════════════════════════════════════════════════════════════════════

def generate_sample(df, dataset_name: str, row_count: int):
    """
    Generate a representative fractional sample and write to Parquet.

    The sampling fraction is dynamically calculated so the output is
    approximately TARGET_SAMPLE_ROWS rows, regardless of source size.

    A 1.2x oversample factor compensates for Bernoulli variance :
    Spark's df.sample() uses independent per-row coin flips, so the
    actual count is stochastic around fraction x N.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Source DataFrame (NOT cached: read directly from Parquet).
    dataset_name : str
        Used to construct the output filename.
    row_count : int
        Pre-computed row count of the source.

    Edge Cases
    ----------
    - row_count == 0      → skip (nothing to sample).
    - row_count <= target → fraction capped at 1.0 (take everything).
    - Very large N        → fraction is very small (e.g. 0.0002).

    Writes To
    ---------
    data/sample_data/<dataset_name>_sample.parquet

    Complexity
    ----------
    O(N) single-pass.  No shuffle.  coalesce(1) ensures a single output
    part file: safe because the sample is small (<=500K rows, <100 MB).
    """
    if row_count == 0:
        print(f"      [SKIP] Empty dataset, cannot sample.")
        return

    # Calculate the raw fraction needed.
    raw_fraction = TARGET_SAMPLE_ROWS / row_count

    if raw_fraction >= 1.0:
        # Source is smaller than target : take everything.
        print(f"      Source ({row_count:,} rows) <= target "
              f"({TARGET_SAMPLE_ROWS:,}). Writing full copy.")
        sample_df = df
    else:
        # Apply oversample factor, capped at 1.0.
        adjusted_fraction = min(raw_fraction * OVERSAMPLE_FACTOR, 1.0)
        print(f"      Sampling fraction: {adjusted_fraction:.6f} "
              f"(target ~{TARGET_SAMPLE_ROWS:,} rows from {row_count:,})")
        sample_df = df.sample(
            withReplacement=False,
            fraction=adjusted_fraction,
            seed=42,  # reproducibility
        )

    # coalesce(1) → single output file.  Safe because sample is small.
    safe_name = dataset_name.replace(".parquet", "")
    output_path = os.path.join(SAMPLE_DIR, f"{safe_name}_sample.parquet")

    sample_df.coalesce(1).write.mode("overwrite").parquet(output_path)
    print(f"      → Sample written: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# ██  UNIFIED run_eda_for_dataset()
# ═══════════════════════════════════════════════════════════════════════════

def run_eda_for_dataset(
    spark: SparkSession,
    dataset_path: str,
    dataset_meta_accumulator: OrderedDict,
    all_range_stats_accumulator: OrderedDict,
    start_phase: int = 1,
) -> float:
    """
    Perform COMPLETE EDA on a single Parquet dataset.

    Phased pipeline
    ---------------
    Phase 1 : Ingest + Schema
    Phase 2 : Row count + Column count
    Phase 3 : Numeric summary statistics (.summary())
    Phase 4 : Metadata collection (completeness, null counts)
    Phase 5 : Distributed histograms (executor-side RDD.histogram())
    Phase 6 : Distributed correlation matrix (executor-side stat.corr())
    Phase 7 : Range statistics (min/max/mean/std)
    Phase 8 : Representative sample generation

    Parameters
    ----------
    spark : SparkSession
    dataset_path : str
        Absolute path to the .parquet file or directory.
    dataset_meta_accumulator : OrderedDict
        Mutable dict to accumulate metadata for cross-dataset figures.
    all_range_stats_accumulator : OrderedDict
        Mutable dict to accumulate range stats for physical ranges figure.
    start_phase : int
        Phase number (1-8) to start from.  Phases before this are skipped.
        Phases 1-2 always run regardless (cheap, required by later phases).

    Returns
    -------
    float
        Wall-clock seconds spent on this dataset's EDA.

    Raises
    ------
    Does not raise : prints errors inline and returns elapsed time.
    Designed so one corrupted dataset never aborts the entire run.

    Complexity
    ----------
    - Schema:        O(1) Parquet footer metadata only.
    - Count:         O(N) single-pass via Parquet row-group counts.
    - Summary:       O(N x C_numeric) single-pass HashAggregate.
    - Completeness:  O(N x C) single-pass HashAggregate.
    - Histograms:    O(N x C_phys) single-pass per column.
    - Correlation:   O(N x C_phys²) pairwise.
    - Range Stats:   O(N x C_phys) single .agg().
    - Sampling:      O(N) single-pass Bernoulli.
    No data shuffles are introduced.
    """
    dataset_name = os.path.basename(dataset_path)
    t_dataset_start = _time.perf_counter()

    if start_phase > 1:
        print(f"\n{'━' * 72}")
        print(f"  DATASET:  {dataset_name}  [RESUMING from Phase {start_phase}]")
        print(f"  PATH:     {dataset_path}")
        print(f"{'━' * 72}")
    else:
        print(f"\n{'━' * 72}")
        print(f"  DATASET:  {dataset_name}")
        print(f"  PATH:     {dataset_path}")
        print(f"{'━' * 72}")

    # ── Phase 1 : Ingest + Schema ─────────────────────────────────────
    print(f"\n  [Phase 1] Ingesting + reading schema ...")
    t0 = _time.perf_counter()

    try:
        df = load_dataset(spark, dataset_path)
    except Exception as exc:
        print(f"  [ERROR] Failed to read {dataset_name}: {exc}")
        elapsed = _time.perf_counter() - t_dataset_start
        print(f"  Elapsed: {elapsed:.2f}s (FAILED)\n")
        return elapsed

    num_cols = len(df.columns)
    print(f"  Schema ({num_cols} columns):")
    print(f"  {'Column Name':<30s}  {'Data Type':<20s}")
    print(f"  {'─' * 30}  {'─' * 20}")
    for col_name, col_type in df.dtypes:
        print(f"  {col_name:<30s}  {col_type:<20s}")
    print(f"  [{_time.perf_counter() - t0:.1f}s]")

    # ── Phase 2 : Row count ───────────────────────────────────────────
    print(f"\n  [Phase 2] Counting rows ...")
    t0 = _time.perf_counter()

    try:
        row_count = df.count()
    except Exception as exc:
        print(f"  [ERROR] Row count failed: {exc}")
        elapsed = _time.perf_counter() - t_dataset_start
        print(f"  Elapsed: {elapsed:.2f}s (PARTIAL)\n")
        return elapsed

    print(f"  Total rows:    {row_count:>14,}")
    print(f"  Total columns: {num_cols:>14,}")
    print(f"  [{_time.perf_counter() - t0:.1f}s]")

    if row_count == 0:
        print(f"  [WARNING] Dataset is empty : skipping all analysis.")
        elapsed = _time.perf_counter() - t_dataset_start
        print(f"  Elapsed: {elapsed:.2f}s\n")
        return elapsed

    # ── Phase 3 : Numeric summary statistics ──────────────────────────
    if start_phase <= 3:
        print(f"\n  [Phase 3] Computing summary statistics ...")
        t0 = _time.perf_counter()

        numeric_cols = [
            field.name
            for field in df.schema.fields
            if isinstance(field.dataType, NumericType)
        ]

        if not numeric_cols:
            print(f"  [WARNING] No numeric columns found : skipping summary.")
        else:
            print(f"  Numeric columns ({len(numeric_cols)}): {numeric_cols}")
            df_numeric = df.select(numeric_cols)
            summary_df = df_numeric.summary("count", "min", "max", "mean", "stddev")
            print(f"\n  Summary Statistics for: {dataset_name}")
            summary_df.show(truncate=False, vertical=True)

        print(f"  [{_time.perf_counter() - t0:.1f}s]")
    else:
        print(f"\n  [Phase 3] SKIPPED (resume)")

    # ── Phase 4 : Metadata collection (completeness, null counts) ─────
    if start_phase <= 4:
        print(f"\n  [Phase 4] Collecting metadata ...")
        t0 = _time.perf_counter()
        meta = collect_dataset_metadata(df, dataset_name)
        dataset_meta_accumulator[dataset_name] = meta
        print(f"  [{_time.perf_counter() - t0:.1f}s]")
    else:
        print(f"\n  [Phase 4] SKIPPED (resume)")

    # ── Phase 5 : Distributed histograms ──────────────────────────────
    phys_cols = PHYS_COLUMNS.get(dataset_name, [])
    colour    = DATASET_COLOURS.get(dataset_name, "#888888")

    if start_phase <= 5:
        if phys_cols:
            print(f"\n  [Phase 5] Building histograms ...")
            t0 = _time.perf_counter()
            fig_histograms_for_dataset(df, dataset_name, phys_cols, colour)
            print(f"  [{_time.perf_counter() - t0:.1f}s]")
        else:
            print(f"\n  [Phase 5] No PHYS_COLUMNS configured : skipping histograms.")
    else:
        print(f"\n  [Phase 5] SKIPPED (resume)")

    # ── Phase 6 : Distributed correlation matrix ──────────────────────
    if start_phase <= 6:
        if phys_cols and len(phys_cols) >= 2:
            print(f"\n  [Phase 6] Building correlation heatmap ...")
            t0 = _time.perf_counter()
            fig_correlation_for_dataset(df, dataset_name, phys_cols)
            print(f"  [{_time.perf_counter() - t0:.1f}s]")
        else:
            print(f"\n  [Phase 6] Need >= 2 PHYS_COLUMNS : skipping correlation.")
    else:
        print(f"\n  [Phase 6] SKIPPED (resume)")

    # ── Phase 7 : Range statistics ────────────────────────────────────
    if start_phase <= 7:
        if phys_cols:
            avail_phys = [c for c in phys_cols if c in set(df.columns)]
            if avail_phys:
                print(f"\n  [Phase 7] Computing range stats ...")
                t0 = _time.perf_counter()
                all_range_stats_accumulator[dataset_name] = (
                    compute_range_stats_spark(df, avail_phys)
                )
                print(f"  [{_time.perf_counter() - t0:.1f}s]")
            else:
                print(f"\n  [Phase 7] No available PHYS_COLUMNS in schema : skipping.")
        else:
            print(f"\n  [Phase 7] No PHYS_COLUMNS configured : skipping range stats.")
    else:
        print(f"\n  [Phase 7] SKIPPED (resume)")

    # ── Phase 8 : Representative sample ───────────────────────────────
    print(f"\n  [Phase 8] Generating representative sample ...")
    t0 = _time.perf_counter()
    try:
        generate_sample(df, dataset_name, row_count)
    except Exception as exc:
        print(f"  [ERROR] Sampling failed for {dataset_name}: {exc}")
    print(f"  [{_time.perf_counter() - t0:.1f}s]")

    # ── Done ──────────────────────────────────────────────────────────
    elapsed = _time.perf_counter() - t_dataset_start
    print(f"\n  ✓ {dataset_name} complete : {elapsed:.2f}s")
    return elapsed


# ═══════════════════════════════════════════════════════════════════════════
# ██  PERFORMANCE REPORT
# ═══════════════════════════════════════════════════════════════════════════

def print_performance_report(
    dataset_times: dict[str, float],
    total_wall_seconds: float,
) -> None:
    """
    Print per-dataset timing and instructions for Speedup / Efficiency.

    Parameters
    ----------
    dataset_times : dict[str, float]
        Mapping of dataset name → EDA wall-clock seconds.
    total_wall_seconds : float
        Total wall-clock time for the entire pipeline.
    """
    print(f"\n{'=' * 72}")
    print(f"  PERFORMANCE REPORT")
    print(f"{'=' * 72}")

    print(f"\n  {'Dataset':<40s}  {'Time (s)':>10s}")
    print(f"  {'─' * 40}  {'─' * 10}")
    for name, t in dataset_times.items():
        print(f"  {name:<40s}  {t:>10.2f}")
    print(f"  {'─' * 40}  {'─' * 10}")
    print(f"  {'TOTAL':<40s}  {total_wall_seconds:>10.2f}")

    print(f"\n  Executor configuration:")
    print(f"    Executor instances  = {EXECUTOR_INSTANCES}")
    print(f"    Executor cores      = {EXECUTOR_CORES}")
    print(f"    Executor memory     = {EXECUTOR_MEMORY_GB}g")
    print(f"    Driver memory       = {DRIVER_MEMORY_GB}g")
    print(f"    Tₙ (this run)       = {total_wall_seconds:.2f}s")

    print(f"""
  ┌────────────────────────────────────────────────────────────────────┐
  │  SPEEDUP & EFFICIENCY (requires a baseline run)                   │
  │                                                                   │
  │  1. Run with TOTAL_CORES=6, EXECUTOR_CORES=5 (→ 1 executor)      │
  │     Record the total wall-clock time as T₁.                       │
  │                                                                   │
  │  2. Run with TOTAL_CORES={TOTAL_CORES:<3d} (→ {EXECUTOR_INSTANCES:<2d} executors)              │
  │     Tₙ = {total_wall_seconds:<8.2f}s (this run).                           │
  │                                                                   │
  │  3. Compute:                                                      │
  │       Speedup    = T₁ / Tₙ                                       │
  │       Efficiency = Speedup / n                                    │
  │                  = T₁ / (Tₙ x n)                                  │
  │                                                                   │
  │  Perfect scaling → Speedup = n, Efficiency = 1.0                  │
  └────────────────────────────────────────────────────────────────────┘
""")


# ═══════════════════════════════════════════════════════════════════════════
# ██  ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    """
    Orchestrate the full unified EDA pipeline.

    Pipeline
    --------
    1. Build HPC SparkSession.
    2. Discover datasets across indiv_data/ and fused_data/.
    3. For each dataset, run the 8-phase EDA pipeline.
    4. Generate cross-dataset visualisations.
    5. Print performance report.

    Returns
    -------
    int
        Exit code: 0 = success, 1 = partial failure, 2 = fatal error.
    """
    wall_start = _time.perf_counter()

    # ── Initialise ─────────────────────────────────────────────────────
    _apply_dark_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    print(f"  Output directories:")
    print(f"    Plots:   {os.path.abspath(OUTPUT_DIR)}")
    print(f"    Samples: {os.path.abspath(SAMPLE_DIR)}")

    # ── Build SparkSession ─────────────────────────────────────────────
    try:
        spark = build_spark_session()
    except Exception as exc:
        print(f"[FATAL] Failed to create SparkSession: {exc}")
        return 2

    # ── Discover Datasets ──────────────────────────────────────────────
    try:
        datasets = discover_parquet_datasets(DATA_DIRS)
    except Exception as exc:
        print(f"[FATAL] {exc}")
        spark.stop()
        return 2

    if not datasets:
        print("[FATAL] No .parquet datasets found.  Nothing to analyse.")
        spark.stop()
        return 2

    print(f"\n  Discovered {len(datasets)} dataset(s):")
    for i, ds in enumerate(datasets, 1):
        print(f"    {i}. {os.path.basename(ds)}")

    # ── Phase 1 : Per-Dataset EDA Loop ─────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  PHASE 1 : Per-Dataset EDA Pipeline")
    print(f"{'═' * 72}")

    dataset_meta       = OrderedDict()
    all_range_stats    = OrderedDict()
    dataset_times      = OrderedDict()
    n_failures         = 0

    for ds_idx, dataset_path in enumerate(datasets):
        name = os.path.basename(dataset_path)

        # Resume logic: skip datasets before RESUME_DATASET_IDX.
        if ds_idx < RESUME_DATASET_IDX:
            print(f"\n  [SKIP] {name} (dataset index {ds_idx} < "
                  f"RESUME_DATASET_IDX={RESUME_DATASET_IDX})")
            continue

        # For the first resumed dataset, apply RESUME_PHASE.
        # Subsequent datasets start from Phase 1.
        start_phase = RESUME_PHASE if ds_idx == RESUME_DATASET_IDX else 1

        try:
            elapsed = run_eda_for_dataset(
                spark, dataset_path,
                dataset_meta, all_range_stats,
                start_phase=start_phase,
            )
            dataset_times[name] = elapsed
        except Exception as exc:
            print(f"  [ERROR] Unhandled exception for {name}: {exc}")
            dataset_times[name] = 0.0
            n_failures += 1

    # ── Phase 2 : Cross-Dataset Figures ────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  PHASE 2 : Cross-Dataset Figures")
    print(f"{'═' * 72}")

    if dataset_meta:
        print(f"\n  Building dataset overview ...")
        fig_dataset_overview(dataset_meta)

        print(f"  Building completeness heatmap ...")
        fig_data_completeness(dataset_meta)

        print(f"  Building null structure chart ...")
        fig_null_structure(dataset_meta)

    if all_range_stats:
        print(f"  Building physical ranges chart ...")
        fig_physical_ranges(all_range_stats)

    # ── Performance Report ─────────────────────────────────────────────
    total_wall = _time.perf_counter() - wall_start
    print_performance_report(dataset_times, total_wall)

    # ── Teardown ───────────────────────────────────────────────────────
    spark.stop()

    if n_failures > 0:
        print(f"[WARNING] {n_failures} dataset(s) encountered errors.")
        return 1

    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  ✓ UNIFIED EDA PIPELINE COMPLETE                    │")
    print(f"  │    Total time: {total_wall:.1f}s  ({total_wall / 60:.1f} min){'':>14s}│")
    print(f"  │    Plots dir:  {os.path.abspath(OUTPUT_DIR):<36s}│")
    print(f"  │    Sample dir: {os.path.abspath(SAMPLE_DIR):<36s}│")
    print(f"  └─────────────────────────────────────────────────────┘")

    return 0


if __name__ == "__main__":
    sys.exit(main())
