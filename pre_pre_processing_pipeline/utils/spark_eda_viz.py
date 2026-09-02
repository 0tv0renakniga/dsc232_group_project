"""
spark_eda_viz.py

-------------------------------------------------------------------------------
COMPUTATIONAL GLACIOLOGY : HPC-OPTIMISED EDA VISUALISATIONS
-------------------------------------------------------------------------------
DATE:   2026-02-18
STATUS: PRODUCTION

Generates publication-quality EDA visualisation PNGs from the four individual
flattened Parquet datasets.  ALL heavy aggregation (histograms, correlations,
completeness counts) runs on Spark executors; only small arrays/scalars are
brought to the driver for matplotlib rendering.

Figure Catalogue
----------------
  1. fig_01_dataset_overview.png       Row counts + column counts per dataset
  2. fig_02_data_completeness.png      Per-column non-null percentage heatmap
  3. fig_03_histograms_<dataset>.png   Distribution histograms (per dataset)
  4. fig_04_correlation_<dataset>.png  Correlation heatmap (per dataset)
  5. fig_05_physical_ranges.png        Min/Max/Mean +/- StdDev range bars
  6. fig_06_null_structure.png         Per-column null counts stacked bar

Output
------
  data/eda_plots/*.png   (created if absent)

HPC Safety
----------
  * matplotlib.use('Agg') : no display backend required.
  * No .toPandas() / .collect() on raw DataFrames.
  * Driver budget: 16 GB (extreme-scale for 1.3B-row file indexing).
  * All transferred data is O(bins x cols) << 1 MB.

Usage
-----
  spark-submit --master local[*] utils/spark_eda_viz.py

-------------------------------------------------------------------------------
!!!! NOTE: YOU HAVE TO ADD BELOW TO GET PLOTS TO SHOW !!!!
from IPython.display import Image, display

# Replace with the actual path to your plot
display(Image(filename='data/eda_plots/fig_03_histograms_grace.png'))

AND MAKE SURE YOU HAVE THE FOLLOWING:
# matplotlib HPC env
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), '.matplotlib_cache')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import matplotlib
if 'ipykernel' in sys.modules:
    # Use the inline backend for notebooks
    %matplotlib inline 
else:
    # Use the headless backend for spark-submit / terminal jobs
    matplotlib.use("Agg")
"""


import os
import sys
import math
import time as _time
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")  # headless backend : MUST be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.types import NumericType, TimestampType
from pyspark.sql import functions as F


# =========================================================================
# CONFIGURATION
# =========================================================================

TOTAL_CORES      = 32
TOTAL_MEMORY_GB  = 128
DRIVER_MEMORY_GB = 16    # Extreme-scale : 1.3B rows need ~16 GB for file
                         # index metadata across thousands of step_*.parquet

EXECUTOR_INSTANCES = max(TOTAL_CORES - 1, 1)
EXECUTOR_MEMORY_GB = max(
    math.floor((TOTAL_MEMORY_GB - DRIVER_MEMORY_GB) / EXECUTOR_INSTANCES),
    1,
)
SHUFFLE_PARTITIONS = 2 * TOTAL_CORES

BASE_DATA_DIR = os.path.join("data", "indiv_data")
OUTPUT_DIR    = os.path.join("data", "eda_plots")

HISTOGRAM_BINS = 50
FIG_DPI        = 150


# ── Physical variables to visualise (exclude sentinels/categoricals) ────
# Maps dataset basename -> list of interesting continuous columns.
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
}

# Short display names for datasets.
DATASET_LABELS = {
    "bedmap3_static.parquet":    "Bedmap3 Static",
    "grace.parquet":             "GRACE",
    "icesat2_dynamic.parquet":   "ICESat-2 Dynamic",
    "ocean_dynamic.parquet":     "Ocean Dynamic",
}

# Colour palette per dataset (viridis-derived).
DATASET_COLOURS = {
    "bedmap3_static.parquet":    "#440154",
    "grace.parquet":             "#31688e",
    "icesat2_dynamic.parquet":   "#35b779",
    "ocean_dynamic.parquet":     "#fde725",
}


# =========================================================================
# SPARK SESSION
# =========================================================================

def build_spark_session() -> SparkSession:
    """Build extreme-scale HPC SparkSession for billion-row datasets."""
    spark = (
        SparkSession.builder
        .appName("HPC_Antarctic_Extreme_Scale_Viz")
        .config("spark.driver.memory",            f"{DRIVER_MEMORY_GB}g")
        .config("spark.executor.instances",        str(EXECUTOR_INSTANCES))
        .config("spark.executor.memory",           f"{EXECUTOR_MEMORY_GB}g")
        .config("spark.sql.shuffle.partitions",    str(SHUFFLE_PARTITIONS))
        # --- Metadata & Stability (billion-row wall breaker) ---
        .config("spark.driver.maxResultSize",      "4g")
        .config("spark.network.timeout",           "1200s")
        .config("spark.sql.sources.parallelPartitionDiscovery.threshold", "32")
        .config("spark.sql.sources.parallelPartitionDiscovery.parallelism", "64")
        # --- AQE & Optimisation ---
        .config("spark.sql.adaptive.enabled",                      "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled",   "true")
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128m")
        # --- Parquet ---
        .config("spark.sql.parquet.filterPushdown",                "true")
        .config("spark.sql.parquet.mergeSchema",                   "false")
        # --- Disk spill safety : use Lustre scratch, NOT /tmp ---
        .config("spark.local.dir",
                os.environ.get("TMPDIR",
                               os.path.join(os.getcwd(), "spark_scratch")))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    # Create scratch dir if using fallback path.
    scratch = spark.conf.get("spark.local.dir")
    os.makedirs(scratch, exist_ok=True)
    return spark


# =========================================================================
# DATASET DISCOVERY & INGESTION
# =========================================================================

def discover_datasets(base_dir: str) -> list[str]:
    """Return sorted list of .parquet paths under base_dir."""
    abs_dir = os.path.abspath(base_dir)
    if not os.path.isdir(abs_dir):
        raise FileNotFoundError(f"Data directory not found: {abs_dir}")
    return sorted(
        os.path.join(abs_dir, e)
        for e in os.listdir(abs_dir)
        if e.endswith(".parquet")
    )


def load_dataset(spark: SparkSession, path: str):
    """Read a Parquet dataset with recursiveFileLookup to bypass Hive."""
    return (
        spark.read
        .option("recursiveFileLookup", "true")
        .option("mergeSchema", "true")
        .parquet(path)
    )


# =========================================================================
# STYLE HELPERS
# =========================================================================

def _apply_dark_style():
    """Apply a dark, modern style globally."""
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
    print(f"  -> Saved: {path}")


# =========================================================================
# FIGURE 1 : DATASET OVERVIEW (row counts + column counts)
# =========================================================================

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


# =========================================================================
# FIGURE 2 : DATA COMPLETENESS HEATMAP
# =========================================================================

def fig_data_completeness(dataset_meta: dict[str, dict]):
    """
    Heatmap showing % non-null per column per dataset.

    Computed on the executor side via:
      df.select([count(when(col(c).isNotNull(), c)) for c in cols])
    Only the resulting 1-row DataFrame (one scalar per column) is
    brought to the driver.
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

    # Build the matrix: rows = datasets, cols = all_cols_ordered.
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

    # Annotate each cell.
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


# =========================================================================
# FIGURE 3 : DISTRIBUTION HISTOGRAMS (per dataset)
# =========================================================================

def compute_histogram_spark(df, col_name: str, n_bins: int = 50):
    """
    Compute histogram bin edges and counts using Spark RDD.histogram().

    This is executor-side computation.  Only the (edges, counts) arrays
    are returned to the driver : O(n_bins) floats.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    col_name : str
    n_bins : int

    Returns
    -------
    (edges, counts) : (list[float], list[int]) or (None, None) on failure.
    """
    try:
        rdd = df.select(col_name).na.drop().rdd.map(lambda row: float(row[0]))
        # RDD.histogram(n_bins) returns (list_of_edges, list_of_counts).
        # Edges has n_bins+1 elements, counts has n_bins elements.
        # This runs entirely on executors as a single-pass aggregate.
        if rdd.isEmpty():
            return None, None
        edges, counts = rdd.histogram(n_bins)
        return edges, counts
    except Exception as exc:
        print(f"    [WARN] Histogram failed for '{col_name}': {exc}")
        return None, None


def fig_histograms_for_dataset(
    df,
    dataset_name: str,
    phys_cols: list[str],
    colour: str,
):
    """
    Create a multi-panel histogram figure for one dataset.

    All histograms are computed via Spark RDD.histogram() (executor-side).
    """
    # Filter to columns that actually exist in the DataFrame.
    available = set(df.columns)
    cols_to_plot = [c for c in phys_cols if c in available]

    if not cols_to_plot:
        print(f"    [SKIP] No plottable columns for {dataset_name}")
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
        print(f"    Computing histogram: {col_name} ...")
        edges, counts = compute_histogram_spark(df, col_name, HISTOGRAM_BINS)

        if edges is None:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="#888888")
            ax.set_title(col_name)
            continue

        # Bar centres from edges.
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


# =========================================================================
# FIGURE 4 : CORRELATION HEATMAP (per dataset)
# =========================================================================

def compute_correlation_matrix_spark(df, cols: list[str]):
    """
    Compute pairwise Pearson correlations using df.stat.corr().

    Each call is a single-pass distributed aggregate.
    Total calls = n*(n-1)/2 (symmetric matrix).
    For n <= 13 columns this is at most 78 Spark jobs : fast.

    Returns
    -------
    np.ndarray of shape (n, n).
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


def fig_correlation_for_dataset(
    df,
    dataset_name: str,
    phys_cols: list[str],
    colour: str,
):
    """
    Render a correlation heatmap for one dataset's physical variables.
    """
    available = set(df.columns)
    cols_to_use = [c for c in phys_cols if c in available]

    if len(cols_to_use) < 2:
        print(f"    [SKIP] Need >= 2 numeric columns for correlation "
              f"({dataset_name})")
        return

    print(f"    Computing {len(cols_to_use)}x{len(cols_to_use)} "
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

    # Annotate cells.
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


# =========================================================================
# FIGURE 5 : PHYSICAL RANGE BARS (min / max / mean +/- stddev)
# =========================================================================

def compute_range_stats_spark(df, cols: list[str]):
    """
    Compute min, max, mean, stddev for each column via a single .agg() call.

    This compiles to a single-pass HashAggregate on executors.
    Returns a dict: {col: {"min": ..., "max": ..., "mean": ..., "std": ...}}.
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


def fig_physical_ranges(all_range_stats: dict[str, dict]):
    """
    Horizontal range bar chart showing [min, max] with mean +/- std
    for each physical variable, grouped by dataset.
    """
    # Flatten all vars with their dataset for plotting.
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
        print("  [SKIP] No range stats to plot.")
        return

    n = len(entries)
    fig, ax = plt.subplots(figsize=(14, max(6, n * 0.45)))
    ax.set_title("Physical Variable Ranges (Min / Max / Mean +/- Std)",
                 fontsize=14, fontweight="bold")

    y_positions = list(range(n))

    for i, e in enumerate(entries):
        # Range bar: [min, max].
        ax.barh(i, e["max"] - e["min"], left=e["min"], height=0.6,
                color=e["colour"], alpha=0.35, edgecolor=e["colour"])
        # Mean marker.
        ax.plot(e["mean"], i, "D", color=e["colour"],
                markersize=6, markeredgecolor="#ffffff", markeredgewidth=0.5)
        # Std whiskers.
        if not np.isnan(e["std"]):
            ax.plot(
                [e["mean"] - e["std"], e["mean"] + e["std"]],
                [i, i],
                color=e["colour"], linewidth=2, solid_capstyle="round",
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([e["label"] for e in entries], fontsize=8)
    ax.set_xlabel("Value")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.invert_yaxis()

    fig.tight_layout()
    _save_fig(fig, "fig_05_physical_ranges.png")


# =========================================================================
# FIGURE 6 : NULL STRUCTURE (per-column null counts, stacked by dataset)
# =========================================================================

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

    names  = list(dataset_meta.keys())

    fig, ax = plt.subplots(figsize=(14, max(5, len(all_cols_ordered) * 0.35)))
    ax.set_title("Null Counts per Column", fontsize=14, fontweight="bold")

    y_pos = np.arange(len(all_cols_ordered))
    bar_height = 0.7 / max(len(names), 1)

    for ds_idx, ds_name in enumerate(names):
        nulls_dict = dataset_meta[ds_name].get("null_counts", {})
        label = DATASET_LABELS.get(ds_name, ds_name)
        colour = DATASET_COLOURS.get(ds_name, "#888888")

        null_vals = []
        for c in all_cols_ordered:
            null_vals.append(nulls_dict.get(c, 0))

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


# =========================================================================
# METADATA COLLECTION (executor-side aggregates)
# =========================================================================

def collect_dataset_metadata(df, dataset_name: str) -> dict:
    """
    Hardened metadata collection for extreme-scale datasets (1B+ rows).

    Splits the work into TWO separate Spark actions to reduce peak
    driver heap pressure :
      1. df.count()           : fast Parquet-footer-optimised count.
      2. df.agg(...).head()   : per-column non-null counts.

    If the completeness aggregate fails (OOM on massive nested dirs),
    partial metadata is returned so the script does not abort.

    Returns
    -------
    dict with keys:
      row_count, col_count, columns, completeness, null_counts
    """
    columns = df.columns

    # Step 1/2 : Global row count.
    # Parquet footer optimisation makes this cheap even on 1.3B rows.
    print(f"      [Step 1/2] Global row count ...")
    row_count = df.count()
    print(f"      -> {row_count:,} rows")

    # Step 2/2 : Per-column completeness.
    # Separate action so the driver can GC between the two heavy scans.
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
        # Return partial metadata so downstream figures degrade gracefully.
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


# =========================================================================
# MAIN PIPELINE
# =========================================================================

def main() -> int:
    """
    Orchestrate the full EDA visualisation pipeline.

    Returns
    -------
    int
        Exit code: 0 = success, 1 = partial failure, 2 = fatal error.
    """
    wall_start = _time.perf_counter()
    _apply_dark_style()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"  Output directory: {os.path.abspath(OUTPUT_DIR)}")

    # ── Spark ───────────────────────────────────────────────────────
    try:
        spark = build_spark_session()
    except Exception as exc:
        print(f"[FATAL] SparkSession creation failed: {exc}")
        return 2

    # ── Discover ────────────────────────────────────────────────────
    try:
        dataset_paths = discover_datasets(BASE_DATA_DIR)
    except FileNotFoundError as exc:
        print(f"[FATAL] {exc}")
        spark.stop()
        return 2

    if not dataset_paths:
        print("[FATAL] No .parquet datasets found.")
        spark.stop()
        return 2

    print(f"\n  Discovered {len(dataset_paths)} dataset(s):")
    for i, p in enumerate(dataset_paths, 1):
        print(f"    {i}. {os.path.basename(p)}")

    # ── Phase 1 : Collect metadata and per-dataset figures ──────────
    print(f"\n{'=' * 72}")
    print(f"  PHASE 1 : Metadata + Per-Dataset Figures")
    print(f"{'=' * 72}")

    dataset_meta = OrderedDict()
    all_range_stats = OrderedDict()

    for path in dataset_paths:
        name = os.path.basename(path)
        label = DATASET_LABELS.get(name, name)
        phys_cols = PHYS_COLUMNS.get(name, [])
        colour = DATASET_COLOURS.get(name, "#888888")

        print(f"\n  ── {label} ──")

        try:
            df = load_dataset(spark, path)
        except Exception as exc:
            print(f"    [ERROR] Failed to load: {exc}")
            continue

        # NO .cache() : Parquet on Lustre is the cache.  Caching 1.38B
        # rows overflows RAM -> spills to /tmp -> "No space left on device".

        # Metadata (completeness, null counts).
        print(f"    Collecting metadata ...")
        t0 = _time.perf_counter()
        meta = collect_dataset_metadata(df, name)
        dataset_meta[name] = meta
        print(f"       {meta['row_count']:,} rows x {meta['col_count']} cols "
              f"[{_time.perf_counter() - t0:.1f}s]")

        # Histograms.
        if phys_cols:
            print(f"    Building histograms ...")
            t0 = _time.perf_counter()
            fig_histograms_for_dataset(df, name, phys_cols, colour)
            print(f"       [{_time.perf_counter() - t0:.1f}s]")

        # Correlation.
        if len(phys_cols) >= 2:
            print(f"    Building correlation heatmap ...")
            t0 = _time.perf_counter()
            fig_correlation_for_dataset(df, name, phys_cols, colour)
            print(f"       [{_time.perf_counter() - t0:.1f}s]")

        # Range stats (for the combined physical ranges figure).
        if phys_cols:
            avail = [c for c in phys_cols if c in set(df.columns)]
            if avail:
                print(f"    Computing range stats ...")
                t0 = _time.perf_counter()
                all_range_stats[name] = compute_range_stats_spark(df, avail)
                print(f"       [{_time.perf_counter() - t0:.1f}s]")

    # ── Phase 2 : Cross-dataset figures ─────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  PHASE 2 : Cross-Dataset Figures")
    print(f"{'=' * 72}")

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

    # ── Cleanup ─────────────────────────────────────────────────────
    spark.stop()

    wall_total = _time.perf_counter() - wall_start
    print(f"\n{'=' * 72}")
    print(f"  EDA VISUALISATION COMPLETE")
    print(f"  Total time: {wall_total:.1f}s  ({wall_total / 60:.1f} min)")
    print(f"  Output dir: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'=' * 72}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
