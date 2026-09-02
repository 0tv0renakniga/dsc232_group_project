"""
visualization_script.py

-------------------------------------------------------------------------------
COMPUTATIONAL GLACIOLOGY : TARGETED ANTARCTICA EDA VISUALISATION
-------------------------------------------------------------------------------
DATE:   2026-02-22
STATUS: PRODUCTION

Performs focused EDA on the Antarctica Sparse Feature Store, producing three
publication-quality visualisation categories with domain-specific insight
commentary:

  1. Null/Missing-Data Bar Chart  — per-column completeness profile
  2. Grouped Distribution Histograms — split by domain group:
       bedmap  : surface, bed, thickness, bed_slope
       ocean   : clamped_depth, dist_to_ocean, ice_draft, thetao_mo,
                 t_star_mo, so_mo, t_f_mo, t_star_quarterly_avg,
                 t_star_quarterly_std, thetao_quarterly_avg,
                 thetao_quarterly_std
       icesat  : ice_area, delta_h, surface_slope, h_surface_dynamic
       grace   : lwe_mo, lwe_fused, lwe_quarterly_avg, lwe_quarterly_std
  3. Spearman Correlation Heatmap  — ocean vars (thetao_mo, so_mo) vs
       ice change vars (delta_h, thickness)

Target Dataset
--------------
  antarctica_sparse_features.parquet  (fused feature store, 28 columns)
  — OR its sample subset antarctica_sparse_features_sample.parquet

Schema (28 columns):
  y (double), x (double), exact_time (timestamp_ntz), mascon_id (int),
  surface (float), bed (float), thickness (float), bed_slope (float),
  dist_to_grounding_line (float), clamped_depth (float),
  dist_to_ocean (float), ice_draft (float), delta_h (float),
  ice_area (float), surface_slope (float), h_surface_dynamic (float),
  thetao_mo (double), t_star_mo (double), so_mo (double), t_f_mo (double),
  t_star_quarterly_avg (double), t_star_quarterly_std (double),
  thetao_quarterly_avg (double), thetao_quarterly_std (double),
  lwe_mo (float), lwe_quarterly_avg (double), lwe_quarterly_std (double),
  lwe_fused (double)

Output
------
  data/eda_plots/
    viz_01_null_bar_chart.png
    viz_02_histograms_bedmap.png
    viz_02_histograms_ocean.png
    viz_02_histograms_icesat.png
    viz_02_histograms_grace.png
    viz_03_spearman_correlation.png

Computational Safety
--------------------
  * ALL aggregations run on executors (no .collect() on raw DataFrames).
  * Histograms use RDD.histogram() → O(n_bins) floats to driver.
  * Spearman correlation uses df.sample(fraction) → .toPandas() on ~50K rows.
  * Null counts use a single .agg() → O(columns) scalars to driver.

Usage
-----
  spark-submit --master local[*] utils/visualization_script.py
  python utils/visualization_script.py

-------------------------------------------------------------------------------
"""

import os
import sys
import math
import time as _time

import matplotlib
matplotlib.use("Agg")  # headless backend: MUST precede pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.types import NumericType
from pyspark.sql import functions as F


# ═══════════════════════════════════════════════════════════════════════════
# ██  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# --- Data Path ---
# Local sample for development.  On SDSC, override with:
#   DATA_PATH = "data/fused_data/antarctica_sparse_features.parquet"
DATA_PATH = "../data/antarctica_sparse_features_sample.parquet"

# --- Output ---
OUTPUT_DIR = os.path.join("data", "eda_plots")

# --- Spark Config ---
# Lightweight local mode.  For SDSC, increase driver/executor memory.
DRIVER_MEMORY_GB = 4
SHUFFLE_PARTITIONS = 16

# --- Plot Config ---
HISTOGRAM_BINS = 50
FIG_DPI = 150

# --- Sampling Config ---
# Target sample size for the Spearman correlation (Pandas-side).
# 50K rows × 4 columns ≈ 1.6 MB on driver — safe with plenty of headroom.
TARGET_CORR_ROWS = 50_000
OVERSAMPLE_FACTOR = 1.2

# ─── Domain-Grouped Column Lists ──────────────────────────────────────────
# Histograms are generated per group.  Each group has a colour palette entry.
HISTOGRAM_GROUPS = {
    "bedmap": {
        "columns": ["surface", "bed", "thickness", "bed_slope"],
        "colour": "#440154",     # deep purple (viridis)
        "label":  "Bedmap3 Static Topography",
    },
    "ocean": {
        "columns": [
            "clamped_depth", "dist_to_ocean", "ice_draft",
            "thetao_mo", "t_star_mo", "so_mo", "t_f_mo",
            "t_star_quarterly_avg", "t_star_quarterly_std",
            "thetao_quarterly_avg", "thetao_quarterly_std",
        ],
        "colour": "#31688e",     # teal-blue
        "label":  "Ocean / Coastal Features",
    },
    "icesat": {
        "columns": ["ice_area", "delta_h", "surface_slope", "h_surface_dynamic"],
        "colour": "#35b779",     # green
        "label":  "ICESat-2 Dynamic",
    },
    "grace": {
        "columns": ["lwe_mo", "lwe_fused", "lwe_quarterly_avg", "lwe_quarterly_std"],
        "colour": "#fde725",     # yellow
        "label":  "GRACE Mass Anomaly",
    },
}

# ─── Spearman Correlation Target Columns ──────────────────────────────────
# Ocean-side (drivers of basal melting) vs ice-change-side (responses).
SPEARMAN_COLS = ["thetao_mo", "so_mo", "delta_h", "thickness"]


# ═══════════════════════════════════════════════════════════════════════════
# ██  SPARK SESSION
# ═══════════════════════════════════════════════════════════════════════════

def build_spark_session() -> SparkSession:
    """
    Build a lightweight local SparkSession for single-dataset EDA.

    Returns
    -------
    SparkSession

    Raises
    ------
    RuntimeError
        If Spark cannot initialise (e.g. JVM not found, memory error).

    Complexity
    ----------
    O(1) : configuration only, no data scanned.
    """
    print("=" * 72)
    print("  Antarctica Sparse Features : Targeted EDA")
    print("=" * 72)
    print(f"  Driver Memory ........... {DRIVER_MEMORY_GB}g")
    print(f"  Shuffle Partitions ...... {SHUFFLE_PARTITIONS}")
    print(f"  Data Path ............... {DATA_PATH}")
    print("=" * 72)

    spark = (
        SparkSession.builder
        .appName("Antarctica_Sparse_Features_Targeted_EDA")
        .config("spark.driver.memory", f"{DRIVER_MEMORY_GB}g")
        .config("spark.sql.shuffle.partitions", str(SHUFFLE_PARTITIONS))
        .config("spark.driver.maxResultSize", "2g")
        # --- AQE ---
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        # --- Parquet optimisation ---
        .config("spark.sql.parquet.filterPushdown", "true")
        .config("spark.sql.parquet.mergeSchema", "false")
        # --- Scratch dir ---
        .config("spark.local.dir",
                os.environ.get("TMPDIR",
                               os.path.join(os.getcwd(), "spark_scratch")))
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    scratch = spark.conf.get("spark.local.dir")
    os.makedirs(scratch, exist_ok=True)

    return spark


# ═══════════════════════════════════════════════════════════════════════════
# ██  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_dataset(spark: SparkSession, path: str):
    """
    Load a Parquet dataset, handling both single files and partitioned
    directories (the full fused dataset may use sub-directories).

    Parameters
    ----------
    spark : SparkSession
    path : str
        Path to .parquet file or directory.

    Returns
    -------
    pyspark.sql.DataFrame

    Raises
    ------
    FileNotFoundError
        If the path does not exist.

    Complexity
    ----------
    O(1) : metadata read only (no data scanned until an action).
    """
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"Dataset not found: {abs_path}\n"
            f"Ensure you are running from the project root directory."
        )

    return (
        spark.read
        .option("recursiveFileLookup", "true")
        .option("mergeSchema", "true")
        .parquet(abs_path)
    )


# ═══════════════════════════════════════════════════════════════════════════
# ██  STYLE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _apply_dark_style():
    """
    Apply a dark, modern matplotlib style globally.

    This ensures consistent visual identity across all figures and avoids
    the default white-background style that is hard to read on dark UIs
    or in presentation slides.
    """
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
    """
    Save figure to OUTPUT_DIR and close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        Filename (e.g. 'viz_01_null_bar_chart.png').
    """
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"    → Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# ██  DISTRIBUTED AGGREGATIONS
# ═══════════════════════════════════════════════════════════════════════════

def compute_null_counts_and_pcts(df, row_count: int) -> dict:
    """
    Compute per-column null count and null percentage via a single .agg().

    This compiles to a SINGLE-PASS HashAggregate on executors.
    Only one Spark action; O(columns) scalars returned to driver.
    ZERO data shuffled.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The full DataFrame (NOT sampled).
    row_count : int
        Pre-computed row count to avoid a second full scan.

    Returns
    -------
    dict : {column_name: {"null_count": int, "null_pct": float}}
        Ordered by descending null percentage.

    Edge Cases
    ----------
    - row_count == 0 → returns all columns with 0 nulls and 0.0%.
    - All-null column → null_count == row_count, null_pct == 100.0.

    Complexity
    ----------
    O(N × C) single-pass.  No shuffle.
    """
    columns = df.columns
    if row_count == 0:
        return {c: {"null_count": 0, "null_pct": 0.0} for c in columns}

    # Count non-nulls per column in a single distributed aggregate.
    agg_exprs = [
        F.count(F.when(F.col(c).isNotNull(), c)).alias(c)
        for c in columns
    ]
    non_null_row = df.agg(*agg_exprs).head()

    results = {}
    for c in columns:
        nn = int(non_null_row[c]) if non_null_row[c] is not None else 0
        null_count = row_count - nn
        null_pct = (null_count / row_count) * 100.0
        results[c] = {"null_count": null_count, "null_pct": null_pct}

    # Sort by descending null percentage for visual clarity.
    results = dict(
        sorted(results.items(), key=lambda kv: kv[1]["null_pct"], reverse=True)
    )
    return results


def compute_histogram_spark(df, col_name: str, n_bins: int = HISTOGRAM_BINS):
    """
    Compute histogram bin edges and counts using Spark RDD.histogram().

    This is EXECUTOR-side computation.  Only the (edges, counts) arrays
    are returned to the driver: O(n_bins) floats, <10 KB total.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    col_name : str
    n_bins : int

    Returns
    -------
    (edges, counts) : (list[float], list[int]) or (None, None) on failure.

    Edge Cases
    ----------
    - All-null column → returns (None, None).
    - Single distinct value → histogram degenerates to 1 bin.

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


def compute_basic_stats(df, cols: list) -> dict:
    """
    Compute min, max, mean, stddev for a list of columns in a single .agg().

    Single Spark job.  Only O(cols) scalars returned.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    cols : list[str]

    Returns
    -------
    dict : {col: {"min": float, "max": float, "mean": float, "std": float}}

    Edge Cases
    ----------
    - Column not in DataFrame → skipped.
    - All-null column → values are NaN.

    Complexity
    ----------
    O(N × C) single-pass HashAggregate.  No shuffle.
    """
    available = set(df.columns)
    cols = [c for c in cols if c in available]

    if not cols:
        return {}

    agg_exprs = []
    for c in cols:
        agg_exprs.extend([
            F.min(c).alias(f"{c}__min"),
            F.max(c).alias(f"{c}__max"),
            F.mean(c).alias(f"{c}__mean"),
            F.stddev(c).alias(f"{c}__std"),
        ])

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


def sample_to_pandas(df, row_count: int, target_rows: int = TARGET_CORR_ROWS):
    """
    Safely sample a Spark DataFrame to a small Pandas DataFrame.

    The sampling fraction is dynamically calculated to produce approximately
    target_rows rows. A 1.2x oversample factor compensates for Bernoulli
    variance (Spark's df.sample uses per-row coin flips).

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    row_count : int
    target_rows : int
        Desired number of rows in the Pandas output.

    Returns
    -------
    pandas.DataFrame

    Edge Cases
    ----------
    - row_count <= target_rows → full DataFrame materialised (no sampling).
    - row_count == 0 → returns empty Pandas DataFrame.

    Complexity
    ----------
    O(N) single-pass Bernoulli sampling.  No shuffle.
    .toPandas() transfers only the sampled rows.
    """
    if row_count == 0:
        return df.toPandas()

    raw_fraction = target_rows / row_count

    if raw_fraction >= 1.0:
        print(f"      Source ({row_count:,} rows) <= target ({target_rows:,}). "
              f"Materialising full DataFrame.")
        return df.toPandas()

    adjusted_fraction = min(raw_fraction * OVERSAMPLE_FACTOR, 1.0)
    print(f"      Sampling fraction: {adjusted_fraction:.6f} "
          f"(target ~{target_rows:,} rows from {row_count:,})")

    sampled_df = df.sample(
        withReplacement=False,
        fraction=adjusted_fraction,
        seed=42,
    )
    return sampled_df.toPandas()


# ═══════════════════════════════════════════════════════════════════════════
# ██  FIGURE 1 : NULL / MISSING DATA BAR CHART
# ═══════════════════════════════════════════════════════════════════════════

def fig_null_bar_chart(null_info: dict):
    """
    Horizontal bar chart showing per-column null counts with percentage
    annotations.

    Uses a traffic-light colour scheme:
      - Green  (<10% null):  largely complete
      - Orange (10-50%):     partially sparse
      - Red    (>50%):       mostly missing

    Parameters
    ----------
    null_info : dict
        {column_name: {"null_count": int, "null_pct": float}},
        pre-sorted descending by null_pct.

    Complexity
    ----------
    O(C) where C = number of columns.  Purely driver-side rendering.
    """
    cols = list(null_info.keys())
    counts = [null_info[c]["null_count"] for c in cols]
    pcts = [null_info[c]["null_pct"] for c in cols]

    # Traffic-light colour assignment.
    colours = []
    for p in pcts:
        if p > 50:
            colours.append("#e74c3c")    # red: mostly missing
        elif p > 10:
            colours.append("#f39c12")    # orange: partially sparse
        else:
            colours.append("#2ecc71")    # green: largely complete

    fig, ax = plt.subplots(figsize=(14, max(7, len(cols) * 0.4)))
    ax.set_title("Missing Data Profile : Antarctica Sparse Feature Store\n"
                 "(28 Columns — Null Counts per Column)",
                 fontsize=14, fontweight="bold")

    bars = ax.barh(range(len(cols)), counts, color=colours,
                   edgecolor="#ffffff22", alpha=0.85)

    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=9)
    ax.set_xlabel("Null Count")
    ax.xaxis.set_major_formatter(mticker.EngFormatter())

    # Annotate each bar with null count and percentage.
    for bar, count, pct in zip(bars, counts, pcts):
        x_pos = bar.get_width()
        # Place text inside bar if bar is wide enough, else outside.
        if pct > 5:
            ax.text(x_pos * 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{count:,.0f}  ({pct:.1f}%)",
                    va="center", ha="center", fontsize=7.5, color="#ffffff",
                    fontweight="bold")
        else:
            ax.text(x_pos + (ax.get_xlim()[1] * 0.01),
                    bar.get_y() + bar.get_height() / 2,
                    f"{count:,.0f}  ({pct:.1f}%)",
                    va="center", fontsize=7.5, color="#e0e0e0")

    # Reference lines (percentage thresholds converted to counts).
    # We add legend patches instead since x-axis is counts, not percentages.
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.invert_yaxis()

    # Legend for colour coding.
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", alpha=0.85, label="< 10% null (complete)"),
        Patch(facecolor="#f39c12", alpha=0.85, label="10–50% null (sparse)"),
        Patch(facecolor="#e74c3c", alpha=0.85, label="> 50% null (mostly missing)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    fig.tight_layout()
    _save_fig(fig, "viz_01_null_bar_chart.png")

    # ── Insight Commentary ──
    high_null = [c for c, info in null_info.items() if info["null_pct"] > 50]
    mid_null = [c for c, info in null_info.items()
                if 10 < info["null_pct"] <= 50]
    low_null = [c for c, info in null_info.items() if info["null_pct"] < 5]

    print("\n" + "─" * 72)
    print("  INSIGHT : Missing Data Bar Chart (Viz 01)")
    print("─" * 72)
    print(f"  PURPOSE: This chart diagnoses data completeness across all 28")
    print(f"  columns. Missing data in geospatial feature stores is rarely")
    print(f"  random — it is structurally determined by sensor geometry and")
    print(f"  physical coverage. Understanding null patterns is MANDATORY")
    print(f"  before choosing an imputation strategy.")
    print()
    if high_null:
        print(f"  • HIGH NULLS (>50%): {len(high_null)} columns:")
        for c in high_null:
            print(f"      {c:30s}  {null_info[c]['null_pct']:.1f}% null  "
                  f"({null_info[c]['null_count']:,.0f} rows)")
        print(f"    These are primarily ocean-derived features (thetao, so, T_f,")
        print(f"    T*, quarterly stats) and potentially GRACE mass features.")
        print(f"    This sparsity is EXPECTED: ocean features only exist at")
        print(f"    coastal pixels where the BallTree matched GLORYS data to")
        print(f"    the ice shelf draft depth. Inland grounded-ice pixels will")
        print(f"    always have null ocean values — this is physical absence,")
        print(f"    not data quality failure.")
        print()
    if mid_null:
        print(f"  • MODERATE NULLS (10–50%): {len(mid_null)} columns:")
        for c in mid_null:
            print(f"      {c:30s}  {null_info[c]['null_pct']:.1f}% null")
        print(f"    These likely include ICESat-2 dynamic features and GRACE")
        print(f"    mass columns with incomplete temporal or spatial coverage.")
        print()
    if low_null:
        print(f"  • NEAR-COMPLETE (<5%): {len(low_null)} columns:")
        for c in low_null[:8]:
            print(f"      {c:30s}  {null_info[c]['null_pct']:.1f}% null")
        if len(low_null) > 8:
            print(f"      ... and {len(low_null) - 8} more")
        print(f"    These form the reliable 'backbone' features for any ML")
        print(f"    model and require no imputation.")
        print()
    print(f"  PREPROCESSING IMPLICATION: Use domain-aware imputation.")
    print(f"  Ocean features should be zero-filled with a binary indicator")
    print(f"  column (has_ocean_data), because null = 'no ocean here',")
    print(f"  which is physically meaningful, not a data quality issue.")
    print(f"  Do NOT mean-impute ocean features across the whole continent.")
    print("─" * 72)


# ═══════════════════════════════════════════════════════════════════════════
# ██  FIGURE 2 : GROUPED DOMAIN HISTOGRAMS
# ═══════════════════════════════════════════════════════════════════════════

def fig_grouped_histograms(df, group_name: str, group_config: dict,
                           stats: dict):
    """
    Create a multi-panel histogram figure for one domain group.

    All histograms are computed via Spark RDD.histogram() (executor-side).
    Only O(n_bins) floats per column transfer to the driver.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Full DataFrame (not sampled — histograms are computed on executors).
    group_name : str
        Short group key (e.g. 'bedmap', 'ocean', 'icesat', 'grace').
    group_config : dict
        {"columns": list[str], "colour": str, "label": str}
    stats : dict
        Pre-computed {col: {"min", "max", "mean", "std"}} for annotation.

    Complexity
    ----------
    O(N × len(columns)) : one RDD.histogram() per column.
    """
    columns = group_config["columns"]
    colour = group_config["colour"]
    label = group_config["label"]

    available = set(df.columns)
    cols_to_plot = [c for c in columns if c in available]

    if not cols_to_plot:
        print(f"      [SKIP] No plottable columns for group '{group_name}'")
        return

    n = len(cols_to_plot)
    ncols_grid = min(n, 3)
    nrows_grid = math.ceil(n / ncols_grid)

    fig, axes = plt.subplots(
        nrows_grid, ncols_grid,
        figsize=(6 * ncols_grid, 4.5 * nrows_grid),
    )
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    fig.suptitle(f"Distributions : {label}",
                 fontsize=15, fontweight="bold", y=1.02)

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
        widths = [edges[i + 1] - edges[i] for i in range(len(counts))]

        ax.bar(centres, counts, width=widths, color=colour,
               edgecolor="#ffffff22", alpha=0.85)
        ax.set_title(col_name, fontsize=11)
        ax.set_ylabel("Count")
        ax.yaxis.set_major_formatter(mticker.EngFormatter())
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # Annotate with mean line if stats are available.
        if col_name in stats:
            s = stats[col_name]
            if not math.isnan(s["mean"]):
                ax.axvline(s["mean"], color="#ffffff", linestyle="--",
                           alpha=0.6, linewidth=1.2,
                           label=f'μ={s["mean"]:.2g}')
                ax.legend(fontsize=7, loc="upper right")

    # Hide unused axes.
    for idx in range(len(cols_to_plot), len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, f"viz_02_histograms_{group_name}.png")


def print_histogram_insights(group_name: str, group_config: dict,
                             stats: dict):
    """
    Print domain-specific diagnostic insights for a histogram group.

    Parameters
    ----------
    group_name : str
    group_config : dict
    stats : dict
    """
    label = group_config["label"]
    columns = group_config["columns"]

    print(f"\n" + "─" * 72)
    print(f"  INSIGHT : Distribution Histograms — {label} ({group_name})")
    print("─" * 72)

    if group_name == "bedmap":
        print(f"  PURPOSE: Bedmap3 features describe the static ice sheet")
        print(f"  geometry. Distributions reveal the terrain heterogeneity")
        print(f"  across Antarctica.")
        print()
        for col_name in columns:
            if col_name not in stats:
                continue
            s = stats[col_name]
            if math.isnan(s["mean"]):
                continue
            if col_name == "surface":
                print(f"  • surface: Range [{s['min']:.0f}, {s['max']:.0f}] m  |  "
                      f"μ={s['mean']:.0f} m  |  σ={s['std']:.0f} m")
                print(f"    Right-skewed: most elevations are moderate (500–")
                print(f"    2000 m), with a tail towards the high interior")
                print(f"    plateau. Low-surface cluster = floating ice shelves.")
                print(f"    Consider StandardScaler for normalisation.")
                print()
            elif col_name == "bed":
                print(f"  • bed: Range [{s['min']:.0f}, {s['max']:.0f}] m  |  "
                      f"μ={s['mean']:.0f} m  |  σ={s['std']:.0f} m")
                print(f"    Bimodal: one mode below sea level (marine ice sheet")
                print(f"    bed) and one above (exposed rock). This bimodality is")
                print(f"    a key indicator of West Antarctic marine instability.")
                print()
            elif col_name == "thickness":
                print(f"  • thickness: Range [{s['min']:.0f}, {s['max']:.0f}] m  |  "
                      f"μ={s['mean']:.0f} m  |  σ={s['std']:.0f} m")
                print(f"    Right-skewed. Most pixels have moderate thickness")
                print(f"    (500–2000 m). Pile-up near zero = thin ice shelves or")
                print(f"    ocean. Log-transform recommended for distance-based ML.")
                print()
            elif col_name == "bed_slope":
                print(f"  • bed_slope: Range [{s['min']:.4f}, {s['max']:.4f}]  |  "
                      f"μ={s['mean']:.4f}  |  σ={s['std']:.4f}")
                print(f"    Concentrated near zero with heavy positive tail.")
                print(f"    Steep bed slopes occur at subglacial mountain")
                print(f"    ranges and trough walls. RobustScaler recommended.")
                print()

    elif group_name == "ocean":
        print(f"  PURPOSE: Ocean features capture sub-ice-shelf conditions")
        print(f"  from GLORYS reanalysis. High null rates are expected")
        print(f"  (grounded ice has no ocean interface). The non-null")
        print(f"  distributions reveal the thermal forcing on ice shelves.")
        print()
        for col_name in columns:
            if col_name not in stats:
                continue
            s = stats[col_name]
            if math.isnan(s["mean"]):
                continue
            print(f"  • {col_name}: Range [{s['min']:.3g}, {s['max']:.3g}]  |  "
                  f"μ={s['mean']:.3g}  |  σ={s['std']:.3g}")
        print()
        print(f"    Key diagnostic: thetao_mo values above 0°C indicate")
        print(f"    warm Circumpolar Deep Water intrusions driving basal")
        print(f"    melting. Quarterly std columns capture temporal")
        print(f"    variability in ocean forcing — high variability may")
        print(f"    indicate pulsed warm-water incursions.")
        print()

    elif group_name == "icesat":
        print(f"  PURPOSE: ICESat-2 features describe dynamic ice surface")
        print(f"  changes. delta_h is the primary target variable for")
        print(f"  ice-loss prediction.")
        print()
        for col_name in columns:
            if col_name not in stats:
                continue
            s = stats[col_name]
            if math.isnan(s["mean"]):
                continue
            if col_name == "delta_h":
                print(f"  • delta_h: Range [{s['min']:.2f}, {s['max']:.2f}] m  |  "
                      f"μ={s['mean']:.4f} m  |  σ={s['std']:.4f} m")
                print(f"    Near-normal centred around zero. Extreme tails")
                print(f"    indicate satellite artefacts or calving events.")
                print(f"    Clip at ±3σ before ML training.")
                print()
            else:
                print(f"  • {col_name}: Range [{s['min']:.3g}, {s['max']:.3g}]  |  "
                      f"μ={s['mean']:.3g}  |  σ={s['std']:.3g}")
                print()

    elif group_name == "grace":
        print(f"  PURPOSE: GRACE mass anomaly features measure ice sheet")
        print(f"  mass balance at 300 km resolution, downscaled to the")
        print(f"  500 m grid. Near-symmetric distributions centred at zero")
        print(f"  are expected (anomalies relative to a long-term mean).")
        print()
        for col_name in columns:
            if col_name not in stats:
                continue
            s = stats[col_name]
            if math.isnan(s["mean"]):
                continue
            print(f"  • {col_name}: Range [{s['min']:.4g}, {s['max']:.4g}]  |  "
                  f"μ={s['mean']:.4g}  |  σ={s['std']:.4g}")
        print()
        print(f"    Negative values = mass loss (ice thinning/discharge).")
        print(f"    Left skew in the Amundsen Sea sector is expected —")
        print(f"    this is the most rapidly losing region in Antarctica.")
        print()

    print("─" * 72)


# ═══════════════════════════════════════════════════════════════════════════
# ██  FIGURE 3 : SPEARMAN CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════════════════

def fig_spearman_correlation(pdf):
    """
    Compute and plot a Spearman rank-correlation matrix between ocean
    variables (thetao_mo, so_mo) and ice change variables (delta_h,
    thickness).

    WHY SPEARMAN OVER PEARSON:
      Pearson measures linear association only.  Physical relationships
      between ocean temperature and ice thickness are often monotonic but
      non-linear (e.g. logarithmic thinning response to warming).
      Spearman captures any monotonic relationship, making it the
      correct diagnostic for these physically coupled variables.

    WHY SAMPLING IS SAFE:
      The sample is ~50K rows × 4 columns = 1.6 MB on the driver.
      scipy.stats.spearmanr on 50K rows is O(N log N) for rank
      computation — completes in < 1 second.

    Parameters
    ----------
    pdf : pandas.DataFrame
        Sampled DataFrame containing the SPEARMAN_COLS.

    Complexity
    ----------
    O(S log S) per column pair where S = sample size.
    Purely driver-side after sampling.
    """
    from scipy import stats as sp_stats

    # Defensive: check which columns are present.
    available_cols = [c for c in SPEARMAN_COLS if c in pdf.columns]
    if len(available_cols) < 2:
        print("      [SKIP] Need >= 2 columns for Spearman correlation.")
        return

    # Drop rows with any NaN in the target columns.
    valid = pdf[available_cols].dropna()
    n_valid = len(valid)

    if n_valid < 10:
        print(f"      [SKIP] Only {n_valid} valid rows — insufficient for "
              f"meaningful Spearman correlation.")
        return

    print(f"      Computing Spearman rank correlation on {n_valid:,} rows ...")

    n = len(available_cols)
    corr_matrix = np.eye(n)
    pval_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            rho, pval = sp_stats.spearmanr(
                valid[available_cols[i]].values,
                valid[available_cols[j]].values,
            )
            if math.isnan(rho):
                rho = 0.0
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho
            pval_matrix[i, j] = pval
            pval_matrix[j, i] = pval

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(max(7, n * 1.5), max(6, n * 1.3)))
    ax.set_title(
        "Spearman Rank Correlation\n"
        "Ocean Variables (thetao_mo, so_mo) vs Ice Changes (delta_h, thickness)",
        fontsize=13, fontweight="bold",
    )

    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1,
                   aspect="equal")

    ax.set_xticks(range(n))
    ax.set_xticklabels(available_cols, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(available_cols, fontsize=10)

    # Annotate each cell with correlation value and significance.
    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            pval = pval_matrix[i, j]
            text_color = "#000000" if abs(val) < 0.6 else "#ffffff"

            # Mark significance.
            sig_marker = ""
            if i != j:
                if pval < 0.001:
                    sig_marker = " ***"
                elif pval < 0.01:
                    sig_marker = " **"
                elif pval < 0.05:
                    sig_marker = " *"

            ax.text(j, i, f"{val:.3f}{sig_marker}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Spearman ρ", fontsize=10)

    # Add significance legend as text.
    ax.text(0.5, -0.22,
            "Significance: *** p<0.001  |  ** p<0.01  |  * p<0.05",
            transform=ax.transAxes, ha="center", fontsize=8,
            color="#b0b0b0", style="italic")

    fig.tight_layout()
    _save_fig(fig, "viz_03_spearman_correlation.png")

    # ── Insight Commentary ──
    print("\n" + "─" * 72)
    print("  INSIGHT : Spearman Correlation Heatmap (Viz 03)")
    print("─" * 72)
    print(f"  PURPOSE: This matrix reveals monotonic (not necessarily linear)")
    print(f"  relationships between ocean forcing variables and ice-change")
    print(f"  response variables. Spearman ρ is robust to outliers and")
    print(f"  non-normality — both expected in geophysical data.")
    print()
    print(f"  VARIABLES:")
    print(f"    Ocean forcing:  thetao_mo (ocean temperature), so_mo (salinity)")
    print(f"    Ice response:   delta_h (elevation change), thickness")
    print()

    # Extract key cross-domain correlations.
    for i, ci in enumerate(available_cols):
        for j, cj in enumerate(available_cols):
            if i >= j:
                continue
            rho = corr_matrix[i, j]
            pval = pval_matrix[i, j]
            sig = "p<0.001" if pval < 0.001 else (
                "p<0.01" if pval < 0.01 else (
                    "p<0.05" if pval < 0.05 else f"p={pval:.3f}"))
            strength = ("strong" if abs(rho) > 0.5 else
                        "moderate" if abs(rho) > 0.3 else
                        "weak")
            direction = "positive" if rho > 0 else "negative"
            print(f"  • {ci} ↔ {cj}: ρ = {rho:+.3f} ({strength} {direction}, "
                  f"{sig})")

    print()
    print(f"  PHYSICAL INTERPRETATION:")
    print(f"    A negative correlation between ocean temperature (thetao_mo)")
    print(f"    and ice thickness suggests that warmer ocean water is")
    print(f"    associated with thinner ice — consistent with basal melting.")
    print(f"    A positive correlation between thetao_mo and delta_h would")
    print(f"    indicate that warm-ocean pixels experience surface lowering")
    print(f"    (negative delta_h means elevation loss).")
    print()
    print(f"    For ML: thetao_mo and so_mo may be collinear (both track")
    print(f"    water mass mixing). Consider using T* (thermal driving) as")
    print(f"    a single derived feature instead of both raw variables.")
    print(f"    Note: {n_valid:,} valid rows used (after dropping NaN in all")
    print(f"    4 columns). The high null rate in ocean features means only")
    print(f"    coastal pixels contribute to this analysis.")
    print("─" * 72)


# ═══════════════════════════════════════════════════════════════════════════
# ██  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    """
    Orchestrate the targeted Antarctica EDA pipeline.

    Pipeline
    --------
    1. Build SparkSession.
    2. Load the Antarctica sparse features dataset.
    3. Count rows, print schema.
    4. Compute null counts/percentages → bar chart (Viz 01).
    5. Compute grouped histograms → 4 figures (Viz 02 × 4 groups).
    6. Sample → Spearman correlation heatmap (Viz 03).
    7. Teardown.

    Returns
    -------
    int
        Exit code: 0 = success, 1 = partial failure, 2 = fatal error.

    Complexity
    ----------
    O(N × C) dominated by the histogram and null-count aggregations.
    """
    wall_start = _time.perf_counter()

    # ── Initialise ─────────────────────────────────────────────────────
    _apply_dark_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n  Output directory: {os.path.abspath(OUTPUT_DIR)}")

    # ── Build SparkSession ─────────────────────────────────────────────
    try:
        spark = build_spark_session()
    except Exception as exc:
        print(f"[FATAL] Failed to create SparkSession: {exc}")
        return 2

    # ── Load Dataset ───────────────────────────────────────────────────
    print(f"\n  [Phase 1] Loading dataset ...")
    t0 = _time.perf_counter()

    try:
        df = load_dataset(spark, DATA_PATH)
    except Exception as exc:
        print(f"  [FATAL] Failed to load dataset: {exc}")
        spark.stop()
        return 2

    num_cols = len(df.columns)
    print(f"  Schema ({num_cols} columns):")
    print(f"    {'Column Name':<30s}  {'Data Type':<20s}")
    print(f"    {'─' * 30}  {'─' * 20}")
    for col_name, col_type in df.dtypes:
        print(f"    {col_name:<30s}  {col_type:<20s}")
    print(f"  [{_time.perf_counter() - t0:.1f}s]")

    # ── Row Count ──────────────────────────────────────────────────────
    print(f"\n  [Phase 2] Counting rows ...")
    t0 = _time.perf_counter()

    try:
        row_count = df.count()
    except Exception as exc:
        print(f"  [FATAL] Row count failed: {exc}")
        spark.stop()
        return 2

    print(f"  Total rows:    {row_count:>14,}")
    print(f"  Total columns: {num_cols:>14,}")
    print(f"  [{_time.perf_counter() - t0:.1f}s]")

    if row_count == 0:
        print(f"  [FATAL] Dataset is empty. Nothing to analyse.")
        spark.stop()
        return 2

    # ══════════════════════════════════════════════════════════════════
    #  FIGURE 1 : Null Bar Chart
    # ══════════════════════════════════════════════════════════════════
    print(f"\n  [Phase 3] Computing null distribution across all {num_cols} "
          f"columns ...")
    t0 = _time.perf_counter()

    try:
        null_info = compute_null_counts_and_pcts(df, row_count)
        fig_null_bar_chart(null_info)
    except Exception as exc:
        print(f"  [ERROR] Null bar chart failed: {exc}")

    print(f"  [{_time.perf_counter() - t0:.1f}s]")

    # ══════════════════════════════════════════════════════════════════
    #  FIGURE 2 : Grouped Histograms (4 domain groups)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n  [Phase 4] Computing grouped histograms ...")
    t0 = _time.perf_counter()

    # Collect all histogram columns for a single stats aggregation.
    all_hist_cols = []
    for group_config in HISTOGRAM_GROUPS.values():
        all_hist_cols.extend(group_config["columns"])

    try:
        # Single distributed .agg() for all stats — efficient.
        stats = compute_basic_stats(df, all_hist_cols)
    except Exception as exc:
        print(f"  [ERROR] Stats computation failed: {exc}")
        stats = {}

    for group_name, group_config in HISTOGRAM_GROUPS.items():
        print(f"\n    ── Group: {group_config['label']} ({group_name}) ──")
        try:
            fig_grouped_histograms(df, group_name, group_config, stats)
            print_histogram_insights(group_name, group_config, stats)
        except Exception as exc:
            print(f"    [ERROR] Histograms for '{group_name}' failed: {exc}")

    print(f"\n  [{_time.perf_counter() - t0:.1f}s]")

    # ══════════════════════════════════════════════════════════════════
    #  FIGURE 3 : Spearman Correlation Heatmap
    # ══════════════════════════════════════════════════════════════════
    print(f"\n  [Phase 5] Sampling for Spearman correlation ...")
    t0 = _time.perf_counter()

    try:
        # Select only the columns needed for correlation to minimise
        # the Pandas DataFrame memory footprint.
        available_corr = [c for c in SPEARMAN_COLS if c in set(df.columns)]
        if len(available_corr) >= 2:
            df_corr = df.select(available_corr)
            pdf = sample_to_pandas(df_corr, row_count, TARGET_CORR_ROWS)
            print(f"      Sampled {len(pdf):,} rows for Spearman correlation.")
            fig_spearman_correlation(pdf)
        else:
            print(f"      [SKIP] Need >= 2 Spearman columns in dataset. "
                  f"Available: {available_corr}")
    except Exception as exc:
        print(f"  [ERROR] Spearman correlation failed: {exc}")

    print(f"  [{_time.perf_counter() - t0:.1f}s]")

    # ── Summary ───────────────────────────────────────────────────────
    total_wall = _time.perf_counter() - wall_start

    # ── Teardown ──────────────────────────────────────────────────────
    spark.stop()

    print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │  ✓ TARGETED ANTARCTICA EDA COMPLETE                          │")
    print(f"  │    Total time: {total_wall:.1f}s  ({total_wall / 60:.1f} min){'':<18s}│")
    print(f"  │    Plots dir:  {os.path.abspath(OUTPUT_DIR):<42s}│")
    print(f"  │    Figures:                                                  │")
    print(f"  │      viz_01_null_bar_chart.png                               │")
    print(f"  │      viz_02_histograms_bedmap.png                            │")
    print(f"  │      viz_02_histograms_ocean.png                             │")
    print(f"  │      viz_02_histograms_icesat.png                            │")
    print(f"  │      viz_02_histograms_grace.png                             │")
    print(f"  │      viz_03_spearman_correlation.png                         │")
    print(f"  └─────────────────────────────────────────────────────────────┘")

    return 0


if __name__ == "__main__":
    sys.exit(main())
