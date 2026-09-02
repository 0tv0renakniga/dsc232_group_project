"""
spark_eda_bonus_plots.py

-------------------------------------------------------------------------------
COMPUTATIONAL GLACIOLOGY : BONUS EDA VISUALISATIONS
-------------------------------------------------------------------------------
DATE:   2026-02-19
STATUS: PRODUCTION

Generates four wish-list figures that complement the main EDA pipeline:

  fig_07  Ice Mask + Ocean Data Coverage  (2x1 spatial scatter)
  fig_08  Height Change + LWE Fused       (2x1 spatial scatter)
  fig_09  Monthly Mean delta_h             (time series + linear trend)
  fig_10  Monthly Mean LWE                 (time series + linear trend)

Data Sources
------------
  Spatial plots   : sample Parquet files from data/sample_data/ (~250K rows).
                    These are small enough for pandas : no Spark required.
  Time series     : full fused dataset via Spark GROUP BY month aggregation.
                    Only ~84 output rows (7 years x 12 months) are collected
                    to the driver.

HPC Runtime Context
-------------------
  Same 32 core / 128 GB shared-partition configuration as the main pipeline.
  The Spark session is only needed for the monthly aggregation (one GROUP BY
  shuffle producing ~84 rows).

Shuffle Analysis
----------------
  One unavoidable shuffle for the GROUP BY month.
  Input  : 1,386,866,499 rows.
  Output : ~84 rows (partial aggregates combine to monthly means).
  AQE auto-coalesces the tiny output partitions.

Output dir: data/eda_plots/

Usage
-----
  spark-submit --master local[*] utils/spark_eda_bonus_plots.py
  spark-submit --master yarn     utils/spark_eda_bonus_plots.py
-------------------------------------------------------------------------------
"""

import os
import sys
import math
import time as _time

import matplotlib
matplotlib.use("Agg")  # headless backend : MUST precede pyplot import
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ═══════════════════════════════════════════════════════════════════════════
# ██  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# --- HPC Resource Allocation (mirrors main pipeline) ---
TOTAL_CORES      = 32
TOTAL_MEMORY_GB  = 128
EXECUTOR_CORES   = 5
EXECUTOR_INSTANCES = max((TOTAL_CORES - 1) // EXECUTOR_CORES, 1)  # -> 6
DRIVER_MEMORY_GB   = 10
EXECUTOR_MEMORY_GB = max(
    math.floor((TOTAL_MEMORY_GB - DRIVER_MEMORY_GB) / EXECUTOR_INSTANCES),
    1,
)
SHUFFLE_PARTITIONS = 2 * TOTAL_CORES

# --- Paths ---
SAMPLE_DIR = os.path.join("data", "sample_data")
FUSED_PATH = os.path.join("data", "fused_data",
                          "antarctica_sparse_features.parquet")
OUTPUT_DIR = os.path.join("data", "eda_plots")

FIG_DPI = 150


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
    print(f"    -> Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# ██  SAMPLE DATA LOADER (pandas)
# ═══════════════════════════════════════════════════════════════════════════

def _load_sample(name: str) -> pd.DataFrame | None:
    """
    Load a sample Parquet directory from data/sample_data/<name>.

    The coalesce(1) writer creates a directory containing a single
    part-*.parquet file plus _SUCCESS.  pandas can read this directly.

    Parameters
    ----------
    name : str
        Directory name, e.g. 'bedmap3_static_sample.parquet'.

    Returns
    -------
    pd.DataFrame or None if not found.
    """
    path = os.path.join(SAMPLE_DIR, name)
    if not os.path.exists(path):
        print(f"    [WARN] Sample not found: {path}")
        return None

    try:
        pdf = pd.read_parquet(path)
        print(f"    Loaded {len(pdf):,} rows from {name}")
        return pdf
    except Exception as exc:
        print(f"    [ERROR] Failed to read {name}: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# ██  SPARK SESSION BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_spark_session() -> SparkSession:
    """
    Construct an HPC-optimised SparkSession for the monthly aggregation.

    Resource Allocation (mirrors main pipeline)
    -------------------
    32 cores / 128 GB / 6 executors x 19 GB / driver 10 GB

    Returns
    -------
    SparkSession
    """
    spark = (
        SparkSession.builder
        .appName("HPC_Antarctic_Bonus_Plots")
        .config("spark.driver.memory",            f"{DRIVER_MEMORY_GB}g")
        .config("spark.executor.instances",        str(EXECUTOR_INSTANCES))
        .config("spark.executor.cores",            str(EXECUTOR_CORES))
        .config("spark.executor.memory",           f"{EXECUTOR_MEMORY_GB}g")
        .config("spark.sql.shuffle.partitions",    str(SHUFFLE_PARTITIONS))
        .config("spark.driver.maxResultSize",      "4g")
        .config("spark.network.timeout",           "1200s")
        .config("spark.sql.adaptive.enabled",                      "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled",   "true")
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128m")
        .config("spark.sql.parquet.filterPushdown",                "true")
        .config("spark.sql.parquet.mergeSchema",                   "false")
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
# ██  MONTHLY AGGREGATION (Spark -> small pandas)
# ═══════════════════════════════════════════════════════════════════════════

def compute_monthly_aggregates(spark: SparkSession) -> pd.DataFrame | None:
    """
    GROUP BY month on the FULL fused dataset to compute monthly means
    for delta_h and lwe_fused.

    Produces ~84 rows (7 years x 12 months).  Safe to .toPandas().

    Complexity
    ----------
    O(N) scan + 1 shuffle (GROUP BY).
    Output is tiny (~84 rows), so AQE auto-coalesces aggressively.
    """
    fused_abs = os.path.abspath(FUSED_PATH)
    if not os.path.exists(fused_abs):
        print(f"    [SKIP] Fused dataset not found: {fused_abs}")
        return None

    print(f"\n  Computing monthly aggregates from full fused dataset ...")
    print(f"    Path: {fused_abs}")
    t0 = _time.perf_counter()

    df = (
        spark.read
        .option("recursiveFileLookup", "true")
        .option("mergeSchema", "true")
        .parquet(fused_abs)
    )

    monthly = (
        df
        .withColumn("month_year", F.date_trunc("month", "exact_time"))
        .groupBy("month_year")
        .agg(
            F.mean("delta_h").alias("mean_delta_h"),
            F.mean("lwe_fused").alias("mean_lwe_fused"),
            F.count("*").alias("n_rows"),
        )
        .orderBy("month_year")
    )

    # ~84 rows -> safe to collect to driver.
    pdf = monthly.toPandas()
    pdf["month_year"] = pd.to_datetime(pdf["month_year"])
    pdf = pdf.dropna(subset=["month_year"]).sort_values("month_year")

    elapsed = _time.perf_counter() - t0
    print(f"    -> {len(pdf)} monthly records aggregated in {elapsed:.1f}s")
    print(f"    Date range: {pdf['month_year'].min()} to {pdf['month_year'].max()}")

    return pdf


# ═══════════════════════════════════════════════════════════════════════════
# ██  FIGURE 7 : Ice Mask + Ocean Data Coverage (2x1 spatial)
# ═══════════════════════════════════════════════════════════════════════════

def fig_ice_mask_ocean_coverage():
    """
    2x1 subplot:
      Top:    Bedmap3 ice mask spatial distribution (x, y coloured by mask)
              mask = {1: grounded ice, 2: floating ice, 3: ocean}
      Bottom: Fused dataset showing where valid ocean data exists
              (thetao_mo is not null)

    Uses sample data (~250K rows each) via pandas.  No Spark needed.

    Complexity
    ----------
    O(S) where S = sample size.  Two pandas reads + two scatter plots.
    """
    print("\n  Building fig_07: Ice Mask + Ocean Coverage ...")

    pdf_bedmap = _load_sample("bedmap3_static_sample.parquet")
    pdf_fused  = _load_sample("antarctica_sparse_features_sample.parquet")

    if pdf_bedmap is None or pdf_fused is None:
        print("    [SKIP] Required sample data not available.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 18))
    fig.suptitle("Ice Mask & Ocean Data Coverage",
                 fontsize=16, fontweight="bold", y=0.95)

    # ── Top: Ice Mask ──────────────────────────────────────────────────
    mask_palette = {1: "#35b779", 2: "#31688e", 3: "#fde725"}
    mask_labels  = {1: "Grounded Ice", 2: "Floating Ice", 3: "Ocean"}

    for mask_val in sorted(pdf_bedmap["mask"].dropna().unique()):
        mv = int(mask_val)
        subset = pdf_bedmap[pdf_bedmap["mask"] == mask_val]
        ax1.scatter(
            subset["x"], subset["y"],
            s=0.15, alpha=0.35, rasterized=True,
            c=mask_palette.get(mv, "#888888"),
            label=mask_labels.get(mv, f"Mask={mv}"),
        )

    ax1.set_title("Bedmap3 Ice Mask Classification")
    ax1.set_xlabel("x (EPSG:3031, m)")
    ax1.set_ylabel("y (EPSG:3031, m)")
    ax1.set_aspect("equal")
    ax1.legend(markerscale=25, loc="upper right", fontsize=9,
               framealpha=0.6)
    ax1.grid(alpha=0.2)
    ax1.xaxis.set_major_formatter(mticker.EngFormatter())
    ax1.yaxis.set_major_formatter(mticker.EngFormatter())

    # ── Bottom: Ocean Data Coverage ────────────────────────────────────
    # Check which column names exist for ocean temperature.
    ocean_col = None
    for candidate in ["thetao_mo", "thetao", "T_star", "t_star_mo"]:
        if candidate in pdf_fused.columns:
            ocean_col = candidate
            break

    if ocean_col is None:
        print("    [WARN] No ocean column found in fused sample.")
        ax2.text(0.5, 0.5, "No Ocean Column Found",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=14, color="#888888")
    else:
        has_ocean = pdf_fused[ocean_col].notna()
        no_ocean  = ~has_ocean

        ax2.scatter(
            pdf_fused.loc[no_ocean, "x"], pdf_fused.loc[no_ocean, "y"],
            s=0.1, alpha=0.12, c="#555555", label="No Ocean Data",
            rasterized=True,
        )
        ax2.scatter(
            pdf_fused.loc[has_ocean, "x"], pdf_fused.loc[has_ocean, "y"],
            s=0.4, alpha=0.55, c="#e76f51", label="Valid Ocean Data",
            rasterized=True,
        )

    ax2.set_title(f"Ocean Data Coverage ({ocean_col})")
    ax2.set_xlabel("x (EPSG:3031, m)")
    ax2.set_ylabel("y (EPSG:3031, m)")
    ax2.set_aspect("equal")
    ax2.legend(markerscale=20, loc="upper right", fontsize=9,
               framealpha=0.6)
    ax2.grid(alpha=0.2)
    ax2.xaxis.set_major_formatter(mticker.EngFormatter())
    ax2.yaxis.set_major_formatter(mticker.EngFormatter())

    fig.tight_layout()
    _save_fig(fig, "fig_07_ice_mask_ocean_coverage.png")


# ═══════════════════════════════════════════════════════════════════════════
# ██  FIGURE 8 : delta_h + lwe_fused Spatial (2x1)
# ═══════════════════════════════════════════════════════════════════════════

def fig_delta_h_vs_lwe_spatial():
    """
    2x1 subplot:
      Top:    Spatial distribution of delta_h (ice height change)
      Bottom: Spatial distribution of lwe_fused (liquid water equivalent)

    Uses the fused sample data (~250K rows) via pandas.
    Colour ranges clipped to [1st, 99th] percentile to mitigate outliers.

    Complexity
    ----------
    O(S) where S = sample size.  One pandas read + two scatter plots.
    """
    print("\n  Building fig_08: delta_h vs LWE Fused (spatial) ...")

    pdf = _load_sample("antarctica_sparse_features_sample.parquet")
    if pdf is None:
        print("    [SKIP] Fused sample not available.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 18))
    fig.suptitle("Height Change & Mass Anomaly : Spatial Distribution",
                 fontsize=16, fontweight="bold", y=0.95)

    # ── Top: delta_h ──────────────────────────────────────────────────
    valid_dh = pdf.dropna(subset=["delta_h", "x", "y"])
    if valid_dh.empty:
        ax1.text(0.5, 0.5, "No delta_h Data", ha="center", va="center",
                 transform=ax1.transAxes, fontsize=14, color="#888888")
    else:
        vmin = valid_dh["delta_h"].quantile(0.01)
        vmax = valid_dh["delta_h"].quantile(0.99)

        sc1 = ax1.scatter(
            valid_dh["x"], valid_dh["y"],
            s=0.2, alpha=0.45, rasterized=True,
            c=valid_dh["delta_h"], cmap="RdBu_r",
            vmin=vmin, vmax=vmax,
        )
        cb1 = fig.colorbar(sc1, ax=ax1, shrink=0.8, pad=0.02)
        cb1.set_label("delta_h (m)", fontsize=9)

    ax1.set_title("Ice Height Change (delta_h)")
    ax1.set_xlabel("x (EPSG:3031, m)")
    ax1.set_ylabel("y (EPSG:3031, m)")
    ax1.set_aspect("equal")
    ax1.grid(alpha=0.2)
    ax1.xaxis.set_major_formatter(mticker.EngFormatter())
    ax1.yaxis.set_major_formatter(mticker.EngFormatter())

    # ── Bottom: lwe_fused ─────────────────────────────────────────────
    # Detect which LWE column is present.
    lwe_col = None
    for candidate in ["lwe_fused", "lwe_mo", "lwe_length"]:
        if candidate in pdf.columns:
            lwe_col = candidate
            break

    if lwe_col is None:
        ax2.text(0.5, 0.5, "No LWE Column Found", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=14, color="#888888")
    else:
        valid_lwe = pdf.dropna(subset=[lwe_col, "x", "y"])
        if valid_lwe.empty:
            ax2.text(0.5, 0.5, f"No valid {lwe_col} Data",
                     ha="center", va="center", transform=ax2.transAxes,
                     fontsize=14, color="#888888")
        else:
            vmin = valid_lwe[lwe_col].quantile(0.01)
            vmax = valid_lwe[lwe_col].quantile(0.99)

            sc2 = ax2.scatter(
                valid_lwe["x"], valid_lwe["y"],
                s=0.2, alpha=0.45, rasterized=True,
                c=valid_lwe[lwe_col], cmap="PiYG",
                vmin=vmin, vmax=vmax,
            )
            cb2 = fig.colorbar(sc2, ax=ax2, shrink=0.8, pad=0.02)
            cb2.set_label(f"{lwe_col} (cm eq. water)", fontsize=9)

    ax2.set_title(f"Liquid Water Equivalent ({lwe_col or 'N/A'})")
    ax2.set_xlabel("x (EPSG:3031, m)")
    ax2.set_ylabel("y (EPSG:3031, m)")
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.2)
    ax2.xaxis.set_major_formatter(mticker.EngFormatter())
    ax2.yaxis.set_major_formatter(mticker.EngFormatter())

    fig.tight_layout()
    _save_fig(fig, "fig_08_delta_h_vs_lwe_spatial.png")


# ═══════════════════════════════════════════════════════════════════════════
# ██  HELPER : Time-Series Plot with Seasonal Shading
# ═══════════════════════════════════════════════════════════════════════════

def _plot_timeseries_with_trend(
    ax: plt.Axes,
    dates: pd.Series,
    values: pd.Series,
    line_colour: str,
    ylabel: str,
    title: str,
    annotation_y_factor: float = 0.95,
):
    """
    Shared logic for both time-series figures:
      - Monthly line plot with markers.
      - Numpy linear trend (red dashed).
      - Antarctic Fall-Winter shading (March-August each year).
      - Seasonal annotation.

    Parameters
    ----------
    ax : matplotlib Axes
    dates : pd.Series of datetime64
    values : pd.Series of float
    line_colour : str   hex colour for the data line
    ylabel : str
    title : str
    annotation_y_factor : float
        Fraction of ymax for the annotation y-position.
    """
    # Data line
    ax.plot(dates, values,
            color=line_colour, linewidth=1.5,
            marker="o", markersize=3,
            label="Monthly Mean")

    # Linear trend
    x_num = mdates.date2num(dates)
    coeffs = np.polyfit(x_num, values, 1)
    y_pred = np.polyval(coeffs, x_num)
    ax.plot(dates, y_pred,
            color="red", linestyle="--", linewidth=1.5,
            label="Linear Trend")

    # Seasonal shading (Antarctic Fall-Winter = March-August)
    year_min = int(dates.dt.year.min())
    year_max = int(dates.dt.year.max())
    for yr in range(year_min, year_max + 1):
        start = pd.Timestamp(f"{yr}-03-01")
        end   = pd.Timestamp(f"{yr}-08-31")
        ax.axvspan(start, end, color="blue", alpha=0.08)

    # Annotation
    y_range = values.max() - values.min()
    anno_y  = values.min() + y_range * annotation_y_factor
    ax.text(
        dates.min() + pd.DateOffset(months=1),
        anno_y,
        "Shaded = Antarctic Fall-Winter (March-August)",
        color="#6688cc", fontsize=11,
    )

    # Formatting
    ax.set_xlabel("Month and Year")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlim(
        dates.min() - pd.DateOffset(months=1),
        dates.max() + pd.DateOffset(months=1),
    )
    ax.legend(loc="lower left", fontsize=10, framealpha=0.6)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


# ═══════════════════════════════════════════════════════════════════════════
# ██  FIGURE 9 : Monthly Mean delta_h Time Series
# ═══════════════════════════════════════════════════════════════════════════

def fig_delta_h_timeseries(pdf_monthly: pd.DataFrame | None):
    """
    Monthly mean delta_h (ice height change) with linear trend
    and Antarctic Fall-Winter shading.

    Parameters
    ----------
    pdf_monthly : pd.DataFrame
        Output of compute_monthly_aggregates().
    """
    print("\n  Building fig_09: delta_h Time Series ...")

    if pdf_monthly is None or pdf_monthly.empty:
        print("    [SKIP] No monthly data available.")
        return

    pdf = pdf_monthly.dropna(subset=["mean_delta_h"])
    if pdf.empty:
        print("    [SKIP] No valid delta_h data in monthly aggregates.")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    _plot_timeseries_with_trend(
        ax=ax,
        dates=pdf["month_year"],
        values=pdf["mean_delta_h"],
        line_colour="#35b779",
        ylabel="Average Ice Height Change (delta_h, m)",
        title="Mean Antarctic Ice Height Change, by Month (2018-2025)",
        annotation_y_factor=0.92,
    )

    fig.tight_layout()
    _save_fig(fig, "fig_09_delta_h_timeseries.png")


# ═══════════════════════════════════════════════════════════════════════════
# ██  FIGURE 10 : Monthly Mean LWE Time Series
# ═══════════════════════════════════════════════════════════════════════════

def fig_lwe_timeseries(pdf_monthly: pd.DataFrame | None):
    """
    Monthly mean LWE (liquid water equivalent, fused) with linear trend
    and Antarctic Fall-Winter shading.

    Parameters
    ----------
    pdf_monthly : pd.DataFrame
        Output of compute_monthly_aggregates().
    """
    print("\n  Building fig_10: LWE Fused Time Series ...")

    if pdf_monthly is None or pdf_monthly.empty:
        print("    [SKIP] No monthly data available.")
        return

    pdf = pdf_monthly.dropna(subset=["mean_lwe_fused"])
    if pdf.empty:
        print("    [SKIP] No valid LWE data in monthly aggregates.")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    _plot_timeseries_with_trend(
        ax=ax,
        dates=pdf["month_year"],
        values=pdf["mean_lwe_fused"],
        line_colour="#fde725",
        ylabel="Average LWE (cm eq. water thickness)",
        title="Mean Antarctic LWE (Fused), by Month (2018-2025)",
        annotation_y_factor=1.05,
    )

    fig.tight_layout()
    _save_fig(fig, "fig_10_lwe_timeseries.png")


# ═══════════════════════════════════════════════════════════════════════════
# ██  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    """
    Orchestrate all four bonus figures.

    Pipeline
    --------
    1. Build HPC SparkSession.
    2. Generate spatial plots from sample data (pandas, no heavy Spark).
    3. Aggregate full fused dataset by month (1 Spark GROUP BY).
    4. Generate time-series plots from the monthly aggregate.
    5. Print summary.

    Returns
    -------
    int
        Exit code: 0 = success, 1 = partial failure, 2 = fatal.
    """
    wall_start = _time.perf_counter()

    _apply_dark_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 72)
    print("  BONUS EDA PLOTS")
    print("=" * 72)
    print(f"    Output dir: {os.path.abspath(OUTPUT_DIR)}")

    # ── Build SparkSession (needed for time-series aggregation) ────────
    try:
        spark = build_spark_session()
    except Exception as exc:
        print(f"[FATAL] Failed to create SparkSession: {exc}")
        return 2

    n_failures = 0

    # ── Spatial plots (sample data via pandas, fast) ──────────────────
    try:
        fig_ice_mask_ocean_coverage()
    except Exception as exc:
        print(f"  [ERROR] fig_07 failed: {exc}")
        n_failures += 1

    try:
        fig_delta_h_vs_lwe_spatial()
    except Exception as exc:
        print(f"  [ERROR] fig_08 failed: {exc}")
        n_failures += 1

    # ── Time-series (Spark aggregation on full fused dataset) ─────────
    pdf_monthly = None
    try:
        pdf_monthly = compute_monthly_aggregates(spark)
    except Exception as exc:
        print(f"  [ERROR] Monthly aggregation failed: {exc}")
        n_failures += 1

    try:
        fig_delta_h_timeseries(pdf_monthly)
    except Exception as exc:
        print(f"  [ERROR] fig_09 failed: {exc}")
        n_failures += 1

    try:
        fig_lwe_timeseries(pdf_monthly)
    except Exception as exc:
        print(f"  [ERROR] fig_10 failed: {exc}")
        n_failures += 1

    # ── Cleanup ───────────────────────────────────────────────────────
    spark.stop()

    elapsed = _time.perf_counter() - wall_start

    print(f"\n{'=' * 72}")
    status_label = "COMPLETE" if n_failures == 0 else f"PARTIAL ({n_failures} failures)"
    print(f"  BONUS PLOTS {status_label}")
    print(f"  Total time: {elapsed:.1f}s  ({elapsed / 60:.1f} min)")
    print(f"  Output:     {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'=' * 72}")

    return 0 if n_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
