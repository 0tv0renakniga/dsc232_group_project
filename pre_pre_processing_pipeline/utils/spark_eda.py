"""
spark_eda.py

-------------------------------------------------------------------------------
COMPUTATIONAL GLACIOLOGY : HPC-OPTIMISED EXPLORATORY DATA ANALYSIS
-------------------------------------------------------------------------------
DATE:   2026-02-18
STATUS: PRODUCTION

Performs iterative, per-dataset descriptive statistics on the individual
flattened Parquet tables extracted from data/tar_files/indiv_data.tar.

Target Datasets
---------------
  1. bedmap3_static.parquet       (2D static topography + mascon + ocean 2D)
  2. grace.parquet                (mascon mass anomaly time series)
  3. icesat2_dynamic.parquet      (3D elevation-change kinematics, multi-step)
  4. ocean_dynamic.parquet        (3D ocean T, S, T_f, T*)

HPC Runtime Context
-------------------
  Cluster:           SDSC Expanse (or equivalent SLURM cluster)
  Resource Template: 32 cores / 128 GB RAM (configurable via constants)
  Driver Memory:     2 GB (fixed : never broadcast large tables to driver)
  Executor Formula:  instances = TOTAL_CORES - 1
                     memory   = floor((TOTAL_MEMORY_GB - DRIVER_MEMORY_GB)
                                      / instances) GB

Shuffle Minimisation
--------------------
  * Uses .summary() → Catalyst single-pass HashAggregates (no shuffle).
  * spark.sql.shuffle.partitions set to 2 x TOTAL_CORES (narrow default).
  * AQE enabled → auto-coalesce of post-shuffle partitions if any wide
    transformation sneaks through.
  * coalesce preferred over repartition wherever applicable.
  * No .collect() / .toPandas() : all output via .show() or .printSchema().

Usage
-----
  spark-submit --master local[*] utils/spark_eda.py
  spark-submit --master yarn     utils/spark_eda.py

  Adjust TOTAL_CORES and TOTAL_MEMORY_GB to match your SLURM allocation.
  For Speedup measurement, run twice:
    1. Single-executor baseline  (TOTAL_CORES = 2, i.e. 1 executor)
    2. Full-cluster run          (TOTAL_CORES = N)
  Record wall-clock times and plug into the formulae printed at the end.

-------------------------------------------------------------------------------
"""


import os
import sys
import math
import time as _time


from pyspark.sql import SparkSession
from pyspark.sql.types import NumericType


# ═══════════════════════════════════════════════════════════════════════════
# ██  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# --- HPC Resource Allocation ---
# Adjust these two constants to match your SLURM job allocation.
# Everything else is derived programmatically.
TOTAL_CORES      = 32     # --ntasks or total cores in the allocation
TOTAL_MEMORY_GB  = 128    # Total RAM in GB across the allocation

# --- Driver Memory (fixed ceiling) ---
# The driver only orchestrates; keep it small.  Raise ONLY if you are
# explicitly broadcasting a table > 1 GB to executors.
DRIVER_MEMORY_GB = 2

# --- Derived Executor Resources ---
EXECUTOR_INSTANCES = max(TOTAL_CORES - 1, 1)  # reserve 1 core for the driver
EXECUTOR_MEMORY_GB = max(
    math.floor((TOTAL_MEMORY_GB - DRIVER_MEMORY_GB) / EXECUTOR_INSTANCES),
    1,  # safety floor: at least 1 GB per executor
)

# --- Shuffle Tuning ---
# 2 x total cores is a conservative default that avoids creating thousands
# of tiny shuffle partitions.  AQE will coalesce further at runtime.
SHUFFLE_PARTITIONS = 2 * TOTAL_CORES

# --- Data Paths ---
# The tarball extracts to indiv_data/ relative to the project root.
# Adjust BASE_DATA_DIR if your extraction location differs.
BASE_DATA_DIR = os.path.join("data", "indiv_data")


# ═══════════════════════════════════════════════════════════════════════════
# ██  SPARK SESSION BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_spark_session() -> SparkSession:
    """
    Construct an HPC-optimised SparkSession.

    Resource Allocation
    -------------------
    driver.memory        = DRIVER_MEMORY_GB  (2 GB default : fixed ceiling)
    executor.instances   = TOTAL_CORES - 1
    executor.memory      = floor((TOTAL_MEMORY_GB - DRIVER_MEMORY_GB)
                                 / EXECUTOR_INSTANCES)

    Shuffle Strategy
    ----------------
    shuffle.partitions   = 2 x TOTAL_CORES
    AQE enabled          → runtime coalesce of post-shuffle partitions.

    Returns
    -------
    SparkSession
        Configured and ready for distributed EDA.

    Raises
    ------
    RuntimeError
        If derived executor memory is less than 1 GB (under-provisioned node).

    Complexity
    ----------
    O(1) : configuration only, no data scanned.
    """
    if EXECUTOR_MEMORY_GB < 1:
        raise RuntimeError(
            f"Derived executor memory is {EXECUTOR_MEMORY_GB} GB. "
            f"The HPC allocation is too small for the requested "
            f"{EXECUTOR_INSTANCES} executors.  Reduce TOTAL_CORES or "
            f"increase TOTAL_MEMORY_GB."
        )

    driver_mem  = f"{DRIVER_MEMORY_GB}g"
    exec_mem    = f"{EXECUTOR_MEMORY_GB}g"

    print("=" * 72)
    print("  HPC SparkSession Configuration")
    print("=" * 72)
    print(f"  TOTAL_CORES ........... {TOTAL_CORES}")
    print(f"  TOTAL_MEMORY_GB ....... {TOTAL_MEMORY_GB}")
    print(f"  DRIVER_MEMORY ......... {driver_mem}")
    print(f"  EXECUTOR_INSTANCES .... {EXECUTOR_INSTANCES}")
    print(f"  EXECUTOR_MEMORY ....... {exec_mem}")
    print(f"  SHUFFLE_PARTITIONS .... {SHUFFLE_PARTITIONS}")
    print("=" * 72)

    spark = (
        SparkSession.builder
        .appName("HPC_Parquet_EDA")
        .config("spark.driver.memory",            driver_mem)
        .config("spark.executor.instances",        str(EXECUTOR_INSTANCES))
        .config("spark.executor.memory",           exec_mem)
        .config("spark.sql.shuffle.partitions",    str(SHUFFLE_PARTITIONS))
        # --- Adaptive Query Execution ---
        .config("spark.sql.adaptive.enabled",                      "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled",   "true")
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128m")
        # --- Parquet pushdown & vectorisation ---
        .config("spark.sql.parquet.filterPushdown",                "true")
        .config("spark.sql.parquet.mergeSchema",                   "false")
        .getOrCreate()
    )

    # Suppress noisy INFO-level Spark logs during EDA.
    spark.sparkContext.setLogLevel("WARN")

    return spark


# ═══════════════════════════════════════════════════════════════════════════
# ██  DATASET DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════

def discover_parquet_datasets(base_dir: str) -> list[str]:
    """
    Scan *base_dir* for top-level entries ending in '.parquet'.

    Parameters
    ----------
    base_dir : str
        Path to the directory containing parquet datasets (e.g. 'data/indiv_data').

    Returns
    -------
    list[str]
        Sorted list of absolute paths to .parquet entries (files or directories).

    Raises
    ------
    FileNotFoundError
        If *base_dir* does not exist.

    Edge Cases
    ----------
    - Empty directory  → returns []
    - Non-.parquet entries silently skipped.
    - Symlinks are followed (os.listdir behaviour).

    Complexity
    ----------
    O(N) where N = number of entries in base_dir (no recursion here).
    """
    abs_dir = os.path.abspath(base_dir)
    if not os.path.isdir(abs_dir):
        raise FileNotFoundError(
            f"Data directory not found: {abs_dir}\n"
            f"Ensure the tarball has been extracted to {base_dir}."
        )

    datasets = []
    for entry in sorted(os.listdir(abs_dir)):
        if entry.endswith(".parquet"):
            datasets.append(os.path.join(abs_dir, entry))

    if not datasets:
        print(f"[WARNING] No .parquet entries found in {abs_dir}")

    return datasets


# ═══════════════════════════════════════════════════════════════════════════
# ██  PER-DATASET EDA
# ═══════════════════════════════════════════════════════════════════════════

def run_eda_for_dataset(spark: SparkSession, dataset_path: str) -> float:
    """
    Perform full EDA on a single Parquet dataset.

    Steps
    -----
    1. Ingest with recursiveFileLookup=true (bypass Hive partition discovery).
    2. Report row count, column count.
    3. Print full schema (column names + data types).
    4. Isolate numeric columns (filter out TimestampType, StringType, etc.).
    5. Compute summary statistics via .summary() : single-pass HashAggregates.
    6. Display results via .show() : never .collect()/.toPandas().

    Parameters
    ----------
    spark : SparkSession
        Active SparkSession.
    dataset_path : str
        Absolute path to the .parquet file or directory.

    Returns
    -------
    float
        Wall-clock seconds spent on this dataset's EDA.

    Raises
    ------
    Does not raise : prints errors inline and returns elapsed time.
    Designed so one corrupted dataset does not abort the entire run.

    Complexity
    ----------
    - Schema:  O(1) : Parquet footer metadata only.
    - Count:   O(N) single-pass via Parquet row-group counts (no shuffle).
    - Summary: O(N x C_numeric) single-pass HashAggregate per partition.
    No shuffles are introduced; all operations are narrow transformations
    followed by a final driver-side reduce of partial aggregates.
    """
    dataset_name = os.path.basename(dataset_path)
    t0 = _time.perf_counter()

    print(f"\n{'─' * 72}")
    print(f"  DATASET:  {dataset_name}")
    print(f"  PATH:     {dataset_path}")
    print(f"{'─' * 72}")

    # ── 1. Ingest ───────────────────────────────────────────────────────
    # recursiveFileLookup=true is CRITICAL for:
    #   - icesat2_dynamic.parquet/ which contains sub-directories
    #     (step_015.parquet, step_016.parquet, ...)
    #   - Any Hive-partitioned layout where partition columns (month_idx)
    #     duplicate data columns → AnalysisException without this flag.
    try:
        df = (
            spark.read
            .option("recursiveFileLookup", "true")
            .option("mergeSchema", "true")
            .parquet(dataset_path)
        )
    except Exception as exc:
        print(f"  [ERROR] Failed to read {dataset_name}: {exc}")
        elapsed = _time.perf_counter() - t0
        print(f"  Elapsed: {elapsed:.2f} s (FAILED)\n")
        return elapsed

    # ── 2. Dimensions ───────────────────────────────────────────────────
    num_cols = len(df.columns)
    print(f"\n  Schema ({num_cols} columns):")
    print(f"  {'Column Name':<30s}  {'Data Type':<20s}")
    print(f"  {'─' * 30}  {'─' * 20}")
    for col_name, col_type in df.dtypes:
        print(f"  {col_name:<30s}  {col_type:<20s}")

    # Row count : triggers a single-pass scan.  On Parquet files with
    # intact footers, Spark can often resolve this from metadata alone.
    print(f"\n  Counting rows ...")
    try:
        row_count = df.count()
    except Exception as exc:
        print(f"  [ERROR] Row count failed: {exc}")
        elapsed = _time.perf_counter() - t0
        print(f"  Elapsed: {elapsed:.2f} s (PARTIAL)\n")
        return elapsed

    print(f"  Total rows:    {row_count:>14,}")
    print(f"  Total columns: {num_cols:>14,}")

    if row_count == 0:
        print(f"  [WARNING] Dataset is empty : skipping summary statistics.")
        elapsed = _time.perf_counter() - t0
        print(f"  Elapsed: {elapsed:.2f} s\n")
        return elapsed

    # ── 3. Isolate Numeric Columns ───────────────────────────────────
    # isinstance(field.dataType, NumericType) catches:
    #   IntegerType, LongType, ShortType, ByteType,
    #   FloatType, DoubleType, DecimalType
    # and correctly EXCLUDES:
    #   TimestampType, DateType, StringType, BooleanType, BinaryType
    # This prevents AnalysisException when .summary() tries to compute
    # mean/stddev on non-numeric columns.
    numeric_cols = [
        field.name
        for field in df.schema.fields
        if isinstance(field.dataType, NumericType)
    ]

    if not numeric_cols:
        print(f"  [WARNING] No numeric columns found : skipping summary.")
        elapsed = _time.perf_counter() - t0
        print(f"  Elapsed: {elapsed:.2f} s\n")
        return elapsed

    print(f"\n  Numeric columns ({len(numeric_cols)}):")
    print(f"    {numeric_cols}")

    # ── 4. Summary Statistics ────────────────────────────────────────
    # .summary() compiles to single-pass HashAggregates per partition.
    # Requesting only the stats we need avoids computing percentiles
    # (which WOULD trigger a shuffle for exact quantiles).
    #
    # The result is a small DataFrame with shape (5, len(numeric_cols)+1)
    # where column 0 = "summary" and remaining columns = stat values.
    # .show() pulls only these 5 rows to the driver : safe even with
    # 2 GB driver memory.
    print(f"\n  Computing summary statistics (min, max, mean, stddev) ...")

    df_numeric = df.select(numeric_cols)
    summary_df = df_numeric.summary("count", "min", "max", "mean", "stddev")

    print(f"\n  Summary Statistics for: {dataset_name}")
    summary_df.show(truncate=False, vertical=True)

    # ── 5. Elapsed Time ──────────────────────────────────────────────
    elapsed = _time.perf_counter() - t0
    print(f"  Elapsed: {elapsed:.2f} s\n")

    return elapsed


# ═══════════════════════════════════════════════════════════════════════════
# ██  SPEEDUP & EFFICIENCY REPORTING
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
        Total wall-clock time for the entire EDA run.

    Notes
    -----
    Speedup and Efficiency cannot be computed in a single run; they
    require comparing T₁ (1-executor baseline) with Tₙ (n-executor run).
    This function prints the measured Tₙ and the formulae for the user
    to complete after the baseline run.
    """
    print("\n" + "=" * 72)
    print("  PERFORMANCE REPORT")
    print("=" * 72)

    print(f"\n  {'Dataset':<35s}  {'Time (s)':>10s}")
    print(f"  {'─' * 35}  {'─' * 10}")
    for name, t in dataset_times.items():
        print(f"  {name:<35s}  {t:>10.2f}")
    print(f"  {'─' * 35}  {'─' * 10}")
    print(f"  {'TOTAL':<35s}  {total_wall_seconds:>10.2f}")

    print(f"\n  Executor configuration:")
    print(f"    n (executor instances)  = {EXECUTOR_INSTANCES}")
    print(f"    Tₙ (this run)          = {total_wall_seconds:.2f} s")

    print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  SPEEDUP & EFFICIENCY (requires a baseline run)                │
  │                                                                │
  │  1. Run this script with TOTAL_CORES = 2 (→ 1 executor)       │
  │     Record the total wall-clock time as T₁.                    │
  │                                                                │
  │  2. Run this script with TOTAL_CORES = {TOTAL_CORES:<3d} (→ {EXECUTOR_INSTANCES:<3d} executors)   │
  │     The total wall-clock time Tₙ = {total_wall_seconds:<8.2f} s (this run).    │
  │                                                                │
  │  3. Compute:                                                   │
  │       Speedup    = T₁ / Tₙ                                    │
  │       Efficiency = Speedup / n                                 │
  │                  = T₁ / (Tₙ x n)                               │
  │                                                                │
  │  Perfect scaling → Speedup = n, Efficiency = 1.0               │
  │  Sub-linear scaling is expected due to:                        │
  │    - shuffle overhead (if any)                                 │
  │    - task scheduling latency                                   │
  │    - Parquet footer / metadata reads (serial on driver)        │
  └─────────────────────────────────────────────────────────────────┘
""")


# ═══════════════════════════════════════════════════════════════════════════
# ██  ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    """
    Orchestrate the full EDA pipeline.

    Returns
    -------
    int
        Exit code: 0 = success, 1 = partial failure, 2 = fatal error.
    """
    wall_start = _time.perf_counter()

    # ── Build SparkSession ──────────────────────────────────────────
    try:
        spark = build_spark_session()
    except Exception as exc:
        print(f"[FATAL] Failed to create SparkSession: {exc}")
        return 2

    # ── Discover Datasets ───────────────────────────────────────────
    try:
        datasets = discover_parquet_datasets(BASE_DATA_DIR)
    except FileNotFoundError as exc:
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

    # ── Run EDA Loop ────────────────────────────────────────────────
    dataset_times: dict[str, float] = {}
    n_failures = 0

    for dataset_path in datasets:
        name = os.path.basename(dataset_path)
        try:
            elapsed = run_eda_for_dataset(spark, dataset_path)
            dataset_times[name] = elapsed
        except Exception as exc:
            # Defensive: catch any unhandled exception so one bad
            # dataset never aborts the entire loop.
            print(f"  [ERROR] Unhandled exception for {name}: {exc}")
            dataset_times[name] = 0.0
            n_failures += 1

    # ── Performance Report ──────────────────────────────────────────
    total_wall = _time.perf_counter() - wall_start
    print_performance_report(dataset_times, total_wall)

    # ── Teardown ────────────────────────────────────────────────────
    spark.stop()

    if n_failures > 0:
        print(f"[WARNING] {n_failures} dataset(s) encountered errors.")
        return 1

    print("[SUCCESS] EDA complete for all datasets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
