
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import NumericType

def main():
    # ---------------------------------------------------------
    # 1. Initialize and Configure Spark
    # ---------------------------------------------------------
    spark = (SparkSession.builder
        .appName("Antarctica_EDA_Production")
        # Memory Tuning for 40GB Scale
        .config("spark.executor.memory", "8g")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.files.maxPartitionBytes", "134217728") # 128MB partitions
        # Adaptive Query Execution (AQE) for optimal post-shuffle coalescing
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        # Vectorized reader is safe to use now that we bypassed partition discovery
        .config("spark.sql.parquet.enableVectorizedReader", "true")
        .getOrCreate())

    spark.sparkContext.setLogLevel("ERROR")

    # ---------------------------------------------------------
    # 2. Implement File Path Toggling
    # ---------------------------------------------------------
    USE_PROTOTYPE = True  # Toggle for testing

    prototype_path = "ml_subset_amundsen_sea.parquet"
    production_path = "antarctica_sparse_features.parquet"

    target_file = prototype_path if USE_PROTOTYPE else production_path
    print(f"[*] Initializing EDA on target dataset: {target_file}\n")

    # ---------------------------------------------------------
    # 3. Ingestion & Schema Anomaly Handling
    # ---------------------------------------------------------
    try:
        # recursiveFileLookup bypasses Hive partition discovery.
        # This prevents the `month_idx` directory name from being appended to 
        # the schema, cleanly resolving the duplicate column crash.
        df = spark.read.option("recursiveFileLookup", "true").parquet(target_file)
    except Exception as e:
        print(f"[ERROR] Failed to read Parquet dataset. Exception: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 4. Calculate Dimensions and Schema
    # ---------------------------------------------------------
    # Row count pushed down to Parquet footer metadata
    total_rows = df.count()
    total_cols = len(df.columns)

    print("-" * 40)
    print(" DATASET DIMENSIONS")
    print("-" * 40)
    print(f"Total Rows:    {total_rows:,}")
    print(f"Total Columns: {total_cols}\n")

    print("-" * 40)
    print(" DATASET SCHEMA")
    print("-" * 40)
    for field in df.schema.fields:
        print(f" - {field.name.ljust(25)} : {field.dataType.typeName()}")
    print("\n")

    # ---------------------------------------------------------
    # 5. Compute Summary Statistics
    # ---------------------------------------------------------
    # Isolate numeric columns to prevent cast errors on timestamps
    numeric_columns = [
        field.name for field in df.schema.fields 
        if isinstance(field.dataType, NumericType)
    ]

    print("-" * 40)
    print(" SUMMARY STATISTICS (Single Pass)")
    print("-" * 40)
    
    # .summary() uses HashAggregates to process the 40GB data in exactly one pass
    stats_df = df.select(*numeric_columns).summary("min", "max", "mean", "stddev")
    
    # Vertical display for wide schemas
    stats_df.show(truncate=False, vertical=True)

    spark.stop()

if __name__ == "__main__":
    main()



"""
import gdown

# Download a file
file_id = "1ZPQsUA5ZuU22vv7XOp0bQytBlyb4FidR"
file_id = "1rjKIbz5zIbW3T2wwNOW4LuHokX9KMEp5"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
output_path = "COMPREHENSIVE_EDA_AND_PREPROCESSING.md"
gdown.download(url, output_path, quiet=False)

# Download a folder (returns a list of downloaded file paths)
#folder_url = "https://drive.google.com"
#file_paths = gdown.download_folder(folder_url, quiet=False)
"""


