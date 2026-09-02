import pandas as pd

# Load the dataset
# Note: make sure you have 'fastparquet' or 'pyarrow' installed to read parquet files
file_path = 'data/antarctica_sparse_features_sample.parquet'


df = pd.read_parquet(file_path)

print("--- Dataset Overview ---")
print(f"Total Rows: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}")

print("\n--- Column List ---")
print(df.columns.tolist())

print("\n--- Data Types & Missing Values ---")
print(df.info())

print("\n--- Summary Statistics (Numerical) ---")
print(df.describe())

print("\n--- First 5 Rows ---")
print(df.head())

