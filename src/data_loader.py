"""
Data Loader Module.

Responsible for loading raw CSV data, validating its schema, and
performing a data quality audit. This is Stage 1 of the ML pipeline.

Functions:
    load_data: Load CSV file into a DataFrame with validation.
    validate_schema: Check expected columns and data types.
    audit_data: Print comprehensive data quality summary.
"""

import os
import pandas as pd


def load_data(filepath):
    """
    Load a CSV file into a Pandas DataFrame.

    Args:
        filepath (str): Absolute or relative path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or cannot be parsed.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)

    if df.empty:
        raise ValueError(f"Data file is empty: {filepath}")

    # Parse timestamp if present
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def validate_schema(df, required_columns=None):
    """
    Validate that the DataFrame contains expected columns.

    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
            If None, checks for a minimal set of expected columns.

    Returns:
        bool: True if validation passes.

    Raises:
        ValueError: If required columns are missing.
    """
    if required_columns is None:
        required_columns = [
            "day", "timestamp", "RoadType", "NumberofLanes",
            "LargeVehicles", "Landmarks", "Temperature", "Weather"
        ]

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    return True


def audit_data(df):
    """
    Perform a comprehensive data quality audit.

    Prints dataset shape, data types, null counts, duplicate counts,
    and target variable distribution. This is the first thing you run
    before any EDA or modeling.

    Args:
        df (pd.DataFrame): DataFrame to audit.

    Returns:
        dict: Audit results containing shape, nulls, duplicates, and
              target distribution.
    """
    audit_results = {}

    print("=" * 60)
    print("  DATA QUALITY AUDIT")
    print("=" * 60)

    # Shape
    print(f"\n1. SHAPE: {df.shape[0]} rows × {df.shape[1]} columns")
    audit_results["shape"] = df.shape

    # Data types
    print(f"\n2. DATA TYPES:")
    for col in df.columns:
        print(f"   {col:30s} -> {df[col].dtype}")
    audit_results["dtypes"] = df.dtypes.to_dict()

    # Null values
    null_counts = df.isnull().sum()
    null_pct = (df.isnull().sum() / len(df) * 100).round(2)
    print(f"\n3. NULL VALUES:")
    has_nulls = False
    for col in df.columns:
        if null_counts[col] > 0:
            has_nulls = True
            print(f"   {col:30s} -> {null_counts[col]:5d} nulls ({null_pct[col]:.2f}%)")
    if not has_nulls:
        print("   No null values found.")
    audit_results["null_counts"] = null_counts.to_dict()

    # Duplicates
    n_duplicates = df.duplicated().sum()
    print(f"\n4. DUPLICATE ROWS: {n_duplicates} ({n_duplicates/len(df)*100:.2f}%)")
    audit_results["n_duplicates"] = n_duplicates

    # Target variable distribution
    if "congestion_level" in df.columns:
        print(f"\n5. TARGET DISTRIBUTION (congestion_level):")
        dist = df["congestion_level"].value_counts()
        dist_pct = df["congestion_level"].value_counts(normalize=True) * 100
        for cls in dist.index:
            print(f"   {cls:10s} -> {dist[cls]:5d} ({dist_pct[cls]:.1f}%)")
        audit_results["target_distribution"] = dist.to_dict()

    # Basic statistics for numerical columns
    print(f"\n6. NUMERICAL STATISTICS:")
    numeric_cols = df.select_dtypes(include=["number"]).columns
    print(df[numeric_cols].describe().round(2).to_string())

    print("\n" + "=" * 60)

    return audit_results
