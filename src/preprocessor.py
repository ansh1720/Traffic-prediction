"""
Preprocessing Module.

Responsible for preparing the engineered DataFrame for model training.
This is Stage 4 of the pipeline. Handles null imputation, categorical
encoding, feature scaling, and stratified train-test splitting.

CRITICAL RULE: Split FIRST, then fit scaler on training data ONLY.
This prevents data leakage.

Functions:
    handle_missing_values: Impute nulls using median/mode strategy.
    encode_categoricals: One-hot encode nominal categorical features.
    scale_features: StandardScaler fitted on train set only.
    split_data: Stratified train-test split.
    preprocess_pipeline: Run the full preprocessing sequence.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def handle_missing_values(df):
    """
    Impute missing values using robust strategies.

    Numerical columns: median imputation (robust to outliers).
    Categorical columns: mode imputation (most frequent value).

    Args:
        df (pd.DataFrame): DataFrame potentially containing nulls.

    Returns:
        pd.DataFrame: DataFrame with no null values.
    """
    df = df.copy()

    # Numerical columns: median imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            n_nulls = df[col].isnull().sum()
            df[col] = df[col].fillna(median_val)
            print(f"    Imputed {n_nulls} nulls in '{col}' with median={median_val:.2f}")

    # Categorical columns: mode imputation
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            n_nulls = df[col].isnull().sum()
            df[col] = df[col].fillna(mode_val)
            print(f"    Imputed {n_nulls} nulls in '{col}' with mode='{mode_val}'")

    assert df.isnull().sum().sum() == 0, "Null values remain after imputation!"
    return df


def encode_categoricals(df, fit=True, encoders=None):
    df = df.copy()

    # Normalize column names to match config casing (handling case and underscores)
    col_mapping = {}
    for col in df.columns:
        col_clean = col.lower().replace("_", "").replace(" ", "")
        for cfg_col in config.CATEGORICAL_FEATURES:
            cfg_clean = cfg_col.lower().replace("_", "").replace(" ", "")
            if col_clean == cfg_clean and col != cfg_col:
                col_mapping[col] = cfg_col
    if col_mapping:
        df = df.rename(columns=col_mapping)

    # Normalize values for categorical columns if they exist
    if "Weather" in df.columns:
        df["Weather"] = df["Weather"].map(config.WEATHER_MAPPING).fillna(df["Weather"])
    if "RoadType" in df.columns:
        df["RoadType"] = df["RoadType"].map(config.ROAD_TYPE_MAPPING).fillna(df["RoadType"])

    # Cast to pandas Categorical with fixed config categories
    for col, categories in config.CATEGORICAL_CATEGORIES.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=categories)

    # Identify categorical columns present in the dataframe
    cat_cols = [c for c in config.CATEGORICAL_FEATURES if c in df.columns]

    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    return df, {}


def encode_target(y, fit=True, encoder=None):
    """
    Encode the target variable (congestion_level) to integers.

    Low=0, Medium=1, High=2 — or as fitted by LabelEncoder.

    Args:
        y (pd.Series): Target variable as strings.
        fit (bool): If True, fit a new LabelEncoder.
        encoder (LabelEncoder): Pre-fitted encoder for transform only.

    Returns:
        tuple: (encoded_y as numpy array, fitted LabelEncoder)
    """
    if fit:
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
    else:
        if encoder is None:
            raise ValueError("Must provide a fitted encoder when fit=False")
        y_encoded = encoder.transform(y)

    return y_encoded, encoder


def scale_features(X_train, X_test, fit=True, scaler=None):
    """
    Scale numerical features using StandardScaler.

    CRITICAL: Fit on X_train ONLY, then transform both X_train and X_test.
    This prevents data leakage from test set distribution.

    Args:
        X_train (np.ndarray or pd.DataFrame): Training features.
        X_test (np.ndarray or pd.DataFrame): Test features.
        fit (bool): If True, fit a new scaler on X_train.
        scaler (StandardScaler): Pre-fitted scaler for transform only.

    Returns:
        tuple: (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    if fit:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        if scaler is None:
            raise ValueError("Must provide a fitted scaler when fit=False")
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def split_data(df, target_column=None, test_size=None, random_seed=None):
    """
    Perform stratified train-test split.

    Stratification ensures each congestion class is proportionally
    represented in both train and test sets.

    Args:
        df (pd.DataFrame): Full DataFrame with features and target.
        target_column (str): Name of the target column.
        test_size (float): Fraction of data for test set.
        random_seed (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train_df, X_test_df, y_train, y_test)
    """
    if target_column is None:
        target_column = config.TARGET_COLUMN
    if test_size is None:
        test_size = config.TEST_SIZE
    if random_seed is None:
        random_seed = config.RANDOM_SEED

    # Separate features and target
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Remove non-feature columns
    cols_to_drop = ["timestamp", "avg_speed_kmh", "demand", "Index", "geohash", "day", "minute"]
    cols_to_drop = [c for c in cols_to_drop if c in X.columns]
    X = X.drop(columns=cols_to_drop)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y
    )

    print(f"    Train set: {X_train.shape[0]} samples")
    print(f"    Test set:  {X_test.shape[0]} samples")
    print(f"    Train class distribution:")
    for cls, count in y_train.value_counts().items():
        print(f"      {cls}: {count} ({count/len(y_train)*100:.1f}%)")

    return X_train, X_test, y_train, y_test


def preprocess_pipeline(df):
    """
    Run the full preprocessing pipeline.

    Sequence: handle nulls -> split -> encode categoricals ->
    encode target -> scale features -> save artifacts.

    Args:
        df (pd.DataFrame): Feature-engineered DataFrame.

    Returns:
        dict: Contains X_train, X_test, y_train, y_test (scaled/encoded),
              plus fitted scaler, label_encoder, and feature_names.
    """
    print("\n  STAGE 4: PREPROCESSING")
    print("  " + "-" * 40)

    # Step 1: Handle missing values
    print("  -> Handling missing values...")
    df = handle_missing_values(df)

    # Step 2: Split first (BEFORE scaling!)
    print("  -> Splitting data (stratified)...")
    X_train_df, X_test_df, y_train, y_test = split_data(df)

    # Step 3: Encode categoricals (on both splits consistently)
    print("  -> Encoding categorical features...")
    X_train_df, _ = encode_categoricals(X_train_df)
    X_test_df, _ = encode_categoricals(X_test_df)

    # Align columns (one-hot encoding may produce different columns)
    X_train_df, X_test_df = X_train_df.align(X_test_df, join="left",
                                              axis=1, fill_value=0)

    feature_names = list(X_train_df.columns)

    # Step 4: Encode target
    print("  -> Encoding target variable...")
    y_train_encoded, label_encoder = encode_target(y_train, fit=True)
    y_test_encoded, _ = encode_target(y_test, fit=False, encoder=label_encoder)
    print(f"    Classes: {list(label_encoder.classes_)}")

    # Step 5: Scale features (fit on train ONLY)
    print("  -> Scaling features (fit on train only)...")
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train_df.values, X_test_df.values
    )

    # Save scaler and encoder
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, config.SCALER_PATH)
    joblib.dump(label_encoder, config.LABEL_ENCODER_PATH)
    print(f"    Saved scaler to: {config.SCALER_PATH}")
    print(f"    Saved label encoder to: {config.LABEL_ENCODER_PATH}")

    # Save processed data
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    df.to_csv(config.PROCESSED_DATA_PATH, index=False)

    print(f"\n  -> Preprocessing complete. Feature count: {len(feature_names)}")

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train_encoded,
        "y_test": y_test_encoded,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        "y_train_raw": y_train,
        "y_test_raw": y_test,
    }
