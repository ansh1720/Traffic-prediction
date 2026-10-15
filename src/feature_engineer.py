"""
Feature Engineering Module.

Responsible for creating derived features from raw data that encode
domain knowledge about traffic patterns. This is Stage 3 of the pipeline.

Compatible with both synthetic traffic data and the historical train.csv dataset.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def extract_time_features(df):
    """
    Extract temporal features from timestamp/day columns.
    Handles both standard datetime timestamps and H:M formatted timestamp strings.
    """
    df = df.copy()

    # Determine columns present
    has_timestamp = "timestamp" in df.columns
    has_day = "day" in df.columns

    # Parse day of week and related features from day column (if present, e.g. in train.csv)
    if has_day:
        if "day_of_week" not in df.columns:
            df["day_of_week"] = df["day"] % 7
        if "day_of_week_name" not in df.columns:
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            df["day_of_week_name"] = df["day_of_week"].map(lambda x: day_names[int(x)] if not pd.isna(x) else "Monday")
        if "is_weekend" not in df.columns:
            df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        if "month" not in df.columns:
            # Approximate month from day count
            df["month"] = (df["day"] // 30) % 12 + 1
    
    # Parse hour and other temporal components from timestamp
    if has_timestamp:
        # Check if it is a simple H:M time string (no spaces/dashes, has a colon)
        sample = str(df["timestamp"].iloc[0])
        is_time_str = False
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            is_time_str = ":" in sample and " " not in sample and "-" not in sample

        if is_time_str:
            # H:M string format
            parts = df["timestamp"].astype(str).str.split(":")
            df["hour"] = pd.to_numeric(parts.str[0], errors="coerce").fillna(0).astype(int)
            df["minute"] = pd.to_numeric(parts.str[1], errors="coerce").fillna(0).astype(int)
        else:
            # Datetime format
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            df["hour"] = ts.dt.hour.fillna(0).astype(int)
            if "day_of_week" not in df.columns:
                df["day_of_week"] = ts.dt.dayofweek.fillna(0).astype(int)
            if "day_of_week_name" not in df.columns:
                df["day_of_week_name"] = ts.dt.day_name()
            if "is_weekend" not in df.columns:
                df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
            if "month" not in df.columns:
                df["month"] = ts.dt.month.fillna(1).astype(int)

        # Common derived fields
        if "is_peak_hour" not in df.columns:
            df["is_peak_hour"] = df["hour"].isin(config.PEAK_HOURS).astype(int)
        
        if "time_of_day_bucket" not in df.columns:
            df["time_of_day_bucket"] = pd.cut(
                df["hour"],
                bins=[-1, 5, 11, 16, 20, 24],
                labels=["Night", "Morning", "Afternoon", "Evening", "Night_late"]
            ).astype(str)
            df["time_of_day_bucket"] = df["time_of_day_bucket"].replace(
                "Night_late", "Night"
            )

    # If is_holiday is not present, default to 0
    if "is_holiday" not in df.columns:
        df["is_holiday"] = 0

    return df


def add_weather_severity(df):
    """
    Map weather categories to numerical severity scores.
    """
    df = df.copy()
    col = "Weather" if "Weather" in df.columns else "weather"

    if col in df.columns:
        df["weather_severity"] = df[col].map(config.WEATHER_SEVERITY)
        # Handle unmapped values
        df["weather_severity"] = df["weather_severity"].fillna(0).astype(int)
    else:
        df["weather_severity"] = 0

    return df


def create_interaction_features(df):
    """
    Create interaction features that capture compound effects.
    """
    df = df.copy()
    weather_col = "Weather" if "Weather" in df.columns else "weather"

    # Simulate incidents for training if not present (since train.csv doesn't have incidents)
    if "incident_nearby" not in df.columns:
        np.random.seed(config.RANDOM_SEED)
        prob = np.where(df["is_peak_hour"] == 1, 0.08, 0.03)
        df["incident_nearby"] = (np.random.random(len(df)) < prob).astype(int)

    # Peak hour × rain
    if "is_peak_hour" in df.columns and weather_col in df.columns:
        is_rain = df[weather_col].astype(str).str.contains("Rain|Rainy", case=False, regex=True).astype(int)
        df["peak_x_rain"] = df["is_peak_hour"] * is_rain

    # Peak hour × incident
    if "is_peak_hour" in df.columns and "incident_nearby" in df.columns:
        df["peak_x_incident"] = df["is_peak_hour"] * df["incident_nearby"]

    # Weekend × weather severity
    if "is_weekend" in df.columns and "weather_severity" in df.columns:
        df["weekend_x_weather_severity"] = (
            df["is_weekend"] * df["weather_severity"]
        )

    return df


def bin_target_variable(df, demand_column="demand", speed_column="avg_speed_kmh"):
    """
    Bin continuous values into congestion level classes.
    Supports either continuous speed (synthetic) or continuous demand (train.csv).
    """
    df = df.copy()

    if demand_column in df.columns:
        df["congestion_level"] = np.where(
            df[demand_column] < config.DEMAND_THRESHOLD_LOW, "Low",
            np.where(
                df[demand_column] < config.DEMAND_THRESHOLD_HIGH, "Medium", "High"
            )
        )
    elif speed_column in df.columns:
        df["congestion_level"] = np.where(
            df[speed_column] < 20, "High",
            np.where(
                df[speed_column] < 50, "Medium", "Low"
            )
        )

    return df


def engineer_all_features(df):
    """
    Run the complete feature engineering pipeline.
    """
    print("  -> Extracting time features...")
    df = extract_time_features(df)

    print("  -> Adding weather severity...")
    df = add_weather_severity(df)

    print("  -> Creating interaction features...")
    df = create_interaction_features(df)

    # Bin target if demand/speed column exists but congestion_level does not
    if "congestion_level" not in df.columns:
        print("  -> Binning target variable...")
        df = bin_target_variable(df)

    print(f"  -> Feature engineering complete. Shape: {df.shape}")

    return df
