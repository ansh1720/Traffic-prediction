"""
Predictor Module.

Responsible for loading the serialized model and scaler, then serving
predictions for new data. This is Stage 9 of the pipeline.

Supports:
1. Loading serialized scikit-learn models, scalers, and encoders.
2. In-memory caching and nearest-neighbor lookup of geohashes in train.csv.
3. Feature preparation and prediction for single feature dictionaries,
   batch dataframes, and coordinate-based road segments.
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# =============================================================================
# GEOHASH FEATURE LOOKUP & CACHE
# =============================================================================
_geohash_features_cache = None
_geohash_coords = None


def load_geohash_features():
    """
    Load and cache the unique geohashes and their static attributes from train.csv.
    Decodes the geohashes to coordinate points to enable nearest-neighbor lookup.
    """
    global _geohash_features_cache, _geohash_coords
    if _geohash_features_cache is not None:
        return _geohash_features_cache

    try:
        # Resolve train.csv path
        filepath = os.path.join(config.PROJECT_ROOT, "train.csv")
        if os.path.exists(filepath):
            print("  -> Loading historical geohash mapping from train.csv...")
            df = pd.read_csv(filepath)
            
            # Group by geohash and take first record's static road attributes
            grouped = df.groupby("geohash").first()
            _geohash_features_cache = {}
            _geohash_coords = []

            # Base32 alphabet for geohash decoding
            base32 = '0123456789bcdefghjkmnpqrstuvwxyz'

            def decode_geohash(geohash):
                """Decode a 6-character geohash into (lat, lon) coordinates."""
                lat_interval = (-90.0, 90.0)
                lon_interval = (-180.0, 180.0)
                is_even = True
                for char in geohash:
                    val = base32.index(char)
                    for mask in [16, 8, 4, 2, 1]:
                        if is_even:
                            mid = (lon_interval[0] + lon_interval[1]) / 2
                            if val & mask:
                                lon_interval = (mid, lon_interval[1])
                            else:
                                lon_interval = (lon_interval[0], mid)
                        else:
                            mid = (lat_interval[0] + lat_interval[1]) / 2
                            if val & mask:
                                lat_interval = (mid, lat_interval[1])
                            else:
                                lat_interval = (lat_interval[0], mid)
                        is_even = not is_even
                lat = (lat_interval[0] + lat_interval[1]) / 2
                lon = (lon_interval[0] + lon_interval[1]) / 2
                return lat, lon

            for gh, row in grouped.iterrows():
                lat_ds, lon_ds = decode_geohash(gh)
                _geohash_features_cache[gh] = {
                    "RoadType": row.get("RoadType", "Residential"),
                    "NumberofLanes": int(row.get("NumberofLanes", 2)) if pd.notna(row.get("NumberofLanes")) else 2,
                    "LargeVehicles": row.get("LargeVehicles", "Allowed"),
                    "Landmarks": row.get("Landmarks", "No")
                }
                _geohash_coords.append((gh, lat_ds, lon_ds))
            
            print(f"     Successfully cached {len(_geohash_features_cache)} geohashes.")
        else:
            print("  -> WARNING: train.csv not found! Using empty geohash cache.")
            _geohash_features_cache = {}
            _geohash_coords = []
    except Exception as e:
        print(f"  -> Error caching geohash features: {e}")
        _geohash_features_cache = {}
        _geohash_coords = []

    return _geohash_features_cache


def lookup_features_by_coord(lat, lon):
    """
    Finds the closest geohash in train.csv by translating Bengaluru coordinates
    and returns its static road properties.
    """
    load_geohash_features()

    if not _geohash_coords:
        return {
            "RoadType": "Residential",
            "NumberofLanes": 2,
            "LargeVehicles": "Allowed",
            "Landmarks": "No"
        }

    # Scale Bengaluru lat/lon into the dataset geohash bounding box
    from src.routing import translate_blr_to_dataset
    lat_ds, lon_ds = translate_blr_to_dataset(lat, lon)

    # Perform nearest-neighbor coordinate matching
    min_dist = float("inf")
    closest_gh = None
    for gh, gh_lat, gh_lon in _geohash_coords:
        dist = (lat_ds - gh_lat) ** 2 + (lon_ds - gh_lon) ** 2
        if dist < min_dist:
            min_dist = dist
            closest_gh = gh

    return _geohash_features_cache.get(closest_gh, {
        "RoadType": "Residential",
        "NumberofLanes": 2,
        "LargeVehicles": "Allowed",
        "Landmarks": "No"
    })


# =============================================================================
# ARTIFACT LOADING FUNCTIONS
# =============================================================================
def load_model(model_path=None):
    """Load serialized model from disk."""
    if model_path is None:
        model_path = config.BEST_MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run the training pipeline first."
        )
    return joblib.load(model_path)


def load_scaler(scaler_path=None):
    """Load serialized scaler from disk."""
    if scaler_path is None:
        scaler_path = config.SCALER_PATH

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler file not found: {scaler_path}. Run the training pipeline first."
        )
    return joblib.load(scaler_path)


def load_label_encoder(encoder_path=None):
    """Load serialized label encoder from disk."""
    if encoder_path is None:
        encoder_path = config.LABEL_ENCODER_PATH

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(
            f"Label encoder not found: {encoder_path}. Run the training pipeline first."
        )
    return joblib.load(encoder_path)


# =============================================================================
# PREDICTION CORE FUNCTIONS
# =============================================================================
def prepare_input(feature_dict, feature_names, scaler):
    """
    Transform a raw feature dictionary into model-ready input.
    """
    # Normalize keys: check for lower-case alternatives and map to capitalized versions
    normalized_dict = {}
    key_mapping = {
        "weather": "Weather",
        "road_type": "RoadType",
        "temperature": "Temperature",
        "large_vehicles": "LargeVehicles",
        "largevehicles": "LargeVehicles",
        "landmarks": "Landmarks",
        "number_of_lanes": "NumberofLanes",
        "numberoflanes": "NumberofLanes"
    }
    for k, v in feature_dict.items():
        mapped_key = key_mapping.get(k.lower(), k)
        normalized_dict[mapped_key] = v

    # Set default values for missing features
    defaults = {
        "is_holiday": 0,
        "month": 6,
        "LargeVehicles": "Allowed",
        "Landmarks": "No",
        "NumberofLanes": 2,
        "incident_nearby": 0,
        "Temperature": 25.0,
        "Weather": "Sunny",
        "RoadType": "Residential"
    }
    for k, v in defaults.items():
        if k not in normalized_dict:
            normalized_dict[k] = v

    df = pd.DataFrame([normalized_dict])

    # Add weather severity
    weather_col = "Weather"
    df["weather_severity"] = df[weather_col].map(config.WEATHER_SEVERITY).fillna(0).astype(int)

    # Add interaction features
    is_rain = df[weather_col].astype(str).str.contains("Rain|Rainy", case=False, regex=True).astype(int)
    df["peak_x_rain"] = df["is_peak_hour"] * is_rain
    df["peak_x_incident"] = df["is_peak_hour"] * df["incident_nearby"]
    df["weekend_x_weather_severity"] = df["is_weekend"] * df["weather_severity"]

    # One-hot encode categoricals using the preprocessor's robust encoder
    from src.preprocessor import encode_categoricals
    df, _ = encode_categoricals(df)

    # Align columns to match training features
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Scale
    X = scaler.transform(df.values)

    return X


def predict_single(feature_dict, model=None, scaler=None,
                   label_encoder=None, feature_names=None):
    """
    Predict congestion level for a single record.
    """
    if model is None:
        model = load_model()
    if scaler is None:
        scaler = load_scaler()
    if label_encoder is None:
        label_encoder = load_label_encoder()
    if feature_names is None:
        feature_names = joblib.load(
            os.path.join(config.MODELS_DIR, "feature_names.pkl")
        )

    X = prepare_input(feature_dict, feature_names, scaler)

    # Predict class
    y_pred = model.predict(X)[0]
    prediction_label = label_encoder.inverse_transform([y_pred])[0]

    # Predict probabilities (if supported)
    probabilities = {}
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        for cls_idx, cls_name in enumerate(label_encoder.classes_):
            probabilities[cls_name] = round(float(proba[cls_idx]), 4)
        confidence = round(float(max(proba)), 4)

    return {
        "prediction": prediction_label,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def predict_batch(input_df, model=None, scaler=None,
                  label_encoder=None, feature_names=None):
    """
    Predict congestion levels for multiple records.
    """
    if model is None:
        model = load_model()
    if scaler is None:
        scaler = load_scaler()
    if label_encoder is None:
        label_encoder = load_label_encoder()
    if feature_names is None:
        feature_names = joblib.load(
            os.path.join(config.MODELS_DIR, "feature_names.pkl")
        )

    results = input_df.copy()
    predictions = []
    confidences = []

    for _, row in input_df.iterrows():
        result = predict_single(
            row.to_dict(), model, scaler, label_encoder, feature_names
        )
        predictions.append(result["prediction"])
        confidences.append(result["confidence"])

    results["predicted_congestion"] = predictions
    results["confidence"] = confidences

    return results
