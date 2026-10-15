"""
Configuration file for Traffic Congestion Prediction project.

All constants, file paths, thresholds, and model parameters are
centralized here to ensure consistency across all modules and
reproducibility of results.
"""

import os

# =============================================================================
# RANDOM SEED — Used everywhere for reproducibility
# =============================================================================
RANDOM_SEED = 42

# =============================================================================
# PROJECT PATHS
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Raw and processed data file paths
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "train.csv")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "traffic_processed.csv")

# Model persistence directory
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

# Reports and figures
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
MODEL_COMPARISON_PATH = os.path.join(REPORTS_DIR, "model_comparison.csv")

# =============================================================================
# DATASET PARAMETERS
# =============================================================================
# Number of synthetic records (legacy, kept for API compatibility)
N_SAMPLES = 12000

# Train-test split ratio (80% train, 20% test)
TEST_SIZE = 0.20

# =============================================================================
# CONGESTION THRESHOLDS FOR DEMAND BINNING
# =============================================================================
# These thresholds define how continuous traffic demand [0.0, 1.0] is binned into
# congestion classes:
#   - Low congestion: demand < 0.05
#   - Medium congestion: 0.05 <= demand < 0.15
#   - High congestion: demand >= 0.15
DEMAND_THRESHOLD_LOW = 0.05
DEMAND_THRESHOLD_HIGH = 0.15

# Target class labels (used consistently across the project)
CONGESTION_CLASSES = ["Low", "Medium", "High"]

# =============================================================================
# PEAK HOUR DEFINITIONS
# =============================================================================
# Morning rush: 7–9 AM, Evening rush: 5–7 PM
# These hours are flagged as peak hours for the is_peak_hour feature.
PEAK_HOURS = {7, 8, 9, 17, 18, 19}

# =============================================================================
# FEATURE LISTS
# =============================================================================
# Categorical features that require encoding
CATEGORICAL_FEATURES = ["Weather", "RoadType", "LargeVehicles", "Landmarks",
                        "day_of_week_name", "time_of_day_bucket"]

# Fixed categories to ensure identical dummy columns during training & prediction
CATEGORICAL_CATEGORIES = {
    "Weather": ["Sunny", "Rainy", "Foggy", "Snowy"],
    "RoadType": ["Street", "Residential"],
    "LargeVehicles": ["Allowed", "Not Allowed"],
    "Landmarks": ["Yes", "No"],
    "day_of_week_name": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "time_of_day_bucket": ["Night", "Morning", "Afternoon", "Evening"]
}

# Value mapping from API response strings to dataset strings
WEATHER_MAPPING = {
    "Clear": "Sunny",
    "Cloudy": "Sunny",
    "Rain": "Rainy",
    "Fog": "Foggy",
    "Snow": "Snowy",
    "Sunny": "Sunny",
    "Rainy": "Rainy",
    "Foggy": "Foggy",
    "Snowy": "Snowy"
}

ROAD_TYPE_MAPPING = {
    "Highway": "Street",
    "Arterial": "Street",
    "Collector": "Street",
    "Local": "Residential",
    "Street": "Street",
    "Residential": "Residential"
}

# Binary features (already 0/1, no encoding needed)
BINARY_FEATURES = ["is_weekend", "is_peak_hour", "is_holiday",
                   "incident_nearby"]

# Numerical features that require scaling
NUMERICAL_FEATURES = ["hour", "day_of_week", "month", "Temperature",
                      "NumberofLanes"]

# Interaction features (created during feature engineering)
INTERACTION_FEATURES = ["peak_x_rain", "peak_x_incident",
                        "weekend_x_weather_severity"]

# Target column name
TARGET_COLUMN = "congestion_level"

# =============================================================================
# MODEL HYPERPARAMETER GRIDS (for GridSearchCV)
# =============================================================================
# Decision Tree tuning grid
DT_PARAM_GRID = {
    "max_depth": [5, 10],
    "min_samples_split": [10],
    "min_samples_leaf": [4],
    "criterion": ["entropy"],
}

# Random Forest tuning grid
RF_PARAM_GRID = {
    "n_estimators": [50],
    "max_depth": [10],
    "min_samples_split": [10],
    "max_features": ["sqrt"],
}

# Cross-validation folds for GridSearchCV
CV_FOLDS = 3

# Scoring metric for GridSearchCV
CV_SCORING = "f1_weighted"

# =============================================================================
# BENGALURU & DATASET SPATIAL MAPPING
# =============================================================================
# Bounding box for Bengaluru city
BENGALURU_LAT_MIN = 12.85
BENGALURU_LAT_MAX = 13.10
BENGALURU_LON_MIN = 77.45
BENGALURU_LON_MAX = 77.75

# Bounding box for the geohash locations in train.csv
DATASET_LAT_MIN = -5.485
DATASET_LAT_MAX = -5.237
DATASET_LON_MIN = 90.587
DATASET_LON_MAX = 90.972

# =============================================================================
# API CONFIGURATIONS
# =============================================================================
NOMINATIM_API_URL = "https://nominatim.openstreetmap.org/search"
OSRM_API_URL = "http://router.project-osrm.org/route/v1/driving"
OPEN_METEO_API_URL = "https://api.open-meteo.com/v1/forecast"
API_USER_AGENT = "TrafficCongestionPredictor/1.0 (contact: intern@example.com)"

# =============================================================================
# WEATHER SEVERITY AND ROAD CONFIGURATIONS
# =============================================================================
# Weather types and their probabilities (for legacy generators)
WEATHER_TYPES = ["Clear", "Cloudy", "Rain", "Fog", "Snow"]
WEATHER_PROBS = [0.40, 0.25, 0.20, 0.10, 0.05]

# Road types and their probabilities
ROAD_TYPES = ["Highway", "Arterial", "Local", "Collector"]
ROAD_PROBS = [0.30, 0.35, 0.20, 0.15]

# Weather severity mapping (used for interaction features)
WEATHER_SEVERITY = {
    "Clear": 0,
    "Sunny": 0,
    "Cloudy": 1,
    "Rain": 2,
    "Rainy": 2,
    "Fog": 3,
    "Foggy": 3,
    "Snow": 4,
    "Snowy": 4,
}

# Nominal road speed limits (km/h) for routing time calculation
ROAD_SPEED_LIMITS = {
    "Highway": 90.0,
    "Arterial": 50.0,
    "Collector": 40.0,
    "Local": 30.0,
    "Street": 40.0,
    "Residential": 30.0
}

# Congestion speed multipliers (factor to multiply duration by based on congestion class)
CONGESTION_TIME_MULTIPLIERS = {
    "Low": 1.0,      # Free flow speed
    "Medium": 1.6,   # Speed reduced to ~60%
    "High": 2.5      # Speed reduced to ~40% (gridlock)
}
