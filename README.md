# Bengaluru Smart Route Traffic Prediction & Journey Planner

An end-to-end Machine Learning pipeline and lightweight web application that predicts road congestion levels (Low / Medium / High) in Bengaluru, India.

## Project Overview

This project anticipates traffic congestion levels across road segments using easily accessible inputs (time of day, day of week, weather) and historical spatial properties mapped from the training dataset (`train.csv`).

The project features:
- Master ML orchestrator pipeline (Stage 1 to Stage 8).
- Decision Tree and Random Forest classifiers.
- Lightweight Flask API server with built-in geohash cache nearest-neighbor lookups.
- Single-page light-themed Leaflet.js route planner.

## Repository Structure

```
traffic/
├── app/
│   └── templates/
│       └── index.html       # Light-theme Leaflet.js Frontend
│   └── static/
│       ├── css/             # Local Leaflet stylesheet
│       └── js/              # Local Leaflet and Chart.js scripts
├── src/
│   ├── data_loader.py       # Data ingestion and schema audit
│   ├── feature_engineer.py  # Time extraction and target binning
│   ├── preprocessor.py      # Null imputation, scaling, and categorical encoding
│   ├── model_trainer.py     # Decision Tree and Random Forest training
│   ├── evaluator.py         # Performance reports and EDA visual figures
│   └── predictor.py         # Spatial geohash cache lookup and inference
├── models/                  # Serialized scaler.pkl, label_encoder.pkl, best_model.pkl
├── reports/
│   ├── figures/             # Auto-generated EDA and evaluation charts
│   └── model_comparison.csv # Tuning comparison metrics
├── tests/                   # Pytest unit tests verifying pipeline stages
├── app.py                   # Lightweight Flask API web server
├── config.py                # Centralized configurations and coordinates mapping
├── train.csv                # Historical training dataset
├── test.csv                 # Inference test dataset
└── run_pipeline.py          # Master orchestrator script
```

## Quickstart

### 1. Install Dependencies
Ensure you are using Python 3.9+.
```bash
pip install -r requirements.txt
```

### 2. Run the Full ML Pipeline
Run the orchestrator script to ingest data, generate EDA figures, split, train, tune, and save the classification models:
```bash
python run_pipeline.py
```
*Note: Evaluator figures and plots are exported to `reports/figures/`.*

### 3. Launch the Lightweight Frontend
Start the Flask web server:
```bash
python app.py
```
Then open http://127.0.0.1:5000 in your browser.

## Spatial-Temporal Infrastructure

### Bounding-Box Coordinate Scaling
The ML classifiers are trained using a dataset (`train.csv`) index-located in a bounding box centered near Latitude -5.35, Longitude 90.78. 
To enable real-world routing inside Bengaluru (centered near Latitude 12.97, Longitude 77.59), the server performs a linear coordinate transformation to shift and scale Bengaluru inputs into the dataset bounding box, decodes it as a 6-character geohash, and retrieves the closest geohash details from the cache.

### Model Performance
- Best Model: Tuned Decision Tree
- Performance: ~51.56% Weighted F1 (improving upon the baseline major-class predictor baseline of 51.3% accuracy).
- Features encoded dynamically using fixed category lists in `config.py` to align predictions and training sets.
