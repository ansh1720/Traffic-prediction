"""
Main Pipeline Orchestrator.

Runs the complete Traffic Congestion Prediction pipeline end-to-end:
    Stage 1: Data Ingestion (load and audit raw CSV)
    Stage 2: Exploratory Data Analysis (generate EDA visualizations)
    Stage 3: Feature Engineering (create derived features)
    Stage 4: Preprocessing (impute, encode, scale, split)
    Stage 5: Model Training (train 6 models with default parameters)
    Stage 6: Model Evaluation (compute metrics, generate plots)
    Stage 7: Hyperparameter Tuning (GridSearchCV on top 2 models)
    Stage 8: Model Persistence (save best model + scaler + encoder)
    Stage 9: Prediction Demo (sanity check predictions)

Usage:
    python run_pipeline.py
"""

import os
import sys
import time
import numpy as np
import joblib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_loader import load_data, validate_schema, audit_data
from src.feature_engineer import engineer_all_features
from src.preprocessor import preprocess_pipeline
from src.model_trainer import get_models, train_all, tune_top_models
from src.evaluator import (
    evaluate_all, compare_models, plot_all_confusion_matrices,
    plot_model_comparison, plot_feature_importance, plot_eda_charts
)
from src.predictor import predict_single


def run_pipeline():
    """Execute the full ML pipeline from data loading to prediction demo."""

    start_time = time.time()

    print("=" * 70)
    print("  TRAFFIC CONGESTION PREDICTION — FULL PIPELINE")
    print("=" * 70)

    # =========================================================================
    # STAGE 1: DATA INGESTION
    # =========================================================================
    print("\n" + "=" * 70)
    print("  STAGE 1: DATA INGESTION")
    print("=" * 70)

    # Check if raw data exists; if not, generate it
    if not os.path.exists(config.RAW_DATA_PATH):
        print("  Raw data not found. Generating synthetic data...")
        from generate_data import main as generate_main
        generate_main()

    df = load_data(config.RAW_DATA_PATH)
    validate_schema(df)
    audit_results = audit_data(df)

    # =========================================================================
    # STAGE 3: FEATURE ENGINEERING (Run before EDA for train.csv schemas)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  STAGE 3: FEATURE ENGINEERING")
    print("=" * 70)

    df = engineer_all_features(df)
    print(f"  Final columns ({len(df.columns)}): {list(df.columns)}")

    # Compute baseline accuracy (moved here since target is engineered in Stage 3)
    majority_class = df["congestion_level"].value_counts().idxmax()
    majority_pct = df["congestion_level"].value_counts(normalize=True).max()
    print(f"\n  BASELINE: Always predicting '{majority_class}' -> "
          f"{majority_pct*100:.1f}% accuracy")
    print(f"  Any model must beat this baseline to be useful.")

    # =========================================================================
    # STAGE 2: EXPLORATORY DATA ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("  STAGE 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    plot_eda_charts(df)

    # =========================================================================
    # STAGE 4: PREPROCESSING
    # =========================================================================
    print("\n" + "=" * 70)
    print("  STAGE 4: PREPROCESSING")
    print("=" * 70)

    prep_result = preprocess_pipeline(df)

    X_train = prep_result["X_train"]
    X_test = prep_result["X_test"]
    y_train = prep_result["y_train"]
    y_test = prep_result["y_test"]
    feature_names = prep_result["feature_names"]
    label_encoder = prep_result["label_encoder"]

    print(f"\n  Feature matrix shapes:")
    print(f"    X_train: {X_train.shape}")
    print(f"    X_test:  {X_test.shape}")

    # Save feature names for prediction
    joblib.dump(feature_names,
                os.path.join(config.MODELS_DIR, "feature_names.pkl"))

    # =========================================================================
    # STAGE 5: MODEL TRAINING (default parameters)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  STAGE 5: MODEL TRAINING")
    print("=" * 70)

    models = get_models()
    trained_results = train_all(models, X_train, y_train)

    # =========================================================================
    # STAGE 6: MODEL EVALUATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("  STAGE 6: MODEL EVALUATION")
    print("=" * 70)

    eval_results = evaluate_all(trained_results, X_train, y_train,
                                 X_test, y_test)

    # Print classification reports
    for name, metrics in eval_results.items():
        print(f"\n  {'='*50}")
        print(f"  Classification Report — {name}")
        print(f"  {'='*50}")
        print(metrics["classification_report"])

    # Generate confusion matrices
    plot_all_confusion_matrices(eval_results, label_encoder)

    # Build comparison table
    comparison_df = compare_models(eval_results)

    # Plot model comparison
    plot_model_comparison(comparison_df)

    # Feature importance for tree-based models
    for name in ["Random Forest", "Decision Tree", "Gradient Boosting"]:
        if name in trained_results:
            print(f"\n  -> Feature importance for {name}:")
            plot_feature_importance(
                trained_results[name]["model"],
                feature_names,
                model_name=name
            )

    # =========================================================================
    # STAGE 7: HYPERPARAMETER TUNING
    # =========================================================================
    print("\n" + "=" * 70)
    print("  STAGE 7: HYPERPARAMETER TUNING")
    print("=" * 70)

    tuned_results = tune_top_models(
        trained_results, eval_results, X_train, y_train
    )

    # Evaluate tuned models
    tuned_eval_results = {}
    for name, result in tuned_results.items():
        if "best_model" in result:
            from src.evaluator import evaluate_model
            metrics = evaluate_model(
                result["best_model"], X_train, y_train, X_test, y_test,
                f"{name} (Tuned)"
            )
            metrics["train_time"] = result.get("tune_time", 0)
            tuned_eval_results[f"{name} (Tuned)"] = metrics

            print(f"\n  {name} (Tuned):")
            print(f"    Test Acc: {metrics['test_accuracy']:.4f} | "
                  f"Weighted F1: {metrics['weighted_f1']:.4f}")
            print(f"    Best params: {result.get('best_params', 'N/A')}")

    # Add tuned results to comparison
    all_eval = {**eval_results, **tuned_eval_results}
    final_comparison = compare_models(all_eval)

    # Plot confusion matrices for tuned models
    plot_all_confusion_matrices(tuned_eval_results, label_encoder)

    # =========================================================================
    # STAGE 8: MODEL PERSISTENCE
    # =========================================================================
    print("\n" + "=" * 70)
    print("  STAGE 8: MODEL PERSISTENCE")
    print("=" * 70)

    # Select best model overall (highest weighted F1)
    best_name = final_comparison.iloc[0]["Model"]
    best_f1 = final_comparison.iloc[0]["Weighted F1"]
    print(f"\n  Best model: {best_name} (Weighted F1: {best_f1:.4f})")

    # Find the best model object
    if best_name in trained_results:
        best_model = trained_results[best_name]["model"]
    else:
        # It's a tuned model
        base_name = best_name.replace(" (Tuned)", "")
        best_model = tuned_results[base_name]["best_model"]

    # Save best model
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, config.BEST_MODEL_PATH)
    print(f"  Saved best model to: {config.BEST_MODEL_PATH}")

    # Feature importance for best model (if tree-based)
    if hasattr(best_model, "feature_importances_"):
        plot_feature_importance(best_model, feature_names,
                               model_name="Best Model")

    # =========================================================================
    # STAGE 9: PREDICTION DEMO (Sanity Checks)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  STAGE 9: PREDICTION DEMO")
    print("=" * 70)

    # Sanity check 1: Monday morning, rain, peak hour, incident -> expect High
    test_high = {
        "hour": 8,
        "day_of_week": 0,  # Monday
        "month": 6,
        "is_weekend": 0,
        "is_peak_hour": 1,
        "is_holiday": 0,
        "Weather": "Rainy",
        "Temperature": 25.0,
        "RoadType": "Highway",
        "LargeVehicles": "Not Allowed",
        "Landmarks": "No",
        "NumberofLanes": 2,
        "incident_nearby": 1,
        "day_of_week_name": "Monday",
        "time_of_day_bucket": "Morning",
    }

    result1 = predict_single(test_high, best_model,
                              prep_result["scaler"],
                              label_encoder, feature_names)
    print(f"\n  Sanity Check 1 (should be HIGH):")
    print(f"    Input: Mon 8AM, Rain, Highway, Incident=Yes")
    print(f"    Prediction: {result1['prediction']} "
          f"(Confidence: {result1['confidence']*100:.1f}%)")
    print(f"    Probabilities: {result1['probabilities']}")

    # Sanity check 2: Sunday 2 AM, clear, no incident -> expect Low
    test_low = {
        "hour": 2,
        "day_of_week": 6,  # Sunday
        "month": 6,
        "is_weekend": 1,
        "is_peak_hour": 0,
        "is_holiday": 0,
        "Weather": "Sunny",
        "Temperature": 20.0,
        "RoadType": "Residential",
        "LargeVehicles": "Allowed",
        "Landmarks": "Yes",
        "NumberofLanes": 1,
        "incident_nearby": 0,
        "day_of_week_name": "Sunday",
        "time_of_day_bucket": "Night",
    }

    result2 = predict_single(test_low, best_model,
                              prep_result["scaler"],
                              label_encoder, feature_names)
    print(f"\n  Sanity Check 2 (should be LOW):")
    print(f"    Input: Sun 2AM, Sunny, Residential, Incident=No")
    print(f"    Prediction: {result2['prediction']} "
          f"(Confidence: {result2['confidence']*100:.1f}%)")
    print(f"    Probabilities: {result2['probabilities']}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Total execution time: {elapsed:.1f} seconds")
    print(f"  Best model: {best_name} (Weighted F1: {best_f1:.4f})")
    print(f"  Models saved to: {config.MODELS_DIR}")
    print(f"  Reports saved to: {config.REPORTS_DIR}")
    print(f"  EDA figures saved to: {config.FIGURES_DIR}")
    print(f"\n  Next steps:")
    print(f"    1. Review EDA plots in reports/figures/")
    print(f"    2. Review model_comparison.csv in reports/")
    print(f"    3. Run: streamlit run app/streamlit_app.py")
    print("=" * 70)

    return {
        "best_model_name": best_name,
        "best_f1": best_f1,
        "comparison_df": final_comparison,
    }


if __name__ == "__main__":
    run_pipeline()
