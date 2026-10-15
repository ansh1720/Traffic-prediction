"""
Model Trainer Module.

Responsible for defining, training, and tuning ML models. This covers
Stages 5 and 7 of the pipeline.

Models trained:
    1. Logistic Regression (baseline ML model)
    2. K-Nearest Neighbors
    3. Support Vector Machine (RBF kernel)
    4. Decision Tree
    5. Random Forest
    6. Gradient Boosting (bonus model)

Functions:
    get_models: Return a dictionary of model instances with default params.
    train_all: Train all models and return fitted objects.
    tune_model: Run GridSearchCV on a specific model.
    tune_top_models: Identify and tune the top 2 performing models.
"""

import sys
import os
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def get_models(random_seed=None):
    """
    Return a dictionary of ML model instances with default parameters.

    All models use the same random_state for reproducibility.
    class_weight='balanced' is used where supported to handle
    potential class imbalance.

    Args:
        random_seed (int): Random seed. Defaults to config.RANDOM_SEED.

    Returns:
        dict: Model name -> model instance.
    """
    if random_seed is None:
        random_seed = config.RANDOM_SEED

    models = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=random_seed,
            class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            random_state=random_seed,
            n_estimators=100,
            class_weight="balanced",
            n_jobs=-1
        ),
    }

    return models


def train_all(models, X_train, y_train):
    """
    Train all models and record training times.

    Args:
        models (dict): Model name -> model instance.
        X_train (np.ndarray): Scaled training features.
        y_train (np.ndarray): Encoded training labels.

    Returns:
        dict: Model name -> dict with 'model' (fitted) and 'train_time' (seconds).
    """
    results = {}

    print("\n  STAGE 5: MODEL TRAINING")
    print("  " + "-" * 40)

    for name, model in models.items():
        print(f"  -> Training {name}...", end=" ", flush=True)
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        print(f"Done ({elapsed:.2f}s)")

        results[name] = {
            "model": model,
            "train_time": round(elapsed, 3),
        }

    print(f"\n  -> All {len(models)} models trained successfully.")
    return results


def tune_model(model, param_grid, X_train, y_train,
               cv_folds=None, scoring=None, random_seed=None):
    """
    Run GridSearchCV on a single model with stratified cross-validation.

    Args:
        model: Scikit-learn model instance.
        param_grid (dict): Hyperparameter grid to search.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        cv_folds (int): Number of CV folds. Defaults to config.CV_FOLDS.
        scoring (str): Scoring metric. Defaults to config.CV_SCORING.
        random_seed (int): Random seed. Defaults to config.RANDOM_SEED.

    Returns:
        dict: Contains best_model, best_params, best_score, cv_results.
    """
    if cv_folds is None:
        cv_folds = config.CV_FOLDS
    if scoring is None:
        scoring = config.CV_SCORING
    if random_seed is None:
        random_seed = config.RANDOM_SEED

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                         random_state=random_seed)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0,
        refit=True
    )

    print(f"    Searching {len(_count_combinations(param_grid))} combinations "
          f"× {cv_folds} folds...")
    start = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"    Best score: {grid_search.best_score_:.4f}")
    print(f"    Best params: {grid_search.best_params_}")
    print(f"    Search time: {elapsed:.2f}s")

    return {
        "best_model": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "cv_results": grid_search.cv_results_,
        "tune_time": round(elapsed, 3),
    }


def _count_combinations(param_grid):
    """Count total parameter combinations in a grid."""
    from itertools import product
    keys = param_grid.keys()
    values = param_grid.values()
    return list(product(*values))


def tune_top_models(trained_results, evaluation_results,
                    X_train, y_train, top_n=2):
    """
    Identify and tune the top N models based on evaluation metrics.

    Uses weighted F1 score to rank models, then applies GridSearchCV
    to the top performers only (as recommended by the plan).

    Args:
        trained_results (dict): Output from train_all().
        evaluation_results (dict): Output from evaluator.evaluate_all().
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        top_n (int): Number of top models to tune.

    Returns:
        dict: Model name -> tuning results.
    """
    print("\n  STAGE 7: HYPERPARAMETER TUNING")
    print("  " + "-" * 40)

    # Rank models by weighted F1
    model_scores = {
        name: metrics["weighted_f1"]
        for name, metrics in evaluation_results.items()
    }
    ranked = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    print("  Model ranking by weighted F1:")
    for i, (name, score) in enumerate(ranked, 1):
        marker = " <- WILL TUNE" if i <= top_n else ""
        print(f"    {i}. {name}: {score:.4f}{marker}")

    # Define param grids for tunable models
    param_grids = {
        "Decision Tree": config.DT_PARAM_GRID,
        "Random Forest": config.RF_PARAM_GRID,
    }

    tuned_results = {}

    for i, (name, score) in enumerate(ranked[:top_n]):
        print(f"\n  -> Tuning {name}...")

        if name in param_grids:
            grid = param_grids[name]
        else:
            # Generic light tuning for other models
            print(f"    No predefined grid for {name}. Skipping tuning.")
            tuned_results[name] = {
                "best_model": trained_results[name]["model"],
                "best_params": "default",
                "best_score": score,
                "tune_time": 0,
            }
            continue

        base_model = type(trained_results[name]["model"])(
            random_state=config.RANDOM_SEED,
            class_weight="balanced" if hasattr(
                trained_results[name]["model"], "class_weight"
            ) else None
        )

        result = tune_model(base_model, grid, X_train, y_train)
        tuned_results[name] = result

    return tuned_results
