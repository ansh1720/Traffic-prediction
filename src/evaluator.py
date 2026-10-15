"""
Evaluator Module.

Responsible for computing metrics, generating visualizations, and
comparing all trained models. This is Stage 6 of the pipeline.

Metrics computed:
    - Train accuracy, Test accuracy
    - Precision, Recall, F1 (per class)
    - Macro F1, Weighted F1
    - Confusion matrix

Visualizations generated:
    - Confusion matrices (per model)
    - Model comparison bar chart
    - Feature importance chart (for tree-based models)
    - Class distribution chart
    - Congestion by hour chart
    - Correlation heatmap

Functions:
    evaluate_model: Compute all metrics for one model.
    evaluate_all: Evaluate all trained models.
    compare_models: Build comparison table and save to CSV.
    plot_confusion_matrix: Generate and save confusion matrix plot.
    plot_model_comparison: Bar chart of all models' metrics.
    plot_feature_importance: Feature importance from tree models.
    plot_eda_charts: Generate all EDA visualizations.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# Set plot style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("viridis")


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Compute comprehensive evaluation metrics for a single model.

    Args:
        model: Fitted scikit-learn model.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        model_name (str): Name for display purposes.

    Returns:
        dict: All computed metrics.
    """
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "model_name": model_name,
        "train_accuracy": round(accuracy_score(y_train, y_pred_train), 4),
        "test_accuracy": round(accuracy_score(y_test, y_pred_test), 4),
        "precision_macro": round(precision_score(y_test, y_pred_test,
                                                  average="macro",
                                                  zero_division=0), 4),
        "recall_macro": round(recall_score(y_test, y_pred_test,
                                            average="macro",
                                            zero_division=0), 4),
        "macro_f1": round(f1_score(y_test, y_pred_test,
                                    average="macro",
                                    zero_division=0), 4),
        "weighted_f1": round(f1_score(y_test, y_pred_test,
                                       average="weighted",
                                       zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred_test),
        "y_pred_test": y_pred_test,
        "classification_report": classification_report(
            y_test, y_pred_test, zero_division=0
        ),
    }

    # Overfitting indicator
    metrics["overfit_gap"] = round(
        metrics["train_accuracy"] - metrics["test_accuracy"], 4
    )

    return metrics


def evaluate_all(trained_results, X_train, y_train, X_test, y_test):
    """
    Evaluate all trained models and print results.

    Args:
        trained_results (dict): Output from model_trainer.train_all().
        X_train, y_train, X_test, y_test: Data splits.

    Returns:
        dict: Model name -> metrics dict.
    """
    print("\n  STAGE 6: MODEL EVALUATION")
    print("  " + "-" * 40)

    all_metrics = {}

    for name, result in trained_results.items():
        model = result["model"]
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test, name)
        metrics["train_time"] = result["train_time"]
        all_metrics[name] = metrics

        print(f"\n  {name}:")
        print(f"    Train Acc: {metrics['train_accuracy']:.4f} | "
              f"Test Acc: {metrics['test_accuracy']:.4f} | "
              f"Gap: {metrics['overfit_gap']:.4f}")
        print(f"    Macro F1:  {metrics['macro_f1']:.4f} | "
              f"Weighted F1: {metrics['weighted_f1']:.4f}")

    return all_metrics


def compare_models(all_metrics, save_path=None):
    """
    Build a comparison DataFrame and save to CSV.

    Args:
        all_metrics (dict): Model name -> metrics dict.
        save_path (str): Path to save CSV. Defaults to config path.

    Returns:
        pd.DataFrame: Comparison table.
    """
    if save_path is None:
        save_path = config.MODEL_COMPARISON_PATH

    rows = []
    for name, m in all_metrics.items():
        rows.append({
            "Model": name,
            "Train Accuracy": m["train_accuracy"],
            "Test Accuracy": m["test_accuracy"],
            "Overfit Gap": m["overfit_gap"],
            "Precision (Macro)": m["precision_macro"],
            "Recall (Macro)": m["recall_macro"],
            "Macro F1": m["macro_f1"],
            "Weighted F1": m["weighted_f1"],
            "Train Time (s)": m.get("train_time", 0),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Weighted F1", ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"\n  Model Comparison Table:")
    print(df.to_string(index=False))
    print(f"\n  Saved to: {save_path}")

    return df


def plot_confusion_matrix(cm, class_names, model_name, save_dir=None):
    """
    Generate and save a confusion matrix heatmap.

    Args:
        cm (np.ndarray): Confusion matrix.
        class_names (list): Class label names.
        model_name (str): Model name for title and filename.
        save_dir (str): Directory to save the plot. Defaults to config.FIGURES_DIR.
    """
    if save_dir is None:
        save_dir = config.FIGURES_DIR
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={"label": "Count"})
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {filepath}")


def plot_all_confusion_matrices(all_metrics, label_encoder):
    """
    Generate confusion matrix plots for all models.

    Args:
        all_metrics (dict): Model name -> metrics dict.
        label_encoder: Fitted LabelEncoder with class names.
    """
    print("\n  -> Generating confusion matrices...")
    class_names = list(label_encoder.classes_)
    for name, metrics in all_metrics.items():
        plot_confusion_matrix(metrics["confusion_matrix"], class_names, name)


def plot_model_comparison(comparison_df, save_dir=None):
    """
    Generate a bar chart comparing all models' metrics.

    Args:
        comparison_df (pd.DataFrame): Model comparison table.
        save_dir (str): Directory to save the plot.
    """
    if save_dir is None:
        save_dir = config.FIGURES_DIR
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Accuracy comparison
    models = comparison_df["Model"]
    x = np.arange(len(models))
    width = 0.35

    axes[0].bar(x - width/2, comparison_df["Train Accuracy"],
                width, label="Train Accuracy", color="#2196F3", alpha=0.8)
    axes[0].bar(x + width/2, comparison_df["Test Accuracy"],
                width, label="Test Accuracy", color="#FF9800", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].set_title("Train vs Test Accuracy", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)

    # F1 comparison
    axes[1].bar(x - width/2, comparison_df["Macro F1"],
                width, label="Macro F1", color="#4CAF50", alpha=0.8)
    axes[1].bar(x + width/2, comparison_df["Weighted F1"],
                width, label="Weighted F1", color="#9C27B0", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    axes[1].set_ylabel("F1 Score", fontsize=12)
    axes[1].set_title("Macro vs Weighted F1", fontsize=14, fontweight="bold")
    axes[1].legend()
    axes[1].set_ylim(0, 1.05)

    plt.suptitle("Model Performance Comparison", fontsize=16,
                 fontweight="bold", y=1.02)
    plt.tight_layout()

    filepath = os.path.join(save_dir, "model_comparison.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {filepath}")


def plot_feature_importance(model, feature_names, model_name="Random Forest",
                           top_n=15, save_dir=None):
    """
    Plot feature importance for tree-based models.

    Args:
        model: Fitted tree-based model with feature_importances_.
        feature_names (list): Names of features.
        model_name (str): Model name for title.
        top_n (int): Number of top features to display.
        save_dir (str): Directory to save the plot.

    Returns:
        pd.DataFrame: Feature importance DataFrame.
    """
    if save_dir is None:
        save_dir = config.FIGURES_DIR
    os.makedirs(save_dir, exist_ok=True)

    if not hasattr(model, "feature_importances_"):
        print(f"    {model_name} does not support feature importance.")
        return None

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    # Take top N
    importance_df = importance_df.tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(importance_df["Feature"], importance_df["Importance"],
                   color=plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df))))
    ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}",
                fontsize=14, fontweight="bold")

    # Add value labels
    for bar, val in zip(bars, importance_df["Importance"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()

    filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {filepath}")

    return importance_df


def plot_eda_charts(df, save_dir=None):
    """
    Generate all EDA visualizations.

    Creates:
    1. Class distribution bar chart
    2. Congestion by hour of day
    3. Congestion by day of week
    4. Congestion by weather
    5. Correlation heatmap
    6. Speed/Demand distribution by road type
    7. Temperature vs congestion

    Args:
        df (pd.DataFrame): Raw or feature-engineered DataFrame.
        save_dir (str): Directory to save plots.
    """
    if save_dir is None:
        save_dir = config.FIGURES_DIR
    os.makedirs(save_dir, exist_ok=True)

    print("\n  EDA VISUALIZATIONS")
    print("  " + "-" * 40)

    # Handle case-insensitive columns
    weather_col = "Weather" if "Weather" in df.columns else "weather"
    road_type_col = "RoadType" if "RoadType" in df.columns else "road_type"
    temp_col = "Temperature" if "Temperature" in df.columns else "temperature"

    # 1. Class Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Low": "#4CAF50", "Medium": "#FF9800", "High": "#F44336"}
    counts = df["congestion_level"].value_counts()
    bars = ax.bar(counts.index, counts.values,
                  color=[colors.get(c, "#999") for c in counts.index],
                  edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{val}\n({val/len(df)*100:.1f}%)",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Congestion Level", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Target Variable Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "class_distribution.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved class_distribution.png")

    # 2. Congestion by Hour
    fig, ax = plt.subplots(figsize=(12, 5))
    hour_cong = df.groupby(["hour", "congestion_level"]).size().unstack(
        fill_value=0)
    for cls in ["Low", "Medium", "High"]:
        if cls in hour_cong.columns:
            ax.plot(hour_cong.index, hour_cong[cls], marker="o",
                    linewidth=2, label=cls, color=colors.get(cls, "#999"))
    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel("Number of Records", fontsize=12)
    ax.set_title("Congestion Levels by Hour of Day", fontsize=14,
                fontweight="bold")
    ax.set_xticks(range(24))
    ax.legend(title="Congestion", fontsize=10)
    ax.axvspan(7, 9, alpha=0.1, color="red", label="Morning Rush")
    ax.axvspan(17, 19, alpha=0.1, color="red", label="Evening Rush")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "congestion_by_hour.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved congestion_by_hour.png")

    # 3. Congestion by Day of Week
    fig, ax = plt.subplots(figsize=(10, 5))
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_cong = df.groupby(["day_of_week", "congestion_level"]).size().unstack(
        fill_value=0)
    day_cong.plot(kind="bar", ax=ax, color=[colors.get(c, "#999")
                  for c in day_cong.columns], edgecolor="white")
    
    # Map index integers to short day names dynamically
    labels = [day_names[int(i)] for i in day_cong.index]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, fontsize=10)
    ax.set_xlabel("Day of Week", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Congestion by Day of Week", fontsize=14, fontweight="bold")
    ax.legend(title="Congestion")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "congestion_by_day.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved congestion_by_day.png")

    # 4. Congestion by Weather
    fig, ax = plt.subplots(figsize=(10, 5))
    weather_cong = pd.crosstab(df[weather_col], df["congestion_level"],
                                normalize="index") * 100
    weather_cong = weather_cong.reindex(columns=["Low", "Medium", "High"],
                                         fill_value=0)
    weather_cong.plot(kind="bar", stacked=True, ax=ax,
                      color=[colors.get(c, "#999")
                             for c in ["Low", "Medium", "High"]],
                      edgecolor="white")
    ax.set_xlabel("Weather Condition", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Congestion Distribution by Weather", fontsize=14,
                fontweight="bold")
    ax.legend(title="Congestion", loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "congestion_by_weather.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved congestion_by_weather.png")

    # 5. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5,
                cbar_kws={"label": "Correlation Coefficient"})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "correlation_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved correlation_heatmap.png")

    # 6. Speed or Demand Distribution by Road Type
    if "avg_speed_kmh" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        road_types = [r for r in df[road_type_col].unique() if pd.notna(r)]
        for rt in sorted(road_types):
            subset = df[df[road_type_col] == rt]["avg_speed_kmh"]
            ax.hist(subset, bins=30, alpha=0.5, label=str(rt), edgecolor="white")
        ax.set_xlabel("Average Speed (km/h)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Speed Distribution by Road Type", fontsize=14,
                    fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "speed_distribution.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  -> Saved speed_distribution.png")
    elif "demand" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        road_types = [r for r in df[road_type_col].unique() if pd.notna(r)]
        for rt in sorted(road_types):
            subset = df[df[road_type_col] == rt]["demand"]
            ax.hist(subset, bins=30, alpha=0.5, label=str(rt), edgecolor="white")
        ax.set_xlabel("Traffic Demand", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Demand Distribution by Road Type", fontsize=14,
                    fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "speed_distribution.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  -> Saved demand distribution as speed_distribution.png")

    # 7. Temperature vs Speed/Demand colored by congestion
    if temp_col in df.columns and "avg_speed_kmh" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sample = df.dropna(subset=[temp_col]).sample(
            min(2000, len(df)), random_state=config.RANDOM_SEED)
        for cls in ["Low", "Medium", "High"]:
            subset = sample[sample["congestion_level"] == cls]
            ax.scatter(subset[temp_col], subset["avg_speed_kmh"],
                      alpha=0.4, label=cls, color=colors.get(cls, "#999"),
                      s=20, edgecolors="none")
        ax.set_xlabel("Temperature (°C)", fontsize=12)
        ax.set_ylabel("Average Speed (km/h)", fontsize=12)
        ax.set_title("Temperature vs Speed by Congestion Level", fontsize=14,
                    fontweight="bold")
        ax.legend(title="Congestion")
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "temperature_vs_speed.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  -> Saved temperature_vs_speed.png")
    elif temp_col in df.columns and "demand" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sample = df.dropna(subset=[temp_col]).sample(
            min(2000, len(df)), random_state=config.RANDOM_SEED)
        for cls in ["Low", "Medium", "High"]:
            subset = sample[sample["congestion_level"] == cls]
            ax.scatter(subset[temp_col], subset["demand"],
                      alpha=0.4, label=cls, color=colors.get(cls, "#999"),
                      s=20, edgecolors="none")
        ax.set_xlabel("Temperature (°C)", fontsize=12)
        ax.set_ylabel("Traffic Demand", fontsize=12)
        ax.set_title("Temperature vs Demand by Congestion Level", fontsize=14,
                    fontweight="bold")
        ax.legend(title="Congestion")
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "temperature_vs_speed.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  -> Saved temperature vs demand as temperature_vs_speed.png")

    print(f"\n  -> All EDA charts saved to: {save_dir}")
