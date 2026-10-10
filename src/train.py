"""
Model training script with MLflow experiment tracking.

Usage:
    python src/train.py --data data/heart.csv --experiment heart-disease

Trains Logistic Regression and Random Forest, logs everything to MLflow,
and saves the best model to models/best_model.pkl.
"""

import argparse
import json
import os
import sys

import joblib
import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data_preprocessing import (FEATURE_COLS, TARGET_COL, get_train_test,
                                    load_data)
from src.feature_engineering import FEATURE_COLS as FE_FEATURES
from src.feature_engineering import (add_interaction_features,
                                     build_preprocessor)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
PLOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "notebooks", "plots"
)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def get_models():
    """Return dict of {name: estimator} to compare."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),
    }


def log_confusion_matrix(y_true, y_pred, title, run_id):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(title)
    path = os.path.join(PLOT_DIR, f"cm_{run_id}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def log_roc_curve(estimator, X_test, y_test, title, run_id):
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(estimator, X_test, y_test, ax=ax)
    ax.set_title(title)
    path = os.path.join(PLOT_DIR, f"roc_{run_id}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def train_and_log(
    name, estimator, X_train, X_test, y_train, y_test, experiment_name: str
):
    """Train one model inside an MLflow run, return (run_id, test_metrics)."""

    with mlflow.start_run(run_name=name) as run:
        run_id = run.info.run_id

        # Build full pipeline: preprocessor → model
        preprocessor = build_preprocessor()
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )

        # ── Cross-validation ──────────────────────────────────────────────
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            return_train_score=False,
        )
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            key = f"test_{metric}"
            mlflow.log_metric(f"cv_mean_{metric}", float(np.mean(cv_results[key])))
            mlflow.log_metric(f"cv_std_{metric}", float(np.std(cv_results[key])))

        # ── Full train + test eval ────────────────────────────────────────
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred),
            "test_roc_auc": roc_auc_score(y_test, y_pred_prob),
        }
        mlflow.log_metrics(metrics)

        # Params
        params = estimator.get_params()
        params["model_type"] = name
        mlflow.log_params({k: str(v) for k, v in params.items()})

        # Artefacts
        cm_path = log_confusion_matrix(
            y_test, y_pred, f"Confusion Matrix – {name}", run_id[:8]
        )
        roc_path = log_roc_curve(
            pipeline, X_test, y_test, f"ROC Curve – {name}", run_id[:8]
        )
        mlflow.log_artifact(cm_path, artifact_path="plots")
        mlflow.log_artifact(roc_path, artifact_path="plots")

        # Model
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=f"heart_disease_{name}",
        )

        print(f"\n[{name}] Metrics: {json.dumps(metrics, indent=2)}")
        print(f"  MLflow Run ID: {run_id}")

    return run_id, metrics, pipeline


def main(data_path: str, experiment_name: str):
    mlflow.set_experiment(experiment_name)

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = get_train_test(df)

    best_model = None
    best_roc_auc = -1
    best_name = ""

    for name, estimator in get_models().items():
        run_id, metrics, pipeline = train_and_log(
            name, estimator, X_train, X_test, y_train, y_test, experiment_name
        )
        if metrics["test_roc_auc"] > best_roc_auc:
            best_roc_auc = metrics["test_roc_auc"]
            best_model = pipeline
            best_name = name

    # Persist best model
    best_path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(best_model, best_path)
    print(f"\nBest model: {best_name} (ROC-AUC={best_roc_auc:.4f})")
    print(f"Saved to {best_path}")

    # Save metadata alongside model
    meta = {"model_name": best_name, "roc_auc": best_roc_auc, "features": FEATURE_COLS}
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/heart.csv")
    parser.add_argument("--experiment", default="heart-disease")
    args = parser.parse_args()
    main(args.data, args.experiment)
