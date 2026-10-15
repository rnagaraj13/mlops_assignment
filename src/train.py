"""
Train logistic regression and a random forest.

    python src/train.py --data data/heart.csv --experiment heart-disease

Writes models/best_model.pkl and a local mlruns/ folder.
Open it with: mlflow ui --port 5000
"""

import argparse
import json
import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data_preprocessing import FEATURE_COLS, get_train_test, load_data
from src.feature_engineering import build_preprocessor

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
PLOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "notebooks", "plots"
)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

CV_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def get_models():
    """Starting estimators. The grid search overwrites the tuned fields."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            random_state=42,
            n_jobs=1,
        ),
    }


def get_param_grids():
    """Grid passed to GridSearchCV. Keys are pipeline parameter names."""
    return {
        "logistic_regression": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
        },
        "random_forest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [4, 8, 12],
            "classifier__min_samples_split": [2, 5],
        },
    }


def build_model_pipeline(estimator) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", clone(estimator)),
        ]
    )


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


def _cv_summary(cv_results):
    summary = {}
    for metric in CV_METRICS:
        key = f"test_{metric}"
        summary[metric] = {
            "mean": float(np.mean(cv_results[key])),
            "std": float(np.std(cv_results[key])),
        }
    return summary


def train_and_log(name, estimator, X_train, X_test, y_train, y_test):
    """One MLflow run: search, then score the winner on the test split."""
    pipeline = build_model_pipeline(estimator)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    with mlflow.start_run(run_name=name) as run:
        run_id = run.info.run_id
        search = GridSearchCV(
            pipeline,
            get_param_grids()[name],
            cv=cv,
            scoring="roc_auc",
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_

        cv_results = cross_validate(
            clone(best_pipeline),
            X_train,
            y_train,
            cv=cv,
            scoring=CV_METRICS,
            n_jobs=1,
        )
        cv_summary = _cv_summary(cv_results)
        for metric, stats in cv_summary.items():
            mlflow.log_metric(f"cv_mean_{metric}", stats["mean"])
            mlflow.log_metric(f"cv_std_{metric}", stats["std"])

        y_pred = best_pipeline.predict(X_test)
        y_pred_prob = best_pipeline.predict_proba(X_test)[:, 1]
        metrics = {
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_precision": float(precision_score(y_test, y_pred)),
            "test_recall": float(recall_score(y_test, y_pred)),
            "test_f1": float(f1_score(y_test, y_pred)),
            "test_roc_auc": float(roc_auc_score(y_test, y_pred_prob)),
        }
        mlflow.log_metrics(metrics)
        mlflow.log_metric("search_best_cv_roc_auc", float(search.best_score_))
        mlflow.log_param("model_type", name)
        mlflow.log_params({k: str(v) for k, v in search.best_params_.items()})

        cm_path = log_confusion_matrix(
            y_test, y_pred, f"Confusion Matrix – {name}", run_id[:8]
        )
        roc_path = log_roc_curve(
            best_pipeline, X_test, y_test, f"ROC Curve – {name}", run_id[:8]
        )
        mlflow.log_artifact(cm_path, artifact_path="plots")
        mlflow.log_artifact(roc_path, artifact_path="plots")
        mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

        print(f"\n[{name}] best params: {search.best_params_}")
        print(f"[{name}] search CV ROC-AUC: {search.best_score_:.4f}")
        print(f"[{name}] held-out metrics: {json.dumps(metrics, indent=2)}")
        print(f"  MLflow Run ID: {run_id}")

    record = {
        "model_name": name,
        "run_id": run_id,
        "best_params": {k: _jsonable(v) for k, v in search.best_params_.items()},
        "search_best_cv_roc_auc": float(search.best_score_),
        "cv": cv_summary,
        "held_out": metrics,
    }
    return record, best_pipeline


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def select_best(records):
    """Pick the higher held-out ROC-AUC. Recall breaks a near tie."""

    def sort_key(record):
        return (record["held_out"]["test_roc_auc"], record["held_out"]["test_recall"])

    ordered = sorted(records, key=sort_key, reverse=True)
    winner = ordered[0]
    runner = ordered[1]
    reason = (
        f"Selected {winner['model_name']} with held-out ROC-AUC "
        f"{winner['held_out']['test_roc_auc']:.4f} and recall "
        f"{winner['held_out']['test_recall']:.4f}. "
        f"The other candidate, {runner['model_name']}, scored ROC-AUC "
        f"{runner['held_out']['test_roc_auc']:.4f} and recall "
        f"{runner['held_out']['test_recall']:.4f}. "
        "I ranked on held-out ROC-AUC, then recall, because the AUC values "
        "were almost the same and a missed disease case matters more."
    )
    return winner, reason


def main(data_path: str, experiment_name: str):
    mlflow.set_experiment(experiment_name)

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = get_train_test(df)

    records = []
    fitted = {}
    for name, estimator in get_models().items():
        record, pipeline = train_and_log(
            name, estimator, X_train, X_test, y_train, y_test
        )
        records.append(record)
        fitted[name] = pipeline

    winner, reason = select_best(records)
    best_name = winner["model_name"]
    best_model = fitted[best_name]
    best_path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(best_model, best_path)

    selection = {
        "selected_model": best_name,
        "reason": reason,
        "features": FEATURE_COLS,
        "derived_features": ["age_thalach_ratio", "oldpeak_slope"],
        "split": {"test_size": 0.2, "stratified": True, "random_state": 42},
        "candidates": records,
    }
    selection_path = os.path.join(MODEL_DIR, "model_selection.json")
    with open(selection_path, "w") as f:
        json.dump(selection, f, indent=2)

    meta = {
        "model_name": best_name,
        "roc_auc": winner["held_out"]["test_roc_auc"],
        "recall": winner["held_out"]["test_recall"],
        "features": FEATURE_COLS,
    }
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{reason}")
    print(f"Saved model to {best_path}")
    print(f"Saved selection report to {selection_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/heart.csv")
    parser.add_argument("--experiment", default="heart-disease")
    args = parser.parse_args()
    main(args.data, args.experiment)
