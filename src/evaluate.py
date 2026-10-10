"""
Standalone evaluation script – loads the saved model and prints a full report.
Usage:  python src/evaluate.py --data data/heart.csv --model models/best_model.pkl
"""

import argparse
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, roc_auc_score)

from src.data_preprocessing import get_train_test, load_data


def evaluate(data_path: str, model_path: str):
    df = load_data(data_path)
    _, X_test, _, y_test = get_train_test(df)

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    print("=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(
        classification_report(
            y_test, y_pred, target_names=["No Disease", "Heart Disease"]
        )
    )

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print(f"\nROC-AUC:           {roc_auc_score(y_test, y_pred_prob):.4f}")
    print(f"Average Precision: {average_precision_score(y_test, y_pred_prob):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/heart.csv")
    parser.add_argument("--model", default="models/best_model.pkl")
    args = parser.parse_args()
    evaluate(args.data, args.model)
