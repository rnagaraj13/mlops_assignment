"""
Data preprocessing utilities for Heart Disease prediction.
Handles loading, cleaning, encoding, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal"
]
TARGET_COL = "target"

CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERICAL_COLS  = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def load_data(path: str) -> pd.DataFrame:
    """Load CSV from *path*, ensuring correct dtypes."""
    df = pd.read_csv(path)
    # Cast categoricals to int for sklearn
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(int)
    return df


def split_features_target(df: pd.DataFrame):
    """Return (X, y) from a dataframe."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


def get_train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split."""
    X, y = split_features_target(df)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def build_preprocessor() -> Pipeline:
    """Return a fitted-ready StandardScaler pipeline for numerical columns."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLS),
        ],
        remainder="passthrough"   # keeps categorical columns as-is
    )
    return preprocessor


def save_preprocessor(preprocessor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"Preprocessor saved to {path}")


def load_preprocessor(path: str):
    return joblib.load(path)
