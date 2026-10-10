"""
Feature engineering helpers for Heart Disease dataset.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

NUMERICAL_COLS  = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS


def build_preprocessor():
    """
    Build a ColumnTransformer that:
      - StandardScales numerical features
      - OneHotEncodes categorical features (drop first to avoid multicollinearity)
    Returns an unfitted sklearn transformer.
    """
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ]
    )
    return preprocessor


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-inspired interaction / derived features."""
    df = df.copy()
    # Age × max heart rate achieved — lower thalach relative to age is suspicious
    df["age_thalach_ratio"] = df["age"] / (df["thalach"] + 1e-6)
    # ST depression normalised by slope
    df["oldpeak_slope"] = df["oldpeak"] * df["slope"]
    return df
