"""Derived columns and the scaler/encoder used by training and the API."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_preprocessing import CATEGORICAL_COLS, FEATURE_COLS, NUMERICAL_COLS

DERIVED_NUMERICAL_COLS = ["age_thalach_ratio", "oldpeak_slope"]
SCALED_COLS = NUMERICAL_COLS + DERIVED_NUMERICAL_COLS


class DerivedFeatures(BaseEstimator, TransformerMixin):
    """Add ratio features from the raw clinical columns."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if not isinstance(X, pd.DataFrame):
            df.columns = FEATURE_COLS
        df = df.copy()
        df["age_thalach_ratio"] = df["age"] / (df["thalach"] + 1e-6)
        df["oldpeak_slope"] = df["oldpeak"] * df["slope"]
        return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same derived columns as DerivedFeatures, for notebooks and tests."""
    return DerivedFeatures().transform(df)


def build_preprocessor() -> Pipeline:
    """Scaler plus one-hot encoder. Derived columns are added first."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), SCALED_COLS),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore", drop="first", sparse_output=False
                ),
                CATEGORICAL_COLS,
            ),
        ]
    )
    return Pipeline(
        [
            ("derived", DerivedFeatures()),
            ("columns", preprocessor),
        ]
    )
