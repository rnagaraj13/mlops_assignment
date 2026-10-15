"""Load the Cleveland CSV and make the train/test split."""

import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET_COL = "target"

CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERICAL_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def load_data(path: str) -> pd.DataFrame:
    """Load the CSV and cast the category columns to int."""
    df = pd.read_csv(path)
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
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
