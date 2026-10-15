"""Checks for the CSV split."""

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    NUMERICAL_COLS,
    TARGET_COL,
    get_train_test,
    split_features_target,
)


@pytest.fixture
def sample_df():
    """Small synthetic DataFrame mimicking the heart dataset."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "age": rng.integers(30, 80, n).astype(float),
            "sex": rng.integers(0, 2, n),
            "cp": rng.integers(0, 4, n),
            "trestbps": rng.integers(90, 200, n).astype(float),
            "chol": rng.integers(150, 400, n).astype(float),
            "fbs": rng.integers(0, 2, n),
            "restecg": rng.integers(0, 3, n),
            "thalach": rng.integers(70, 200, n).astype(float),
            "exang": rng.integers(0, 2, n),
            "oldpeak": rng.uniform(0, 6, n),
            "slope": rng.integers(0, 3, n),
            "ca": rng.integers(0, 5, n),
            "thal": rng.integers(0, 4, n),
            "target": rng.integers(0, 2, n),
        }
    )
    return df


class TestFeatureTargetSplit:
    def test_columns_present(self, sample_df):
        X, y = split_features_target(sample_df)
        assert list(X.columns) == FEATURE_COLS
        assert y.name == TARGET_COL

    def test_no_target_in_X(self, sample_df):
        X, _ = split_features_target(sample_df)
        assert TARGET_COL not in X.columns

    def test_lengths_match(self, sample_df):
        X, y = split_features_target(sample_df)
        assert len(X) == len(y) == len(sample_df)


class TestTrainTestSplit:
    def test_split_size(self, sample_df):
        X_train, X_test, y_train, y_test = get_train_test(sample_df, test_size=0.2)
        total = len(X_train) + len(X_test)
        assert total == len(sample_df)

    def test_no_leakage(self, sample_df):
        X_train, X_test, _, _ = get_train_test(sample_df)
        common = set(X_train.index) & set(X_test.index)
        assert len(common) == 0

    def test_stratified(self, sample_df):
        _, _, y_train, y_test = get_train_test(sample_df)
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        # Allow up to 15 % difference due to small n
        assert abs(train_rate - test_rate) < 0.15


class TestColumnConstants:
    def test_numerical_subset(self):
        for col in NUMERICAL_COLS:
            assert col in FEATURE_COLS

    def test_categorical_subset(self):
        for col in CATEGORICAL_COLS:
            assert col in FEATURE_COLS

    def test_no_overlap_between_sets(self):
        overlap = set(NUMERICAL_COLS) & set(CATEGORICAL_COLS)
        assert len(overlap) == 0
