"""
Unit tests for model training pipeline.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_preprocessing import split_features_target, FEATURE_COLS
from src.feature_engineering import build_preprocessor
from src.train import get_models


@pytest.fixture
def tiny_df():
    rng = np.random.default_rng(0)
    n = 60
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


class TestPreprocessor:
    def test_output_shape(self, tiny_df):
        X, y = split_features_target(tiny_df)
        preprocessor = build_preprocessor()
        X_transformed = preprocessor.fit_transform(X)
        # Should have more columns than input due to OHE
        assert X_transformed.shape[0] == len(X)
        assert X_transformed.shape[1] > len(FEATURE_COLS)

    def test_no_nan_after_transform(self, tiny_df):
        X, y = split_features_target(tiny_df)
        preprocessor = build_preprocessor()
        X_t = preprocessor.fit_transform(X)
        assert not np.isnan(X_t).any()


class TestModels:
    @pytest.mark.parametrize("name,estimator", list(get_models().items()))
    def test_fit_predict(self, tiny_df, name, estimator):
        X, y = split_features_target(tiny_df)
        preprocessor = build_preprocessor()
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        probs = pipeline.predict_proba(X)

        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})
        assert probs.shape == (len(y), 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    @pytest.mark.parametrize("name,estimator", list(get_models().items()))
    def test_prediction_range(self, tiny_df, name, estimator):
        X, y = split_features_target(tiny_df)
        preprocessor = build_preprocessor()
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
        pipeline.fit(X, y)
        probs = pipeline.predict_proba(X)
        assert (probs >= 0).all() and (probs <= 1).all()
