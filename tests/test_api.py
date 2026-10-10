"""
Integration tests for the FastAPI app (uses TestClient – no real model needed).
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

SAMPLE_PATIENT = {
    "age": 54,
    "sex": 1,
    "cp": 0,
    "trestbps": 122,
    "chol": 286,
    "fbs": 0,
    "restecg": 0,
    "thalach": 116,
    "exang": 1,
    "oldpeak": 3.2,
    "slope": 1,
    "ca": 2,
    "thal": 2,
}


@pytest.fixture
def mock_model():
    """Return a lightweight mock pipeline."""
    model = MagicMock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.2, 0.8]])
    return model


@pytest.fixture
def client(mock_model):
    """Provide a TestClient with the model monkeypatched."""
    with patch("api.app.get_model", return_value=mock_model):
        from api.app import app

        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_healthy(self, client, mock_model):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


class TestPredictEndpoint:
    def test_predict_valid(self, client):
        resp = client.post("/predict", json=SAMPLE_PATIENT)
        assert resp.status_code == 200
        body = resp.json()
        assert "prediction" in body
        assert "confidence" in body
        assert body["prediction"] in (0, 1)
        assert 0.0 <= body["confidence"] <= 1.0

    def test_predict_positive(self, client):
        resp = client.post("/predict", json=SAMPLE_PATIENT)
        assert resp.json()["prediction"] == 1
        assert resp.json()["label"] == "Heart Disease"

    def test_predict_missing_field(self, client):
        incomplete = {k: v for k, v in SAMPLE_PATIENT.items() if k != "age"}
        resp = client.post("/predict", json=incomplete)
        assert resp.status_code == 422  # Unprocessable Entity

    def test_predict_invalid_range(self, client):
        bad = dict(SAMPLE_PATIENT)
        bad["age"] = 999  # Out of valid range
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422


class TestBatchEndpoint:
    def test_batch_predict(self, client, mock_model):
        mock_model.predict.return_value = np.array([1, 0])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])
        resp = client.post(
            "/predict/batch", json={"patients": [SAMPLE_PATIENT, SAMPLE_PATIENT]}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["predictions"]) == 2


class TestMetricsEndpoint:
    def test_metrics_format(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "heart_api_requests_total" in resp.text
