"""
FastAPI serving application for Heart Disease prediction.

Endpoints:
  GET  /health        – liveness probe
  GET  /metrics       – Prometheus-compatible plain-text metrics
  POST /predict       – predict from JSON input, returns prediction + confidence
"""

import logging
import os
import time
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("heart-api")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts the probability of heart disease from patient health data.",
    version="1.0.0",
)

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/best_model.pkl")
_model = None


def get_model():
    global _model
    if _model is None:
        if not Path(MODEL_PATH).exists():
            raise RuntimeError(f"Model not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
    return _model


# ── Prometheus-style counters (in-memory) ─────────────────────────────────────
_counters = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_error": 0,
    "predictions_positive": 0,
    "predictions_negative": 0,
}
_latency_sum = 0.0
_latency_count = 0


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class PatientFeatures(BaseModel):
    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="0=female 1=male")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure")
    chol: float = Field(..., ge=100, le=600, description="Serum cholesterol mg/dL")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar >120 mg/dL")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=50, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels (0-4)")
    thal: int = Field(
        ..., ge=0, le=3, description="Thal (0=normal 1=fixed 2=reversible)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class BatchRequest(BaseModel):
    patients: List[PatientFeatures]


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0=No Disease, 1=Heart Disease")
    confidence: float = Field(..., description="Probability of heart disease")
    label: str = Field(..., description="Human-readable label")


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


# ── Middleware – request logging ──────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    global _latency_sum, _latency_count
    _counters["requests_total"] += 1
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    _latency_sum += elapsed
    _latency_count += 1
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} duration={elapsed:.3f}s"
    )
    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
def health():
    """Liveness + readiness probe."""
    try:
        get_model()
        return {"status": "healthy", "model": "loaded"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
def metrics():
    """Expose Prometheus-compatible text metrics."""
    avg_latency = (_latency_sum / _latency_count) if _latency_count else 0.0
    lines = [
        "# HELP heart_api_requests_total Total HTTP requests",
        "# TYPE heart_api_requests_total counter",
        f'heart_api_requests_total {_counters["requests_total"]}',
        "",
        "# HELP heart_api_requests_success_total Successful predictions",
        "# TYPE heart_api_requests_success_total counter",
        f'heart_api_requests_success_total {_counters["requests_success"]}',
        "",
        "# HELP heart_api_requests_error_total Prediction errors",
        "# TYPE heart_api_requests_error_total counter",
        f'heart_api_requests_error_total {_counters["requests_error"]}',
        "",
        "# HELP heart_api_predictions_positive_total Positive (disease) predictions",
        "# TYPE heart_api_predictions_positive_total counter",
        f'heart_api_predictions_positive_total {_counters["predictions_positive"]}',
        "",
        "# HELP heart_api_predictions_negative_total Negative predictions",
        "# TYPE heart_api_predictions_negative_total counter",
        f'heart_api_predictions_negative_total {_counters["predictions_negative"]}',
        "",
        "# HELP heart_api_avg_latency_seconds Average request latency",
        "# TYPE heart_api_avg_latency_seconds gauge",
        f"heart_api_avg_latency_seconds {avg_latency:.6f}",
    ]
    return "\n".join(lines)


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(patient: PatientFeatures):
    """Single-patient prediction."""
    try:
        model = get_model()
        row = pd.DataFrame([patient.model_dump()])
        pred = int(model.predict(row)[0])
        confidence = float(model.predict_proba(row)[0][1])

        _counters["requests_success"] += 1
        if pred == 1:
            _counters["predictions_positive"] += 1
        else:
            _counters["predictions_negative"] += 1

        label = "Heart Disease" if pred == 1 else "No Heart Disease"
        logger.info(f"Prediction: {label} (confidence={confidence:.4f})")

        return PredictionResponse(prediction=pred, confidence=confidence, label=label)

    except Exception as exc:
        _counters["requests_error"] += 1
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
def predict_batch(batch: BatchRequest):
    """Batch prediction endpoint."""
    try:
        model = get_model()
        rows = pd.DataFrame([p.model_dump() for p in batch.patients])
        preds = model.predict(rows).tolist()
        confidences = model.predict_proba(rows)[:, 1].tolist()

        results = []
        for pred, conf in zip(preds, confidences):
            label = "Heart Disease" if pred == 1 else "No Heart Disease"
            results.append(
                PredictionResponse(
                    prediction=int(pred), confidence=float(conf), label=label
                )
            )
            _counters["requests_success"] += 1
            if pred == 1:
                _counters["predictions_positive"] += 1
            else:
                _counters["predictions_negative"] += 1

        return BatchPredictionResponse(predictions=results)

    except Exception as exc:
        _counters["requests_error"] += 1
        logger.error(f"Batch prediction error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
