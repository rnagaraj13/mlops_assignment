# Heart Disease MLOps Pipeline
> **AIMLCZG523 – Assignment 01**

End-to-end MLOps solution: data ingestion → EDA → model training with MLflow → FastAPI serving → Docker → Kubernetes → CI/CD → Monitoring.

---

## Repository Layout

```
heart-disease-mlops/
├── data/
│   ├── download_data.py       # Dataset download & cleaning script
│   └── heart.csv              # Cleaned dataset (generated)
├── notebooks/
│   ├── eda.py                 # EDA script (outputs plots/)
│   └── plots/                 # All EDA visualisations
├── src/
│   ├── data_preprocessing.py  # Load, split, encode
│   ├── feature_engineering.py # ColumnTransformer preprocessor
│   ├── train.py               # Train + MLflow logging
│   └── evaluate.py            # Evaluation report
├── api/
│   └── app.py                 # FastAPI – /predict /health /metrics
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_api.py
├── models/                    # Saved best_model.pkl (generated)
├── .github/workflows/
│   └── ci-cd.yml              # GitHub Actions pipeline
├── k8s/
│   ├── namespace.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   └── ingress.yaml
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/provisioning/  # Auto-provisioned Grafana datasource + dashboard
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Quick Start (local)

### 1 – Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-mlops.git
cd heart-disease-mlops
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2 – Download dataset

```bash
python data/download_data.py
```

### 3 – Run EDA

```bash
python notebooks/eda.py
# Plots saved to notebooks/plots/
```

### 4 – Train models (with MLflow)

```bash
python src/train.py --data data/heart.csv --experiment heart-disease
# Open MLflow UI:
mlflow ui --port 5000
```

### 5 – Evaluate

```bash
python src/evaluate.py --data data/heart.csv --model models/best_model.pkl
```

### 6 – Run unit tests

```bash
pytest tests/ -v --cov=src --cov=api
```

### 7 – Run API locally

```bash
uvicorn api.app:app --reload --port 8000
# Test:
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":54,"sex":1,"cp":0,"trestbps":122,"chol":286,"fbs":0,"restecg":0,"thalach":116,"exang":1,"oldpeak":3.2,"slope":1,"ca":2,"thal":2}'
```

### 8 – Docker (full stack)

```bash
# Build & run API only
docker build -t heart-disease-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models heart-disease-api

# Full stack: API + MLflow + Prometheus + Grafana
docker compose up -d
# API:        http://localhost:8000
# MLflow:     http://localhost:5000
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin / admin123)
```

### 9 – Kubernetes (Minikube)

```bash
minikube start --driver=docker
kubectl apply -f k8s/
kubectl get pods -n heart-mlops
minikube service heart-disease-api-svc -n heart-mlops
```

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci-cd.yml`) runs on every push to `main`:

| Stage | What it does |
|-------|-------------|
| **Lint** | Black, isort, Flake8 |
| **Test** | pytest + coverage report |
| **Train** | Downloads data, trains models, uploads artefacts |
| **Build & Push** | Multi-stage Docker build → GHCR |
| **Deploy** | `kubectl apply` to the configured cluster |

Set the following repository secrets:
- `KUBE_CONFIG` – base64-encoded kubeconfig for your cluster

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness / readiness probe |
| GET | `/metrics` | Prometheus text metrics |
| POST | `/predict` | Single-patient prediction |
| POST | `/predict/batch` | Batch predictions |
| GET | `/docs` | Swagger UI |

**Sample request:**
```json
POST /predict
{
  "age": 54, "sex": 1, "cp": 0, "trestbps": 122,
  "chol": 286, "fbs": 0, "restecg": 0, "thalach": 116,
  "exang": 1, "oldpeak": 3.2, "slope": 1, "ca": 2, "thal": 2
}
```

**Sample response:**
```json
{
  "prediction": 1,
  "confidence": 0.82,
  "label": "Heart Disease"
}
```

---

## Models

| Model | CV ROC-AUC | Notes |
|-------|-----------|-------|
| Logistic Regression | ~0.89 | Fast, interpretable baseline |
| Random Forest | ~0.92 | Best performer; saved as `best_model.pkl` |

Preprocessor: `StandardScaler` on numerical features + `OneHotEncoder` on categorical features, wrapped in a single `sklearn.Pipeline`.

---

## Monitoring

- **Prometheus** scrapes `/metrics` every 10 s  
- **Grafana** dashboard `heart_api.json` is auto-provisioned and shows: total requests, success/error rates, average latency, prediction class distribution, and request-rate time series.

---

## Dataset

- **Source:** UCI Machine Learning Repository – Heart Disease (Cleveland)  
- **Features:** 13 clinical attributes (age, sex, chest pain type, blood pressure, cholesterol, …)  
- **Target:** Binary – 0 = No Disease, 1 = Heart Disease  
- **Size:** 303 rows after cleaning

---

*AIMLCZG523 – Machine Learning Operations | Assignment 01*
