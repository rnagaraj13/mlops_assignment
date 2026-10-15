# Heart Disease MLOps Pipeline

AIMLCZG523 – Assignment 01

Classifier for the UCI Cleveland heart-disease file. Training logs to a local MLflow folder. The API, Docker image, and Kubernetes manifests all load `models/best_model.pkl`. Render is an optional public copy of that same API.

Repository: https://github.com/rnagaraj13/mlops_assignment

Use Python 3.11 or 3.12. The pinned NumPy wheel does not install on Python 3.14.

## College lab setup

Run these from the repo root, in this order. Python 3.11 or 3.12. Stop `uvicorn` before the Docker commands if port 8000 is still taken.

```bash
git clone https://github.com/rnagaraj13/mlops_assignment.git
cd mlops_assignment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python data/download_data.py
python notebooks/eda.py
python src/train.py --data data/heart.csv --experiment heart-disease
mlflow ui --port 5000
```

Leave `mlflow ui` running and open http://localhost:5000. That UI reads the local `mlruns/` directory created by training. It is the experiment log for the demo.

In a second terminal, with the same virtual environment:

```bash
pytest tests/ -v --cov=src --cov=api
python src/evaluate.py --data data/heart.csv --model models/best_model.pkl
uvicorn api.app:app --port 8000
```

Sample prediction (first Cleveland row; original UCI codes):

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```

Docker, after the model file exists:

```bash
docker build -t heart-disease-api:latest .
docker run --rm -p 8000:8000 heart-disease-api:latest
```

Monitoring stack (API, Prometheus, Grafana). This does not start a second MLflow server:

```bash
docker compose up -d
```

- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin / admin123)

Kubernetes, when Minikube or Docker Desktop Kubernetes is installed:

```bash
docker build -t heart-disease-api:latest .
minikube start --driver=docker
minikube image load heart-disease-api:latest
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl rollout status deployment/heart-disease-api -n heart-mlops
minikube service heart-disease-api-svc -n heart-mlops
```

The Service type is LoadBalancer. On Minikube, `minikube service` opens it. The Ingress has no hostname, so the Minikube ingress addon can route `/` without a certificate issuer.

`data/heart.csv` and `models/best_model.pkl` are already in the repository, so Docker can build even if the UCI download is blocked. Training overwrites the model and refreshes `mlruns/`.

## Architecture

![Heart disease pipeline](docs/architecture.png)

GitHub Actions (`.github/workflows/ci-cd.yml`) runs lint, tests, training, the Docker build, a Minikube smoke test, and the Render hook on `main`. Set `RENDER_DEPLOY_HOOK_URL` in the repo secrets if you want that hook to fire. Render builds the Dockerfile from git. It does not run MLflow.

## API

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Loads the model. Returns 503 if the file is missing |
| GET | `/metrics` | Prometheus text metrics |
| POST | `/predict` | One patient. Returns prediction, confidence, and label |
| POST | `/predict/batch` | Several patients |
| GET | `/docs` | Swagger UI |

Codes match the UCI Cleveland file, not the Kaggle 0-based recoding:

| Field | Values |
| --- | --- |
| cp | 1 typical, 2 atypical, 3 non-anginal, 4 asymptomatic |
| slope | 1 upsloping, 2 flat, 3 downsloping |
| thal | 3 normal, 6 fixed defect, 7 reversible defect |
| target | 0 no disease, 1 disease (original 1–4 collapsed to 1) |

## Models

Held-out split: 20% stratified, `random_state=42`, 60 rows. Grid search and the cross-validation metrics use only the training 237 rows.

| Model | Search | CV ROC-AUC | Held-out ROC-AUC | Held-out recall |
| --- | --- | --- | --- | --- |
| Logistic regression | `C` in {0.01, 0.1, 1, 10}. Best `C=1.0` | 0.904 | 0.916 | 0.643 |
| Random forest | depth {4, 8, 12}, trees {100, 200}, min split {2, 5}. Best depth 4, 200 trees, min split 2 | 0.879 | 0.917 | 0.786 |

I saved the random forest. Held-out ROC-AUC is 0.917 against 0.916 for logistic regression, and recall is 0.786 against 0.643. Cross-validation ROC-AUC actually preferred logistic regression (0.904 vs 0.879). Details are in `models/model_selection.json`.

The pickle is one pipeline: `age / thalach` and `oldpeak * slope`, then scaling, one-hot encoding, then the classifier. `/predict` still takes the 13 raw fields.

## Dataset

- Source: UCI Machine Learning Repository, Heart Disease, processed Cleveland file
- Download: `python data/download_data.py` (uses `certifi` so a lab Python without system certificates can still fetch it)
- 297 rows after dropping 6 rows with `?` missing values
- Class counts: 160 no disease, 137 disease

## Monitoring

Prometheus scrapes `api:8000/metrics` every 10 seconds. Grafana loads `monitoring/grafana/provisioning/dashboards/heart_api.json` on startup: request count, success and error counts, average latency, prediction mix, and request rate.

Request lines are also written to the API process log.

Code lives in `data/`, `notebooks/`, `src/`, `api/`, and `tests/`. The saved model is `models/best_model.pkl`. Kubernetes YAML is in `k8s/`. Prometheus and Grafana config is in `monitoring/`. The write-up is `REPORT.md`.
