# Heart Disease MLOps Project

This project implements an end-to-end MLOps workflow for the UCI Heart Disease dataset. It covers dataset acquisition, exploratory analysis, model training, experiment tracking, automated testing, packaging, deployment manifests, and a production-ready API.

## Project goals

- Build a reproducible ML pipeline for binary heart disease prediction.
- Train and compare baseline classification models.
- Track experiments with MLflow.
- Package the trained model into a reusable artifact.
- Expose a FastAPI prediction service.
- Run automated tests and CI with GitHub Actions.
- Containerize and deploy the model API locally or in Kubernetes.

## Dataset

Source: UCI Machine Learning Repository (Cleveland heart disease dataset)

## Project structure

- `src/`: data processing, EDA, training, and inference logic
- `app/`: FastAPI service
- `tests/`: unit tests
- `scripts/`: dataset download helper
- `deployment/kubernetes/`: Kubernetes deployment manifest
- `.github/workflows/`: CI pipeline
- `docs/`: project report and architecture docs
- `data/`: dataset downloads and processed outputs

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download and prepare the dataset:
   ```bash
   python scripts/download_dataset.py
   ```
4. Run EDA and model training:
   ```bash
   python src/eda.py
   python src/train.py
   ```
5. Start the API:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
6. Open the Swagger UI:
   `http://localhost:8000/docs`

## API usage

Example request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52,
    "sex": 1,
    "cp": 0,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 2,
    "thal": 3
  }'
```

## CI/CD

The GitHub Actions workflow runs linting and tests on every push or pull request. It installs dependencies, runs the test suite, and verifies model training still succeeds.

## Deployment

The Kubernetes deployment manifest configures a load-balanced service for the model API.

```bash
kubectl apply -f deployment/kubernetes/deployment.yaml
```

## Documentation

The final report is available under `docs/report.md`.

## Notes

This project uses a reproducible, local MLflow tracking directory (`mlruns/`) so experiments remain portable and easy to review in a clean environment.
