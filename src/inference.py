from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

try:
    from src.data_pipeline import FEATURE_COLUMNS
    from src.train import MODEL_PATH
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.data_pipeline import FEATURE_COLUMNS
    from src.train import MODEL_PATH

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "heart_disease_model.joblib"


def ensure_model_exists() -> None:
    if not MODEL_PATH.exists():
        from src.train import run_training

        run_training()


def prepare_features(payload: dict) -> pd.DataFrame:
    row = {feature: payload.get(feature, 0) for feature in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_record(payload: dict) -> dict:
    ensure_model_exists()
    model = joblib.load(MODEL_PATH)
    features = prepare_features(payload)
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][prediction])
    return {
        "prediction": prediction,
        "risk_label": "Heart disease detected" if prediction == 1 else "No heart disease detected",
        "confidence": round(probability, 4),
    }


if __name__ == "__main__":
    sample = {
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
        "thal": 3,
    }
    print(predict_record(sample))
