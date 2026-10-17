from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
MODEL_PATH = MODEL_DIR / "heart_disease_model.joblib"
RAW_DATA_PATH = DATA_DIR / "heart_disease_raw.csv"
PROCESSED_DATA_PATH = DATA_DIR / "heart_disease_clean.csv"

FEATURE_COLUMNS = [
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
TARGET_COLUMN = "target"


def download_dataset() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_DATA_PATH.exists():
        import urllib.request

        urllib.request.urlretrieve(DATASET_URL, RAW_DATA_PATH)
    return RAW_DATA_PATH


def load_and_clean_data() -> pd.DataFrame:
    download_dataset()
    df = pd.read_csv(RAW_DATA_PATH, header=None, names=FEATURE_COLUMNS + [TARGET_COLUMN], engine="python")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] > 0).astype(int)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    return df


def build_models() -> dict[str, Pipeline]:
    logistic_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")),
        ]
    )
    forest_pipeline = Pipeline(
        steps=[
            ("model", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")),
        ]
    )
    return {
        "logistic_regression": logistic_pipeline,
        "random_forest": forest_pipeline,
    }


def evaluate_model(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "roc_auc": "roc_auc",
        },
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    result = {
        "cv_accuracy": float(scores["test_accuracy"].mean()),
        "cv_precision": float(scores["test_precision"].mean()),
        "cv_recall": float(scores["test_recall"].mean()),
        "cv_roc_auc": float(scores["test_roc_auc"].mean()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }
    return result


def train_model() -> dict:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    df = load_and_clean_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    mlflow.set_tracking_uri(str(ROOT / "mlruns"))
    mlflow.set_experiment("heart-disease-prediction")

    best_model_name = ""
    best_roc_auc = -1.0
    model_results: dict[str, dict] = {}

    with mlflow.start_run(run_name="heart_disease_training") as parent_run:
        for model_name, model in build_models().items():
            with mlflow.start_run(run_name=model_name, nested=True):
                metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
                model_results[model_name] = metrics

                mlflow.log_param("model_name", model_name)
                mlflow.log_params(model.named_steps["model"].get_params())
                for key, value in metrics.items():
                    mlflow.log_metric(key, float(value))

                if metrics["roc_auc"] > best_roc_auc:
                    best_roc_auc = metrics["roc_auc"]
                    best_model_name = model_name

        final_model = build_models()[best_model_name]
        final_model.fit(X_train, y_train)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_param("selected_model", best_model_name)

    summary = {
        "best_model": best_model_name,
        "metrics": model_results,
        "model_path": str(MODEL_PATH),
        "run_id": parent_run.info.run_id,
    }
    return summary


def predict_record(payload: dict) -> dict:
    if not MODEL_PATH.exists():
        train_model()

    model = joblib.load(MODEL_PATH)
    features = {feature: payload.get(feature, 0) for feature in FEATURE_COLUMNS}
    record = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    prediction = int(model.predict(record)[0])
    probability = float(model.predict_proba(record)[0][prediction])
    return {
        "prediction": prediction,
        "risk_label": "Heart disease detected" if prediction == 1 else "No heart disease detected",
        "confidence": round(probability, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Heart disease MLOps pipeline")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", help="Predict for a JSON payload")
    args = parser.parse_args()

    if args.train:
        result = train_model()
        print(json.dumps(result, indent=2))
        return

    if args.predict:
        payload = json.loads(args.predict)
        print(json.dumps(predict_record(payload), indent=2))
        return

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
    print(json.dumps(predict_record(sample), indent=2))


if __name__ == "__main__":
    main()
