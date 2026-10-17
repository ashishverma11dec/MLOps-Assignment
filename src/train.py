from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

try:
    from src.data_pipeline import FEATURE_COLUMNS, TARGET_COLUMN, load_processed_dataset
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.data_pipeline import FEATURE_COLUMNS, TARGET_COLUMN, load_processed_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "heart_disease_model.joblib"
REPORTS_PATH = PROJECT_ROOT / "reports"
REPORTS_PATH.mkdir(exist_ok=True)


def build_models() -> dict[str, Pipeline]:
    logistic_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")),
        ]
    )
    forest_model = Pipeline(
        steps=[
            ("model", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"))
        ]
    )
    return {
        "logistic_regression": logistic_model,
        "random_forest": forest_model,
    }


def evaluate_model(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={"accuracy": "accuracy", "precision": "precision", "recall": "recall", "roc_auc": "roc_auc"},
        n_jobs=None,
    )

    result = {
        "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
        "cv_precision_mean": float(scores["test_precision"].mean()),
        "cv_recall_mean": float(scores["test_recall"].mean()),
        "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
    }

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)

    result.update(
        {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "confusion_matrix": cm.tolist(),
        }
    )
    return result


def save_training_artifacts(model_name: str, y_test: pd.Series, y_pred: pd.Series, y_prob: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{model_name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(REPORTS_PATH / f"{model_name}_confusion_matrix.png", dpi=200)
    plt.close(fig)


def run_training() -> dict:
    df = load_processed_dataset()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))
    mlflow.set_experiment("heart-disease-classification")

    model_summaries: dict[str, dict] = {}
    best_model_name = ""
    best_score = -1.0

    with mlflow.start_run(run_name="heart_disease_training") as run:
        for model_name, model in build_models().items():
            with mlflow.start_run(run_name=model_name, nested=True) as model_run:
                metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
                model_summaries[model_name] = metrics

                mlflow.log_param("model_name", model_name)
                mlflow.log_params(model.named_steps["model"].get_params())
                for metric_name, metric_value in metrics.items():
                    if metric_name != "confusion_matrix":
                        mlflow.log_metric(metric_name, float(metric_value))

                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                save_training_artifacts(model_name, y_test, y_pred, y_prob)
                mlflow.log_artifact(str(REPORTS_PATH / f"{model_name}_confusion_matrix.png"))

                if metrics["roc_auc"] > best_score:
                    best_score = metrics["roc_auc"]
                    best_model_name = model_name

        final_model = build_models()[best_model_name]
        final_model.fit(X_train, y_train)
        joblib.dump(final_model, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_param("selected_model", best_model_name)

    metadata = {
        "run_id": run.info.run_id,
        "best_model": best_model_name,
        "metrics": model_summaries,
    }
    metadata_path = REPORTS_PATH / "training_summary.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata


if __name__ == "__main__":
    result = run_training()
    print(json.dumps(result, indent=2))
