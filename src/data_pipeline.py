from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "heart_disease_raw.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "heart_disease_clean.csv"
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

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


def download_dataset(destination: Path | None = None) -> Path:
    destination = destination or RAW_DATA_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urlretrieve(DATASET_URL, destination)
    return destination


def load_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    raw_path = download_dataset(path) if path is not None else download_dataset()
    return pd.read_csv(raw_path, header=None, names=FEATURE_COLUMNS + [TARGET_COLUMN], sep=",")


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    processed = df.copy()
    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        if column in processed.columns:
            processed[column] = pd.to_numeric(processed[column], errors="coerce")

    processed = processed.dropna().reset_index(drop=True)
    processed[TARGET_COLUMN] = processed[TARGET_COLUMN].astype(int)
    processed[TARGET_COLUMN] = (processed[TARGET_COLUMN] > 0).astype(int)
    return processed


def prepare_dataset() -> pd.DataFrame:
    raw_df = load_raw_dataset()
    cleaned_df = preprocess_dataset(raw_df)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(PROCESSED_DATA_PATH, index=False)
    return cleaned_df


def load_processed_dataset(path: Path | None = None) -> pd.DataFrame:
    dataset_path = path or PROCESSED_DATA_PATH
    if not dataset_path.exists():
        return prepare_dataset()
    return pd.read_csv(dataset_path)


if __name__ == "__main__":
    df = prepare_dataset()
    print(f"Saved cleaned dataset to {PROCESSED_DATA_PATH} with shape {df.shape}.")
