from __future__ import annotations

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from src.inference import ensure_model_exists, predict_record

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")


class HeartDiseaseRequest(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


@app.on_event("startup")
def startup_event() -> None:
    ensure_model_exists()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "heart-disease-prediction"}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Heart Disease Prediction API is running."}


@app.post("/predict")
def predict(payload: HeartDiseaseRequest) -> dict[str, object]:
    result = predict_record(payload.model_dump())
    return {
        "input": payload.model_dump(),
        "prediction": result["prediction"],
        "risk_label": result["risk_label"],
        "confidence": result["confidence"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
