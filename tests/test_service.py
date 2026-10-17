from src.inference import predict_record


def test_prediction_output_is_valid():
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

    result = predict_record(sample)
    assert set(result.keys()) == {"prediction", "risk_label", "confidence"}
    assert result["prediction"] in {0, 1}
    assert isinstance(result["confidence"], float)
