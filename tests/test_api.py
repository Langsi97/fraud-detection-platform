from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.api.main import app


client = TestClient(app)


class DummyModel:
    def predict_proba(self, X):
        return [[0.1, 0.9] for _ in range(X.shape[0])]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "Credit Card Fraud Detection"


@patch("src.api.main.load_model")
@patch("src.api.main.load_preprocessor")
def test_predict_endpoint_success(mock_preprocessor, mock_load_model):
    dummy_model = DummyModel()
    df = pd.DataFrame({"V1": [1.0], "V2": [2.0], "Amount": [15.0]})
    pipeline = Pipeline([("scaler", StandardScaler())])
    pipeline.fit(df)

    mock_load_model.return_value = dummy_model
    mock_preprocessor.return_value = pipeline

    response = client.post(
        "/predict",
        json={"transaction": {"V1": 1.0, "V2": 2.0, "Amount": 15.0}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "Fraud"
    assert 0.0 <= body["fraud_probability"] <= 1.0
