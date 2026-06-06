import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.predict import predict_transaction, predict_batch


class DummyModel:
    def predict_proba(self, X):
        return [[0.1, 0.9] for _ in range(X.shape[0])]


def test_predict_transaction_returns_fraud():
    dummy_model = DummyModel()
    preprocessor = Pipeline([("scaler", StandardScaler())])
    df = pd.DataFrame({"V1": [1.0], "V2": [2.0], "Amount": [50.0]})
    preprocessor.fit(df)

    result = predict_transaction(dummy_model, preprocessor, df.iloc[0].to_dict())

    assert result["prediction"] == "Fraud"
    assert 0.0 <= result["fraud_probability"] <= 1.0


def test_predict_batch_produces_expected_columns():
    dummy_model = DummyModel()
    preprocessor = Pipeline([("scaler", StandardScaler())])
    df = pd.DataFrame({"V1": [1.0, 2.0], "V2": [2.0, 3.0], "Amount": [10.0, 20.0]})
    preprocessor.fit(df)
    scored = predict_batch(dummy_model, preprocessor, df)

    assert "fraud_probability" in scored.columns
    assert "prediction" in scored.columns
    assert len(scored) == 2
