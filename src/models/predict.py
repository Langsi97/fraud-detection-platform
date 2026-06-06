import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def load_preprocessor(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessing pipeline not found: {path}")
    return joblib.load(path)


def predict_transaction(model, preprocessor, transaction: Dict[str, float]) -> Dict[str, object]:
    features_df = pd.DataFrame([transaction])
    validate_input_features(preprocessor, features_df)
    transformed = preprocessor.transform(features_df)
    probabilities = model.predict_proba(transformed)[:, 1]
    prediction = int(probabilities[0] >= 0.5)
    return {
        "fraud_probability": float(probabilities[0]),
        "prediction": "Fraud" if prediction == 1 else "Legitimate",
        "prediction_label": prediction,
    }


def predict_batch(model, preprocessor, df: pd.DataFrame) -> pd.DataFrame:
    validate_input_features(preprocessor, df)
    transformed = preprocessor.transform(df)
    probabilities = model.predict_proba(transformed)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    output = df.copy()
    output["fraud_probability"] = probabilities
    output["prediction"] = np.where(predictions == 1, "Fraud", "Legitimate")
    output["prediction_label"] = predictions
    return output


def validate_input_features(preprocessor, df: pd.DataFrame) -> None:
    if hasattr(preprocessor, "transformers"):
        numeric_features = preprocessor.transformers_[0][2]
        missing = [col for col in numeric_features if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required input fields: {missing}")


def persist_artifacts(object_to_save, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(object_to_save, path)
