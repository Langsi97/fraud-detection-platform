import os
from pathlib import Path
from typing import Dict

import pandas as pd


def compute_monitoring_metrics(predictions: pd.DataFrame) -> Dict[str, float]:
    fraud_rate = float((predictions["prediction_label"] == 1).mean())
    average_score = float(predictions["fraud_probability"].mean())
    return {
        "prediction_volume": int(len(predictions)),
        "fraud_rate": fraud_rate,
        "average_fraud_probability": average_score,
    }


def create_monitoring_exports(predictions: pd.DataFrame, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    daily_summary = (
        predictions.assign(date=pd.to_datetime("today").normalize())
        .groupby("date")
        .agg(prediction_volume=("prediction_label", "count"), fraud_rate=("prediction_label", "mean"), average_fraud_probability=("fraud_probability", "mean"))
        .reset_index()
    )
    daily_summary.to_csv(os.path.join(output_dir, "daily_prediction_summary.csv"), index=False)

    fraud_rate_summary = (
        predictions.groupby("prediction")
        .agg(prediction_volume=("prediction_label", "count"), average_fraud_probability=("fraud_probability", "mean"))
        .reset_index()
    )
    fraud_rate_summary.to_csv(os.path.join(output_dir, "fraud_rate_summary.csv"), index=False)

    trends = (
        predictions.assign(date=pd.to_datetime("today").normalize())
        .groupby("date")
        .agg(fraud_transactions=("prediction_label", "sum"), total_transactions=("prediction_label", "count"))
        .reset_index()
    )
    trends.to_csv(os.path.join(output_dir, "prediction_trends.csv"), index=False)

    top_risk = predictions.sort_values("fraud_probability", ascending=False).head(100)
    top_risk.to_csv(os.path.join(output_dir, "high_risk_transactions.csv"), index=False)
