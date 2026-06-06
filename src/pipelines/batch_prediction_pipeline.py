import os
from pathlib import Path
from typing import Optional

import pandas as pd

from src.database.db import get_session, log_prediction_record
from src.models.predict import load_model, load_preprocessor, predict_batch
from src.monitoring.drift_detection import generate_drift_report, summarize_feature_drift
from src.monitoring.metrics_tracking import compute_monitoring_metrics, create_monitoring_exports

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"
PREPROCESSOR_PATH = ROOT_DIR / "models" / "preprocessor.pkl"
TABLEAU_OUTPUT_DIR = ROOT_DIR / "tableau" / "dashboard_data"
MONITORING_REPORT_PATH = ROOT_DIR / "reports" / "monitoring" / "batch_drift_report.html"
FEATURE_DRIFT_PATH = TABLEAU_OUTPUT_DIR / "feature_drift_summary.csv"


def run_batch_prediction(input_csv: str, database_url: Optional[str] = None) -> pd.DataFrame:
    data = pd.read_csv(input_csv)
    model = load_model(str(MODEL_PATH))
    preprocessor = load_preprocessor(str(PREPROCESSOR_PATH))

    predictions = predict_batch(model, preprocessor, data)
    predictions.to_csv(str(ROOT_DIR / "data" / "processed" / "batch_predictions.csv"), index=False)

    with get_session(database_url) as session:
        for _, row in predictions.iterrows():
            log_prediction_record(
                session=session,
                fraud_score=float(row["fraud_probability"]),
                prediction_label=row["prediction"],
                model_version="production",
                transaction_amount=float(row.get("Amount", 0.0)),
            )

    create_monitoring_exports(predictions, str(TABLEAU_OUTPUT_DIR))
    reference_data = pd.read_csv(str(ROOT_DIR / "data" / "processed" / "processed_data.csv"))
    generate_drift_report(reference_data, data, str(MONITORING_REPORT_PATH))
    summarize_feature_drift(reference_data, data, str(FEATURE_DRIFT_PATH))

    summary = compute_monitoring_metrics(predictions)
    print("Batch prediction summary:", summary)
    return predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run batch prediction for fraud detection.")
    parser.add_argument("--input", required=True, help="Path to an input CSV file to score.")
    parser.add_argument("--database-url", default=None, help="Optional SQLite database URL.")
    args = parser.parse_args()
    run_batch_prediction(args.input, args.database_url)
