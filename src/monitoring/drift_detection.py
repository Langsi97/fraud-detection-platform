import os
from pathlib import Path
from typing import Optional

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


def generate_drift_report(reference_data: pd.DataFrame, current_data: pd.DataFrame, output_html: str) -> None:
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    Path(output_html).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(output_html)


def summarize_feature_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame, output_path: str) -> pd.DataFrame:
    numeric_features = [col for col in current_data.columns if col in reference_data.columns and pd.api.types.is_numeric_dtype(current_data[col])]
    drift_summary = []
    for feature in numeric_features:
        reference_series = reference_data[feature].dropna()
        current_series = current_data[feature].dropna()
        drift_summary.append(
            {
                "feature": feature,
                "reference_mean": float(reference_series.mean()),
                "current_mean": float(current_series.mean()),
                "mean_delta": float(current_series.mean() - reference_series.mean()),
                "reference_std": float(reference_series.std()),
                "current_std": float(current_series.std()),
                "std_delta": float(current_series.std() - reference_series.std()),
            }
        )
    df = pd.DataFrame(drift_summary)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
