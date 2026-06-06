import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def compute_classification_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }


def compute_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def save_metrics(metrics: Dict[str, float], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def save_feature_importance(feature_names: List[str], importances: List[float], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    feature_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    feature_df.sort_values("importance", ascending=False, inplace=True)
    feature_df.to_csv(output_path, index=False)
