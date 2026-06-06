import os
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.models.evaluate import compute_classification_metrics, compute_confusion_matrix


def _get_scale_pos_weight(y_train):
    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) != 2:
        return 1.0
    majority = max(counts)
    minority = min(counts)
    return float(majority / minority)


def train_candidate_models(X_train, y_train, cv: int = 3) -> Dict[str, Dict]:
    """Train candidate classifiers and return evaluation metadata."""
    scale_pos_weight = _get_scale_pos_weight(y_train)
    class_weights = "balanced"

    models = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, class_weight=class_weights, random_state=42),
            {"C": [0.01, 0.1, 1.0]},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=200, class_weight=class_weights, n_jobs=-1, random_state=42),
            {"max_depth": [8, 12, None], "min_samples_split": [2, 5]},
        ),
        "xgboost": (
            XGBClassifier(use_label_encoder=False, eval_metric="logloss", scale_pos_weight=scale_pos_weight, n_jobs=-1, random_state=42),
            {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
        ),
    }

    results = {}
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    for name, (model, params) in models.items():
        grid = GridSearchCV(
            estimator=model,
            param_grid=params,
            scoring="average_precision",
            cv=splitter,
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        results[name] = {
            "model": grid.best_estimator_,
            "best_params": grid.best_params_,
            "best_score": grid.best_score_,
        }

    return results


def evaluate_model(name: str, model, X_test, y_test) -> Dict[str, object]:
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)
    metrics = compute_classification_metrics(y_test, predictions, probabilities)
    metrics["confusion_matrix"] = compute_confusion_matrix(y_test, predictions)
    metrics["model_name"] = name
    return metrics


def select_best_model(results: Dict[str, Dict]) -> Tuple[str, object, Dict]:
    best_name, best_score = None, -1.0
    for name, entry in results.items():
        if entry["metrics"]["pr_auc"] > best_score:
            best_score = entry["metrics"]["pr_auc"]
            best_name = name
    return best_name, results[best_name]["model"], results[best_name]["metrics"]


def persist_model(model, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
