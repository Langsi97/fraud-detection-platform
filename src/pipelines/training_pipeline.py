import json
from pathlib import Path

import mlflow
from mlflow import sklearn as mlflow_sklearn

from src.data.load_data import prepare_dataset
from src.data.preprocessing import build_preprocessing_pipeline, serialize_preprocessor, split_data
from src.models.evaluate import save_feature_importance, save_metrics
from src.models.train import evaluate_model, persist_model, select_best_model, train_candidate_models
from src.monitoring.drift_detection import generate_drift_report, summarize_feature_drift


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_CANDIDATES = [
    ROOT_DIR / "data" / "raw" / "phpKo8OWT.arff",
    ROOT_DIR / "data" / "raw" / "credit_card_fraud.xlsx",
    ROOT_DIR / "data" / "raw" / "credit_card_fraud.csv",
]
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "processed_data.csv"
PREPROCESSOR_PATH = ROOT_DIR / "models" / "preprocessor.pkl"
BEST_MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"
METRICS_PATH = ROOT_DIR / "models" / "metrics.json"
FEATURE_IMPORTANCE_PATH = ROOT_DIR / "models" / "feature_importance.csv"
DRIFT_REPORT_PATH = ROOT_DIR / "reports" / "monitoring" / "data_drift_report.html"
DRIFT_SUMMARY_PATH = ROOT_DIR / "tableau" / "dashboard_data" / "feature_drift_summary.csv"


def get_raw_data_path() -> Path:
    for path in RAW_DATA_CANDIDATES:
        if path.exists():
            return path
    candidate_paths = ", ".join(str(path) for path in RAW_DATA_CANDIDATES)
    raise FileNotFoundError(
        f"No supported raw dataset found. Checked: {candidate_paths}."
    )


def run_training_pipeline():
    required_columns = [
        "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
        "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
        "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
        "Amount", "Class",
    ]
    raw_data_path = get_raw_data_path()
    df = prepare_dataset(str(raw_data_path), str(PROCESSED_DATA_PATH), required_columns)

    X_train, X_test, y_train, y_test = split_data(df, target_column="Class")
    feature_columns = list(X_train.columns)
    preprocessor = build_preprocessing_pipeline(feature_columns)
    preprocessor.fit(X_train)

    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    mlflow.set_tracking_uri(Path(root_mlflow_dir()).as_uri())
    mlflow.set_experiment("credit_card_fraud")

    with mlflow.start_run(run_name="training_run") as run:
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("validation_samples", len(X_test))
        mlflow.log_param("feature_count", len(feature_columns))

        candidate_results = train_candidate_models(X_train_transformed, y_train)
        best_results = {}

        for model_name, result in candidate_results.items():
            metrics = evaluate_model(model_name, result["model"], X_test_transformed, y_test)
            result["metrics"] = metrics
            mlflow.log_metric(f"{model_name}_pr_auc", metrics["pr_auc"])
            mlflow.log_metric(f"{model_name}_roc_auc", metrics["roc_auc"])
            best_results[model_name] = result

        best_name, best_model, best_metrics = select_best_model(best_results)

        persist_model(best_model, str(BEST_MODEL_PATH))
        serialize_preprocessor(preprocessor, str(PREPROCESSOR_PATH))

        save_metrics({model: entry["metrics"] for model, entry in best_results.items()}, str(METRICS_PATH))
        feature_importances = compute_feature_importance(best_model, feature_columns)
        save_feature_importance(feature_columns, feature_importances, str(FEATURE_IMPORTANCE_PATH))

        mlflow_sklearn.log_model(best_model, "model")
        mlflow.log_artifact(str(METRICS_PATH))
        mlflow.log_artifact(str(FEATURE_IMPORTANCE_PATH))

        generate_drift_report(X_train, X_test, str(DRIFT_REPORT_PATH))
        summarize_feature_drift(X_train, X_test, str(DRIFT_SUMMARY_PATH))

        mlflow.log_param("best_model", best_name)
        mlflow.log_artifact(str(DRIFT_REPORT_PATH))
        mlflow.log_artifact(str(DRIFT_SUMMARY_PATH))

        print(f"Best model: {best_name}")
        print(json.dumps(best_metrics, indent=2))


def compute_feature_importance(model, feature_columns):
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_.tolist()
    if hasattr(model, "coef_"):
        coef = model.coef_.ravel().tolist()
        return [abs(value) for value in coef]
    return [0.0] * len(feature_columns)


def root_mlflow_dir() -> str:
    root = ROOT_DIR / "mlruns"
    root.mkdir(parents=True, exist_ok=True)
    return str(root.resolve())


if __name__ == "__main__":
    run_training_pipeline()
