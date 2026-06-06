import os
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_feature_columns(df: pd.DataFrame, target_column: str = "Class") -> List[str]:
    return [col for col in df.columns if col != target_column]


def split_data(df: pd.DataFrame, target_column: str = "Class", test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features = get_feature_columns(df, target_column)
    X = df[features]
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def build_preprocessing_pipeline(feature_columns: List[str]) -> Pipeline:
    numeric_transformer = Pipeline([
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, feature_columns),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
    ])
    return pipeline


def validate_feature_set(expected_features: List[str], current_features: List[str]) -> None:
    missing = [col for col in expected_features if col not in current_features]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")


def serialize_preprocessor(pipeline: Pipeline, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(pipeline, output_path)


def load_preprocessor(file_path: str) -> Pipeline:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Preprocessing pipeline not found: {file_path}")
    return joblib.load(file_path)
