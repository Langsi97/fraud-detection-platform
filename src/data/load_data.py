import os
from pathlib import Path
from typing import List

import liac_arff
import numpy as np
import pandas as pd


def _decode_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def load_arff(file_path: str) -> pd.DataFrame:
    """Load an ARFF file and return a Pandas DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ARFF file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as handle:
        dataset = liac_arff.load(handle)

    columns = [name for name, _ in dataset["attributes"]]
    values = []
    for row in dataset["data"]:
        decoded = [_decode_value(x) for x in row]
        values.append(decoded)

    df = pd.DataFrame(values, columns=columns)
    return df


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load a raw dataset from ARFF, CSV, or Excel."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw dataset not found: {file_path}")

    extension = Path(file_path).suffix.lower()
    if extension == ".arff":
        return load_arff(file_path)
    if extension == ".csv":
        return pd.read_csv(file_path)
    if extension in {".xls", ".xlsx"}:
        return pd.read_excel(file_path)

    raise ValueError(
        f"Unsupported raw data format '{extension}'. Provide .arff, .csv, .xls, or .xlsx."
    )


def validate_schema(df: pd.DataFrame, expected_columns: List[str]) -> None:
    """Validate that the DataFrame contains the expected schema."""
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.isnull().any().any():
        raise ValueError("Input dataset contains missing values. Please clean or impute before proceeding.")


def clean_dataframe(df: pd.DataFrame, target_column: str = "Class") -> pd.DataFrame:
    """Convert numeric columns, fill missing values, and ensure the target is integer."""
    df = df.copy()
    numeric_columns = [col for col in df.columns if col != target_column]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if target_column in df.columns:
        df[target_column] = pd.to_numeric(df[target_column], errors="coerce").astype(int)

    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    return df


def prepare_dataset(raw_path: str, processed_path: str, required_columns: List[str]) -> pd.DataFrame:
    """Load raw data, validate schema, clean, and save a processed CSV."""
    df = load_raw_data(raw_path)
    validate_schema(df, required_columns)
    df = clean_dataframe(df, target_column="Class")
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    return df
