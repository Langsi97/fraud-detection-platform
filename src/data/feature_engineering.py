import pandas as pd


def extract_features_and_target(df: pd.DataFrame, target_column: str = "Class"):
    """Split the dataset into features and target arrays."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def feature_summary(df: pd.DataFrame, target_column: str = "Class") -> pd.DataFrame:
    """Compute feature-level summary statistics for reporting."""
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    return df[numeric_columns].describe().transpose()
