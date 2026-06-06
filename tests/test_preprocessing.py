import pandas as pd

from src.data.preprocessing import build_preprocessing_pipeline, split_data


def test_split_data_preserves_shape():
    df = pd.DataFrame(
        {
            "V1": [0.1, 0.2, 0.3, 0.4],
            "V2": [1.1, 1.2, 1.3, 1.4],
            "Amount": [10.0, 20.0, 30.0, 40.0],
            "Class": [0, 1, 0, 1],
        }
    )
    X_train, X_test, y_train, y_test = split_data(df, target_column="Class", test_size=0.5, random_state=0)

    assert len(X_train) == 2
    assert len(X_test) == 2
    assert set(X_train.columns) == {"V1", "V2", "Amount"}
    assert y_train.dtype == int or y_train.dtype == object


def test_build_preprocessing_pipeline_transforms_data():
    df = pd.DataFrame(
        {
            "V1": [1.0, 2.0, 3.0],
            "V2": [4.0, 5.0, 6.0],
            "Amount": [10.0, 20.0, 30.0],
        }
    )
    pipeline = build_preprocessing_pipeline(["V1", "V2", "Amount"])
    transformed = pipeline.fit_transform(df)

    assert transformed.shape == (3, 3)
    assert abs(transformed.mean()) < 1e-6
