import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.features.build_features import TARGET_COLUMN, get_feature_pipeline, split_features_target
from src.models.predict_model import predict


def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_type": ["yellow", "green", "yellow"],
            "pickup_datetime": [
                "2025-01-03 08:15:00",
                "2025-01-26 17:45:00",
                "2025-01-27 06:10:00",
            ],
            "passenger_count": [1, 2, 1],
            "estimated_distance": [1.5, 7.2, 3.1],
            "pickup_location_id": [161, 237, 132],
            "dropoff_location_id": [237, 142, 170],
            "vendor_id": [1, 2, 1],
            "ratecode_id": [1, 1, 1],
            TARGET_COLUMN: [12.5, 31.8, 18.4],
        }
    )


def test_predict_supports_dense_artifacts():
    df = sample_dataframe()
    X, y = split_features_target(df)
    preprocessor = get_feature_pipeline()
    transformed = preprocessor.fit_transform(X, y)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    model = GradientBoostingRegressor(random_state=42).fit(transformed, y)

    artifact = {
        "model": model,
        "preprocessor": preprocessor,
        "input_matrix_format": "dense",
    }

    predictions = predict(artifact, df.drop(columns=[TARGET_COLUMN]))

    assert len(predictions) == len(df)
    assert all(isinstance(value, float) for value in predictions)
