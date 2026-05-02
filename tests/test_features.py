import pandas as pd

import pytest

from src.features.build_features import (
    LEAKAGE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    assert_no_leakage_columns,
    get_feature_pipeline,
    prepare_feature_frame,
    split_features_target,
)


def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pickup_datetime": ["2025-01-03 08:15:00", "2025-01-26 17:45:00"],
            "passenger_count": [1, 2],
            "trip_distance": [1.5, 7.2],
            "pickup_location_id": [161, 237],
            "dropoff_location_id": [237, 142],
            "vendor_id": [1, 2],
            "ratecode_id": [1, 1],
            TARGET_COLUMN: [12.5, 31.8],
        }
    )


def test_leakage_columns_are_rejected():
    assert "total_amount" in LEAKAGE_COLUMNS
    with pytest.raises(ValueError):
        assert_no_leakage_columns(["trip_distance", "tip_amount"])


def test_prepare_feature_frame_creates_temporal_features():
    features = prepare_feature_frame(sample_dataframe().drop(columns=[TARGET_COLUMN]))

    assert list(features.columns) == MODEL_FEATURE_COLUMNS
    assert features["pickup_hour"].tolist() == [8, 17]
    assert features["pickup_month"].tolist() == [1, 1]


def test_feature_pipeline_transforms_small_sample():
    df = sample_dataframe()
    X, y = split_features_target(df)
    pipeline = get_feature_pipeline()
    transformed = pipeline.fit_transform(X, y)

    assert transformed.shape[0] == len(df)
