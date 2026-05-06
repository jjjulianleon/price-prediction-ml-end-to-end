import pandas as pd

import pytest

from src.features.build_features import (
    build_candidate_modeling_frame,
    DISTANCE_COLUMN,
    FEATURE_CONTRACT_VERSION,
    LEAKAGE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    normalize_raw_taxi_frame,
    RAW_SOURCE_COLUMN_MAP,
    raw_quality_mask,
    TARGET_COLUMN,
    assert_no_leakage_columns,
    get_feature_audit_payload,
    get_feature_pipeline,
    prepare_feature_frame,
    split_features_target,
)


def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_type": ["yellow", "green"],
            "pickup_datetime": ["2025-01-03 08:15:00", "2025-01-26 17:45:00"],
            "passenger_count": [1, 2],
            DISTANCE_COLUMN: [1.5, 7.2],
            "pickup_location_id": [161, 237],
            "dropoff_location_id": [237, 142],
            "vendor_id": [1, 2],
            "ratecode_id": [1, 1],
            TARGET_COLUMN: [12.5, 31.8],
        }
    )


def test_leakage_columns_are_rejected():
    assert "total_amount" in LEAKAGE_COLUMNS
    assert "tpep_dropoff_datetime" in LEAKAGE_COLUMNS
    with pytest.raises(ValueError):
        assert_no_leakage_columns([DISTANCE_COLUMN, "tip_amount"])


def test_prepare_feature_frame_creates_temporal_features():
    features = prepare_feature_frame(sample_dataframe().drop(columns=[TARGET_COLUMN]))

    assert list(features.columns) == MODEL_FEATURE_COLUMNS
    assert features["trip_type"].tolist() == ["yellow", "green"]
    assert features["pickup_hour"].tolist() == [8, 17]
    assert features["pickup_month"].tolist() == [1, 1]
    assert "log_estimated_distance" in features.columns
    assert "route_id" in features.columns
    assert "same_zone" in features.columns


def test_feature_pipeline_transforms_small_sample():
    df = sample_dataframe()
    X, y = split_features_target(df)
    pipeline = get_feature_pipeline()
    transformed = pipeline.fit_transform(X, y)

    assert transformed.shape[0] == len(df)


def test_feature_audit_payload_matches_contract():
    audit = get_feature_audit_payload()

    assert audit["feature_contract_version"] == FEATURE_CONTRACT_VERSION
    assert audit["raw_source_column_map"] == RAW_SOURCE_COLUMN_MAP
    assert DISTANCE_COLUMN in audit["raw_feature_columns"]
    assert "trip_duration_min" in audit["diagnostic_only_columns"]


def test_normalize_raw_taxi_frame_maps_raw_schema():
    raw_df = pd.DataFrame(
        {
            "tpep_pickup_datetime": ["2025-01-03 08:15:00"],
            "tpep_dropoff_datetime": ["2025-01-03 08:35:00"],
            "trip_distance": [1.5],
            "passenger_count": [2],
            "vendorid": [1],
            "ratecodeid": [1],
            "pulocationid": [161],
            "dolocationid": [237],
            TARGET_COLUMN: [12.5],
        }
    )

    normalized = normalize_raw_taxi_frame(raw_df)

    assert "pickup_datetime" in normalized.columns
    assert "dropoff_datetime" in normalized.columns
    assert DISTANCE_COLUMN in normalized.columns
    assert normalized.loc[0, "pickup_location_id"] == 161


def test_build_candidate_modeling_frame_filters_invalid_rows():
    raw_df = pd.DataFrame(
        {
            "tpep_pickup_datetime": ["2025-01-03 08:15:00", "2025-01-04 09:00:00"],
            "tpep_dropoff_datetime": ["2025-01-03 08:35:00", "2025-01-04 08:55:00"],
            "trip_type": ["yellow", "green"],
            "trip_distance": [1.5, 2.0],
            "passenger_count": [2, 1],
            "vendorid": [1, 2],
            "ratecodeid": [1, 1],
            "pulocationid": [161, 132],
            "dolocationid": [237, 132],
            TARGET_COLUMN: [12.5, 10.0],
        }
    )

    mask = raw_quality_mask(raw_df, start_date="2025-01-01", end_date="2025-01-31")
    filtered = build_candidate_modeling_frame(
        raw_df,
        start_date="2025-01-01",
        end_date="2025-01-31",
    )

    assert mask.tolist() == [True, False]
    assert len(filtered) == 1
    assert filtered.iloc[0][DISTANCE_COLUMN] == 1.5
    assert filtered.iloc[0]["trip_type"] == "yellow"
