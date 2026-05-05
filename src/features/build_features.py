"""Feature selection and preprocessing helpers for NYC Taxi fare modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "fare_amount"
FEATURE_CONTRACT_VERSION = "v2_estimated_distance"
LEGACY_DISTANCE_COLUMN = "trip_distance"
DISTANCE_COLUMN = "estimated_distance"
RAW_SOURCE_COLUMN_MAP = {
    "tpep_pickup_datetime": "pickup_datetime",
    "vendorid": "vendor_id",
    "ratecodeid": "ratecode_id",
    "pulocationid": "pickup_location_id",
    "dolocationid": "dropoff_location_id",
}
RAW_FEATURE_COLUMNS = [
    "pickup_datetime",
    "passenger_count",
    DISTANCE_COLUMN,
    "pickup_location_id",
    "dropoff_location_id",
    "vendor_id",
    "ratecode_id",
]
MODEL_FEATURE_COLUMNS = [
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "is_weekend",
    "is_rush_hour",
    "is_night",
    "passenger_count",
    DISTANCE_COLUMN,
    "log_estimated_distance",
    "pickup_location_id",
    "dropoff_location_id",
    "vendor_id",
    "ratecode_id",
    "route_id",
    "same_zone",
]
LEAKAGE_COLUMNS = {
    "payment_type",
    "tip_amount",
    "tolls_amount",
    "mta_tax",
    "extra",
    "improvement_surcharge",
    "congestion_surcharge",
    "airport_fee",
    "total_amount",
    "tpep_dropoff_datetime",
    "trip_duration_min",
    "trip_duration_minutes",
    "speed_mph",
    "avg_speed_mph",
    "tip_pct",
    "fare_per_mile",
}
NUMERIC_FEATURES = [
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "is_weekend",
    "is_rush_hour",
    "is_night",
    "passenger_count",
    DISTANCE_COLUMN,
    "log_estimated_distance",
    "same_zone",
]
CATEGORICAL_FEATURES = [
    "pickup_location_id",
    "dropoff_location_id",
    "vendor_id",
    "ratecode_id",
    "route_id",
]
DIAGNOSTIC_ONLY_COLUMNS = [
    "tpep_dropoff_datetime",
    "trip_duration_min",
    "speed_mph",
    "payment_type",
    "tip_amount",
    "tolls_amount",
    "mta_tax",
    "extra",
    "improvement_surcharge",
    "congestion_surcharge",
    "airport_fee",
    "total_amount",
]


class PickupTimeFeatures(BaseEstimator, TransformerMixin):
    """Derive temporal features from pickup timestamp and select the modeling frame."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return prepare_feature_frame(X)


def assert_no_leakage_columns(columns) -> None:
    forbidden = sorted(set(columns).intersection(LEAKAGE_COLUMNS))
    if forbidden:
        joined = ", ".join(forbidden)
        raise ValueError(f"Leakage columns found in feature set: {joined}")


def with_official_distance_name(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if DISTANCE_COLUMN not in normalized.columns and LEGACY_DISTANCE_COLUMN in normalized.columns:
        normalized[DISTANCE_COLUMN] = normalized[LEGACY_DISTANCE_COLUMN]
    return normalized


def normalize_raw_taxi_frame(df: pd.DataFrame) -> pd.DataFrame:
    normalized = with_official_distance_name(df)
    normalized = normalized.rename(
        columns={
            source: target
            for source, target in RAW_SOURCE_COLUMN_MAP.items()
            if source in normalized.columns and target not in normalized.columns
        }
    )
    return normalized


def raw_quality_mask(
    df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    require_positive_fare: bool = True,
) -> pd.Series:
    normalized = normalize_raw_taxi_frame(df)
    pickup_dt = pd.to_datetime(normalized.get("pickup_datetime"), errors="coerce")
    dropoff_dt = pd.to_datetime(normalized.get("tpep_dropoff_datetime"), errors="coerce")
    distance = pd.to_numeric(normalized.get(DISTANCE_COLUMN), errors="coerce")
    fare_amount = pd.to_numeric(normalized.get(TARGET_COLUMN), errors="coerce")
    passenger_count = pd.to_numeric(normalized.get("passenger_count"), errors="coerce")

    mask = (
        pickup_dt.notna()
        & dropoff_dt.notna()
        & (dropoff_dt > pickup_dt)
        & (distance > 0)
        & passenger_count.between(1, 6)
        & normalized.get("pickup_location_id").notna()
        & normalized.get("dropoff_location_id").notna()
    )

    if require_positive_fare:
        mask = mask & (fare_amount > 0)
    else:
        mask = mask & fare_amount.notna() & (fare_amount >= 0)

    if start_date is not None and end_date is not None:
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        mask = mask & pickup_dt.between(start_dt, end_dt)

    return mask.fillna(False)


def build_candidate_modeling_frame(
    df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    require_positive_fare: bool = True,
) -> pd.DataFrame:
    normalized = normalize_raw_taxi_frame(df)
    mask = raw_quality_mask(
        normalized,
        start_date=start_date,
        end_date=end_date,
        require_positive_fare=require_positive_fare,
    )
    return normalized.loc[mask].copy()


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    transformed = normalize_raw_taxi_frame(df)
    pickup_dt = pd.to_datetime(transformed["pickup_datetime"], errors="coerce")
    transformed["pickup_hour"] = pickup_dt.dt.hour
    transformed["pickup_dayofweek"] = pickup_dt.dt.dayofweek
    transformed["pickup_month"] = pickup_dt.dt.month
    transformed["is_weekend"] = pickup_dt.dt.dayofweek.isin([5, 6]).astype("Int64")
    transformed["is_rush_hour"] = pickup_dt.dt.hour.isin([7, 8, 9, 16, 17, 18, 19]).astype("Int64")
    transformed["is_night"] = pickup_dt.dt.hour.isin([22, 23, 0, 1, 2, 3, 4, 5]).astype("Int64")
    transformed["log_estimated_distance"] = np.log1p(
        pd.to_numeric(transformed[DISTANCE_COLUMN], errors="coerce")
    )
    transformed["route_id"] = (
        transformed["pickup_location_id"].astype("Int64").astype(str)
        + "_"
        + transformed["dropoff_location_id"].astype("Int64").astype(str)
    )
    transformed["same_zone"] = (
        transformed["pickup_location_id"] == transformed["dropoff_location_id"]
    ).astype("Int64")
    return transformed


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    assert_no_leakage_columns(df.columns)
    transformed = add_temporal_features(df)
    missing = sorted(set(RAW_FEATURE_COLUMNS).difference(transformed.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required feature columns: {missing_text}")
    return transformed.loc[:, MODEL_FEATURE_COLUMNS].copy()


def select_raw_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    assert_no_leakage_columns(df.columns)
    transformed = normalize_raw_taxi_frame(df)
    missing = sorted(set(RAW_FEATURE_COLUMNS).difference(transformed.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required feature columns: {missing_text}")
    return transformed.loc[:, RAW_FEATURE_COLUMNS].copy()


def split_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise ValueError(f"Target column `{target_column}` not found in DataFrame.")
    X = select_raw_feature_columns(df.drop(columns=[target_column]))
    y = pd.to_numeric(df[target_column], errors="coerce")
    return X, y


def get_feature_audit_payload() -> dict[str, object]:
    return {
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "raw_source_column_map": RAW_SOURCE_COLUMN_MAP,
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "diagnostic_only_columns": DIAGNOSTIC_ONLY_COLUMNS,
        "leakage_columns": sorted(LEAKAGE_COLUMNS),
    }


def get_feature_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("pickup_time_features", PickupTimeFeatures()),
            ("preprocessor", preprocessor),
        ]
    )
