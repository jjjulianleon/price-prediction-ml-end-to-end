"""Feature selection and preprocessing helpers for NYC Taxi fare modeling."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "fare_amount"
RAW_FEATURE_COLUMNS = [
    "pickup_datetime",
    "passenger_count",
    "trip_distance",
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
    "passenger_count",
    "trip_distance",
    "pickup_location_id",
    "dropoff_location_id",
    "vendor_id",
    "ratecode_id",
]
LEAKAGE_COLUMNS = {
    "payment_type",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "congestion_surcharge",
    "airport_fee",
    "mta_tax",
    "total_amount",
    "trip_duration_minutes",
    "avg_speed_mph",
    "tip_pct",
    "fare_per_mile",
}
NUMERIC_FEATURES = [
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "is_weekend",
    "passenger_count",
    "trip_distance",
]
CATEGORICAL_FEATURES = [
    "pickup_location_id",
    "dropoff_location_id",
    "vendor_id",
    "ratecode_id",
]


class PickupTimeFeatures(BaseEstimator, TransformerMixin):
    """Derive temporal features from pickup timestamp and select the modeling frame."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return prepare_feature_frame(X)


def assert_no_leakage_columns(columns: Iterable[str]) -> None:
    forbidden = sorted(set(columns).intersection(LEAKAGE_COLUMNS))
    if forbidden:
        joined = ", ".join(forbidden)
        raise ValueError(f"Leakage columns found in feature set: {joined}")


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    transformed = df.copy()
    pickup_dt = pd.to_datetime(transformed["pickup_datetime"], errors="coerce")
    transformed["pickup_hour"] = pickup_dt.dt.hour
    transformed["pickup_dayofweek"] = pickup_dt.dt.dayofweek
    transformed["pickup_month"] = pickup_dt.dt.month
    transformed["is_weekend"] = pickup_dt.dt.dayofweek.isin([5, 6]).astype("Int64")
    return transformed


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    assert_no_leakage_columns(df.columns)

    missing = sorted(set(RAW_FEATURE_COLUMNS).difference(df.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required feature columns: {missing_text}")

    transformed = add_temporal_features(df)
    return transformed.loc[:, MODEL_FEATURE_COLUMNS].copy()


def select_raw_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    assert_no_leakage_columns(df.columns)
    missing = sorted(set(RAW_FEATURE_COLUMNS).difference(df.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required feature columns: {missing_text}")
    return df.loc[:, RAW_FEATURE_COLUMNS].copy()


def split_features_target(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise ValueError(f"Target column `{target_column}` not found in DataFrame.")
    X = select_raw_feature_columns(df.drop(columns=[target_column]))
    y = pd.to_numeric(df[target_column], errors="coerce")
    return X, y


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
