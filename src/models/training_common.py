"""Shared training helpers for experiment and production pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.data.ingestion import fetch_data_in_batches
from src.features.build_features import (
    TARGET_COLUMN,
    get_feature_pipeline,
    split_features_target,
)
from src.utils.config import Settings


@dataclass(frozen=True)
class EstimatorSpec:
    name: str
    build_estimator: Callable[[], Any]
    training_strategy: str
    description: str
    matrix_format: str = "sparse"
    required: bool = False
    dependency: str | None = None
    rubric_status: str = "optional"


def iter_date_windows(start_date: str, end_date: str, grain: str) -> list[tuple[str, str]]:
    start_dt = date.fromisoformat(start_date)
    end_dt = date.fromisoformat(end_date)
    windows: list[tuple[str, str]] = []
    current = start_dt

    while current <= end_dt:
        if grain == "week":
            window_end = min(current + timedelta(days=6), end_dt)
        else:
            if current.month == 12:
                next_month = current.replace(year=current.year + 1, month=1, day=1)
            else:
                next_month = current.replace(month=current.month + 1, day=1)
            window_end = min(next_month - timedelta(days=1), end_dt)

        windows.append((current.isoformat(), window_end.isoformat()))
        current = window_end + timedelta(days=1)

    return windows


def split_window_queries(
    table_name: str,
    start_date: str,
    end_date: str,
    grain: str,
) -> list[str]:
    queries: list[str] = []
    for window_start, window_end in iter_date_windows(start_date, end_date, grain):
        queries.append(
            f"""
            SELECT * FROM {table_name}
            WHERE CAST(pickup_datetime AS DATE) BETWEEN TO_DATE('{window_start}') AND TO_DATE('{window_end}')
            ORDER BY pickup_datetime
            """.strip()
        )
    return queries


def split_ranges(settings: Settings) -> dict[str, tuple[str, str]]:
    train_end = date.fromisoformat(settings.train_end_date)
    val_end = date.fromisoformat(settings.val_end_date)
    data_end = date.fromisoformat(settings.data_end_date)
    return {
        "train": (settings.data_start_date, settings.train_end_date),
        "validation": ((train_end + timedelta(days=1)).isoformat(), settings.val_end_date),
        "test": ((val_end + timedelta(days=1)).isoformat(), data_end.isoformat()),
    }


def iter_split_batches(
    table_name: str,
    start_date: str,
    end_date: str,
    grain: str,
    batch_size: int,
    settings: Settings | None = None,
):
    for query in split_window_queries(table_name, start_date, end_date, grain):
        yield from fetch_data_in_batches(query, batch_size=batch_size, settings=settings)


def fit_preprocessor_from_sample(sample_df: pd.DataFrame):
    X_sample, _ = split_features_target(sample_df)
    pipeline = get_feature_pipeline()
    pipeline.fit(X_sample)
    return pipeline


TRIP_TYPE_WEIGHT_COLUMN = "trip_type"


def compute_trip_type_weights(trip_type_series: pd.Series) -> dict[str, float]:
    """Inverse-frequency weights per trip_type, normalized so the mean weight equals 1.0."""
    counts = trip_type_series.astype(str).str.lower().value_counts()
    if counts.empty:
        return {}
    total = float(counts.sum())
    n_classes = float(len(counts))
    return {trip_type: total / (n_classes * float(count)) for trip_type, count in counts.items()}


def trip_type_weights_for_frame(
    df: pd.DataFrame,
    weight_map: dict[str, float] | None,
) -> np.ndarray | None:
    if not weight_map or TRIP_TYPE_WEIGHT_COLUMN not in df.columns:
        return None
    series = df[TRIP_TYPE_WEIGHT_COLUMN].astype(str).str.lower()
    weights = series.map(weight_map).astype(float).fillna(1.0)
    return weights.to_numpy()


def materialize_matrix(matrix, matrix_format: str):
    if matrix_format == "dense" and hasattr(matrix, "toarray"):
        return matrix.toarray()
    return matrix


def train_incremental_snowflake(
    table_name: str,
    start_date: str,
    end_date: str,
    grain: str,
    preprocessor,
    estimator,
    matrix_format: str = "sparse",
    batch_size: int = 50_000,
    settings: Settings | None = None,
    log_fn: Callable[[str], None] | None = None,
    log_every_n_batches: int = 5,
    trip_type_weights: dict[str, float] | None = None,
):
    seen_rows = 0
    batch_counter = 0
    sample_weight_supported = trip_type_weights is not None
    for window_idx, (window_start, window_end) in enumerate(
        iter_date_windows(start_date, end_date, grain),
        start=1,
    ):
        if log_fn:
            log_fn(f"Train window {window_idx}: {window_start} -> {window_end}")
        query = (
            f"SELECT * FROM {table_name} "
            f"WHERE CAST(pickup_datetime AS DATE) BETWEEN TO_DATE('{window_start}') AND TO_DATE('{window_end}') "
            "ORDER BY pickup_datetime"
        )
        for batch_df in fetch_data_in_batches(query, batch_size=batch_size, settings=settings):
            if batch_df.empty:
                continue
            X_batch, y_batch = split_features_target(batch_df)
            X_transformed = materialize_matrix(preprocessor.transform(X_batch), matrix_format)
            batch_weights = trip_type_weights_for_frame(batch_df, trip_type_weights)
            if sample_weight_supported and batch_weights is not None:
                try:
                    estimator.partial_fit(X_transformed, y_batch, sample_weight=batch_weights)
                except TypeError:
                    sample_weight_supported = False
                    if log_fn:
                        log_fn(
                            "Estimator partial_fit does not accept sample_weight; "
                            "falling back to unweighted incremental training."
                        )
                    estimator.partial_fit(X_transformed, y_batch)
            else:
                estimator.partial_fit(X_transformed, y_batch)
            seen_rows += len(batch_df)
            batch_counter += 1
            if log_fn and batch_counter % log_every_n_batches == 0:
                log_fn(f"Train batches={batch_counter} rows_seen={seen_rows:,}")

    if seen_rows == 0:
        raise ValueError("Training query returned no rows.")
    return estimator


def evaluate_model(
    model,
    preprocessor,
    table_name: str,
    start_date: str,
    end_date: str,
    grain: str,
    matrix_format: str = "sparse",
    batch_size: int = 50_000,
    settings: Settings | None = None,
    split_name: str = "evaluation",
    log_fn: Callable[[str], None] | None = None,
    log_every_n_batches: int = 5,
) -> float:
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    total_rows = 0
    start_ts = perf_counter()

    if log_fn:
        log_fn(
            f"Evaluating {split_name} | matrix_format={matrix_format} | batch_size={batch_size}"
        )

    for window_idx, (window_start, window_end) in enumerate(
        iter_date_windows(start_date, end_date, grain),
        start=1,
    ):
        if log_fn:
            log_fn(f"{split_name} window {window_idx}: {window_start} -> {window_end}")
        query = (
            f"SELECT * FROM {table_name} "
            f"WHERE CAST(pickup_datetime AS DATE) BETWEEN TO_DATE('{window_start}') AND TO_DATE('{window_end}') "
            "ORDER BY pickup_datetime"
        )
        for batch_idx, batch_df in enumerate(
            fetch_data_in_batches(query, batch_size=batch_size, settings=settings),
            start=1,
        ):
            if batch_df.empty:
                continue
            X_batch, y_batch = split_features_target(batch_df)
            X_transformed = materialize_matrix(preprocessor.transform(X_batch), matrix_format)
            predictions = model.predict(X_transformed)
            y_true_parts.append(y_batch.to_numpy())
            y_pred_parts.append(np.asarray(predictions))
            total_rows += len(batch_df)
            if log_fn and batch_idx % log_every_n_batches == 0:
                log_fn(f"{split_name} batches={batch_idx} in current window")

    if not y_true_parts:
        raise ValueError("Evaluation query returned no rows.")

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if log_fn:
        elapsed = perf_counter() - start_ts
        log_fn(
            f"Completed {split_name} | rows={total_rows:,} | rmse={rmse:.4f} | elapsed={elapsed:.1f}s"
        )
    return rmse


def save_artifact(
    model,
    preprocessor,
    model_name: str,
    metrics: dict[str, object],
    feature_audit: dict[str, object],
    zone_lookup_enabled: bool,
    output_path: str | Path,
    input_matrix_format: str,
    artifact_role: str,
    extra_metadata: dict[str, object] | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "preprocessor": preprocessor,
        "target_column": TARGET_COLUMN,
        "model_name": model_name,
        "input_matrix_format": input_matrix_format,
        "artifact_role": artifact_role,
        "feature_audit": feature_audit,
        "zone_lookup_enabled": zone_lookup_enabled,
        "metrics": metrics,
    }
    if extra_metadata:
        artifact["training_metadata"] = extra_metadata
    joblib.dump(artifact, path)
    return path
