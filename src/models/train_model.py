"""Training entrypoint for baseline and incremental NYC Taxi fare models."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

from src.data.ingestion import fetch_data_in_batches, fetch_sample
from src.features.build_features import TARGET_COLUMN, get_feature_pipeline, split_features_target
from src.utils.config import Settings, ensure_model_dir, get_settings


def fit_preprocessor_from_sample(sample_df: pd.DataFrame):
    X_sample, _ = split_features_target(sample_df)
    pipeline = get_feature_pipeline()
    pipeline.fit(X_sample)
    return pipeline


def train_dummy_regressor(sample_df: pd.DataFrame):
    X_sample, y_sample = split_features_target(sample_df)
    preprocessor = get_feature_pipeline()
    X_transformed = preprocessor.fit_transform(X_sample)
    model = DummyRegressor(strategy="mean")
    model.fit(X_transformed, y_sample)
    return preprocessor, model


def train_incremental_model(
    train_query: str,
    preprocessor,
    batch_size: int = 50_000,
    settings: Settings | None = None,
) -> SGDRegressor:
    model = SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=0.0001,
        learning_rate="invscaling",
        eta0=0.01,
        random_state=42,
    )

    seen_rows = 0
    for batch_df in fetch_data_in_batches(train_query, batch_size=batch_size, settings=settings):
        if batch_df.empty:
            continue

        X_batch, y_batch = split_features_target(batch_df)
        X_transformed = preprocessor.transform(X_batch)
        model.partial_fit(X_transformed, y_batch)
        seen_rows += len(batch_df)

    if seen_rows == 0:
        raise ValueError("Training query returned no rows.")
    return model


def train_hist_gradient_boosting(sample_df: pd.DataFrame):
    X_sample, y_sample = split_features_target(sample_df)
    preprocessor = get_feature_pipeline()
    X_transformed = preprocessor.fit_transform(X_sample, y_sample)
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=150,
        min_samples_leaf=50,
        random_state=42,
    )
    model.fit(X_transformed, y_sample)
    return preprocessor, model


def evaluate_model(
    model,
    preprocessor,
    query: str,
    batch_size: int = 50_000,
    settings: Settings | None = None,
) -> float:
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []

    for batch_df in fetch_data_in_batches(query, batch_size=batch_size, settings=settings):
        if batch_df.empty:
            continue

        X_batch, y_batch = split_features_target(batch_df)
        X_transformed = preprocessor.transform(X_batch)
        predictions = model.predict(X_transformed)
        y_true_parts.append(y_batch.to_numpy())
        y_pred_parts.append(np.asarray(predictions))

    if not y_true_parts:
        raise ValueError("Evaluation query returned no rows.")

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def save_artifact(model, preprocessor, metrics: dict[str, float], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "preprocessor": preprocessor,
        "target_column": TARGET_COLUMN,
        "metrics": metrics,
    }
    joblib.dump(artifact, path)
    return path


def train_out_of_core(
    settings: Settings | None = None,
    sample_limit: int = 5_000,
    batch_size: int = 50_000,
) -> dict[str, object]:
    effective_settings = settings or get_settings()

    train_query = f"SELECT * FROM {effective_settings.train_table} ORDER BY pickup_datetime"
    val_query = f"SELECT * FROM {effective_settings.val_table} ORDER BY pickup_datetime"
    test_query = f"SELECT * FROM {effective_settings.test_table} ORDER BY pickup_datetime"

    train_sample = fetch_sample(train_query, limit=sample_limit, settings=effective_settings)
    if train_sample.empty:
        raise ValueError("Train sample is empty. Execute Snowflake SQL scripts before training.")

    dummy_preprocessor, dummy_model = train_dummy_regressor(train_sample)
    dummy_metrics = {
        "val_rmse": evaluate_model(dummy_model, dummy_preprocessor, val_query, batch_size, effective_settings),
        "test_rmse": evaluate_model(dummy_model, dummy_preprocessor, test_query, batch_size, effective_settings),
    }

    incremental_preprocessor = fit_preprocessor_from_sample(train_sample)
    incremental_model = train_incremental_model(
        train_query,
        incremental_preprocessor,
        batch_size=batch_size,
        settings=effective_settings,
    )
    incremental_metrics = {
        "val_rmse": evaluate_model(
            incremental_model,
            incremental_preprocessor,
            val_query,
            batch_size,
            effective_settings,
        ),
        "test_rmse": evaluate_model(
            incremental_model,
            incremental_preprocessor,
            test_query,
            batch_size,
            effective_settings,
        ),
    }

    histgb_preprocessor, histgb_model = train_hist_gradient_boosting(train_sample)
    histgb_metrics = {
        "val_rmse": evaluate_model(histgb_model, histgb_preprocessor, val_query, batch_size, effective_settings),
        "test_rmse": evaluate_model(histgb_model, histgb_preprocessor, test_query, batch_size, effective_settings),
    }
    candidates = {
        "dummy_regressor": (dummy_model, dummy_preprocessor, dummy_metrics),
        "sgd_regressor": (incremental_model, incremental_preprocessor, incremental_metrics),
        "hist_gradient_boosting": (histgb_model, histgb_preprocessor, histgb_metrics),
    }
    best_name = min(candidates, key=lambda name: candidates[name][2]["val_rmse"])
    best_model, best_preprocessor, best_metrics = candidates[best_name]

    model_dir = ensure_model_dir(effective_settings)
    artifact_path = save_artifact(best_model, best_preprocessor, best_metrics, model_dir / "nyc_taxi_fare_baseline.joblib")

    return {
        "selected_model": best_name,
        "artifact_path": str(artifact_path),
        "dummy_metrics": dummy_metrics,
        "incremental_metrics": incremental_metrics,
        "hist_gradient_boosting_metrics": histgb_metrics,
    }


if __name__ == "__main__":
    results = train_out_of_core()
    print(results)
