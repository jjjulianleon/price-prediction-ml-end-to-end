"""Training entrypoint — compares all models from MODEL_ZOO via Snowflake."""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.data.ingestion import fetch_data_in_batches, fetch_sample
from src.features.build_features import TARGET_COLUMN, get_feature_pipeline, split_features_target
from src.models.model_zoo import MODEL_ZOO
from src.utils.config import Settings, ensure_model_dir, get_settings


def fit_preprocessor_from_sample(sample_df: pd.DataFrame):
    X_sample, _ = split_features_target(sample_df)
    pipeline = get_feature_pipeline()
    pipeline.fit(X_sample)
    return pipeline


def train_incremental_snowflake(
    train_query: str,
    preprocessor,
    estimator,
    batch_size: int = 50_000,
    settings: Settings | None = None,
):
    seen_rows = 0
    for batch_df in fetch_data_in_batches(train_query, batch_size=batch_size, settings=settings):
        if batch_df.empty:
            continue
        X_batch, y_batch = split_features_target(batch_df)
        X_transformed = preprocessor.transform(X_batch)
        estimator.partial_fit(X_transformed, y_batch)
        seen_rows += len(batch_df)

    if seen_rows == 0:
        raise ValueError("Training query returned no rows.")
    return estimator


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


def save_artifact(
    model,
    preprocessor,
    model_name: str,
    metrics: dict,
    all_models_metrics: dict,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "preprocessor": preprocessor,
        "target_column": TARGET_COLUMN,
        "model_name": model_name,
        "metrics": {
            **metrics,
            "all_models": all_models_metrics,
        },
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
        raise ValueError("Train sample is empty. Run bootstrap before training.")

    # Fit preprocessor once on training sample
    preprocessor = fit_preprocessor_from_sample(train_sample)
    X_train, y_train = split_features_target(train_sample)
    X_train_t = preprocessor.transform(X_train)

    all_models_metrics: dict[str, dict[str, float]] = {}
    candidates: dict[str, tuple] = {}

    for entry in MODEL_ZOO:
        model = entry.estimator

        if entry.needs_incremental:
            # SGD: true out-of-core via Snowflake batches
            model = train_incremental_snowflake(
                train_query, preprocessor, model, batch_size, effective_settings
            )
        else:
            # sample_only models: fit on training sample
            model.fit(X_train_t, y_train)

        val_rmse = evaluate_model(model, preprocessor, val_query, batch_size, effective_settings)
        test_rmse = evaluate_model(model, preprocessor, test_query, batch_size, effective_settings)

        metrics = {"val_rmse": val_rmse, "test_rmse": test_rmse}
        all_models_metrics[entry.name] = metrics
        candidates[entry.name] = (model, metrics)

    # Print comparison table
    print("\n=== Model Comparison ===")
    print(f"{'Model':<30} {'Val RMSE':>10} {'Test RMSE':>10}")
    print("-" * 52)
    best_name = min(candidates, key=lambda n: candidates[n][1]["val_rmse"])
    for name, (_, m) in sorted(candidates.items(), key=lambda x: x[1][1]["val_rmse"]):
        marker = " <-- WINNER" if name == best_name else ""
        print(f"{name:<30} {m['val_rmse']:>10.4f} {m['test_rmse']:>10.4f}{marker}")

    best_model, best_metrics = candidates[best_name]
    model_dir = ensure_model_dir(effective_settings)
    artifact_path = save_artifact(
        best_model,
        preprocessor,
        best_name,
        best_metrics,
        all_models_metrics,
        model_dir / "nyc_taxi_fare_baseline.joblib",
    )

    return {
        "selected_model": best_name,
        "artifact_path": str(artifact_path),
        "all_models_metrics": all_models_metrics,
    }


if __name__ == "__main__":
    results = train_out_of_core()
    print(f"\nWinner: {results['selected_model']}")
    print(f"Artifact: {results['artifact_path']}")
