"""Training entrypoint — compares all models from MODEL_ZOO via Snowflake."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.data.ingestion import fetch_data_in_batches, fetch_sample
from src.features.build_features import (
    TARGET_COLUMN,
    get_feature_audit_payload,
    get_feature_pipeline,
    split_features_target,
)
from src.models.model_zoo import available_model_entries, unavailable_required_models
from src.utils.config import Settings, ensure_model_dir, get_settings


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


def train_incremental_snowflake(
    table_name: str,
    start_date: str,
    end_date: str,
    grain: str,
    preprocessor,
    estimator,
    batch_size: int = 50_000,
    settings: Settings | None = None,
):
    seen_rows = 0
    for batch_df in iter_split_batches(
        table_name,
        start_date,
        end_date,
        grain,
        batch_size,
        settings,
    ):
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
    table_name: str,
    start_date: str,
    end_date: str,
    grain: str,
    batch_size: int = 50_000,
    settings: Settings | None = None,
) -> float:
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []

    for batch_df in iter_split_batches(
        table_name,
        start_date,
        end_date,
        grain,
        batch_size,
        settings,
    ):
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
    model_results: list[dict[str, object]],
    unavailable_models: list[str],
    feature_audit: dict[str, object],
    zone_lookup_enabled: bool,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "preprocessor": preprocessor,
        "target_column": TARGET_COLUMN,
        "model_name": model_name,
        "feature_audit": feature_audit,
        "zone_lookup_enabled": zone_lookup_enabled,
        "metrics": {
            **metrics,
            "all_models": model_results,
            "unavailable_required_models": unavailable_models,
        },
    }
    joblib.dump(artifact, path)
    return path


def train_out_of_core(
    settings: Settings | None = None,
    sample_limit: int | None = None,
    batch_size: int | None = None,
) -> dict[str, object]:
    effective_settings = settings or get_settings()
    effective_sample_limit = sample_limit or effective_settings.train_sample_limit
    effective_batch_size = batch_size or effective_settings.batch_size
    ranges = split_ranges(effective_settings)
    train_sample_query = f"SELECT * FROM {effective_settings.train_table}"

    train_sample = fetch_sample(
        train_sample_query,
        sample_pct=effective_settings.train_sample_pct,
        limit=effective_sample_limit,
        settings=effective_settings,
    )
    if train_sample.empty:
        raise ValueError(
            "Train sample is empty. Run raw ingestion first and then `python3 -m src.data.ingestion transform` before training."
        )

    preprocessor = fit_preprocessor_from_sample(train_sample)
    X_train, y_train = split_features_target(train_sample)
    X_train_t = preprocessor.transform(X_train)

    model_results: list[dict[str, object]] = []
    candidates: dict[str, tuple] = {}
    available_entries = available_model_entries()
    missing_required = unavailable_required_models()
    feature_audit = get_feature_audit_payload()

    if not available_entries:
        raise ValueError("No models are available. Install project requirements first.")

    for entry in available_entries:
        model = entry.build_estimator()

        if entry.training_strategy == "incremental":
            model = train_incremental_snowflake(
                effective_settings.train_table,
                ranges["train"][0],
                ranges["train"][1],
                effective_settings.training_batch_grain,
                preprocessor,
                model,
                effective_batch_size,
                effective_settings,
            )
        else:
            model.fit(X_train_t, y_train)

        val_rmse = evaluate_model(
            model,
            preprocessor,
            effective_settings.val_table,
            ranges["validation"][0],
            ranges["validation"][1],
            effective_settings.training_batch_grain,
            effective_batch_size,
            effective_settings,
        )
        test_rmse = evaluate_model(
            model,
            preprocessor,
            effective_settings.test_table,
            ranges["test"][0],
            ranges["test"][1],
            effective_settings.training_batch_grain,
            effective_batch_size,
            effective_settings,
        )

        metrics = {
            "model": entry.name,
            "training_strategy": entry.training_strategy,
            "required": entry.required,
            "rubric_status": entry.rubric_status,
            "val_rmse": val_rmse,
            "test_rmse": test_rmse,
            "sample_rows": len(train_sample),
            "features_used": feature_audit["model_feature_columns"],
            "feature_contract_version": feature_audit["feature_contract_version"],
            "zone_lookup_enabled": effective_settings.enable_zone_lookup,
        }
        model_results.append(metrics)
        candidates[entry.name] = (model, metrics)

    print("\n=== Model Comparison ===")
    print(f"{'Model':<24} {'Strategy':<12} {'Val RMSE':>10} {'Test RMSE':>10}")
    print("-" * 64)
    best_name = min(candidates, key=lambda n: candidates[n][1]["val_rmse"])
    for name, (_, m) in sorted(candidates.items(), key=lambda x: x[1][1]["val_rmse"]):
        marker = " <-- WINNER" if name == best_name else ""
        print(
            f"{name:<24} {m['training_strategy']:<12} "
            f"{m['val_rmse']:>10.4f} {m['test_rmse']:>10.4f}{marker}"
        )

    if missing_required:
        print("\nMissing required optional models:", ", ".join(sorted(missing_required)))

    best_model, best_metrics = candidates[best_name]
    model_dir = ensure_model_dir(effective_settings)
    artifact_path = save_artifact(
        best_model,
        preprocessor,
        best_name,
        best_metrics,
        model_results,
        missing_required,
        feature_audit,
        effective_settings.enable_zone_lookup,
        model_dir / "nyc_taxi_fare_baseline.joblib",
    )

    return {
        "selected_model": best_name,
        "artifact_path": str(artifact_path),
        "model_results": sorted(model_results, key=lambda item: item["val_rmse"]),
        "unavailable_required_models": missing_required,
        "sample_rows": len(train_sample),
        "batch_size": effective_batch_size,
        "training_batch_grain": effective_settings.training_batch_grain,
        "feature_audit": feature_audit,
        "zone_lookup_enabled": effective_settings.enable_zone_lookup,
    }


if __name__ == "__main__":
    results = train_out_of_core()
    print(f"\nWinner: {results['selected_model']}")
    print(f"Artifact: {results['artifact_path']}")
