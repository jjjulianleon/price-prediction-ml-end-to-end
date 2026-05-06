"""Experiment runner for the curated model shortlist."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from src.data.ingestion import fetch_sample
from src.features.build_features import get_feature_audit_payload, split_features_target
from src.models.model_zoo import recommended_experiment_entries
from src.models.training_common import (
    EstimatorSpec,
    compute_trip_type_weights,
    evaluate_model,
    fit_preprocessor_from_sample,
    materialize_matrix,
    split_ranges,
    train_incremental_snowflake,
    trip_type_weights_for_frame,
)
from src.utils.config import Settings, get_settings


def _default_logger(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}")


def prepare_experiment_context(
    settings: Settings | None = None,
    sample_limit: int | None = None,
    sample_pct: float | None = None,
) -> dict[str, Any]:
    effective_settings = settings or get_settings()
    effective_sample_limit = sample_limit or min(effective_settings.train_sample_limit, 20_000)
    effective_sample_pct = sample_pct or min(effective_settings.train_sample_pct, 1.0)

    # TABLESAMPLE evita ORDER BY RANDOM() sobre 756M rows (~15 min) -> ~30 seg
    train_sample = fetch_sample(
        f"SELECT * FROM {effective_settings.train_table}",
        sample_pct=effective_sample_pct,
        limit=effective_sample_limit,
        sample_seed=effective_settings.eda_sample_seed,
        settings=effective_settings,
        use_tablesample=True,
    )
    if train_sample.empty:
        raise ValueError(
            "Train sample is empty. Run raw ingestion first and then `python3 -m src.data.ingestion transform` before experimentation."
        )

    preprocessor = fit_preprocessor_from_sample(train_sample)
    X_train_sample_raw, y_train_sample = split_features_target(train_sample)
    X_train_sample_t = preprocessor.transform(X_train_sample_raw)
    trip_type_weights = compute_trip_type_weights(train_sample["trip_type"])
    train_sample_weights = trip_type_weights_for_frame(train_sample, trip_type_weights)

    return {
        "settings": effective_settings,
        "ranges": split_ranges(effective_settings),
        "train_sample": train_sample,
        "X_train_sample_raw": X_train_sample_raw,
        "y_train_sample": y_train_sample,
        "X_train_sample_t": X_train_sample_t,
        "preprocessor": preprocessor,
        "feature_audit": get_feature_audit_payload(),
        "trip_type_weights": trip_type_weights,
        "train_sample_weights": train_sample_weights,
    }


def run_single_experiment(
    entry: EstimatorSpec,
    context: dict[str, Any],
    sparse_eval_batch_size: int = 25_000,
    dense_eval_batch_size: int = 5_000,
    # None = evaluacion completa (produccion). Int = muestra aleatoria rapida (notebooks).
    eval_sample_limit: int | None = None,
    # True = fuerza training sobre muestra aunque el entry diga "incremental".
    # Necesario en notebooks: SGDRegressor incremental recorre 108 meses = horas.
    force_sample_strategy: bool = False,
    logger=None,
) -> tuple[Any, dict[str, object]]:
    log = logger or _default_logger
    settings = context["settings"]
    ranges = context["ranges"]
    preprocessor = context["preprocessor"]
    train_sample = context["train_sample"]
    X_train_sample_t = context["X_train_sample_t"]
    y_train_sample = context["y_train_sample"]
    feature_audit = context["feature_audit"]

    model = entry.build_estimator()
    trip_type_weights = context.get("trip_type_weights")
    train_sample_weights = context.get("train_sample_weights")
    log(f"=== Entrenando modelo: {entry.name} ===")

    effective_strategy = "sample" if force_sample_strategy else entry.training_strategy
    if effective_strategy != entry.training_strategy:
        log(f"[notebook mode] Overriding strategy {entry.training_strategy} -> sample para {entry.name}")

    if effective_strategy == "incremental":
        model = train_incremental_snowflake(
            settings.train_table,
            ranges["train"][0],
            ranges["train"][1],
            settings.training_batch_grain,
            preprocessor,
            model,
            matrix_format=entry.matrix_format,
            batch_size=min(settings.batch_size, 50_000),
            settings=settings,
            trip_type_weights=trip_type_weights,
        )
    else:
        X_fit = materialize_matrix(X_train_sample_t, entry.matrix_format)
        log(f"Fit input for {entry.name}: shape={X_fit.shape}")
        try:
            model.fit(X_fit, y_train_sample, sample_weight=train_sample_weights)
        except TypeError:
            log(f"Estimator {entry.name} does not accept sample_weight; fitting unweighted.")
            model.fit(X_fit, y_train_sample)

    eval_batch_size = dense_eval_batch_size if entry.matrix_format == "dense" else sparse_eval_batch_size

    def _quick_rmse(table: str, limit: int) -> float:
        """Evaluacion rapida sobre muestra aleatoria: para notebooks, no para produccion.

        Para modelos dense (GradientBoosting, etc.) se limita a 10K filas porque la
        conversion sparse->dense con columnas OHE genera matrices de 8-40 GB que crashean
        WSL. Modelos sparse (XGBoost, LightGBM) pueden usar el limite completo.
        """
        import numpy as np
        from sklearn.metrics import mean_squared_error
        # Modelos dense: OHE con 10K cols × 200K rows × 8 bytes = 16 GB -> OOM crash
        effective_limit = min(limit, 10_000) if entry.matrix_format == "dense" else limit
        sample = fetch_sample(
            f"SELECT * FROM {table}",
            limit=effective_limit,
            sample_seed=settings.eda_sample_seed,
            settings=settings,
            use_tablesample=True,
        )
        X_s, y_s = split_features_target(sample)
        X_t = materialize_matrix(preprocessor.transform(X_s), entry.matrix_format)
        preds = model.predict(X_t)
        return float(np.sqrt(mean_squared_error(y_s.to_numpy(), np.asarray(preds))))

    if eval_sample_limit:
        log(f"Eval rapido sobre muestra de {eval_sample_limit:,} filas (notebook mode)")
        val_rmse = _quick_rmse(settings.val_table, eval_sample_limit)
        test_rmse = _quick_rmse(settings.test_table, eval_sample_limit)
    else:
        val_rmse, *_ = evaluate_model(
            model, preprocessor, settings.val_table,
            ranges["validation"][0], ranges["validation"][1],
            settings.training_batch_grain, matrix_format=entry.matrix_format,
            batch_size=eval_batch_size, settings=settings,
        )
        test_rmse, *_ = evaluate_model(
            model, preprocessor, settings.test_table,
            ranges["test"][0], ranges["test"][1],
            settings.training_batch_grain, matrix_format=entry.matrix_format,
            batch_size=eval_batch_size, settings=settings,
        )

    metrics = {
        "model": entry.name,
        "training_strategy": entry.training_strategy,
        "rubric_status": entry.rubric_status,
        "required": entry.required,
        "matrix_format": entry.matrix_format,
        "sample_rows": len(train_sample),
        "val_rmse": val_rmse,
        "test_rmse": test_rmse,
        "feature_contract_version": feature_audit["feature_contract_version"],
        "zone_lookup_enabled": settings.enable_zone_lookup,
        "trip_types": list(settings.trip_types),
        "trip_type_weights": {
            k: round(v, 6) for k, v in (trip_type_weights or {}).items()
        },
    }
    log(f"{entry.name} listo | val_rmse={val_rmse:.4f} | test_rmse={test_rmse:.4f}")
    return model, metrics


def run_curated_experiment_benchmark(
    settings: Settings | None = None,
    sample_limit: int | None = None,
    sample_pct: float | None = None,
    logger=None,
) -> dict[str, object]:
    context = prepare_experiment_context(settings=settings, sample_limit=sample_limit, sample_pct=sample_pct)
    results: list[dict[str, object]] = []

    for entry in recommended_experiment_entries():
        _, metrics = run_single_experiment(entry, context, logger=logger)
        results.append(metrics)

    comparison = pd.DataFrame(results).sort_values("val_rmse").reset_index(drop=True)
    return {
        "comparison": comparison,
        "results": comparison.to_dict(orient="records"),
        "feature_audit": context["feature_audit"],
        "sample_rows": len(context["train_sample"]),
    }


if __name__ == "__main__":
    benchmark = run_curated_experiment_benchmark()
    print(benchmark["comparison"].to_string(index=False))
