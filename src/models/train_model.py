"""Production training entrypoint for the selected NYC Taxi model."""

from __future__ import annotations

import gc
import logging
import time
from time import perf_counter

from src.data.ingestion import fetch_sample, fetch_stratified_train_sample, iter_stratified_train_strata
from src.features.build_features import get_feature_audit_payload, split_features_target
from src.models.production_model import (
    PRODUCTION_ARTIFACT_NAME,
    PRODUCTION_DENSE_EVAL_BATCH_SIZE,
    PRODUCTION_MAX_SAMPLE_ROWS,
    PRODUCTION_SELECTION_EVIDENCE,
    get_production_model_spec,
)
from src.models.training_common import (
    compute_trip_type_weights,
    evaluate_model,
    fit_preprocessor_from_sample,
    materialize_matrix,
    save_artifact,
    split_ranges,
    train_incremental_snowflake,
    trip_type_weights_for_frame,
)
from src.utils.config import ensure_model_dir, get_settings

LOGGER = logging.getLogger("src.models.train_model")


def configure_logging() -> None:
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def _train_xgboost_out_of_core(
    spec,
    preprocessor,
    trip_type_weights: dict,
    total_limit: int,
    settings,
    log_fn,
) -> tuple:
    """Entrenamiento XGBoost out-of-core real: un lote por año×flota.

    Nunca tiene mas de ~500K filas (un estrato) en memoria al mismo tiempo.
    Usa xgb.train con xgb_model=booster_anterior para entrenamiento continuo:
    cada estrato aporta rondas nuevas al booster acumulado.

    Estrategia:
    - 18 estratos (9 años × 2 flotas) × 555K filas = 10M filas totales vistas
    - Cada estrato contribuye total_rounds / 18 ≈ 22 rondas
    - Total: ~400 arboles como el modelo original
    - Pico de RAM por lote: ~300 MB (DataFrame + sparse matrix)
    """
    import gc
    import math
    import tempfile

    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost requerido para out-of-core training.")

    xgb_params = {p: v for p, v in spec.build_estimator().get_params().items()
                  if p not in ("n_estimators", "verbosity", "callbacks")}
    # Normalizar parametros al formato nativo de xgb.train
    param_map = {
        "learning_rate": "eta",
        "max_depth": "max_depth",
        "subsample": "subsample",
        "colsample_bytree": "colsample_bytree",
        "n_jobs": "nthread",
        "random_state": "seed",
        "tree_method": "tree_method",
        "device": "device",
        "objective": "objective",
    }
    params = {param_map[k]: v for k, v in xgb_params.items() if k in param_map}
    params["verbosity"] = 0

    total_rounds = spec.build_estimator().get_params().get("n_estimators", 400)
    start_year = int(settings.data_start_date[:4])
    end_year   = int(settings.train_end_date[:4])
    n_years    = end_year - start_year + 1
    n_fleets   = len(settings.trip_types)
    n_strata   = n_years * n_fleets
    rows_per_stratum = max(1, math.ceil(total_limit / n_strata))
    rounds_per_stratum = max(1, math.ceil(total_rounds / n_strata))

    log_fn(
        "XGBoost out-of-core | strata=%d | rows_per_stratum=%d | rounds_per_stratum=%d | total_rounds~=%d",
        n_strata, rows_per_stratum, rounds_per_stratum, rounds_per_stratum * n_strata,
    )
    log_fn(
        "Memoria pico por lote: ~%.0f MB (vs %.1f GB si se cargara todo de una vez)",
        rows_per_stratum * 17 * 12 / 1e6 * 2.5,
        total_limit * 17 * 12 / 1e9 * 2.5,
    )

    booster = None
    total_rows_seen = 0
    fit_start = perf_counter()

    for stratum_idx, (year, fleet, batch_df) in enumerate(
        iter_stratified_train_strata(rows_per_stratum, settings=settings), start=1
    ):
        X_b, y_b = split_features_target(batch_df)
        X_t = materialize_matrix(preprocessor.transform(X_b), "sparse")
        w_b = trip_type_weights_for_frame(batch_df, trip_type_weights)
        dtrain = xgb.DMatrix(X_t, label=y_b.to_numpy(), weight=w_b)
        del X_t, X_b, batch_df  # liberar inmediatamente

        booster = xgb.train(
            params, dtrain, rounds_per_stratum,
            xgb_model=booster,
            verbose_eval=False,
        )
        del dtrain
        gc.collect()

        total_rows_seen += len(y_b)
        elapsed = perf_counter() - fit_start
        rate = stratum_idx / elapsed
        eta = (n_strata - stratum_idx) / rate if rate > 0 else 0
        log_fn(
            "Stratum %d/%d | year=%d fleet=%s | rows_seen=%s | rounds=%d | elapsed=%.0fs | eta=%.0fs (~%.1f min)",
            stratum_idx, n_strata, year, fleet,
            f"{total_rows_seen:,}", booster.num_boosted_rounds(),
            elapsed, eta, eta / 60,
        )

    log_fn(
        "XGBoost out-of-core complete | total_rows=%s | total_rounds=%d | elapsed=%.1fs",
        f"{total_rows_seen:,}", booster.num_boosted_rounds(), perf_counter() - fit_start,
    )

    # Envolver el booster nativo en XGBRegressor para compatibilidad con el pipeline
    from xgboost import XGBRegressor
    with tempfile.NamedTemporaryFile(suffix=".ubj", delete=False) as f:
        tmp_path = f.name
    booster.save_model(tmp_path)
    sklearn_model = XGBRegressor(**spec.build_estimator().get_params())
    sklearn_model.load_model(tmp_path)
    import os as _os
    _os.unlink(tmp_path)

    return sklearn_model, total_rows_seen


def train_production_model(
    settings=None,
    sample_limit: int | None = None,
    batch_size: int | None = None,
) -> dict[str, object]:
    configure_logging()
    start_ts = perf_counter()
    effective_settings = settings or get_settings()
    spec = get_production_model_spec()
    requested_sample_limit = sample_limit or effective_settings.train_sample_limit
    effective_sample_limit = requested_sample_limit
    requested_batch_size = batch_size or effective_settings.batch_size
    effective_batch_size = requested_batch_size

    if spec.training_strategy == "sample":
        effective_sample_limit = min(effective_sample_limit, PRODUCTION_MAX_SAMPLE_ROWS)

    if spec.matrix_format == "dense":
        effective_batch_size = min(effective_batch_size, PRODUCTION_DENSE_EVAL_BATCH_SIZE)

    ranges = split_ranges(effective_settings)

    LOGGER.info(
        "Starting production training | model=%s | strategy=%s | matrix_format=%s",
        spec.name,
        spec.training_strategy,
        spec.matrix_format,
    )
    LOGGER.info(
        "Training config | requested_sample_limit=%s | effective_sample_limit=%s | sample_pct=%s | requested_batch_size=%s | effective_batch_size=%s | grain=%s | trip_types=%s",
        requested_sample_limit,
        effective_sample_limit,
        effective_settings.train_sample_pct,
        requested_batch_size,
        effective_batch_size,
        effective_settings.training_batch_grain,
        ",".join(effective_settings.trip_types),
    )
    LOGGER.info(
        "Split ranges | train=%s..%s | validation=%s..%s | test=%s..%s",
        ranges["train"][0],
        ranges["train"][1],
        ranges["validation"][0],
        ranges["validation"][1],
        ranges["test"][0],
        ranges["test"][1],
    )

    # ---------------------------------------------------------------------------
    # Fit preprocessor sobre muestra pequeña (10K filas).
    # El OHE ve solo las categorias de esa muestra; categorias nuevas en batches
    # posteriores quedan como "unknown" (handle_unknown='ignore' en el pipeline).
    # Esto es correcto y necesario para el modo out-of-core.
    # ---------------------------------------------------------------------------
    preprocessing_start = perf_counter()
    bootstrap_sample = fetch_sample(
        f"SELECT * FROM {effective_settings.train_table}",
        limit=10_000,
        use_tablesample=True,
        settings=effective_settings,
    )
    if bootstrap_sample.empty:
        raise ValueError(
            "Bootstrap sample vacio. Ejecuta ingestion transform primero."
        )
    preprocessor = fit_preprocessor_from_sample(bootstrap_sample)
    trip_type_weights = compute_trip_type_weights(bootstrap_sample["trip_type"])
    trip_type_counts = bootstrap_sample["trip_type"].astype(str).str.lower().value_counts().to_dict()
    LOGGER.info("Preprocessor fitted on %d rows | elapsed=%.1fs", len(bootstrap_sample), perf_counter() - preprocessing_start)
    LOGGER.info(
        "Trip-type imbalance | counts=%s | inverse_frequency_weights=%s",
        {k: int(v) for k, v in trip_type_counts.items()},
        {k: round(v, 4) for k, v in trip_type_weights.items()},
    )
    del bootstrap_sample  # liberar memoria antes del entrenamiento

    model = spec.build_estimator()

    if spec.training_strategy == "xgb_out_of_core":
        model, train_total_rows = _train_xgboost_out_of_core(
            spec, preprocessor, trip_type_weights, effective_sample_limit,
            effective_settings, LOGGER.info,
        )
    elif spec.training_strategy == "incremental":
        LOGGER.info("Training incremental model over Snowflake batches")
        model = train_incremental_snowflake(
            effective_settings.train_table,
            ranges["train"][0],
            ranges["train"][1],
            effective_settings.training_batch_grain,
            preprocessor,
            model,
            matrix_format=spec.matrix_format,
            batch_size=effective_batch_size,
            settings=effective_settings,
            log_fn=LOGGER.info,
            trip_type_weights=trip_type_weights,
        )
        train_total_rows = effective_sample_limit
    else:
        # sample: carga el dataset completo de entrenamiento en memoria (2M filas estratificadas)
        train_sample = fetch_stratified_train_sample(
            total_limit=effective_sample_limit,
            settings=effective_settings,
        )
        if train_sample.empty:
            raise ValueError("Train sample vacio.")
        LOGGER.info("Fetched train sample | rows=%s", f"{len(train_sample):,}")

        # Bug fix 1: refit preprocessor sobre los DATOS DE ENTRENAMIENTO reales (2M filas).
        # El bootstrap de 10K filas solo cubria el 13% de route_ids (70K posibles).
        # Con 2M filas el OHE ve ~100% de rutas → route_id funciona correctamente.
        preprocessing_start = perf_counter()
        preprocessor = fit_preprocessor_from_sample(train_sample)
        LOGGER.info("Preprocessor refitted on %d training rows | elapsed=%.1fs",
                    len(train_sample), perf_counter() - preprocessing_start)

        # Bug fix 2: recalcular trip_type_weights desde la muestra real (no del bootstrap).
        # El bootstrap tenia 91.8% yellow / 8.2% green → green recibia 11x mas peso.
        # La muestra estratificada es 50/50 → weights deben ser iguales (1.0 cada uno).
        trip_type_weights = compute_trip_type_weights(train_sample["trip_type"])
        trip_type_counts = train_sample["trip_type"].astype(str).str.lower().value_counts().to_dict()
        LOGGER.info(
            "Trip-type balance from train sample | counts=%s | weights=%s",
            {k: int(v) for k, v in trip_type_counts.items()},
            {k: round(v, 4) for k, v in trip_type_weights.items()},
        )

        X_train, y_train = split_features_target(train_sample)
        fit_start = perf_counter()
        X_train_t = materialize_matrix(preprocessor.transform(X_train), spec.matrix_format)
        LOGGER.info("Fit input materialized | shape=%s", getattr(X_train_t, "shape", "unknown"))
        sample_weights = trip_type_weights_for_frame(train_sample, trip_type_weights)
        try:
            model.fit(X_train_t, y_train, sample_weight=sample_weights)
        except TypeError:
            model.fit(X_train_t, y_train)
        LOGGER.info("Model fit completed | elapsed=%.1fs", perf_counter() - fit_start)
        train_total_rows = len(train_sample)
        # Liberar memoria de entrenamiento antes de la evaluación (~2GB con 5M filas)
        del X_train_t, X_train, y_train, train_sample, sample_weights
        gc.collect()
        LOGGER.info("Training memory released before evaluation.")

    val_rmse, val_mae, val_med_ae, val_mape, val_r2 = evaluate_model(
        model,
        preprocessor,
        effective_settings.val_table,
        ranges["validation"][0],
        ranges["validation"][1],
        effective_settings.training_batch_grain,
        matrix_format=spec.matrix_format,
        batch_size=effective_batch_size,
        settings=effective_settings,
        split_name="validation",
        log_fn=LOGGER.info,
    )
    test_rmse, test_mae, test_med_ae, test_mape, test_r2 = evaluate_model(
        model,
        preprocessor,
        effective_settings.test_table,
        ranges["test"][0],
        ranges["test"][1],
        effective_settings.training_batch_grain,
        matrix_format=spec.matrix_format,
        batch_size=effective_batch_size,
        settings=effective_settings,
        split_name="test",
        log_fn=LOGGER.info,
    )

    feature_audit = get_feature_audit_payload()
    metrics = {
        "model": spec.name,
        "training_strategy": spec.training_strategy,
        "rubric_status": spec.rubric_status,
        "matrix_format": spec.matrix_format,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "val_med_ae": val_med_ae,
        "val_mape": round(val_mape, 2),
        "val_r2": round(val_r2, 4),
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_med_ae": test_med_ae,
        "test_mape": round(test_mape, 2),
        "test_r2": round(test_r2, 4),
        "sample_rows": train_total_rows,
        "features_used": feature_audit["model_feature_columns"],
        "feature_contract_version": feature_audit["feature_contract_version"],
        "zone_lookup_enabled": effective_settings.enable_zone_lookup,
        "trip_types": list(effective_settings.trip_types),
        "trip_type_counts": {k: int(v) for k, v in trip_type_counts.items()},
        "trip_type_weights": {k: round(v, 6) for k, v in trip_type_weights.items()},
    }

    artifact_path = save_artifact(
        model,
        preprocessor,
        spec.name,
        metrics,
        feature_audit,
        effective_settings.enable_zone_lookup,
        ensure_model_dir(effective_settings) / PRODUCTION_ARTIFACT_NAME,
        input_matrix_format=spec.matrix_format,
        artifact_role="production",
        extra_metadata={
            "selection_evidence": PRODUCTION_SELECTION_EVIDENCE,
            "training_batch_grain": effective_settings.training_batch_grain,
            "requested_sample_limit": requested_sample_limit,
            "effective_sample_limit": effective_sample_limit,
            "requested_batch_size": requested_batch_size,
            "effective_batch_size": effective_batch_size,
        },
    )
    total_elapsed = perf_counter() - start_ts
    LOGGER.info(
        "Production training finished | model=%s | "
        "val_rmse=%.4f | val_mae=%.4f | val_med_ae=%.4f | val_mape=%.2f%% | val_r2=%.4f | "
        "test_rmse=%.4f | test_mae=%.4f | test_med_ae=%.4f | test_mape=%.2f%% | test_r2=%.4f | "
        "artifact=%s | elapsed=%.1fs",
        spec.name,
        val_rmse, val_mae, val_med_ae, val_mape, val_r2,
        test_rmse, test_mae, test_med_ae, test_mape, test_r2,
        artifact_path,
        total_elapsed,
    )

    return {
        "selected_model": spec.name,
        "artifact_path": str(artifact_path),
        "metrics": metrics,
        "sample_rows": train_total_rows,
        "batch_size": effective_batch_size,
        "training_batch_grain": effective_settings.training_batch_grain,
        "feature_audit": feature_audit,
        "zone_lookup_enabled": effective_settings.enable_zone_lookup,
        "trip_types": list(effective_settings.trip_types),
        "selection_evidence": PRODUCTION_SELECTION_EVIDENCE,
    }


if __name__ == "__main__":
    results = train_production_model()
    LOGGER.info("Selected production model: %s", results["selected_model"])
    LOGGER.info("Artifact: %s", results["artifact_path"])
