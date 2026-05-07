"""Production training entrypoint for the selected NYC Taxi model."""

from __future__ import annotations

import logging
from time import perf_counter

from src.data.ingestion import fetch_sample
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

    train_sample = fetch_sample(
        f"SELECT * FROM {effective_settings.train_table}",
        sample_pct=effective_settings.train_sample_pct,
        limit=effective_sample_limit,
        sample_seed=effective_settings.eda_sample_seed,
        settings=effective_settings,
    )
    if train_sample.empty:
        raise ValueError(
            "Train sample is empty. Run raw ingestion first and then `python3 -m src.data.ingestion transform` before production training."
        )
    LOGGER.info("Fetched train sample | rows=%s", f"{len(train_sample):,}")

    preprocessing_start = perf_counter()
    preprocessor = fit_preprocessor_from_sample(train_sample)
    LOGGER.info("Preprocessor fitted | elapsed=%.1fs", perf_counter() - preprocessing_start)
    X_train, y_train = split_features_target(train_sample)
    model = spec.build_estimator()

    trip_type_counts = train_sample["trip_type"].astype(str).str.lower().value_counts().to_dict()
    trip_type_weights = compute_trip_type_weights(train_sample["trip_type"])
    LOGGER.info(
        "Trip-type imbalance | counts=%s | inverse_frequency_weights=%s",
        {k: int(v) for k, v in trip_type_counts.items()},
        {k: round(v, 4) for k, v in trip_type_weights.items()},
    )

    if spec.training_strategy == "incremental":
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
    else:
        fit_start = perf_counter()
        X_train_t = materialize_matrix(preprocessor.transform(X_train), spec.matrix_format)
        LOGGER.info("Fit input materialized | shape=%s", getattr(X_train_t, "shape", "unknown"))
        sample_weights = trip_type_weights_for_frame(train_sample, trip_type_weights)
        try:
            model.fit(X_train_t, y_train, sample_weight=sample_weights)
        except TypeError:
            LOGGER.warning(
                "Estimator %s does not accept sample_weight; fitting without trip-type weights.",
                spec.name,
            )
            model.fit(X_train_t, y_train)
        LOGGER.info("Model fit completed | elapsed=%.1fs", perf_counter() - fit_start)

    val_rmse = evaluate_model(
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
    test_rmse = evaluate_model(
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
        "test_rmse": test_rmse,
        "sample_rows": len(train_sample),
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
        "Production training finished | model=%s | val_rmse=%.4f | test_rmse=%.4f | artifact=%s | elapsed=%.1fs",
        spec.name,
        val_rmse,
        test_rmse,
        artifact_path,
        total_elapsed,
    )

    return {
        "selected_model": spec.name,
        "artifact_path": str(artifact_path),
        "metrics": metrics,
        "sample_rows": len(train_sample),
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
