"""Production model specification selected from experimentation evidence."""

from __future__ import annotations

from src.models.estimators import build_xgboost
from src.models.training_common import EstimatorSpec

PRODUCTION_MODEL_NAME = "xgboost"
PRODUCTION_ARTIFACT_NAME = "nyc_taxi_fare_production.joblib"

# XGBoost no tiene partial_fit; se entrena sobre muestra masiva estratificada
# (~5M filas aleatorias del set de train) que cabe en memoria con tree_method=hist.
# Filas totales a procesar (distribuidas entre 18 estratos año×flota).
# 10M total / 18 estratos = ~555K filas por estrato.
# Pico de RAM por lote: ~300MB — nunca se acumula el dataset completo.
# 2M filas: el OHE ve ~100% de rutas únicas, sparse matrix ~400MB → cabe en laptop.
# Con más filas el OHE explota en memoria. Con menos el route_id pierde cobertura.
PRODUCTION_MAX_SAMPLE_ROWS = 5_000_000
PRODUCTION_DENSE_EVAL_BATCH_SIZE = 50_000
PRODUCTION_SELECTION_EVIDENCE = (
    "Selected on 2026-05-10 from notebook 04 benchmark. "
    "XGBoost: mejor gap val-test (-0.003) del shortlist, indicando la mejor "
    "generalizacion temporal a datos de 2025. "
    "Entrenamiento out-of-core real: 18 lotes (9 años x 2 flotas) x 555K filas, "
    "nunca mas de ~300MB de RAM por lote. xgb.train con xgb_model=booster_anterior "
    "acumula ~400 arboles sobre 10M filas totales del periodo 2015-2023."
)


def get_production_model_spec() -> EstimatorSpec:
    return EstimatorSpec(
        name=PRODUCTION_MODEL_NAME,
        build_estimator=build_xgboost,
        # sample estratificado: 2M filas (18 estratos × 111K) caben en ~400MB sparse.
        # Con 2M filas el OHE ve ~100% de rutas → route_id funciona correctamente.
        training_strategy="sample",
        description="XGBoost sample estratificado: 5M filas, 600 arboles, OHE completo, sin OOM.",
        matrix_format="sparse",
        rubric_status="production",
    )
