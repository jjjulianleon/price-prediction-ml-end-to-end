"""Production model specification selected from experimentation evidence."""

from __future__ import annotations

from src.models.estimators import build_gradient_boosting
from src.models.training_common import EstimatorSpec

PRODUCTION_MODEL_NAME = "gradient_boosting"
PRODUCTION_ARTIFACT_NAME = "nyc_taxi_fare_production.joblib"
PRODUCTION_MAX_SAMPLE_ROWS = 20_000
PRODUCTION_DENSE_EVAL_BATCH_SIZE = 5_000
PRODUCTION_SELECTION_EVIDENCE = (
    "Selected on 2026-05-05 from notebook 04 and notebooks/temp.txt because it "
    "had the best completed validation RMSE among the stable runs."
)


def get_production_model_spec() -> EstimatorSpec:
    return EstimatorSpec(
        name=PRODUCTION_MODEL_NAME,
        build_estimator=build_gradient_boosting,
        training_strategy="sample",
        description="Selected production booster.",
        matrix_format="dense",
        rubric_status="production",
    )
