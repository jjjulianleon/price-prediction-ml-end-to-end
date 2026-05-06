"""Experimentation-only model catalog for notebook 04 and benchmark runs."""

from __future__ import annotations

from src.models.estimators import (
    CatBoostRegressor,
    LGBMRegressor,
    XGBRegressor,
    build_adaboost,
    build_bagging,
    build_catboost,
    build_dummy_regressor,
    build_gradient_boosting,
    build_hist_gradient_boosting,
    build_lightgbm,
    build_pasting,
    build_random_forest,
    build_ridge_regressor,
    build_sgd_regressor,
    build_voting,
    build_xgboost,
)
from src.models.training_common import EstimatorSpec

CURATED_EXPERIMENT_MODEL_NAMES = (
    "dummy_regressor",
    "sgd_regressor",
    "random_forest",
    "gradient_boosting",
    "hist_gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
)

MODEL_ZOO: list[EstimatorSpec] = [
    EstimatorSpec(
        name="dummy_regressor",
        build_estimator=build_dummy_regressor,
        training_strategy="sample",
        description="Baseline mean regressor.",
        rubric_status="baseline",
    ),
    EstimatorSpec(
        name="ridge_regressor",
        build_estimator=build_ridge_regressor,
        training_strategy="sample",
        description="Regularized linear baseline over bounded sample.",
        rubric_status="archived",
    ),
    EstimatorSpec(
        name="sgd_regressor",
        build_estimator=build_sgd_regressor,
        training_strategy="incremental",
        description="True partial_fit baseline over Snowflake batches.",
        rubric_status="baseline",
    ),
    EstimatorSpec(
        name="random_forest",
        build_estimator=build_random_forest,
        training_strategy="sample",
        description="Reference tree ensemble on bounded sample.",
        rubric_status="recommended",
    ),
    EstimatorSpec(
        name="adaboost",
        build_estimator=build_adaboost,
        training_strategy="sample",
        description="Classical boosting baseline retained only for extended experiments.",
        rubric_status="archived",
    ),
    EstimatorSpec(
        name="gradient_boosting",
        build_estimator=build_gradient_boosting,
        training_strategy="sample",
        description="Best completed validation score in the current evidence set.",
        matrix_format="dense",
        required=True,
        rubric_status="recommended",
    ),
    EstimatorSpec(
        name="hist_gradient_boosting",
        build_estimator=build_hist_gradient_boosting,
        training_strategy="sample",
        description="Histogram boosting on bounded sample.",
        matrix_format="dense",
        rubric_status="recommended",
    ),
    EstimatorSpec(
        name="bagging",
        build_estimator=build_bagging,
        training_strategy="sample",
        description="Bagging ensemble archived after being dominated by stronger boosters.",
        rubric_status="archived",
    ),
    EstimatorSpec(
        name="pasting",
        build_estimator=build_pasting,
        training_strategy="sample",
        description="Pasting ensemble archived after being dominated by stronger boosters.",
        rubric_status="archived",
    ),
    EstimatorSpec(
        name="voting",
        build_estimator=build_voting,
        training_strategy="sample",
        description="Voting ensemble archived because it added heavy runtime without winning.",
        matrix_format="dense",
        rubric_status="archived",
    ),
    EstimatorSpec(
        name="xgboost",
        build_estimator=build_xgboost,
        training_strategy="sample",
        description="Modern gradient boosting implementation with hist backend.",
        required=True,
        dependency="xgboost",
        rubric_status="recommended",
    ),
    EstimatorSpec(
        name="lightgbm",
        build_estimator=build_lightgbm,
        training_strategy="sample",
        description="Modern gradient boosting implementation with leaf-wise growth.",
        required=True,
        dependency="lightgbm",
        rubric_status="recommended",
    ),
    EstimatorSpec(
        name="catboost",
        build_estimator=build_catboost,
        training_strategy="sample",
        description="Modern gradient boosting implementation pending stable completion.",
        matrix_format="dense",
        required=True,
        dependency="catboost",
        rubric_status="recommended",
    ),
]


def _is_available(entry: EstimatorSpec) -> bool:
    if entry.dependency == "xgboost" and XGBRegressor is None:
        return False
    if entry.dependency == "lightgbm" and LGBMRegressor is None:
        return False
    if entry.dependency == "catboost" and CatBoostRegressor is None:
        return False
    return True


def available_model_entries(model_names: tuple[str, ...] | list[str] | None = None) -> list[EstimatorSpec]:
    allowed = set(model_names) if model_names else None
    available: list[EstimatorSpec] = []
    for entry in MODEL_ZOO:
        if allowed is not None and entry.name not in allowed:
            continue
        if not _is_available(entry):
            continue
        available.append(entry)
    return available


def recommended_experiment_entries() -> list[EstimatorSpec]:
    return available_model_entries(CURATED_EXPERIMENT_MODEL_NAMES)


def unavailable_required_models(model_names: tuple[str, ...] | list[str] | None = None) -> list[str]:
    allowed = set(model_names) if model_names else None
    missing: list[str] = []
    for entry in MODEL_ZOO:
        if not entry.required:
            continue
        if allowed is not None and entry.name not in allowed:
            continue
        if not _is_available(entry):
            missing.append(entry.name)
    return missing
