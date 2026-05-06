from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import SGDRegressor

from src.models import experiment_runner
from src.models import train_model as training_module
from src.models.training_common import EstimatorSpec
from src.models import training_common
from src.utils.config import Settings


def synthetic_batches():
    rows = []
    for day in range(1, 31):
        rows.append(
            {
                "trip_type": "yellow" if day % 2 else "green",
                "pickup_datetime": f"2025-01-{day:02d} 08:00:00",
                "passenger_count": 1 + (day % 3),
                "estimated_distance": 1.0 + day / 10.0,
                "pickup_location_id": 100 + (day % 5),
                "dropoff_location_id": 200 + (day % 7),
                "vendor_id": 1 + (day % 2),
                "ratecode_id": 1,
                "fare_amount": 8.0 + day * 0.8,
            }
        )
    return pd.DataFrame(rows)


def demo_settings(tmp_path: Path) -> Settings:
    return Settings(
        snowflake_account="demo",
        snowflake_user="user",
        snowflake_password="password",
        snowflake_role="role",
        snowflake_warehouse="warehouse",
        snowflake_database="database",
        trip_types=("yellow", "green"),
        snowflake_schema_raw="RAW",
        snowflake_schema_staging="STAGING",
        snowflake_schema_analytics="ANALYTICS",
        snowflake_schema_ml="ML",
        data_start_date="2025-01-01",
        data_end_date="2025-01-31",
        train_end_date="2025-01-21",
        val_end_date="2025-01-27",
        nyc_taxi_month="2025-01",
        nyc_tlc_base_url="https://d37ci6vzurychx.cloudfront.net/trip-data",
        local_data_dir=Path(tmp_path / "raw"),
        model_dir=Path(tmp_path),
        enable_download=True,
        enable_stage_upload=True,
        enable_copy_into=True,
        enable_zone_lookup=False,
        zone_lookup_path=Path(tmp_path / "taxi_zone_lookup.csv"),
        eda_sample_limit=100,
        eda_sample_seed=42,
        train_sample_limit=12,
        train_sample_pct=1.0,
        batch_size=10,
        training_batch_grain="month",
        target_column="fare_amount",
    )


def install_fake_snowflake(monkeypatch, df: pd.DataFrame) -> None:
    def fake_fetch_sample(query, sample_pct=None, limit=5000, sample_seed=None, settings=None):
        return df.iloc[: min(limit, len(df))].copy()

    def fake_fetch_batches(query, batch_size=50000, settings=None):
        if "VAL_SET_DEV" in query:
            yield df.iloc[21:27].copy()
        elif "TEST_SET_DEV" in query:
            yield df.iloc[27:].copy()
        else:
            yield df.iloc[:21].copy()

    monkeypatch.setattr(experiment_runner, "fetch_sample", fake_fetch_sample)
    monkeypatch.setattr(training_module, "fetch_sample", fake_fetch_sample)
    monkeypatch.setattr(training_common, "fetch_data_in_batches", fake_fetch_batches)


def test_run_curated_experiment_benchmark_with_mocked_snowflake(monkeypatch, tmp_path):
    df = synthetic_batches()
    install_fake_snowflake(monkeypatch, df)

    monkeypatch.setattr(
        experiment_runner,
        "recommended_experiment_entries",
        lambda: [
            EstimatorSpec(
                name="dummy_regressor",
                build_estimator=lambda: DummyRegressor(strategy="mean"),
                training_strategy="sample",
                description="dummy",
                rubric_status="baseline",
            ),
            EstimatorSpec(
                name="sgd_regressor",
                build_estimator=lambda: SGDRegressor(random_state=42),
                training_strategy="incremental",
                description="sgd",
                rubric_status="baseline",
            ),
            EstimatorSpec(
                name="hist_gradient_boosting",
                build_estimator=lambda: HistGradientBoostingRegressor(random_state=42),
                training_strategy="sample",
                description="hgb",
                matrix_format="dense",
                rubric_status="recommended",
            ),
        ],
    )

    benchmark = experiment_runner.run_curated_experiment_benchmark(
        settings=demo_settings(tmp_path),
        sample_limit=12,
        sample_pct=1.0,
        logger=lambda message: None,
    )

    assert benchmark["sample_rows"] == 12
    assert len(benchmark["results"]) == 3
    assert benchmark["comparison"].iloc[0]["val_rmse"] >= 0
    assert benchmark["feature_audit"]["feature_contract_version"] == "v3_multi_taxi_estimated_distance"


def test_train_production_model_with_mocked_snowflake(monkeypatch, tmp_path):
    df = synthetic_batches()
    install_fake_snowflake(monkeypatch, df)

    monkeypatch.setattr(
        training_module,
        "get_production_model_spec",
        lambda: EstimatorSpec(
            name="hist_gradient_boosting",
            build_estimator=lambda: HistGradientBoostingRegressor(random_state=42),
            training_strategy="sample",
            description="prod",
            matrix_format="dense",
            rubric_status="production",
        ),
    )

    result = training_module.train_production_model(
        settings=demo_settings(tmp_path),
        sample_limit=12,
        batch_size=10,
    )

    artifact_path = Path(result["artifact_path"])
    artifact = joblib.load(artifact_path)

    assert result["selected_model"] == "hist_gradient_boosting"
    assert artifact_path.exists()
    assert result["sample_rows"] == 12
    assert result["batch_size"] == 10
    assert result["training_batch_grain"] == "month"
    assert artifact["artifact_role"] == "production"
    assert artifact["input_matrix_format"] == "dense"
    assert artifact["metrics"]["val_rmse"] >= 0
