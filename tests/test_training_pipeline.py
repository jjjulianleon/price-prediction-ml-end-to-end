from pathlib import Path

import pandas as pd

from src.models import train_model as training_module
from src.utils.config import Settings


def synthetic_batches():
    rows = []
    for day in range(1, 31):
        rows.append(
            {
                "pickup_datetime": f"2025-01-{day:02d} 08:00:00",
                "passenger_count": 1 + (day % 3),
                "trip_distance": 1.0 + day / 10.0,
                "pickup_location_id": 100 + (day % 5),
                "dropoff_location_id": 200 + (day % 7),
                "vendor_id": 1 + (day % 2),
                "ratecode_id": 1,
                "fare_amount": 8.0 + day * 0.8,
            }
        )
    return pd.DataFrame(rows)


def test_train_out_of_core_with_mocked_snowflake(monkeypatch, tmp_path):
    df = synthetic_batches()

    def fake_fetch_sample(query, sample_pct=None, limit=5000, settings=None):
        return df.iloc[: min(limit, len(df))].copy()

    def fake_fetch_batches(query, batch_size=50000, settings=None):
        if "VAL_SET_DEV" in query:
            yield df.iloc[21:27].copy()
        elif "TEST_SET_DEV" in query:
            yield df.iloc[27:].copy()
        else:
            yield df.iloc[:21].copy()

    monkeypatch.setattr(training_module, "fetch_sample", fake_fetch_sample)
    monkeypatch.setattr(training_module, "fetch_data_in_batches", fake_fetch_batches)

    settings = Settings(
        snowflake_account="demo",
        snowflake_user="user",
        snowflake_password="password",
        snowflake_role="role",
        snowflake_warehouse="warehouse",
        snowflake_database="database",
        snowflake_schema_raw="RAW",
        snowflake_schema_analytics="ANALYTICS",
        snowflake_schema_ml="ML",
        data_start_date="2025-01-01",
        data_end_date="2025-01-31",
        train_end_date="2025-01-21",
        val_end_date="2025-01-27",
        nyc_taxi_month="2025-01",
        nyc_tlc_base_url="https://d37ci6vzurychx.cloudfront.net/trip-data",
        model_dir=Path(tmp_path),
    )

    result = training_module.train_out_of_core(settings=settings, sample_limit=12, batch_size=10)

    assert result["selected_model"] in {"dummy_regressor", "sgd_regressor", "hist_gradient_boosting"}
    assert Path(result["artifact_path"]).exists()
    assert result["dummy_metrics"]["val_rmse"] >= 0
    assert result["incremental_metrics"]["test_rmse"] >= 0
    assert result["hist_gradient_boosting_metrics"]["val_rmse"] >= 0
