from pathlib import Path

import pytest

from src.utils.config import get_settings, get_snowflake_connection_params, missing_required_settings


def test_get_settings_supports_legacy_spark_env_names(monkeypatch):
    for key in [
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_ROLE",
        "SNOWFLAKE_WAREHOUSE",
        "SNOWFLAKE_DATABASE",
        "SNOWFLAKE_SCHEMA_RAW",
        "SNOWFLAKE_SCHEMA_STAGING",
        "SNOWFLAKE_SCHEMA_ANALYTICS",
        "SNOWFLAKE_SCHEMA_ML",
        "TRIP_TYPE",
        "LOCAL_DATA_DIR",
        "ENABLE_DOWNLOAD",
        "ENABLE_STAGE_UPLOAD",
        "ENABLE_COPY_INTO",
        "ENABLE_ZONE_LOOKUP",
        "ZONE_LOOKUP_PATH",
        "DATA_START_DATE",
        "DATA_END_DATE",
        "TRAIN_END_DATE",
        "VAL_END_DATE",
        "NYC_TAXI_MONTH",
        "MODEL_DIR",
        "MODEL_TARGET",
        "EDA_SAMPLE_LIMIT",
        "EDA_SAMPLE_SEED",
        "TRAIN_SAMPLE_LIMIT",
        "TRAIN_SAMPLE_PCT",
        "BATCH_SIZE",
        "TRAINING_BATCH_GRAIN",
    ]:
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("SF_HOST", "demo-account.snowflakecomputing.com")
    monkeypatch.setenv("SF_USER", "legacy_user")
    monkeypatch.setenv("SF_PASSWORD", "legacy_password")
    monkeypatch.setenv("SF_ROLE", "ACCOUNTADMIN")
    monkeypatch.setenv("SF_WAREHOUSE", "COMPUTE_WH")
    monkeypatch.setenv("SF_DATABASE", "DM_DB")
    monkeypatch.setenv("TRIP_TYPE", "yellow,green")
    monkeypatch.setenv("SF_RAW_SCHEMA", "RAW_STAGE")
    monkeypatch.setenv("SNOWFLAKE_SCHEMA_STAGING", "STAGING_STAGE")
    monkeypatch.setenv("SF_ANALYTICS_SCHEMA", "ANALYTICS_STAGE")
    monkeypatch.delenv("SNOWFLAKE_SCHEMA_ML", raising=False)
    monkeypatch.setenv("LOCAL_DATA_DIR", "data/raw")
    monkeypatch.setenv("ENABLE_DOWNLOAD", "true")
    monkeypatch.setenv("ENABLE_STAGE_UPLOAD", "false")
    monkeypatch.setenv("ENABLE_COPY_INTO", "true")
    monkeypatch.setenv("ENABLE_ZONE_LOOKUP", "false")
    monkeypatch.setenv("ZONE_LOOKUP_PATH", "data/raw/taxi_zone_lookup.csv")
    monkeypatch.setenv("DATA_START_DATE", "2025-01-01")
    monkeypatch.setenv("DATA_END_DATE", "2025-01-31")
    monkeypatch.setenv("TRAIN_END_DATE", "2025-01-21")
    monkeypatch.setenv("VAL_END_DATE", "2025-01-27")
    monkeypatch.setenv("MODEL_DIR", "data/models")
    monkeypatch.setenv("MODEL_TARGET", "fare_amount")
    monkeypatch.setenv("EDA_SAMPLE_LIMIT", "7000")
    monkeypatch.setenv("EDA_SAMPLE_SEED", "99")
    monkeypatch.setenv("TRAIN_SAMPLE_LIMIT", "25000")
    monkeypatch.setenv("TRAIN_SAMPLE_PCT", "2.5")
    monkeypatch.setenv("BATCH_SIZE", "10000")
    monkeypatch.setenv("TRAINING_BATCH_GRAIN", "week")

    settings = get_settings(reload=True, load_env_file=False)

    assert settings.snowflake_account == "demo-account"
    assert settings.snowflake_user == "legacy_user"
    assert settings.trip_type == "yellow"
    assert settings.trip_types == ("yellow", "green")
    assert settings.snowflake_schema_raw == "RAW_STAGE"
    assert settings.snowflake_schema_staging == "STAGING_STAGE"
    assert settings.snowflake_schema_analytics == "ANALYTICS_STAGE"
    assert settings.snowflake_schema_ml == "ML"
    assert settings.data_start_date == "2025-01-01"
    assert settings.val_end_date == "2025-01-27"
    assert settings.model_dir == Path("data/models")
    assert settings.enable_stage_upload is False
    assert settings.enable_copy_into is True
    assert settings.eda_sample_limit == 7000
    assert settings.eda_sample_seed == 99
    assert settings.train_sample_limit == 25000
    assert settings.train_sample_pct == 2.5
    assert settings.batch_size == 10000
    assert settings.training_batch_grain == "week"


def test_missing_required_settings_reports_empty_values(monkeypatch):
    for key in [
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_ROLE",
        "SNOWFLAKE_WAREHOUSE",
        "SNOWFLAKE_DATABASE",
        "SF_HOST",
        "SF_USER",
        "SF_PASSWORD",
        "SF_ROLE",
        "SF_WAREHOUSE",
        "SF_DATABASE",
    ]:
        monkeypatch.delenv(key, raising=False)

    settings = get_settings(reload=True, validate=False, load_env_file=False)
    missing = missing_required_settings(settings)

    assert "SNOWFLAKE_ACCOUNT" in missing
    assert "SNOWFLAKE_USER" in missing

    with pytest.raises(ValueError):
        get_snowflake_connection_params(settings)


def test_get_settings_rejects_invalid_date_order(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "demo-account")
    monkeypatch.setenv("SNOWFLAKE_USER", "user")
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "password")
    monkeypatch.setenv("SNOWFLAKE_ROLE", "role")
    monkeypatch.setenv("SNOWFLAKE_WAREHOUSE", "warehouse")
    monkeypatch.setenv("SNOWFLAKE_DATABASE", "database")
    monkeypatch.setenv("TRIP_TYPE", "yellow")
    monkeypatch.setenv("DATA_START_DATE", "2025-01-01")
    monkeypatch.setenv("DATA_END_DATE", "2025-06-30")
    monkeypatch.setenv("TRAIN_END_DATE", "2025-05-31")
    monkeypatch.setenv("VAL_END_DATE", "2025-05-15")

    with pytest.raises(ValueError):
        get_settings(reload=True, load_env_file=False)


def test_get_settings_rejects_invalid_trip_type(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "demo-account")
    monkeypatch.setenv("SNOWFLAKE_USER", "user")
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "password")
    monkeypatch.setenv("SNOWFLAKE_ROLE", "role")
    monkeypatch.setenv("SNOWFLAKE_WAREHOUSE", "warehouse")
    monkeypatch.setenv("SNOWFLAKE_DATABASE", "database")
    monkeypatch.setenv("TRIP_TYPE", "blue")

    with pytest.raises(ValueError):
        get_settings(reload=True, load_env_file=False)


def test_get_settings_accepts_green_only(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "demo-account")
    monkeypatch.setenv("SNOWFLAKE_USER", "user")
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "password")
    monkeypatch.setenv("SNOWFLAKE_ROLE", "role")
    monkeypatch.setenv("SNOWFLAKE_WAREHOUSE", "warehouse")
    monkeypatch.setenv("SNOWFLAKE_DATABASE", "database")
    monkeypatch.setenv("TRIP_TYPE", "green")

    settings = get_settings(reload=True, load_env_file=False)

    assert settings.trip_types == ("green",)
    assert settings.green_enabled is True
    assert settings.yellow_enabled is False
