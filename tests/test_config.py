from pathlib import Path

import pytest

from src.utils.config import get_settings, get_snowflake_connection_params, missing_required_settings


def test_get_settings_supports_legacy_spark_env_names(monkeypatch):
    monkeypatch.setenv("SF_HOST", "demo-account.snowflakecomputing.com")
    monkeypatch.setenv("SF_USER", "legacy_user")
    monkeypatch.setenv("SF_PASSWORD", "legacy_password")
    monkeypatch.setenv("SF_ROLE", "ACCOUNTADMIN")
    monkeypatch.setenv("SF_WAREHOUSE", "COMPUTE_WH")
    monkeypatch.setenv("SF_DATABASE", "DM_DB")
    monkeypatch.setenv("SF_RAW_SCHEMA", "RAW_STAGE")
    monkeypatch.setenv("SF_ANALYTICS_SCHEMA", "ANALYTICS_STAGE")
    monkeypatch.delenv("SNOWFLAKE_SCHEMA_ML", raising=False)
    monkeypatch.setenv("DATA_START_DATE", "2025-01-01")
    monkeypatch.setenv("DATA_END_DATE", "2025-01-31")
    monkeypatch.setenv("TRAIN_END_DATE", "2025-01-21")
    monkeypatch.setenv("VAL_END_DATE", "2025-01-27")
    monkeypatch.setenv("MODEL_DIR", "data/models")

    settings = get_settings(reload=True)

    assert settings.snowflake_account == "demo-account"
    assert settings.snowflake_user == "legacy_user"
    assert settings.snowflake_schema_raw == "RAW_STAGE"
    assert settings.snowflake_schema_analytics == "ANALYTICS_STAGE"
    assert settings.snowflake_schema_ml == "ML"
    assert settings.data_start_date == "2025-01-01"
    assert settings.val_end_date == "2025-01-27"
    assert settings.model_dir == Path("data/models")


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

    settings = get_settings(reload=True, validate=False)
    missing = missing_required_settings(settings)

    assert "SNOWFLAKE_ACCOUNT" in missing
    assert "SNOWFLAKE_USER" in missing

    with pytest.raises(ValueError):
        get_snowflake_connection_params(settings)
