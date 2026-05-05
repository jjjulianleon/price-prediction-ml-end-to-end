"""Centralized project configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


def _first_env(*keys: str, default: str | None = None) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value not in (None, ""):
            return value
    return default


def _normalize_snowflake_account(account: str | None) -> str | None:
    if account is None:
        return None
    return account.replace("https://", "").replace(".snowflakecomputing.com", "").strip("/")


@dataclass(frozen=True)
class Settings:
    snowflake_account: str | None
    snowflake_user: str | None
    snowflake_password: str | None
    snowflake_role: str | None
    snowflake_warehouse: str | None
    snowflake_database: str | None
    trip_type: str
    snowflake_schema_raw: str
    snowflake_schema_staging: str
    snowflake_schema_analytics: str
    snowflake_schema_ml: str
    data_start_date: str
    data_end_date: str
    train_end_date: str
    val_end_date: str
    nyc_taxi_month: str | None
    nyc_tlc_base_url: str
    local_data_dir: Path
    model_dir: Path
    enable_download: bool
    enable_stage_upload: bool
    enable_copy_into: bool
    enable_zone_lookup: bool
    zone_lookup_path: Path
    eda_sample_limit: int
    eda_sample_seed: int
    train_sample_limit: int
    train_sample_pct: float
    batch_size: int
    training_batch_grain: str
    target_column: str

    @property
    def required_env(self) -> dict[str, str | None]:
        return {
            "SNOWFLAKE_ACCOUNT": self.snowflake_account,
            "SNOWFLAKE_USER": self.snowflake_user,
            "SNOWFLAKE_PASSWORD": self.snowflake_password,
            "SNOWFLAKE_ROLE": self.snowflake_role,
            "SNOWFLAKE_WAREHOUSE": self.snowflake_warehouse,
            "SNOWFLAKE_DATABASE": self.snowflake_database,
        }

    @property
    def raw_table(self) -> str:
        return f"{self.snowflake_database}.{self.snowflake_schema_raw}.YELLOW_TRIPS_DEV"

    @property
    def obt_table(self) -> str:
        return f"{self.snowflake_database}.{self.snowflake_schema_analytics}.OBT_TRIPS_DEV"

    @property
    def staging_table(self) -> str:
        return f"{self.snowflake_database}.{self.snowflake_schema_staging}.TRIPS_STAGE_DEV"

    @property
    def train_table(self) -> str:
        return f"{self.snowflake_database}.{self.snowflake_schema_ml}.TRAIN_SET_DEV"

    @property
    def val_table(self) -> str:
        return f"{self.snowflake_database}.{self.snowflake_schema_ml}.VAL_SET_DEV"

    @property
    def test_table(self) -> str:
        return f"{self.snowflake_database}.{self.snowflake_schema_ml}.TEST_SET_DEV"

    @property
    def processing_window_label(self) -> str:
        if self.nyc_taxi_month:
            return f"{self.nyc_taxi_month} ({self.data_start_date} to {self.data_end_date})"
        return f"{self.data_start_date} to {self.data_end_date}"

    @property
    def raw_stage(self) -> str:
        return f"{self.snowflake_database}.{self.snowflake_schema_raw}.NYC_TAXI_STAGE"

    @property
    def raw_file_format(self) -> str:
        return f"{self.snowflake_database}.{self.snowflake_schema_raw}.NYC_TAXI_PARQUET"

    @property
    def raw_load_audit_table(self) -> str:
        return f"{self.snowflake_database}.{self.snowflake_schema_raw}.RAW_LOAD_AUDIT"

    @property
    def taxi_zone_lookup_table(self) -> str:
        return f"{self.snowflake_database}.{self.snowflake_schema_staging}.TAXI_ZONE_LOOKUP"


def _parse_bool(raw_value: str | None, default: bool) -> bool:
    if raw_value in (None, ""):
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError("Expected a boolean-like configuration value.")


def _parse_positive_int(raw_value: str | None, default: int) -> int:
    if raw_value in (None, ""):
        return default
    value = int(raw_value)
    if value <= 0:
        raise ValueError("Expected a positive integer configuration value.")
    return value


def _parse_positive_float(raw_value: str | None, default: float) -> float:
    if raw_value in (None, ""):
        return default
    value = float(raw_value)
    if value <= 0:
        raise ValueError("Expected a positive float configuration value.")
    return value


def _normalize_batch_grain(raw_value: str | None) -> str:
    grain = (raw_value or "month").strip().lower()
    if grain not in {"month", "week"}:
        raise ValueError("TRAINING_BATCH_GRAIN must be `month` or `week`.")
    return grain


def _normalize_trip_type(raw_value: str | None) -> str:
    trip_type = (raw_value or "yellow").strip().lower()
    if trip_type != "yellow":
        raise ValueError("TRIP_TYPE must currently be `yellow`.")
    return trip_type


def build_settings(load_env_file: bool = True) -> Settings:
    if load_env_file:
        load_dotenv(override=False)
    account = _first_env("SNOWFLAKE_ACCOUNT", "SF_ACCOUNT", "SF_HOST")
    return Settings(
        snowflake_account=_normalize_snowflake_account(account),
        snowflake_user=_first_env("SNOWFLAKE_USER", "SF_USER"),
        snowflake_password=_first_env("SNOWFLAKE_PASSWORD", "SF_PASSWORD"),
        snowflake_role=_first_env("SNOWFLAKE_ROLE", "SF_ROLE"),
        snowflake_warehouse=_first_env("SNOWFLAKE_WAREHOUSE", "SF_WAREHOUSE"),
        snowflake_database=_first_env("SNOWFLAKE_DATABASE", "SF_DATABASE"),
        trip_type=_normalize_trip_type(_first_env("TRIP_TYPE", default="yellow")),
        snowflake_schema_raw=_first_env("SNOWFLAKE_SCHEMA_RAW", "SF_RAW_SCHEMA", default="RAW") or "RAW",
        snowflake_schema_staging=_first_env("SNOWFLAKE_SCHEMA_STAGING", default="STAGING") or "STAGING",
        snowflake_schema_analytics=_first_env(
            "SNOWFLAKE_SCHEMA_ANALYTICS",
            "SF_ANALYTICS_SCHEMA",
            default="ANALYTICS",
        )
        or "ANALYTICS",
        snowflake_schema_ml=_first_env("SNOWFLAKE_SCHEMA_ML", default="ML") or "ML",
        data_start_date=_first_env("DATA_START_DATE", default="2025-01-01") or "2025-01-01",
        data_end_date=_first_env("DATA_END_DATE", default="2025-01-31") or "2025-01-31",
        train_end_date=_first_env("TRAIN_END_DATE", default="2025-01-21") or "2025-01-21",
        val_end_date=_first_env("VAL_END_DATE", default="2025-01-27") or "2025-01-27",
        nyc_taxi_month=_first_env("NYC_TAXI_MONTH"),
        nyc_tlc_base_url=_first_env(
            "NYC_TLC_BASE_URL",
            default="https://d37ci6vzurychx.cloudfront.net/trip-data",
        )
        or "https://d37ci6vzurychx.cloudfront.net/trip-data",
        local_data_dir=Path(_first_env("LOCAL_DATA_DIR", default="data/raw") or "data/raw"),
        model_dir=Path(_first_env("MODEL_DIR", default="data/models") or "data/models"),
        enable_download=_parse_bool(_first_env("ENABLE_DOWNLOAD"), default=True),
        enable_stage_upload=_parse_bool(_first_env("ENABLE_STAGE_UPLOAD"), default=True),
        enable_copy_into=_parse_bool(_first_env("ENABLE_COPY_INTO"), default=True),
        enable_zone_lookup=_parse_bool(_first_env("ENABLE_ZONE_LOOKUP"), default=False),
        zone_lookup_path=Path(
            _first_env("ZONE_LOOKUP_PATH", default="data/raw/taxi_zone_lookup.csv")
            or "data/raw/taxi_zone_lookup.csv"
        ),
        eda_sample_limit=_parse_positive_int(_first_env("EDA_SAMPLE_LIMIT"), default=10_000),
        eda_sample_seed=_parse_positive_int(_first_env("EDA_SAMPLE_SEED"), default=42),
        train_sample_limit=_parse_positive_int(_first_env("TRAIN_SAMPLE_LIMIT"), default=50_000),
        train_sample_pct=_parse_positive_float(_first_env("TRAIN_SAMPLE_PCT"), default=1.0),
        batch_size=_parse_positive_int(_first_env("BATCH_SIZE"), default=50_000),
        training_batch_grain=_normalize_batch_grain(_first_env("TRAINING_BATCH_GRAIN")),
        target_column=_first_env("MODEL_TARGET", default="fare_amount") or "fare_amount",
    )


@lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    return build_settings()


def get_settings(
    reload: bool = False,
    validate: bool = True,
    load_env_file: bool = True,
) -> Settings:
    if reload:
        _cached_settings.cache_clear()
    settings = build_settings(load_env_file=load_env_file) if reload else _cached_settings()
    if validate:
        validate_required_settings(settings)
        validate_date_settings(settings)
    return settings


def missing_required_settings(settings: Settings | None = None) -> list[str]:
    effective_settings = settings or get_settings(validate=False)
    return [key for key, value in effective_settings.required_env.items() if not value]


def validate_required_settings(settings: Settings | None = None) -> None:
    missing = missing_required_settings(settings)
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required environment variables: {missing_text}")


def validate_date_settings(settings: Settings | None = None) -> None:
    effective_settings = settings or get_settings()
    start_dt = date.fromisoformat(effective_settings.data_start_date)
    train_end_dt = date.fromisoformat(effective_settings.train_end_date)
    val_end_dt = date.fromisoformat(effective_settings.val_end_date)
    end_dt = date.fromisoformat(effective_settings.data_end_date)

    if not (start_dt <= train_end_dt < val_end_dt < end_dt):
        raise ValueError(
            "Expected DATA_START_DATE <= TRAIN_END_DATE < VAL_END_DATE < DATA_END_DATE."
        )

    if effective_settings.target_column != "fare_amount":
        raise ValueError("This project currently supports only MODEL_TARGET=`fare_amount`.")


def ensure_model_dir(settings: Settings | None = None) -> Path:
    effective_settings = settings or get_settings()
    effective_settings.model_dir.mkdir(parents=True, exist_ok=True)
    return effective_settings.model_dir


def get_snowflake_connection_params(settings: Settings | None = None) -> dict[str, str]:
    effective_settings = settings or get_settings()
    validate_required_settings(effective_settings)
    return {
        "account": effective_settings.snowflake_account or "",
        "user": effective_settings.snowflake_user or "",
        "password": effective_settings.snowflake_password or "",
        "role": effective_settings.snowflake_role or "",
        "warehouse": effective_settings.snowflake_warehouse or "",
        "database": effective_settings.snowflake_database or "",
    }


def sql_file_paths(settings: Settings | None = None) -> Iterable[Path]:
    _ = settings or get_settings(validate=False)
    base_dir = Path(__file__).resolve().parents[1] / "data" / "sql"
    return [
        base_dir / "00_create_schemas.sql",
        base_dir / "01_create_external_or_stage_tables.sql",
        base_dir / "02_create_staging_trips_dev.sql",
        base_dir / "03_create_obt_trips_dev.sql",
        base_dir / "04_create_time_splits_dev.sql",
    ]


def sql_template_context(settings: Settings | None = None) -> dict[str, str]:
    effective_settings = settings or get_settings(validate=False)
    return {
        "DATABASE": effective_settings.snowflake_database or "",
        "RAW_SCHEMA": effective_settings.snowflake_schema_raw,
        "STAGING_SCHEMA": effective_settings.snowflake_schema_staging,
        "ANALYTICS_SCHEMA": effective_settings.snowflake_schema_analytics,
        "ML_SCHEMA": effective_settings.snowflake_schema_ml,
        "DATA_START_DATE": effective_settings.data_start_date,
        "DATA_END_DATE": effective_settings.data_end_date,
        "TRAIN_END_DATE": effective_settings.train_end_date,
        "VAL_END_DATE": effective_settings.val_end_date,
        "ENABLE_ZONE_LOOKUP": "TRUE" if effective_settings.enable_zone_lookup else "FALSE",
    }
