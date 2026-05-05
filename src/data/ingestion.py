"""Snowflake-first ingestion and batch download utilities."""

from __future__ import annotations

import argparse
import io
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Iterator, Sequence
from urllib.request import urlretrieve

import pandas as pd

from src.utils.config import Settings, get_settings, sql_file_paths, sql_template_context
from src.utils.snowflake_conn import close_quietly, create_connection

LOGGER = logging.getLogger("src.data.ingestion")


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(column).lower() for column in normalized.columns]
    return normalized


def configure_logging() -> None:
    if LOGGER.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def read_sql_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def render_sql_template(sql_text: str, settings: Settings | None = None) -> str:
    rendered = sql_text
    for key, value in sql_template_context(settings).items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def execute_sql(sql: str, settings: Settings | None = None) -> None:
    connection = create_connection(settings=settings)
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql)
    finally:
        close_quietly(connection)


def execute_sql_file(path: str | Path, settings: Settings | None = None) -> None:
    effective_settings = settings or get_settings()
    file_path = Path(path)
    sql_text = render_sql_template(read_sql_file(path), settings=effective_settings)
    connection = create_connection(settings=effective_settings)
    try:
        LOGGER.info("Running SQL file: %s", file_path.name)
        for cursor in connection.execute_stream(io.StringIO(sql_text), remove_comments=True):
            close_quietly(cursor)
        LOGGER.info("Completed SQL file: %s", file_path.name)
    except Exception:
        LOGGER.exception(
            "Failed SQL file: %s | database=%s | window=%s",
            file_path.name,
            effective_settings.snowflake_database,
            effective_settings.processing_window_label,
        )
        raise
    finally:
        close_quietly(connection)


def execute_sql_files(paths: Sequence[str | Path], settings: Settings | None = None) -> None:
    for path in paths:
        execute_sql_file(path, settings=settings)


def execute_sql_group(group: str, settings: Settings | None = None) -> None:
    effective_settings = settings or get_settings()
    paths = list(sql_file_group_paths(group, settings=effective_settings))
    LOGGER.info(
        "Starting SQL group `%s` | database=%s | window=%s",
        group,
        effective_settings.snowflake_database,
        effective_settings.processing_window_label,
    )
    execute_sql_files(paths, settings=effective_settings)
    if group in {"transform", "all"}:
        log_transform_summary(effective_settings)
    LOGGER.info("Finished SQL group `%s`", group)


def fetch_data_in_batches(
    query: str,
    batch_size: int = 50_000,
    settings: Settings | None = None,
) -> Iterator[pd.DataFrame]:
    connection = create_connection(settings=settings)
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        cursor.arraysize = batch_size

        if hasattr(cursor, "fetch_pandas_batches"):
            for batch_df in cursor.fetch_pandas_batches():
                if not batch_df.empty:
                    yield normalize_dataframe_columns(batch_df)
            return

        columns = [meta[0] for meta in cursor.description]
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            yield normalize_dataframe_columns(pd.DataFrame(rows, columns=columns))
    finally:
        close_quietly(cursor)
        close_quietly(connection)


def fetch_sample(
    query: str,
    sample_pct: float | None = None,
    limit: int = 5_000,
    sample_seed: int | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    wrapped_query = query.strip().rstrip(";")
    seed_clause = f" SEED ({sample_seed})" if sample_pct is not None and sample_seed is not None else ""
    sample_clause = f" SAMPLE ({sample_pct}){seed_clause}" if sample_pct is not None else ""
    sample_query = f"SELECT * FROM ({wrapped_query}) AS base{sample_clause} LIMIT {limit}"
    batches = list(fetch_data_in_batches(sample_query, batch_size=limit, settings=settings))
    if not batches:
        return pd.DataFrame()
    return pd.concat(batches, ignore_index=True)


def default_train_query(settings: Settings | None = None) -> str:
    effective_settings = settings or get_settings()
    return f"SELECT * FROM {effective_settings.train_table} ORDER BY pickup_datetime"


def weekly_query(table_name: str, week_start_day: int, week_end_day: int) -> str:
    return (
        f"SELECT * FROM {table_name} "
        f"WHERE EXTRACT(DAY FROM pickup_datetime) BETWEEN {week_start_day} AND {week_end_day} "
        f"ORDER BY pickup_datetime"
    )


def sql_file_group_paths(group: str, settings: Settings | None = None) -> Sequence[Path]:
    all_paths = list(sql_file_paths(settings))
    groups = {
        "setup": all_paths[:2],
        "transform": all_paths[2:],
        "all": all_paths,
    }
    if group not in groups:
        valid = ", ".join(sorted(groups))
        raise ValueError(f"Unknown SQL group `{group}`. Expected one of: {valid}")
    return groups[group]


def fetch_scalar(query: str, settings: Settings | None = None) -> int:
    connection = create_connection(settings=settings)
    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            row = cursor.fetchone()
            if row is None:
                return 0
            return int(row[0] or 0)
    finally:
        close_quietly(connection)


def fetch_exists(query: str, settings: Settings | None = None) -> bool:
    connection = create_connection(settings=settings)
    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchone() is not None
    finally:
        close_quietly(connection)


def object_exists(object_name: str, object_type: str, settings: Settings | None = None) -> bool:
    location = ".".join(object_name.split(".")[:-1])
    name = object_name.split(".")[-1]
    query = f"SHOW {object_type} LIKE '{name}' IN {location}"
    connection = create_connection(settings=settings)
    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchone() is not None
    finally:
        close_quietly(connection)


def log_transform_summary(settings: Settings) -> None:
    LOGGER.info("Transform summary for window=%s", settings.processing_window_label)
    objects = [
        ("raw_rows", settings.raw_table, "TABLES"),
        ("staging_rows", settings.staging_table, "TABLES"),
        ("obt_rows", settings.obt_table, "TABLES"),
        ("train_rows", settings.train_table, "VIEWS"),
        ("val_rows", settings.val_table, "VIEWS"),
        ("test_rows", settings.test_table, "VIEWS"),
    ]
    for label, object_name, object_type in objects:
        try:
            if not object_exists(object_name, object_type, settings=settings):
                LOGGER.warning("%s | object missing: %s", label, object_name)
                continue
            count = fetch_scalar(f"SELECT COUNT(*) FROM {object_name}", settings=settings)
            LOGGER.info("%s=%s | object=%s", label, count, object_name)
        except Exception:
            LOGGER.exception("Failed to collect summary metric for %s", object_name)
    log_obt_filter_diagnostics(settings)


def fetch_one_row(query: str, settings: Settings | None = None):
    connection = create_connection(settings=settings)
    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchone()
    finally:
        close_quietly(connection)


def log_obt_filter_diagnostics(settings: Settings) -> None:
    diagnostic_query = f"""
    SELECT
        COUNT(*) AS raw_total,
        COUNT_IF(tpep_pickup_datetime IS NOT NULL) AS nonnull_pickup,
        COUNT_IF(tpep_dropoff_datetime IS NOT NULL) AS nonnull_dropoff,
        COUNT_IF(CAST(tpep_pickup_datetime AS DATE) BETWEEN TO_DATE('{settings.data_start_date}') AND TO_DATE('{settings.data_end_date}')) AS in_date_range,
        COUNT_IF(tpep_dropoff_datetime > tpep_pickup_datetime) AS valid_time_order,
        COUNT_IF(trip_distance > 0) AS positive_trip_distance,
        COUNT_IF(passenger_count BETWEEN 1 AND 6) AS passenger_range_ok,
        COUNT_IF(fare_amount > 0) AS positive_fare_amount,
        COUNT_IF(pulocationid IS NOT NULL) AS nonnull_pulocationid,
        COUNT_IF(dolocationid IS NOT NULL) AS nonnull_dolocationid,
        COUNT_IF(
            tpep_pickup_datetime IS NOT NULL
            AND tpep_dropoff_datetime IS NOT NULL
            AND CAST(tpep_pickup_datetime AS DATE) BETWEEN TO_DATE('{settings.data_start_date}') AND TO_DATE('{settings.data_end_date}')
            AND tpep_dropoff_datetime > tpep_pickup_datetime
            AND trip_distance > 0
            AND passenger_count BETWEEN 1 AND 6
            AND fare_amount > 0
            AND pulocationid IS NOT NULL
            AND dolocationid IS NOT NULL
        ) AS rows_passing_all_filters
    FROM {settings.raw_table}
    """
    try:
        result = fetch_one_row(diagnostic_query, settings=settings)
        if result is None:
            LOGGER.warning("OBT diagnostics returned no rows for %s", settings.raw_table)
            return

        columns = [
            "raw_total",
            "nonnull_pickup",
            "nonnull_dropoff",
            "in_date_range",
            "valid_time_order",
            "positive_trip_distance",
            "passenger_range_ok",
            "positive_fare_amount",
            "nonnull_pulocationid",
            "nonnull_dolocationid",
            "rows_passing_all_filters",
        ]
        LOGGER.info("OBT filter diagnostics:")
        for name, value in zip(columns, result):
            LOGGER.info("  %s=%s", name, value)

        if int(result[-1] or 0) == 0:
            log_problem_samples(settings)
    except Exception:
        LOGGER.exception("Failed to compute OBT diagnostics for %s", settings.raw_table)


def log_problem_samples(settings: Settings) -> None:
    sample_query = f"""
    SELECT
        vendorid,
        tpep_pickup_datetime,
        tpep_dropoff_datetime,
        passenger_count,
        trip_distance,
        pulocationid,
        dolocationid,
        fare_amount,
        CASE WHEN tpep_pickup_datetime IS NULL THEN 1 ELSE 0 END AS bad_pickup_null,
        CASE WHEN tpep_dropoff_datetime IS NULL THEN 1 ELSE 0 END AS bad_dropoff_null,
        CASE WHEN CAST(tpep_pickup_datetime AS DATE) NOT BETWEEN TO_DATE('{settings.data_start_date}') AND TO_DATE('{settings.data_end_date}') THEN 1 ELSE 0 END AS bad_date_range,
        CASE WHEN tpep_dropoff_datetime <= tpep_pickup_datetime THEN 1 ELSE 0 END AS bad_time_order,
        CASE WHEN trip_distance <= 0 THEN 1 ELSE 0 END AS bad_trip_distance,
        CASE WHEN passenger_count NOT BETWEEN 1 AND 6 THEN 1 ELSE 0 END AS bad_passenger_count,
        CASE WHEN fare_amount <= 0 THEN 1 ELSE 0 END AS bad_fare_amount,
        CASE WHEN pulocationid IS NULL THEN 1 ELSE 0 END AS bad_pulocationid,
        CASE WHEN dolocationid IS NULL THEN 1 ELSE 0 END AS bad_dolocationid
    FROM {settings.raw_table}
    WHERE NOT (
        tpep_pickup_datetime IS NOT NULL
        AND tpep_dropoff_datetime IS NOT NULL
        AND CAST(tpep_pickup_datetime AS DATE) BETWEEN TO_DATE('{settings.data_start_date}') AND TO_DATE('{settings.data_end_date}')
        AND tpep_dropoff_datetime > tpep_pickup_datetime
        AND trip_distance > 0
        AND passenger_count BETWEEN 1 AND 6
        AND fare_amount > 0
        AND pulocationid IS NOT NULL
        AND dolocationid IS NOT NULL
    )
    LIMIT 10
    """
    try:
        sample_df = fetch_sample(sample_query, limit=10, settings=settings)
        if sample_df.empty:
            LOGGER.warning("No invalid sample rows found for diagnostics.")
            return
        LOGGER.warning("Sample rows failing OBT filters:\n%s", sample_df.to_string(index=False))
    except Exception:
        LOGGER.exception("Failed to fetch invalid sample rows for OBT diagnostics")


def iter_month_starts(settings: Settings) -> list[date]:
    start_dt = date.fromisoformat(settings.data_start_date).replace(day=1)
    end_dt = date.fromisoformat(settings.data_end_date).replace(day=1)
    month_starts: list[date] = []
    current = start_dt
    while current <= end_dt:
        month_starts.append(current)
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return month_starts


def month_file_name(month_start: date) -> str:
    return f"yellow_tripdata_{month_start.year:04d}-{month_start.month:02d}.parquet"


def month_file_url(settings: Settings, month_start: date) -> str:
    return f"{settings.nyc_tlc_base_url.rstrip('/')}/{month_file_name(month_start)}"


def download_tlc_month_files(settings: Settings) -> list[Path]:
    target_dir = settings.local_data_dir / settings.trip_type
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files: list[Path] = []

    for month_start in iter_month_starts(settings):
        file_name = month_file_name(month_start)
        destination = target_dir / file_name
        if destination.exists():
            LOGGER.info("Skipping download; parquet already exists | path=%s", destination)
            downloaded_files.append(destination)
            continue
        if not settings.enable_download:
            raise FileNotFoundError(
                f"Download disabled and parquet missing locally: {destination}"
            )
        url = month_file_url(settings, month_start)
        LOGGER.info("Downloading NYC TLC parquet | url=%s", url)
        urlretrieve(url, destination)
        LOGGER.info("Downloaded parquet to %s", destination)
        downloaded_files.append(destination)

    return downloaded_files


def file_already_loaded(file_name: str, settings: Settings) -> bool:
    query = f"""
    SELECT 1
    FROM {settings.raw_load_audit_table}
    WHERE file_name = '{file_name}'
      AND copy_status = 'COPIED'
    LIMIT 1
    """
    return fetch_exists(query, settings=settings)


def log_load_audit(
    file_name: str,
    local_path: Path,
    settings: Settings,
    copy_status: str,
    rows_loaded: int,
) -> None:
    month_label = file_name.replace("yellow_tripdata_", "").replace(".parquet", "")
    query = f"""
    MERGE INTO {settings.raw_load_audit_table} AS target
    USING (
        SELECT
            '{file_name}' AS file_name,
            '{settings.trip_type}' AS trip_type,
            '{month_label}' AS period_label,
            '{str(local_path)}' AS local_path,
            '{copy_status}' AS copy_status,
            {rows_loaded} AS rows_loaded,
            CURRENT_TIMESTAMP() AS loaded_at
    ) AS source
    ON target.file_name = source.file_name
    WHEN MATCHED THEN UPDATE SET
        trip_type = source.trip_type,
        period_label = source.period_label,
        local_path = source.local_path,
        copy_status = source.copy_status,
        rows_loaded = source.rows_loaded,
        loaded_at = source.loaded_at
    WHEN NOT MATCHED THEN INSERT (
        file_name,
        trip_type,
        period_label,
        local_path,
        copy_status,
        rows_loaded,
        loaded_at
    ) VALUES (
        source.file_name,
        source.trip_type,
        source.period_label,
        source.local_path,
        source.copy_status,
        source.rows_loaded,
        source.loaded_at
    )
    """
    execute_sql(query, settings=settings)


def put_file_to_stage(local_path: Path, settings: Settings) -> None:
    if not settings.enable_stage_upload:
        LOGGER.info("Skipping PUT because ENABLE_STAGE_UPLOAD=false | path=%s", local_path)
        return
    connection = create_connection(settings=settings)
    try:
        with connection.cursor() as cursor:
            put_sql = (
                f"PUT 'file://{local_path}' @{settings.raw_stage} "
                "AUTO_COMPRESS=FALSE OVERWRITE=TRUE PARALLEL=8"
            )
            LOGGER.info("Uploading parquet to Snowflake stage | stage=%s", settings.raw_stage)
            cursor.execute(put_sql)
            for row in cursor.fetchall():
                LOGGER.info("PUT result: %s", row)
    finally:
        close_quietly(connection)


def copy_stage_to_raw(settings: Settings, file_name: str) -> None:
    if not settings.enable_copy_into:
        LOGGER.info("Skipping COPY INTO because ENABLE_COPY_INTO=false | file=%s", file_name)
        return
    copy_sql = f"""
    COPY INTO {settings.raw_table}
    FROM @{settings.raw_stage}/{file_name}
    FILE_FORMAT = (
        TYPE = PARQUET
        USE_LOGICAL_TYPE = TRUE
    )
    MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE
    ON_ERROR = 'ABORT_STATEMENT'
    """
    connection = create_connection(settings=settings)
    try:
        with connection.cursor() as cursor:
            LOGGER.info("Copying staged parquet into raw table | raw_table=%s", settings.raw_table)
            LOGGER.info(
                "COPY options | file_format=PARQUET logical_type=true | file=%s",
                file_name,
            )
            cursor.execute(copy_sql)
            for row in cursor.fetchall():
                LOGGER.info("COPY result: %s", row)
    finally:
        close_quietly(connection)


def remove_stage_file(settings: Settings, file_name: str) -> None:
    if not settings.enable_stage_upload:
        return
    connection = create_connection(settings=settings)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"REMOVE @{settings.raw_stage}/{file_name}")
    finally:
        close_quietly(connection)


def ingest_tlc_period(settings: Settings | None = None, overwrite: bool = True) -> None:
    effective_settings = settings or get_settings()
    parquet_paths = download_tlc_month_files(effective_settings)
    file_names = [path.name for path in parquet_paths]
    try:
        for parquet_path in parquet_paths:
            if file_already_loaded(parquet_path.name, effective_settings) and not overwrite:
                LOGGER.info(
                    "Skipping Snowflake load; parquet already audited as copied | file=%s",
                    parquet_path.name,
                )
                continue
            put_file_to_stage(parquet_path, effective_settings)
            copy_stage_to_raw(effective_settings, parquet_path.name)
            log_load_audit(
                parquet_path.name,
                parquet_path,
                effective_settings,
                "COPIED" if effective_settings.enable_copy_into else "STAGED_ONLY",
                0,
            )
        raw_count = fetch_scalar(
            f"SELECT COUNT(*) FROM {effective_settings.raw_table}",
            settings=effective_settings,
        )
        LOGGER.info(
            "Completed automatic TLC ingestion | raw_table=%s | total_rows=%s | files=%s | window=%s",
            effective_settings.raw_table,
            raw_count,
            len(file_names),
            effective_settings.processing_window_label,
        )
    except Exception:
        LOGGER.exception(
            "Automatic TLC ingestion failed | raw_table=%s | window=%s",
            effective_settings.raw_table,
            effective_settings.processing_window_label,
        )
        raise
    finally:
        for file_name in file_names:
            try:
                remove_stage_file(effective_settings, file_name)
            except Exception:
                LOGGER.warning("Could not clean staged file %s", file_name)
        # Remove local parquet files after staging/copy to free disk space on large runs
        for local_path in parquet_paths:
            try:
                if local_path.exists():
                    local_path.unlink()
                    LOGGER.info("Removed local parquet file: %s", local_path)
            except Exception:
                LOGGER.warning("Could not remove local parquet file %s", local_path)


def ingest_tlc_month(settings: Settings | None = None, overwrite: bool = True) -> None:
    ingest_tlc_period(settings=settings, overwrite=overwrite)


def bootstrap_raw(settings: Settings | None = None) -> None:
    effective_settings = settings or get_settings()
    execute_sql_group("setup", settings=effective_settings)
    ingest_tlc_period(settings=effective_settings, overwrite=False)


def transform_model_data(settings: Settings | None = None) -> None:
    effective_settings = settings or get_settings()
    execute_sql_group("transform", settings=effective_settings)


def bootstrap_full(settings: Settings | None = None) -> None:
    effective_settings = settings or get_settings()
    bootstrap_raw(settings=effective_settings)
    transform_model_data(settings=effective_settings)


def bootstrap(settings: Settings | None = None) -> None:
    bootstrap_raw(settings=settings)


def preview_raw_sample(settings: Settings | None = None) -> None:
    effective_settings = settings or get_settings()
    sample_df = fetch_sample(
        f"SELECT * FROM {effective_settings.raw_table}",
        sample_pct=1.0,
        limit=effective_settings.eda_sample_limit,
        sample_seed=effective_settings.eda_sample_seed,
        settings=effective_settings,
    )
    LOGGER.info(
        "RAW sample preview for EDA | table=%s | rows=%s | columns=%s",
        effective_settings.raw_table,
        len(sample_df),
        list(sample_df.columns),
    )
    if not sample_df.empty:
        LOGGER.info("RAW sample head:\n%s", sample_df.head(10).to_string(index=False))


def preview_obt_sample(settings: Settings | None = None) -> None:
    effective_settings = settings or get_settings()
    sample_df = fetch_sample(
        f"SELECT * FROM {effective_settings.obt_table}",
        sample_pct=effective_settings.train_sample_pct,
        limit=effective_settings.eda_sample_limit,
        sample_seed=effective_settings.eda_sample_seed,
        settings=effective_settings,
    )
    LOGGER.info(
        "OBT sample preview | table=%s | rows=%s | columns=%s",
        effective_settings.obt_table,
        len(sample_df),
        list(sample_df.columns),
    )
    if not sample_df.empty:
        LOGGER.info("OBT sample head:\n%s", sample_df.head(10).to_string(index=False))


def preview_eda_sample(settings: Settings | None = None) -> None:
    preview_raw_sample(settings=settings)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Execute Snowflake ingestion and SQL workflows.")
    parser.add_argument(
        "command",
        nargs="?",
        default="bootstrap",
        choices=[
            "setup",
            "ingest",
            "transform",
            "all",
            "bootstrap",
            "bootstrap_raw",
            "bootstrap_full",
            "sample",
            "sample_raw",
            "sample_obt",
            "sql_all",
        ],
        help="Workflow command to execute.",
    )
    args = parser.parse_args()

    if args.command == "setup":
        execute_sql_group("setup")
        return
    if args.command == "transform":
        transform_model_data()
        return
    if args.command == "sql_all":
        execute_sql_group("all")
        return
    if args.command == "ingest":
        ingest_tlc_period()
        return
    if args.command in {"bootstrap", "bootstrap_raw"}:
        bootstrap_raw()
        return
    if args.command in {"bootstrap_full", "all"}:
        bootstrap_full()
        return
    if args.command in {"sample", "sample_raw"}:
        preview_raw_sample()
        return
    if args.command == "sample_obt":
        preview_obt_sample()
        return


if __name__ == "__main__":
    main()
