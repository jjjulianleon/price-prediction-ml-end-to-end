"""Snowflake-first ingestion and batch download utilities."""

from __future__ import annotations

import argparse
import io
import logging
import tempfile
from datetime import datetime
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
    settings: Settings | None = None,
) -> pd.DataFrame:
    wrapped_query = query.strip().rstrip(";")
    sample_clause = f" SAMPLE ({sample_pct})" if sample_pct is not None else ""
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


def month_file_name(settings: Settings) -> str:
    start_dt = datetime.fromisoformat(settings.data_start_date)
    end_dt = datetime.fromisoformat(settings.data_end_date)
    if start_dt.year != end_dt.year or start_dt.month != end_dt.month:
        raise ValueError(
            "Automatic TLC ingestion currently supports a single monthly parquet per run. "
            "Set DATA_START_DATE and DATA_END_DATE within the same month."
        )
    return f"yellow_tripdata_{start_dt.year:04d}-{start_dt.month:02d}.parquet"


def month_file_url(settings: Settings) -> str:
    return f"{settings.nyc_tlc_base_url.rstrip('/')}/{month_file_name(settings)}"


def download_tlc_month_file(settings: Settings) -> Path:
    url = month_file_url(settings)
    target_dir = Path(tempfile.mkdtemp(prefix="nyc_taxi_ingest_"))
    destination = target_dir / month_file_name(settings)
    LOGGER.info("Downloading NYC TLC parquet | url=%s", url)
    urlretrieve(url, destination)
    LOGGER.info("Downloaded parquet to %s", destination)
    return destination


def truncate_raw_dev_table(settings: Settings) -> None:
    LOGGER.info("Truncating raw dev table: %s", settings.raw_table)
    execute_sql(f"TRUNCATE TABLE {settings.raw_table}", settings=settings)


def put_file_to_stage(local_path: Path, settings: Settings) -> None:
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
    connection = create_connection(settings=settings)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"REMOVE @{settings.raw_stage}/{file_name}")
    finally:
        close_quietly(connection)


def ingest_tlc_month(settings: Settings | None = None, overwrite: bool = True) -> None:
    effective_settings = settings or get_settings()
    parquet_path = download_tlc_month_file(effective_settings)
    file_name = parquet_path.name
    try:
        if overwrite:
            truncate_raw_dev_table(effective_settings)
        put_file_to_stage(parquet_path, effective_settings)
        copy_stage_to_raw(effective_settings, file_name)
        raw_count = fetch_scalar(f"SELECT COUNT(*) FROM {effective_settings.raw_table}", settings=effective_settings)
        LOGGER.info(
            "Completed automatic TLC ingestion | raw_table=%s | total_rows=%s | month=%s",
            effective_settings.raw_table,
            raw_count,
            effective_settings.processing_window_label,
        )
    except Exception:
        LOGGER.exception(
            "Automatic TLC ingestion failed | raw_table=%s | month=%s",
            effective_settings.raw_table,
            effective_settings.processing_window_label,
        )
        raise
    finally:
        try:
            remove_stage_file(effective_settings, file_name)
        except Exception:
            LOGGER.warning("Could not clean staged file %s", file_name)


def bootstrap(settings: Settings | None = None) -> None:
    effective_settings = settings or get_settings()
    execute_sql_group("setup", settings=effective_settings)
    ingest_tlc_month(settings=effective_settings, overwrite=True)
    execute_sql_group("transform", settings=effective_settings)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Execute Snowflake ingestion and SQL workflows.")
    parser.add_argument(
        "command",
        nargs="?",
        default="bootstrap",
        choices=["setup", "ingest", "transform", "all", "bootstrap"],
        help="Workflow command to execute.",
    )
    args = parser.parse_args()

    if args.command in {"setup", "transform", "all"}:
        execute_sql_group(args.command)
        return
    if args.command == "ingest":
        ingest_tlc_month()
        return
    if args.command == "bootstrap":
        bootstrap()
        return


if __name__ == "__main__":
    main()
