-- Base raw tables for Yellow and Green Taxi data in Snowflake.
-- The default project path now performs automatic ingestion from the official NYC TLC parquet
-- through `python3 -m src.data.ingestion ingest` or `python3 -m src.data.ingestion bootstrap_raw`.
-- The commented alternatives below remain useful if the team later decides to load from an internal stage
-- or from an already materialized raw table.

CREATE OR REPLACE FILE FORMAT {{DATABASE}}.{{RAW_SCHEMA}}.NYC_TAXI_PARQUET
    TYPE = PARQUET
    USE_LOGICAL_TYPE = TRUE;

CREATE OR REPLACE FILE FORMAT {{DATABASE}}.{{RAW_SCHEMA_GREEN}}.NYC_TAXI_PARQUET
    TYPE = PARQUET
    USE_LOGICAL_TYPE = TRUE;

CREATE OR REPLACE STAGE {{DATABASE}}.{{RAW_SCHEMA}}.NYC_TAXI_STAGE
    FILE_FORMAT = {{DATABASE}}.{{RAW_SCHEMA}}.NYC_TAXI_PARQUET;

CREATE OR REPLACE STAGE {{DATABASE}}.{{RAW_SCHEMA_GREEN}}.NYC_TAXI_STAGE
    FILE_FORMAT = {{DATABASE}}.{{RAW_SCHEMA_GREEN}}.NYC_TAXI_PARQUET;

CREATE TABLE IF NOT EXISTS {{DATABASE}}.{{RAW_SCHEMA}}.YELLOW_TRIPS_DEV (
    vendorid NUMBER,
    tpep_pickup_datetime TIMESTAMP_NTZ,
    tpep_dropoff_datetime TIMESTAMP_NTZ,
    passenger_count NUMBER,
    trip_distance FLOAT,
    ratecodeid NUMBER,
    store_and_fwd_flag STRING,
    pulocationid NUMBER,
    dolocationid NUMBER,
    payment_type NUMBER,
    fare_amount FLOAT,
    extra FLOAT,
    mta_tax FLOAT,
    tip_amount FLOAT,
    tolls_amount FLOAT,
    improvement_surcharge FLOAT,
    total_amount FLOAT,
    congestion_surcharge FLOAT,
    airport_fee FLOAT
);

CREATE TABLE IF NOT EXISTS {{DATABASE}}.{{RAW_SCHEMA_GREEN}}.GREEN_TRIPS_DEV (
    vendorid NUMBER,
    lpep_pickup_datetime TIMESTAMP_NTZ,
    lpep_dropoff_datetime TIMESTAMP_NTZ,
    store_and_fwd_flag STRING,
    ratecodeid NUMBER,
    pulocationid NUMBER,
    dolocationid NUMBER,
    passenger_count NUMBER,
    trip_distance FLOAT,
    fare_amount FLOAT,
    extra FLOAT,
    mta_tax FLOAT,
    tip_amount FLOAT,
    tolls_amount FLOAT,
    ehail_fee FLOAT,
    improvement_surcharge FLOAT,
    total_amount FLOAT,
    payment_type NUMBER,
    trip_type STRING,
    congestion_surcharge FLOAT
);

CREATE TABLE IF NOT EXISTS {{DATABASE}}.{{RAW_SCHEMA}}.RAW_LOAD_AUDIT (
    file_name STRING,
    trip_type STRING,
    period_label STRING,
    local_path STRING,
    copy_status STRING,
    rows_loaded NUMBER,
    loaded_at TIMESTAMP_NTZ,
    PRIMARY KEY (file_name)
);

CREATE TABLE IF NOT EXISTS {{DATABASE}}.{{RAW_SCHEMA_GREEN}}.RAW_LOAD_AUDIT (
    file_name STRING,
    trip_type STRING,
    period_label STRING,
    local_path STRING,
    copy_status STRING,
    rows_loaded NUMBER,
    loaded_at TIMESTAMP_NTZ,
    PRIMARY KEY (file_name)
);

CREATE TABLE IF NOT EXISTS {{DATABASE}}.{{STAGING_SCHEMA}}.TAXI_ZONE_LOOKUP (
    location_id NUMBER,
    borough STRING,
    zone STRING,
    service_zone STRING,
    is_airport NUMBER
);
--
-- Example load for one month:
-- COPY INTO {{DATABASE}}.{{RAW_SCHEMA}}.YELLOW_TRIPS_DEV
-- FROM @{{DATABASE}}.{{RAW_SCHEMA}}.NYC_TAXI_STAGE/yellow_tripdata_2025-01.parquet
-- MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;
--
-- COPY INTO {{DATABASE}}.{{RAW_SCHEMA_GREEN}}.GREEN_TRIPS_DEV
-- FROM @{{DATABASE}}.{{RAW_SCHEMA_GREEN}}.NYC_TAXI_STAGE/green_tripdata_2025-01.parquet
-- MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;

-- Alternative if data already exists in a broader RAW table:
-- INSERT OVERWRITE INTO {{DATABASE}}.{{RAW_SCHEMA}}.YELLOW_TRIPS_DEV
-- SELECT
--     VendorID,
--     tpep_pickup_datetime,
--     tpep_dropoff_datetime,
--     passenger_count,
--     trip_distance,
--     RatecodeID,
--     store_and_fwd_flag,
--     PULocationID,
--     DOLocationID,
--     payment_type,
--     fare_amount,
--     extra,
--     mta_tax,
--     tip_amount,
--     tolls_amount,
--     improvement_surcharge,
--     total_amount,
--     congestion_surcharge,
--     airport_fee
-- FROM {{DATABASE}}.{{RAW_SCHEMA}}.YELLOW_TRIPS
-- WHERE CAST(tpep_pickup_datetime AS DATE) BETWEEN TO_DATE('{{DATA_START_DATE}}') AND TO_DATE('{{DATA_END_DATE}}');
