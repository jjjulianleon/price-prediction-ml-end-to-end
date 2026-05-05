-- Build a dev OBT with only prediction-safe columns for pre-trip fare estimation.
CREATE OR REPLACE TABLE {{DATABASE}}.{{ANALYTICS_SCHEMA}}.OBT_TRIPS_DEV AS
SELECT
    pickup_datetime,
    EXTRACT(HOUR FROM pickup_datetime) AS pickup_hour,
    DAYOFWEEKISO(pickup_datetime) AS pickup_dayofweek,
    EXTRACT(MONTH FROM pickup_datetime) AS pickup_month,
    IFF(DAYOFWEEKISO(pickup_datetime) IN (6, 7), 1, 0) AS is_weekend,
    IFF(EXTRACT(HOUR FROM pickup_datetime) IN (7, 8, 9, 16, 17, 18, 19), 1, 0) AS is_rush_hour,
    IFF(EXTRACT(HOUR FROM pickup_datetime) IN (22, 23, 0, 1, 2, 3, 4, 5), 1, 0) AS is_night,
    passenger_count,
    estimated_distance,
    estimated_distance AS trip_distance,
    LN(1 + estimated_distance) AS log_estimated_distance,
    pickup_location_id,
    dropoff_location_id,
    vendor_id,
    ratecode_id,
    CONCAT(pickup_location_id, '_', dropoff_location_id) AS route_id,
    IFF(pickup_location_id = dropoff_location_id, 1, 0) AS same_zone,
    fare_amount
FROM {{DATABASE}}.{{STAGING_SCHEMA}}.TRIPS_STAGE_DEV
WHERE tpep_dropoff_datetime > pickup_datetime
  AND fare_amount > 0
  AND trip_duration_min IS NOT NULL
  AND trip_duration_min > 0;

-- Leakage guardrails for this dev OBT:
-- Excluded from features: payment_type, tip_amount, tolls_amount, mta_tax,
-- improvement_surcharge, congestion_surcharge, airport_fee, total_amount,
-- trip_duration-derived columns, and any value known only after trip completion.
-- `trip_distance` is retained here only as a temporary compatibility alias for notebooks and legacy analysis.
