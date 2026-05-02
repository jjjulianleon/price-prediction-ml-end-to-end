-- Build a dev OBT with only prediction-safe columns for pre-trip fare estimation.
CREATE OR REPLACE TABLE {{DATABASE}}.{{ANALYTICS_SCHEMA}}.OBT_TRIPS_DEV AS
SELECT
    tpep_pickup_datetime AS pickup_datetime,
    EXTRACT(HOUR FROM tpep_pickup_datetime) AS pickup_hour,
    DAYOFWEEKISO(tpep_pickup_datetime) AS pickup_dayofweek,
    EXTRACT(MONTH FROM tpep_pickup_datetime) AS pickup_month,
    IFF(DAYOFWEEKISO(tpep_pickup_datetime) IN (6, 7), 1, 0) AS is_weekend,
    passenger_count,
    trip_distance,
    pulocationid AS pickup_location_id,
    dolocationid AS dropoff_location_id,
    vendorid AS vendor_id,
    ratecodeid AS ratecode_id,
    fare_amount
FROM {{DATABASE}}.{{RAW_SCHEMA}}.YELLOW_TRIPS_DEV
WHERE tpep_pickup_datetime IS NOT NULL
  AND tpep_dropoff_datetime IS NOT NULL
  AND CAST(tpep_pickup_datetime AS DATE) BETWEEN TO_DATE('{{DATA_START_DATE}}') AND TO_DATE('{{DATA_END_DATE}}')
  AND tpep_dropoff_datetime > tpep_pickup_datetime
  AND trip_distance > 0
  AND passenger_count BETWEEN 1 AND 6
  AND fare_amount > 0
  AND pulocationid IS NOT NULL
  AND dolocationid IS NOT NULL;

-- Leakage guardrails for this dev OBT:
-- Excluded from features: payment_type, tip_amount, tolls_amount, mta_tax,
-- improvement_surcharge, congestion_surcharge, airport_fee, total_amount,
-- trip_duration-derived columns, and any value known only after trip completion.
