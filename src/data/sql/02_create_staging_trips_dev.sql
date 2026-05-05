-- Build a staging table with typed, cleaned and documented modeling inputs.
CREATE OR REPLACE TABLE {{DATABASE}}.{{STAGING_SCHEMA}}.TRIPS_STAGE_DEV AS
SELECT
    tpep_pickup_datetime AS pickup_datetime,
    tpep_dropoff_datetime,
    fare_amount,
    trip_distance,
    trip_distance AS estimated_distance,
    passenger_count,
    vendorid AS vendor_id,
    ratecodeid AS ratecode_id,
    pulocationid AS pickup_location_id,
    dolocationid AS dropoff_location_id,
    payment_type,
    extra,
    mta_tax,
    tip_amount,
    tolls_amount,
    improvement_surcharge,
    congestion_surcharge,
    airport_fee,
    total_amount,
    DATEDIFF('minute', tpep_pickup_datetime, tpep_dropoff_datetime) AS trip_duration_min,
    IFF(
        DATEDIFF('second', tpep_pickup_datetime, tpep_dropoff_datetime) > 0,
        trip_distance / NULLIF(DATEDIFF('second', tpep_pickup_datetime, tpep_dropoff_datetime) / 3600.0, 0),
        NULL
    ) AS speed_mph
FROM {{DATABASE}}.{{RAW_SCHEMA}}.YELLOW_TRIPS_DEV
WHERE CAST(tpep_pickup_datetime AS DATE) BETWEEN TO_DATE('{{DATA_START_DATE}}') AND TO_DATE('{{DATA_END_DATE}}')
  AND tpep_pickup_datetime IS NOT NULL
  AND fare_amount IS NOT NULL
  AND fare_amount >= 0
  AND trip_distance > 0
  AND passenger_count BETWEEN 1 AND 6
  AND pulocationid IS NOT NULL
  AND dolocationid IS NOT NULL;

-- This layer keeps diagnostic columns that are valid for quality checks but forbidden for features.
