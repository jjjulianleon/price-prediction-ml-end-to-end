-- Build STAGING from RAW with the structural validations learned in EDA and cleaning.
-- Goal of this layer:
-- 1. unify Yellow and Green Taxi under one canonical schema
-- 2. keep only structurally valid trips for pre-trip fare modeling
-- 3. preserve diagnostic columns needed for QA, while keeping leakage out of the final OBT
CREATE OR REPLACE TABLE {{DATABASE}}.{{STAGING_SCHEMA}}.TRIPS_STAGE_DEV AS
WITH yellow_source AS (
    -- Yellow Taxi canonicalization:
    -- map vendor/location/ratecode names to the shared project contract
    SELECT
        'yellow' AS trip_type,
        tpep_pickup_datetime AS pickup_datetime,
        tpep_dropoff_datetime AS dropoff_datetime,
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
        total_amount
    FROM {{DATABASE}}.{{RAW_SCHEMA}}.YELLOW_TRIPS_DEV
    WHERE {{ENABLE_YELLOW}}
),
green_source AS (
    -- Green Taxi canonicalization:
    -- keep the same contract as Yellow; airport_fee is absent here and stays nullable
    SELECT
        'green' AS trip_type,
        lpep_pickup_datetime AS pickup_datetime,
        lpep_dropoff_datetime AS dropoff_datetime,
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
        NULL::FLOAT AS airport_fee,
        total_amount
    FROM {{DATABASE}}.{{RAW_SCHEMA}}.GREEN_TRIPS_DEV
    WHERE {{ENABLE_GREEN}}
),
unioned AS (
    -- At this point both fleets share one column contract and can be validated together.
    SELECT * FROM yellow_source
    UNION ALL
    SELECT * FROM green_source
)
SELECT
    -- Fleet identifier preserved as a modeling feature to capture structural differences
    trip_type,
    -- Core trip timestamps
    pickup_datetime,
    dropoff_datetime,
    -- Supervised target
    fare_amount,
    -- Historical distance retained and published as estimated_distance for modeling
    trip_distance,
    estimated_distance,
    -- Candidate safe inputs
    passenger_count,
    vendor_id,
    ratecode_id,
    pickup_location_id,
    dropoff_location_id,
    -- Diagnostic-only columns preserved for QA and audit, never for final features
    payment_type,
    extra,
    mta_tax,
    tip_amount,
    tolls_amount,
    improvement_surcharge,
    congestion_surcharge,
    airport_fee,
    total_amount,
    DATEDIFF('minute', pickup_datetime, dropoff_datetime) AS trip_duration_min,
    IFF(
        DATEDIFF('second', pickup_datetime, dropoff_datetime) > 0,
        trip_distance / NULLIF(DATEDIFF('second', pickup_datetime, dropoff_datetime) / 3600.0, 0),
        NULL
    ) AS speed_mph
FROM unioned
WHERE CAST(pickup_datetime AS DATE) BETWEEN TO_DATE('{{DATA_START_DATE}}') AND TO_DATE('{{DATA_END_DATE}}')
  -- Cleaning rule 1: pickup timestamp must exist
  AND pickup_datetime IS NOT NULL
  -- Cleaning rule 2: dropoff timestamp must exist for duration/order diagnostics
  AND dropoff_datetime IS NOT NULL
  -- Cleaning rule 3: temporal order must be valid
  AND dropoff_datetime > pickup_datetime
  -- Cleaning rule 4: target must exist and be strictly positive for this regression setup
  AND fare_amount IS NOT NULL
  AND fare_amount > 0
  -- Cleaning rule 5: distance proxy must be strictly positive
  AND trip_distance > 0
  -- Cleaning rule 6: passenger_count must stay in the business-supported range
  AND passenger_count BETWEEN 1 AND 6
  -- Cleaning rule 7: both zones must be present to support spatial features
  AND pickup_location_id IS NOT NULL
  AND dropoff_location_id IS NOT NULL;

-- This layer keeps diagnostic columns that are valid for quality checks but forbidden for features.
-- Cleaning rules intentionally stop at structural validity. Extreme but plausible fares/distances
-- are preserved for later review instead of being clipped here without business justification.
