-- Build the final OBT used by ML.
-- This layer starts from structurally valid STAGING rows and keeps only prediction-safe fields.
-- Temporal derivations are aligned with pandas notebook logic:
-- pickup_dayofweek uses 0=Monday ... 6=Sunday.
CREATE OR REPLACE TABLE {{DATABASE}}.{{ANALYTICS_SCHEMA}}.OBT_TRIPS_DEV AS
WITH base AS (
    SELECT
        -- Preserve fleet so the model can learn Yellow vs Green differences
        trip_type,
        -- Raw-safe input timestamp
        pickup_datetime,
        -- Safe numeric inputs
        passenger_count,
        estimated_distance,
        -- Safe spatial and categorical inputs
        pickup_location_id,
        dropoff_location_id,
        vendor_id,
        ratecode_id,
        -- Regression target
        fare_amount,
        -- Deterministic temporal derivations from pickup only
        EXTRACT(HOUR FROM pickup_datetime) AS pickup_hour,
        DAYOFWEEKISO(pickup_datetime) - 1 AS pickup_dayofweek,
        EXTRACT(MONTH FROM pickup_datetime) AS pickup_month
    FROM {{DATABASE}}.{{STAGING_SCHEMA}}.TRIPS_STAGE_DEV
    -- OBT rule 1: keep trips with valid positive duration for diagnostic consistency
    WHERE trip_duration_min IS NOT NULL
      AND trip_duration_min > 0
)
SELECT
    trip_type,
    pickup_datetime,
    -- Time-derived features
    pickup_hour,
    pickup_dayofweek,
    pickup_month,
    IFF(pickup_dayofweek IN (5, 6), 1, 0) AS is_weekend,
    IFF(pickup_hour IN (7, 8, 9, 16, 17, 18, 19), 1, 0) AS is_rush_hour,
    IFF(pickup_hour IN (22, 23, 0, 1, 2, 3, 4, 5), 1, 0) AS is_night,
    -- Direct trip descriptors
    passenger_count,
    estimated_distance,
    -- Compatibility alias kept temporarily for legacy notebook outputs
    estimated_distance AS trip_distance,
    -- Deterministic numeric transform
    LN(1 + estimated_distance) AS log_estimated_distance,
    -- Spatial/categorical features
    pickup_location_id,
    dropoff_location_id,
    vendor_id,
    ratecode_id,
    -- Route-level derivations
    CONCAT(pickup_location_id, '_', dropoff_location_id) AS route_id,
    IFF(pickup_location_id = dropoff_location_id, 1, 0) AS same_zone,
    -- Target
    fare_amount
FROM base;

-- Leakage guardrails for this dev OBT:
-- Excluded from features: payment_type, tip_amount, tolls_amount, mta_tax,
-- improvement_surcharge, congestion_surcharge, airport_fee, total_amount,
-- trip_duration-derived columns, and any value known only after trip completion.
-- `trip_distance` is retained here only as a temporary compatibility alias for notebooks and legacy analysis.
