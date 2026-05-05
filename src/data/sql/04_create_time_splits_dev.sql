-- Temporal splits for the base phase using date boundaries from the environment.
-- Current recommended base window: 6 months with monthly split boundaries.
-- Example base window:
--   DATA_START_DATE = 2025-01-01
--   TRAIN_END_DATE = 2025-04-30
--   VAL_END_DATE   = 2025-05-31
--   DATA_END_DATE  = 2025-06-30

CREATE OR REPLACE VIEW {{DATABASE}}.{{ML_SCHEMA}}.TRAIN_SET_DEV AS
SELECT *
FROM {{DATABASE}}.{{ANALYTICS_SCHEMA}}.OBT_TRIPS_DEV
WHERE CAST(pickup_datetime AS DATE) BETWEEN TO_DATE('{{DATA_START_DATE}}') AND TO_DATE('{{TRAIN_END_DATE}}');

CREATE OR REPLACE VIEW {{DATABASE}}.{{ML_SCHEMA}}.VAL_SET_DEV AS
SELECT *
FROM {{DATABASE}}.{{ANALYTICS_SCHEMA}}.OBT_TRIPS_DEV
WHERE CAST(pickup_datetime AS DATE) > TO_DATE('{{TRAIN_END_DATE}}')
  AND CAST(pickup_datetime AS DATE) <= TO_DATE('{{VAL_END_DATE}}');

CREATE OR REPLACE VIEW {{DATABASE}}.{{ML_SCHEMA}}.TEST_SET_DEV AS
SELECT *
FROM {{DATABASE}}.{{ANALYTICS_SCHEMA}}.OBT_TRIPS_DEV
WHERE CAST(pickup_datetime AS DATE) > TO_DATE('{{VAL_END_DATE}}')
  AND CAST(pickup_datetime AS DATE) <= TO_DATE('{{DATA_END_DATE}}');

-- How to scale this later without changing the modeling code:
--   TRAIN: pickup year 2015-2023
--   VAL:   pickup year 2024
--   TEST:  pickup year 2025
