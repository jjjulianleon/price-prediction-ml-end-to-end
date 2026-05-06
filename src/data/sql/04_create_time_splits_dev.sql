-- Splits temporales materializados como TABLA para el periodo oficial del proyecto.
-- Se usa TABLE (no VIEW) porque:
--   - La OBT tiene ~150M filas (2015-2025). Con VIEW, cada query de training y
--     evaluacion re-escanea la OBT completa antes de filtrar por fecha.
--   - ORDER BY RANDOM() LIMIT 5M sobre una VIEW de 100M filas es muy costoso.
--   - Con TABLE, Snowflake opera sobre un set pre-filtrado y aplica micro-partition
--     pruning mucho mas eficientemente.
-- Las fechas vienen de .env; no requieren cambios en el SQL.
-- Ventana oficial:
--   TRAIN:      DATA_START_DATE (2015-01-01) a TRAIN_END_DATE (2023-12-31)
--   VALIDATION: TRAIN_END_DATE+1 (2024-01-01) a VAL_END_DATE (2024-12-31)
--   TEST:       VAL_END_DATE+1   (2025-01-01) a DATA_END_DATE (2025-12-31)
-- Sin solapamiento temporal garantizado por las condiciones BETWEEN / >.
-- TEST se evalua una sola vez al final; no usar para seleccion de hiperparametros.

CREATE OR REPLACE TABLE {{DATABASE}}.{{ML_SCHEMA}}.TRAIN_SET_DEV AS
SELECT *
FROM {{DATABASE}}.{{ANALYTICS_SCHEMA}}.OBT_TRIPS_DEV
WHERE CAST(pickup_datetime AS DATE) BETWEEN TO_DATE('{{DATA_START_DATE}}') AND TO_DATE('{{TRAIN_END_DATE}}');

CREATE OR REPLACE TABLE {{DATABASE}}.{{ML_SCHEMA}}.VAL_SET_DEV AS
SELECT *
FROM {{DATABASE}}.{{ANALYTICS_SCHEMA}}.OBT_TRIPS_DEV
WHERE CAST(pickup_datetime AS DATE) > TO_DATE('{{TRAIN_END_DATE}}')
  AND CAST(pickup_datetime AS DATE) <= TO_DATE('{{VAL_END_DATE}}');

CREATE OR REPLACE TABLE {{DATABASE}}.{{ML_SCHEMA}}.TEST_SET_DEV AS
SELECT *
FROM {{DATABASE}}.{{ANALYTICS_SCHEMA}}.OBT_TRIPS_DEV
WHERE CAST(pickup_datetime AS DATE) > TO_DATE('{{VAL_END_DATE}}')
  AND CAST(pickup_datetime AS DATE) <= TO_DATE('{{DATA_END_DATE}}');
