# Feature Audit

## Version de contrato

- `feature_contract_version`: `v3_multi_taxi_estimated_distance`

## Features aprobadas de entrada

- `pickup_datetime`
- `trip_type`
- `passenger_count`
- `estimated_distance`
- `pickup_location_id`
- `dropoff_location_id`
- `vendor_id`
- `ratecode_id`

## Derivadas deterministicas aprobadas

- `pickup_hour`
- `pickup_dayofweek`
- `pickup_month`
- `is_weekend`
- `is_rush_hour`
- `is_night`
- `log_estimated_distance`
- `route_id`
- `same_zone`

## Variables diagnosticas

Estas columnas pueden usarse en calidad o EDA, pero no como features:

- `tpep_dropoff_datetime`
- `trip_duration_min`
- `speed_mph`
- `payment_type`
- `tip_amount`
- `tolls_amount`
- `mta_tax`
- `extra`
- `improvement_surcharge`
- `congestion_surcharge`
- `airport_fee`
- `total_amount`

## Variables prohibidas por leakage

- `payment_type`
- `tip_amount`
- `tolls_amount`
- `mta_tax`
- `extra`
- `improvement_surcharge`
- `congestion_surcharge`
- `airport_fee`
- `total_amount`
- `tpep_dropoff_datetime`
- `trip_duration_min`
- `speed_mph`

## Mejora opcional prioritaria

Cuando se active `ENABLE_ZONE_LOOKUP=true` y exista lookup cargado, se podran evaluar como extension:

- `pickup_borough`
- `dropoff_borough`
- `pickup_is_airport`
- `dropoff_is_airport`

Esas variables no forman parte del contrato base actual.

## Nota Multi-Fleet

- `trip_type` forma parte del set oficial y entra al preprocesamiento como variable categorica.
- si `TRIP_TYPE=yellow,green`, el EDA y la experimentacion deben reportar comparativas por flota.
