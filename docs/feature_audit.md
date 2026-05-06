# Feature Audit

## Version del contrato

- `feature_contract_version`: `v3_multi_taxi_estimated_distance`
- Fecha de cierre: 2026-05-09

## Features aprobadas de entrada (RAW_FEATURE_COLUMNS en build_features.py)

| Feature | Tipo | Fuente en OBT | Descripcion |
|---|---|---|---|
| `trip_type` | categorica | directo | flota: `yellow` o `green` |
| `pickup_datetime` | timestamp | directo | usado para derivadas temporales, no entra raw al modelo |
| `passenger_count` | numerica | directo | 1..6, limpiado en STAGING |
| `estimated_distance` | numerica | directo (alias de trip_distance) | distancia estimada pre-viaje |
| `pickup_location_id` | categorica | directo | zona TLC de origen (1-265) |
| `dropoff_location_id` | categorica | directo | zona TLC de destino (1-265) |
| `vendor_id` | categorica | directo | ID del proveedor del sistema de despacho |
| `ratecode_id` | categorica | directo | codigo de tarifa TLC (1-6) |

## Derivadas deterministicas aprobadas

Calculadas en la OBT (SQL) y/o en `src/features/build_features.py::prepare_feature_frame`.

| Feature derivada | Calculo | Fuente |
|---|---|---|
| `pickup_hour` | `EXTRACT(HOUR FROM pickup_datetime)` | OBT SQL |
| `pickup_dayofweek` | `DAYOFWEEKISO(pickup_datetime) - 1` (0=Lun..6=Dom) | OBT SQL |
| `pickup_month` | `EXTRACT(MONTH FROM pickup_datetime)` | OBT SQL |
| `is_weekend` | `1 si pickup_dayofweek IN (5,6)` | OBT SQL |
| `is_rush_hour` | `1 si pickup_hour IN (7,8,9,16,17,18,19)` | OBT SQL |
| `is_night` | `1 si pickup_hour IN (22,23,0,1,2,3,4,5)` | OBT SQL |
| `log_estimated_distance` | `LN(1 + estimated_distance)` | OBT SQL |
| `route_id` | `CONCAT(pickup_location_id, '_', dropoff_location_id)` | OBT SQL |
| `same_zone` | `1 si pickup_location_id = dropoff_location_id` | OBT SQL |

## Columnas de control (presentes en OBT, no entran al modelo)

| Columna | Proposito |
|---|---|
| `pickup_datetime` | usado para derivadas temporales y splits; no entra como feature raw |

## Columnas diagnosticas (solo en STAGING, no en OBT ni splits ML)

| Columna | Razon de exclusion |
|---|---|
| `tpep_dropoff_datetime` / `lpep_dropoff_datetime` | post-viaje — leakage |
| `trip_duration_min` | derivado de dropoff — leakage |
| `speed_mph` | derivado de duracion real — leakage |
| `payment_type` | post-pago — leakage |
| `tip_amount` | post-pago — leakage |
| `tolls_amount` | post-viaje — leakage |
| `mta_tax` | cargo regulatorio post-viaje — leakage |
| `extra` | cargo variable post-viaje — leakage |
| `improvement_surcharge` | cargo regulatorio — leakage |
| `congestion_surcharge` | cargo por zona post-viaje — leakage |
| `airport_fee` | cargo por destino post-viaje — leakage |
| `total_amount` | suma de todos los cargos — leakage directo del target |

## Verificacion anti-leakage en codigo

La lista `LEAKAGE_COLUMNS` esta codificada en `src/features/build_features.py` y tiene cobertura de test en `tests/test_features.py::test_leakage_columns_are_rejected`. El notebook 03 también verifica explicitamente que ninguna columna de la lista prohibida entre al frame de modelado.

## Pipeline de preprocesamiento

Definido en `src/features/build_features.py::get_feature_pipeline()`:

- **Numericas** (`NUMERIC_FEATURES`): `SimpleImputer(median)` + `StandardScaler`
- **Categoricas** (`CATEGORICAL_FEATURES`): `SimpleImputer(constant='missing')` + `OneHotEncoder(handle_unknown='ignore')`

El espacio transformado post-encoding supera las 2700 columnas principalmente por el one-hot de location IDs y route_id.

## Nota sobre `trip_distance`

El alias `trip_distance` en la OBT fue removido en la version v3 del contrato. Solo existe `estimated_distance` como nombre oficial del proxy de distancia historica. El codigo en `build_features.py` conserva un fallback de compatibilidad para frames de RAW que aun usen el nombre original.

## Extension futura (no activa en v3)

Cuando se active `ENABLE_ZONE_LOOKUP=true` con el lookup oficial de zonas TLC:

- `pickup_borough`, `dropoff_borough`
- `pickup_is_airport`, `dropoff_is_airport`

Estas variables no forman parte del contrato v3 y requeririan una nueva version del preprocesador.
