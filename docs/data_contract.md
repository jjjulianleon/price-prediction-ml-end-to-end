# Data Contract

## Version del contrato

- `feature_contract_version`: `v3_multi_taxi_estimated_distance`
- Fecha de cierre: 2026-05-09

## Estructura de columnas por capa

La OBT (`ANALYTICS.OBT_TRIPS_DEV`) y los splits ML contienen **exclusivamente** columnas aptas para modelado. Las columnas diagnosticas existen solo en `STAGING` para auditoria de calidad.

### Columnas de la OBT y splits ML

| Columna | Tipo | Rol | Feature | Disponible pre-viaje | Regla de calidad |
|---|---|---|:---:|:---:|---|
| `trip_type` | string | flota identidad | si | si | valor en {yellow, green} |
| `pickup_datetime` | timestamp | tiempo base | si | si | no nulo, dentro del periodo |
| `pickup_hour` | int | derivada temporal | si | si | 0..23 |
| `pickup_dayofweek` | int | derivada temporal | si | si | 0=Lun..6=Dom |
| `pickup_month` | int | derivada temporal | si | si | 1..12 |
| `is_weekend` | int | derivada temporal | si | si | 0 o 1 |
| `is_rush_hour` | int | derivada temporal | si | si | 0 o 1 |
| `is_night` | int | derivada temporal | si | si | 0 o 1 |
| `passenger_count` | int | demanda declarada | si | si | 1..6 (limpieza en STAGING) |
| `estimated_distance` | float | distancia estimada | si | si | 0.1..150 (limpieza en STAGING) |
| `log_estimated_distance` | float | derivada numerica | si | si | calculado con LN(1 + estimated_distance) |
| `pickup_location_id` | int | zona origen | si | si | no nulo |
| `dropoff_location_id` | int | zona destino | si | si | no nulo |
| `vendor_id` | int | operador TLC | si | si | no nulo cuando aplique |
| `ratecode_id` | int | codigo de tarifa | si | si (usualmente) | 1..6 (limpieza en STAGING) |
| `route_id` | string | par origen-destino | si | si | concatenacion de location IDs |
| `same_zone` | int | flag ruta | si | si | 0 o 1 |
| `fare_amount` | float | **TARGET** | no | no aplica | 2.50..300 (limpieza en STAGING) |

### Columnas presentes en STAGING pero excluidas de la OBT

Estas columnas existen en `STAGING.TRIPS_STAGE_DEV` para auditoria de calidad y EDA, pero **no aparecen en la OBT ni en los splits ML**.

| Columna | Razon de exclusion |
|---|---|
| `payment_type` | conocido solo al pago del viaje — leakage |
| `tip_amount` | conocido solo al pago — leakage |
| `tolls_amount` | conocido solo al finalizar el viaje — leakage |
| `mta_tax` | cargo post-viaje — leakage |
| `extra` | cargo variable post-viaje — leakage |
| `improvement_surcharge` | cargo regulatorio post-viaje — leakage |
| `congestion_surcharge` | cargo post-viaje segun zona — leakage |
| `airport_fee` | cargo post-viaje segun destino — leakage |
| `total_amount` | suma de fare + todos los cargos post-pago — leakage directo |
| `dropoff_datetime` | conocido solo al finalizar — leakage |
| `trip_duration_min` | derivado de dropoff — leakage |
| `speed_mph` | derivado de duracion real — leakage |

## Definicion de `estimated_distance`

`trip_distance` (nombre en RAW) se publica como `estimated_distance` en STAGING, OBT y ML:

- **entrenamiento**: proxy historico construido desde `trip_distance` observado
- **serving**: el usuario o sistema externo provee la distancia estimada antes del viaje
- el alias de compatibilidad `trip_distance` fue **removido de la OBT** en la version v3 del contrato para evitar confusion en el pipeline de serving

## Reglas de calidad en STAGING

Aplicadas por pushdown SQL en `02_create_staging_trips_dev.sql`. Validadas con evidencia muestral en `notebooks/02_data_cleaning.ipynb`.

| # | Regla SQL | Columas afectadas | Justificacion |
|---|---|---|---|
| 1 | `pickup_datetime IS NOT NULL` | pickup_datetime | timestamp de inicio requerido |
| 2 | `dropoff_datetime IS NOT NULL` | dropoff_datetime | necesario para validacion de orden |
| 3 | `CAST(pickup_datetime AS DATE) BETWEEN start AND end` | pickup_datetime | ventana del proyecto |
| 4 | `dropoff_datetime > pickup_datetime` | ambas | viaje con duracion positiva |
| 5 | `trip_distance BETWEEN 0.1 AND 150` | trip_distance | 0.1 mi minimo GPS; 150 mi techo fisico |
| 6 | `passenger_count BETWEEN 1 AND 6` | passenger_count | catalogo TLC |
| 7 | `fare_amount BETWEEN 2.50 AND 300` | fare_amount | $2.50 minimo TLC; $300 elimina errores |
| 8 | `pulocationid IS NOT NULL` | pickup_location_id | zona espacial requerida |
| 9 | `dolocationid IS NOT NULL` | dropoff_location_id | zona espacial requerida |
| 10 | `ratecode_id BETWEEN 1 AND 6` | ratecode_id | catalogo oficial TLC (Standard=1..Group=6) |

## Lista prohibida de features

Ninguna de estas columnas puede entrar al pipeline de entrenamiento ni de inferencia:

```
total_amount, tip_amount, tolls_amount, mta_tax, extra,
improvement_surcharge, congestion_surcharge, airport_fee,
payment_type, tpep_dropoff_datetime, lpep_dropoff_datetime,
trip_duration_min, speed_mph
```

La lista esta codificada en `src/features/build_features.py::LEAKAGE_COLUMNS` y tiene cobertura de test en `tests/test_features.py::test_leakage_columns_are_rejected`.

## Compatibilidad multi-flota

- Yellow RAW: `tpep_pickup_datetime`, `tpep_dropoff_datetime`, `airport_fee`
- Green RAW: `lpep_pickup_datetime`, `lpep_dropoff_datetime`, sin `airport_fee` (NULL en STAGING)
- Ambas flotas se unifican en STAGING bajo el mismo contrato canonico
- `trip_type` = `yellow` o `green` es feature de modelado, no solo metadato operativo

## Extension opcional (no activa en contrato base)

Cuando se active `ENABLE_ZONE_LOOKUP=true` y exista lookup oficial cargado, se podrian evaluar:

- `pickup_borough`, `dropoff_borough`
- `pickup_is_airport`, `dropoff_is_airport`

Estas variables no forman parte del contrato v3 actual.
