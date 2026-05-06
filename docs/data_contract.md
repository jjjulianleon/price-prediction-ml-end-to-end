# Data Contract

## Contrato Oficial de Modelado

El contrato oficial de modelado aplica a las capas `STAGING`, `ANALYTICS` y `ML`. La capa `RAW` conserva nombres de origen para trazabilidad.

| Columna | Tipo esperado | Rol | Feature | Disponible pre-viaje | Riesgo de leakage | Regla de calidad | Accion ante invalido |
| :-- | :-- | :-- | :--: | :--: | :-- | :-- | :-- |
| `pickup_datetime` | timestamp | tiempo base | si | si | bajo | no nulo, dentro del periodo | descartar fila |
| `trip_type` | string | flota (`yellow` o `green`) | si | si | bajo | valor en catalogo permitido | imputar a `yellow` en serving legacy o descartar si invalido en training |
| `pickup_location_id` | int | zona origen | si | si | bajo | no nulo | descartar fila |
| `dropoff_location_id` | int | zona destino | si | si | bajo | no nulo | descartar fila |
| `passenger_count` | int | demanda declarada | si | si | bajo | rango razonable `1..6` | descartar fila |
| `vendor_id` | int | operador | si | si | bajo | no nulo cuando aplique | imputar o descartar segun capa |
| `ratecode_id` | int | tarifa declarada | si | usualmente si | medio | no nulo preferible | imputar o mantener nullable controlado |
| `estimated_distance` | float | distancia estimada pre-viaje | si | si | medio | `> 0` | descartar fila |
| `fare_amount` | float | target | no | no aplica | no aplica | no nulo, `>= 0`; preferible `> 0` para entrenamiento | descartar fila invalida |
| `pickup_hour` | int | derivada temporal | si | si | bajo | `0..23` | recalcular |
| `pickup_dayofweek` | int | derivada temporal | si | si | bajo | `0..6` o consistente con pipeline | recalcular |
| `pickup_month` | int | derivada temporal | si | si | bajo | `1..12` | recalcular |
| `is_weekend` | int | derivada temporal | si | si | bajo | `0/1` | recalcular |
| `is_rush_hour` | int | derivada temporal | si | si | bajo | `0/1` | recalcular |
| `is_night` | int | derivada temporal | si | si | bajo | `0/1` | recalcular |
| `log_estimated_distance` | float | derivada numerica | si | si | bajo | consistente con `log1p` | recalcular |
| `route_id` | string | ruta origen-destino | si | si | bajo | no vacio | recalcular |
| `same_zone` | int | flag espacial | si | si | bajo | `0/1` | recalcular |
| `tpep_dropoff_datetime` | timestamp | diagnostico | no | no | alto | no usar en modelado | solo diagnostico |
| `trip_duration_min` | float | diagnostico | no | no | alto | derivada valida | solo diagnostico |
| `speed_mph` | float | diagnostico | no | no | alto | derivada valida | solo diagnostico |
| `payment_type` | int | diagnostico | no | no | alto | no usar en modelado | solo diagnostico |
| `tip_amount` | float | diagnostico | no | no | alto | no usar en modelado | solo diagnostico |
| `tolls_amount` | float | diagnostico | no | no | alto | no usar en modelado | solo diagnostico |
| `mta_tax` | float | diagnostico | no | no | alto | no usar en modelado | solo diagnostico |
| `extra` | float | diagnostico | no | no | alto | no usar en modelado | solo diagnostico |
| `improvement_surcharge` | float | diagnostico | no | no | alto | no usar en modelado | solo diagnostico |
| `congestion_surcharge` | float | diagnostico | no | no | alto | no usar en modelado | solo diagnostico |
| `airport_fee` | float | diagnostico | no | no | alto | no usar en modelado | solo diagnostico |
| `total_amount` | float | diagnostico | no | no | alto | no usar en modelado | solo diagnostico |

## Lista Prohibida de Features

No pueden entrar al modelo final:

- `total_amount`
- `tip_amount`
- `tolls_amount`
- `mta_tax`
- `extra`
- `improvement_surcharge`
- `congestion_surcharge`
- `airport_fee`
- `payment_type`
- `tpep_dropoff_datetime`
- `trip_duration_min`
- `speed_mph`

## Compatibilidad Legacy

- `trip_distance` se conserva en `RAW`
- `ANALYTICS` puede exponer temporalmente un alias de compatibilidad para notebooks legacy
- el contrato oficial del pipeline y del serving usa `estimated_distance`
- el contrato oficial incorpora `trip_type` como feature categorica para capturar diferencias estructurales entre flotas
