# Problem Definition

## Contexto

El proyecto utiliza `NYC TLC Taxi Trip Record Data` para predecir la tarifa base de un viaje individual (`fare_amount`) como problema de regresion supervisada.

La arquitectura sigue un esquema **Snowflake-first**: toda la ingesta, limpieza estructural, construccion de la OBT y materializacion de los splits temporales ocurre en Snowflake mediante SQL. Python y los notebooks solo consumen muestras representativas; nunca descargan la base completa a pandas.

## Fuente de datos

- Fuente principal: `NYC TLC Trip Record Data` (parquet mensual, CDN oficial)
- URL base: `https://d37ci6vzurychx.cloudfront.net/trip-data`
- Servicios soportados: `Yellow Taxi` y `Green Taxi` combinados (`TRIP_TYPE=yellow,green`)
- Unidad de analisis: un viaje individual
- Volumen estimado: ~150M filas en la OBT para el periodo 2015-2025

## Objetivo del modelo

Predecir `fare_amount` en un momento **antes de iniciar el viaje**.

## Variable objetivo

- Target oficial: `fare_amount`
- Metricas de evaluacion: `RMSE` (principal), `MAE`, `R2`

## Por que `fare_amount` y no `total_amount`

`total_amount` incorpora cargos conocidos solo al cierre o pago del viaje: propinas, peajes, recargos de congestion, airport fee y otros. Usarlo como target o como feature introduce leakage y hace el problema no reproducible en serving.

`fare_amount` representa la tarifa base asociada al servicio y permite modelar un escenario de prediccion pre-viaje defendible y verificable.

## Momento de prediccion

El modelo simula una inferencia hecha antes de iniciar el viaje. Solo pueden usarse variables conocidas o estimables en ese instante.

## Variables conocidas antes del viaje

- `pickup_datetime`
- `trip_type`
- `pickup_location_id`
- `dropoff_location_id`
- `passenger_count`
- `vendor_id`
- `ratecode_id`
- `estimated_distance`

## Definicion de `estimated_distance`

En la data historica original existe `trip_distance`. El contrato oficial de modelado la publica como `estimated_distance` en `STAGING`, `ANALYTICS` y `ML`.

- en entrenamiento historico, `estimated_distance` es un **proxy** construido desde `trip_distance`
- en serving, el usuario o sistema externo proporciona una distancia estimada antes del viaje
- el alias `trip_distance` fue removido de la OBT en la version final del contrato (v3) para evitar confusion entre la variable historica y el input de serving

## Variables excluidas por leakage

No deben usarse como features:

- `total_amount`, `tip_amount`, `tolls_amount`, `mta_tax`, `extra`
- `improvement_surcharge`, `congestion_surcharge`, `airport_fee`
- `payment_type`
- `tpep_dropoff_datetime` / `lpep_dropoff_datetime`
- `trip_duration_min`, `speed_mph`

Estas columnas existen en `STAGING` para diagnostico y auditoria de calidad, pero **no aparecen en la OBT ni en los splits ML**.

## Arquitectura de capas Snowflake

```
RAW
  YELLOW_TRIPS_DEV   <- parquet TLC yellow por mes, append-only, nombres originales
  GREEN_TRIPS_DEV    <- parquet TLC green por mes, append-only, nombres originales

STAGING
  TRIPS_STAGE_DEV    <- TABLE  yellow + green unificados bajo contrato canonico
                        aplica 8 reglas de limpieza estructural validadas en notebooks
                        conserva columnas diagnosticas (payment_type, tip_amount, etc.)
                        para auditoria; estas columnas NO llegan a la OBT

ANALYTICS
  OBT_TRIPS_DEV      <- TABLE  contrato anti-leakage estricto
                        incluye solo columnas aptas para modelado pre-viaje
                        agrega derivadas deterministicas (pickup_hour, route_id, etc.)
                        es la unica fuente valida para splits y entrenamiento

ML
  TRAIN_SET_DEV      <- TABLE  2015-01-01 a 2023-12-31  (~100M filas)
  VAL_SET_DEV        <- TABLE  2024-01-01 a 2024-12-31  (~15M filas)
  TEST_SET_DEV       <- TABLE  2025-01-01 a 2025-12-31  (~15M filas)
```

### Por que los splits ML son tablas materializadas y no views

Con `VIEW` sobre la OBT (~150M filas 2015-2025), cada query de training o evaluacion re-escanea la OBT completa y aplica el filtro de fecha en tiempo de ejecucion. El `ORDER BY RANDOM() LIMIT 5_000_000` del entrenamiento sobre el train view (~100M filas 2015-2023) requiere generar y ordenar numeros aleatorios para todas esas filas antes de tomar los 5M de muestra.

Con `TABLE` pre-materializada, Snowflake opera sobre el subconjunto ya filtrado, aplica micro-partition pruning eficientemente y reduce el costo de computo por query. El training y cada batch de evaluacion son significativamente mas rapidos, especialmente con el volumen de 2015-2025.

La contrapartida es mayor uso de storage (los 3 splits suman similar volumen a la OBT). Aceptable dado que el ahorro en compute por multiples queries supera el costo de almacenamiento.

## Reglas de limpieza en STAGING

Todas las reglas estan validadas con evidencia muestral en `notebooks/02_data_cleaning.ipynb`.

| Regla | Condicion SQL | Justificacion |
|---|---|---|
| pickup no nulo | `pickup_datetime IS NOT NULL` | timestamp de inicio requerido para splits temporales |
| dropoff no nulo | `dropoff_datetime IS NOT NULL` | necesario para validacion de orden temporal |
| ventana temporal | `CAST(pickup_datetime AS DATE) BETWEEN data_start_date AND data_end_date` | parametrizado desde `.env` |
| orden logico | `dropoff_datetime > pickup_datetime` | viaje con duracion positiva |
| distancia en rango | `trip_distance BETWEEN 0.1 AND 150` | 0.1 mi minimo GPS valido; 150 mi techo fisico NYC-area |
| pasajeros en rango | `passenger_count BETWEEN 1 AND 6` | rango soportado por regulacion TLC |
| tarifa en rango | `fare_amount BETWEEN 2.50 AND 300` | $2.50 minimo TLC; $300 elimina errores de entrada de datos |
| zona origen | `pulocationid IS NOT NULL` | requerida para features espaciales |
| zona destino | `dolocationid IS NOT NULL` | requerida para features espaciales |
| ratecode valido | `ratecode_id BETWEEN 1 AND 6` | catalogo oficial TLC (1=Standard, 2=JFK, 3=Newark, 4=Nassau/Westchester, 5=Negotiated, 6=Group) |

## Splits temporales

| Split | Periodo | Filas estimadas | Uso |
|---|---|---|---|
| `TRAIN_SET_DEV` | 2015-01-01 a 2023-12-31 | ~100M | entrenamiento del modelo productivo |
| `VAL_SET_DEV` | 2024-01-01 a 2024-12-31 | ~15M | seleccion de modelo e hiperparametros |
| `TEST_SET_DEV` | 2025-01-01 a 2025-12-31 | ~15M | evaluacion final, **una sola vez** |

Las fechas son parametrizadas desde `.env`; no requieren cambios en el SQL del pipeline.

## Estrategia de entrenamiento out-of-core

XGBoost no tiene `partial_fit` nativo. La estrategia elegida es **muestra masiva estratificada**:

- `fetch_sample` ejecuta `SELECT * FROM TRAIN_SET_DEV ORDER BY RANDOM() LIMIT 5_000_000`
- Snowflake aplica aleatoriedad global sobre el set pre-materializado
- 5M filas aleatorias garantizan cobertura de todos los anos, meses y flotas del periodo 2015-2023
- La evaluacion en `VAL_SET_DEV` y `TEST_SET_DEV` se hace por lotes (`BATCH_SIZE=200000`) sin cargar todo en memoria

## Modelo productivo

- Modelo seleccionado: `XGBoost` (`tree_method=hist`, `n_estimators=400`)
- Seleccion basada en `val_rmse` del benchmark formal en `notebooks/04_model_experimentation.ipynb`
- Entrenado sobre muestra masiva estratificada (~5M filas) desde `TRAIN_SET_DEV`
- Usa `sample_weight` para corregir desbalance Yellow/Green
- Artefacto final: `data/models/nyc_taxi_fare_production.joblib`
- Serving via `FastAPI` + `Streamlit`

## Hipotesis del modelo

- la distancia estimada es el predictor mas fuerte del target
- el patron horario, semanal y mensual influye en la tarifa
- el par origen-destino aporta estructura espacial relevante
- `trip_type` captura diferencias estructurales entre Yellow y Green que no captura solo la distancia
- un booster (XGBoost) supera al baseline lineal y al Random Forest en este problema tabular

## Referencias

- NYC TLC Trip Record Data: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Enunciado oficial: `ENUNCIADO.md`
- Decisiones de diseno: `docs/decisions_log.md`
- Contrato de datos: `docs/data_contract.md`
- Auditoria de features: `docs/feature_audit.md`
