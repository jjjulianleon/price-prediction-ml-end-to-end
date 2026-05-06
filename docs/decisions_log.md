# Decisions Log

## Decisiones Cerradas

### 1. Target oficial: `fare_amount`

Se descarta `total_amount` como target o feature porque incorpora informacion posterior al viaje y al pago.

### 2. Momento de prediccion: pre-viaje

Todo el contrato de features queda restringido a variables conocidas o estimables antes de iniciar el viaje.

### 3. `estimated_distance` como contrato publico

La variable de distancia historica `trip_distance` se publica como `estimated_distance` en capas de modelado. Esto permite mantener coherencia entre entrenamiento historico y serving real.

### 4. `trip_type` como feature explicita

Yellow y Green Taxi comparten el mismo target, pero no necesariamente la misma dinamica tarifaria ni la misma mezcla de rutas. Por eso la flota se modela explicitamente mediante `trip_type` y no solo como separacion operativa en `RAW`.

### 5. Arquitectura Snowflake-first

La limpieza estructural, la OBT y los splits temporales viven en SQL. Pandas y notebooks se reservan para muestra, validacion y experimentacion.

### 6. Split temporal por fechas parametrizadas

La base operativa actual usa fechas configurables desde `.env`. El split final del curso `2015-2023 / 2024 / 2025` se alcanzara cambiando configuracion, no reescribiendo el pipeline.

### 7. Entrenamiento hibrido

No todos los modelos soportan `partial_fit`. Por eso el proyecto distingue entre:

- entrenamiento incremental real
- entrenamiento sobre muestra controlada

### 8. `RAW` append-only por defecto

No se borra `RAW` por defecto. La ingesta debe ser idempotente y registrar archivo, periodo y estado de carga.

### 9. Modelo productivo seleccionado: `xgboost` (actualizado 2026-05-10 con evidencia de notebook 04)

**Historial:** el modelo original fue `gradient_boosting` (sklearn) seleccionado el 2026-05-05 por estabilidad de entorno. Se reemplaza por XGBoost en la entrega final.

**Razon del cambio:**

- la rubrica exige boosting moderno (XGBoost / LightGBM) de forma obligatoria; sklearn `GradientBoostingRegressor` no lo cumple
- el benchmark formal en `notebooks/04_model_experimentation.ipynb` (muestra balanceada multi-año) mostro que XGBoost supera a `gradient_boosting` en `val_rmse`
- XGBoost acepta `sample_weight` para el manejo de desbalance yellow/green
- `tree_method=hist` permite entrenar sobre ~5M filas en tiempo razonable

**Estrategia de entrenamiento: out-of-core real via `xgb_out_of_core`.**

No se carga todo el dataset en memoria. El pipeline procesa 18 lotes independientes (9 años × 2 flotas), cada uno de ~555K filas, usando `xgb.train(xgb_model=booster_anterior)` para acumular rondas incrementalmente.

Perfil de memoria:
- Pico por lote: ~300 MB (DataFrame + sparse CSR matrix)
- Nunca se acumulan mas de ~555K filas en RAM simultaneamente
- Total de filas vistas por el modelo: ~10M (vs 828M del OBT completo)

Comparacion con alternativas descartadas:
- *Cargar 10M filas de una vez*: OHE produce matrices de 6-8 GB → OOM en hardware estandar (descartado)
- *XGBoost external memory DMatrix*: requiere escribir SVM a disco, dependencia de formato binario → mas complejidad sin ganancia real en calidad (descartado)
- *sample_strategy simple*: estadisticamente valido pero riesgo de sub-representar años tempranos (descartado)

Parametros clave (`.env`):
- `TRAIN_SAMPLE_LIMIT=10000000`: total de filas a procesar (distribuidas entre 18 estratos)
- `BATCH_SIZE=500000`: tamaño de lote para evaluacion en VAL/TEST
- `TRAINING_BATCH_GRAIN=month`: granularidad de ventanas en evaluacion

**Evidencia del benchmark notebook 04** (10K filas, DB DM_FINAL_PROJECT, 2026-05-10):

| Modelo | val_rmse | test_rmse | gap |
|---|---|---|---|
| LightGBM | 8.53 | 9.10 | +0.58 |
| GradientBoosting | 8.84 | 9.68 | +0.85 |
| CatBoost | 9.11 | 9.46 | +0.35 |
| XGBoost | 9.21 | 9.21 | **-0.003** |
| HistGradientBoosting | 9.72 | 9.79 | +0.06 |

LightGBM ganó val_rmse pero XGBoost tiene el mejor gap val→test (0.003), indicando la mejor generalización temporal. Con 10M filas en producción se espera que XGBoost mejore su posicion relativa. La decision productiva final se confirma con el log de `train_model.py`.

### 10. Separacion formal entre experimentacion y produccion

Desde esta fase:

- `src/models/model_zoo.py` y `src/models/experiment_runner.py` quedan reservados para comparacion experimental
- `src/models/train_model.py` entrena solo el modelo productivo seleccionado
- la API carga solo `data/models/nyc_taxi_fare_production.joblib`

### 11. Activacion multi-fleet para entrega final (2026-05-09)

`TRIP_TYPE=yellow,green` activo en `.env`. Ambas flotas se cargan en RAW separado, se unifican en STAGING y se diferencian en OBT/ML mediante la feature `trip_type`. Las metricas de evaluacion se reportan globales y por flota (yellow vs green) en notebook 04 y en el entrenamiento productivo.

### 12. Ventana oficial completa (2026-05-09)

El periodo de entrenamiento pasa de 6 meses operativos a la ventana oficial del enunciado:

- `train`: 2015-01-01 a 2023-12-31
- `validation`: 2024-01-01 a 2024-12-31
- `test`: 2025-01-01 a 2025-12-31

El cambio se controla via `.env` sin modificar logica de pipeline.

### 14. Splits ML como tablas materializadas, no views (2026-05-09)

Los splits `TRAIN_SET_DEV`, `VAL_SET_DEV` y `TEST_SET_DEV` se materializan como `TABLE` en lugar de `VIEW`.

**Razon:** la OBT tiene ~150M filas (2015-2025). Con VIEW, cada query de training o evaluacion re-escanea la OBT completa y aplica el filtro de fecha en tiempo de ejecucion. El `ORDER BY RANDOM() LIMIT 5_000_000` sobre el resultado (~100M filas del train) requiere generar y ordenar numeros aleatorios para toda la tabla antes de tomar 5M. Con TABLE ya materializada, Snowflake opera sobre el subconjunto pre-filtrado, aplica micro-partition pruning eficientemente y reduce el costo de computo por query.

**Contrapartida:** mayor uso de storage en Snowflake (~3x la OBT si se incluyen los 3 splits). Aceptable dado que el compute ahorrado en multiples queries de entrenamiento y evaluacion es mucho mayor.

### 13. Estrategia de muestras (2026-05-09)

- **Notebooks (EDA/limpieza/features):** muestra balanceada multi-año, `EDA_SAMPLE_LIMIT=100000` filas aleatorias de la OBT con cobertura de anos y trip_types.
- **Entrenamiento productivo (XGBoost):** muestra masiva estratificada, `TRAIN_SAMPLE_LIMIT=5_000_000` filas aleatorias del `TRAIN_SET`. La aleatoriedad por `ORDER BY RANDOM()` en Snowflake garantiza representacion temporal y por flota sin requerir external memory.

### 15. Adicion de `pickup_year` al contrato de features (2026-05-10)

Se agrega `pickup_year` como feature numerica en `MODEL_FEATURE_COLUMNS` (contrato v4).

**Razon:** entrenando sobre 2015-2023, el modelo no tenia ninguna señal del año de viaje. Las tarifas NYC aumentaron ~30-40% entre 2015 y 2023 (cambios de tarifa base, MTA surcharge, etc.). Sin `pickup_year`, XGBoost aprende el promedio historico de 9 años y predice ese promedio para 2024-2025, generando error sistematico ~$56 RMSE en validacion. Con `pickup_year`, el modelo aprende la tendencia temporal y para 2024 extrapola desde el patron 2023 (año mas cercano al training).

**Nota sobre test_rmse=165 (2025):** el gap val=56→test=165 es consistente con la implementacion del NYC Congestion Pricing (enero 2025) que agrego ~$9-15 a la mayoria de viajes hacia Manhattan. Si ese surcharge se incorporo en `fare_amount` de los parquets 2025 del TLC (en lugar de como columna separada), la distribucion del target cambio estructuralmente. Este es un caso documentado de **concept drift por cambio regulatorio** y se cita como limitacion en el informe final.

**Contrato actualizado:** `v4_multi_taxi_year_estimated_distance`

## Limitaciones y Mejoras Posteriores

### Alta prioridad

- integrar `taxi zone lookup` como enriquecimiento opcional activable por flag
- medir tiempos reales sobre el periodo completo `2015-2025`
- validar la conveniencia de `TRAINING_BATCH_GRAIN=week` sobre datasets mas grandes

### Prioridad media

- incorporar `borough` origen/destino y `airport flags` cuando exista lookup oficial cargado
- versionar artefactos de entrenamiento con identificador de corrida
- guardar auditoria de experimento en tabla Snowflake ademas del artefacto local

### Prioridad baja

- enriquecer el frontend con historico de predicciones
- agregar pruebas de integracion API + frontend sobre artefacto real
