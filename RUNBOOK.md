# Runbook Operativo

Guia paso a paso para ejecutar, validar y presentar el proyecto final de prediccion de tarifas NYC Yellow Taxi.

## Objetivo de Esta Guia

Este documento sirve para tres cosas:

1. levantar el pipeline end-to-end sin improvisar comandos
2. validar que la base de 6 meses funciona correctamente
3. dejar una secuencia reproducible para que cualquier integrante del grupo pueda continuar

## Configuracion Recomendada

La base actual esta preparada para una ventana de `6 meses`:

- `DATA_START_DATE=2025-01-01`
- `DATA_END_DATE=2025-06-30`
- `TRAIN_END_DATE=2025-04-30`
- `VAL_END_DATE=2025-05-31`
- `TRAINING_BATCH_GRAIN=month`

Interpretacion del split:

- `train`: enero a abril
- `validation`: mayo
- `test`: junio

Se usa `month` como granularidad por defecto porque es la opcion mas estable para esta primera base operativa. Solo cambia a `week` si despues de medir tiempos o memoria descubres que el lote mensual es demasiado pesado.

## Paso 0. Preparar Entorno

Desde la raiz del proyecto:

```bash
cd price-prediction-ml-end-to-end
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Completa `.env` con las credenciales reales de Snowflake antes de seguir.

Variables criticas a revisar:

- `SNOWFLAKE_ACCOUNT`
- `SNOWFLAKE_USER`
- `SNOWFLAKE_PASSWORD`
- `SNOWFLAKE_ROLE`
- `SNOWFLAKE_WAREHOUSE`
- `SNOWFLAKE_DATABASE`
- `SNOWFLAKE_SCHEMA_RAW`
- `SNOWFLAKE_SCHEMA_STAGING`
- `SNOWFLAKE_SCHEMA_ANALYTICS`
- `SNOWFLAKE_SCHEMA_ML`
- `TRIP_TYPE`
- `LOCAL_DATA_DIR`

Variables de operacion recomendadas:

```env
ENABLE_DOWNLOAD=true
ENABLE_STAGE_UPLOAD=true
ENABLE_COPY_INTO=true
ENABLE_ZONE_LOOKUP=false
ZONE_LOOKUP_PATH=data/raw/taxi_zone_lookup.csv
DATA_START_DATE=2025-01-01
DATA_END_DATE=2025-06-30
TRAIN_END_DATE=2025-04-30
VAL_END_DATE=2025-05-31
EDA_SAMPLE_LIMIT=10000
EDA_SAMPLE_SEED=42
MODEL_TARGET=fare_amount
TRAIN_SAMPLE_LIMIT=50000
TRAIN_SAMPLE_PCT=1.0
BATCH_SIZE=50000
TRAINING_BATCH_GRAIN=month
MODEL_DIR=data/models
```

## Paso 1. Ingestar Solo RAW

Ejecuta:

```bash
python3 -m src.data.ingestion bootstrap_raw
```

Que hace internamente:

1. crea esquemas y objetos base en Snowflake
2. descarga todos los parquets mensuales de la ventana configurada
3. sube cada archivo al stage interno
4. ejecuta `COPY INTO` hacia `RAW.YELLOW_TRIPS_DEV`
5. registra auditoria minima de carga en `RAW.RAW_LOAD_AUDIT`
6. deja disponible la base raw para trabajar `EDA`, `cleaning` y `feature engineering` sin depender todavia de la OBT

Notas operativas importantes:

- `RAW` no se trunca por defecto
- los archivos ya descargados o auditados se saltan salvo que fuerces `overwrite`
- `ENABLE_ZONE_LOOKUP=false` mantiene el flujo base activo sin enriquecimiento geografico opcional

Que debes validar en logs:

- `raw_rows`

## Paso 2. Validar RAW y Auditar la Ingesta

Corre estas consultas:

```sql
SELECT COUNT(*) FROM DM_FINAL_PROJECT.RAW.YELLOW_TRIPS_DEV;
SELECT COUNT(*) FROM DM_FINAL_PROJECT.RAW.RAW_LOAD_AUDIT;
```

Chequeos temporales recomendados:

```sql
SELECT MIN(tpep_pickup_datetime), MAX(tpep_pickup_datetime)
FROM DM_FINAL_PROJECT.RAW.YELLOW_TRIPS_DEV;
```

Esperado:

- `RAW` con filas mayores a cero
- `RAW_LOAD_AUDIT` con una fila por archivo procesado
- fechas contenidas dentro de la ventana configurada

Para revisar rapidamente una muestra raw desde terminal:

```bash
python3 -m src.data.ingestion sample_raw
```

## Paso 3. Trabajar EDA, Cleaning y Feature Engineering Sobre RAW

Corre en este orden:

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_data_cleaning.ipynb`
3. `notebooks/03_feature_engineering.ipynb`

Qué debe probar cada notebook:

- `01_eda`: EDA descriptivo global sobre muestra de `RAW`, con leakage visible e identificacion de problemas estructurales
- `02_data_cleaning`: reglas estructurales sobre `RAW` y medicion del impacto de los filtros antes de llevarlos a SQL
- `03_feature_engineering`: contrato candidato seguro con `estimated_distance`, derivadas deterministicas y frontera clara contra leakage, todavia sin depender de OBT

Resultado esperado de esta etapa:

- una lista clara de reglas de limpieza
- una muestra candidata de modelado ya filtrada
- un contrato de features aprobado para automatizar en `STAGING/OBT`

## Paso 4. Materializar STAGING, OBT y Splits

Una vez cerradas las reglas y el contrato de features, ejecuta:

```bash
python3 -m src.data.ingestion transform
```

Que hace esta etapa:

1. crea `STAGING.TRIPS_STAGE_DEV`
2. crea `ANALYTICS.OBT_TRIPS_DEV`
3. crea `ML.TRAIN_SET_DEV`, `ML.VAL_SET_DEV` y `ML.TEST_SET_DEV`
4. imprime conteos y diagnosticos de filtros

Valida luego:

```sql
SELECT COUNT(*) FROM DM_FINAL_PROJECT.STAGING.TRIPS_STAGE_DEV;
SELECT COUNT(*) FROM DM_FINAL_PROJECT.ANALYTICS.OBT_TRIPS_DEV;
SELECT COUNT(*) FROM DM_FINAL_PROJECT.ML.TRAIN_SET_DEV;
SELECT COUNT(*) FROM DM_FINAL_PROJECT.ML.VAL_SET_DEV;
SELECT COUNT(*) FROM DM_FINAL_PROJECT.ML.TEST_SET_DEV;
```

Y si quieres una vista rapida de la OBT:

```bash
python3 -m src.data.ingestion sample_obt
```

## Paso 5. Revisar Model Experimentation y Entrenar

Primero corre:

1. `notebooks/04_model_experimentation.ipynb`

Luego ejecuta:

```bash
python3 -m src.models.train_model
```

Modelos esperados:

- `DummyRegressor`
- `SGDRegressor`
- `RandomForestRegressor`
- `AdaBoostRegressor`
- `GradientBoostingRegressor`
- `HistGradientBoostingRegressor`
- `XGBoost`
- `LightGBM`
- `CatBoost`
- `Bagging`
- `Pasting`
- `Voting`

Qué debes revisar en salida:

- tabla comparativa con `training_strategy`
- `val_rmse`
- `test_rmse`
- nombre del modelo ganador
- ruta del artefacto guardado
- `feature_audit`
- `zone_lookup_enabled`
- `unavailable_required_models` vacio

Interpretacion correcta:

- el modelo se elige por `validation`, no por `test`
- `test` se usa solo como evaluacion final
- si `SGDRegressor` gana o compite bien, confirma que la ruta incremental funciona
- si gana otro modelo, igual debes poder justificar por que el costo computacional vale la pena

## Paso 6. Ejecutar Tests

Ejecuta:

```bash
python3 -m pytest
```

La suite debe validar:

- configuracion y parseo de entorno
- presencia de documentos clave y enlaces del README
- proteccion minima contra leakage
- consistencia del pipeline de features
- entrenamiento mock sin depender de Snowflake real

## Paso 7. Levantar la API

Ejecuta:

```bash
uvicorn src.api.main:app --reload
```

Prueba minima:

```bash
curl http://127.0.0.1:8000/health
```

Esperado:

- `status=ok`
- `model_loaded=true`
- `model_name` con el nombre del artefacto entrenado

## Paso 8. Levantar Streamlit

En otra terminal, con el mismo entorno activado:

```bash
streamlit run app/frontend.py
```

Chequeos minimos:

- la app carga sin error
- muestra el estado de la API
- permite ingresar los campos del viaje
- devuelve una tarifa estimada
- muestra el payload enviado y el modelo usado

## Validacion End-to-End Recomendada

Secuencia minima para confirmar que todo funciona:

1. `python3 -m src.data.ingestion bootstrap_raw`
2. validar `RAW` y `RAW_LOAD_AUDIT`
3. correr `01_eda` a `03_feature_engineering`
4. ejecutar `python3 -m src.data.ingestion transform`
5. validar `STAGING/OBT/ML`
6. correr `04_model_experimentation`
7. correr `python3 -m src.models.train_model`
8. correr `python3 -m pytest`
9. levantar `uvicorn`
10. levantar `streamlit`
11. hacer una prediccion manual desde la UI

## Criterios de Aceptacion

Puedes considerar esta base operativa como valida si se cumple todo esto:

- la ingesta cubre los 6 meses configurados
- los notebooks `01-03` pueden correrse usando solo muestra de `RAW`
- la OBT no contiene columnas de leakage en el contrato de modelado
- el contrato publico usa `estimated_distance` fuera de `RAW`
- los splits estan bien separados en el tiempo
- el entrenamiento genera comparacion completa de modelos
- las dependencias requeridas estan instaladas
- los tests quedan en verde
- la API responde
- el frontend consume la API correctamente

## Siguientes Mejoras Despues de Esta Base

Cuando esta ventana de 6 meses quede estable, lo correcto es:

1. extender exactamente el mismo pipeline al rango historico completo
2. medir tiempos reales de bootstrap y entrenamiento
3. ajustar `TRAIN_SAMPLE_LIMIT` y `BATCH_SIZE` con evidencia
4. enriquecer features solo si no rompen la regla de no leakage
5. cerrar la defensa con tablas finales de RMSE y decisiones justificadas
