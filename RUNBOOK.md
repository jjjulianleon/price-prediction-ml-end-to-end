# Runbook Operativo

Guia oficial para ejecutar, validar y presentar el proyecto final de prediccion de tarifas NYC TLC Taxi.

## Objetivo

Este runbook deja una secuencia unica y reproducible para:

1. cargar datos en Snowflake
2. auditar calidad y features
3. materializar `STAGING`, `OBT` y splits temporales
4. comparar modelos
5. publicar el mejor modelo en API y frontend

## Configuracion Recomendada

La ventana historica oficial del proyecto es configurable desde `.env`. El esquema final esperado por la rubrica es:

- `train`: 2015-01-01 a 2023-12-31
- `validation`: 2024-01-01 a 2024-12-31
- `test`: 2025-01-01 a 2025-12-31

La misma arquitectura puede correrse sobre una ventana mas corta de validacion tecnica sin cambiar el pipeline. Para evitar mezclar objetos, se recomienda usar una base dedicada como `DM_EXP_FINAL_PROJECT`.

## Paso 0. Preparar Entorno

```bash
cd /home/pabseb/DataMining/final-project/price-prediction-ml-end-to-end
source .venv/bin/activate
```

Si necesitas recrear el entorno:

```bash
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Variables criticas:

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
- `DATA_START_DATE`
- `DATA_END_DATE`
- `TRAIN_END_DATE`
- `VAL_END_DATE`
- `MODEL_DIR`
- `TRIP_TYPE`

Configuraciones validas de flota:

- `TRIP_TYPE=yellow`: ingesta solo Yellow Taxi
- `TRIP_TYPE=green`: ingesta solo Green Taxi
- `TRIP_TYPE=yellow,green`: ingesta ambas flotas y las integra en la OBT con `trip_type` como feature

## Paso 1. Setup e Ingesta RAW

```bash
python3 -m src.data.ingestion bootstrap_raw
```

Esto crea esquemas y objetos base, descarga los parquet configurados y carga `RAW.YELLOW_TRIPS_DEV`, `RAW_TEST.GREEN_TRIPS_DEV` o ambos segun `TRIP_TYPE`, cada uno con su propia auditoria `RAW_LOAD_AUDIT`.

Validacion minima en Snowflake:

```sql
SELECT COUNT(*) FROM DM_EXP_FINAL_PROJECT.RAW_TEST.YELLOW_TRIPS_DEV;
SELECT COUNT(*) FROM DM_EXP_FINAL_PROJECT.RAW_TEST.RAW_LOAD_AUDIT;

SELECT COUNT(*) FROM DM_EXP_FINAL_PROJECT.RAW_TEST.GREEN_TRIPS_DEV;
SELECT COUNT(*) FROM DM_EXP_FINAL_PROJECT.RAW_TEST.RAW_LOAD_AUDIT;

SELECT MIN(tpep_pickup_datetime), MAX(tpep_pickup_datetime)
FROM DM_EXP_FINAL_PROJECT.RAW_TEST.YELLOW_TRIPS_DEV;
```

Muestra rapida desde terminal:

```bash
python3 -m src.data.ingestion sample_raw
```

## Paso 2. EDA, Cleaning y Feature Engineering

Corre los notebooks en este orden:

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_data_cleaning.ipynb`
3. `notebooks/03_feature_engineering.ipynb`

Qué validar:

- `01_eda`: estructura raw, leakage visible, distribuciones y outliers
- si `TRIP_TYPE=yellow,green`, revisar diferencias descriptivas por `trip_type`
- `02_data_cleaning`: impacto de reglas de calidad y ejemplos invalidos
- `03_feature_engineering`: contrato final de features, derivadas y compatibilidad con el pipeline reusable

Resultado esperado:

- reglas de limpieza cerradas
- contrato de features aprobado
- evidencia suficiente para automatizar `STAGING/OBT`

## Paso 3. Materializar STAGING, OBT y Splits

```bash
python3 -m src.data.ingestion transform
```

Esto ejecuta:

- `02_create_staging_trips_dev.sql`
- `03_create_obt_trips_dev.sql`
- `04_create_time_splits_dev.sql`

Si prefieres correrlo manualmente en Snowflake Worksheet, ejecuta esos tres scripts en ese mismo orden.

Validacion minima:

```sql
SELECT COUNT(*) FROM DM_EXP_FINAL_PROJECT.STAGING_TEST.TRIPS_STAGE_DEV;
SELECT COUNT(*) FROM DM_EXP_FINAL_PROJECT.ANALYTICS_TEST.OBT_TRIPS_DEV;
SELECT COUNT(*) FROM DM_EXP_FINAL_PROJECT.ML_TEST.TRAIN_SET_DEV;
SELECT COUNT(*) FROM DM_EXP_FINAL_PROJECT.ML_TEST.VAL_SET_DEV;
SELECT COUNT(*) FROM DM_EXP_FINAL_PROJECT.ML_TEST.TEST_SET_DEV;

SELECT MIN(pickup_datetime), MAX(pickup_datetime)
FROM DM_EXP_FINAL_PROJECT.ML_TEST.TRAIN_SET_DEV;

SELECT MIN(pickup_datetime), MAX(pickup_datetime)
FROM DM_EXP_FINAL_PROJECT.ML_TEST.VAL_SET_DEV;

SELECT MIN(pickup_datetime), MAX(pickup_datetime)
FROM DM_EXP_FINAL_PROJECT.ML_TEST.TEST_SET_DEV;
```

Vista rapida de la OBT:

```bash
python3 -m src.data.ingestion sample_obt
```

## Paso 4. Experimentacion de Modelos

Corre primero el notebook:

1. `notebooks/04_model_experimentation.ipynb`

Que cambia en esta version:

- cada modelo del shortlist corre en su propia celda
- el notebook guarda progreso en `data/models/notebook04_progress.csv`
- el `model_zoo` queda solo para experimentacion; ya no define el artefacto productivo

Qué revisar:

- `comparison`
- `val_rmse`
- `test_rmse`
- `notebook04_progress.csv`

La seleccion final se hace por `validation`. `test` queda reservado para verificacion final.

Como referencia actual, el log `notebooks/temp.txt` ya deja suficiente evidencia para promover `gradient_boosting` a produccion mientras `catboost` siga inconcluso por estabilidad de entorno.

Opcional desde terminal:

```bash
python3 -m src.models.experiment_runner
```

## Paso 5. Entrenamiento Productivo Final

Entrena solo el modelo seleccionado para serving:

```bash
python3 -m src.models.train_model
```

Resultado esperado:

- artefacto fijo en `data/models/nyc_taxi_fare_production.joblib`
- metricas de `validation` y `test` del modelo productivo
- metadata de contrato y evidencia de seleccion dentro del artefacto

## Paso 6. Pruebas del Proyecto

```bash
python3 -m pytest
```

La suite debe quedar en verde y cubrir:

- configuracion
- contrato de features
- proteccion minima contra leakage
- documentacion obligatoria
- entrenamiento mock sin Snowflake real

## Paso 7. Publicar API y Frontend

API:

```bash
uvicorn src.api.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Frontend:

```bash
streamlit run app/frontend.py
```

Verificaciones minimas:

- la API levanta y detecta `nyc_taxi_fare_production.joblib`
- Streamlit consume la API sin errores
- una prediccion manual devuelve tarifa y nombre del modelo

## Paso 8. Opcion Recomendada Con Docker Compose

Si quieres levantar backend y frontend de forma consistente, usa Docker Compose. En esta version, Docker se usa solo para la app ya servida; el entrenamiento final ocurre fuera de contenedor.

Primero entrena el artefacto productivo en tu entorno local:

```bash
python3 -m src.models.train_model
```

Luego levanta API y frontend:

```bash
docker compose up --build
```

Puertos esperados:

- API: `http://127.0.0.1:8000`
- Frontend: `http://127.0.0.1:8501`

Notas operativas:

- el frontend usa `API_BASE_URL=http://api:8000` dentro de la red de Compose
- el artefacto productivo se persiste en `./data/models`
- Compose asume que ya existe `./data/models/nyc_taxi_fare_production.joblib`
- si cambias el modelo y reentrenas, reinicia Compose para que la app cargue el nuevo artefacto

## Secuencia Recomendada Completa

```bash
python3 -m src.data.ingestion bootstrap_raw
python3 -m src.data.ingestion sample_raw
# correr notebooks 01, 02, 03
python3 -m src.data.ingestion transform
python3 -m src.data.ingestion sample_obt
# correr notebook 04
python3 -m src.models.train_model
python3 -m pytest
uvicorn src.api.main:app --reload
streamlit run app/frontend.py
```

## Criterios de Aceptacion

El proyecto puede considerarse listo para presentacion cuando:

- `RAW`, `STAGING`, `ANALYTICS` y `ML` quedan pobladas en Snowflake
- el contrato final usa `estimated_distance` y excluye leakage
- los splits temporales respetan el corte configurado
- existe benchmark del shortlist con metricas comparables
- el modelo productivo queda guardado en `MODEL_DIR/nyc_taxi_fare_production.joblib`
- API y frontend funcionan sobre ese artefacto
