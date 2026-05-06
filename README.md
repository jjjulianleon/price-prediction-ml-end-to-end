# Price Prediction ML End to End

Proyecto final de prediccion de `fare_amount` para `NYC TLC Taxi Trip Records`, diseñado con arquitectura `Snowflake-first`, entrenamiento reproducible y serving mediante `FastAPI + Streamlit`.

## Resumen

El objetivo es predecir la tarifa base de un viaje individual en un momento previo al inicio del trayecto. La solucion evita dos errores criticos para esta entrega:

- mover la logica pesada fuera de Snowflake
- usar variables conocidas solo al finalizar o pagar el viaje

La arquitectura oficial del proyecto es:

- `RAW`: ingesta de parquet TLC y auditoria de carga
- `STAGING`: tipado, validaciones estructurales y columnas diagnosticas
- `ANALYTICS`: OBT final segura para analisis y modelado
- `ML`: splits temporales y datasets listos para entrenamiento

## Alcance del Proyecto

El alcance oficial del proyecto contempla el periodo historico completo definido en el enunciado:

- `train`: 2015 a 2023
- `validation`: 2024
- `test`: 2025

El pipeline ya esta preparado para ese esquema. La ventana exacta de trabajo se controla desde `.env`, por lo que puede ejecutarse sobre una muestra operativa de 6 meses para validacion tecnica sin cambiar la logica del proyecto.

## Decisiones Tecnicas Clave

- `Snowflake-first`: ingesta, limpieza estructural, OBT y splits se ejecutan en SQL.
- `Multi-fleet`: el pipeline soporta `yellow`, `green` o ambas flotas, con `RAW` separado por tipo y `trip_type` como feature categorica.
- `EDA sobre muestra`: los notebooks trabajan con muestras representativas; no se descarga toda la base a Pandas.
- `No leakage`: el contrato final excluye variables post-viaje o post-pago.
- `Estimated distance`: `trip_distance` se usa como proxy historico y se publica como `estimated_distance` fuera de `RAW`.
- `Out-of-core pragmático`: entrenamiento incremental real donde el modelo lo permite y evaluacion por lotes para `validation/test`.

## Feature Set Final

### Features aprobadas

- `pickup_datetime`
- `trip_type`
- `passenger_count`
- `estimated_distance`
- `pickup_location_id`
- `dropoff_location_id`
- `vendor_id`
- `ratecode_id`

### Derivadas finales

- `pickup_hour`
- `pickup_dayofweek`
- `pickup_month`
- `is_weekend`
- `is_rush_hour`
- `is_night`
- `log_estimated_distance`
- `route_id`
- `same_zone`

### Variables prohibidas por leakage

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

## Experimentacion Y Produccion

Desde esta etapa el proyecto separa explicitamente dos rutas:

- `src/models/model_zoo.py` y `src/models/experiment_runner.py`: solo para experimentacion
- `src/models/train_model.py`: solo para entrenamiento productivo final

Shortlist curado actual para comparar en `notebooks/04_model_experimentation.ipynb`:

- `DummyRegressor`
- `SGDRegressor`
- `RandomForestRegressor`
- `GradientBoostingRegressor`
- `HistGradientBoostingRegressor`
- `XGBoost`
- `LightGBM`
- `CatBoost`

Modelos como `Ridge`, `AdaBoost`, `Bagging`, `Pasting` y `Voting` quedan archivados en el zoo para trazabilidad experimental, pero salen del flujo principal porque ya mostraron peor tradeoff entre calidad y costo.

Modelo productivo seleccionado al 2026-05-05:

- `GradientBoostingRegressor`

El entrenamiento final genera un artefacto fijo en `MODEL_DIR/nyc_taxi_fare_production.joblib`, que es el archivo esperado por la API y el frontend.

## Estructura

```text
├── app/                # Frontend Streamlit
├── data/               # Datos locales y artefactos
├── docs/               # Definicion del problema, contrato y trazabilidad
├── notebooks/          # EDA, cleaning, feature engineering y experimentacion
├── src/
│   ├── api/            # FastAPI
│   ├── data/           # Ingesta y SQL
│   ├── features/       # Contrato de features y preprocessing
│   ├── models/         # Model zoo, training e inference
│   └── utils/          # Configuracion y conexion
├── tests/
├── .env.example
├── README.md
└── RUNBOOK.md
```

## Notebooks

- `01_eda.ipynb`: audita una muestra de `RAW` y documenta distribuciones, outliers, leakage visible y estructura de origen.
- cuando `TRIP_TYPE=yellow,green`, el EDA consume muestras de ambas flotas y deja comparativas por `trip_type`
- `02_data_cleaning.ipynb`: justifica con evidencia las reglas de calidad que luego se automatizan en SQL.
- `03_feature_engineering.ipynb`: valida el contrato final de features desde una muestra candidata ya filtrada.
- `04_model_experimentation.ipynb`: revisa los splits materializados, corre un shortlist por celdas independientes y persiste progreso en `data/models/notebook04_progress.csv`.

## Configuracion

La configuracion operativa vive en [.env.example](/home/pabseb/DataMining/final-project/price-prediction-ml-end-to-end/.env.example:1). Los parametros mas importantes son:

- `SNOWFLAKE_DATABASE`
- `SNOWFLAKE_SCHEMA_RAW`
- `SNOWFLAKE_SCHEMA_STAGING`
- `SNOWFLAKE_SCHEMA_ANALYTICS`
- `SNOWFLAKE_SCHEMA_ML`
- `DATA_START_DATE`
- `DATA_END_DATE`
- `TRAIN_END_DATE`
- `VAL_END_DATE`
- `EDA_SAMPLE_LIMIT`
- `TRAIN_SAMPLE_LIMIT`
- `TRAIN_SAMPLE_PCT`
- `BATCH_SIZE`
- `TRAINING_BATCH_GRAIN`
- `MODEL_DIR`
- `TRIP_TYPE`

Valores soportados para `TRIP_TYPE`:

- `yellow`
- `green`
- `yellow,green`

## Ejecucion

El flujo operativo completo esta documentado en [RUNBOOK.md](/home/pabseb/DataMining/final-project/price-prediction-ml-end-to-end/RUNBOOK.md:1). Resumen corto:

```bash
python3 -m src.data.ingestion bootstrap_raw
python3 -m src.data.ingestion transform
python3 -m src.models.train_model
python3 -m pytest
uvicorn src.api.main:app --reload
streamlit run app/frontend.py
```

Levantada conjunta de la app con Docker Compose:

```bash
python3 -m src.models.train_model
docker compose up --build
```

Con eso:

- API: `http://127.0.0.1:8000`
- Frontend: `http://127.0.0.1:8501`

Importante:

- el entrenamiento productivo ocurre fuera de Docker
- Compose asume que ya existe `data/models/nyc_taxi_fare_production.joblib`
- Docker se usa solo para servir la app (`api` + `frontend`)

## Documentacion

- [problem_definition.md](/home/pabseb/DataMining/final-project/price-prediction-ml-end-to-end/docs/problem_definition.md:1)
- [data_contract.md](/home/pabseb/DataMining/final-project/price-prediction-ml-end-to-end/docs/data_contract.md:1)
- [feature_audit.md](/home/pabseb/DataMining/final-project/price-prediction-ml-end-to-end/docs/feature_audit.md:1)
- [model_rubric_matrix.md](/home/pabseb/DataMining/final-project/price-prediction-ml-end-to-end/docs/model_rubric_matrix.md:1)
- [decisions_log.md](/home/pabseb/DataMining/final-project/price-prediction-ml-end-to-end/docs/decisions_log.md:1)

## Resultado Esperado

Una corrida correcta debe dejar:

- `RAW` cargada y auditada
- `STAGING`, `ANALYTICS.OBT` y `ML` materializados
- notebooks consistentes con el contrato final
- benchmark de shortlist con `val_rmse` y `test_rmse`
- artefacto final `MODEL_DIR/nyc_taxi_fare_production.joblib`
- API y frontend operativos
