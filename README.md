# Price Prediction ML End to End

Proyecto final de prediccion de tarifas para `NYC Yellow Taxi` con arquitectura `Snowflake-first`, entrenamiento controlado para volumen grande y despliegue mediante `FastAPI + Streamlit`.

La solucion esta diseñada para cumplir el enunciado del curso evitando dos fallas criticas: descargar toda la data a memoria local y contaminar el entrenamiento con variables post-viaje. La base actual opera sobre una ventana estable de `6 meses` y deja el pipeline listo para escalar despues al periodo historico completo.

## Resumen Ejecutivo

El flujo final del proyecto resuelve cinco necesidades del enunciado:

1. construir la OBT y los splits temporales del lado de Snowflake
2. explorar y validar la data solo sobre muestras
3. entrenar modelos con una estrategia compatible con volumen grande
4. comparar baselines, boostings obligatorios y ensambles con `RMSE`
5. servir el modelo seleccionado mediante API y frontend

Decisiones tecnicas principales:

- `Pushdown computation`: la limpieza estructural y los splits viven en SQL
- `No leakage`: solo se usan variables disponibles antes de iniciar el viaje
- `Out-of-core pragmático`: entrenamiento incremental real donde el algoritmo lo soporta, muestra controlada donde no existe `partial_fit`
- `Evaluacion por lotes`: validation y test se recorren en batches, no se cargan completos en memoria
- `Split mensual 4/1/1`: para la base de 6 meses se usa `train=enero-abril`, `validation=mayo`, `test=junio`

## Cumplimiento del Enunciado

La implementacion actual cubre lo que exige la entrega tecnica:

- OBT materializada en Snowflake con filtros y reglas de limpieza estructural
- splits temporales construidos en la base, sin `train_test_split` local
- notebooks separados para `EDA`, `data cleaning`, `feature engineering` y `model experimentation`
- pipeline modular en `src/` para ingesta, features, entrenamiento e inferencia
- API en FastAPI y frontend en Streamlit
- suite de tests para configuracion, leakage y entrenamiento mock

Puntos clave de rubrica que este repo deja explicitamente resueltos:

- no se usan columnas como `payment_type`, `tip_amount`, `tolls_amount` o `total_amount` como input
- la data completa no se procesa en pandas
- los modelos obligatorios quedan integrados en el catalogo del pipeline
- el artefacto final guarda modelo, preprocesador y metricas para trazabilidad

## Arquitectura

### 1. Ingesta y modelado en Snowflake

La ventana actual se define en `.env` y la arquitectura queda separada en dos etapas:

- etapa 1: `setup + ingest` para poblar `RAW.YELLOW_TRIPS_DEV` y trabajar `EDA / cleaning / feature engineering` sobre muestras raw
- etapa 2: `transform` para construir `STAGING.TRIPS_STAGE_DEV`, `ANALYTICS.OBT_TRIPS_DEV` y los splits de `ML`

Los scripts SQL obligatorios estan en `src/data/sql/`:

- `00_create_schemas.sql`
- `01_create_external_or_stage_tables.sql`
- `02_create_staging_trips_dev.sql`
- `03_create_obt_trips_dev.sql`
- `04_create_time_splits_dev.sql`

### 2. Splits temporales

La base actual usa un split mensual pensado para estabilizar la experimentacion:

- `train`: `2025-01-01` a `2025-04-30`
- `validation`: `2025-05-01` a `2025-05-31`
- `test`: `2025-06-01` a `2025-06-30`

La razon de esta decision es simple: para una primera ventana de 6 meses, un split mensual es mas interpretable, mas facil de auditar y menos ruidoso que un split semanal. El parametro `TRAINING_BATCH_GRAIN` existe para evaluar lotes `month` o `week` sin cambiar el resto del pipeline.

### 3. Entrenamiento y evaluacion

La estrategia del proyecto no promete `out-of-core` falso. Se hace distincion entre:

- modelos incrementales reales: `SGDRegressor`
- modelos sin `partial_fit`: entrenamiento sobre muestra representativa controlada

La evaluacion se ejecuta siempre por lotes sobre `validation` y `test`, de modo que la memoria local no escala con el tamano completo de los splits.

## Features Finales

### Variables de entrada aprobadas

Estas son las variables finales del contrato de features:

- `pickup_datetime`
- `passenger_count`
- `estimated_distance`
- `pickup_location_id`
- `dropoff_location_id`
- `vendor_id`
- `ratecode_id`

### Variables derivadas en el pipeline

El preprocesador transforma `pickup_datetime` en:

- `pickup_hour`
- `pickup_dayofweek`
- `pickup_month`
- `is_weekend`
- `is_rush_hour`
- `is_night`

Ademas construye derivadas deterministicas y trazables:

- `log_estimated_distance`
- `route_id`
- `same_zone`

### Variables excluidas por leakage

Se excluyen de forma explicita columnas conocidas solo al cierre del viaje o derivadas de ese cierre:

- `payment_type`
- `tip_amount`
- `tolls_amount`
- `mta_tax`
- `improvement_surcharge`
- `congestion_surcharge`
- `airport_fee`
- `total_amount`
- `trip_duration_minutes`
- `avg_speed_mph`
- `tip_pct`
- `fare_per_mile`

La regla es estricta: si una variable depende del fin del viaje o del monto final liquidado, no entra al modelo.

## Modelos Integrados

El pipeline compara:

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

Interpretacion del catalogo:

- `DummyRegressor` sirve de baseline minimo
- `SGDRegressor` representa la ruta incremental real
- los ensambles y boostings obligatorios cubren la parte comparativa de la rubrica
- `Voting`, `Bagging` y `Pasting` dejan evidencia de comparacion entre familias de ensamble

## Estructura del Repositorio

```text
├── app/                # Frontend Streamlit
├── data/               # Modelos y directorios de trabajo
├── notebooks/          # Evidencia analitica del flujo
├── src/
│   ├── api/            # FastAPI
│   ├── data/           # Ingesta y SQL
│   ├── features/       # Contrato de features y preprocessing
│   ├── models/         # Model zoo, training e inference
│   └── utils/          # Configuracion y conexion
├── tests/              # Tests unitarios y mocks del pipeline
├── .env.example
├── ENUNCIADO.md
├── README.md
└── RUNBOOK.md
```

## Notebooks

Cada notebook tiene un rol claro y complementario:

- `01_eda.ipynb`: explora una muestra de `RAW`, identifica leakage visible y documenta la calidad base
- `02_data_cleaning.ipynb`: mide el impacto de las reglas de limpieza sobre `RAW` antes de automatizarlas en SQL
- `03_feature_engineering.ipynb`: construye el contrato de features candidato desde una muestra raw ya filtrada
- `04_model_experimentation.ipynb`: usa los splits ya materializados despues de `transform`

La intencion no es dejar notebooks decorativos, sino evidencia legible para defensa y continuidad del equipo.

## Configuracion

La plantilla operativa esta en [.env.example](/home/pabseb/DataMining/final-project/price-prediction-ml-end-to-end/.env.example:1). Las variables mas importantes son:

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
- `ENABLE_DOWNLOAD`
- `ENABLE_STAGE_UPLOAD`
- `ENABLE_COPY_INTO`
- `ENABLE_ZONE_LOOKUP`
- `ZONE_LOOKUP_PATH`
- `DATA_START_DATE`
- `DATA_END_DATE`
- `TRAIN_END_DATE`
- `VAL_END_DATE`
- `MODEL_TARGET`
- `EDA_SAMPLE_LIMIT`
- `EDA_SAMPLE_SEED`
- `TRAIN_SAMPLE_LIMIT`
- `TRAIN_SAMPLE_PCT`
- `BATCH_SIZE`
- `TRAINING_BATCH_GRAIN`
- `MODEL_DIR`

## Documentos Clave

La definicion formal del problema, el contrato de datos y la trazabilidad metodologica quedaron separados del README para que el equipo tenga una referencia operativa mas clara:

- [problem_definition.md](/home/pabseb/DataMining/final-project/price-prediction-ml-end-to-end/docs/problem_definition.md:1)
- [data_contract.md](/home/pabseb/DataMining/final-project/price-prediction-ml-end-to-end/docs/data_contract.md:1)
- [feature_audit.md](/home/pabseb/DataMining/final-project/price-prediction-ml-end-to-end/docs/feature_audit.md:1)
- [model_rubric_matrix.md](/home/pabseb/DataMining/final-project/price-prediction-ml-end-to-end/docs/model_rubric_matrix.md:1)
- [decisions_log.md](/home/pabseb/DataMining/final-project/price-prediction-ml-end-to-end/docs/decisions_log.md:1)

Default recomendado para la base actual:

- ventana de `6 meses`
- sample de entrenamiento controlado
- batches de `50000`
- granularidad de lote `month`

## Ejecucion

La guia operativa completa esta en [RUNBOOK.md](/home/pabseb/DataMining/final-project/price-prediction-ml-end-to-end/RUNBOOK.md:1). El flujo corto de validacion queda separado por etapas:

```bash
python3 -m src.data.ingestion bootstrap_raw
python3 -m src.data.ingestion sample_raw
python3 -m src.data.ingestion transform
python3 -m src.models.train_model
python3 -m pytest
uvicorn src.api.main:app --reload
streamlit run app/frontend.py
```

## Validacion Esperada

Una corrida correcta debe dejar:

- data cargada en `RAW.YELLOW_TRIPS_DEV`
- muestra raw disponible para `EDA / cleaning / feature engineering`
- OBT poblada en `ANALYTICS.OBT_TRIPS_DEV`
- splits temporales sin solapamiento
- artefacto de modelo en `data/models/`
- `pytest` en verde
- `GET /health` con `model_loaded=true`
- Streamlit consumiendo la API sin errores

## Referencias

- Enunciado oficial preservado en [ENUNCIADO.md](/home/pabseb/DataMining/final-project/price-prediction-ml-end-to-end/ENUNCIADO.md:1)
- Guia operativa del proyecto en [RUNBOOK.md](/home/pabseb/DataMining/final-project/price-prediction-ml-end-to-end/RUNBOOK.md:1)
