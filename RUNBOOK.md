# Runbook Operativo

Guia oficial para ejecutar, validar y presentar el proyecto final NYC TLC Taxi Fare Prediction.

## Arquitectura de ejecucion

```
[Snowflake]  RAW → STAGING → ANALYTICS.OBT → ML (TRAIN/VAL/TEST)
[Python]     notebooks EDA/cleaning/FE  →  experimentacion  →  train productivo
[Serving]    FastAPI + Streamlit (Docker Compose o local)
```

- Todo el trabajo pesado (ingesta, limpieza, OBT, splits) vive en Snowflake
- Notebooks consumen muestras de ~100K filas; nunca descargan la base completa
- Entrenamiento XGBoost sobre muestra masiva de 10M filas desde TRAIN_SET_DEV (756M filas)
- Evaluacion en validation y test por lotes de 200K filas

---

## Paso 0. Preparar Entorno

```bash
cd price-prediction-ml-end-to-end/
source .venv/bin/activate          # si ya existe
# o recrear:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

El archivo `.env` debe tener:

| Variable | Valor para produccion |
|---|---|
| `SNOWFLAKE_DATABASE` | `DM_FINAL_PROJECT` |
| `SNOWFLAKE_SCHEMA_RAW` | `RAW` |
| `SNOWFLAKE_SCHEMA_STAGING` | `STAGING` |
| `SNOWFLAKE_SCHEMA_ANALYTICS` | `ANALYTICS` |
| `SNOWFLAKE_SCHEMA_ML` | `ML` |
| `TRIP_TYPE` | `yellow,green` |
| `DATA_START_DATE` | `2015-01-01` |
| `DATA_END_DATE` | `2025-12-31` |
| `TRAIN_END_DATE` | `2023-12-31` |
| `VAL_END_DATE` | `2024-12-31` |
| `TRAIN_SAMPLE_LIMIT` | `10000000` |
| `BATCH_SIZE` | `200000` |

Verificar que la configuracion carga correctamente:

```bash
python3 -c "
from src.utils.config import get_settings
s = get_settings()
print('database:', s.snowflake_database)
print('trip_types:', s.trip_types)
print('train_table:', s.train_table)
print('train_sample_limit:', s.train_sample_limit)
"
```

---

## Paso 1. Ingesta RAW (solo si no se ha hecho)

Solo ejecutar si las tablas RAW estan vacias o incompletas.

```bash
# Descarga parquets del CDN TLC y los carga a Snowflake RAW
# ENABLE_DOWNLOAD/STAGE_UPLOAD/COPY_INTO=true requeridos
python3 -m src.data.ingestion bootstrap_raw
```

**Volumenes esperados (ya cargados al 2026-05-10):**
- `RAW.YELLOW_TRIPS_DEV`: 801 553 240 filas
- `RAW.GREEN_TRIPS_DEV`: 68 239 054 filas

---

## Paso 2. Transformar — STAGING, OBT y Splits (solo si no se ha hecho)

```bash
# Ejecuta en secuencia:
# 02_create_staging_trips_dev.sql  (yellow+green → STAGING con 10 reglas de limpieza)
# 03_create_obt_trips_dev.sql      (STAGING → OBT con features derivadas, sin leakage)
# 04_create_time_splits_dev.sql    (OBT → TRAIN/VAL/TEST como tablas materializadas)
python3 -m src.data.ingestion transform
```

**Volumenes esperados (ya materializados al 2026-05-10):**

| Tabla | Filas | Tipo |
|---|---|---|
| `STAGING.TRIPS_STAGE_DEV` | 829 956 201 | TABLE |
| `ANALYTICS.OBT_TRIPS_DEV` | 828 426 876 | TABLE |
| `ML.TRAIN_SET_DEV` | 756 778 077 | TABLE (2015-2023) |
| `ML.VAL_SET_DEV` | 36 148 221 | TABLE (2024) |
| `ML.TEST_SET_DEV` | 35 500 578 | TABLE (2025) |

Verificar sin solapamiento temporal:

```sql
-- Ejecutar en Snowflake para verificar integridad de splits
SELECT
    'train' AS split, MIN(pickup_datetime) AS min_dt, MAX(pickup_datetime) AS max_dt, COUNT(*) AS rows
FROM DM_FINAL_PROJECT.ML.TRAIN_SET_DEV
UNION ALL
SELECT 'val', MIN(pickup_datetime), MAX(pickup_datetime), COUNT(*) FROM DM_FINAL_PROJECT.ML.VAL_SET_DEV
UNION ALL
SELECT 'test', MIN(pickup_datetime), MAX(pickup_datetime), COUNT(*) FROM DM_FINAL_PROJECT.ML.TEST_SET_DEV;
```

---

## Paso 3. Notebooks de exploración (ejecutar en orden)

Los notebooks 01, 02 y 03 validan las decisiones de diseno sobre muestras representativas.
El notebook 04 compara el shortlist de modelos y elige el productivo.

```bash
jupyter notebook notebooks/
```

| Notebook | Fuente de datos | Descripcion |
|---|---|---|
| `01_eda.ipynb` | `RAW` (100K balanceado) | distribucion, outliers, leakage visible |
| `02_data_cleaning.ipynb` | `RAW` (100K balanceado) | valida 10 reglas de limpieza, impacto por flota |
| `03_feature_engineering.ipynb` | `OBT` (100K) | contrato de features v3, anti-leakage, yellow/green |
| `04_model_experimentation.ipynb` | `ML splits` (100K train, full val/test en lotes) | shortlist, val_rmse, metricas por flota |

---

## Paso 4. Entrenamiento Productivo

Entrena XGBoost sobre 10M filas aleatorias de `TRAIN_SET_DEV` (756M filas).
Evalua en `VAL_SET_DEV` y `TEST_SET_DEV` por lotes. Guarda el artefacto.

```bash
python3 -m src.models.train_model
```

**Estrategia out-of-core:**
- 756M filas de train viven en Snowflake; nunca se descargan completas
- `ORDER BY RANDOM() LIMIT 10_000_000` en Snowflake extrae 10M filas aleatorias representativas
- XGBoost usa sparse CSR matrix (memoria eficiente, ~1-2GB RAM para 10M filas)
- Evaluacion en val (36M) y test (35M) por lotes de 200K sin carga en memoria

**Tiempo estimado:**
- Fetch 10M filas desde Snowflake: ~10-15 min
- Preprocesamiento + fit XGBoost 400 rondas: ~30-60 min
- Evaluacion val + test en batches: ~20-30 min
- **Total: ~60-100 min**

**Log esperado al terminar:**
```
Production training finished | model=xgboost | val_rmse=X.XXXX | test_rmse=X.XXXX | artifact=data/models/nyc_taxi_fare_production.joblib | elapsed=XXXs
```

**Verificar el artefacto:**
```bash
python3 -c "
import joblib
art = joblib.load('data/models/nyc_taxi_fare_production.joblib')
print('modelo:    ', art['model_name'])
print('val_rmse:  ', art['metrics']['val_rmse'])
print('test_rmse: ', art['metrics']['test_rmse'])
print('n_filas:   ', art['metrics']['sample_rows'])
print('trip_types:', art['metrics']['trip_types'])
print('features:  ', art['feature_audit']['model_feature_columns'])
"
```

---

## Paso 5. Tests

```bash
python3 -m pytest -v
```

Los tests validan:
- contrato de features y exclusion de leakage
- pipeline de preprocesamiento
- prediccion con el artefacto productivo
- consistencia de documentacion

---

## Paso 6. Serving Local

```bash
# Terminal 1 — API
uvicorn src.api.main:app --reload
# -> http://127.0.0.1:8000/docs  (Swagger interactivo)

# Terminal 2 — Frontend
streamlit run app/frontend.py
# -> http://127.0.0.1:8501
```

**Prueba de prediccion rapida:**
```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"trip_type":"yellow","pickup_datetime":"2025-03-15T08:30:00","pickup_location_id":161,"dropoff_location_id":237,"passenger_count":1,"estimated_distance":3.2,"vendor_id":2,"ratecode_id":1}' | python3 -m json.tool
```

---

## Paso 7. Serving con Docker Compose

El artefacto debe existir antes de levantar Docker.

```bash
# El entrenamiento productivo ocurre fuera de Docker:
python3 -m src.models.train_model

# Luego levantar API + frontend en contenedores:
docker compose up --build

# API:      http://127.0.0.1:8000
# Frontend: http://127.0.0.1:8501
```

---

## Criterios de Aceptacion

| Check | Como verificar |
|---|---|
| RAW cargado | logs de `bootstrap_raw` muestran yellow ~800M, green ~68M |
| STAGING correcto | `staging_rows ≈ 829M` (95.4% del RAW) |
| OBT sin leakage | `ANALYTICS.OBT_TRIPS_DEV` no tiene `payment_type`, `tip_amount`, etc. |
| Splits sin solapamiento | query SQL del Paso 2 muestra rangos disjuntos |
| Modelo entrena sin OOM | log muestra `Fetched train sample \| rows=10,000,000` sin crash |
| val_rmse < baseline | `val_rmse` de XGBoost menor que `DummyRegressor` |
| test evaluado 1 sola vez | `test_rmse` aparece solo en el log final del `train_model` |
| API responde | `curl /predict` devuelve `{"predicted_fare": X.XX, ...}` |
| Tests verdes | `pytest` pasa sin errores |

---

## Documentacion de Referencia

- `docs/problem_definition.md` — objetivo, arquitectura, estrategia out-of-core
- `docs/data_contract.md` — columnas por capa, reglas de calidad, lista prohibida
- `docs/feature_audit.md` — contrato v3, derivadas, anti-leakage
- `docs/model_rubric_matrix.md` — shortlist, modelo productivo, configuracion
- `docs/decisions_log.md` — decisiones de diseno con fecha y justificacion
- `ENUNCIADO.md` — enunciado original del proyecto
