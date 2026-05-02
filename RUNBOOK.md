# Runbook Operativo

## Proposito

Este documento resume como ejecutar la fase base del proyecto de prediccion de precios NYC Yellow Taxi sobre Snowflake, desde la preparacion del entorno hasta la validacion inicial de datos y el entrenamiento experimental.

## Ejecucion completa

Si quieres correr todo el flujo base de la forma recomendada:

```bash
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m src.data.ingestion bootstrap
python3 -m src.models.train_model
python3 -m pytest
```

Que hace `bootstrap`:

1. crea esquemas y tabla raw dev en Snowflake
2. descarga automaticamente el parquet mensual oficial de NYC TLC
3. sube el archivo a un stage interno de Snowflake
4. ejecuta `COPY INTO` directo a `RAW.YELLOW_TRIPS_DEV`
5. construye `ANALYTICS.OBT_TRIPS_DEV`
6. crea `ML.TRAIN_SET_DEV`, `ML.VAL_SET_DEV` y `ML.TEST_SET_DEV`
7. imprime conteos en logs

## Paso 1. Configurar y ejecutar la carga completa

Antes de correr el flujo, completa `.env` con:

- credenciales de Snowflake
- base, schemas y warehouse
- rango de fechas del mes a procesar

Ejemplo:

```env
DATA_START_DATE=2025-01-01
DATA_END_DATE=2025-01-31
TRAIN_END_DATE=2025-01-21
VAL_END_DATE=2025-01-27
```

Luego ejecuta:

```bash
python3 -m src.data.ingestion bootstrap
```

## Paso 2. Probar y validar la data

Despues de `bootstrap`, valida primero que Snowflake quedo bien poblado.

La validacion minima es revisar los logs del comando. Debes ver conteos para:

- `raw_rows`
- `obt_rows`
- `train_rows`
- `val_rows`
- `test_rows`

Si quieres probarlo manualmente en Snowflake, corre:

```sql
SELECT COUNT(*) FROM DM_FINAL_PROJECT.RAW.YELLOW_TRIPS_DEV;
SELECT COUNT(*) FROM DM_FINAL_PROJECT.ANALYTICS.OBT_TRIPS_DEV;
SELECT COUNT(*) FROM DM_FINAL_PROJECT.ML.TRAIN_SET_DEV;
SELECT COUNT(*) FROM DM_FINAL_PROJECT.ML.VAL_SET_DEV;
SELECT COUNT(*) FROM DM_FINAL_PROJECT.ML.TEST_SET_DEV;
```

Chequeos recomendados:

```sql
SELECT MIN(tpep_pickup_datetime), MAX(tpep_pickup_datetime)
FROM DM_FINAL_PROJECT.RAW.YELLOW_TRIPS_DEV;

SELECT MIN(pickup_datetime), MAX(pickup_datetime)
FROM DM_FINAL_PROJECT.ANALYTICS.OBT_TRIPS_DEV;
```

Esperado:

- `RAW.YELLOW_TRIPS_DEV` con filas mayores a cero
- `OBT_TRIPS_DEV` con filas mayores a cero
- splits con filas y sin solapamiento temporal evidente

## Paso 3. Explorar datos en notebooks

Corre en este orden:

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_data_cleaning.ipynb`
3. `notebooks/03_feature_engineering.ipynb`
4. `notebooks/04_model_experimentation.ipynb`

Objetivo:

- revisar distribuciones y nulos
- validar leakage
- confirmar reglas de limpieza
- probar el pipeline experimental sobre muestras

## Paso 4. Entrenar el baseline experimental

```bash
python3 -m src.models.train_model
```

Modelos incluidos:

- `DummyRegressor`
- `SGDRegressor`
- `HistGradientBoostingRegressor`

Salida esperada:

- RMSE de validation
- RMSE de test
- artefacto guardado en `MODEL_DIR`

## Paso 5. Ejecutar tests

```bash
python3 -m pytest
```

Los tests validan:

- configuracion
- contrato de features
- leakage
- entrenamiento minimo mock

Los tests no reemplazan la corrida real en Snowflake.

## Ejecucion por partes

Si no quieres usar `bootstrap`, el flujo corto por etapas es:

```bash
python3 -m src.data.ingestion setup
python3 -m src.data.ingestion ingest
python3 -m src.data.ingestion transform
python3 -m src.models.train_model
python3 -m pytest
```

Que hace cada comando:

- `setup`: crea esquemas y tabla raw dev
- `ingest`: descarga el parquet mensual y lo carga a Snowflake
- `transform`: construye OBT y splits y muestra conteos

## Notas operativas

- La ingesta automatica actual asume que `DATA_START_DATE` y `DATA_END_DATE` pertenecen al mismo mes.
- El foco de esta fase es experimentacion; API y frontend vienen despues.
- Si `bootstrap` falla, revisa primero el log del archivo SQL, del `PUT` o del `COPY INTO` que fallo.

## Siguiente validacion recomendada

Cuando `bootstrap` termine bien, el siguiente paso practico es abrir `01_eda.ipynb` y confirmar que la muestra de `ANALYTICS.OBT_TRIPS_DEV` ya tiene datos reales y distribuciones razonables.
