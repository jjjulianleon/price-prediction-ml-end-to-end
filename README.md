# Price Prediction ML End to End

Proyecto base para prediccion de precios de viajes NYC Yellow Taxi con enfoque Snowflake-first. El objetivo de esta fase es establecer una plataforma reproducible de ingenieria de datos y experimentacion de modelos sobre un rango de fechas controlado, evitando mover el procesamiento masivo fuera de Snowflake.

## Objetivo del proyecto

El repositorio organiza el flujo inicial para:

- cargar un subconjunto controlado de viajes Yellow Taxi en Snowflake
- construir una OBT de modelado con reglas de limpieza y control de leakage
- materializar conjuntos `train`, `validation` y `test` con logica temporal
- explorar datos mediante muestras
- entrenar y comparar modelos iniciales de regresion

En esta etapa el foco esta en la **experimentacion**. Las decisiones de API, frontend y serving se postergan hasta validar la estrategia de modelado.

## Alcance de la fase actual

- servicio inicial: `Yellow Taxi`
- target inicial: `fare_amount`
- rango de datos configurable por fechas en `.env`
- OBT y splits construidos en Snowflake
- entrenamiento local solo con muestras o batches

## Arquitectura de trabajo

1. Snowflake centraliza la carga dev, limpieza estructural, OBT y particiones temporales.
2. Los notebooks se usan para EDA y validacion en muestras, no para procesar la base completa.
3. Los scripts en `src/` encapsulan configuracion, acceso a Snowflake, transformaciones y entrenamiento.
4. Los tests validan contratos de codigo y supuestos del pipeline, pero no reemplazan la ejecucion de la data real en Snowflake.

## Dataset de modelado

La tabla de modelado `ANALYTICS.OBT_TRIPS_DEV` conserva solo variables seguras para prediccion pre-viaje:

- `pickup_datetime`
- `pickup_hour`
- `pickup_dayofweek`
- `pickup_month`
- `is_weekend`
- `passenger_count`
- `trip_distance`
- `pickup_location_id`
- `dropoff_location_id`
- `vendor_id`
- `ratecode_id`
- `fare_amount` como target

Se excluyen del contrato de features variables que introducen leakage o dependen del cierre del viaje, por ejemplo `payment_type`, `tip_amount`, `tolls_amount`, `mta_tax`, `airport_fee`, `total_amount` y derivadas post-viaje.

## Modelos incluidos en la base experimental

- `DummyRegressor` como baseline
- `SGDRegressor` para entrenamiento incremental por lotes
- `HistGradientBoostingRegressor` sobre muestra controlada

Los modelos boosting adicionales requeridos por el proyecto quedan pendientes para la fase comparativa.

## Estructura principal

```text
├── data/
│   ├── models/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_experimentation.ipynb
├── src/
│   ├── data/sql/
│   ├── data/ingestion.py
│   ├── features/
│   ├── models/
│   └── utils/
├── tests/
├── .env.example
├── RUNBOOK.md
├── ENUNCIADO.md
└── README.md
```

## Configuracion

La plantilla de entorno esta en [.env.example](/home/pabseb/DataMining/final-project/price-prediction-ml-end-to-end/.env.example:1). Las variables mas importantes son:

- `SNOWFLAKE_ACCOUNT`
- `SNOWFLAKE_USER`
- `SNOWFLAKE_PASSWORD`
- `SNOWFLAKE_ROLE`
- `SNOWFLAKE_WAREHOUSE`
- `SNOWFLAKE_DATABASE`
- `SNOWFLAKE_SCHEMA_RAW`
- `SNOWFLAKE_SCHEMA_ANALYTICS`
- `SNOWFLAKE_SCHEMA_ML`
- `DATA_START_DATE`
- `DATA_END_DATE`
- `TRAIN_END_DATE`
- `VAL_END_DATE`
- `MODEL_DIR`

Aunque la configuracion actual esta pensada para un solo mes, el diseño ya queda preparado para ampliar el rango de fechas sin reescribir el pipeline.

## Ejecucion

La guia operativa completa, orden de ejecucion y expectativas de salida estan en [RUNBOOK.md](/home/pabseb/DataMining/final-project/price-prediction-ml-end-to-end/RUNBOOK.md:1).

Comando recomendado para la fase base:

```bash
python3 -m src.data.ingestion bootstrap
```

Ese comando prepara la estructura en Snowflake, ingiere automaticamente el parquet mensual oficial de NYC TLC hacia `RAW.YELLOW_TRIPS_DEV` y luego materializa OBT y splits.
La ruta principal de carga usa `PUT` a un stage interno y `COPY INTO` directo en Snowflake.

## Estado de pruebas

Los tests del repositorio validan codigo y contratos del pipeline. No ejecutan por si solos la carga real en Snowflake ni reemplazan la corrida operativa del flujo. Primero debe existir la data dev en Snowflake; despues se ejecutan notebooks y entrenamiento experimental.

## Referencia academica

El enunciado original del proyecto se conserva en [ENUNCIADO.md](/home/pabseb/DataMining/final-project/price-prediction-ml-end-to-end/ENUNCIADO.md:1).
