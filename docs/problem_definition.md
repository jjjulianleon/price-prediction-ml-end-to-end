# Problem Definition

## Contexto

El proyecto utiliza `NYC TLC Taxi Trip Record Data` para predecir la tarifa base de un viaje individual (`fare_amount`) como problema de regresion supervisada.

La arquitectura debe respetar un escenario Big Data:

- Snowflake absorbe la ingesta, limpieza estructural, OBT y splits temporales
- el EDA se realiza sobre muestras representativas
- el entrenamiento local parte de muestra controlada o entrenamiento por lotes

## Fuente de datos

- Fuente principal: `NYC TLC Trip Record Data`
- Servicios soportados: `Yellow Taxi`, `Green Taxi` o ambas flotas combinadas
- Unidad de analisis: un viaje individual

## Objetivo del modelo

Predecir `fare_amount` en un momento **antes de iniciar el viaje**.

## Variable objetivo

- Target oficial: `fare_amount`
- Metrica principal: `RMSE`

## Por que `fare_amount` y no `total_amount`

`total_amount` incorpora cargos y componentes conocidos solo al cierre o pago del viaje, por ejemplo propinas, peajes, recargos o condiciones posteriores al trayecto. Usarlo como target o como feature haria el problema menos interpretable y aumentaria el riesgo de leakage.

`fare_amount` representa mejor la tarifa base asociada al servicio y permite modelar un escenario de prediccion pre-viaje mas defendible.

## Momento de prediccion

El modelo simula una inferencia hecha antes del viaje. Por tanto, solo pueden usarse variables conocidas o estimables en ese instante.

## Variables conocidas antes del viaje

- `pickup_datetime`
- `trip_type`
- `pickup_location_id`
- `dropoff_location_id`
- `passenger_count`
- `vendor_id`
- `ratecode_id` cuando aplica
- `estimated_distance`

## Definicion de `estimated_distance`

En la data historica original existe `trip_distance`. Para esta fase, el contrato oficial de modelado la publica como `estimated_distance` en `STAGING`, `ANALYTICS` y `ML`.

Interpretacion:

- en entrenamiento historico, `estimated_distance` es un **proxy** construido desde `trip_distance`
- en serving, el usuario o sistema externo debe proporcionar una distancia estimada antes del viaje

Esta decision queda documentada para que el equipo no confunda una variable historica observada con una variable perfectamente conocida ex-ante.

## Variables excluidas por leakage

No deben usarse como features:

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

Estas columnas pueden servir para diagnostico, reglas de calidad o EDA, pero no para el modelo final.

## Splits temporales

Split final esperado por el proyecto:

- `train`: 2015 a 2023
- `validation`: 2024
- `test`: 2025

La implementacion base actual usa fechas parametrizadas desde `.env`, lo que permite reproducir primero una ventana de 6 meses y luego escalar al split final sin cambiar arquitectura.

## Estrategia Big Data

- Snowflake realiza el trabajo pesado de ingesta, tipado, limpieza y materializacion de datasets
- el EDA consume muestras, no tablas completas
- el entrenamiento usa una estrategia hibrida:
  - incremental real para algoritmos con `partial_fit`
  - muestra controlada para algoritmos sin entrenamiento out-of-core nativo
- `validation` selecciona modelo; `test` queda reservado para evaluacion final

## Hipotesis iniciales

- la distancia estimada es uno de los predictores mas fuertes del target
- el patron horario, semanal y mensual influye en la tarifa
- el par origen-destino aporta estructura espacial util
- un ensamblado boosting deberia superar al baseline simple y al baseline lineal

## Referencias

- NYC TLC Trip Record Data
- Documentacion oficial del curso y enunciado preservado en `ENUNCIADO.md`
- Documentacion de XGBoost, LightGBM y CatBoost para entrenamiento sobre muestra o lotes
