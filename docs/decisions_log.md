# Decisions Log

## Decisiones Cerradas

### 1. Target oficial: `fare_amount`

Se descarta `total_amount` como target o feature porque incorpora informacion posterior al viaje y al pago.

### 2. Momento de prediccion: pre-viaje

Todo el contrato de features queda restringido a variables conocidas o estimables antes de iniciar el viaje.

### 3. `estimated_distance` como contrato publico

La variable de distancia historica `trip_distance` se publica como `estimated_distance` en capas de modelado. Esto permite mantener coherencia entre entrenamiento historico y serving real.

### 4. Arquitectura Snowflake-first

La limpieza estructural, la OBT y los splits temporales viven en SQL. Pandas y notebooks se reservan para muestra, validacion y experimentacion.

### 5. Split temporal por fechas parametrizadas

La base operativa actual usa fechas configurables desde `.env`. El split final del curso `2015-2023 / 2024 / 2025` se alcanzara cambiando configuracion, no reescribiendo el pipeline.

### 6. Entrenamiento hibrido

No todos los modelos soportan `partial_fit`. Por eso el proyecto distingue entre:

- entrenamiento incremental real
- entrenamiento sobre muestra controlada

### 7. `RAW` append-only por defecto

No se borra `RAW` por defecto. La ingesta debe ser idempotente y registrar archivo, periodo y estado de carga.

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
