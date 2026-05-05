# NYC Taxi Price Prediction — End-to-End ML

## What This Is

Sistema end-to-end de predicción de tarifas para viajes NYC Yellow Taxi, construido como proyecto final de Data Mining. Toma variables de entrada pre-viaje (sin leakage), entrena múltiples modelos de regresión y sirve predicciones vía API FastAPI + UI Streamlit.

## Core Value

El usuario ingresa datos de un viaje en la UI y recibe una estimación del `fare_amount` — el sistema debe correr completo de extremo a extremo sin Snowflake para la demo.

## Requirements

### Validated

- ✓ Feature engineering sin leakage (`pickup_location_id`, `dropoff_location_id`, `trip_distance`, temporales) — existing
- ✓ Pipeline de entrenamiento modular en `src/models/train_model.py` — existing
- ✓ DummyRegressor + SGDRegressor (out-of-core) + HistGradientBoostingRegressor — existing
- ✓ FastAPI backend skeleton en `src/api/main.py` — existing
- ✓ Streamlit frontend skeleton en `app/frontend.py` — existing
- ✓ Snowflake ingestion pipeline en `src/data/ingestion.py` — existing

### Active

- [ ] Modo de entrenamiento local sin Snowflake (descarga parquet NYC TLC, split temporal, entrena, guarda artefacto)
- [ ] XGBoost y LightGBM agregados al pipeline de comparación de modelos
- [ ] RandomForest como comparación adicional de ensemble
- [ ] API `TripInput` alineada con features reales del modelo (location_id en lugar de lat/lon, sin leakage payment_type)
- [ ] Frontend Streamlit actualizado con inputs correctos que coincidan con el modelo
- [ ] Sistema corriendo end-to-end: `uvicorn` + `streamlit` → predicción visible en browser

### Out of Scope

- Autenticación de usuarios — proyecto académico
- Despliegue en producción (cloud) — demo local es suficiente
- Múltiples meses de datos — 1 mes es el alcance actual
- Coordenadas GPS → location_id mapping automático — complejidad no justificada para el demo

## Context

- **Base de código existente**: repo clonado con estructura completa, tiene entrenamiento funcional pero conectado a Snowflake (credenciales expiradas)
- **Problema crítico identificado**: la API y el frontend usan `pickup_longitude/latitude` y `payment_type` (leakage), pero el modelo usa `pickup_location_id`/`dropoff_location_id` sin payment_type — están desconectados
- **Compañero de grupo**: ya entrenó un modelo SGD out-of-core sobre 1 mes. Solo falta producción y UI
- **Entorno**: Linux, Python 3, conda/venv, FastAPI + Streamlit
- **Penalizaciones críticas a evitar**: data leakage en features, carga completa en pandas, omitir boosting

## Constraints

- **Snowflake**: Sin credenciales activas — modo local obligatorio para poder desarrollar y demostrar
- **Timeline**: Proyecto académico — entrega inmediata, prioridad sobre elegancia
- **Rubrica**: Exige modelos boosting (XGBoost, LightGBM) como obligatorios; HistGradientBoosting ya existe
- **Data**: NYC TLC Yellow Taxi parquet mensual (~400MB por mes, descargable desde CDN oficial)
- **Anti-leakage**: payment_type, tip_amount, tolls_amount, total_amount nunca deben entrar al modelo

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Modo local sin Snowflake | Credenciales expiradas, necesitamos poder entrenar y demostrar | — Pending |
| Location ID en lugar de lat/lon | El modelo ya usa location_id — mantener consistencia, no agregar complejidad de geocoding | — Pending |
| XGBoost + LightGBM + RandomForest | Rubrica pide boosting; RandomForest añade punto de comparación ensemble sin costo | — Pending |
| 1 mes de datos para training local | Equilibrio entre tiempo de descarga y representatividad del modelo | — Pending |

## Evolution

Este documento evoluciona en cada transición de fase y al completar milestones.

**Después de cada fase:**
1. ¿Requisitos invalidados? → Mover a Out of Scope con razón
2. ¿Requisitos validados? → Mover a Validated con referencia de fase
3. ¿Nuevos requisitos emergieron? → Agregar a Active
4. ¿Decisiones a registrar? → Agregar a Key Decisions

---
*Last updated: 2026-05-05 after initialization*
