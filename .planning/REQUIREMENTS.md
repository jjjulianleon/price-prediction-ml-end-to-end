# Requirements: NYC Taxi Price Prediction — End-to-End ML

**Defined:** 2026-05-05
**Core Value:** Usuario ingresa datos del viaje en la UI → recibe estimación de fare_amount vía modelo entrenado

## v1 Requirements

### Local Training

- [ ] **TRAIN-01**: Sistema puede descargar parquet NYC TLC de 1 mes directamente desde la CDN oficial sin Snowflake
- [ ] **TRAIN-02**: Script de entrenamiento local divide data en train/val/test con split temporal (no aleatorio)
- [ ] **TRAIN-03**: Entrenamiento local respeta el contrato de features sin leakage (mismas columnas que el pipeline Snowflake)
- [ ] **TRAIN-04**: Artefacto `model.joblib` se guarda en `data/models/` con modelo ganador + preprocessor + métricas

### Models

- [ ] **MODEL-01**: Pipeline incluye XGBoost como modelo boosting
- [ ] **MODEL-02**: Pipeline incluye LightGBM como modelo boosting
- [ ] **MODEL-03**: Pipeline incluye RandomForest como comparación ensemble
- [ ] **MODEL-04**: Selección automática del mejor modelo por RMSE de validación
- [ ] **MODEL-05**: Métricas de todos los modelos (val_rmse, test_rmse) quedan registradas en el artefacto

### API

- [ ] **API-01**: `TripInput` usa `pickup_location_id` y `dropoff_location_id` en lugar de coordenadas lat/lon
- [ ] **API-02**: `TripInput` excluye `payment_type` y cualquier variable de leakage
- [ ] **API-03**: Endpoint `/predict` carga el artefacto entrenado y retorna predicción real (no mock)
- [ ] **API-04**: Endpoint `/health` indica si el modelo está cargado o no
- [ ] **API-05**: La API maneja correctamente el formato de entrada: `pickup_datetime` + features del modelo

### Frontend

- [ ] **UI-01**: Formulario Streamlit usa `pickup_location_id` y `dropoff_location_id` (selectbox con IDs 1-265)
- [ ] **UI-02**: Formulario excluye `payment_type`, `store_and_fwd_flag` y otras variables de leakage
- [ ] **UI-03**: Formulario incluye `pickup_datetime` (date + time pickers), `trip_distance`, `passenger_count`, `vendor_id`, `ratecode_id`
- [ ] **UI-04**: Resultado de predicción se muestra claramente con el nombre del modelo usado
- [ ] **UI-05**: Errores de conexión con la API se muestran con mensaje amigable

### Integration

- [ ] **INT-01**: `src/api/main.py` carga automáticamente el artefacto desde `data/models/` al iniciar
- [ ] **INT-02**: Sistema completo corre con: `uvicorn src.api.main:app` + `streamlit run app/frontend.py`
- [ ] **INT-03**: `requirements.txt` incluye xgboost, lightgbm y todas las dependencias necesarias

## v2 Requirements

### Mejoras futuras

- **V2-01**: Modo Snowflake con credenciales activas (retomar cuando estén disponibles)
- **V2-02**: Mapeo automático de coordenadas GPS a location_id via lookup table
- **V2-03**: Múltiples meses de datos para mayor precisión
- **V2-04**: Reentrenamiento programado

## Out of Scope

| Feature | Reason |
|---------|--------|
| Autenticación de usuarios | Proyecto académico, no necesario |
| Despliegue cloud | Demo local suficiente para entrega |
| GPS → location_id automático | Complejidad innecesaria, user selecciona ID directamente |
| payment_type como feature | Leakage — penalización crítica en rúbrica |
| Coordenadas lat/lon en el modelo | El modelo ya usa location_id, cambiar sería reescribir el contrato |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| TRAIN-01 | Phase 1 | Pending |
| TRAIN-02 | Phase 1 | Pending |
| TRAIN-03 | Phase 1 | Pending |
| TRAIN-04 | Phase 1 | Pending |
| MODEL-01 | Phase 2 | Pending |
| MODEL-02 | Phase 2 | Pending |
| MODEL-03 | Phase 2 | Pending |
| MODEL-04 | Phase 2 | Pending |
| MODEL-05 | Phase 2 | Pending |
| API-01 | Phase 3 | Pending |
| API-02 | Phase 3 | Pending |
| API-03 | Phase 3 | Pending |
| API-04 | Phase 3 | Pending |
| API-05 | Phase 3 | Pending |
| UI-01 | Phase 3 | Pending |
| UI-02 | Phase 3 | Pending |
| UI-03 | Phase 3 | Pending |
| UI-04 | Phase 3 | Pending |
| UI-05 | Phase 3 | Pending |
| INT-01 | Phase 4 | Pending |
| INT-02 | Phase 4 | Pending |
| INT-03 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 22 total
- Mapped to phases: 22
- Unmapped: 0 ✓

---
*Requirements defined: 2026-05-05*
*Last updated: 2026-05-05 after initial definition*
