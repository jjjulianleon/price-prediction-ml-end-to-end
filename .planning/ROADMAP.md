# Roadmap: NYC Taxi Price Prediction — End-to-End ML

**Milestone:** v1 — Sistema funcional end-to-end para entrega académica
**Requirements coverage:** 22/22 ✓
**Phases:** 4

---

## Phase 1: Local Training Pipeline

**Goal:** El sistema puede entrenar modelos completos sin Snowflake descargando datos NYC TLC directamente.

**Requirements:** TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04

**Deliverables:**
- `src/models/train_local.py` — script de entrenamiento local completo
- `src/data/local_ingestion.py` — descarga parquet desde CDN y hace split temporal
- Artefacto `data/models/nyc_taxi_fare_local.joblib` generado tras correr el script
- `requirements.txt` actualizado con xgboost + lightgbm

**Success Criteria:**
1. `python -m src.models.train_local` descarga el parquet, entrena y guarda artefacto sin error
2. El split es temporal (train < val < test por fecha), no aleatorio
3. El artefacto contiene `model`, `preprocessor`, `metrics`, `model_name`
4. Las features del pipeline local son idénticas al contrato de features existente (sin leakage)

---

## Phase 2: Boosting Models

**Goal:** Pipeline de comparación incluye XGBoost, LightGBM y RandomForest además de los modelos existentes.

**Requirements:** MODEL-01, MODEL-02, MODEL-03, MODEL-04, MODEL-05

**Deliverables:**
- `src/models/model_zoo.py` — catálogo de todos los modelos con sus hiperparámetros
- `train_local.py` actualizado para comparar todos los modelos y elegir el mejor por val_rmse
- Tabla de métricas impresa al final del entrenamiento

**Plans:** 2 plans

Plans:
- [ ] 02-01-PLAN.md — Create model_zoo.py catalog and add xgboost/lightgbm to requirements.txt
- [ ] 02-02-PLAN.md — Update train_local.py to iterate MODEL_ZOO and save enriched artifact

**Success Criteria:**
1. El entrenamiento compara al menos 5 modelos: Dummy, SGD, RandomForest, XGBoost, LightGBM
2. El modelo con menor val_rmse se guarda como artefacto principal
3. Las métricas de todos los modelos quedan en el artefacto bajo `metrics.all_models`
4. XGBoost y LightGBM entrenan sin error con los datos del parquet local

---

## Phase 3: API + Frontend Alignment

**Goal:** La API y la UI usan exactamente las mismas features que el modelo — sin lat/lon, sin leakage.

**Requirements:** API-01, API-02, API-03, API-04, API-05, UI-01, UI-02, UI-03, UI-04, UI-05

**Deliverables:**
- `src/api/main.py` — `TripInput` corregido con `pickup_location_id`, `dropoff_location_id`, sin `payment_type`
- `src/models/predict_model.py` — función `predict` actualizada para manejar el nuevo formato de entrada
- `app/frontend.py` — formulario corregido con los inputs correctos (location IDs, sin leakage)

**Success Criteria:**
1. `TripInput` tiene exactamente los campos: `pickup_datetime`, `pickup_location_id`, `dropoff_location_id`, `passenger_count`, `trip_distance`, `vendor_id`, `ratecode_id`
2. No existe `payment_type` ni coordenadas lat/lon en el esquema de la API
3. El frontend muestra selectboxes para location_id (1-265), no campos de lat/lon
4. Una petición `POST /predict` con datos válidos retorna `estimated_fare_amount` numérico

---

## Phase 4: End-to-End Integration

**Goal:** Sistema completo corriendo: API carga el artefacto, Streamlit conecta a la API, predicción visible.

**Requirements:** INT-01, INT-02, INT-03

**Deliverables:**
- `src/api/main.py` — startup event carga artefacto desde `data/models/` automáticamente
- `run.sh` o instrucciones claras en README para levantar el sistema completo
- Prueba manual verificada: abrir Streamlit, llenar formulario, ver predicción real

**Success Criteria:**
1. `uvicorn src.api.main:app --reload` inicia sin error y `/health` retorna `{"model_loaded": true}`
2. `streamlit run app/frontend.py` abre la UI sin error
3. Llenando el formulario y enviando → la UI muestra un precio predicho (no mock)
4. El nombre del modelo ganador aparece en la respuesta (`"model": "xgboost"` o similar)

---

## Dependencies

```
Phase 1 → Phase 2 (Phase 2 extiende el pipeline de Phase 1)
Phase 1 → Phase 3 (Phase 3 necesita el artefacto para probar predict)
Phase 2 + Phase 3 → Phase 4 (integración final)
```

## Requirement → Phase Mapping

| Phase | Requirements |
|-------|-------------|
| 1 | TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04 |
| 2 | MODEL-01, MODEL-02, MODEL-03, MODEL-04, MODEL-05 |
| 3 | API-01, API-02, API-03, API-04, API-05, UI-01, UI-02, UI-03, UI-04, UI-05 |
| 4 | INT-01, INT-02, INT-03 |
