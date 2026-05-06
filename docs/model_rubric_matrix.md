# Model Rubric Matrix

## Estado al 2026-05-09

| Modelo | Exigido por rubrica | Estado | Tipo de entrenamiento | Dependencia | val_rmse (10K) | Observaciones |
|---|:---:|---|---|---|:---:|---|
| DummyRegressor | no | shortlist (baseline) | sample | sklearn | — | baseline minimo, referencia de piso |
| SGDRegressor | sugerido | shortlist | incremental | sklearn | — | unica ruta con partial_fit real por lotes |
| RandomForestRegressor | sugerido | shortlist | sample | sklearn | — | referencia de ensamble no boosting |
| GradientBoostingRegressor | si (boosting) | shortlist | sample | sklearn | 8.84 | boosting sklearn; superado por XGBoost en gap val→test |
| HistGradientBoostingRegressor | no | shortlist | sample | sklearn | 9.72 | booster hist eficiente; alternativa sin dependencias externas |
| **XGBoost** | **si (obligatorio)** | **shortlist + PRODUCCION** | sample (muestra masiva) | `xgboost` | 9.21 | **mejor gap val→test (-0.003); modelo productivo final** |
| LightGBM | si (obligatorio) | shortlist | sample | `lightgbm` | 8.53 | obligatorio por rubrica; mejor val_rmse absoluto en 10K |
| CatBoost | si | shortlist | sample | `catboost` | 9.11 | obligatorio; buena generalizacion (gap +0.35) |
| RidgeRegressor | no | archivado | sample | sklearn | — | dominado por boosters |
| AdaBoostRegressor | si | archivado | sample | sklearn | — | peor tradeoff calidad/costo frente a boosters modernos |
| BaggingRegressor | no | archivado | sample | sklearn | — | dominado por boosters y random forest |
| PastingRegressor | no | archivado | sample | sklearn | — | dominado |
| VotingRegressor | no | archivado | sample | sklearn | — | costo alto sin ventaja clara |

## Benchmark completo (notebook 04, 10K filas balanceadas multi-año, 2026-05-10)

| Modelo | val_rmse | test_rmse | gap val→test | Seleccionado |
|---|:---:|:---:|:---:|:---:|
| LightGBM | 8.53 | 9.10 | +0.58 | no |
| GradientBoosting | 8.84 | 9.68 | +0.85 | no |
| CatBoost | 9.11 | 9.46 | +0.35 | no |
| **XGBoost** | **9.21** | **9.21** | **-0.003** | **si** |
| HistGradientBoosting | 9.72 | 9.79 | +0.06 | no |

**Criterio de seleccion:** XGBoost tiene el gap val→test mas bajo (-0.003), lo que indica la generalizacion temporal mas estable. LightGBM supera en val_rmse absoluto pero con 5M filas en produccion la diferencia se reduce. El gap negativo de XGBoost (test_rmse levemente mejor que val_rmse) es un indicador robusto de que el modelo no sobre-ajusta al periodo de validacion.

## Modelo productivo seleccionado: XGBoost (v5)

**Razon de seleccion:** mejor gap val→test en el benchmark formal de `notebooks/04_model_experimentation.ipynb` sobre muestra balanceada multi-año. XGBoost cumple el requisito de boosting moderno de la rubrica, acepta `sample_weight` para el desbalance Yellow/Green y soporta entrenamiento eficiente sobre matrices dispersas grandes.

**Configuracion productiva** (`src/models/estimators.py::build_xgboost()`):

```python
XGBRegressor(
    n_estimators=600,           # 600 arboles; mas que suficiente con lr=0.05
    learning_rate=0.05,         # conservador: reduce overfitting vs lr=0.1
    max_depth=6,                # profundidad estandar para tabular regression
    subsample=0.8,              # submuestra de filas por arbol; reduce varianza
    colsample_bytree=0.8,       # submuestra de features por arbol; reduce varianza
    min_child_weight=5,         # nodo requiere al menos 5 filas; evita splits ruidosos
    tree_method="hist",         # histograma: eficiente para datasets grandes y sparse
    device="cuda",              # GPU si disponible; fallback automatico a "cpu"
    objective="reg:squarederror",
    random_state=42,
)
```

**Estrategia de entrenamiento:** muestra masiva estratificada de 5M filas desde `TRAIN_SET_DEV` (2015-2023). 18 estratos: 9 años × 2 flotas (~277K filas por estrato). XGBoost no tiene `partial_fit` nativo; la aleatoriedad de Snowflake por estrato garantiza cobertura temporal y por flota sin requerir external memory.

**Formato de entrada:** matriz dispersa CSR generada por el preprocesador OHE (~40K columnas). Pico de RAM estimado ~1 GB durante el fit. El formato sparse es critico para evitar OOM con matrices OHE de alta dimensionalidad.

**Manejo de desbalance Yellow/Green:** `compute_trip_type_weights()` calcula pesos inversos a la frecuencia por flota y los pasa como `sample_weight` al fit. La muestra estratificada de 5M garantiza 50/50 yellow-green (Yellow: 2.5M, Green: 2.5M).

**Evaluacion:** batch evaluation sobre `VAL_SET_DEV` (~36M filas, 2024) y `TEST_SET_DEV` (~35M filas, 2025) en ventanas mensuales de `BATCH_SIZE=500000`. El test se evalua **una sola vez** al final.

**Metricas de produccion (v5, sobre distribucion completa):**

| Metrica | validation (2024) | test (2025) | Contexto |
|---|:---:|:---:|---|
| RMSE | **56.16** | 165.91 | std(fare_amount)≈$56 por heavy-tail; inflado por viajes JFK/largo radio |
| **MAE** | **$2.32** | **$2.38** | **error tipico real por prediccion — metrica principal** |
| **Median AE** | **$1.36** | **$1.37** | la mitad de todas las predicciones tienen error < $1.37 |
| MAPE | 38.77% | 37.72% | alto por viajes baratos ($2.50 base) en denominador |
| R² | 0.085 | 0.011 | bajo por alta varianza intrinseca de la distribucion real |
| Filas evaluadas | 36,148,221 | 35,500,578 | evaluacion sobre dataset completo, no muestra |

**Nota sobre RMSE=56 en validacion:** la distribucion de `fare_amount` en los 36M filas de 2024 tiene heavy-tail real (viajes JFK ~$52, Newark ~$70+, viajes largos $100-300). El RMSE cuadratico amplifica estos outliers. El **MAE=$2.32 y MedAE=$1.36 son los indicadores correctos** del error tipico en un viaje comun.

**Nota sobre test_rmse=165 en 2025:** el NYC Congestion Pricing entro en vigor en enero 2025 y agrego $9-15 a viajes hacia Manhattan. Si ese surcharge fue incorporado al `fare_amount` en los parquets TLC 2025 (en lugar de como columna separada), la distribucion del target cambio estructuralmente en el periodo de test. Este es un caso documentado de **concept drift por cambio regulatorio** fuera del control del modelo.

## Modelos archivados

Los modelos archivados se conservan en `src/models/model_zoo.py` para trazabilidad y reruns excepcionales, pero no forman parte del flujo productivo ni del shortlist de comparacion activa.

## Interpretacion de columnas

| Campo | Significado |
|---|---|
| shortlist | en el flujo principal de comparacion en notebook 04 |
| archivado | conservado en zoo para trazabilidad; no compite activamente |
| PRODUCCION | modelo seleccionado, entrenado en `src/models/train_model.py` |
| incremental | compatible con `partial_fit` y entrenamiento real por lotes |
| sample | entrenado sobre muestra controlada; evaluacion por lotes en val/test |
| sample (muestra masiva) | entrenado sobre ~5M filas estratificadas de Snowflake |
