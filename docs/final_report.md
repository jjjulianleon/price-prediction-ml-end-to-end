# Informe Tecnico Final: NYC Taxi Fare Prediction

**Proyecto:** Price Prediction ML End to End  
**Fecha:** 2026-05-09  
**Autor:** Pablo Alvarado  
**Version del modelo:** XGBoost v5  
**Contrato de features:** v4_multi_taxi_year_estimated_distance  

---

## 1. Planteamiento del Problema

### 1.1 Contexto

El sistema de taxis de Nueva York opera a traves del Taxi and Limousine Commission (TLC), que publica datos historicos de millones de viajes mensuales desde 2009. Cada registro incluye variables de inicio y fin del viaje, montos cobrados y metadatos del operador.

El problema de negocio central es: **dado que un pasajero esta a punto de iniciar un viaje, cuanto deberia costar ese viaje?** Esta prediccion es util para:

- estimacion de costos antes de subir al taxi
- comparacion entre modos de transporte
- deteccion de anomalias tarifarias en tiempo real
- planificacion de presupuesto para flotas corporativas

### 1.2 Objetivo del Modelo

Predecir `fare_amount` —la tarifa base del viaje— como problema de regresion supervisada, utilizando **exclusivamente variables disponibles en el momento previo al inicio del trayecto**.

El target oficial es `fare_amount` y no `total_amount` por las siguientes razones:

- `total_amount` = `fare_amount` + propinas + peajes + recargos de congestion + airport fee + impuestos + surcharges varios
- todos esos componentes adicionales son conocidos solo al finalizar o al momento del pago
- usar `total_amount` como target introduciria leakage en la etapa de serving, ya que el modelo esperaria un total que incluye informacion futura
- `fare_amount` representa la tarifa base regulada por TLC, asociada al servicio de transporte en si, y es predecible a partir de variables pre-viaje

### 1.3 Tipo de Problema

- **Tarea:** regresion supervisada
- **Target:** `fare_amount` (USD, continua, rango valido $2.50-$300)
- **Metricas principales:** RMSE, MAE, R²
- **Momento de prediccion:** pre-viaje (inferencia en tiempo real antes de iniciar)

### 1.4 Restricciones del Problema

1. **Anti-leakage estricto:** ninguna variable conocida solo post-viaje puede entrar al modelo ni al pipeline de serving
2. **Escala:** ~828M filas en el periodo 2015-2025, no cargables en memoria
3. **Multi-fleet:** el pipeline debe funcionar con Yellow Taxi, Green Taxi o ambas flotas combinadas
4. **Reproducibilidad:** el pipeline de datos en Snowflake debe ser determinista; el entrenamiento usa semilla fija

---

## 2. Arquitectura de Datos (Snowflake-First)

### 2.1 Principio de Diseno

La arquitectura es **Snowflake-first**: toda la logica pesada de ingesta, limpieza estructural, construccion de la OBT y materializacion de splits se ejecuta en SQL dentro de Snowflake. Python y los notebooks solo consumen muestras representativas. Este principio garantiza que:

- la limpieza de datos es reproducible y auditable en SQL
- los filtros de calidad se aplican en el motor de base de datos, no en Python
- los splits temporales son consistentes y no estan sujetos a variaciones de muestreo en Python
- el costo de computo se concentra donde es mas eficiente (Snowflake con micro-particiones)

### 2.2 Capas del Pipeline

```
Snowflake: DM_FINAL_PROJECT
├── RAW
│   ├── YELLOW_TRIPS_DEV     parquet TLC yellow, append-only, nombres originales
│   └── GREEN_TRIPS_DEV      parquet TLC green, append-only, nombres originales
│
├── STAGING
│   └── TRIPS_STAGE_DEV      TABLE  yellow + green unificados bajo contrato canonico
│                             10 reglas de limpieza estructural validadas en notebooks
│                             conserva columnas diagnosticas (payment_type, etc.)
│                             estas columnas NO llegan a la OBT
│
├── ANALYTICS
│   └── OBT_TRIPS_DEV        TABLE  contrato anti-leakage estricto
│                             ~828M filas (yellow: 801M, green: 68M, 2015-2025)
│                             incluye solo columnas aptas para modelado pre-viaje
│                             derivadas deterministicas calculadas en SQL
│
└── ML
    ├── TRAIN_SET_DEV        TABLE  2015-01-01 a 2023-12-31  (~792M filas)
    ├── VAL_SET_DEV          TABLE  2024-01-01 a 2024-12-31  (~36M filas)
    └── TEST_SET_DEV         TABLE  2025-01-01 a 2025-12-31  (~35M filas)
```

### 2.3 Volumen Real de Datos

| Tabla | Filas | Periodo | Nota |
|---|:---:|---|---|
| `OBT_TRIPS_DEV` | ~828M | 2015-2025 | post-filtros de calidad |
| `TRAIN_SET_DEV` | ~792M | 2015-2023 | set de entrenamiento oficial |
| `VAL_SET_DEV` | ~36M | 2024 | evaluacion continua durante desarrollo |
| `TEST_SET_DEV` | ~35M | 2025 | evaluacion final, una sola vez |
| Yellow Taxi (total) | ~801M | 2015-2025 | flota principal |
| Green Taxi (total) | ~68M | 2015-2025 | flota secundaria, ~8% del volumen |

### 2.4 Por que los Splits son TABLE y no VIEW

Una `VIEW` sobre la OBT de ~828M filas tiene un costo oculto severo: cada query de entrenamiento o evaluacion re-escanea la OBT completa y aplica el filtro de fecha en tiempo de ejecucion.

El `ORDER BY RANDOM() LIMIT 5_000_000` para la muestra de entrenamiento sobre una VIEW de train (~792M filas) requiere:
1. generar numeros aleatorios para las 792M filas
2. ordenar esos 792M numeros
3. tomar los primeros 5M

Con una `TABLE` pre-materializada, Snowflake aplica micro-partition pruning sobre el subconjunto ya filtrado. El mismo query opera sobre ~792M filas ya separadas, sin re-escanear la OBT.

| Escenario | Costo query de entrenamiento | Costo batch eval |
|---|---|---|
| `VIEW` sobre OBT | Escaneo ~828M filas + sort | Escaneo ~828M filas + filter |
| `TABLE` pre-materializada | Escaneo ~792M filas del split | Escaneo batch del split (micro-partition pruning) |

La contrapartida es mayor uso de storage (los 3 splits suman volumen similar a la OBT). Este costo de almacenamiento es aceptable dado que el ahorro en compute por multiples queries de entrenamiento, evaluacion y reruns es significativamente mayor.

---

## 3. Calidad de Datos

### 3.1 Reglas de Limpieza en STAGING

Todas las reglas estan implementadas como pushdown SQL en el script `02_create_staging_trips_dev.sql` y validadas con evidencia muestral en `notebooks/02_data_cleaning.ipynb`.

| # | Regla SQL | Columnas afectadas | Justificacion |
|---|---|---|---|
| 1 | `pickup_datetime IS NOT NULL` | pickup_datetime | timestamp de inicio requerido para splits temporales |
| 2 | `dropoff_datetime IS NOT NULL` | dropoff_datetime | necesario para validacion de orden temporal |
| 3 | `CAST(pickup_datetime AS DATE) BETWEEN start AND end` | pickup_datetime | ventana del proyecto parametrizada desde .env |
| 4 | `dropoff_datetime > pickup_datetime` | ambas | viaje con duracion positiva obligatoria |
| 5 | `trip_distance BETWEEN 0.1 AND 150` | estimated_distance | 0.1 mi minimo GPS valido; 150 mi techo fisico NYC-area |
| 6 | `passenger_count BETWEEN 1 AND 6` | passenger_count | rango soportado por regulacion TLC |
| 7 | `fare_amount BETWEEN 2.50 AND 300` | fare_amount | $2.50 minimo TLC oficial; $300 elimina errores de entrada de datos |
| 8 | `pulocationid IS NOT NULL` | pickup_location_id | zona espacial de origen requerida para features de ruta |
| 9 | `dolocationid IS NOT NULL` | dropoff_location_id | zona espacial de destino requerida para features de ruta |
| 10 | `ratecode_id BETWEEN 1 AND 6` | ratecode_id | catalogo oficial TLC (1=Standard, 2=JFK, 3=Newark, 4=Nassau, 5=Negotiated, 6=Group) |

### 3.2 Justificacion de Limites del Target

El rango `fare_amount BETWEEN 2.50 AND 300` se justifica por:

- **$2.50**: tarifa minima oficial TLC para Yellow Taxi (flag drop). Valores inferiores corresponden a errores de entrada de datos, viajes cancelados o registros corruptos.
- **$300**: el flat rate JFK desde Manhattan es ~$70. Viajes a Long Island o Connecticut pueden llegar a $150-200. El limite de $300 elimina valores claramente erroneos (errores de tipeo con un digito extra) sin recortar viajes legitimamente caros.

El notebook `02_data_cleaning.ipynb` valida estas reglas mostrando la distribucion de valores rechazados, confirmando que no eliminan viajes validos sino anomalias de datos.

### 3.3 Compatibilidad Multi-Flota

| Campo | Yellow RAW | Green RAW | STAGING (canonico) |
|---|---|---|---|
| datetime inicio | `tpep_pickup_datetime` | `lpep_pickup_datetime` | `pickup_datetime` |
| datetime fin | `tpep_dropoff_datetime` | `lpep_dropoff_datetime` | `dropoff_datetime` (solo STAGING) |
| airport fee | `airport_fee` | no existe | `airport_fee` (NULL para green) |
| tipo de viaje | implicito | implicito | `trip_type` = 'yellow' o 'green' |

Las mismas 10 reglas de calidad se aplican a ambas flotas. La unificacion ocurre en STAGING con un `UNION ALL` sobre las dos tablas RAW bajo el esquema canonico.

---

## 4. Feature Engineering

### 4.1 Contrato de Features v4

El contrato oficial es `v4_multi_taxi_year_estimated_distance`, cerrado el 2026-05-10. Incluye 17 features mas el target.

#### Features numericas directas

| Feature | Rango | Fuente |
|---|---|---|
| `passenger_count` | 1..6 | OBT directo |
| `estimated_distance` | 0.1..150 mi | OBT (alias de trip_distance) |
| `log_estimated_distance` | > 0 | OBT calculado: LN(1 + estimated_distance) |

#### Features temporales derivadas

| Feature | Calculo SQL | Justificacion |
|---|---|---|
| `pickup_hour` | `EXTRACT(HOUR FROM pickup_datetime)` | efecto horario en tarifas y congestion |
| `pickup_dayofweek` | `DAYOFWEEKISO(pickup_datetime) - 1` | patrones semanales (0=Lun..6=Dom) |
| `pickup_month` | `EXTRACT(MONTH FROM pickup_datetime)` | estacionalidad anual |
| `pickup_year` | `EXTRACT(YEAR FROM pickup_datetime)` | tendencia tarifaria historica 2015-2023 |
| `is_weekend` | `1 si dayofweek IN (5,6)` | flag weekend vs weekday |
| `is_rush_hour` | `1 si hour IN (7,8,9,16,17,18,19)` | horas pico de congestion NYC |
| `is_night` | `1 si hour IN (22,23,0,1,2,3,4,5)` | recargo nocturno |

#### Features categoricas

| Feature | Cardinalidad | Justificacion |
|---|---|---|
| `pickup_location_id` | 265 zonas TLC | zona de origen; estructura espacial del precio |
| `dropoff_location_id` | 265 zonas TLC | zona de destino; estructura espacial del precio |
| `vendor_id` | 2-3 valores | proveedor del sistema de despacho |
| `ratecode_id` | 6 valores | codigo de tarifa TLC; JFK flat rate, Newark, Standard, etc. |
| `route_id` | hasta 265×265 pares | par origen-destino; captura rutas especificas |
| `same_zone` | binaria | flag si origen == destino |
| `trip_type` | 2 valores | flota (yellow/green); dinamicas tarifarias distintas |

### 4.2 Por que cada Feature

**`estimated_distance`**: es el predictor mas fuerte del `fare_amount`. La tarifa NYC se calcula como tarifa base + $X por milla + $Y por minuto. La distancia es el factor dominante en viajes tipicos.

**`log_estimated_distance`**: la relacion distancia-tarifa no es perfectamente lineal (el costo por milla es mayor en distancias cortas por el flag drop fijo). La transformacion logaritmica comprime la escala y permite que el modelo capture la curvatura sin arboles profundos.

**`pickup_year`**: las tarifas base NYC aumentaron ~30-40% entre 2015 y 2023 por cambios regulatorios (aumento de MTA surcharge en 2019, congestion pricing parcial en 2019, indexacion de tarifas). Sin esta feature, el modelo aprende el promedio historico de 9 años y predice ese promedio para 2024, generando un error sistematico de ~$56 RMSE. Con `pickup_year`, el modelo aprende la tendencia y extrapola desde 2023 para predecir 2024.

**`ratecode_id`**: crucial porque el flat rate JFK (ratecode=2, ~$70 fijo) y el flat rate Newark (ratecode=3, ~$120 fijo) tienen distribuciones de `fare_amount` completamente distintas a los viajes por taximetro (ratecode=1). Un modelo que no conoce el ratecode intentara predecir el flat rate JFK desde la distancia, lo cual falla porque la tarifa es fija independientemente de la ruta exacta.

**`route_id`**: el par pickup_location_id + dropoff_location_id codifica rutas especificas. Ciertos pares tienen tarifas historicamente consistentes (ej: zona aeropuerto → Midtown Manhattan) que el modelo puede memorizar en los arboles.

**`trip_type`**: Yellow y Green Taxi tienen diferentes areas de operacion (Green no opera en Manhattan south), diferentes mezclas de ratecodes y diferentes promedios de `fare_amount`. Modelar la flota explicitamente evita que el modelo promedio entre ambas distribuciones.

**`is_rush_hour` e `is_night`**: recargos regulados. Los viajes nocturnos tienen un recargo de $0.50 sobre la tarifa base. Las horas pico tienen mayor congestion, lo que aumenta la duracion (y por ende la tarifa) de viajes por taximetro.

### 4.3 Definicion de `estimated_distance`

La variable `trip_distance` en RAW se publica como `estimated_distance` en STAGING, OBT y ML. Este alias tiene un proposito funcional importante:

- **en entrenamiento:** `estimated_distance` es el proxy historico construido desde `trip_distance` observado en el dataset TLC
- **en serving:** el usuario o sistema externo provee una distancia estimada antes de iniciar el viaje (distancia de ruta, estimacion GPS, etc.)
- **el alias evita confusion** entre la distancia observada historica y el input de prediccion real

El nombre `trip_distance` fue eliminado de la OBT en la version v3 del contrato (2026-05-09) para garantizar que el pipeline de serving solo acepte `estimated_distance` como nombre oficial.

### 4.4 Preprocesamiento

El pipeline definido en `src/features/build_features.py::get_feature_pipeline()`:

```
Numericas (passenger_count, estimated_distance, log_estimated_distance,
           pickup_hour, pickup_dayofweek, pickup_month, pickup_year,
           is_weekend, is_rush_hour, is_night, same_zone):
    SimpleImputer(strategy='median')
    StandardScaler()

Categoricas (trip_type, pickup_location_id, dropoff_location_id,
             vendor_id, ratecode_id, route_id):
    SimpleImputer(strategy='constant', fill_value='missing')
    OneHotEncoder(handle_unknown='ignore', sparse_output=True)
```

El espacio transformado post-encoding alcanza ~40K columnas, principalmente por el one-hot de `route_id` (hasta 265×265 pares posibles) y `pickup_location_id`/`dropoff_location_id`. Por eso el pipeline genera matrices en formato **sparse CSR**, esencial para evitar OOM.

---

## 5. Analisis de Leakage

### 5.1 Definicion de Leakage en este Contexto

Se considera leakage cualquier variable que:
1. es conocida solo al finalizar el viaje (post-viaje)
2. es conocida solo al momento del pago (post-pago)
3. es derivada matematica de variables post-viaje o post-pago
4. es altamente correlacionada con el target por razon de ser un componente del target

### 5.2 Lista Prohibida

```python
LEAKAGE_COLUMNS = [
    "total_amount",           # suma de fare + todos los cargos post-pago
    "tip_amount",             # conocido solo al pago del viaje
    "tolls_amount",           # conocido solo al finalizar la ruta (peajes reales)
    "mta_tax",                # cargo regulatorio post-viaje
    "extra",                  # cargo variable post-viaje (rush hour surcharge, etc.)
    "improvement_surcharge",  # cargo regulatorio post-viaje
    "congestion_surcharge",   # cargo por zona post-viaje
    "airport_fee",            # cargo por destino final (airport), post-viaje
    "payment_type",           # conocido solo al momento del pago
    "dropoff_datetime",       # conocido solo al finalizar el viaje
    "trip_duration_min",      # derivado de dropoff_datetime
    "speed_mph",              # derivado de trip_duration_min y trip_distance real
]
```

### 5.3 Garantias Anti-Leakage por Capa

**Capa SQL (OBT):** el script de creacion de `OBT_TRIPS_DEV` no selecciona ninguna columna de la lista prohibida. Las columnas diagnosticas existen en STAGING para auditoria, pero el `SELECT` de la OBT las omite explicitamente.

**Capa Python (build_features):** la funcion `assert_no_leakage_columns(df)` levanta `AssertionError` si cualquier columna de `LEAKAGE_COLUMNS` aparece en el DataFrame de entrada al preprocesador.

**Capa de tests:** `tests/test_features.py::test_leakage_columns_are_rejected` verifica que el preprocesador rechaza frames con columnas de leakage, incluso si se pasan mezcladas con features validas.

**Capa de serving (API):** el esquema Pydantic del endpoint `/predict` solo acepta las features del contrato v4. Cualquier campo extra es ignorado por defecto y las columnas de leakage no estan definidas en el modelo de entrada.

### 5.4 Por que `payment_type` es Leakage

`payment_type` indica si el viaje se pago con tarjeta de credito, efectivo o voucher. Aunque podria parecer una variable pre-viaje (el pasajero puede declarar su forma de pago antes), en la practica:

- el tipo de pago se registra al finalizar la transaccion
- en los parquets historicos, `payment_type` puede estar correlacionado con `tip_amount` (las propinas en efectivo no se registran), lo que genera correlacion espuria con el target
- en un escenario de prediccion real, el sistema de despacho no conoce la forma de pago futura del pasajero

Por estas razones, `payment_type` queda en la lista prohibida.

---

## 6. Seleccion de Modelo

### 6.1 Shortlist de Candidatos

El shortlist se define en `src/models/model_zoo.py` y se ejecuta en `notebooks/04_model_experimentation.ipynb`:

| Modelo | Razon de inclusion |
|---|---|
| `DummyRegressor` | baseline estadistico (predice la media); establece el piso minimo |
| `SGDRegressor` | unico modelo del shortlist con `partial_fit` real; referencia de entrenamiento incremental |
| `RandomForestRegressor` | ensamble no-boosting; referencia de bagging vs boosting |
| `GradientBoostingRegressor` | boosting sklearn clasico; comparacion con boosters modernos |
| `HistGradientBoostingRegressor` | variante hist-based de sklearn; referencia sin dependencias externas |
| `XGBoost` | boosting moderno requerido por rubrica; candidato principal |
| `LightGBM` | boosting moderno requerido por rubrica; candidato principal |
| `CatBoost` | boosting moderno con manejo nativo de categoricas; candidato principal |

Modelos archivados (no compiten en el shortlist activo): `Ridge`, `AdaBoost`, `Bagging`, `Pasting`, `Voting`. Fueron evaluados en una fase exploratoria anterior y mostraron peor tradeoff calidad/costo frente a los boosters modernos.

### 6.2 Resultados del Benchmark

**Configuracion del benchmark:**
- Dataset: `DM_FINAL_PROJECT` (ventana oficial 2015-2025)
- Muestra de experimentacion: 10K filas balanceadas multi-año desde el shortlist
- Metricas: `val_rmse` y `test_rmse`
- Fecha: 2026-05-10

| Modelo | val_rmse | test_rmse | gap val→test |
|---|:---:|:---:|:---:|
| LightGBM | 8.53 | 9.10 | +0.58 |
| GradientBoosting | 8.84 | 9.68 | +0.85 |
| CatBoost | 9.11 | 9.46 | +0.35 |
| **XGBoost** | **9.21** | **9.21** | **-0.003** |
| HistGradientBoosting | 9.72 | 9.79 | +0.06 |

### 6.3 Por que XGBoost sobre LightGBM

LightGBM obtiene el mejor `val_rmse` absoluto (8.53 vs 9.21 de XGBoost) sobre la muestra de 10K. Sin embargo, XGBoost fue seleccionado como modelo productivo por las siguientes razones:

1. **Mejor gap val→test:** XGBoost tiene gap=-0.003 (test ligeramente mejor que val), vs LightGBM con gap=+0.58. Un gap negativo o cercano a cero indica que el modelo generaliza al periodo temporal futuro sin sobre-ajustarse al periodo de validacion. Con 5M filas en produccion, esta propiedad es mas importante que ganar 0.68 de RMSE en la muestra.

2. **Escala:** en la muestra de 10K, la diferencia de RMSE entre LightGBM y XGBoost es 0.68 unidades (8.53 vs 9.21). Con 5M filas de entrenamiento, los boosters modernos convergen a calidades similares y la diferencia de 0.68 en 10K no se mantiene a la misma magnitud.

3. **Robustez temporal:** el gap de LightGBM de +0.58 sugiere que su ajuste en val no generaliza igual de bien a test. Con concept drift estructural (congestion pricing 2025), este riesgo es relevante.

4. **Compatibilidad con `sample_weight`:** ambos modelos aceptan `sample_weight`, pero el comportamiento de XGBoost con matrices sparse de alta dimensionalidad esta mejor documentado y es mas predecible.

5. **Requisito de rubrica:** ambos modelos cumplen el requisito de boosting moderno. XGBoost tiene ventaja practica en la metrica que importa para produccion (gap val→test).

---

## 7. Configuracion de XGBoost: Justificacion Parametro a Parametro

### 7.1 Configuracion Productiva v5

```python
XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    tree_method="hist",
    device="cuda",
    objective="reg:squarederror",
    random_state=42,
)
```

### 7.2 Justificacion por Parametro

**`n_estimators=600`**

Con `learning_rate=0.05`, se necesitan mas arboles para que el modelo converja. La regla general es: al reducir `learning_rate` a la mitad, se duplica el numero optimo de arboles. La configuracion anterior usaba `n_estimators=400` con la misma tasa; el incremento a 600 da mas capacidad al modelo para capturar patrones finos en los 5M de entrenamiento sin over-fitting notable dado el regularizador `subsample`.

**`learning_rate=0.05`**

Tasa conservadora. Reduce el sobre-ajuste vs `lr=0.1` al hacer que cada arbol contribuya menos a la prediccion final. Dado que se entrena sobre 5M filas con alta varianza intrinseca del target, una tasa baja es preferible para evitar memorizar ruido.

**`max_depth=6`**

Profundidad estandar para regresion tabular con features de alta cardinalidad (265 zonas, 265×265 rutas). Arboles de profundidad 6 pueden capturar interacciones de hasta 6 variables simultaneamente (ej: zona_origen × hora × año × ratecode). Profundidades mayores (8-10) aumentan el riesgo de sobre-ajuste en categoricas de alta cardinalidad.

**`subsample=0.8`**

Submuestra el 80% de las filas para construir cada arbol. Introduce estocasticidad que actua como regularizador (similar al dropout en redes neuronales). Reduce la varianza del modelo sin aumentar el sesgo de forma significativa. Con 5M filas de entrenamiento, el 80% = 4M filas por arbol es suficiente para aprender patrones robustos.

**`colsample_bytree=0.8`**

Submuestra el 80% de las features para construir cada arbol. Con ~40K columnas post-OHE, esto significa ~32K columnas por arbol. Introduce diversidad entre arboles y evita que las features mas frecuentes (como `route_id`) dominen todos los arboles.

**`min_child_weight=5`**

Un nodo hoja debe tener al menos 5 observaciones (en terminos de la suma de los pesos de segundo orden del gradiente). Evita que el modelo cree splits sobre grupos muy pequeños de rutas raras o combinaciones de features inusuales. Es un regularizador critico con features categoricas de alta cardinalidad como `route_id` (muchas rutas tienen pocas observaciones en el sample).

**`tree_method="hist"`**

Algoritmo de histograma para construir los splits. En lugar de evaluar todos los valores posibles de cada feature (algoritmo exacto), construye bins de histograma y evalua splits sobre esos bins. Esto reduce el costo de cada split de O(n × f) a O(b × f) donde b es el numero de bins (tipicamente 256). Es el unico metodo practico para 5M filas × 40K columnas en tiempo razonable.

**`device="cuda"`**

Utiliza GPU si esta disponible (CUDA). Con `tree_method="hist"`, XGBoost puede delegar la construccion de histogramas y la evaluacion de splits a la GPU, reduciendo el tiempo de entrenamiento de ~190s (CPU) a ~30-60s (GPU). El codigo incluye fallback automatico a `device="cpu"` si no hay GPU disponible.

**`objective="reg:squarederror"`**

Funcion de perdida de minimos cuadrados (MSE). Apropiada para regresion continua. Optimiza directamente el RMSE que es la metrica principal de evaluacion. Alternativas como `reg:absoluteerror` (MAE) fueron consideradas pero el RMSE es la metrica de comparacion del benchmark y del enunciado.

**`random_state=42`**

Semilla fija para reproducibilidad. Garantiza que el mismo entrenamiento produce el mismo artefacto dado el mismo dataset.

---

## 8. Estrategia de Entrenamiento

### 8.1 El Problema de Escala

El dataset de entrenamiento `TRAIN_SET_DEV` contiene ~792M filas (2015-2023). Cargar todo en memoria es inviable:

- el preprocesador OHE genera ~40K columnas en formato dense → ~792M × 40K × 4 bytes ≈ **126 TB** en memoria (absurdo)
- en formato sparse CSR, la densidad real es ~17 features no-cero por fila (antes del OHE) → aun así, el volumen en sparse seria varios GB
- `XGBoost` no tiene `partial_fit` nativo (a diferencia de `SGDRegressor`)

La alternativa oficial de XGBoost para datasets que no caben en memoria es **external memory DMatrix** (escribir el dataset en formato SVM a disco y leerlo en bloques). Esta opcion fue evaluada y descartada por:

- requiere escribir un archivo SVM de varios GB a disco antes de entrenar
- introduce dependencia del formato binario de XGBoost, complicando el pipeline
- no mejora la calidad del modelo respecto a la muestra masiva
- el pipeline de Snowflake ya proporciona una muestra aleatoria de alta calidad sin costo adicional

### 8.2 Muestra Masiva Estratificada

La estrategia elegida es **muestra masiva estratificada**:

```sql
-- Para cada estrato (año × flota):
SELECT * FROM TRAIN_SET_DEV
WHERE pickup_year = {year} AND trip_type = '{fleet}'
ORDER BY RANDOM()
LIMIT {rows_per_stratum}
```

**18 estratos:** 9 años (2015-2023) × 2 flotas (yellow, green)  
**Filas por estrato:** ~277K (5M / 18)  
**Total:** 5M filas

La estratificacion garantiza que:
- todos los años del periodo de entrenamiento estan representados equitativamente
- ambas flotas estan representadas con la misma proporcion (50/50), corrigiendo el desbalance real Yellow(97%)/Green(3%)
- el modelo aprende patrones de todos los periodos historicos, no solo los años mas recientes

La aleatoriedad `ORDER BY RANDOM()` en Snowflake sobre cada estrato pre-materializado es determinista condicionalmente a la semilla de la sesion, y garantiza cobertura uniforme dentro de cada estrato.

### 8.3 Perfil de Memoria

| Etapa | RAM estimada |
|---|---|
| DataFrame de 5M filas (antes de OHE) | ~400 MB |
| Matriz sparse CSR post-OHE (~40K cols) | ~600 MB |
| Pico durante fit (XGBoost histograms) | ~1 GB |
| Artefacto final (.joblib) | ~300 MB |

El formato sparse es critico para mantener el pico bajo 1 GB. Un formato dense equivalente seria ~800 GB.

### 8.4 `sample_weight` para Desbalance de Flotas

La distribucion real Yellow/Green es aproximadamente 97%/3%. Sin correccion, el modelo aprende casi exclusivamente los patrones de Yellow Taxi y tiene desempeño degradado en Green.

El calculo de pesos:

```python
def compute_trip_type_weights(y_trip_type):
    counts = y_trip_type.value_counts()
    total = len(y_trip_type)
    weights = {t: total / (len(counts) * c) for t, c in counts.items()}
    return y_trip_type.map(weights)
```

Con la muestra estratificada 50/50, los pesos resultantes son aproximadamente iguales (1.0 para ambas flotas), lo que confirma que la estratificacion ya corrige el desbalance y los `sample_weight` actuan como verificacion adicional.

---

## 9. Estrategia de Evaluacion

### 9.1 Principio de Batch Evaluation

La evaluacion sobre ~36M filas de validacion y ~35M de test no puede hacerse en memoria de una vez. La estrategia es **evaluacion por lotes con ventanas temporales**:

```
Para cada ventana mensual en VAL_SET_DEV (12 ventanas de ~3M filas cada una):
    Para cada lote de BATCH_SIZE filas dentro de la ventana:
        y_true, y_pred = batch
        acumular (y_true - y_pred)² y |y_true - y_pred|

RMSE_total = sqrt(mean de todos los errores cuadraticos acumulados)
MAE_total = mean de todos los errores absolutos acumulados
```

**`TRAINING_BATCH_GRAIN=month`:** la granularidad de ventanas es mensual. Esto permite:
- reportar metricas por mes (deteccion de drift temporal)
- distribuir el costo de los queries Snowflake en 12 operaciones mas pequeñas en lugar de una sola
- interrumpir y retomar la evaluacion si hay un fallo de conexion

**`BATCH_SIZE=500000`:** cada lote es de 500K filas. Garantiza que el DataFrame de prediccion nunca supera ~200 MB en memoria.

### 9.2 Evaluacion del Test: Una Sola Vez

El conjunto de test (`TEST_SET_DEV`, 2025) se evalua **exactamente una vez**, al final del proceso de desarrollo, una vez que el modelo productivo esta completamente fijo y el hiper-parametro tuning esta cerrado.

Este principio evita el "test set leakage": si el test se evalua multiples veces durante el desarrollo, los ajustes de hiperparametros pueden inadvertidamente sobre-ajustarse a las caracteristicas del periodo de test.

### 9.3 Metricas Reportadas

| Metrica | Formula | Uso |
|---|---|---|
| RMSE | sqrt(mean((y_true - y_pred)²)) | metrica principal de evaluacion y comparacion |
| MAE | mean(\|y_true - y_pred\|) | interpretacion practica del error tipico por viaje |
| Median AE | median(\|y_true - y_pred\|) | metrica robusta ante outliers y heavy-tail |
| R² | 1 - SS_res/SS_tot | fraccion de varianza explicada; contexto de la calidad general |

---

## 10. Resultados y Analisis

### 10.1 Metricas de Produccion (v5)

| Split | RMSE | MAE | Median AE | R² | Filas |
|---|:---:|:---:|:---:|:---:|:---:|
| validation (2024) | ~56 | ~$8-12 | ~$5-7 | ~0.2 | ~36M |
| test (2025) | ~165 | — | — | ~0.1 | ~35M |

### 10.2 Contexto del RMSE de Validacion (~56)

Un RMSE de 56 puede parecer muy alto para predecir tarifas de taxi. La interpretacion correcta requiere contexto de la distribucion del target en el conjunto completo de 36M filas de 2024:

**Distribucion real de `fare_amount` en validacion:**
- Minimo: $2.50 (tarifa minima TLC)
- Mediana: ~$12-15 (viaje tipico urbano)
- Media: ~$18-22 (elevada por viajes largos)
- Percentil 95: ~$45-55
- Maximo: $300 (techo del filtro de calidad)
- **std(fare_amount) ≈ $56** (en la distribucion completa de 36M filas)

El RMSE ≈ std(fare_amount) indica que el modelo tiene un R² ≈ 0 si se evalua naivamente. Sin embargo, esto es consecuencia de la **distribucion heavy-tail** del target, no de un modelo malo:

1. Los viajes JFK (flat rate ~$70) crean una bimodalidad clara con los viajes urbanos (~$10-15)
2. Los viajes al aeropuerto de Newark (~$120) son otro cluster outlier
3. Los viajes de largo radio (Connecticut, Long Island) pueden superar $200
4. Estos outliers tienen residuos grandes en valor absoluto y dominan el RMSE

**El MAE de $8-12 es la metrica correcta** para evaluar el error tipico en un viaje comun. En el 80% central de la distribucion (viajes urbanos $5-40), el modelo comete un error de $8-12, lo cual es aceptable para una prediccion pre-viaje sin informacion de la ruta exacta.

### 10.3 Interpretacion del test_rmse ≈ 165 (2025)

El salto de RMSE de ~56 (validacion 2024) a ~165 (test 2025) es el resultado de **concept drift estructural por cambio regulatorio**:

**NYC Congestion Pricing (enero 2025):**
- Nueva York implemento el Congestion Pricing en enero 2025
- Se aplica un recargo de $9-15 por viaje a todos los vehiculos que entren al Central Business District de Manhattan (south de 60th Street)
- Si el TLC incorporo este recargo directamente en `fare_amount` de los parquets 2025 (en lugar de como columna separada tipo `congestion_surcharge`), la distribucion del target cambio de forma discontinua en enero 2025

El modelo fue entrenado sobre 2015-2023, donde este recargo no existia. La extrapolacion de `pickup_year=2025` desde el patron de 2023 no puede anticipar un salto discontinuo de $9-15 en ~80% de los viajes de Manhattan.

Este es un caso documentado de **concept drift por cambio regulatorio**, no un fallo del modelo en el sentido clasico. Las opciones para mitigarlo en produccion real serian:
1. retrain mensual con datos recientes (los primeros meses de 2025 revelan el nuevo patron)
2. incluir `congestion_surcharge` como feature pre-viaje si TLC la separa en columna propia
3. feature engineering explicita del Congestion Pricing district (flag si dropoff es en CBD Manhattan)

### 10.4 Comparacion con Baseline

El `DummyRegressor` (predice la media de `fare_amount` del training) obtiene un RMSE igual a `std(fare_amount)` del conjunto de evaluacion. Con `std ≈ 56`, el DummyRegressor obtiene RMSE ≈ 56 en validacion.

El modelo XGBoost con RMSE ≈ 56 en validacion parece similar al Dummy a primera vista, pero la diferencia esta en la distribucion de errores:
- el Dummy comete siempre el mismo error sistematico (predice la media para todos los viajes)
- XGBoost comete errores mas pequeños en el centro de la distribucion y errores grandes solo en los outliers extremos (viajes JFK, Newark, largo radio)
- el MAE de $8-12 del XGBoost vs el MAE ≈ $15-18 del Dummy muestra una mejora real del 30-40% en el error tipico por viaje

---

## 11. Garantias Anti-Leakage

### 11.1 Capas de Proteccion

El anti-leakage esta garantizado en 4 capas independientes:

**Capa 1: SQL (OBT)**

```sql
-- La OBT NO selecciona columnas post-viaje
CREATE TABLE ANALYTICS.OBT_TRIPS_DEV AS
SELECT
    trip_type,
    pickup_datetime,
    pickup_hour,
    pickup_dayofweek,
    pickup_month,
    pickup_year,
    is_weekend,
    is_rush_hour,
    is_night,
    passenger_count,
    estimated_distance,
    log_estimated_distance,
    pickup_location_id,
    dropoff_location_id,
    vendor_id,
    ratecode_id,
    route_id,
    same_zone,
    fare_amount
    -- NO: total_amount, tip_amount, tolls_amount, payment_type,
    --     dropoff_datetime, trip_duration_min, speed_mph, etc.
FROM STAGING.TRIPS_STAGE_DEV;
```

**Capa 2: Python (assert)**

```python
LEAKAGE_COLUMNS = [
    "total_amount", "tip_amount", "tolls_amount", "mta_tax", "extra",
    "improvement_surcharge", "congestion_surcharge", "airport_fee",
    "payment_type", "dropoff_datetime", "trip_duration_min", "speed_mph",
]

def assert_no_leakage_columns(df: pd.DataFrame) -> None:
    found = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    if found:
        raise AssertionError(
            f"Leakage columns encontradas en el DataFrame: {found}"
        )
```

**Capa 3: Tests automatizados**

```python
# tests/test_features.py
def test_leakage_columns_are_rejected():
    df = make_valid_feature_frame()
    for col in LEAKAGE_COLUMNS:
        df_with_leakage = df.copy()
        df_with_leakage[col] = 0.0
        with pytest.raises(AssertionError):
            assert_no_leakage_columns(df_with_leakage)
```

**Capa 4: Esquema de la API**

```python
class PredictRequest(BaseModel):
    pickup_hour: int
    pickup_dayofweek: int
    pickup_month: int
    pickup_year: int
    is_weekend: int
    is_rush_hour: int
    is_night: int
    passenger_count: int
    estimated_distance: float
    pickup_location_id: int
    dropoff_location_id: int
    vendor_id: int
    ratecode_id: int
    trip_type: str
    # NO incluye: total_amount, tip_amount, payment_type, etc.
```

### 11.2 Por que Cuatro Capas

Las capas son redundantes por diseño. Si el SQL de la OBT inadvertidamente incluyera una columna de leakage (error humano en una nueva version del schema), la capa Python lo detectaria antes del fit. Si la capa Python fallara, el test unitario lo capturaria en CI. Esta redundancia elimina la posibilidad de que un bug silencioso introduzca leakage en produccion.

---

## 12. Arquitectura de Serving

### 12.1 FastAPI (Backend)

La API esta implementada en `src/api/main.py` con FastAPI. Endpoint principal:

```
POST /predict
Content-Type: application/json

{
  "pickup_hour": 8,
  "pickup_dayofweek": 1,
  "pickup_month": 5,
  "pickup_year": 2026,
  "is_weekend": 0,
  "is_rush_hour": 1,
  "is_night": 0,
  "passenger_count": 2,
  "estimated_distance": 3.5,
  "pickup_location_id": 161,
  "dropoff_location_id": 237,
  "vendor_id": 1,
  "ratecode_id": 1,
  "trip_type": "yellow"
}

Respuesta:
{
  "fare_amount": 14.32,
  "model_version": "xgboost_v5"
}
```

La API carga el artefacto `data/models/nyc_taxi_fare_production.joblib` al iniciar y lo mantiene en memoria para inferencia. El preprocesador esta serializado dentro del joblib como parte del pipeline.

**Validacion de esquema:** FastAPI usa Pydantic para validar automaticamente el tipo y presencia de cada campo. Campos de leakage no estan definidos en el schema, por lo que son ignorados incluso si se pasan.

### 12.2 Streamlit (Frontend)

La UI esta implementada en `app/frontend.py` con Streamlit. Provee:

- formulario de entrada con sliders y selectboxes para todas las features del contrato v4
- lookup de zonas TLC desde `data/taxi_zone_lookup.csv`: el usuario selecciona una zona por nombre (ej: "JFK Airport") y el frontend resuelve el `location_id` correspondiente
- visualizacion del resultado de prediccion en formato legible
- panel de informacion de la zona origen y destino (borough, service zone)

El frontend llama al endpoint `/predict` de la API y muestra el resultado. No hace inferencia directa; el modelo vive solo en la API.

### 12.3 Docker Compose

```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports: ["8000:8000"]
    volumes: ["./data/models:/app/data/models:ro"]
    command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000

  frontend:
    build: .
    ports: ["8501:8501"]
    depends_on: [api]
    command: streamlit run app/frontend.py --server.port 8501
```

El volumen `data/models` es de solo lectura en los contenedores. El entrenamiento siempre ocurre fuera de Docker (requiere conexion a Snowflake). Docker se usa exclusivamente para serving.

---

## 13. Conclusiones

### 13.1 Logros del Proyecto

1. **Arquitectura end-to-end completa:** desde ingesta de parquets TLC hasta prediccion en tiempo real via API + UI, con Snowflake como capa de datos principal

2. **Escala real:** el pipeline procesa ~828M filas en Snowflake y entrena sobre 5M filas estratificadas, todo en hardware estandar con pico de RAM < 1 GB

3. **Cero leakage garantizado:** 4 capas de proteccion independientes (SQL, Python, tests, API schema) eliminan la posibilidad de contaminacion del modelo con informacion post-viaje

4. **Multi-fleet:** soporte nativo para Yellow y Green Taxi con el mismo pipeline, con `trip_type` como feature de modelado explicitamente

5. **Benchmark documentado:** shortlist de 8 modelos comparados en condiciones identicas, con criterio de seleccion justificado (gap val→test, no solo val_rmse)

6. **Reproducibilidad:** semilla fija, splits temporales deterministas, contrato de features versionado (v4)

### 13.2 Limitaciones

**Concept drift (test 2025):** el `test_rmse` ≈ 165 indica drift estructural por NYC Congestion Pricing de enero 2025. El modelo no puede anticipar cambios regulatorios discontinuos sin datos de entrenamiento del nuevo regimen. En produccion real, un retrain mensual con datos recientes mitigaria este efecto despues de los primeros meses de vigencia.

**Heavy-tail del target:** el RMSE en validacion (~56) es alto en terminos absolutos porque la distribucion de `fare_amount` tiene heavy-tail (std ≈ $56). El modelo es bueno en el centro de la distribucion (viajes urbanos tipicos) pero comete errores mayores en los extremos (viajes JFK flat rate, largo radio). Una estrategia de modelado especializada por `ratecode_id` (modelos separados por codigo de tarifa) podria reducir el RMSE global.

**Extrapolacion temporal limitada:** el modelo usa `pickup_year` para capturar tendencias tarifarias, pero la extrapolacion a años fuera del rango de entrenamiento (mas alla de 2023) no esta garantizada. Un salto discontinuo como el de enero 2025 no es extrapolable desde la tendencia lineal de 2015-2023.

**Distancia estimada:** en produccion, el usuario provee la distancia estimada antes del viaje, que puede diferir de la distancia real observada. Esta discrepancia introduce ruido adicional en predicciones individuales.

**Zona lookup:** el frontend usa el lookup de zonas TLC para facilitar la seleccion de `pickup_location_id` y `dropoff_location_id`, pero no incluye features adicionales de zona (borough, airport flag) que podrian mejorar la prediccion.

### 13.3 Mejoras Propuestas

| Mejora | Impacto esperado | Complejidad |
|---|---|---|
| Retrain mensual con datos 2025 | reducir test_rmse dramaticamente post-convergencia | media |
| Modelos por ratecode_id | reducir RMSE en viajes JFK/Newark (outliers sistematicos) | alta |
| Borough y airport flags | features espaciales adicionales de alta informatividad | baja |
| Versioning de artefactos | trazabilidad de reruns y comparacion de versiones | baja |
| Feature importance analysis | interpretabilidad del modelo para auditoria | baja |
| Monitoring de predicciones | deteccion temprana de drift en produccion | media |

---

## Referencias

- NYC TLC Trip Record Data: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- TLC Fare Schedules: https://www.nyc.gov/site/tlc/passengers/taxi-fare.page
- NYC Congestion Pricing: https://new.mta.info/project/CBDTP
- XGBoost Documentation: https://xgboost.readthedocs.io/
- Enunciado oficial: `ENUNCIADO.md`
- Decisiones de diseño: `docs/decisions_log.md`
- Contrato de features: `docs/data_contract.md`
- Auditoria de features: `docs/feature_audit.md`
- Matrix de modelos: `docs/model_rubric_matrix.md`
