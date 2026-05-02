# Enunciado Original

Este archivo conserva el enunciado base del proyecto final, separado del README operativo de esta fase.

## Proyecto Final: Predicción de Precios (End-to-End ML con Big Data)

Este repositorio es la plantilla oficial para el proyecto final.

### Arquitectura para grandes volumenes

Procesamiento de aproximadamente `20GB` de informacion de viajes. La limpieza estructurada, la construccion de la OBT y la division `train/validation/test` deben ejecutarse del lado de Snowflake mediante SQL. No se debe descargar toda la data ni hacer `train_test_split` tradicional sobre un DataFrame completo.

### Objetivo general

Consumir `analytics.obt_trips` desde Snowflake, hacer exploracion en muestras, entrenar de forma modular en Python y dejar el proyecto listo para evolucionar hacia API y frontend.

### Flujo esperado

1. Modelado de datos en `src/data/sql/`.
2. Exploracion y limpieza en `notebooks/` usando muestras.
3. Experimentacion de modelado con entrenamiento por lotes.
4. Refactorizacion a codigo productivo en `src/`.
5. Despliegue posterior por FastAPI y Streamlit.

### Estructura esperada

```text
├── data/
├── notebooks/
├── src/
│   ├── data/sql/
│   ├── features/
│   ├── models/
│   ├── api/
│   └── utils/
├── app/
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

### Penalizaciones criticas del proyecto final

- Data leakage en features de entrada.
- Carga completa de la base en pandas.
- No usar lotes o muestras.
- Omitir algoritmos boosting obligatorios en la fase completa del curso.

Para la fase base implementada en este repositorio, el README principal ya documenta la adaptacion Snowflake-first y el alcance reducido a un mes de datos.
