import datetime as dt

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"
HEALTH_URL = "http://127.0.0.1:8000/health"

st.set_page_config(page_title="NYC Taxi Fare Predictor", page_icon="🚕", layout="centered")

st.title("NYC Taxi Fare Predictor")
st.caption("Estimación de `fare_amount` para viajes NYC Yellow Taxi — sin data leakage.")

# API health check
try:
    h = requests.get(HEALTH_URL, timeout=3)
    info = h.json()
    if info.get("model_loaded"):
        st.success(f"API conectada · Modelo: **{info.get('model_name', 'desconocido')}**")
    else:
        st.warning("API conectada pero el modelo no está cargado. ¿Corriste el entrenamiento?")
except requests.RequestException:
    st.error("No se puede conectar a la API. Inicia el servidor con: `uvicorn src.api.main:app --reload`")

st.divider()

with st.form("trip_form"):
    st.subheader("Datos del viaje")

    col1, col2 = st.columns(2)
    with col1:
        pickup_date = st.date_input("Fecha de recogida", value=dt.date(2025, 1, 15))
        pickup_time = st.time_input("Hora de recogida", value=dt.time(14, 35))
        pickup_location_id = st.selectbox(
            "Zona de recogida (LocationID)",
            options=list(range(1, 266)),
            index=236,
            help="ID de zona NYC TLC (1-265). Ej: 237 = Upper East Side",
        )
        dropoff_location_id = st.selectbox(
            "Zona de destino (LocationID)",
            options=list(range(1, 266)),
            index=140,
            help="ID de zona NYC TLC (1-265). Ej: 141 = Lenox Hill",
        )

    with col2:
        trip_distance = st.number_input(
            "Distancia del viaje (millas)", min_value=0.1, value=7.8, step=0.1
        )
        passenger_count = st.number_input(
            "Número de pasajeros", min_value=1, max_value=8, value=2
        )
        vendor_id = st.selectbox(
            "Vendor", options=[1, 2], index=0, help="1=Creative Mobile, 2=VeriFone"
        )
        ratecode_id = st.selectbox(
            "Rate Code",
            options=[1, 2, 3, 4, 5, 6],
            index=0,
            help="1=Standard, 2=JFK, 3=Newark, 4=Nassau/Westchester, 5=Negotiated, 6=Group",
        )

    submitted = st.form_submit_button("Estimar tarifa", use_container_width=True)

if submitted:
    pickup_datetime = dt.datetime.combine(pickup_date, pickup_time).strftime("%Y-%m-%d %H:%M:%S")

    payload = {
        "pickup_datetime": pickup_datetime,
        "pickup_location_id": int(pickup_location_id),
        "dropoff_location_id": int(dropoff_location_id),
        "passenger_count": int(passenger_count),
        "trip_distance": float(trip_distance),
        "vendor_id": int(vendor_id),
        "ratecode_id": int(ratecode_id),
    }

    with st.spinner("Consultando la API..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=15)
            response.raise_for_status()
            data = response.json()
            fare = data.get("estimated_fare_amount")
            model = data.get("model", "desconocido")
            st.success(f"### Tarifa estimada: **${fare:.2f}**")
            st.caption(f"Modelo: {model}")
            with st.expander("Ver datos enviados"):
                st.json(payload)
        except requests.HTTPError as exc:
            st.error(f"Error de API ({exc.response.status_code}): {exc.response.text}")
        except requests.RequestException as exc:
            st.error(f"No se pudo conectar a la API: {exc}")
