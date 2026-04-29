import datetime as dt

import requests
import streamlit as st

# URL de la API (suponiendo que FastAPI corre en el puerto 8000)
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="NYC Taxi Price Predictor", page_icon="🚕")

st.title("NYC Taxi Price Predictor")
st.write(
	"Estimador de `total_amount` para viajes de NYC Taxi usando un modelo ML. "
	"Usa variables sin leakage (sin propinas ni peajes)."
)

with st.form("trip_form"):
	st.subheader("Datos del viaje")

	col1, col2 = st.columns(2)
	with col1:
		pickup_date = st.date_input("Pickup date", value=dt.date(2024, 3, 10))
		pickup_time = st.time_input("Pickup time", value=dt.time(14, 35))
		pickup_longitude = st.number_input("Pickup longitude", value=-73.9857, format="%.6f")
		pickup_latitude = st.number_input("Pickup latitude", value=40.7484, format="%.6f")

	with col2:
		dropoff_longitude = st.number_input("Dropoff longitude", value=-73.7769, format="%.6f")
		dropoff_latitude = st.number_input("Dropoff latitude", value=40.6413, format="%.6f")
		passenger_count = st.number_input("Passenger count", min_value=1, max_value=8, value=2)
		trip_distance = st.number_input("Trip distance (miles)", min_value=0.0, value=7.8, step=0.1)

	col3, col4, col5 = st.columns(3)
	with col3:
		ratecodeid = st.selectbox("Rate code", options=[1, 2, 3, 4, 5, 6], index=0)
	with col4:
		payment_type = st.selectbox("Payment type", options=[1, 2, 3, 4, 5, 6], index=0)
	with col5:
		vendorid = st.selectbox("Vendor", options=[1, 2], index=0)

	store_and_fwd_flag = st.selectbox("Store and forward flag", options=["N", "Y"], index=0)

	submitted = st.form_submit_button("Predict")

if submitted:
	pickup_datetime = dt.datetime.combine(pickup_date, pickup_time).strftime("%Y-%m-%d %H:%M:%S")

	payload = {
		"pickup_datetime": pickup_datetime,
		"pickup_longitude": float(pickup_longitude),
		"pickup_latitude": float(pickup_latitude),
		"dropoff_longitude": float(dropoff_longitude),
		"dropoff_latitude": float(dropoff_latitude),
		"passenger_count": int(passenger_count),
		"trip_distance": float(trip_distance),
		"ratecodeid": int(ratecodeid),
		"payment_type": int(payment_type),
		"vendorid": int(vendorid),
		"store_and_fwd_flag": store_and_fwd_flag,
	}

	with st.spinner("Consultando la API..."):
		try:
			response = requests.post(API_URL, json=payload, timeout=10)
			response.raise_for_status()
			data = response.json()
			st.success(f"Estimated total amount: ${data['estimated_total_amount']}")
			st.caption(f"Model: {data.get('model', 'unknown')}")
		except requests.RequestException as exc:
			st.error(f"API error: {exc}")