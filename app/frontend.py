import datetime as dt

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"
HEALTH_URL = "http://127.0.0.1:8000/health"

st.set_page_config(
    page_title="NYC Taxi Fare Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #f5efe5;
        --paper: #fffaf3;
        --ink: #1f2933;
        --muted: #5f6c7b;
        --accent: #d97706;
        --accent-dark: #9a3412;
        --accent-soft: #fde7c2;
        --line: #ead8bf;
        --ok: #0f766e;
        --warn: #b45309;
        --err: #b91c1c;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(217, 119, 6, 0.12), transparent 28%),
            radial-gradient(circle at top right, rgba(15, 118, 110, 0.10), transparent 24%),
            linear-gradient(180deg, #fcf7ef 0%, var(--bg) 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .hero {
        background: linear-gradient(135deg, rgba(255, 250, 243, 0.96), rgba(253, 231, 194, 0.92));
        border: 1px solid var(--line);
        border-radius: 28px;
        padding: 2rem 2.2rem;
        box-shadow: 0 24px 60px rgba(120, 83, 31, 0.08);
        margin-bottom: 1.25rem;
    }

    .hero-kicker {
        color: var(--accent-dark);
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }

    .hero-title {
        color: var(--ink);
        font-size: 2.6rem;
        line-height: 1.05;
        font-weight: 800;
        margin-bottom: 0.8rem;
    }

    .hero-copy {
        color: var(--muted);
        font-size: 1.02rem;
        max-width: 760px;
        margin-bottom: 0;
    }

    .card {
        background: rgba(255, 250, 243, 0.94);
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1.25rem 1.2rem;
        box-shadow: 0 16px 40px rgba(120, 83, 31, 0.06);
    }

    .card-title {
        color: var(--accent-dark);
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }

    .card-value {
        color: var(--ink);
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .card-copy {
        color: var(--muted);
        font-size: 0.95rem;
        margin-bottom: 0;
    }

    .result-box {
        background: linear-gradient(135deg, rgba(15, 118, 110, 0.12), rgba(255, 250, 243, 0.96));
        border: 1px solid rgba(15, 118, 110, 0.24);
        border-radius: 24px;
        padding: 1.5rem;
        margin-top: 1rem;
    }

    .result-label {
        color: var(--ok);
        font-size: 0.84rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .result-value {
        color: var(--ink);
        font-size: 2.35rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 0.45rem;
    }

    .result-copy {
        color: var(--muted);
        font-size: 0.96rem;
        margin-bottom: 0;
    }

    div[data-testid="stForm"] {
        background: rgba(255, 250, 243, 0.94);
        border: 1px solid var(--line);
        border-radius: 26px;
        padding: 1.35rem 1.35rem 0.8rem 1.35rem;
        box-shadow: 0 18px 42px rgba(120, 83, 31, 0.06);
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fffaf3 0%, #f7ecda 100%);
        border-right: 1px solid var(--line);
    }

    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
        color: white;
        border: 0;
        border-radius: 14px;
        font-weight: 700;
        min-height: 3rem;
    }

    .stButton > button:hover, .stFormSubmitButton > button:hover {
        filter: brightness(1.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_health() -> tuple[bool, str, str]:
    try:
        response = requests.get(HEALTH_URL, timeout=3)
        response.raise_for_status()
        info = response.json()
        if info.get("model_loaded"):
            return True, "Modelo cargado", info.get("model_name", "desconocido")
        return False, "API conectada sin modelo", "desconocido"
    except requests.RequestException:
        return False, "API no disponible", "desconocido"


api_ok, api_status, model_name = api_health()

st.markdown(
    """
    <section class="hero">
        <div class="hero-kicker">Proyecto Final · Prediccion de Tarifas</div>
        <div class="hero-title">NYC Taxi Fare Predictor</div>
        <p class="hero-copy">
            Interfaz de validacion para el modelo final de estimacion de <code>fare_amount</code>.
            La app consume la API de FastAPI y usa exclusivamente variables disponibles antes del viaje,
            manteniendo la politica de no leakage exigida por el proyecto.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

status_col, model_col, policy_col = st.columns(3)
with status_col:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">Estado de API</div>
            <div class="card-value">{api_status}</div>
            <p class="card-copy">Endpoint esperado: <code>{HEALTH_URL}</code></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with model_col:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">Modelo Activo</div>
            <div class="card-value">{model_name}</div>
            <p class="card-copy">Debe existir un artefacto entrenado en <code>MODEL_DIR</code>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with policy_col:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Politica de Features</div>
            <div class="card-value">Sin leakage</div>
            <p class="card-copy">No se usan variables post-viaje como <code>tip_amount</code> o <code>total_amount</code>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

main_col, side_col = st.columns([1.75, 0.85], gap="large")

with side_col:
    st.markdown("### Guia rapida")
    st.markdown(
        """
        - Levanta la API con `uvicorn src.api.main:app --reload`
        - Confirma que el entrenamiento ya genero un `.joblib`
        - Usa valores razonables de distancia y zonas
        - Revisa el payload final antes de documentar resultados
        """
    )

    st.markdown("### Features admitidas")
    st.markdown(
        """
        - `pickup_datetime`
        - `pickup_location_id`
        - `dropoff_location_id`
        - `passenger_count`
        - `estimated_distance`
        - `vendor_id`
        - `ratecode_id`
        """
    )

    st.markdown("### Validacion minima")
    st.markdown(
        """
        - API conectada
        - modelo cargado
        - prediccion devuelta sin error HTTP
        - tarifa positiva y coherente con la distancia
        """
    )

with main_col:
    with st.form("trip_form"):
        st.subheader("Datos del viaje")
        st.caption("Ingresa solo variables disponibles antes de iniciar el trayecto.")

        col1, col2 = st.columns(2, gap="large")
        with col1:
            pickup_date = st.date_input("Fecha de recogida", value=dt.date(2025, 1, 15))
            pickup_time = st.time_input("Hora de recogida", value=dt.time(14, 35))
            pickup_location_id = st.selectbox(
                "Zona de recogida (LocationID)",
                options=list(range(1, 266)),
                index=236,
                help="ID de zona NYC TLC entre 1 y 265.",
            )
            dropoff_location_id = st.selectbox(
                "Zona de destino (LocationID)",
                options=list(range(1, 266)),
                index=140,
                help="ID de zona de destino NYC TLC entre 1 y 265.",
            )

        with col2:
            estimated_distance = st.number_input(
                "Distancia estimada del viaje (millas)",
                min_value=0.1,
                value=7.8,
                step=0.1,
            )
            passenger_count = st.number_input(
                "Numero de pasajeros",
                min_value=1,
                max_value=8,
                value=2,
            )
            vendor_id = st.selectbox(
                "Vendor",
                options=[1, 2],
                index=0,
                help="1 = Creative Mobile, 2 = VeriFone.",
            )
            ratecode_id = st.selectbox(
                "Rate Code",
                options=[1, 2, 3, 4, 5, 6],
                index=0,
                help="1 = Standard, 2 = JFK, 3 = Newark, 4 = Nassau/Westchester, 5 = Negotiated, 6 = Group.",
            )

        submitted = st.form_submit_button("Estimar tarifa", use_container_width=True)

if not api_ok:
    st.warning("La API no esta lista. Inicia `uvicorn src.api.main:app --reload` y vuelve a intentar.")

if submitted:
    pickup_datetime = dt.datetime.combine(pickup_date, pickup_time).strftime("%Y-%m-%d %H:%M:%S")
    payload = {
        "pickup_datetime": pickup_datetime,
        "pickup_location_id": int(pickup_location_id),
        "dropoff_location_id": int(dropoff_location_id),
        "passenger_count": int(passenger_count),
        "estimated_distance": float(estimated_distance),
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

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-label">Tarifa estimada</div>
                    <div class="result-value">${fare:.2f}</div>
                    <p class="result-copy">Prediccion generada por <strong>{model}</strong> a partir de variables pre-viaje.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            details_col, payload_col = st.columns([1, 1], gap="large")
            with details_col:
                st.markdown("### Resumen de validacion")
                st.markdown(
                    f"""
                    - Fecha y hora de pickup: `{pickup_datetime}`
                    - Distancia estimada: `{estimated_distance:.1f}` millas
                    - Pasajeros: `{passenger_count}`
                    - Modelo: `{model}`
                    """
                )
            with payload_col:
                with st.expander("Ver payload enviado a la API", expanded=False):
                    st.json(payload)

        except requests.HTTPError as exc:
            st.error(f"Error de API ({exc.response.status_code}): {exc.response.text}")
        except requests.RequestException as exc:
            st.error(f"No se pudo conectar a la API: {exc}")
