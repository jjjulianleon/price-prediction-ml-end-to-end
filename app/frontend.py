import csv
import datetime as dt
import os
from pathlib import Path

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_URL = f"{API_BASE_URL}/predict"
HEALTH_URL = f"{API_BASE_URL}/health"

ZONE_LOOKUP_LOCAL = Path(__file__).parent.parent / "data" / "taxi_zone_lookup.csv"
ZONE_LOOKUP_CDN = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

VENDOR_LABELS = {1: "Creative Mobile Technologies", 2: "Curb Mobility (VeriFone)"}
RATECODE_LABELS = {
    1: "Standard rate",
    2: "JFK flat rate (~$52)",
    3: "Newark (metered)",
    4: "Nassau / Westchester",
    5: "Negotiated fare",
    6: "Group ride",
}

st.set_page_config(
    page_title="NYC Taxi Fare Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* System font stack — no external dependencies */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui,
                     "Helvetica Neue", Arial, sans-serif;
    }

    :root {
        --bg: #eef3f8;
        --surface: rgba(255,255,255,0.92);
        --ink: #132033;
        --muted: #5b6b80;
        --line: #cfd9e6;
        --navy: #17324d;
        --navy-soft: #284868;
        --teal: #0f766e;
        --teal-soft: #d9f3f0;
        --amber: #c66a1b;
        --amber-deep: #9f4b12;
        --shadow-lg: 0 24px 70px rgba(24,39,56,0.10);
        --shadow-md: 0 18px 40px rgba(24,39,56,0.08);
    }

    .stApp {
        background: linear-gradient(180deg, #f7fafc 0%, var(--bg) 60%, #edf2f7 100%);
        color: var(--ink);
    }
    .block-container { max-width: 1260px; padding-top: 2rem; padding-bottom: 2.5rem; }
    h1,h2,h3,h4 { color: var(--ink); letter-spacing: -0.02em; }
    p,li,label,span { color: var(--ink); }
    code {
        color: #0f2d45; background: #e7eef6;
        border: 1px solid #d0dbe7; border-radius: 8px;
        padding: 0.12rem 0.42rem; font-size: 0.92em; font-weight: 700;
    }

    /* Hero */
    .hero-panel {
        background: linear-gradient(135deg,rgba(255,255,255,0.93),rgba(255,255,255,0.83));
        border: 1px solid rgba(175,193,214,0.95); border-radius: 28px;
        padding: 2rem 2.2rem; box-shadow: var(--shadow-lg); margin-bottom: 1.1rem;
    }
    .eyebrow {
        font-size: 0.78rem; font-weight: 800; letter-spacing: 0.10em;
        text-transform: uppercase; color: var(--amber-deep); margin-bottom: 0.7rem;
    }
    .hero-title {
        font-size: 2.8rem; line-height: 1.0; font-weight: 800;
        color: var(--ink); margin-bottom: 0.8rem;
    }
    .hero-copy {
        color: var(--muted); max-width: 760px;
        font-size: 1rem; line-height: 1.72; margin: 0;
    }
    .hero-badge-row { display:flex; flex-wrap:wrap; gap:0.7rem; margin-top:1.1rem; }
    .hero-badge {
        display:inline-flex; align-items:center; gap:0.5rem;
        background:rgba(255,255,255,0.78); border:1px solid rgba(175,193,214,0.95);
        border-radius:999px; color:var(--navy); padding:0.55rem 0.85rem;
        font-size:0.88rem; font-weight:700;
    }

    /* Status chips */
    .status-chip {
        display:inline-flex; align-items:center; gap:0.5rem; border-radius:999px;
        padding:0.38rem 0.8rem; font-size:0.82rem; font-weight:800;
        letter-spacing:0.02em; border:1px solid transparent;
    }
    .status-chip.ok   { color:#0b5e57; background:#e3f6ef; border-color:rgba(15,118,110,0.18); }
    .status-chip.warn { color:var(--amber-deep); background:#fff1df; border-color:rgba(198,106,27,0.16); }
    .status-chip.err  { color:#9b2c2c; background:#fde7e7; border-color:rgba(194,65,65,0.16); }
    .status-chip-dot  { width:0.52rem; height:0.52rem; border-radius:999px; background:currentColor; }

    /* Cards */
    .metric-card {
        background:linear-gradient(180deg,rgba(255,255,255,0.94),rgba(255,255,255,0.82));
        border:1px solid rgba(175,193,214,0.9); border-radius:22px;
        padding:1.1rem 1.1rem; box-shadow:var(--shadow-md); min-height:160px;
    }
    .metric-label  { color:var(--navy-soft); font-size:0.74rem; font-weight:800; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.55rem; }
    .metric-value  { color:var(--ink); font-size:1.3rem; line-height:1.15; font-weight:800; margin-bottom:0.4rem; }
    .metric-copy   { color:var(--muted); font-size:0.92rem; line-height:1.55; margin:0; }

    /* Side panels */
    .side-panel, .result-panel, .empty-panel {
        background:linear-gradient(180deg,rgba(255,255,255,0.95),rgba(255,255,255,0.84));
        border:1px solid rgba(175,193,214,0.9); border-radius:24px;
        padding:1.2rem 1.2rem 1.1rem; box-shadow:var(--shadow-md); margin-bottom:1rem;
    }
    .result-panel {
        background: linear-gradient(135deg,rgba(15,118,110,0.08),rgba(255,255,255,0.94));
        border-color:rgba(15,118,110,0.22);
    }
    .empty-panel { border-style:dashed; }
    .panel-kicker { color:var(--navy-soft); font-size:0.74rem; font-weight:800; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.45rem; }
    .panel-title  { color:var(--ink); font-size:1.22rem; font-weight:800; margin-bottom:0.45rem; }
    .panel-copy   { color:var(--muted); font-size:0.92rem; line-height:1.65; }
    .result-amount { color:var(--ink); font-size:2.9rem; line-height:1; font-weight:800; margin:0.2rem 0 0.4rem; }
    .result-subtitle { color:var(--muted); font-size:0.93rem; line-height:1.65; margin-bottom:0.85rem; }

    .mini-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:0.65rem; margin-top:0.9rem; }
    .mini-item {
        background:rgba(255,255,255,0.72); border:1px solid rgba(175,193,214,0.72);
        border-radius:16px; padding:0.7rem 0.75rem;
    }
    .mini-label { color:var(--navy-soft); font-size:0.7rem; font-weight:800; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.22rem; }
    .mini-value { color:var(--ink); font-size:0.96rem; font-weight:800; }

    /* Metrics row */
    .metrics-row { display:flex; flex-wrap:wrap; gap:0.6rem; margin-top:0.85rem; }
    .metrics-pill {
        background:rgba(255,255,255,0.72); border:1px solid rgba(175,193,214,0.72);
        border-radius:999px; padding:0.4rem 0.85rem;
        font-size:0.82rem; font-weight:700; color:var(--navy-soft);
    }
    .metrics-pill strong { color:var(--ink); }

    /* Form */
    div[data-testid="stForm"] {
        background:linear-gradient(180deg,rgba(255,255,255,0.96),rgba(255,255,255,0.90));
        border:1px solid rgba(175,193,214,0.95); border-radius:26px;
        padding:1.45rem 1.4rem 1rem; box-shadow:var(--shadow-lg);
    }
    div[data-testid="stForm"] h3 { margin-top:0.1rem; margin-bottom:0.1rem; }
    div[data-testid="stForm"] p  { color:var(--muted); }
    label { color:var(--ink) !important; font-weight:700 !important; font-size:0.92rem !important; }
    div[data-baseweb="select"]>div, div[data-baseweb="input"]>div,
    div[data-testid="stDateInputField"]>div, div[data-testid="stTimeInput"]>div {
        background:#fff !important; border-radius:14px !important;
        border:1px solid var(--line) !important; min-height:48px !important;
    }
    div[data-baseweb="select"] * { color:var(--ink) !important; }
    input, textarea { color:var(--ink) !important; font-weight:700 !important; }
    div[data-baseweb="select"]>div:focus-within, div[data-baseweb="input"]>div:focus-within {
        border-color:rgba(15,118,110,0.55) !important;
        box-shadow:0 0 0 4px rgba(15,118,110,0.12) !important;
    }
    .stButton>button, .stFormSubmitButton>button {
        border:0 !important; border-radius:14px !important; min-height:3.2rem !important;
        background:linear-gradient(135deg,var(--navy) 0%,var(--teal) 100%) !important;
        color:#fff !important; font-size:0.98rem !important; font-weight:800 !important;
        box-shadow:0 14px 26px rgba(15,118,110,0.18);
        transition:transform 0.16s ease, box-shadow 0.16s ease;
    }
    .stButton>button:hover, .stFormSubmitButton>button:hover {
        transform:translateY(-1px); box-shadow:0 18px 32px rgba(15,118,110,0.22);
    }
    .stButton>button:disabled, .stFormSubmitButton>button:disabled {
        background:linear-gradient(135deg,#95a5b7,#7c8ea2) !important; box-shadow:none;
    }
    div[data-testid="stAlert"] { border-radius:16px; border-width:1px; }
    div[data-testid="stExpander"] {
        border-radius:18px; border:1px solid rgba(175,193,214,0.86);
        background:rgba(255,255,255,0.72);
    }
    .subtle-note { color:var(--muted); font-size:0.88rem; line-height:1.6; margin-top:0.4rem; }
    .bullet-list { margin:0.75rem 0 0 0; padding-left:1.1rem; }
    .bullet-list li { color:var(--muted); margin-bottom:0.5rem; line-height:1.55; }
    @media (max-width:960px) {
        .hero-title { font-size:2.2rem; }
        .mini-grid  { grid-template-columns:1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Zone lookup ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_taxi_zones() -> dict[int, str]:
    """Returns {location_id: 'ID - Zone (Borough)'}. Local file first, CDN fallback."""
    def _parse(rows) -> dict[int, str]:
        return {
            int(r["LocationID"]): f"{int(r['LocationID'])} — {r['Zone']} ({r['Borough']})"
            for r in rows
        }
    if ZONE_LOOKUP_LOCAL.exists():
        with open(ZONE_LOOKUP_LOCAL, newline="") as f:
            return _parse(csv.DictReader(f))
    try:
        import io
        raw = requests.get(ZONE_LOOKUP_CDN, timeout=6).text
        return _parse(csv.DictReader(io.StringIO(raw)))
    except Exception:
        return {i: f"Zone {i}" for i in range(1, 266)}


# ── API helpers ───────────────────────────────────────────────────────────────

def api_health() -> dict:
    info: dict = {
        "reachable": False, "model_loaded": False,
        "api_status": "API no disponible",
        "model_name": "—",
        "detail": f"Endpoint esperado: {HEALTH_URL}",
        "val_rmse": None, "val_mae": None, "val_med_ae": None, "val_r2": None,
        "test_rmse": None, "sample_rows": None,
        "training_strategy": None, "trip_types": None,
        "feature_contract_version": None,
    }
    try:
        r = requests.get(HEALTH_URL, timeout=4)
        r.raise_for_status()
        p = r.json()
        info["reachable"] = True
        info["model_loaded"] = bool(p.get("model_loaded"))
        info["model_name"] = str(p.get("model_name", "—"))
        info["val_rmse"]   = p.get("val_rmse")
        info["val_mae"]    = p.get("val_mae")
        info["val_med_ae"] = p.get("val_med_ae")
        info["val_r2"]     = p.get("val_r2")
        info["test_rmse"]  = p.get("test_rmse")
        info["sample_rows"] = p.get("sample_rows")
        info["training_strategy"] = p.get("training_strategy")
        info["trip_types"] = p.get("trip_types")
        info["feature_contract_version"] = p.get("feature_contract_version")
        if info["model_loaded"]:
            info["api_status"] = "Servicio listo para inferencia"
            info["detail"] = "API conectada y artefacto productivo cargado."
        else:
            info["api_status"] = "API online — modelo no cargado"
            info["detail"] = "La API responde, pero no encontró el joblib productivo."
    except requests.RequestException:
        pass
    return info


# ── Render helpers ────────────────────────────────────────────────────────────

def render_metric_card(label: str, value: str, copy: str) -> None:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<p class="metric-copy">{copy}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_empty_result_panel() -> None:
    st.markdown(
        '<div class="empty-panel">'
        '<div class="panel-kicker">Resultado</div>'
        '<div class="panel-title">Esperando una predicción</div>'
        '<p class="panel-copy">Completa el formulario y envía el payload para ver la '
        'tarifa estimada, el modelo que respondió y un resumen del viaje consultado.</p>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_result_panel(result: dict, zones: dict[int, str]) -> None:
    fare    = float(result["fare"])
    model   = str(result["model"])
    payload = result["payload"]
    pu_name = zones.get(int(payload["pickup_location_id"]),  f"Zone {payload['pickup_location_id']}")
    do_name = zones.get(int(payload["dropoff_location_id"]), f"Zone {payload['dropoff_location_id']}")
    rate_lbl = RATECODE_LABELS.get(int(payload["ratecode_id"]), f"Rate {payload['ratecode_id']}")
    st.markdown(
        f'<div class="result-panel">'
        f'<div class="panel-kicker">Predicción confirmada</div>'
        f'<div class="status-chip ok"><span class="status-chip-dot"></span>Respuesta válida</div>'
        f'<div class="result-amount">${fare:,.2f}</div>'
        f'<p class="result-subtitle">Estimación generada por <strong>{model}</strong> '
        f'usando variables disponibles antes del viaje.</p>'
        f'<div class="mini-grid">'
        f'<div class="mini-item"><div class="mini-label">Fleet</div>'
        f'<div class="mini-value">{payload["trip_type"].title()} Taxi</div></div>'
        f'<div class="mini-item"><div class="mini-label">Distance</div>'
        f'<div class="mini-value">{float(payload["estimated_distance"]):.1f} mi</div></div>'
        f'<div class="mini-item"><div class="mini-label">Pickup zone</div>'
        f'<div class="mini-value" style="font-size:0.82rem">{pu_name}</div></div>'
        f'<div class="mini-item"><div class="mini-label">Dropoff zone</div>'
        f'<div class="mini-value" style="font-size:0.82rem">{do_name}</div></div>'
        f'<div class="mini-item"><div class="mini-label">Rate code</div>'
        f'<div class="mini-value" style="font-size:0.82rem">{rate_lbl}</div></div>'
        f'<div class="mini-item"><div class="mini-label">Datetime</div>'
        f'<div class="mini-value" style="font-size:0.82rem">{payload["pickup_datetime"]}</div></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Init ──────────────────────────────────────────────────────────────────────

zones = load_taxi_zones()
zone_options = sorted(zones.keys())
zone_labels  = [zones[z] for z in zone_options]

health = api_health()
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

# ── Hero ──────────────────────────────────────────────────────────────────────

hero_cls = "ok" if health["model_loaded"] else "warn" if health["reachable"] else "err"
hero_txt = (
    "Modelo listo" if health["model_loaded"]
    else "API online — sin modelo" if health["reachable"]
    else "Servicio fuera de línea"
)

st.markdown(
    f'<section class="hero-panel">'
    f'<div class="eyebrow">Proyecto final · predicción de tarifas NYC TLC</div>'
    f'<div class="hero-title">NYC Taxi Fare Predictor</div>'
    f'<p class="hero-copy">'
    f'Interfaz de validación del modelo productivo <strong>XGBoost</strong> entrenado sobre '
    f'5M viajes estratificados (2015–2023, yellow + green). La predicción es pre-viaje: '
    f'solo se usan variables conocidas antes de iniciar el trayecto — '
    f'<code>fare_amount</code> es el target, sin leakage de pagos ni duración.'
    f'</p>'
    f'<div class="hero-badge-row">'
    f'<div class="status-chip {hero_cls}"><span class="status-chip-dot"></span>{hero_txt}</div>'
    f'<div class="hero-badge"><strong>API</strong>&nbsp;{API_BASE_URL}</div>'
    f'<div class="hero-badge"><strong>Modelo</strong>&nbsp;{health["model_name"]}</div>',
    unsafe_allow_html=True,
)

# Add metric pills inline in hero if available
if health["val_rmse"] is not None:
    parts = []
    if health["val_mae"]    is not None: parts.append(f'<div class="metrics-pill">val MAE <strong>${health["val_mae"]:.2f}</strong></div>')
    if health["val_med_ae"] is not None: parts.append(f'<div class="metrics-pill">val MedAE <strong>${health["val_med_ae"]:.2f}</strong></div>')
    if health["val_r2"]     is not None: parts.append(f'<div class="metrics-pill">R² <strong>{health["val_r2"]:.3f}</strong></div>')
    if health["val_rmse"]   is not None: parts.append(f'<div class="metrics-pill">val RMSE <strong>{health["val_rmse"]:.2f}</strong></div>')
    if health["test_rmse"]  is not None: parts.append(f'<div class="metrics-pill">test RMSE <strong>{health["test_rmse"]:.2f}</strong></div>')
    st.markdown(
        f'<div class="metrics-row">' + "".join(parts) + '</div></div></section>',
        unsafe_allow_html=True,
    )
else:
    st.markdown('</div></section>', unsafe_allow_html=True)

# ── Status banner ─────────────────────────────────────────────────────────────

if not health["reachable"]:
    st.error("La API no responde. Levanta el servicio con `uvicorn src.api.main:app --reload`.")
elif not health["model_loaded"]:
    st.warning("API activa, pero no cargó el artefacto. Verifica que `nyc_taxi_fare_production.joblib` exista en `data/models/`.")
else:
    st.success("API conectada y modelo productivo cargado.")

# ── Info cards ────────────────────────────────────────────────────────────────

metric_cols = st.columns(4, gap="medium")
with metric_cols[0]:
    render_metric_card("Estado API", health["api_status"], health["detail"])
with metric_cols[1]:
    sr = health["sample_rows"]
    sr_fmt = f"{sr:,}" if sr else "—"
    render_metric_card(
        "Datos de entrenamiento",
        f"{sr_fmt} filas",
        "Muestra estratificada por año × flota, 2015–2023. OHE completo sobre ~100% de rutas únicas.",
    )
with metric_cols[2]:
    mae_val = f"${health['val_mae']:.2f}" if health["val_mae"] else "—"
    med_val = f"${health['val_med_ae']:.2f}" if health["val_med_ae"] else "—"
    render_metric_card(
        "Calidad del modelo (val 2024)",
        f"MAE {mae_val}",
        f"Mediana absoluta: {med_val}. El MAE y MedAE miden el error típico por predicción, "
        "robustos frente al heavy tail de la distribución de tarifas.",
    )
with metric_cols[3]:
    render_metric_card(
        "Contrato de features",
        "Pre-viaje · Anti-leakage",
        "Contrato v4: pickup_datetime, trip_type, distancia estimada, zonas, vendor, ratecode. "
        "total_amount, tip, duración y payment_type excluidos por contrato.",
    )

st.write("")

# ── Main layout ───────────────────────────────────────────────────────────────

form_col, side_col = st.columns([1.45, 0.9], gap="large")

with form_col:
    with st.form("trip_form"):
        st.markdown("### Simulador de viaje")
        st.caption(
            "Completa un viaje realista. Todos los campos se envían directamente al endpoint "
            "`POST /predict`. Las zonas corresponden al listado oficial TLC."
        )

        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("#### Ruta y flota")

            trip_type = st.selectbox(
                "Tipo de taxi",
                options=["yellow", "green"],
                index=0,
                help="Yellow opera en Manhattan y aeropuertos; Green en outer boroughs.",
            )

            # Zona pickup — nombre oficial TLC
            pu_default_idx = zone_options.index(237) if 237 in zone_options else 0
            pu_label = st.selectbox(
                "Zona de recogida",
                options=zone_labels,
                index=pu_default_idx,
                help="Búsqueda por nombre de zona o borough. Fuente: TLC taxi_zone_lookup.",
            )
            pickup_location_id = zone_options[zone_labels.index(pu_label)]

            # Zona dropoff — nombre oficial TLC
            do_default_idx = zone_options.index(141) if 141 in zone_options else 0
            do_label = st.selectbox(
                "Zona de destino",
                options=zone_labels,
                index=do_default_idx,
                help="Búsqueda por nombre de zona o borough. Fuente: TLC taxi_zone_lookup.",
            )
            dropoff_location_id = zone_options[zone_labels.index(do_label)]

            pickup_date = st.date_input(
                "Fecha de recogida",
                value=dt.date(2025, 1, 15),
                help="Fecha esperada del inicio del trayecto.",
            )
            pickup_time = st.time_input(
                "Hora de recogida",
                value=dt.time(14, 35),
                help="Hora estimada del pickup.",
            )

        with right:
            st.markdown("#### Parámetros del viaje")

            estimated_distance = st.number_input(
                "Distancia estimada (millas)",
                min_value=0.1,
                value=7.8,
                step=0.1,
                format="%.1f",
                help="Distancia aproximada antes de iniciar el viaje (proxy de trip_distance histórico).",
            )
            passenger_count = st.number_input(
                "Pasajeros",
                min_value=1, max_value=8, value=2,
                help="Entero entre 1 y 8.",
            )
            vendor_id = st.selectbox(
                "Proveedor (Vendor)",
                options=[1, 2],
                format_func=lambda v: f"{v} — {VENDOR_LABELS.get(v, 'Unknown')}",
                index=0,
                help="Empresa proveedora del sistema de despacho.",
            )
            ratecode_id = st.selectbox(
                "Código de tarifa",
                options=[1, 2, 3, 4, 5, 6],
                format_func=lambda r: f"{r} — {RATECODE_LABELS.get(r, 'Unknown')}",
                index=0,
                help="Régimen tarifario. JFK (2) y Newark (3) tienen tarifas planas o especiales.",
            )

            st.markdown(
                '<p class="subtle-note">Variables post-viaje como '
                '<code>tip_amount</code>, <code>total_amount</code> y '
                '<code>trip_distance</code> observada no se aceptan — '
                'están bloqueadas por contrato anti-leakage.</p>',
                unsafe_allow_html=True,
            )

        submitted = st.form_submit_button(
            "Calcular tarifa estimada",
            use_container_width=True,
            disabled=not health["reachable"],
        )

    # ── Submission logic ──────────────────────────────────────────────────────
    if submitted:
        pickup_datetime = dt.datetime.combine(pickup_date, pickup_time).strftime("%Y-%m-%d %H:%M:%S")
        payload = {
            "trip_type": str(trip_type),
            "pickup_datetime": pickup_datetime,
            "pickup_location_id": int(pickup_location_id),
            "dropoff_location_id": int(dropoff_location_id),
            "passenger_count": int(passenger_count),
            "estimated_distance": float(estimated_distance),
            "vendor_id": int(vendor_id),
            "ratecode_id": int(ratecode_id),
        }

        if pickup_location_id == dropoff_location_id and estimated_distance > 12:
            st.warning(
                "Origen y destino son la misma zona pero la distancia es alta. "
                "Verifica el caso antes de documentar."
            )
        if estimated_distance >= 35:
            st.info("Estás probando una distancia larga — puede ser válido para aeropuertos o viajes suburbanos.")

        status_box = st.status("Preparando consulta de inferencia...", expanded=True)
        try:
            status_box.write("Validando payload pre-envío.")
            response = requests.post(API_URL, json=payload, timeout=15)
            status_box.write("Esperando respuesta de `POST /predict`.")
            response.raise_for_status()
            data = response.json()
            fare  = float(data.get("estimated_fare_amount"))
            model = str(data.get("model", "—"))
            st.session_state.latest_result = {"fare": fare, "model": model, "payload": payload}
            status_box.update(label="Predicción completada", state="complete", expanded=False)
            st.toast(f"Tarifa estimada: ${fare:,.2f}")
        except requests.HTTPError as exc:
            status_box.update(label="Error de API", state="error", expanded=True)
            st.error(f"HTTP {exc.response.status_code}: {exc.response.text}")
        except requests.RequestException as exc:
            status_box.update(label="Sin conexión con la API", state="error", expanded=True)
            st.error(f"No se pudo conectar: {exc}")

# ── Side panel ────────────────────────────────────────────────────────────────

with side_col:
    if st.session_state.latest_result:
        render_result_panel(st.session_state.latest_result, zones)
    else:
        render_empty_result_panel()

    st.markdown(
        '<div class="side-panel">'
        '<div class="panel-kicker">Checklist operativo</div>'
        '<div class="panel-title">Antes de documentar</div>'
        '<p class="panel-copy">Usa esta UI como smoke test antes de la defensa.</p>'
        '<ul class="bullet-list">'
        '<li><code>/health</code> responde y confirma modelo cargado y métricas.</li>'
        '<li>La predicción llega sin error HTTP ni timeout.</li>'
        '<li>La tarifa es positiva y coherente con la distancia y el ratecode.</li>'
        '<li>Prueba al menos un viaje JFK (ratecode=2, ~18 mi) y uno urbano (ratecode=1, ~3 mi).</li>'
        '<li>Prueba con Yellow y Green para confirmar que ambas flotas responden.</li>'
        '</ul>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="side-panel">'
        '<div class="panel-kicker">Política anti-leakage</div>'
        '<div class="panel-title">Contrato de entrada v4</div>'
        '<p class="panel-copy">'
        'Features permitidas: <code>pickup_datetime</code>, <code>trip_type</code>, '
        '<code>pickup_location_id</code>, <code>dropoff_location_id</code>, '
        '<code>estimated_distance</code>, <code>passenger_count</code>, '
        '<code>vendor_id</code>, <code>ratecode_id</code>.<br><br>'
        'Bloqueadas por contrato: <code>total_amount</code>, <code>tip_amount</code>, '
        '<code>congestion_surcharge</code>, <code>trip_duration_min</code>, '
        '<code>payment_type</code> y cualquier variable post-pago.'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.latest_result:
        with st.expander("Ver payload JSON enviado", expanded=False):
            st.json(st.session_state.latest_result["payload"])
