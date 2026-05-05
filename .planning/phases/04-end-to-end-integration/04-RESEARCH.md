# Phase 4: End-to-End Integration - Research

**Researched:** 2026-05-05
**Domain:** FastAPI lifespan events, Streamlit HTTP client, local run orchestration, requirements management
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| INT-01 | `src/api/main.py` carga automáticamente el artefacto desde `data/models/` al iniciar | API already has `load_artifacts()` wired to `@app.on_event("startup")` — needs migration to `lifespan` pattern to remove deprecation warning. Model discovery via `_find_latest_model()` is correct. |
| INT-02 | Sistema completo corre con `uvicorn src.api.main:app` + `streamlit run app/frontend.py` | Both commands work today. Missing: `run.sh` wrapper and updated README section with exact commands. |
| INT-03 | `requirements.txt` incluye xgboost, lightgbm y todas las dependencias necesarias | xgboost>=2.0.0 and lightgbm>=4.0.0 already present. **Critical gap: `streamlit` is absent from requirements.txt.** Also `requests` (used by frontend) is absent. |
</phase_requirements>

---

## Summary

Phase 4 is an integration and polish phase — not a build phase. The hard work (API, model loading, frontend, feature pipeline) was completed in Phases 1-3. The codebase already has a working `src/api/main.py` that discovers and loads the joblib artifact, and `app/frontend.py` that calls the API and renders predictions. A manual end-to-end test was successfully executed: model loaded, prediction returned `$31.93` for a sample trip.

Three gaps remain before the success criteria are fully met:

1. `@app.on_event("startup")` in FastAPI 0.128 triggers a `DeprecationWarning` — the planner should decide whether to migrate to `lifespan` or leave it (functionally identical, purely cosmetic for a demo).
2. `requirements.txt` is missing `streamlit>=1.35.0` and `requests>=2.28.0` (both used by `app/frontend.py`).
3. No `run.sh` or clear README section with the two-command startup sequence exists.

**Primary recommendation:** Fix requirements.txt, add `run.sh`, and update README. Optionally migrate `on_event` to `lifespan` to silence the deprecation warning cleanly.

---

## Current State Audit (CRITICAL for Planning)

### What Already Works (verified 2026-05-05)

| Component | File | Status | Verified |
|-----------|------|--------|---------|
| API startup loads model | `src/api/main.py` | Working | `load_artifacts()` returns MODEL=xgboost, MODEL_NAME="xgboost" |
| `/health` endpoint | `src/api/main.py:68` | Correct shape | Returns `{"status":"ok","model_loaded":true,"model_name":"xgboost"}` |
| `/predict` endpoint | `src/api/main.py:73` | Working | Returns `{"estimated_fare_amount":31.93,"model":"xgboost"}` |
| Frontend form | `app/frontend.py` | Correct fields | location_id, no leakage, date+time pickers |
| Frontend API call | `app/frontend.py:78` | Correct | POST to `http://127.0.0.1:8000/predict` |
| Model artifact | `data/models/nyc_taxi_fare_baseline.joblib` | Present | 683 KB, model_name="xgboost", has `all_models` metrics |
| Feature pipeline | `src/features/build_features.py` | Complete | `select_raw_feature_columns` + `PickupTimeFeatures` chain |
| XGBoost in requirements | `requirements.txt` | Present | `xgboost>=2.0.0` |
| LightGBM in requirements | `requirements.txt` | Present | `lightgbm>=4.0.0` |

### What Is Missing (gaps to close in Phase 4)

| Gap | File | Severity |
|-----|------|---------|
| `streamlit` missing from requirements.txt | `requirements.txt` | HIGH — frontend cannot be installed from requirements alone |
| `requests` missing from requirements.txt | `requirements.txt` | HIGH — frontend import fails without it |
| No `run.sh` or startup instructions in README | root | MEDIUM — INT-02 success criterion |
| `@app.on_event` deprecated in FastAPI 0.128 | `src/api/main.py:55` | LOW — prints warning on startup, does not break functionality |

---

## Standard Stack

### Core (already in use)
| Library | Version (installed) | Purpose | Notes |
|---------|--------------------|---------|----|
| fastapi | 0.128.0 | REST API framework | Already wired; `on_event` deprecated, `lifespan` preferred |
| uvicorn | 0.39.0 | ASGI server | `uvicorn src.api.main:app --reload` |
| pydantic | >=2.0.0 | Request validation | `TripInput` already using v2 API (`model_dump`, `model_config`) |
| streamlit | 1.57.0 (latest) | Frontend UI | **Not in requirements.txt** |
| requests | 2.32.5 (installed) | HTTP client in frontend | **Not in requirements.txt** |
| joblib | >=1.3.0 | Artifact serialization | In requirements.txt |

### Installation Fix
```bash
# Add to requirements.txt:
streamlit>=1.35.0
requests>=2.28.0
```

---

## Architecture Patterns

### INT-01: API Artifact Loading

The current implementation uses `@app.on_event("startup")` which is functionally correct but deprecated in FastAPI 0.95+. Two options:

**Option A — Keep `on_event` (minimal change):**
```python
# src/api/main.py — current, works but prints DeprecationWarning
@app.on_event("startup")
def load_artifacts() -> None:
    global MODEL, MODEL_NAME
    model_path = _find_latest_model()
    ...
```

**Option B — Migrate to `lifespan` (recommended, silences warning):**
```python
# src/api/main.py — FastAPI lifespan pattern (FastAPI docs, 2024)
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    global MODEL, MODEL_NAME
    model_path = _find_latest_model()
    if model_path:
        MODEL = load_model(model_path)
        MODEL_NAME = MODEL.get("model_name", "unknown") if isinstance(MODEL, dict) else "unknown"
    yield
    # shutdown (nothing needed)

app = FastAPI(title="NYC Taxi Fare Prediction API", version="2.0", lifespan=lifespan)
```

**Decision for planner:** Option B is cleaner and future-proof. Recommend migrating. The `on_event` version still runs correctly if left alone.

### INT-02: Two-Command Startup Pattern

The standard local run pattern for this stack:

```bash
# Terminal 1
uvicorn src.api.main:app --reload

# Terminal 2
streamlit run app/frontend.py
```

`run.sh` should encode this as two background processes with a note that they require separate terminals (or use `&` with trap for cleanup):

```bash
#!/usr/bin/env bash
# run.sh — start the full system
set -e
echo "Starting NYC Taxi Fare Prediction System..."
echo ""
echo "Run these in separate terminals:"
echo "  Terminal 1: uvicorn src.api.main:app --reload"
echo "  Terminal 2: streamlit run app/frontend.py"
```

Alternatively, a single-script approach using background processes:

```bash
#!/usr/bin/env bash
uvicorn src.api.main:app --reload &
API_PID=$!
streamlit run app/frontend.py
kill $API_PID
```

The simpler "instructions" approach is safer for a demo and avoids process management complexity.

### INT-03: Requirements.txt Pattern

Standard practice: all runtime imports must appear in `requirements.txt`. The current file is missing two packages that `app/frontend.py` imports directly:

```
# app/frontend.py line 3:
import requests        # MISSING from requirements.txt
import streamlit as st # MISSING from requirements.txt
```

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ASGI lifecycle hooks | Custom middleware for model load | `lifespan` context manager | FastAPI-native, zero extra deps |
| HTTP client for frontend | `urllib.request` or `http.client` | `requests` (already installed) | Error handling, JSON, timeout all built-in |
| Process management for demo | Custom daemon manager | Simple `run.sh` + two terminals | Complexity adds no value for academic demo |

**Key insight:** All integration work in Phase 4 is wiring and packaging — not new functionality. Any "clever" solution is over-engineering.

---

## Common Pitfalls

### Pitfall 1: MODEL global not visible after refactor to lifespan
**What goes wrong:** If `lifespan` function does `MODEL = ...` but `MODEL` is not declared `global` inside the function, the assignment creates a local variable. The `/predict` endpoint sees `None`.
**Why it happens:** Python closure scoping — inner function assignment shadows the module global.
**How to avoid:** Declare `global MODEL, MODEL_NAME` at the top of the lifespan startup block.
**Warning signs:** `/health` returns `{"model_loaded": false}` even after startup logs succeed.

### Pitfall 2: Streamlit reruns on every widget interaction
**What goes wrong:** Every slider/selectbox change re-runs the entire `frontend.py` script, including the health check `requests.get(HEALTH_URL)`. With a 3-second timeout this adds latency on every interaction.
**Why it happens:** Streamlit's execution model re-runs the full script on state change.
**How to avoid:** The current code uses `timeout=3` which is acceptable. Do not increase. Alternatively wrap health check in `st.cache_resource` — but that's out of scope for Phase 4.
**Warning signs:** UI feels sluggish when changing dropdown values.

### Pitfall 3: uvicorn --reload breaks relative paths
**What goes wrong:** `uvicorn src.api.main:app --reload` must be run from the project root. If run from `src/`, the `_find_latest_model()` path `data/models/` resolves incorrectly.
**Why it happens:** `Path("data/models")` is relative to `cwd`, not the module file.
**How to avoid:** Document in `run.sh` and README: always run from project root. The `MODEL_DIR` env var can override if needed.
**Warning signs:** `/health` returns `{"model_loaded": false}` — model dir not found.

### Pitfall 4: snowflake-connector-python import fails at startup
**What goes wrong:** `requirements.txt` includes `snowflake-connector-python[pandas]` but Snowflake is not available locally. If any import at module load time tries to connect, startup fails.
**Why it happens:** The current code in `src/utils/snowflake_conn.py` and `src/utils/config.py` uses `load_dotenv()` and `validate_required_settings()` which raises `ValueError` if Snowflake env vars are missing.
**How to avoid:** The API currently calls `load_model` and `_find_latest_model` — neither touches Snowflake config. The `get_settings()` call with `validate=True` is NOT called at startup. This is already safe. Do not add any Snowflake import to the startup path.
**Warning signs:** ImportError or ValueError about missing SNOWFLAKE_ACCOUNT at `uvicorn` startup.

### Pitfall 5: Missing streamlit causes silent failure
**What goes wrong:** If `streamlit` is not in `requirements.txt` and someone does a fresh `pip install -r requirements.txt`, running `streamlit run app/frontend.py` fails with `command not found`.
**Why it happens:** The package is installed separately by the developer but not recorded as a dependency.
**How to avoid:** Add `streamlit>=1.35.0` and `requests>=2.28.0` to `requirements.txt` (INT-03).

---

## Code Examples

### Lifespan migration (FastAPI docs pattern)
```python
# Source: https://fastapi.tiangolo.com/advanced/events/
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup
    global MODEL, MODEL_NAME
    model_path = _find_latest_model()
    if model_path is not None:
        try:
            MODEL = load_model(model_path)
            MODEL_NAME = MODEL.get("model_name", "unknown") if isinstance(MODEL, dict) else "unknown"
        except Exception:
            MODEL = None
    yield
    # Runs once at shutdown (nothing to clean up)

app = FastAPI(title="NYC Taxi Fare Prediction API", version="2.0", lifespan=lifespan)
```

### run.sh (simple, reliable for demo)
```bash
#!/usr/bin/env bash
set -e
echo "NYC Taxi Fare Predictor — startup instructions"
echo "================================================"
echo ""
echo "1. Start the API (Terminal 1):"
echo "   uvicorn src.api.main:app --reload"
echo ""
echo "2. Start the UI (Terminal 2):"
echo "   streamlit run app/frontend.py"
echo ""
echo "3. Open: http://localhost:8501"
echo "4. Health: http://localhost:8000/health"
```

### Health check verification
```bash
# After starting API:
curl http://localhost:8000/health
# Expected: {"status":"ok","model_loaded":true,"model_name":"xgboost"}

# Manual predict test:
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pickup_datetime":"2025-01-15 14:35:00","pickup_location_id":237,"dropoff_location_id":141,"passenger_count":2,"trip_distance":7.8,"vendor_id":1,"ratecode_id":1}'
# Expected: {"estimated_fare_amount":31.93,"model":"xgboost"}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|---|---|---|---|
| `@app.on_event("startup")` | `lifespan` context manager | FastAPI 0.95 (2023) | `on_event` still works but prints DeprecationWarning in 0.128 |
| Pydantic v1 `.dict()` | Pydantic v2 `.model_dump()` | Pydantic 2.0 (2023) | Already using v2 API — no action needed |

**Deprecated/outdated in this codebase:**
- `@app.on_event("startup")`: Still functional in FastAPI 0.128, deprecated since 0.95. Will be removed in a future major version. The `lifespan` pattern is the replacement.

---

## Open Questions

1. **Should `run.sh` attempt to start both processes, or just print instructions?**
   - What we know: Single-script background approach works (`uvicorn & streamlit; kill $!`) but requires signal handling to clean up properly.
   - What's unclear: Whether the grader/demo will use `run.sh` directly or just read it.
   - Recommendation: Print-instructions approach is safer and clearer for an academic demo.

2. **Should `@app.on_event` be migrated to `lifespan`?**
   - What we know: Both work. `on_event` prints a deprecation warning but does not affect behavior.
   - Recommendation: Yes, migrate — it removes a distracting warning from `uvicorn` output and is a 10-line change.

3. **Should `snowflake-connector-python` be removed from requirements.txt?**
   - What we know: No code path in the demo run imports it.
   - What's unclear: Whether removing it would break something in the Snowflake path (future v2 work).
   - Recommendation: Leave it — removing it is out of scope for Phase 4 (INT-03 only asks to ADD missing packages, not remove).

---

## Sources

### Primary (HIGH confidence)
- Direct code audit of `src/api/main.py`, `app/frontend.py`, `requirements.txt`, `src/features/build_features.py`, `src/models/predict_model.py` — all verified 2026-05-05
- Live execution test: `load_artifacts()` + `predict()` returning `31.93` — verified 2026-05-05
- FastAPI installed version: 0.128.0 — verified via `pip show fastapi`
- FastAPI `on_event` deprecation: confirmed via live DeprecationWarning in Python 3.13

### Secondary (MEDIUM confidence)
- FastAPI lifespan docs pattern: https://fastapi.tiangolo.com/advanced/events/ — pattern confirmed by deprecation warning message text pointing to this URL

### Tertiary (LOW confidence)
- None

---

## Metadata

**Confidence breakdown:**
- Current state audit: HIGH — verified by running code and inspecting files
- Standard stack versions: HIGH — verified via pip show and live import
- Pitfalls: HIGH — Pitfall 4 (Snowflake) and Pitfall 1 (global scope) confirmed by code reading; others confirmed by known Python/FastAPI behavior
- Architecture patterns: HIGH — lifespan pattern from official FastAPI docs (URL in warning message)

**Research date:** 2026-05-05
**Valid until:** 2026-06-05 (stable ecosystem — FastAPI, Streamlit, requests change slowly)
