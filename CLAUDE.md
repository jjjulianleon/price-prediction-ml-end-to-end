<!-- GSD:project-start source:PROJECT.md -->
## Project

**NYC Taxi Price Prediction — End-to-End ML**

Sistema end-to-end de predicción de tarifas para viajes NYC Yellow Taxi, construido como proyecto final de Data Mining. Toma variables de entrada pre-viaje (sin leakage), entrena múltiples modelos de regresión y sirve predicciones vía API FastAPI + UI Streamlit.

**Core Value:** El usuario ingresa datos de un viaje en la UI y recibe una estimación del `fare_amount` — el sistema debe correr completo de extremo a extremo sin Snowflake para la demo.

### Constraints

- **Snowflake**: Sin credenciales activas — modo local obligatorio para poder desarrollar y demostrar
- **Timeline**: Proyecto académico — entrega inmediata, prioridad sobre elegancia
- **Rubrica**: Exige modelos boosting (XGBoost, LightGBM) como obligatorios; HistGradientBoosting ya existe
- **Data**: NYC TLC Yellow Taxi parquet mensual (~400MB por mes, descargable desde CDN oficial)
- **Anti-leakage**: payment_type, tip_amount, tolls_amount, total_amount nunca deben entrar al modelo
<!-- GSD:project-end -->

<!-- GSD:stack-start source:STACK.md -->
## Technology Stack

Technology stack not yet documented. Will populate after codebase mapping or first phase.
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
