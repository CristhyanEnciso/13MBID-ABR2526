# Resumen del proyecto (AG1 + AG2) – US24

## Propósito del MVP
El MVP permite predecir si un cliente **realizará un depósito** a partir de variables de contexto/campaña. Se entrega con dos canales de uso:
- **API (FastAPI)** para inferencia online.
- **UI (Streamlit)** para interacción de usuario final consumiendo la API.

---

## Ítems desarrollados (dos iteraciones)

### Iteración 1 (AG1) – CRISP-DM: Entendimiento / Preparación de datos
- Ingesta del dataset y análisis descriptivo/EDA (scripts + reportes).
- Validaciones de calidad de datos (tests integrados al pipeline).
- Transformaciones, limpieza, construcción e integración de features (pipeline reproducible con DVC).
- Generación del dataset final de entrenamiento (`data/processed/bank_formatted.csv`).
- Estructura base del repositorio y evidencias en `data/`, `reports/`, `docs/`, `src/`, `tests/`.

### Iteración 2 (AG2) – CRISP-DM: Modelado / Evaluación / Despliegue / Operación
- **Experimentación y trazabilidad (MLflow)** de alternativas de modelado + shortlist (`reports/selection/shortlist_modelos.csv`).
- Entrenamiento del modelo final y generación de artefactos (`models/model.pkl`, `models/feature_columns.json`).
- Evaluación del modelo y baseline para control de calidad (`metrics/eval_metrics.json`, `metrics/baseline_eval_metrics.json`).
- **Test de regresión** integrado al pipeline y evidencia HTML (`metrics/regression_test.json`, `reports/tests/us18_regression_report.html`).
- Despliegue del MVP:
  - **API FastAPI en Render** (endpoints `/health`, `/docs`, `/predict`).
  - **UI Streamlit** en Streamlit Community Cloud, conectada a la API.
- Contrato I/O + plan de despliegue (US21) y propuesta de operación (US22):
  - drift (data/score/label si aplica), SLA/latencia, alertas y política de retraining (incluye tabla CSV de alertas).

---

## Evidencias clave (entrega)
- **Reproducibilidad** demostrada con `dvc repro` y ejecución de `pytest -q` (capturas en Memoria).
- **Enlaces de despliegue** y trazabilidad en `docs/deploy_links.md` (incluye URLs, Commit SHA y fecha).
- Documentos de soporte:
  - US21 (plan despliegue): `docs/deployment_plan.md`
  - US22 (monitoreo/mantenimiento): `docs/monitoring_maintenance.md` + `reports/monitoring/monitoring_plan_table.csv`
  - US24 (QA final): `reports/review/delivery_checklist.md`

---

## URLs productivas (producción)
- **API Render:** https://api-13mbid-qmrc.onrender.com/
- **Health:** https://api-13mbid-qmrc.onrender.com/health
- **Docs:** https://api-13mbid-qmrc.onrender.com/docs
- **UI Streamlit:** https://13mbid-abr2526-cenciso.streamlit.app/

---

## Versionado y release
Tags principales:
- `v1.7_us21_deployment_plan` (plan + despliegue)
- `v1.8_us22_monitoring_policy` (monitoreo/mantenimiento)
- `v2.0_ap2_release` (release final AP2)

Trazabilidad:
- **Commit release (tag `v2.0_ap2_release`):** `<COMMIT_RELEASE>`

---

## Resultados de revisión del MVP (según indicación del docente)
Sobre **9 casos nuevos (sin clasificación previa)**:
- **No realizará un depósito:** 7 clientes (77.8%)
- **Sí realizará un depósito:** 2 clientes (22.2%)

> Estos resultados se mantienen al usar las herramientas generadas (API/UI) y quedan listos para validarse contra datos reales (ground truth) en una fase posterior.

---

## Resultado para decisión
Los resultados de Iteración 2 habilitan una **fase de verificación** con datos reales:
- comparar predicción vs métodos actuales de la organización,
- medir efectividad y definir pasos para adopción (piloto, monitoreo y posible retraining).
