# Checklist final de entrega (US24) – AP2 Release

## Objetivo
Dejar evidencia de cierre de Iteración 2 (Modelado, Evaluación, Despliegue y Operación) con:
- reproducibilidad del pipeline (DVC),
- verificación de predicciones en datos nuevos,
- enlaces de despliegue y tags de versión.

---

## 1) Reproducibilidad (local)

- [x] Ejecución completa: `dvc repro` sin errores  
  - Evidencia: captura de consola con stages ejecutados y estado OK.
- [x] Tests: `pytest -q` sin errores  
  - Evidencia: captura de consola con salida OK.  
  - Reporte (si aplica): `reports/tests/us18_regression_report.html`

Artefactos verificados tras `dvc repro`:
- [x] `models/model.pkl`
- [x] `models/preprocessor.pkl` *(si aplica)*
- [x] `models/feature_columns.json`
- [x] `metrics/baseline_eval_metrics.json`
- [x] `reports/` (carpetas generadas/actualizadas según pipeline)

---

## 2) Validación del MVP con datos nuevos (9 casos)

- [x] Se ejecutaron predicciones sobre **datos nuevos (sin clasificación previa)** usando API/UI.
- [x] Se registró el resumen solicitado por el docente (9 casos):
  - **No realizará un depósito:** 7 clientes (77.8%)
  - **Sí realizará un depósito:** 2 clientes (22.2%)

Evidencias mínimas:
- [x] Captura de la UI mostrando predicción (al menos 1 caso).
- [x] Captura de la API (POST `/predict`) o tabla resumen con los 9 resultados.

---

## 3) Evidencia de despliegue (producción)

- [x] API Render accesible:
  - [x] `/health` OK (modelo + schema cargados)
  - [x] `/docs` accesible
- [x] UI Streamlit accesible y conectada a la API (“API conectada”)
- [x] Predicción ejecutada desde la UI en producción

URLs (producción):
- API: https://api-13mbid-qmrc.onrender.com/
- Health: https://api-13mbid-qmrc.onrender.com/health
- Docs: https://api-13mbid-qmrc.onrender.com/docs
- UI: https://13mbid-abr2526-cenciso.streamlit.app/

---

## 4) Versionado y trazabilidad

Tags por historia:
- [x] `v1.7_us21_deployment_plan`
- [x] `v1.8_us22_monitoring_policy`
- [x] `v1.9_us23_memoria_final` *(si aplica / si se creó)*
- [x] **Tag final release:** `v2.0_ap2_release`

Trazabilidad:
- [x] Commit release: `484001e` *(US24: QA final)*
- [x] `docs/deploy_links.md` actualizado con URLs + Commit SHA + fecha.
- [x] Tags presentes en remoto (`git ls-remote --tags origin`).
- [x] Evidencias (capturas) incorporadas en la Memoria final (PDF).

---

## Resultado
Checklist completo, reproducible y verificable para auditoría y entrega final.
