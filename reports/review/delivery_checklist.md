# Checklist final de entrega (US24) – AP2 Release

## Objetivo
Dejar evidencia de cierre de Iteración 2 (Modelado, Evaluación, Despliegue y Operación) con:
- reproducibilidad del pipeline (DVC),
- verificación de predicciones en datos nuevos,
- enlaces de despliegue y tags de versión.

---

## 1) Reproducibilidad (local)

- [ ] Ejecución completa: `dvc repro` sin errores  
  - Evidencia: captura de consola con stages ejecutados y estado OK.
- [ ] Tests (si aplica): `pytest -q` sin errores  
  - Evidencia: captura de consola con salida OK.

Artefactos verificados tras `dvc repro`:
- [ ] `models/model.pkl`
- [ ] `models/preprocessor.pkl` *(si aplica)*
- [ ] `models/feature_columns.json`
- [ ] `metrics/baseline_eval_metrics.json`
- [ ] `reports/` (carpetas generadas/actualizadas según pipeline)

---

## 2) Validación del MVP con datos nuevos (9 casos)

- [ ] Se ejecutaron predicciones sobre **datos nuevos (sin clasificación previa)** usando API/UI.
- [ ] Se registró el resumen solicitado por docente (9 casos):
  - **No realizará un depósito:** 7 clientes (77.8%)
  - **Sí realizará un depósito:** 2 clientes (22.2%)

Evidencias mínimas:
- [ ] Captura de la UI mostrando predicción (al menos 1 caso).
- [ ] Captura de la API (POST /predict) o tabla resumen con los 9 resultados.

---

## 3) Evidencia de despliegue (producción)

- [ ] API Render accesible:
  - `/health` OK
  - `/docs` accesible
- [ ] UI Streamlit accesible y conectada a la API (“API conectada”)

URLs (producción):
- API: https://api-13mbid-qmrc.onrender.com/
- Health: https://api-13mbid-qmrc.onrender.com/health
- Docs: https://api-13mbid-qmrc.onrender.com/docs
- UI: https://13mbid-abr2526-cenciso.streamlit.app/

---

## 4) Versionado y trazabilidad

Tags por historia:
- [ ] `v1.7_us21_deployment_plan`
- [ ] `v1.8_us22_monitoring_policy`
- [ ] `v1.9_us23_memoria_final` *(si aplica)*
- [ ] **Tag final release:** `v2.0_ap2_release`

- [ ] `docs/deploy_links.md` actualizado con URLs + Commit SHA + fecha.
- [ ] Evidencias (capturas) incorporadas en la Memoria final (PDF).

---

## Resultado
Checklist completo, reproducible y verificable para auditoría y entrega final.
