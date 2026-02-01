# US19 – Quality checklist (Iteración 2)

**Fase CRISP-DM:** Evaluación / Despliegue (preparación)  
**Método:** SCRUM / DataOps / MLOps  
**Objetivo:** Verificar consistencia, reproducibilidad y preparación del MVP.

---

## 1. Estructura del repositorio
- [ ] Existen carpetas base (`src/`, `data/`, `models/`, `metrics/`, `reports/`, `tests/`, `docs/`).
- [ ] Las rutas usadas por scripts y pipeline son relativas al repo (evitar rutas absolutas).
- [ ] Existen `.gitignore`/`.dvcignore` coherentes para evitar ruido y conflictos Git vs DVC.

## 2. Reproducibilidad (DVC)
- [ ] `dvc repro` ejecuta el pipeline de punta a punta sin errores.
- [ ] `dvc dag` muestra un grafo consistente (sin salidas duplicadas entre stages).
- [ ] `dvc.lock` se actualiza únicamente tras reproducir cambios reales.
- [ ] Los outputs pesados o generados (modelos, métricas, reportes) están bajo DVC y no bajo Git.

## 3. Calidad de datos (scripts)
- [ ] La visualización/EDA está implementada en scripts (no solo notebooks).
- [ ] Las transformaciones/limpieza están implementadas en scripts.
- [ ] La verificación de calidad de datos se ejecuta vía pipeline (stages de pruebas).

## 4. Entrenamiento y artefactos (MVP)
- [ ] `train_model` genera `models/model.pkl` y `models/preprocessor.pkl` (outputs bajo DVC).
- [ ] `metrics/train_metrics.json` se genera y se actualiza (output bajo DVC).
- [ ] Reportes de entrenamiento se generan en `reports/model/` (output bajo DVC).

## 5. Evaluación (US17/US18)
- [ ] `evaluate_model` genera `metrics/eval_metrics.json` (output bajo DVC).
- [ ] Evidencias de evaluación se generan en `reports/evaluation/` (output bajo DVC).
- [ ] Documentos de decisión y comparación están versionados en `docs/us18/`.

## 6. Regresión (MLOps)
- [ ] Existe baseline para regresión de evaluación: `metrics/baseline_eval_metrics.json` (Git).
- [ ] Existe test de regresión: `tests/test_model_regression.py` (Git).
- [ ] El test de regresión está integrado al pipeline DVC (stage `regression_test`).
- [ ] Existe evidencia HTML del test: `reports/tests/us18_regression_report.html` (Git).

## 7. Tests (Pytest)
- [ ] `pytest -q` finaliza sin errores.
- [ ] Existe baseline de entrenamiento para tests: `metrics/metrics.json` (Git).
- [ ] El preprocessor es cargable desde `models/preprocessor.pkl` y transforma un sample sin fallar.

## 8. Trazabilidad (MLflow)
- [ ] Los runs se registran en `mlruns/` (local) con params/métricas.
- [ ] Los runs incluyen tags mínimos (`iteration`, `us`, `model_name`, `stage`).
- [ ] Artefactos relevantes (reportes/métricas/modelo) se adjuntan al run.

## 9. Consistencia MVP (API + UI)
- [ ] FastAPI consume el modelo publicado (mismo artefacto/versionado).
- [ ] Streamlit utiliza la API para inferencia con rutas coherentes.
- [ ] Las rutas de carga de modelo/preprocessor están estabilizadas (sin hardcode local).

---

## Resultado del checklist
- **Estado:** ✅ Completo / ⚠️ Con observaciones  
- **Observaciones clave:** (completar en peer review notes)
