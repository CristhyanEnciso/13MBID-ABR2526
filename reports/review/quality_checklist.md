# US19 – Quality checklist (Iteración 2)

**Fase CRISP-DM:** Evaluación / Despliegue (preparación)  
**Método:** SCRUM / DataOps / MLOps  
**Objetivo:** Verificar consistencia, reproducibilidad y preparación del MVP.

---

## 1. Estructura del repositorio
- [x] Existen carpetas base (`src/`, `data/`, `models/`, `metrics/`, `reports/`, `tests/`, `docs/`).
- [x] Las rutas usadas por scripts y pipeline son relativas al repo (evitar rutas absolutas).
- [x] Existen `.gitignore`/`.dvcignore` coherentes para evitar ruido y conflictos Git vs DVC.

## 2. Reproducibilidad (DVC)
- [x] `dvc repro` ejecuta el pipeline de punta a punta sin errores.
- [x] `dvc dag` muestra un grafo consistente (sin salidas duplicadas entre stages).
- [x] `dvc.lock` se actualiza únicamente tras reproducir cambios reales.
- [x] Los outputs pesados o generados (datasets procesados, modelos, métricas, reportes) están bajo control del pipeline (DVC) y no generan conflictos con Git.

## 3. Calidad de datos (scripts)
- [x] La visualización/EDA está implementada en scripts (no solo notebooks).
- [x] Las transformaciones/limpieza están implementadas en scripts.
- [x] La verificación de calidad de datos se ejecuta vía pipeline (stages de pruebas).

## 4. Entrenamiento y artefactos (MVP)
- [x] `train_model` genera `models/model.pkl` (artefacto del MVP).
- [x] `models/preprocessor.pkl` existe *(si aplica / informativo para el servicio)*.
- [x] `metrics/train_metrics.json` se genera y se actualiza.
- [x] Reportes de entrenamiento se generan en `reports/model/`.

## 5. Evaluación (US17/US18)
- [x] `evaluate_model` genera `metrics/eval_metrics.json`.
- [x] Evidencias de evaluación se generan en `reports/evaluation/` (curvas/insumos/artefactos).
- [x] Existe evidencia de corroboración/decisión en el repo (ej.: `reports/evaluation/corroboracion_resultados.md` y baseline en `metrics/baseline_eval_metrics.json`).

## 6. Regresión (MLOps)
- [x] Existe baseline de evaluación: `metrics/baseline_eval_metrics.json` (control de calidad).
- [x] Existe test de regresión: `tests/test_model_regression.py`.
- [x] El test de regresión está integrado al pipeline DVC (stage `regression_test`).
- [x] Existe evidencia HTML del test: `reports/tests/us18_regression_report.html`.
- [x] Existe métrica de estado de regresión: `metrics/regression_test.json`.

## 7. Tests (Pytest)
- [x] `pytest -q` finaliza sin errores.
- [x] Métricas relevantes existen y son trazables (`metrics/train_metrics.json`, `metrics/eval_metrics.json`, `metrics/baseline_eval_metrics.json`).
- [x] El pipeline produce artefactos necesarios para ejecutar inferencia sin fallar (modelo + schema).

## 8. Trazabilidad (MLflow)
- [x] Los runs se registran en `mlruns/` (local) con params/métricas.
- [x] Artefactos relevantes (métricas/reportes) se adjuntan al run (según configuración local).

## 9. Consistencia MVP (API + UI)
- [x] FastAPI consume el modelo publicado (artefacto coherente con el repositorio).
- [x] Streamlit utiliza la API para inferencia con rutas coherentes.
- [x] Las rutas de carga de artefactos están estabilizadas (sin hardcode local).

---

## Resultado del checklist
- **Estado:** ✅ Completo  
- **Observaciones clave:**  
  - *(Opcional para sobresaliente)* Si se desea mayor trazabilidad en MLflow, agregar tags mínimos por run (`iteration`, `us`, `stage`, `model_name`).  
