# US19 – Peer review notes (Iteración 2)

**Fecha:** 2026-01-31  
**Alcance:** Pipeline DVC + scripts de datos + entrenamiento/evaluación + tests + trazabilidad (MLflow) + coherencia MVP (API/UI).

---

## 1. Resumen de ejecución reproducible

### DVC
- Comando: `dvc repro`
- Resultado: **(pendiente de completar con evidencia de consola / sin errores)**

### Pytest
- Comando: `pytest -q`
- Resultado: **(pendiente de completar con evidencia de consola / sin errores)**

---

## 2. Hallazgos (Findings)

### 2.1 Reproducibilidad y outputs (Git vs DVC)
- Hallazgo: Validar que outputs generados por pipeline no estén versionados en Git para evitar el error `output already tracked by SCM`.
- Estado: **(OK / pendiente)**
- Acción: **(si aplica)**

### 2.2 DVC stages y salidas duplicadas
- Hallazgo: Verificar que no existan outputs solapados entre `model_selection` y `train_model`.
- Estado: **(OK / pendiente)**
- Acción: **(si aplica)**

### 2.3 Tests y baselines
- Hallazgo: Baseline de entrenamiento para `tests/test_training.py` debe estar versionado en Git (`metrics/metrics.json`) y permitido por `.gitignore`.
- Estado: **(OK / pendiente)**
- Acción: **(si aplica)**

### 2.4 MLflow (trazabilidad)
- Hallazgo: Confirmar runs con params/métricas/artefactos y tags mínimos.
- Estado: **(OK / pendiente)**
- Evidencia: **(capturas o referencia local)**

### 2.5 Coherencia MVP (API/UI)
- Hallazgo: Confirmar que FastAPI/UI apuntan al mismo modelo y preprocessor publicados.
- Estado: **(OK / pendiente)**
- Acción: **(si aplica)**

---

## 3. Acciones correctivas aplicadas (si existieran)
- **AC-01:** (describir)  
- **AC-02:** (describir)

---

## 4. Evidencias
- Logs `dvc repro`: (ruta o captura)
- Logs `pytest -q`: (ruta o captura)
- Capturas MLflow: (ruta o captura)
