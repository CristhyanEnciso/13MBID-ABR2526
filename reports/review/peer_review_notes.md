# US19 – Peer review notes (Iteración 2)

**Fecha:** 2026-02-05  
**Alcance:** Pipeline DVC + scripts de datos + entrenamiento/evaluación + tests + trazabilidad (MLflow) + coherencia MVP (API/UI).

---

## 1. Resumen de ejecución reproducible

### DVC
- Comando: `dvc repro`
- Resultado: **OK – ejecución sin errores (ver captura en memoria)**

### Pytest
- Comando: `pytest -q`
- Resultado: **OK – 12 passed (ver captura en memoria)**

---

## 2. Hallazgos (Findings)

### 2.1 Reproducibilidad y outputs (Git vs DVC)
- Hallazgo: Validar que outputs del pipeline no estén versionados en Git cuando corresponda, evitando conflictos tipo `output already tracked by SCM`.
- Estado: **OK**
- Acción: **No aplica (control verificado).**

### 2.2 DVC stages y coherencia de salidas
- Hallazgo: Verificar coherencia de rutas de salida entre stages (p. ej., `model_selection` genera shortlist donde el pipeline lo espera).
- Estado: **OK**
- Acción: **Aplicada AC-01 (normalización de ruta de shortlist en DVC).**

### 2.3 Tests y baselines (versionado mínimo)
- Hallazgo: Asegurar que los baselines necesarios para tests se encuentren versionados en Git (p. ej., `metrics/metrics.json`, `metrics/baseline_eval_metrics.json`), sin introducir artefactos pesados.
- Estado: **OK**
- Acción: **Aplicada AC-03 (ajuste de reglas de ignore para permitir baselines).**

### 2.4 MLflow (trazabilidad)
- Hallazgo: Confirmar runs con params/métricas/artefactos y tags mínimos (iteración, us, stage, model_name, balancing_method).
- Estado: **OK**
- Evidencia: **<imagen: MLflow – runs con params/métricas/artefactos>**

### 2.5 Coherencia MVP (API/UI)
- Hallazgo: Confirmar que API/UI consumen el mismo modelo y configuración esperada.
- Estado: **OK**
- Acción: **Verificado mediante endpoints `/health` y documentación `/docs`, y prueba de predicción en UI.**
- Evidencia: **<imagen: /health OK>, <imagen: /docs>, <imagen: predicción en UI>**

---

## 3. Acciones correctivas aplicadas

- **AC-01:** Alineación de la ruta de salida del shortlist en DVC (`reports/selection/shortlist_modelos.csv`) para que coincida con lo generado por el script de experimentación y lo definido en `dvc.yaml`.
- **AC-02:** Separación explícita Git vs outputs generados: se mantienen en Git únicamente artefactos ligeros y de auditoría; se evitan conflictos con salidas del pipeline mediante reglas de ignore adecuadas.
- **AC-03:** Ajuste de `.gitignore` (carpeta `metrics/`) para permitir baselines necesarios para tests (`metrics.json`, `baseline_eval_metrics.json`) sin versionar artefactos pesados.
- **AC-04:** Compatibilidad entre `train_model()` y `tests/test_training.py`: se ajustó la interfaz/estructura de salida esperada para garantizar `pytest -q` sin errores, manteniendo reproducibilidad del pipeline.

---

## 4. Evidencias
- Logs `dvc repro`: <imagen: consola – dvc repro OK>
- Logs `pytest -q`: <imagen: consola – 12 passed>
- Capturas MLflow: <imagen: MLflow – runs>
- Evidencia API/UI: <imagen: /health OK>, <imagen: /docs>, <imagen: predicción en UI>
