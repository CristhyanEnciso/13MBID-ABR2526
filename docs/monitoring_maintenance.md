# Supervisión y mantenimiento (US22) – MVP API FastAPI + UI Streamlit

## Objetivo
Definir un plan **operable** de supervisión y mantenimiento del MVP desplegado (API + UI) para:
- Monitorear **salud del servicio** (SLA, latencia, errores, disponibilidad).
- Monitorear **calidad funcional** de predicciones (distribución de clases, inputs inválidos).
- Detectar **drift** (data drift y score drift; label drift si se dispone de ground truth posterior).
- Definir **alertas** (umbral, severidad y acción).
- Definir **política de retraining** (gatillos, proceso, versionado y rollback).

---

## Alcance del MVP monitoreado
### Componentes
- **API (FastAPI)**: expone endpoints `/health`, `/docs`, `/predict`.
- **UI (Streamlit)**: consume la API pública para generar predicciones.
- **Artefactos clave del modelo**:
  - `models/model.pkl`
  - `models/preprocessor.pkl` (informativo)
  - `models/feature_columns.json` (schema final one-hot)
  - `metrics/baseline_eval_metrics.json` (baseline para control de calidad)

### Fuera de alcance (por ahora)
- Implementación real de Prometheus/Grafana y alerting automatizado.
- Registro persistente de predicciones en una base de datos productiva.
- Captura de **labels reales** (puede depender del negocio).

> Este documento deja la propuesta lista para implementar en una iteración futura.

---

## 1) Monitoreo operativo (servicio)

### Métricas operativas mínimas (API)
**Qué medimos**
- **Disponibilidad**: % de tiempo con `/health` en estado OK.
- **Latencia**:
  - p50, p95, p99 de respuesta en `/predict`.
- **Errores**:
  - tasa de 5xx (fallos del servicio)
  - tasa de 4xx (errores controlados por validación/entrada)
- **Throughput** (opcional): requests/min (RPM) en `/predict`.

**SLA sugerido (simple y defendible)**
- Disponibilidad API: **≥ 99%**
- Latencia `/predict`:
  - p95 **< 800 ms**
- Error rate 5xx:
  - **< 1%** por ventana

**Frecuencia**
- Revisión diaria (manual) o semanal (si no hay automatización).
- En producción real: muestreo continuo + alertas en tiempo real.

---

## 2) Monitoreo funcional (predicciones e inputs)

### Métricas funcionales (inferencia online)
**Qué medimos**
- **Distribución de predicciones**:
  - ratio `yes` vs `no` por ventana (día/semana).
- **Distribución de probabilidades** (si aplica):
  - cambios bruscos en proba promedio o varianza (score drift).
- **Inputs inválidos**:
  - porcentaje de 422 (validación Pydantic)
  - categorías desconocidas (si aparecen; deberían mapearse a ceros por el one-hot)
- **Campos críticos**:
  - `duration` (muy influyente en dataset bancario típico): detectar valores fuera de rango esperado.

**Objetivo**
Detectar:
- Cambios extremos en la proporción `yes` sin explicación.
- Entrada “basura” o inputs fuera de distribución.
- Errores por cambios de contrato I/O o UI.

---

## 3) Monitoreo de drift

### 3.1 Data drift (features)
**Propuesta de método (práctico para MVP)**
- Calcular **PSI (Population Stability Index)** por feature numérica y/o score.
- Alternativa: **KS test** para variables numéricas principales.

**Umbrales sugeridos (PSI)**
- PSI < 0.10 → sin drift relevante
- 0.10 ≤ PSI < 0.20 → drift moderado (vigilar)
- PSI ≥ 0.20 → drift significativo (investigar / posible retraining)

**Frecuencia**
- Semanal (si el volumen de predicciones es bajo).
- Diaria si hay alto volumen.

**Requisito técnico**
Para medir drift real necesitas **capturar un log de inputs** (aunque sea en CSV/JSON) por ventana.
- Opción simple (futura): guardar cada request en `reports/monitoring/pred_requests_YYYYMMDD.csv`.
- Opción más robusta (futura): persistencia en BD/Storage.

### 3.2 Score drift (distribución de probabilidad)
- Registrar distribución del score (probabilidad de `yes`).
- Aplicar PSI o comparar percentiles (p50/p95) vs baseline.

### 3.3 Label drift (si aplica)
Si posteriormente se dispone de la etiqueta real (ground truth):
- Comparar desempeño real (accuracy, f1, roc_auc, etc.) contra:
  - `metrics/baseline_eval_metrics.json`
- Definir “degradación significativa” como delta en métrica clave (ej. -5% relativo o -0.03 absoluto).

> Si no se dispone de labels reales, el foco debe ser data/score drift + salud del servicio.

---

## 4) Alertas (umbral → severidad → acción)

### Tabla de alertas (resumen)
Las alertas se disparan por ventanas (ej. 15 min / 1 hora / 1 día) según la métrica.

- **Severidad P1**: impacto directo (servicio caído o 5xx alto).
- **Severidad P2**: degradación operativa (latencia alta).
- **Severidad P3**: señales de drift o comportamiento anómalo (investigar).

(Ver tabla exportable en `reports/monitoring/monitoring_plan_table.csv`).

---

## 5) Política de retraining (gatillos + proceso + versionado + rollback)

### Gatillos (triggers) de retraining
Se recomienda aplicar retraining si ocurre al menos uno:

1) **Degradación de métricas** respecto a baseline  
- Si se cuenta con labels reales:
  - Si una métrica clave cae por debajo del baseline en más de un delta definido
  - Ejemplo: F1 cae **≥ 0.05** o AUC cae **≥ 0.03**

2) **Drift significativo sostenido**
- PSI ≥ 0.20 en variables críticas o score drift
- sostenido por **2 periodos** consecutivos (ej. 2 semanas)

3) **Ventana temporal**
- Retraining mensual o trimestral (si el negocio lo requiere)
- Útil si el entorno económico/campañas cambian por temporada

### Proceso (pasos)
1) Recolectar datos recientes (nuevos inputs; y labels si existen).
2) Ejecutar pipeline de entrenamiento/validación (DVC).
3) Evaluar métricas y comparar con baseline (`metrics/baseline_eval_metrics.json`).
4) Pasar **quality gate** (aprobación para promover modelo).
5) Versionar:
   - actualizar artefacto `models/model.pkl` y/o `models/feature_columns.json` si corresponde
   - actualizar baseline si el nuevo modelo es el productivo
   - crear tag de release (ej. `vX.Y_us22_monitoring_policy` o `vX.Y_model_release`)
6) Desplegar (redeploy Render/Streamlit por commit).
7) Validar:
   - `/health` OK
   - predicción desde UI
   - métricas/latencia no degradadas

### Rollback (si falla)
Aplicar rollback si:
- 5xx persistente
- `/health` falla (model/schema)
- degradación de métricas post-release

Pasos:
1) Volver a tag estable anterior:
```bash
git checkout <tag_estable_anterior>
```
2. Re-deploy (Render manual deploy o redeploy por commit)  
3. Validación rápida:
   - `/health`
   - 1 predicción desde UI

---

## 6) Opciones adicionales de seguimiento

### Opción A: Data Quality Checks (antes de inferencia y/o en batch)
- Validar rangos (edad, `duration`, `euribor3m`, etc.)
- Validar categorías permitidas (`job`, `marital`, `education`, `month`, `day_of_week`, `poutcome`)
- Registrar % de registros con valores fuera de rango o categorías “raras”

### Opción B: Regression test del modelo
- Comparar métricas del modelo candidato vs baseline
- Si el candidato no supera el gate → no se promueve a producción
- Evidencia: reporte de métricas y comparación contra baseline

---

## Evidencias (recomendadas)
- Captura de `/health` en producción (API)
- Captura de UI prediciendo y mostrando probabilidades
- Baseline en `metrics/baseline_eval_metrics.json`
- Este documento (`docs/monitoring_maintenance.md`)
