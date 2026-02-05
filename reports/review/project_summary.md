# Resumen del proyecto (AG1 + AG2) – US24

## Propósito del MVP
El MVP permite predecir si un cliente **realizará un depósito** a partir de variables de contexto/campaña, con dos canales de uso:
- **API (FastAPI)** para inferencia online.
- **UI (Streamlit)** para interacción de usuario final.

---

## Ítems desarrollados (dos iteraciones)

### Iteración 1 (AG1) – Preparación / Entendimiento / Base del pipeline
- Ingesta y preparación de datos.
- Transformaciones y generación de dataset de entrenamiento.
- Entrenamiento inicial del modelo y artefactos (modelo + preprocesamiento).
- Estructura de repositorio y pipeline con enfoque reproducible (DVC).
- Documentación inicial del proceso y reportes de transformación.

### Iteración 2 (AG2) – Modelado / Evaluación / Despliegue / Operación
- Experimentación y trazabilidad (MLflow) de modelos/parametría/métricas.
- Selección de modelo final y baseline de métricas (`metrics/baseline_eval_metrics.json`).
- Pruebas de regresión/validación (si aplica) para control de calidad.
- Despliegue del MVP:
  - API FastAPI en Render
  - UI Streamlit en Streamlit Community Cloud
- Contrato I/O y endpoints documentados.
- Propuesta de monitoreo y mantenimiento (drift, SLA/latencia, alertas, retraining) con tabla de alertas.

---

## Evidencias clave (entrega)
- URLs productivas (API/UI) + capturas (UI + /health + /docs + logs Render).
- Tags de versionado por historia y release final.
- Reproducibilidad demostrada con `dvc repro`.

---

## Resultado para decisión
Los resultados de Iteración 2 habilitan una **fase de verificación** con datos reales:
- comparar predicción vs métodos actuales de la organización,
- medir efectividad y definir pasos para adopción.
