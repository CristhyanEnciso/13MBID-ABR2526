# Plan de pruebas – Iteración 2 (Modelado/Evaluación)

## 1. Objetivo
Se define un conjunto de pruebas para asegurar la calidad del pipeline de modelado y evaluación: control de desbalance, detección de overfitting, reproducibilidad local (CI local) y trazabilidad (MLflow).

## 2. Alcance
Aplica a:
- Dataset procesado: `data/processed/bank_formatted.csv`
- Selección de técnica (US14), entrenamiento (US16), evaluación (US17) y decisión final (US18)

## 3. Métricas objetivo (calidad del modelo)
Métricas mínimas a reportar:
- Accuracy, Precision, Recall, F1
- AUC-ROC (si aplica), Curva PR
- MCC, KS (si aplica), Matriz de confusión
- Análisis por clase (clase positiva/negativa)

Criterio de priorización:
- Se prioriza **Recall** para minimizar falsos negativos.
- Se usa **F1** como criterio secundario de balance.

## 4. Pruebas de desbalance
- Se verifica la distribución de clases (ratio de clase positiva).
- Se confirma la estrategia de balanceo aplicada (p. ej., undersampling) y se registra en MLflow.

## 5. Pruebas de overfitting
- Se comparan métricas train vs test (y/o CV vs test).
- Se considera riesgo de overfitting si existe brecha relevante (umbral de alerta referencial):
  - Alerta si (F1_train - F1_test) > 0.10 o (Recall_train - Recall_test) > 0.10
  - Si ocurre, se ajustan hiperparámetros/regularización o se cambia de técnica.

## 6. Pruebas de reproducibilidad (CI local)
- Se ejecutan pruebas automatizadas con `pytest`.
- Se genera reporte HTML como evidencia:
  - `reports/tests/tests_report.html`

Comando estándar:
```bash
pytest -q --disable-warnings --maxfail=1 --html=reports/tests/tests_report.html --self-contained-html
