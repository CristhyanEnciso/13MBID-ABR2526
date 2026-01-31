cat > reports/evaluation/decision_summary.md <<'MD'
# US18 – Decisión final del modelo MVP

## Comparación y elección
Se consolidan los resultados de experimentación (US14) y evaluación (US17) en:
- `reports/evaluation/model_comparison.csv`

Se selecciona **DecisionTreeClassifier** como MVP por su mejor desempeño en Recall y F1 frente a los candidatos evaluados.

## Umbral de decisión
Se mantiene el umbral por defecto (0.5) para asegurar reproducibilidad del MVP.
La optimización de threshold por objetivo (Recall/F1/costo) queda como mejora incremental.

## Coste/beneficio (trade-off FP/FN)
Se prioriza Recall por el impacto de falsos negativos en el objetivo del MVP.
La evidencia se respalda con:
- matriz de confusión
- reporte por clase
- curvas ROC/PR

## Control MLOps: test de regresión
Se congela baseline en `metrics/baseline_eval_metrics.json` y se valida no degradación con:
- `tests/test_model_regression.py`
- `metrics/regression_test.json`
- ejecución integrada en DVC (`regression_test`)
MD
