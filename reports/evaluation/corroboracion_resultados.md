# Corroboración de resultados (US17)

Comparo valores documentados vs. los obtenidos en mi ejecución actual (US14/shortlist).
Si los documentados están en %, los normalizo a 0-1 para comparar.

| Modelo | Doc cv_recall_mean | Actual cv_recall_mean | Diff | Doc cv_f1_mean | Actual cv_f1_mean | Diff |
|---|---:|---:|---:|---:|---:|---:|
| LogisticRegression | 0.4950 | 0.6151740935848332 | 0.12012822202520013 | 0.4977 | 0.4494601886538047 | -0.048192371855527105 |
| LinearSVC | 0.4987 | 0.611926205757659 | 0.11321060942738381 | 0.5007 | 0.4518958705117112 | -0.04884561687524036 |
| KNN | 0.5923 | 0.6647908741933657 | 0.0724972962117143 | 0.5794 | 0.3439319996286274 | -0.23550284775244412 |
| DecisionTree | 0.7468 | 0.570293634505867 | -0.17649535631982105 | 0.7056 | 0.3078324031861288 | -0.39776038543512615 |

## Si hay diferencias, registro causas típicas
- random_state / split / folds
- balanceo aplicado (y dónde se aplica)
- cambios en dataset/pipeline de features
- hiperparámetros
