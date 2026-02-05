# Cuestiones a revisar para siguiente iteración (US24)

## Objetivo
Registrar mejoras y lecciones aprendidas a partir del cierre del MVP (Iteración 2),
enfocadas en escalabilidad, operación y adopción.

---

## 1) Verificación con datos reales (prioridad alta)
- Integrar la **etiqueta real** (ground truth) posterior a campañas para medir desempeño real.
- Ejecutar comparación contra baseline y métodos actuales de la organización.
- Definir criterios de aceptación del modelo para adopción (quality gate).

---

## 2) Observabilidad y monitoreo (operación)
- Implementar logging estructurado y persistente (requests, latencias, errores).
- Automatizar métricas operativas: disponibilidad, p95/p99, tasa 4xx/5xx.
- Implementar drift real:
  - registrar inputs y score por ventanas
  - calcular PSI/KS y reportar automáticamente

---

## 3) MLOps / automatización
- Integrar un flujo automatizado de evaluación (retraining → evaluación → gate → release).
- Publicar reportes (tests / drift / métricas) como artefactos de pipeline.
- (Opcional) CI en GitHub Actions para ejecutar tests y validación en cada PR.

---

## 4) Mejora del producto (UI/API)
- Mejorar explicabilidad (feature importance / SHAP opcional).
- Mejorar UX UI (validaciones, rangos esperados, tooltips).
- Manejo robusto de categorías nuevas (catálogo/control de valores permitidos).

---

## 5) Habilidades y adopción organizacional
- Capacitación en metodologías ágiles / prácticas DataOps-MLOps para el equipo.
- Definir roles y responsabilidades operativas (quién monitorea, quién aprueba releases).

---

## Conclusión
El MVP está listo para fase de verificación. La siguiente iteración debe enfocarse en:
(1) validación con datos reales, (2) observabilidad/drift, (3) automatización del ciclo de vida del modelo.
