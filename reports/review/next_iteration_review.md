# Cuestiones a revisar para siguiente iteración (US24)

## Objetivo
Registrar mejoras y lecciones aprendidas tras el cierre del MVP (Iteración 2),
enfocadas en **adopción**, **operación** y **escalabilidad** del producto (API/UI + modelo).

> Referencia: esta propuesta extiende y operacionaliza lo definido en **US22** (drift, SLA/latencia, alertas y retraining).

---

## 1) Verificación con datos reales (P1 – prioridad alta)
**Qué hacer**
- Integrar la **etiqueta real** (ground truth) posterior a campañas para medir desempeño real.
- Ejecutar comparación del desempeño real vs:
  - baseline (`metrics/baseline_eval_metrics.json`)
  - métodos actuales de la organización (criterio de negocio).
- Definir criterios de aceptación para adopción (quality gate) y responsables de aprobación.

**Evidencia / entregables**
- Reporte con métricas reales (ej. F1/AUC/Recall) y comparación vs baseline.
- Criterio de adopción explícito (umbral/condición de aprobación).
- Registro de decisión (aprobado / no aprobado + justificación).

**Riesgo si no se implementa**
- El MVP queda “demostrativo” pero **no validado** frente a la realidad del negocio.

---

## 2) Observabilidad y monitoreo (P1 – operación)
**Qué hacer**
- Implementar logging estructurado y persistente (requests, latencia, status codes, errores).
- Automatizar métricas operativas y funcionales definidas en US22:
  - disponibilidad, p95/p99, tasa 4xx/5xx
  - ratio de clase positiva (yes/no), inputs inválidos (422)
- Implementar drift real:
  - registrar inputs y score por ventanas (día/semana)
  - calcular PSI/KS y reportar automáticamente (data/score drift)

**Evidencia / entregables**
- Log de predicciones (CSV/JSON por ventana) + resumen de métricas operativas.
- Reporte PSI/KS por ventana y alerta cuando supere umbrales (US22).
- Evidencia de alertas disparadas (al menos en modo “simulado” si no hay tooling real).

**Riesgo si no se implementa**
- No se detectan fallos, degradación o drift a tiempo (riesgo de **decisiones incorrectas**).

---

## 3) MLOps / automatización del ciclo de vida (P2)
**Qué hacer**
- Integrar un flujo automatizado: retraining → evaluación → quality gate → release.
- Publicar reportes (tests / drift / métricas) como artefactos del pipeline.
- (Opcional) CI en GitHub Actions para ejecutar `dvc repro` parcial + `pytest` por PR.

**Evidencia / entregables**
- Pipeline documentado (pasos + rutas) y ejecutable.
- Artefactos generados automáticamente: métricas, reporte tests, reporte drift.
- Evidencia de ejecución CI (si se implementa).

**Riesgo si no se implementa**
- Releases manuales con mayor probabilidad de errores y sin trazabilidad consistente.

---

## 4) Mejora del producto (UI/API) (P2)
**Qué hacer**
- Mejorar explicabilidad (feature importance / SHAP opcional según complejidad).
- Mejorar UX en UI: validaciones, rangos esperados, tooltips, mensajes de error claros.
- Manejo robusto de categorías nuevas (catálogo/control de valores permitidos).

**Evidencia / entregables**
- UI con validaciones y ayudas visibles + captura.
- Manejo de errores coherente (422/400/500) documentado.
- (Opcional) reporte de explicabilidad o sección explicativa en la Memoria.

**Riesgo si no se implementa**
- Baja adopción por falta de confianza/entendimiento y fricción en el uso.

---

## 5) Habilidades y adopción organizacional (P3)
**Qué hacer**
- Capacitación básica en prácticas ágiles y DataOps/MLOps para el equipo.
- Definir roles y responsabilidades operativas (quién monitorea, quién aprueba releases, quién responde incidentes).

**Evidencia / entregables**
- RACI simple (roles → responsabilidades).
- Plan de capacitación breve (temas + frecuencia).

**Riesgo si no se implementa**
- Ambigüedad operativa (nadie “dueño” del monitoreo/retraining) y adopción incompleta.

---

## Definition of Done (siguiente iteración)
- [ ] Existe evaluación con **ground truth** y comparación vs baseline y método actual.
- [ ] Existe monitoreo operativo/funcional mínimo (logs + métricas por ventana).
- [ ] Existe medición de drift (PSI/KS) con umbrales (US22) y evidencia de revisión.
- [ ] Existe política de retraining aplicada (al menos una ejecución controlada end-to-end).
- [ ] Roles y criterios de adopción definidos (quality gate + responsable).

---

## Conclusión
El MVP está listo para iniciar fase de verificación. La siguiente iteración debe enfocarse en:
**(1)** validación con datos reales, **(2)** observabilidad + drift, **(3)** automatización del ciclo de vida del modelo.
