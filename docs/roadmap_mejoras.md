# US20 – Roadmap de mejoras posteriores al MVP (futuras tareas)

**Fase CRISP-DM:** Evaluación / Despliegue (mejora continua)  
**Método:** SCRUM / DataOps / MLOps  
**Historia:** US20 – Determinación de futuras tareas  
**Objetivo:** Definir un conjunto priorizado de mejoras para evolucionar el MVP hacia una solución más robusta, reproducible y operable.

---

## Contexto

Con el **MVP construido** (pipeline reproducible con **DVC**, experimentación registrada en **MLflow** y artefactos de modelo disponibles para consumo), se identifican líneas de trabajo para una siguiente iteración enfocada en:

- Mejoras en **gestión y calidad de datos**, asegurando reproducción end-to-end.
- **Automatización** de prácticas (DataOps/MLOps) para reducir esfuerzo manual.
- **Optimización del modelo** (tuning) y evaluación de variantes para aumentar generalización.
- **Monitoreo** de la solución en uso (API/UI), tanto a nivel técnico como de negocio.
- Evaluaciones adicionales (**fairness/bias**, drift) para robustez y sostenibilidad.

Estas líneas incorporan explícitamente las recomendaciones del docente (reproducibilidad, automatización, tuning del árbol, evaluación de Random Forest/XGBoost, e instancias de monitoreo de uso/resultados y métricas complementarias).

---

## Roadmap priorizado (backlog propuesto)

> Convención de prioridad: **Alta** (próxima iteración), **Media** (siguiente), **Baja** (mejora incremental).

### A) Gestión y calidad de datos (reproducibilidad)

1. **Data contract / schema validation (entrada y dataset procesado)**  
   - **Objetivo:** Garantizar que las columnas, tipos y rangos esperados se mantengan estables.  
   - **Valor esperado:** Menos fallos por cambios silenciosos en datos; mayor reproducibilidad.  
   - **Prioridad:** Alta

2. **Versionado estricto de dataset y parámetros (etiquetas por release)**  
   - **Objetivo:** Estandarizar releases del pipeline y datos (tagging + snapshots).  
   - **Valor esperado:** Poder reproducir exactamente cualquier versión del MVP (auditoría).  
   - **Prioridad:** Alta

3. **Reglas de calidad ampliadas (más tests automáticos en `tests/`)**  
   - **Objetivo:** Extender checks (nulos, rangos, duplicados, cardinalidad, outliers) por etapa.  
   - **Valor esperado:** Prevención temprana de degradación por datos.  
   - **Prioridad:** Media

4. **Dataset “golden sample” para validación rápida**  
   - **Objetivo:** Mantener un subconjunto estable para smoke-tests del pipeline.  
   - **Valor esperado:** Ejecuciones rápidas y verificables al cambiar código/config.  
   - **Prioridad:** Media

---

### B) Automatización (DataOps/MLOps)

5. **Ejecución automatizada local/CI del pipeline + quality gates**  
   - **Objetivo:** Estandarizar comandos (`dvc repro`, `pytest -q`) como “puerta de calidad”.  
   - **Valor esperado:** Menos errores humanos y consistencia antes de publicar cambios.  
   - **Prioridad:** Alta

6. **Estandarizar estructura de outputs y limpieza (artefactos efímeros)**  
   - **Objetivo:** Asegurar que outputs de ejecución queden en DVC y docs en Git sin conflictos.  
   - **Valor esperado:** Repo más limpio y sin errores tipo “already tracked by SCM”.  
   - **Prioridad:** Media

7. **Automatizar actualización controlada de baselines (con aprobación)**  
   - **Objetivo:** Definir proceso para actualizar baseline de métricas solo cuando corresponda.  
   - **Valor esperado:** Gobernanza del performance; evita “mejoras” accidentales sin revisión.  
   - **Prioridad:** Media

---

### C) Optimización y modelos (tuning / generalización)

8. **Tuning de hiperparámetros del Árbol de Decisión (Grid/Random Search)**  
   - **Objetivo:** Mejorar capacidad de generalización optimizando profundidad, min_samples, etc.  
   - **Valor esperado:** Mejor equilibrio precision/recall/F1 y menor overfitting.  
   - **Prioridad:** Alta

9. **Evaluar variantes: Random Forest**  
   - **Objetivo:** Comparar con el MVP (mejor generalización y estabilidad).  
   - **Valor esperado:** Menor varianza; posible mejora de métricas sin perder demasiado interpretabilidad.  
   - **Prioridad:** Media

10. **Evaluar variantes: XGBoost (si se habilita dependencia)**  
   - **Objetivo:** Probar boosting para performance superior en clasificación tabular.  
   - **Valor esperado:** Potencial mejora significativa de métricas.  
   - **Prioridad:** Media

11. **Calibración de probabilidades + ajuste de threshold**  
   - **Objetivo:** Mejorar decisiones basadas en score (calibrated probabilities, PR trade-off).  
   - **Valor esperado:** Threshold más estable y justificado; mejor control de FP/FN.  
   - **Prioridad:** Media

---

### D) Fairness / Bias (análisis por subgrupos)

12. **Métricas por subgrupos (edad, job, marital, etc.)**  
   - **Objetivo:** Detectar diferencias de rendimiento entre segmentos.  
   - **Valor esperado:** Identificación temprana de sesgos; mejora de confianza del modelo.  
   - **Prioridad:** Media

13. **Estrategias de mitigación (si se detecta bias)**  
   - **Objetivo:** Aplicar técnicas como reweighing o ajuste de umbral (solo si es justificable).  
   - **Valor esperado:** Reducción de disparidades, manteniendo performance global.  
   - **Prioridad:** Baja

---

### E) Drift / Monitoreo (operación del MVP)

14. **Monitoreo de uso (API): requests, latencia, errores, timeouts**  
   - **Objetivo:** Medir salud y disponibilidad del servicio.  
   - **Valor esperado:** Observabilidad operativa mínima para producción.  
   - **Prioridad:** Alta

15. **Monitoreo de resultados: distribución de scores y ratio de clase positiva**  
   - **Objetivo:** Detectar cambios en comportamiento del modelo (sin etiquetas).  
   - **Valor esperado:** Señales tempranas de drift o cambios de campaña/mercado.  
   - **Prioridad:** Alta

16. **Detección de drift de features (PSI / KS por variable clave)**  
   - **Objetivo:** Medir desviaciones estadísticas entre datos históricos y nuevos.  
   - **Valor esperado:** Trigger para investigación o retraining.  
   - **Prioridad:** Media

17. **Estrategia de retraining (frecuencia o evento gatillado por drift)**  
   - **Objetivo:** Definir cuándo reentrenar y cómo validar (quality gate).  
   - **Valor esperado:** Sostenibilidad; mejora continua controlada.  
   - **Prioridad:** Media

---

### F) Seguridad y robustez de despliegue (preparación)

18. **Hardening básico del servicio (auth, rate limit, validación de inputs)**  
   - **Objetivo:** Preparar la API para consumo real minimizando riesgos.  
   - **Valor esperado:** Mayor resiliencia, menos abuso/errores por entradas inválidas.  
   - **Prioridad:** Media

19. **Estandarizar versionado del modelo consumido por API/UI**  
   - **Objetivo:** Asegurar que API/UI consuman el mismo artefacto/versionado del modelo.  
   - **Valor esperado:** Coherencia en despliegue; reproducibilidad del “modelo en producción”.  
   - **Prioridad:** Alta

---

## Enfoque recomendado por iteración (propuesta)

- **Corto plazo (Alta):** #1, #2, #5, #8, #14, #15, #19  
- **Medio plazo (Media):** #3, #4, #6, #7, #9, #10, #11, #12, #16, #17, #18  
- **Bajo (Baja):** #13 (solo si se detecta bias relevante)

---

## Criterios de cierre de US20 (para el backlog)

- Roadmap documentado con **≥10 mejoras** agrupadas por categoría.  
- Cada mejora incluye: **objetivo, valor esperado y prioridad**.  
- Se incluye explícitamente **monitoreo** (uso + resultados + métricas operativas).  
- El documento queda listo para transformarse en backlog de una siguiente iteración Scrum.

---

## Versionado (Git)

- **Commit sugerido:** `US20: roadmap de mejoras (features, tuning, fairness, drift, MLOps)`  
- **Tag sugerido:** `v1.6_us20_roadmap`
