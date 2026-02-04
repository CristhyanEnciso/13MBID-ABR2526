# Enlaces de despliegue (US21 / US23)

> Nota: Estos enlaces se actualizan cada vez que se realiza un despliegue estable (release/tag).
> Sirve como evidencia para la Memoria (Iteración 2) y como referencia operativa del MVP.

---

## API (Render)

- **Base URL:** https://api-13mbid-qmrc.onrender.com/
- **Health:** https://api-13mbid-qmrc.onrender.com/health
- **Docs (OpenAPI/Swagger):** https://api-13mbid-qmrc.onrender.com/docs

### Endpoints (mínimos)
- `GET /` → mensaje “API online”
- `GET /health` → estado de carga de artefactos (model/schema)
- `POST /predict` → predicción

---

## UI (Streamlit)

- **URL:** https://13mbid-abr2526-cenciso.streamlit.app/

### Configuración (UI)
- **API URL (producción):** https://api-13mbid-qmrc.onrender.com/

---

## Repositorio

- **Repositorio GitHub:**
- **Tags relevantes:**
  - `v1.7_us21_deployment_plan` (plan de implementación)
  - `v1.8_us22_monitoring_policy` (monitoreo + alertas + retraining)
  - `v1.9_us23_memoria_final` (informe/memoria final con evidencias)

---

## Versión desplegada (trazabilidad)

- **Commit SHA:** `4b6112a1cdcb79f6bd02cc699d158cbebb06bced`
- **Fecha (último commit):** `2026-02-02 22:04:55 -0500`

---

## Verificación rápida

- [x] API responde `GET /health` en producción.
- [x] Documentación disponible en `/docs`.
- [x] UI carga y permite enviar un request a la API.
- [x] Se obtiene al menos 1 predicción exitosa desde la UI.

---

## Evidencias asociadas (Memoria)

- Captura API `/health` en producción.
- Captura API `/docs` (Swagger/OpenAPI).
- Captura Render: servicio “Live” + URL pública.
- Captura Render logs (arranque uvicorn / carga de artefactos).
- Captura UI Streamlit: “API conectada” + resultado de predicción.
