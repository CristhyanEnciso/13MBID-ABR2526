# Plan de despliegue (US21) – API FastAPI + UI Streamlit

## Objetivo
Publicar el MVP de inferencia del modelo de predicción mediante:

- **API (FastAPI)** para predicción online (online inference).
- **UI (Streamlit)** para consumo por usuario final (web).

Este plan describe **cómo servir el modelo**, el **contrato I/O**, la estrategia de **versionado** y un procedimiento de **rollback**, además de evidencias y URLs del despliegue.

---

## Alcance del MVP
- **Canal principal (MVP):** API online (FastAPI).
- **UI:** interfaz web conectada a la API pública.
- **Batch:** no implementado en esta iteración (queda como mejora futura). Se justifica porque el caso de uso del MVP es **predicción individual interactiva**.

---

## Estructura esperada del repositorio
- `app/main.py` → API FastAPI
- `app/ui.py` → Interfaz Streamlit
- `models/model.pkl` → modelo entrenado (artefacto productivo)
- `models/preprocessor.pkl` → preprocesador (artefacto del pipeline)
- `models/feature_columns.json` → **esquema final (one-hot)** esperado por el modelo
- `app/api_post_query_data.json` → ejemplo de payload para pruebas

---

## Nota sobre datos (DVC / despliegue)
El archivo `data/processed/bank_formatted.csv` se mantiene **local** (controlado con DVC) para entrenamiento y preparación de datos.

Para el despliegue **online**, la API **no depende** de ese CSV: utiliza `models/feature_columns.json` para garantizar que el request se convierta a las **columnas one-hot exactas** del entrenamiento (p. ej., 42 features), evitando fallos por diferencias de esquema en producción.

---

## Endpoints (API)
- **GET /** → mensaje “API online”
- **GET /health** → estado del servicio y carga de artefactos (modelo / preprocessor / schema)
- **POST /predict** → predicción

---

## Contrato I/O (API)

### POST `/predict`

#### Request (JSON)
Campos requeridos (tipos y validaciones principales):

- `age` (int, >= 0)
- `job` (str) — ej: `"technician"`, `"admin."`, `"management"`, `"retired"`
- `marital` (str) — ej: `"single"`, `"married"`
- `education` (str) — ej: `"basic.4y"`, `"university.degree"`, `"professional.course"`
- `housing` (str) — `"yes"` | `"no"`
- `loan` (str) — `"yes"` | `"no"`
- `contact` (str) — `"cellular"` | `"telephone"`
- `month` (str) — ej: `"may"`, `"jun"`, `"jul"`
- `day_of_week` (str) — `"mon"`, `"tue"`, `"wed"`, `"thu"`, `"fri"`
- `duration` (int, >= 0) — duración del último contacto (segundos)
- `campaign` (int, >= 0) — contactos en campaña actual
- `previous` (int, >= 0) — contactos previos
- `poutcome` (str) — `"success"` | `"nonexistent"` | `"failure"`
- `emp_var_rate` (float)
- `cons_price_idx` (float)
- `cons_conf_idx` (float)
- `euribor3m` (float)
- `nr_employed` (float)
- `contacted_before` (str) — `"yes"` | `"no"`

**Ejemplo válido:**
```json
{
  "age": 36,
  "job": "technician",
  "marital": "single",
  "education": "university.degree",
  "housing": "no",
  "loan": "no",
  "contact": "cellular",
  "month": "may",
  "day_of_week": "tue",
  "duration": 520,
  "campaign": 1,
  "previous": 1,
  "poutcome": "success",
  "emp_var_rate": -0.1,
  "cons_price_idx": 98.893,
  "cons_conf_idx": -42.7,
  "euribor3m": 1.334,
  "nr_employed": 5099.1,
  "contacted_before": "yes"
}
```
## Response (JSON)

**Estructura esperada:**
- `prediction` → `"yes"` o `"no"`
- `probability` → probabilidades por clase (ej. `{ "no": 0.80, "yes": 0.20 }`)
- `model_info` → metadata del modelo en ejecución

**Ejemplo:**
```json
{
  "prediction": "no",
  "probability": { "no": 0.8, "yes": 0.2 },
  "model_info": {
    "model_type": "KNeighborsClassifier",
    "preprocessor_type": "FunctionTransformer",
    "n_features": 42
  }
}
```

## Errores esperados

- **422** → error de validación (campo faltante o tipo inválido)  
- **500** → artefactos faltantes (modelo no cargado / schema no disponible)  
- **400** → error controlado en inferencia (entrada incompatible o error interno)  

---

## Ejecución local (verificación)

### API (FastAPI)
```bash
python -m pip install -r config/requirements.txt  
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

**Verificación local:**
- `http://127.0.0.1:8000/health`  
- `http://127.0.0.1:8000/docs`  

---

## Prueba rápida (API local)
```bash
curl -s -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  --data-binary "@app/api_post_query_data.json"
```

---

## UI (Streamlit) – ejecución local
```bash
python -m pip install -r config/requirements.txt  
streamlit run app/ui.py
```

**Configuración en la UI:**
- **URL API:** `http://localhost:8000`

---

## Despliegue – Opción A (sin GitHub Actions)

### Render (API – FastAPI)
- **Tipo:** Web Service (Python)

**Build command**
```bash
pip install -r config/requirements.txt
```

**Start command**
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Verificación**
- `https://api-13mbid-qmrc.onrender.com/health`  
- `https://api-13mbid-qmrc.onrender.com/docs`  

---

### Streamlit Community Cloud (UI)
- Conectar repo GitHub, branch: `main`  
- **Main file path:** `app/ui.py`  

**Verificación**
- La UI carga correctamente  
- La UI apunta a la URL pública de Render  
- La UI muestra **“API conectada”**  
- Se puede realizar al menos una predicción  

---

## URLs del despliegue (producción)

- **API Render:** `https://api-13mbid-qmrc.onrender.com/`  
- **Health:** `https://api-13mbid-qmrc.onrender.com/health`  
- **OpenAPI/Swagger:** `https://api-13mbid-qmrc.onrender.com/docs`  
- **UI Streamlit:** `https://13mbid-abr2526-cenciso.streamlit.app/`  

---

## Estrategia de versionado

### Versionado del servicio (repo)
- Cada entrega estable se etiqueta con tags Git: `vX.Y_us21_deployment_plan`.
- La release estable incluye:
  - código `app/main.py` + `app/ui.py`
  - artefactos `models/model.pkl`, `models/preprocessor.pkl`
  - esquema `models/feature_columns.json`
  - baseline de métricas (si aplica) en `metrics/baseline_eval_metrics.json`

### Versionado del modelo
- Artefacto productivo: `models/model.pkl`
- Baseline de métricas: `metrics/baseline_eval_metrics.json`
- La “versión del modelo” queda asociada al **tag** que contiene el artefacto + métricas + schema.

---

## Estrategia de rollback

### Criterios (cuándo aplicar rollback)
Ejecutar rollback si ocurre al menos uno:
- la API responde **5xx** de forma persistente  
- `/health` indica `model_loaded=false` o `feature_columns_loaded=false`  
- se detecta degradación significativa respecto a `metrics/baseline_eval_metrics.json` tras reentrenar (si aplica)

### Procedimiento (pasos)
1. Identificar el último tag estable anterior (ej. `v1.6_...`).  
2. Volver a la versión estable:  
```bash
   git checkout <tag_estable>
```
3. Publicar cambios:  
```bash
   git push origin main --tags
```
4. Redeploy:
   - Render: “Manual Deploy” o redeploy automático por commit  
   - Streamlit: redeploy automático por commit  
5. Validar:
   - `/health` en verde  
   - una predicción desde la UI  

---

## Evidencias (capturas requeridas)

Capturas mínimas recomendadas (5):
1. Render: servicio “Live” + URL pública visible  
2. Render: logs mostrando arranque correcto (uvicorn / service live)  
3. API: `/health` en producción (con `model_loaded=true`, `feature_columns_loaded=true`)  
4. API: `/docs` en producción (Swagger/OpenAPI)  
5. Streamlit: UI cargada + “API conectada” + resultado de predicción  

---

## Checklist de cierre (US21)

- [x] API desplegada y accesible (Render)  
- [x] UI desplegada y conectada a la API (Streamlit)  
- [x] Contrato I/O documentado (request/response + errores)  
- [x] Estrategia de versionado definida (tag + artefactos + baseline)  
- [x] Estrategia de rollback definida (criterios + pasos)  
- [x] URLs finales documentadas + evidencias (capturas)  
