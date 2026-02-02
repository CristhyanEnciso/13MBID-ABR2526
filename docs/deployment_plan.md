# Plan de despliegue (US21) – API FastAPI + UI Streamlit

## Objetivo
Publicar el MVP de inferencia:
- **API (FastAPI)** para predicción.
- **UI (Streamlit)** para consumo por usuario final.

## Estructura esperada del repo
- `app/main.py` -> API FastAPI
- `app/ui.py` -> Interfaz Streamlit
- `models/model.pkl` -> modelo entrenado
- `models/preprocessor.pkl` -> preprocesador
- `app/api_post_query_data.json` -> ejemplo de payload

## Ejecución local (verificación)
### API
```bash
python -m pip install -r config/requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
## Endpoints

- **GET /** → mensaje API online  
- **GET /health** → estado (modelo/preprocesador cargados)  
- **POST /predict** → predicción  

---

## Prueba rápida (API)

```bash
curl -s -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  --data-binary "@app/api_post_query_data.json" | python -m json.tool
```
## UI (ejecución local)

<bash>
python -m pip install -r config/requirements.txt
streamlit run app/ui.py
</bash>

---

## Configurar en la UI

- **URL API:** `http://localhost:8000`

---

## Despliegue Opción A (sin GitHub Actions)

### Render (API)

- **Tipo:** Web Service (Python)

**Build command:**

```bash
pip install -r config/requirements.txt
```

**Start command:**

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

**Verificación:**

- Abrir: `https://<tu-api>.onrender.com/health`
- Abrir: `https://<tu-api>.onrender.com/docs`

---

### Streamlit Community Cloud (UI)

- **App:** conectar repo GitHub, branch `main`
- **Main file path:** `app/ui.py`

**Verificación:**

- La UI carga
- La UI apunta a la URL pública de Render y muestra **“API conectada”**
- Se realiza una predicción

---

## Evidencias (capturas requeridas)

- **Render:** servicio + logs + URL pública
- **API:** `/docs` y `/health` en URL pública
- **Streamlit:** UI cargada + “API conectada” + resultado de predicción
