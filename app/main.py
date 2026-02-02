from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("api-13mbid")

# ---------------------------
# Helpers de rutas (robusto)
# ---------------------------
def find_root_dir() -> Path:
    """
    Encuentra la raíz del repo buscando marcadores típicos.
    Funciona si este archivo está en app/main.py o en la raíz.
    """
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "dvc.yaml").exists() or (p / ".git").exists() or (p / "models").exists():
            return p
    return here.parent


# app/main.py está dentro de /app, así que el root del repo es parents[1]
ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "models" / "model.pkl"
PREPROCESSOR_PATH = ROOT_DIR / "models" / "preprocessor.pkl"

if not MODEL_PATH.exists():
    alt = ROOT_DIR / "models" / "decision_tree_model.pkl"
    if alt.exists():
        MODEL_PATH = alt

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    logger.info("Modelo cargado correctamente: %s", type(model).__name__)
    logger.info("Preprocesador cargado correctamente: %s", type(preprocessor).__name__)
except FileNotFoundError as e:
    logger.exception("Error cargando artefactos del modelo")

# Fuente de columnas esperadas (para alinear one-hot)
FEATURE_COLUMNS_PATH = Path(os.getenv("FEATURE_COLUMNS_PATH", str(ROOT_DIR / "models" / "feature_columns.json")))
FORMATTED_DATA_PATH = ROOT_DIR / "data" / "processed" / "bank_formatted.csv"


# ---------------------------
# Esquemas I/O
# ---------------------------
class PredictionRequest(BaseModel):
    age: int = Field(..., ge=0)
    job: str
    marital: str
    education: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    duration: int = Field(..., ge=0)
    campaign: int = Field(..., ge=0)
    previous: int = Field(..., ge=0)
    poutcome: str
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float
    contacted_before: str  # "yes" / "no"


class PredictionResponse(BaseModel):
    prediction: str  # "yes" o "no"
    probability: Dict[str, float]  # {"no": 0.7, "yes": 0.3}
    model_info: Dict[str, Any]


# ---------------------------
# Carga de artefactos
# ---------------------------
model = None
preprocessor = None
feature_columns: Optional[List[str]] = None

model_loaded = False
preprocessor_loaded = False
feature_columns_loaded = False


def load_feature_columns() -> Optional[List[str]]:
    # 1) Intentar models/feature_columns.json
    if FEATURE_COLUMNS_PATH.exists():
        import json

        cols = json.loads(FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))
        if isinstance(cols, list) and len(cols) > 0:
            return cols

    # 2) Fallback: leer cabecera de bank_formatted.csv
    if FORMATTED_DATA_PATH.exists():
        df_head = pd.read_csv(FORMATTED_DATA_PATH, nrows=1)
        cols = list(df_head.columns)
        if "y" in cols:
            cols.remove("y")
        return cols

    return None


def encode_request_to_model_features(req: PredictionRequest, expected_cols: List[str]) -> pd.DataFrame:
    """
    Convierte el request "crudo" a one-hot como bank_formatted,
    y reindexa a las columnas exactas esperadas por el modelo.
    """
    raw = pd.DataFrame([req.dict()])

    categorical_cols = [
        "job",
        "marital",
        "education",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "poutcome",
        "contacted_before",
    ]

    encoded = pd.get_dummies(raw, columns=categorical_cols, dtype=int)

    # Alinear EXACTAMENTE a columnas del entrenamiento
    encoded = encoded.reindex(columns=expected_cols, fill_value=0)

    return encoded


# Cargar modelo/preprocessor (si existe) y columnas
try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No existe el modelo en: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model_loaded = False
    model = None
    logger.warning("No se pudo cargar el modelo: %s", e)

try:
    if PREPROCESSOR_PATH.exists():
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        preprocessor_loaded = True
    else:
        preprocessor = None
        preprocessor_loaded = False
except Exception as e:
    preprocessor_loaded = False
    preprocessor = None
    logger.warning("No se pudo cargar el preprocessor: %s", e)

try:
    feature_columns = load_feature_columns()
    feature_columns_loaded = feature_columns is not None
except Exception as e:
    feature_columns = None
    feature_columns_loaded = False
    logger.warning("No se pudo cargar feature_columns: %s", e)


# ---------------------------
# API
# ---------------------------
app = FastAPI(title="API-13MBID", version="1.0.0")


@app.get("/")
def root():
    return {"message": "API-13MBID Online"}


@app.get("/health")
def health():
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "preprocessor_loaded": preprocessor_loaded,
        "feature_columns_loaded": feature_columns_loaded,
        "model_type": type(model).__name__ if model_loaded else None,
        "preprocessor_type": type(preprocessor).__name__ if preprocessor_loaded else None,
        "paths": {
            "ROOT_DIR": str(ROOT_DIR),
            "MODEL_PATH": str(MODEL_PATH),
            "PREPROCESSOR_PATH": str(PREPROCESSOR_PATH),
            "FEATURE_COLUMNS_PATH": str(FEATURE_COLUMNS_PATH),
            "FORMATTED_DATA_PATH": str(FORMATTED_DATA_PATH),
        },
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if not model_loaded or model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado. Ejecuta el pipeline y verifica models/model.pkl")

    if not feature_columns_loaded or not feature_columns:
        raise HTTPException(
            status_code=500,
            detail="No se pudo determinar el esquema de features. "
                   "Asegura models/feature_columns.json o data/processed/bank_formatted.csv en el repo.",
        )

    try:
        X = encode_request_to_model_features(req, feature_columns)

        # Predicción
        y_pred = model.predict(X)[0]

        # Probabilidades
        prob_yes = None
        prob_no = None

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            classes = list(getattr(model, "classes_", []))

            # Detectar clase positiva (yes o 1)
            pos_label = "yes" if "yes" in classes else (1 if 1 in classes else None)
            if pos_label is None and len(classes) > 0:
                pos_label = classes[-1]  # fallback

            pos_idx = classes.index(pos_label) if pos_label in classes else 1
            prob_yes = float(proba[pos_idx])
            prob_no = float(1.0 - prob_yes)
        else:
            # fallback sin proba
            prob_yes = 0.0
            prob_no = 0.0

        # Normalizar etiqueta
        if str(y_pred).lower() in ("1", "yes", "true"):
            prediction_label = "yes"
        else:
            prediction_label = "no"

        return {
            "prediction": prediction_label,
            "probability": {"no": float(prob_no), "yes": float(prob_yes)},
            "model_info": {
                "model_type": type(model).__name__,
                "preprocessor_type": type(preprocessor).__name__ if preprocessor is not None else None,
                "n_features": int(X.shape[1]),
            },
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {str(e)}")
