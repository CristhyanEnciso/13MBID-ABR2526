from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------
# Logging
# ---------------------------
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
    Funciona local y en Render.
    """
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "dvc.yaml").exists() or (p / ".git").exists() or (p / "models").exists():
            return p
    return here.parent


ROOT_DIR = Path(os.getenv("ROOT_DIR", str(find_root_dir()))).resolve()
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data" / "processed"

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODELS_DIR / "model.pkl")))
PREPROCESSOR_PATH = Path(os.getenv("PREPROCESSOR_PATH", str(MODELS_DIR / "preprocessor.pkl")))
FEATURE_COLUMNS_PATH = Path(os.getenv("FEATURE_COLUMNS_PATH", str(MODELS_DIR / "feature_columns.json")))
FORMATTED_DATA_PATH = Path(os.getenv("FORMATTED_DATA_PATH", str(DATA_DIR / "bank_formatted.csv")))

# compatibilidad con tu repo (si usas decision_tree_model.pkl)
if not MODEL_PATH.exists():
    alt = MODELS_DIR / "decision_tree_model.pkl"
    if alt.exists():
        MODEL_PATH = alt


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
    prediction: str
    probability: Dict[str, float]
    model_info: Dict[str, Any]


# ---------------------------
# Estado global (artefactos)
# ---------------------------
model = None
preprocessor = None
feature_columns: Optional[List[str]] = None

model_loaded = False
preprocessor_loaded = False
feature_columns_loaded = False

model_error: Optional[str] = None
preprocessor_error: Optional[str] = None
feature_columns_error: Optional[str] = None


def _req_to_dict(req: PredictionRequest) -> Dict[str, Any]:
    # Pydantic v2: model_dump; v1: dict
    if hasattr(req, "model_dump"):
        return req.model_dump()
    return req.dict()


def load_feature_columns() -> Optional[List[str]]:
    """
    1) Intentar models/feature_columns.json (preferido)
       Soporta:
         - ["col1","col2",...]
         - {"columns": ["col1",...]}
    2) Fallback: leer cabecera de bank_formatted.csv (solo si existe)
    """
    # 1) JSON
    if FEATURE_COLUMNS_PATH.exists():
        obj = json.loads(FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))

        if isinstance(obj, dict) and "columns" in obj:
            obj = obj["columns"]

        if isinstance(obj, list) and len(obj) > 0 and all(isinstance(x, str) for x in obj):
            return obj

        raise ValueError("feature_columns.json inválido: debe ser list[str] o {'columns': list[str]}")

    # 2) CSV (fallback)
    if FORMATTED_DATA_PATH.exists():
        df_head = pd.read_csv(FORMATTED_DATA_PATH, nrows=1)
        cols = list(df_head.columns)
        if "y" in cols:
            cols.remove("y")
        if len(cols) > 0:
            return cols

    return None


def encode_request_to_model_features(req: PredictionRequest, expected_cols: List[str]) -> pd.DataFrame:
    """
    Convierte el request "crudo" a one-hot y lo alinea a expected_cols.
    (Solo se usa si NO hay preprocessor o si quieres forzar schema.)
    """
    raw = pd.DataFrame([_req_to_dict(req)])

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
    encoded = encoded.reindex(columns=expected_cols, fill_value=0)
    return encoded


def load_artifacts() -> None:
    global model, preprocessor, feature_columns
    global model_loaded, preprocessor_loaded, feature_columns_loaded
    global model_error, preprocessor_error, feature_columns_error

    # Modelo
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No existe el modelo en: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        model_loaded = True
        model_error = None
        logger.info("Modelo cargado correctamente: %s (%s)", type(model).__name__, MODEL_PATH)
    except Exception as e:
        model = None
        model_loaded = False
        model_error = f"{type(e).__name__}: {e}"
        logger.warning("No se pudo cargar el modelo: %s", model_error)

    # Preprocessor
    try:
        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(f"No existe el preprocessor en: {PREPROCESSOR_PATH}")
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        preprocessor_loaded = True
        preprocessor_error = None
        logger.info("Preprocesador cargado correctamente: %s (%s)", type(preprocessor).__name__, PREPROCESSOR_PATH)
    except Exception as e:
        preprocessor = None
        preprocessor_loaded = False
        preprocessor_error = f"{type(e).__name__}: {e}"
        logger.warning("No se pudo cargar el preprocessor: %s", preprocessor_error)

    # Feature columns
    try:
        feature_columns = load_feature_columns()
        feature_columns_loaded = feature_columns is not None
        feature_columns_error = None if feature_columns_loaded else "No encontrado (JSON/CSV)"
        if feature_columns_loaded:
            logger.info(
                "feature_columns cargado (%d cols) desde %s",
                len(feature_columns),
                FEATURE_COLUMNS_PATH if FEATURE_COLUMNS_PATH.exists() else FORMATTED_DATA_PATH,
            )
        else:
            logger.warning("feature_columns NO cargado (no hay JSON/CSV).")
    except Exception as e:
        feature_columns = None
        feature_columns_loaded = False
        feature_columns_error = f"{type(e).__name__}: {e}"
        logger.warning("No se pudo cargar feature_columns: %s", feature_columns_error)


# ---------------------------
# API
# ---------------------------
app = FastAPI(title="API-13MBID", version="1.0.0")


@app.on_event("startup")
def _startup():
    load_artifacts()


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
        "model_type": type(model).__name__ if model_loaded and model is not None else None,
        "preprocessor_type": type(preprocessor).__name__ if preprocessor_loaded and preprocessor is not None else None,
        "errors": {
            "model_error": model_error,
            "preprocessor_error": preprocessor_error,
            "feature_columns_error": feature_columns_error,
        },
        "paths": {
            "ROOT_DIR": str(ROOT_DIR),
            "MODEL_PATH": str(MODEL_PATH),
            "PREPROCESSOR_PATH": str(PREPROCESSOR_PATH),
            "FEATURE_COLUMNS_PATH": str(FEATURE_COLUMNS_PATH),
            "FORMATTED_DATA_PATH": str(FORMATTED_DATA_PATH),
        },
        "exists": {
            "MODEL_PATH": MODEL_PATH.exists(),
            "PREPROCESSOR_PATH": PREPROCESSOR_PATH.exists(),
            "FEATURE_COLUMNS_PATH": FEATURE_COLUMNS_PATH.exists(),
            "FORMATTED_DATA_PATH": FORMATTED_DATA_PATH.exists(),
        },
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if not model_loaded or model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado. Verifica models/model.pkl (o decision_tree_model.pkl).")

    try:
        # Caso A (recomendado): usar preprocessor si existe (no dependes del CSV)
        if preprocessor_loaded and preprocessor is not None:
            raw_df = pd.DataFrame([_req_to_dict(req)])
            X = preprocessor.transform(raw_df)

        # Caso B: si no hay preprocessor, usar schema feature_columns (JSON/CSV)
        else:
            if not feature_columns_loaded or not feature_columns:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "No se pudo determinar el esquema de features. "
                        "Asegura models/feature_columns.json en el repo (recomendado)."
                    ),
                )
            X = encode_request_to_model_features(req, feature_columns)

        # Predicción
        y_pred = model.predict(X)[0]

        # Probabilidades
        prob_yes, prob_no = 0.0, 0.0
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            classes = list(getattr(model, "classes_", []))

            # localizar yes/no si existen
            if "yes" in classes and "no" in classes:
                prob_yes = float(proba[classes.index("yes")])
                prob_no = float(proba[classes.index("no")])
            else:
                # fallback: tomar última como positiva
                prob_yes = float(proba[-1])
                prob_no = float(1.0 - prob_yes)

        # Normalizar etiqueta
        prediction_label = "yes" if str(y_pred).lower() in ("1", "yes", "true") else "no"

        return {
            "prediction": prediction_label,
            "probability": {"no": float(prob_no), "yes": float(prob_yes)},
            "model_info": {
                "model_type": type(model).__name__,
                "preprocessor_type": type(preprocessor).__name__ if preprocessor is not None else None,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error en predicción")
        raise HTTPException(status_code=400, detail=f"Error en predicción: {str(e)}")
