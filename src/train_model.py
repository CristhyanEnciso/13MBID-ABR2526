"""
US16 - Entrenamiento del modelo MVP:
- Split train/test reproducible
- Balanceo por undersampling SOLO en train
- Entrenamiento del modelo seleccionado (params.yaml)
- Export: models/model.pkl
- Métricas: metrics/train_metrics.json
- Reportes: reports/model/
- Registro completo en MLflow (params/métricas/artefactos)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Dict, Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


ROOT = Path(".")
PARAMS_PATH = ROOT / "params.yaml"

OUT_MODEL = ROOT / "models" / "model.pkl"
OUT_METRICS = ROOT / "metrics" / "train_metrics.json"

REPORT_DIR = ROOT / "reports" / "model"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUT_REPORT_TRAIN = REPORT_DIR / "classification_report_train.txt"
OUT_REPORT_TEST = REPORT_DIR / "classification_report_test.txt"
OUT_CM_TRAIN = REPORT_DIR / "confusion_matrix_train.png"
OUT_CM_TEST = REPORT_DIR / "confusion_matrix_test.png"


def _load_params() -> Dict[str, Any]:
    params = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))
    if "train" not in params:
        raise KeyError("Falta la sección 'train' en params.yaml")
    return params["train"]


def _undersample(X: pd.DataFrame, y: pd.Series, random_state: int) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Undersampling simple: reduce la clase mayoritaria al tamaño de la minoritaria.
    Se aplica SOLO a train.
    """
    rng = np.random.default_rng(random_state)

    # Asegurar índice alineado
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError(f"Se esperaba clasificación binaria. Clases detectadas: {classes}")

    minority_class = classes[np.argmin(counts)]
    majority_class = classes[np.argmax(counts)]

    idx_min = np.where(y.values == minority_class)[0]
    idx_maj = np.where(y.values == majority_class)[0]

    n_min = len(idx_min)
    idx_maj_down = rng.choice(idx_maj, size=n_min, replace=False)

    idx_bal = np.concatenate([idx_min, idx_maj_down])
    rng.shuffle(idx_bal)

    return X.iloc[idx_bal].copy(), y.iloc[idx_bal].copy()


def _build_model(train_cfg: Dict[str, Any]):
    model_name = str(train_cfg["model"]["name"]).strip()
    mcfg = train_cfg["model"]

    if model_name == "KNN":
        cfg = mcfg.get("knn", {})
        return KNeighborsClassifier(
            n_neighbors=int(cfg.get("n_neighbors", 5)),
            weights=str(cfg.get("weights", "uniform")),
            n_jobs=int(cfg.get("n_jobs", -1)),
        )

    if model_name == "DecisionTree":
        cfg = mcfg.get("decision_tree", {})
        return DecisionTreeClassifier(
            random_state=int(train_cfg["random_state"]),
            max_depth=cfg.get("max_depth", None),
            min_samples_leaf=int(cfg.get("min_samples_leaf", 1)),
        )

    if model_name == "RandomForest":
        cfg = mcfg.get("random_forest", {})
        return RandomForestClassifier(
            n_estimators=int(cfg.get("n_estimators", 200)),
            random_state=int(train_cfg["random_state"]),
            n_jobs=int(cfg.get("n_jobs", -1)),
        )

    if model_name == "LogisticRegression":
        cfg = mcfg.get("logistic_regression", {})
        return LogisticRegression(
            max_iter=int(cfg.get("max_iter", 200)),
            solver=str(cfg.get("solver", "liblinear")),
            random_state=int(train_cfg["random_state"]),
        )

    if model_name == "LinearSVC":
        cfg = mcfg.get("linear_svc", {})
        return LinearSVC(
            C=float(cfg.get("C", 1.0)),
            random_state=int(train_cfg["random_state"]),
        )

    raise ValueError(f"Modelo no soportado en params.yaml: {model_name}")


def _compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _save_confusion_matrix(y_true, y_pred, out_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    # Anotar valores
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    train_cfg = _load_params()

    input_csv = Path(train_cfg.get("input_csv", "data/processed/bank_formatted.csv"))
    target_col = str(train_cfg.get("target_col", "y"))
    test_size = float(train_cfg.get("test_size", 0.2))
    random_state = int(train_cfg.get("random_state", 42))
    balancing_method = str(train_cfg.get("balancing_method", "undersample")).strip().lower()

    df = pd.read_csv(input_csv)
    if target_col not in df.columns:
        raise KeyError(f"No existe la columna target '{target_col}' en {input_csv}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split reproducible + estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Balanceo SOLO train
    X_train_bal, y_train_bal = X_train, y_train
    if balancing_method == "undersample":
        X_train_bal, y_train_bal = _undersample(X_train, y_train, random_state=random_state)

    model = _build_model(train_cfg)

    # MLflow tracking local robusto
    tracking_uri = f"file:{(ROOT / 'mlruns').resolve().as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Proyecto 13MBID-ABR2526 - Experimentación")

    run_name = f"US16-train-{train_cfg['model']['name']}"
    with mlflow.start_run(run_name=run_name):
        # Tags mínimos
        mlflow.set_tag("iteration", "2")
        mlflow.set_tag("us", "16")
        mlflow.set_tag("stage", "train")
        mlflow.set_tag("balancing_method", balancing_method)
        mlflow.set_tag("model_name", str(train_cfg["model"]["name"]))

        # Params relevantes
        mlflow.log_param("input_csv", str(input_csv))
        mlflow.log_param("target_col", target_col)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("balancing_method", balancing_method)

        # Entrenar
        model.fit(X_train_bal, y_train_bal)

        # Predicciones train/test (para detectar señales tempranas de overfitting)
        y_pred_train = model.predict(X_train_bal)
        y_pred_test = model.predict(X_test)

        m_train = _compute_metrics(y_train_bal, y_pred_train)
        m_test = _compute_metrics(y_test, y_pred_test)

        # Log métricas MLflow
        mlflow.log_metrics({f"train_{k}": v for k, v in m_train.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in m_test.items()})

        # Reportes
        OUT_REPORT_TRAIN.write_text(classification_report(y_train_bal, y_pred_train, zero_division=0), encoding="utf-8")
        OUT_REPORT_TEST.write_text(classification_report(y_test, y_pred_test, zero_division=0), encoding="utf-8")

        _save_confusion_matrix(y_train_bal, y_pred_train, OUT_CM_TRAIN, "Confusion Matrix (Train)")
        _save_confusion_matrix(y_test, y_pred_test, OUT_CM_TEST, "Confusion Matrix (Test)")

        # Export modelo + métricas
        OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, OUT_MODEL)

        OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": str(train_cfg["model"]["name"]),
            "balancing_method": balancing_method,
            "random_state": random_state,
            "test_size": test_size,
            "train_samples": int(len(X_train)),
            "train_samples_balanced": int(len(X_train_bal)),
            "test_samples": int(len(X_test)),
            "metrics_train": m_train,
            "metrics_test": m_test,
        }
        OUT_METRICS.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        # Artefactos a MLflow
        mlflow.log_artifact(str(OUT_REPORT_TRAIN))
        mlflow.log_artifact(str(OUT_REPORT_TEST))
        mlflow.log_artifact(str(OUT_CM_TRAIN))
        mlflow.log_artifact(str(OUT_CM_TEST))
        mlflow.log_artifact(str(OUT_METRICS))
        mlflow.log_artifact(str(OUT_MODEL))

    print(f"[OK] Modelo exportado: {OUT_MODEL}")
    print(f"[OK] Métricas exportadas: {OUT_METRICS}")
    print(f"[OK] Reportes: {REPORT_DIR}")


if __name__ == "__main__":
    main()
