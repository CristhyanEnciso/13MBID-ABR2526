# src/evaluate_model.py
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import yaml
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    matthews_corrcoef, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)

import matplotlib.pyplot as plt


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ks_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def plot_and_save_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_and_save_roc(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> dict:
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist()}


def plot_and_save_pr(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> dict:
    precision, recall, thr = precision_recall_curve(y_true, y_score)
    fig = plt.figure()
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return {"precision": precision.tolist(), "recall": recall.tolist(), "thresholds": thr.tolist()}


def to_01_if_needed(x: float) -> float:
    # Si viene en % (p.ej. 49.7), lo pasamos a 0.497
    return float(x / 100.0) if x is not None and x > 1.0 else float(x)


def main(
    data_path: str = "data/processed/bank_formatted.csv",
    model_path: str = "models/model.pkl",
    out_metrics: str = "metrics/eval_metrics.json",
    out_dir: str = "reports/evaluation",
    corroboracion_path: str = "reports/evaluation/corroboracion_resultados.md",
    params_path: str = "params.yaml",
) -> None:

    params = load_params(params_path)

    # Intenta tomar config desde params.yaml (si existe); si no, usa defaults razonables
    random_state = int(params.get("model", {}).get("random_state", 42))
    test_size = float(params.get("model", {}).get("test_size", 0.2))
    target = params.get("format", {}).get("target_name", "y")

    out_dir = Path(out_dir)
    out_metrics = Path(out_metrics)

    df = pd.read_csv(data_path)
    if target not in df.columns:
        raise ValueError(f"No encuentro la columna target '{target}' en {data_path}")

    y = df[target].astype(int) if str(df[target].dtype) != "int64" else df[target]
    X = df.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = joblib.load(model_path)

    # Predicción
    y_pred = model.predict(X_test)

    # Scores (para AUC/ROC/PR/KS)
    y_score = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)

    # Métricas base
    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_path": data_path,
        "model_path": model_path,
        "split": {"test_size": test_size, "random_state": random_state, "stratify": True},
        "samples": {
            "total": int(len(df)),
            "train": int(len(X_train)),
            "test": int(len(X_test)),
            "positive_rate_total": float(np.mean(y)),
            "positive_rate_test": float(np.mean(y_test)),
        },
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_test, y_pred)),
        },
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }

    # Métricas que requieren score
    if y_score is not None:
        metrics["metrics"]["roc_auc"] = float(roc_auc_score(y_test, y_score))
        metrics["metrics"]["avg_precision_pr_auc"] = float(average_precision_score(y_test, y_score))
        metrics["metrics"]["ks"] = ks_from_scores(np.array(y_test), np.array(y_score))
    else:
        metrics["metrics"]["roc_auc"] = None
        metrics["metrics"]["avg_precision_pr_auc"] = None
        metrics["metrics"]["ks"] = None

    # Guardar artefactos
    save_json(out_metrics, metrics)

    # Reporte por clase
    report_txt = classification_report(y_test, y_pred, zero_division=0)
    save_text(out_dir / "classification_report.txt", report_txt)

    # Confusion matrix
    cm = np.array(metrics["confusion_matrix"])
    plot_and_save_confusion_matrix(cm, out_dir / "confusion_matrix.png")

    # Curvas
    if y_score is not None:
        roc_data = plot_and_save_roc(np.array(y_test), np.array(y_score), out_dir / "roc_curve.png")
        pr_data = plot_and_save_pr(np.array(y_test), np.array(y_score), out_dir / "pr_curve.png")
        save_json(out_dir / "roc_curve_data.json", roc_data)
        save_json(out_dir / "pr_curve_data.json", pr_data)

    # Corroboración vs valores documentados (docente) usando CV mean/std
    # Esos valores pueden estar en % (49.7) o en 0-1 (0.497)
    documented = {
        "LogisticRegression": {"cv_f1_mean": 49.76525605093318, "cv_f1_std": 0.0132401153018114,
                               "cv_recall_mean": 49.50458715596331, "cv_recall_std": 0.01545646949625883},
        "LinearSVC": {"cv_f1_mean": 50.07414873869516, "cv_f1_std": 0.01355814515763869,
                      "cv_recall_mean": 49.871559633027516, "cv_recall_std": 0.013896584388081803},
        "KNN": {"cv_f1_mean": 57.94348473810715, "cv_f1_std": 0.019763296183430562,
                "cv_recall_mean": 59.22935779816514, "cv_recall_std": 0.02654913357458485},
        "DecisionTree": {"cv_f1_mean": 70.5592788621255, "cv_f1_std": 0.010256765940153307,
                         "cv_recall_mean": 74.6788990825688, "cv_recall_std": 0.010444219795051512},
    }

    shortlist_path = Path("reports/model/shortlist_modelos.csv")
    current_rows = {}
    if shortlist_path.exists():
        s = pd.read_csv(shortlist_path)
        for _, r in s.iterrows():
            current_rows[str(r["model"])] = {
                "cv_recall_mean": float(r.get("cv_recall_mean", np.nan)),
                "cv_f1_mean": float(r.get("cv_f1_mean", np.nan)),
            }

    lines = []
    lines.append("# Corroboración de resultados (US17)")
    lines.append("")
    lines.append("Comparo valores documentados vs. los obtenidos en mi ejecución actual (US14/shortlist).")
    lines.append("Si los documentados están en %, los normalizo a 0-1 para comparar.")
    lines.append("")
    lines.append("| Modelo | Doc cv_recall_mean | Actual cv_recall_mean | Diff | Doc cv_f1_mean | Actual cv_f1_mean | Diff |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for m, dv in documented.items():
        doc_recall = to_01_if_needed(dv["cv_recall_mean"])
        doc_f1 = to_01_if_needed(dv["cv_f1_mean"])

        act = current_rows.get(m, {})
        act_recall = float(act.get("cv_recall_mean", np.nan)) if act else np.nan
        act_f1 = float(act.get("cv_f1_mean", np.nan)) if act else np.nan

        diff_recall = (act_recall - doc_recall) if np.isfinite(act_recall) else np.nan
        diff_f1 = (act_f1 - doc_f1) if np.isfinite(act_f1) else np.nan

        lines.append(
            f"| {m} | {doc_recall:.4f} | {act_recall if np.isfinite(act_recall) else 'NA'} | "
            f"{diff_recall if np.isfinite(diff_recall) else 'NA'} | "
            f"{doc_f1:.4f} | {act_f1 if np.isfinite(act_f1) else 'NA'} | {diff_f1 if np.isfinite(diff_f1) else 'NA'} |"
        )

    lines.append("")
    lines.append("## Si hay diferencias, registro causas típicas")
    lines.append("- random_state / split / folds")
    lines.append("- balanceo aplicado (y dónde se aplica)")
    lines.append("- cambios en dataset/pipeline de features")
    lines.append("- hiperparámetros")
    lines.append("")

    save_text(Path(corroboracion_path), "\n".join(lines))

    # Log a MLflow para evidencia
    try:
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Proyecto 13MBID-ABR2526 - Evaluación")
        with mlflow.start_run(run_name="US17_Evaluation"):
            mlflow.log_params({"us": 17, "random_state": random_state, "test_size": test_size})
            for k, v in metrics["metrics"].items():
                if v is not None:
                    mlflow.log_metric(k, v)
            mlflow.log_artifact(str(out_metrics))
            for p in ["classification_report.txt", "confusion_matrix.png", "roc_curve.png", "pr_curve.png", "corroboracion_resultados.md"]:
                fp = out_dir / p
                if fp.exists():
                    mlflow.log_artifact(str(fp))
    except Exception as e:
        # No fallo la evaluación si MLflow no está disponible; solo lo reporta por consola.
        print(f"[WARN] No pude registrar en MLflow: {e}")

    print(f"[OK] Métricas: {out_metrics}")
    print(f"[OK] Reportes: {out_dir}")
    print(f"[OK] Corroboración: {corroboracion_path}")


if __name__ == "__main__":
    main()
