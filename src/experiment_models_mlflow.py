"""
US14 - Comparación de técnicas de modelado y shortlist (MLflow)
Entrada:  data/processed/bank_formatted.csv
Salida:   reports/model/shortlist_modelos.csv
MLflow:   ./mlruns (local)
"""

from pathlib import Path
import json
import pandas as pd
import yaml

import mlflow

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import resample

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

PARAMS = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))


def undersample_xy(X: pd.DataFrame, y: pd.Series, random_state: int):
    df = X.copy()
    df["__y__"] = y.values

    c0 = df[df["__y__"] == 0]
    c1 = df[df["__y__"] == 1]
    n = min(len(c0), len(c1))

    c0b = resample(c0, n_samples=n, random_state=random_state)
    c1b = resample(c1, n_samples=n, random_state=random_state)

    bal = pd.concat([c0b, c1b]).sample(frac=1.0, random_state=random_state)
    yb = bal["__y__"].astype(int)
    Xb = bal.drop(columns=["__y__"])
    return Xb, yb


def proba_or_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        return s
    return None


def eval_model_cv(model, X_train, y_train, X_test, y_test, cv_folds, random_state, balancing_method):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    cv_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}

    for tr_idx, val_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        if balancing_method == "undersampling":
            X_tr, y_tr = undersample_xy(X_tr, y_tr, random_state)

        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)
        y_score = proba_or_score(model, X_val)

        cv_metrics["accuracy"].append(accuracy_score(y_val, y_pred))
        cv_metrics["precision"].append(precision_score(y_val, y_pred, zero_division=0))
        cv_metrics["recall"].append(recall_score(y_val, y_pred, zero_division=0))
        cv_metrics["f1"].append(f1_score(y_val, y_pred, zero_division=0))
        if y_score is not None:
            cv_metrics["roc_auc"].append(roc_auc_score(y_val, y_score))

    X_fit, y_fit = (X_train, y_train)
    if balancing_method == "undersampling":
        X_fit, y_fit = undersample_xy(X_train, y_train, random_state)

    model.fit(X_fit, y_fit)

    y_pred_test = model.predict(X_test)
    y_score_test = proba_or_score(model, X_test)

    test_metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "test_precision": precision_score(y_test, y_pred_test, zero_division=0),
        "test_recall": recall_score(y_test, y_pred_test, zero_division=0),
        "test_f1": f1_score(y_test, y_pred_test, zero_division=0),
    }
    if y_score_test is not None:
        test_metrics["test_roc_auc"] = roc_auc_score(y_test, y_score_test)

    summary = {}
    for k, vals in cv_metrics.items():
        if vals:
            summary[f"cv_{k}_mean"] = float(pd.Series(vals).mean())
            summary[f"cv_{k}_std"] = float(pd.Series(vals).std())

    return summary, test_metrics


def main():
    cfg = PARAMS.get("experiment", {})

    data_path = cfg.get("input_path", "data/processed/bank_formatted.csv")
    target = cfg.get("target_name", "y")
    test_size = float(cfg.get("test_size", 0.2))
    random_state = int(cfg.get("random_state", 1))
    cv_folds = int(cfg.get("cv_folds", 5))
    balancing_method = cfg.get("balancing_method", "undersampling")

    tracking_uri = cfg.get("tracking_uri", "file:./mlruns")
    experiment_name = cfg.get("experiment_name", "Proyecto 13MBID-ABR2526 - Experimentación")

    Path("reports/model").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    # Si y viene como "yes/no", lo convertimos a 1/0
    if df[target].dtype == "object":
        df[target] = df[target].map({"yes": 1, "no": 0})

    X = df.drop(columns=[target])
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    candidates = [
        ("LogisticRegression", LogisticRegression(max_iter=200, solver="liblinear", random_state=random_state)),
        ("LinearSVC", LinearSVC(random_state=random_state)),
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
        ("DecisionTree", DecisionTreeClassifier(random_state=random_state)),
    ]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    rows = []
    for name, model in candidates:
        with mlflow.start_run(run_name=f"US14_{name}"):
            mlflow.log_param("model", name)
            mlflow.log_param("balancing_method", balancing_method)
            mlflow.log_param("cv_folds", cv_folds)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)

            cv_summary, test_metrics = eval_model_cv(
                model, X_train, y_train, X_test, y_test, cv_folds, random_state, balancing_method
            )

            for k, v in {**cv_summary, **test_metrics}.items():
                mlflow.log_metric(k, float(v))

            out_json = Path(f"reports/model/{name}_metrics.json")
            out_json.write_text(json.dumps({**cv_summary, **test_metrics}, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(out_json))

            rows.append({"model": name, **cv_summary, **test_metrics})

    df_out = pd.DataFrame(rows).sort_values(by=["cv_recall_mean", "cv_f1_mean"], ascending=False)
    df_out["recommended_baseline"] = False
    if len(df_out) > 0:
        df_out.loc[df_out.index[0], "recommended_baseline"] = True

    out_csv = Path("reports/model/shortlist_modelos.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"[OK] Generado: {out_csv}")


if __name__ == "__main__":
    main()
