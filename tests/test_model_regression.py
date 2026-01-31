import json
import yaml

BASELINE_PATH = "metrics/baseline_eval_metrics.json"
CURRENT_PATH = "metrics/eval_metrics.json"
PARAMS_PATH = "params.yaml"

def _load_metrics(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # El JSON puede guardar {"metrics": {...}} según la implementación.
    return data["metrics"] if "metrics" in data else data

def _load_thresholds() -> dict:
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params["regression"]["max_drop"]

def test_model_metrics_do_not_regress_beyond_threshold():
    baseline = _load_metrics(BASELINE_PATH)
    current = _load_metrics(CURRENT_PATH)
    max_drop = _load_thresholds()

    for k in ["recall", "f1", "roc_auc"]:
        if k not in baseline or k not in current:
            raise AssertionError(f"Falta métrica '{k}' en baseline o current")

        drop = baseline[k] - current[k]  # si current baja, drop > 0
        assert drop <= max_drop[k], (
            f"Regresión detectada en {k}: baseline={baseline[k]:.6f}, "
            f"current={current[k]:.6f}, drop={drop:.6f} > max_drop={max_drop[k]:.6f}"
        )
