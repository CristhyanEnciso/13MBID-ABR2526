import pandas as pd
from pathlib import Path
import yaml
from datetime import datetime

PARAMS_PATH = Path("params.yaml")
REPORT_DIR = Path("reports/selection")
INPUT = Path("data/raw/bank-additional-full.csv")  
OUTPUT = Path("data/interim/banking_selected.csv")

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    params = yaml.safe_load(PARAMS_PATH.read_text())
    drop_cols_cfg = set(params.get("clean", {}).get("drop_columns", []) or [])
    keep_cols_cfg = params.get("select", {}).get("keep_columns", []) or []

    df = pd.read_csv(INPUT, sep=";") if INPUT.suffix == ".csv" else pd.read_csv(INPUT)

    original_cols = list(df.columns)

    # 1) Eliminar columnas declaradas (p.ej., 'default')
    drop_existing = [c for c in drop_cols_cfg if c in df.columns]
    df = df.drop(columns=drop_existing, errors="ignore")

    # 2) Selección explícita
    if keep_cols_cfg:
        keep_existing = [c for c in keep_cols_cfg if c in df.columns]
        df = df[keep_existing]

    # Evidencias
    kept = pd.DataFrame({"feature": list(df.columns)})
    removed = pd.DataFrame({
        "feature": [c for c in original_cols if c not in df.columns],
        "reason": [
            ("declared_drop" if c in drop_cols_cfg else "not_in_keep_list")
            for c in original_cols if c not in df.columns
        ]
    })

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    kept.to_csv(REPORT_DIR / "features_selected.csv", index=False)
    removed.to_csv(REPORT_DIR / "removed_features.csv", index=False)
    Path(REPORT_DIR / "selection_info.txt").write_text(
        f"timestamp: {ts}\ninput: {INPUT}\noutput: {OUTPUT}\n"
        f"dropped_declared: {drop_existing}\nkept_from_config: {keep_cols_cfg}\n",
        encoding="utf-8"
    )

    # Guardar dataset resultante
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
   
    df.to_csv(OUTPUT, index=False)

if __name__ == "__main__":
    main()
