import pandas as pd
import numpy as np
from pathlib import Path
import yaml

# --- Config ---
PARAMS = yaml.safe_load(Path("params.yaml").read_text())
FEAT = PARAMS.get("features", {})
OUT_CSV = FEAT.get("out_csv", "data/interim/banking_features.csv")
REPORT_DIR = Path("reports/features")

DERIVE_PREV = FEAT.get("derive_prev_contact_from_pdays", True)
PDAYS_NEVER = FEAT.get("pdays_never_value", 999)
DROP_PDAYS = FEAT.get("drop_pdays_after_derive", True)
YESNO_COLS = FEAT.get("yes_no_to_binary_cols", ["housing", "loan", "y"])
REPORT_SUMMARY = FEAT.get("report_summary", True)

# --- IO ---
IN_CSV = "data/interim/banking_clean.csv"   # salida de #10
df = pd.read_csv(IN_CSV)
df = df.replace("unknown", pd.NA) 

# 1) prev_contact a partir de pdays (1 = fue contactado antes, 0 = nunca)
if DERIVE_PREV and "pdays" in df.columns:
    df["prev_contact"] = np.where(df["pdays"] == PDAYS_NEVER, 0, 1).astype(int)
    if DROP_PDAYS:
        df = df.drop(columns=["pdays"])

# 2) Mapear yes/no → 1/0
for col in YESNO_COLS:
    if col in df.columns:
        df[col] = (df[col].astype(str).str.lower().map({"yes": 1, "no": 0})).astype("Int64")

# 3) Asegurar target al final
fmt = PARAMS.get("format", {})
if fmt.get("order_target_last", True):
    tgt = fmt.get("target_name", "y")
    if tgt in df.columns:
        cols = [c for c in df.columns if c != tgt] + [tgt]
        df = df[cols]

# 4) Normalizar nombres a snake_case
df.columns = [c.strip().lower().replace(".", "_") for c in df.columns]

# --- Salidas ---
Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)

# Metrica simple que pide DVC (lista de columnas construidas)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
pd.Series(df.columns, name="column").to_csv(REPORT_DIR / "features_built.csv", index=False)

# Reportes de evidencia ampliados
if REPORT_SUMMARY:
    (REPORT_DIR / "feature_dtypes.csv").write_text(
        df.dtypes.to_frame("dtype").to_csv()
    )
    if "y" in df.columns:
        y_counts = df["y"].value_counts(dropna=False, normalize=True).mul(100).round(2)
        y_counts.to_csv(REPORT_DIR / "target_balance_after_features_pct.csv", header=["porcentaje"])
    desc = pd.DataFrame({
        "col": df.columns,
        "n_nulls": [df[c].isna().sum() for c in df.columns],
        "n_unique": [df[c].nunique(dropna=True) for c in df.columns],
    })
    desc.to_csv(REPORT_DIR / "feature_summary.csv", index=False)

print(f"[OK] Features generadas en: {OUT_CSV}")
