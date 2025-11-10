import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import yaml

# --- Configuración ---
PARAMS = yaml.safe_load(Path("params.yaml").read_text())
OUT_PATH = Path("data/processed/bank_formatted.csv")
REPORT_DIR = Path("reports/format")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# --- Entrada ---
df = pd.read_csv("data/processed/bank_final.csv")

# 1. Validación y formateo de tipos
df = df.apply(lambda col: col.astype(str).str.lower() if col.dtype == "object" else col)

# 2. Codificación One-Hot para categóricas
cat_cols = df.select_dtypes(include="object").columns.drop("y", errors="ignore")
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# 3. Escalado de variables numéricas
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop("y", errors="ignore")
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# 4. Ordenar el target al final
if "y" in df.columns:
    cols = [c for c in df.columns if c != "y"] + ["y"]
    df = df[cols]

# --- Salida ---
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)

# --- Reporte simple ---
df.info(verbose=False, buf=open(REPORT_DIR / "format_summary.txt", "w"))
pd.Series(df.columns).to_csv(REPORT_DIR / "formatted_columns.csv", index=False, header=False)

print(f"[OK] Dataset formateado y guardado en {OUT_PATH}")
