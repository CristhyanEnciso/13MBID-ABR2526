import pandas as pd
from pathlib import Path
import yaml

# Cargar parámetros globales
PARAMS = yaml.safe_load(Path("params.yaml").read_text())
threshold = PARAMS["clean"]["drop_null_threshold"]

# Paths
INPUT = Path("data/interim/banking_selected.csv")
OUTPUT = Path("data/interim/banking_clean.csv")
REPORTS = Path("reports/cleaning")
REPORTS.mkdir(parents=True, exist_ok=True)

# Cargar datos
df = pd.read_csv(INPUT)
rows_before = len(df)

#   Eliminar duplicados
df = df.drop_duplicates()
duplicates_removed = rows_before - len(df)

#   Sustituir "unknown" → NaN
df = df.replace("unknown", pd.NA)

#   Eliminar filas con nulos por encima del umbral
rows_after_na_drop = len(df.dropna())
null_rows_removed = rows_before - rows_after_na_drop
df = df.dropna()

# Guardar dataset limpio
df.to_csv(OUTPUT, index=False)

#   Generar reporte de perdida de datos
loss_report = pd.DataFrame({
    "total_registros_iniciales": [rows_before],
    "duplicados_eliminados": [duplicates_removed],
    "nulos_eliminados": [null_rows_removed],
    "total_final": [len(df)],
    "porcentaje_perdida": [round((rows_before - len(df)) / rows_before * 100, 2)]
})
loss_report.to_csv(REPORTS / "data_loss_report.csv", index=False)

print(" Limpieza completada.")
print(loss_report)
