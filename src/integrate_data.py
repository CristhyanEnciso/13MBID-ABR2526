import pandas as pd
from pathlib import Path

CLEAN_CSV = "data/interim/banking_clean.csv"
FEATS_CSV = "data/interim/banking_features.csv"

OUT_FINAL = "data/processed/bank_final.csv"
OUT_REPORT = "reports/integration/integration_summary.csv"

def main():
    df_clean = pd.read_csv(CLEAN_CSV)
    df_feat  = pd.read_csv(FEATS_CSV)

    # Chequeos basicos
    rows_clean, rows_feat = len(df_clean), len(df_feat)
    assert rows_clean == rows_feat, (
        f"Inconsistencia: filas clean={rows_clean} vs features={rows_feat}"
    )

    # Las features ya incluyen las columnas limpias
    df_final = df_feat.copy()

    # Reordenar target al final si existe
    if "y" in df_final.columns:
        cols = [c for c in df_final.columns if c != "y"] + ["y"]
        df_final = df_final[cols]

    # Salidas
    Path(OUT_FINAL).parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUT_FINAL, index=False)

    # Reporte para DVC
    Path(OUT_REPORT).parent.mkdir(parents=True, exist_ok=True)
    added   = sorted(set(df_final.columns) - set(df_clean.columns))
    removed = sorted(set(df_clean.columns) - set(df_final.columns))

    rep = pd.DataFrame(
        {
            "rows_clean": [rows_clean],
            "rows_feat": [rows_feat],
            "rows_final": [len(df_final)],
            "cols_final": [df_final.shape[1]],
            "added_cols": [", ".join(added) if added else ""],
            "removed_cols": [", ".join(removed) if removed else ""],
        }
    )
    rep.to_csv(OUT_REPORT, index=False)

if __name__ == "__main__":
    main()
