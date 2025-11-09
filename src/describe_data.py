from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Análisis descriptivo del dataset")
    p.add_argument("--input", type=Path, default=Path("data/raw/bank-additional-full.csv"),
                   help="Ruta del CSV de entrada")
    p.add_argument("--sep", type=str, default=";", help="Separador del CSV")
    p.add_argument("--outdir", type=Path, default=Path("reports/summary"),
                   help="Directorio de salida para resúmenes")
    p.add_argument("--figdir", type=Path, default=Path("reports/figures/desc"),
                   help="Directorio de salida para figuras")
    return p.parse_args()

def main():
    args = parse_args()

    # preparar directorios
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.figdir.mkdir(parents=True, exist_ok=True)

    # cargar datos
    df = pd.read_csv(args.input, sep=args.sep)
    print(f"[INFO] dataset: {args.input} | shape={df.shape}")

    # 1) describe()
    try:
        desc = df.describe(include="all", datetime_is_numeric=True).transpose()
    except TypeError:
        desc = df.describe(include="all").transpose()

    desc.to_csv(args.outdir / "summary_describe.csv")

    # 2) tipos
    dtypes = pd.DataFrame({"dtype": df.dtypes.astype(str)})
    dtypes.to_csv(args.outdir / "summary_dtypes.csv")

    # 3) nulos (conteo y %)
    na_counts = df.isna().sum()
    na_pct = (na_counts / len(df) * 100).round(2)
    nulls = pd.DataFrame({"null_count": na_counts, "null_pct": na_pct})
    nulls.to_csv(args.outdir / "summary_nulls.csv")

    # 4) cardinalidad (nunique)
    cardinality = df.nunique().sort_values(ascending=False)
    cardinality = cardinality.rename("nunique").to_frame()
    cardinality.to_csv(args.outdir / "summary_cardinality.csv")

    # 5) value_counts (largo: columna, valor, count, pct)
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    vc_rows = []
    for c in cat_cols:
        vc = df[c].value_counts(dropna=False)
        vc_pct = (vc / len(df) * 100).round(2)
        tmp = pd.DataFrame({"column": c, "value": vc.index.astype(str), "count": vc.values, "pct": vc_pct.values})
        vc_rows.append(tmp)
    if vc_rows:
        value_counts = pd.concat(vc_rows, ignore_index=True)
        value_counts.to_csv(args.outdir / "summary_value_counts.csv", index=False)

    # 6) figuras básicas
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # hist para numéricos
    for c in num_cols:
        ax = df[c].hist(bins=20)
        fig = ax.get_figure()
        fig.savefig(args.figdir / f"hist_{c}.png", bbox_inches="tight")
        fig.clf()

    # barras top-10 para categóricas
    for c in cat_cols:
        top = df[c].value_counts().head(10)
        ax = top.plot(kind="bar")
        fig = ax.get_figure()
        fig.savefig(args.figdir / f"bar_top10_{c}.png", bbox_inches="tight")
        fig.clf()

    # nota metodológica sobre 'duration'
    if "duration" in df.columns:
        with open(args.outdir / "NOTA_duration.txt", "w", encoding="utf-8") as f:
            f.write(
                "La variable 'duration' influye fuertemente en el target, pero solo se conoce a posteriori de la llamada.\n"
                "No debe utilizarse como feature en modelos predictivos realistas.\n"
            )

    print(f"[OK] resúmenes en: {args.outdir} | figuras en: {args.figdir}")

if __name__ == "__main__":
    main()
