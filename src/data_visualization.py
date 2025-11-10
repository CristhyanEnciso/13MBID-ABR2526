import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (9, 5)

def savefig(ax, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ax.get_figure().savefig(path, bbox_inches="tight", dpi=120)
    plt.close(ax.get_figure())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/bank-additional-full.csv", help="CSV de entrada")
    p.add_argument("--sep", default=";", help="Separador del CSV")
    p.add_argument("--figdir", default="reports/figures/eda", help="Carpeta de salida para PNGs")
    p.add_argument("--summarydir", default="reports/summary", help="Carpeta de salida para CSVs")
    args = p.parse_args()

    figdir = Path(args.figdir); figdir.mkdir(parents=True, exist_ok=True)
    sdir = Path(args.summarydir); sdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, sep=args.sep)

    # 1) Distribución de la variable objetivo (y)
    if "y" in df.columns:
        ax = sns.countplot(x="y", data=df)
        ax.set_title("Distribución de la variable objetivo (y)")
        ax.set_xlabel("¿Suscribió un depósito a plazo?")
        ax.set_ylabel("Cantidad de clientes")
        savefig(ax, figdir / "target_balance.png")

        df["y"].value_counts(normalize=True).mul(100).round(2).to_csv(
            sdir / "target_balance_pct.csv", header=["porcentaje"]
        )

    # 2) Distribución de una categórica (education), más loop por todas
    cats = df.select_dtypes(include=["object"]).columns
    if "y" in cats:
        cats = cats.drop("y")
    if "education" in df.columns:
        order = df["education"].value_counts().index
        ax = sns.countplot(y="education", data=df, order=order)
        ax.set_title("Distribución de education")
        ax.set_xlabel("Cantidad")
        savefig(ax, figdir / "cat_education.png")

    for col in cats:
        order = df[col].value_counts().index
        ax = sns.countplot(y=col, data=df, order=order)
        ax.set_title(f"Distribución de {col}")
        ax.set_xlabel("Cantidad")
        savefig(ax, figdir / f"cat_{col}.png")

    # 3) Heatmap de correlaciones (numéricas)
    num_df = df.select_dtypes(include=["float64", "int64"])
    if not num_df.empty:
        corr = num_df.corr(numeric_only=True)
        ax = sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        ax.set_title("Matriz de correlaciones")
        savefig(ax, figdir / "corr_heatmap.png")
        corr.to_csv(sdir / "corr_matrix.csv")

    # 4) Extra: Boxplot de duration por y (si existen)
    if {"duration", "y"}.issubset(df.columns):
        ax = sns.boxplot(x="y", y="duration", data=df)
        ax.set_title("Duration por clase (y)")
        savefig(ax, figdir / "box_duration_by_y.png")

    # 5) Extra: Histograma de age (si existe)
    if "age" in df.columns:
        ax = df["age"].hist(bins=20)
        ax.set_title("Distribución de age")
        savefig(ax, figdir / "hist_age.png")

    # 6) Evidencias en CSV (nulos y cardinalidad)
    df.isna().sum().to_frame("nulos").to_csv(sdir / "nulls_by_column.csv")
    if len(cats) > 0:
        pd.Series({c: df[c].nunique() for c in cats}, name="cardinalidad").to_csv(
            sdir / "categorical_cardinality.csv"
        )

if __name__ == "__main__":
    main()
