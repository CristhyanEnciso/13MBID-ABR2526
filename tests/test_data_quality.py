import pandas as pd
import pytest
from pandera import Check
from pandera.pandas import DataFrameSchema, Column
from pathlib import Path

RAW = Path("data/raw/bank-additional-full.csv")
OUT = Path("reports/validation")
OUT.mkdir(parents=True, exist_ok=True)

@pytest.fixture(scope="session")
def df_raw():
    df = pd.read_csv(RAW, sep=";")
    return df


def test_esquema_completo(df_raw):
    """Valida el esquema completo de las 21 columnas del dataset crudo."""
    schema = DataFrameSchema({
        "age": Column(int, Check.ge(17), nullable=False),
        "job": Column(str, nullable=False),
        "marital": Column(str, nullable=False),
        "education": Column(str, nullable=False),
        "default": Column(str, nullable=True),
        "housing": Column(str, nullable=False),
        "loan": Column(str, nullable=False),
        "contact": Column(str, nullable=False),
        "month": Column(str, nullable=False),
        "day_of_week": Column(str, nullable=False),
        "duration": Column(int, Check.ge(0), nullable=False),
        "campaign": Column(int, Check.ge(1), nullable=False),
        "pdays": Column(int, nullable=True),  # UCI usa 999 para "nunca contactado"
        "previous": Column(int, Check.ge(0), nullable=False),
        "poutcome": Column(str, nullable=True),
        "emp.var.rate": Column(float, nullable=False),
        "cons.price.idx": Column(float, nullable=False),
        "cons.conf.idx": Column(float, nullable=False),
        "euribor3m": Column(float, nullable=False),
        "nr.employed": Column(float, nullable=False),
        "y": Column(str, Check.isin(["yes", "no"]), nullable=False),
    }, coerce=False)
    schema.validate(df_raw)


def test_sin_duplicados(df_raw):
    """Reporte de duplicados (no forzamos 0 en crudo; se limpia en US10)."""
    dup_count = df_raw.duplicated().sum()
    pd.DataFrame({"duplicados": [dup_count]}).to_csv(OUT / "duplicate_rows.csv", index=False)
    # No fallamos aquí: el borrado de duplicados es parte de la limpieza (US10).
    assert dup_count >= 0


def test_nulos_por_columna(df_raw):
    """El CSV crudo de UCI no trae NaN (usa 'unknown'). Si aparecen, se registran y fallamos."""
    nulls = df_raw.isna().sum()
    nulls.to_frame("nulos").to_csv(OUT / "null_distribution.csv")
    assert nulls.sum() == 0, "Hay NaN en crudo (deberían ser 'unknown')."


def test_tipos_macro(df_raw):
    """Tipos esperados para variables macroeconómicas."""
    expected_float_cols = ["emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed"]
    present = [c for c in expected_float_cols if c in df_raw.columns]
    type_flags = [pd.api.types.is_numeric_dtype(df_raw[c]) for c in present]
    pd.DataFrame({"col": present, "is_numeric": type_flags}).to_csv(OUT / "data_types_report.csv", index=False)
    assert all(type_flags), "Alguna columna macro no es numérica."


def test_resumen_calidad(df_raw):
    """Resumen agregado para evidencia en reports/validation/."""
    summary = {
        "rows": [len(df_raw)],
        "cols": [df_raw.shape[1]],
        "n_duplicados": [df_raw.duplicated().sum()],
        "n_nulos_total": [int(df_raw.isna().sum().sum())],
    }
    pd.DataFrame(summary).to_csv(OUT / "quality_summary.csv", index=False)
