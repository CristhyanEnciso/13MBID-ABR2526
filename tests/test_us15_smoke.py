import os
import pandas as pd

DATA_PATH = "data/processed/bank_formatted.csv"

def test_dataset_exists():
    assert os.path.exists(DATA_PATH), f"No existe el dataset esperado: {DATA_PATH}"

def test_target_exists_and_is_binary():
    df = pd.read_csv(DATA_PATH)
    assert "y" in df.columns, "No existe la columna objetivo 'y'"
    vals = set(df["y"].dropna().unique().tolist())
    assert vals.issubset({0, 1}), f"La columna 'y' no es binaria (0/1). Valores: {vals}"

def test_class_distribution_is_imbalanced():
    df = pd.read_csv(DATA_PATH)
    ratio_pos = df["y"].mean()
    assert ratio_pos < 0.30, f"No se observa desbalance suficiente (ratio_pos={ratio_pos:.3f})"
