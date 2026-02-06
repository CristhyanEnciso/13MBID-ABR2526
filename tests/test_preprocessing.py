import pandas as pd
from pathlib import Path
import joblib

def test_preprocessor_can_transform_sample():
    df = pd.read_csv("data/processed/bank_formatted.csv")
    y = df["y"]
    X = df.drop(columns=["y"]).head(10)

    preprocessor = joblib.load("models/preprocessor.pkl")
    X_tr = preprocessor.transform(X)

    assert X_tr.shape[0] == X.shape[0]
    # al menos algún número razonable de columnas
    assert X_tr.shape[1] > 10  
