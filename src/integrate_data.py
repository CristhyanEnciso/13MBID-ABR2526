import pandas as pd
from pathlib import Path

df = pd.read_csv("data/interim/banking_features.csv")
Path("data/processed").mkdir(parents=True, exist_ok=True)
df.to_csv("data/processed/banking_model_base.csv", index=False)
