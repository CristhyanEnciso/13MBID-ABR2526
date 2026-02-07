# src/feature_aligner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureAligner(BaseEstimator, TransformerMixin):
    """
    Alinea columnas de entrada para que coincidan con un esquema esperado.
    - Si faltan columnas, las crea con 0.
    - Si sobran columnas, las elimina.
    - Mantiene el orden de columnas.
    """

    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.feature_columns = feature_columns
        self.feature_columns_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        if self.feature_columns is None:
            self.feature_columns_ = list(X.columns)
        else:
            self.feature_columns_ = list(self.feature_columns)
        return self

    def transform(self, X: pd.DataFrame):
        if self.feature_columns_ is None:
            raise ValueError("FeatureAligner no está fitted. Ejecuta fit() primero.")

        X = X.copy()

        # crear faltantes
        for col in self.feature_columns_:
            if col not in X.columns:
                X[col] = 0

        # eliminar sobrantes y ordenar
        X = X[self.feature_columns_]
        return X
