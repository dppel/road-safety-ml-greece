"""
Preprocessing Pipelines (`preprocessing.py`)
============================================
Scikit-learn compatible transformers and ColumnTransformer builders that
clean sentinel values (-1, 99 -> NaN) and fit imputers/scalers/encoders
exclusively on the training partition.
"""

from typing import Iterable, List, Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from road_safety.data_validation import validate_no_leakage


class SentinelCleaner(BaseEstimator, TransformerMixin):
    """
    Replaces statistical sentinel codes (e.g., -1, 99 in STATS19 / ELSTAT tables)
    with np.nan prior to train-only imputation.
    """

    def __init__(
        self,
        sentinel_values: Sequence[int | float] = (-1, 99),
        columns: Optional[Sequence[str]] = None,
    ) -> None:
        self.sentinel_values = sentinel_values
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "SentinelCleaner":
        validate_no_leakage(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        validate_no_leakage(X)
        X_out = X.copy()
        target_cols = (
            list(self.columns)
            if self.columns is not None
            else [
                c
                for c in X_out.columns
                if c
                in {
                    "speed_limit",
                    "light_conditions",
                    "weather_conditions",
                    "road_surface_conditions",
                    "road_type",
                    "area_type",
                }
            ]
        )
        for col in target_cols:
            if col in X_out.columns:
                X_out[col] = X_out[col].replace(list(self.sentinel_values), np.nan)
        return X_out


def build_preprocessing_transformer(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    """
    Constructs a leak-free ColumnTransformer:
      - Numeric pipeline: SimpleImputer(strategy='median') -> StandardScaler()
      - Categorical pipeline: SimpleImputer(strategy='most_frequent') -> OneHotEncoder(handle_unknown='ignore')
    """
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, list(numeric_features)),
            ("cat", cat_pipe, list(categorical_features)),
        ],
        remainder="drop",
    )
