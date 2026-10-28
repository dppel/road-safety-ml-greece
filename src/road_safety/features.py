"""
Feature Engineering (`features.py`)
===================================
Provides strictly pre-period feature engineering for the Official ELSTAT Regional
Monthly Panel (13 Greek NUTS-2 Regions, 2021-2025).

Crucial Scientific Rule:
  - Every engineered feature for month `t` is computed strictly from months `<= t - 1`
    or structural/calendar attributes known prior to month `t`.
  - No function or transformer in this module accepts or modifies the target vector `y`.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from road_safety.data_validation import validate_no_leakage

REGIONAL_NUMERIC_FEATURES: List[str] = [
    "month",
    "month_sin",
    "month_cos",
    "is_summer",
    "urban_ratio",
    "motorcycle_share",
    "lag_1m_ksi_rate",
    "lag_2m_ksi_rate",
    "lag_12m_ksi_rate",
    "rolling_3m_ksi_rate",
    "yoy_ksi_rate_diff",
    "lag_1m_total_accidents",
    "lag_12m_total_accidents",
]

REGIONAL_CATEGORICAL_FEATURES: List[str] = ["region"]

# Feature subsets for the Week 4 Empirical Feature Ablation Study
ABLATION_FEATURE_GROUPS: Dict[str, Tuple[List[str], List[str]]] = {
    "1. Calendar & Seasonality Only": (
        ["month", "month_sin", "month_cos", "is_summer"],
        [],
    ),
    "2. Calendar + NUTS-2 Region": (
        ["month", "month_sin", "month_cos", "is_summer"],
        ["region"],
    ),
    "3. Calendar + Structural Exposure (Urban & Moto Share)": (
        ["month", "month_sin", "month_cos", "is_summer", "urban_ratio", "motorcycle_share"],
        [],
    ),
    "4. Full Baseline (+ Autoregressive Historical Lags t-1, t-3, t-12)": (
        REGIONAL_NUMERIC_FEATURES,
        REGIONAL_CATEGORICAL_FEATURES,
    ),
}


def build_regional_panel_features(
    panel_df: pd.DataFrame,
    high_ksi_threshold: float = 0.21,
) -> pd.DataFrame:
    """
    Computes strictly lagged historical features for each Greek NUTS-2 region in
    chronological order (sorted by region, year, month) so that month `t` only uses
    observations from `<= t - 1`.
    """
    df = panel_df.sort_values(["region", "year", "month"]).reset_index(drop=True).copy()

    # Cyclical seasonality
    df["month_sin"] = np.round(np.sin(2.0 * np.pi * df["month"] / 12.0), 6)
    df["month_cos"] = np.round(np.cos(2.0 * np.pi * df["month"] / 12.0), 6)
    df["is_summer"] = df["month"].isin([6, 7, 8, 9]).astype(int)

    # Strictly lagged features per region (shifted by >= 1 month so current month t is never used)
    grp = df.groupby("region", sort=False)
    df["lag_1m_ksi_rate"] = grp["ksi_severity_rate"].shift(1)
    df["lag_2m_ksi_rate"] = grp["ksi_severity_rate"].shift(2)
    df["lag_12m_ksi_rate"] = grp["ksi_severity_rate"].shift(12)
    df["rolling_3m_ksi_rate"] = (
        grp["ksi_severity_rate"]
        .shift(1)
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["lag_1m_total_accidents"] = grp["total_accidents"].shift(1)
    df["lag_12m_total_accidents"] = grp["total_accidents"].shift(12)

    # Backfill initial 2021 lag values using only each region's earliest observation
    df["lag_1m_ksi_rate"] = df.groupby("region")["lag_1m_ksi_rate"].bfill()
    df["lag_2m_ksi_rate"] = df["lag_2m_ksi_rate"].fillna(df["lag_1m_ksi_rate"])
    df["lag_12m_ksi_rate"] = df["lag_12m_ksi_rate"].fillna(df["lag_1m_ksi_rate"])
    df["rolling_3m_ksi_rate"] = df["rolling_3m_ksi_rate"].fillna(df["lag_1m_ksi_rate"])
    df["yoy_ksi_rate_diff"] = np.round(df["lag_1m_ksi_rate"] - df["lag_12m_ksi_rate"], 6)

    df["lag_1m_total_accidents"] = df.groupby("region")["lag_1m_total_accidents"].bfill()
    df["lag_12m_total_accidents"] = df["lag_12m_total_accidents"].fillna(df["lag_1m_total_accidents"])

    # Define binary target for elevated regional KSI regime (>= 16% KSI rate)
    df["target_high_ksi"] = (df["ksi_severity_rate"] >= high_ksi_threshold).astype(int)

    return df


def extract_regional_X_y(
    panel_fe_df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts strictly valid pre-period features X and binary target y from the regional panel,
    verifying via `validate_no_leakage(X)` that no contemporaneous outcome column leaks into X.
    """
    num_c = REGIONAL_NUMERIC_FEATURES if numeric_cols is None else numeric_cols
    cat_c = REGIONAL_CATEGORICAL_FEATURES if categorical_cols is None else categorical_cols
    feature_cols = num_c + cat_c

    X = panel_fe_df[feature_cols].copy()
    validate_no_leakage(X, additional_forbidden=["total_accidents"])
    y = panel_fe_df["target_high_ksi"].astype(int).copy()
    return X, y


class TemporalCyclicalTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn transformer that adds bounded trigonometric cyclical encodings
    (`month_sin`, `month_cos`) and summer tourism indicators (`is_summer`) strictly on X.
    """

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TemporalCyclicalTransformer":
        validate_no_leakage(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        validate_no_leakage(X)
        d = X.copy()
        if "month" in d.columns:
            d["month_sin"] = np.sin(2.0 * np.pi * d["month"] / 12.0)
            d["month_cos"] = np.cos(2.0 * np.pi * d["month"] / 12.0)
            d["is_summer"] = d["month"].isin([6, 7, 8, 9]).astype(int)
        return d
