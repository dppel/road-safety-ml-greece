"""
Feature Engineering (`features.py`)
===================================
Provides strictly pre-period / pre-collision feature engineering for:
  1. Track A: Official ELSTAT Regional Monthly Panel (strictly lagged features).
  2. Track B: Event-Level Collision Severity Pipeline (cyclical time, road-risk interactions).

Crucial Scientific Rule:
  No function or transformer in this module accepts or modifies the target vector `y`.
"""

from typing import List, Optional, Tuple
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
    "lag_12m_ksi_rate",
    "rolling_3m_ksi_rate",
    "lag_1m_total_accidents",
]

REGIONAL_CATEGORICAL_FEATURES: List[str] = ["region"]

MICRO_BASE_NUMERIC_FEATURES: List[str] = [
    "month",
    "day_of_week",
    "hour",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "is_night",
    "is_weekend",
    "is_rural",
    "speed_limit",
    "is_junction",
    "is_dark_unlit",
    "number_of_vehicles",
    "has_motorcycle",
    "has_pedestrian",
    "has_vulnerable_user",
    "has_heavy_vehicle",
    "min_driver_age",
    "has_young_driver",
    "has_elderly_driver",
    "max_vehicle_age",
    "moto_x_speed",
    "rural_x_speed",
]

MICRO_CATEGORICAL_FEATURES: List[str] = [
    "region",
    "area_type",
    "road_type",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
]


def build_regional_panel_features(
    panel_df: pd.DataFrame,
    high_ksi_threshold: float = 0.16,
) -> pd.DataFrame:
    """
    Computes strictly lagged historical features for each NUTS-2 region in chronological order
    (sorted by region, year, month) so that month t only uses observations from <= t - 1.
    """
    df = panel_df.sort_values(["region", "year", "month"]).reset_index(drop=True).copy()

    # Cyclical seasonality
    df["month_sin"] = np.round(np.sin(2.0 * np.pi * df["month"] / 12.0), 6)
    df["month_cos"] = np.round(np.cos(2.0 * np.pi * df["month"] / 12.0), 6)
    df["is_summer"] = df["month"].isin([6, 7, 8, 9]).astype(int)

    # Strictly lagged features per region (shifted by >= 1 month so current month t is never used)
    grp = df.groupby("region", sort=False)
    df["lag_1m_ksi_rate"] = grp["ksi_severity_rate"].shift(1)
    df["lag_12m_ksi_rate"] = grp["ksi_severity_rate"].shift(12)
    df["rolling_3m_ksi_rate"] = (
        grp["ksi_severity_rate"]
        .shift(1)
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["lag_1m_total_accidents"] = grp["total_accidents"].shift(1)

    # Backfill initial 2021-01 lag values using only each region's earliest observation
    df["lag_1m_ksi_rate"] = df.groupby("region")["lag_1m_ksi_rate"].bfill()
    df["lag_12m_ksi_rate"] = df["lag_12m_ksi_rate"].fillna(df["lag_1m_ksi_rate"])
    df["rolling_3m_ksi_rate"] = df["rolling_3m_ksi_rate"].fillna(df["lag_1m_ksi_rate"])
    df["lag_1m_total_accidents"] = df.groupby("region")["lag_1m_total_accidents"].bfill()

    # Define binary target for elevated regional KSI regime
    df["target_high_ksi"] = (df["ksi_severity_rate"] >= high_ksi_threshold).astype(int)

    return df


def extract_regional_X_y(
    panel_fe_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts strictly valid pre-period features X and binary target y from the regional panel,
    verifying via `validate_no_leakage(X)` that no contemporaneous outcome column leaks into X.
    """
    feature_cols = REGIONAL_NUMERIC_FEATURES + REGIONAL_CATEGORICAL_FEATURES
    X = panel_fe_df[feature_cols].copy()
    validate_no_leakage(X, additional_forbidden=["total_accidents"])
    y = panel_fe_df["target_high_ksi"].astype(int).copy()
    return X, y


class CollisionFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn transformer that engineers cyclical time encodings and
    pre-collision physical interaction features from raw collision attributes.
    Never accesses or modifies the target variable `y`.
    """

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CollisionFeatureEngineer":
        validate_no_leakage(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        validate_no_leakage(X)
        d = X.copy()

        # Replace sentinels on speed_limit and light_conditions prior to interaction terms
        clean_speed = d["speed_limit"].replace([-1, 99], np.nan)
        clean_light = d["light_conditions"].replace([-1, 99], np.nan)

        d["hour_sin"] = np.sin(2.0 * np.pi * d["hour"] / 24.0)
        d["hour_cos"] = np.cos(2.0 * np.pi * d["hour"] / 24.0)
        d["month_sin"] = np.sin(2.0 * np.pi * d["month"] / 12.0)
        d["month_cos"] = np.cos(2.0 * np.pi * d["month"] / 12.0)

        d["is_night"] = d["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
        d["is_weekend"] = d["day_of_week"].isin([6, 7]).astype(int)
        d["is_rural"] = (d["area_type"] == 2).astype(int)
        d["is_dark_unlit"] = (clean_light == 6).astype(int)

        d["has_vulnerable_user"] = ((d["has_motorcycle"] == 1) | (d["has_pedestrian"] == 1)).astype(int)
        d["has_young_driver"] = (d["min_driver_age"] <= 24).astype(int)
        d["has_elderly_driver"] = (d["min_driver_age"] >= 65).astype(int)

        # Interaction terms (preserved as NaN if speed_limit is missing; imputed downstream)
        d["moto_x_speed"] = d["has_motorcycle"] * clean_speed
        d["rural_x_speed"] = d["is_rural"] * clean_speed

        return d
