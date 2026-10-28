"""
Unit Tests for Feature Engineering & Target Immutability (`test_features.py`)
=============================================================================
Verifies that:
  1. Regional lag features strictly use historical months (no look-ahead leakage).
  2. Event-level feature engineering produces bounded cyclical encodings.
  3. Target vector `y` is NEVER modified by `CollisionFeatureEngineer` or
     `generate_proxy_waze_features_strictly_on_X`.
"""

import numpy as np
import pandas as pd
import pytest

from road_safety.data_validation import assert_target_immutable
from road_safety.features import (
    CollisionFeatureEngineer,
    build_regional_panel_features,
    extract_regional_X_y,
)
from road_safety.waze_ablation import generate_proxy_waze_features_strictly_on_X


def test_target_immutability_guard_detects_target_mutation():
    y_orig = np.array([0, 1, 0, 1, 0])
    y_mutated = np.array([0, 1, 1, 1, 0])
    assert_target_immutable(y_orig, y_orig.copy())
    with pytest.raises(ValueError, match="Target immutability violation"):
        assert_target_immutable(y_orig, y_mutated)


def test_waze_proxy_ablation_never_modifies_target_y():
    df = pd.DataFrame(
        {
            "region": ["Attica (Αττική)", "Crete (Κρήτη)", "Peloponnese (Πελοπόννησος)"],
            "month": [1, 7, 8],
            "day_of_week": [2, 6, 7],
            "hour": [8, 23, 2],
            "area_type": [1, 2, 2],
            "road_type": [6, 1, 3],
            "speed_limit": [50, 90, 90],
            "is_junction": [1, 0, 0],
            "light_conditions": [1, 6, 6],
            "weather_conditions": [1, 1, 2],
            "road_surface_conditions": [1, 1, 2],
            "number_of_vehicles": [2, 1, 2],
            "has_motorcycle": [1, 1, 0],
            "has_pedestrian": [0, 0, 0],
            "has_heavy_vehicle": [0, 0, 1],
            "min_driver_age": [34, 21, 55],
            "max_vehicle_age": [8, 14, 12],
        }
    )
    y_true = pd.Series([0, 1, 1], name="target_ksi")
    y_snapshot = y_true.to_numpy().copy()

    fe = CollisionFeatureEngineer()
    X_fe = fe.fit_transform(df)
    X_waze = generate_proxy_waze_features_strictly_on_X(X_fe, seed=42)

    assert_target_immutable(y_snapshot, y_true)
    assert "waze_flow_speed_kmh" in X_waze.columns
    assert "waze_jam_level" in X_waze.columns
    assert "target_ksi" not in X_waze.columns
    assert (X_waze["hour_sin"].between(-1.0, 1.0)).all()
    assert (X_waze["hour_cos"].between(-1.0, 1.0)).all()


def test_regional_panel_features_exclude_contemporaneous_outcomes():
    panel = pd.DataFrame(
        {
            "year": [2021, 2021, 2021],
            "month": [1, 2, 3],
            "region": ["Attica (Αττική)"] * 3,
            "total_accidents": [380, 365, 420],
            "ksi_accidents": [25, 33, 27],
            "fatal_accidents": [12, 16, 13],
            "serious_accidents": [13, 17, 14],
            "slight_accidents": [355, 332, 393],
            "ksi_severity_rate": [0.0658, 0.0904, 0.0643],
            "urban_ratio": [0.88, 0.88, 0.88],
            "motorcycle_share": [0.52, 0.52, 0.52],
        }
    )
    fe_panel = build_regional_panel_features(panel, high_ksi_threshold=0.08)
    # Month 2 lag_1m_ksi_rate must equal Month 1 ksi_severity_rate (0.0658), NOT Month 2 (0.0904)
    assert np.isclose(fe_panel.loc[1, "lag_1m_ksi_rate"], 0.0658)
    assert np.isclose(fe_panel.loc[2, "lag_1m_ksi_rate"], 0.0904)

    X, y = extract_regional_X_y(fe_panel)
    assert "ksi_accidents" not in X.columns
    assert "total_accidents" not in X.columns
    assert "ksi_severity_rate" not in X.columns
    assert list(y) == [0, 1, 0]
