"""
Unit Tests for Official ELSTAT Feature Engineering & Target Immutability (`test_features.py`)
=============================================================================================
Verifies that:
  1. Regional lag features strictly use historical months (no look-ahead leakage).
  2. Cyclical time encodings (`month_sin`, `month_cos`) are bounded in [-1, 1].
  3. Target vector `y` is NEVER modified during feature engineering (`assert_target_immutable`).
"""

import numpy as np
import pandas as pd
import pytest

from road_safety.data_validation import assert_target_immutable
from road_safety.features import (
    TemporalCyclicalTransformer,
    build_regional_panel_features,
    extract_regional_X_y,
)


def test_target_immutability_guard_detects_target_mutation():
    y_orig = np.array([0, 1, 0, 1, 0])
    y_mutated = np.array([0, 1, 1, 1, 0])
    assert_target_immutable(y_orig, y_orig.copy())
    with pytest.raises(ValueError, match="Target immutability violation"):
        assert_target_immutable(y_orig, y_mutated)


def test_temporal_cyclical_transformer_bounds_and_target_immutability():
    df = pd.DataFrame(
        {
            "region": ["Attica (Αττική)", "Crete (Κρήτη)", "Peloponnese (Πελοπόννησος)"],
            "month": [1, 7, 12],
            "urban_ratio": [0.88, 0.42, 0.38],
        }
    )
    y_true = pd.Series([0, 1, 1], name="target_high_ksi")
    y_snapshot = y_true.to_numpy().copy()

    tr = TemporalCyclicalTransformer()
    X_trans = tr.fit_transform(df)

    assert_target_immutable(y_snapshot, y_true)
    assert (X_trans["month_sin"].between(-1.0, 1.0)).all()
    assert (X_trans["month_cos"].between(-1.0, 1.0)).all()
    assert list(X_trans["is_summer"]) == [0, 1, 0]


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
