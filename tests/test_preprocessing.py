"""
Unit Tests for Data Validation & Preprocessing (`test_preprocessing.py`)
========================================================================
Tests sentinel replacement (-1, 99 -> NaN), leakage detection guards,
chronological train/val/test splitting, and unseen category handling.
"""

import numpy as np
import pandas as pd
import pytest

from road_safety.data_validation import (
    assert_target_immutable,
    temporal_train_val_test_split,
    validate_binary_target,
    validate_no_leakage,
)
from road_safety.preprocessing import SentinelCleaner, build_preprocessing_transformer


def test_leakage_guard_rejects_post_accident_columns():
    df_leaked = pd.DataFrame(
        {
            "speed_limit": [50, 90],
            "number_of_casualties": [1, 3],
        }
    )
    with pytest.raises(ValueError, match="Data leakage detected"):
        validate_no_leakage(df_leaked)


def test_sentinel_cleaner_replaces_minus_one_and_99():
    df = pd.DataFrame(
        {
            "speed_limit": [50, -1, 99, 90],
            "light_conditions": [1, 6, -1, 99],
            "weather_conditions": [1, 2, 99, -1],
        }
    )
    cleaner = SentinelCleaner()
    cleaned = cleaner.fit_transform(df)
    assert cleaned["speed_limit"].isna().sum() == 2
    assert cleaned["light_conditions"].isna().sum() == 2
    assert cleaned["weather_conditions"].isna().sum() == 2
    assert cleaned.loc[0, "speed_limit"] == 50.0


def test_temporal_split_strictly_chronological():
    df = pd.DataFrame(
        {
            "year": [2021, 2022, 2023, 2024, 2025],
            "x": [1, 2, 3, 4, 5],
        }
    )
    train_df, val_df, test_df = temporal_train_val_test_split(
        df, year_col="year", train_end_year=2023, val_year=2024, test_year=2025
    )
    assert train_df["year"].max() < val_df["year"].min()
    assert val_df["year"].max() < test_df["year"].min()
    assert len(train_df) == 3
    assert len(val_df) == 1
    assert len(test_df) == 1


def test_preprocessing_transformer_handles_unseen_categories_and_nans():
    X_train = pd.DataFrame(
        {
            "speed_limit": [50.0, 90.0, np.nan, 50.0],
            "region": ["Attica", "Crete", "Attica", "Peloponnese"],
        }
    )
    X_test = pd.DataFrame(
        {
            "speed_limit": [np.nan, 110.0],
            "region": ["Attica", "Unseen_Island_Region"],
        }
    )
    prep = build_preprocessing_transformer(
        numeric_features=["speed_limit"],
        categorical_features=["region"],
    )
    X_tr_trans = prep.fit_transform(X_train)
    X_te_trans = prep.transform(X_test)
    assert not np.isnan(X_tr_trans).any()
    assert not np.isnan(X_te_trans).any()
    assert X_tr_trans.shape[1] == X_te_trans.shape[1]
