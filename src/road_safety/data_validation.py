"""
Data Validation & Anti-Leakage Guards (`data_validation.py`)
============================================================
Enforces schema completeness, chronological split boundaries, and strict
exclusion of post-collision or contemporaneous target-derived columns.
"""

from typing import Iterable, List, Tuple
import numpy as np
import pandas as pd

# Canonical post-collision / target-derived columns that must NEVER appear in feature matrix X
DEFAULT_FORBIDDEN_LEAKAGE_COLUMNS = frozenset({
    # Micro-level post-accident / endogenous columns
    "target_ksi",
    "collision_severity",
    "number_of_casualties",
    "fatal_casualties",
    "serious_casualties",
    "slight_casualties",
    "did_police_officer_attend_scene_of_accident",
    # Regional panel contemporaneous outcome columns
    "target_high_ksi",
    "ksi_accidents",
    "fatal_accidents",
    "serious_accidents",
    "slight_accidents",
    "ksi_severity_rate",
})


def validate_no_leakage(
    X: pd.DataFrame,
    additional_forbidden: Iterable[str] = (),
) -> None:
    """
    Raises a ValueError if any forbidden post-collision or target-derived
    column is present in the feature matrix X.
    """
    forbidden = set(DEFAULT_FORBIDDEN_LEAKAGE_COLUMNS).union(set(additional_forbidden))
    leaked = sorted(forbidden.intersection(set(X.columns)))
    if leaked:
        raise ValueError(
            f"Data leakage detected! Forbidden post-accident or outcome columns found in X: {leaked}"
        )


def validate_schema(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str = "dataset",
) -> None:
    """
    Verifies that all required columns exist in the DataFrame and that the DataFrame is non-empty.
    """
    if df is None or df.empty:
        raise ValueError(f"{dataset_name} is empty or None.")
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def validate_binary_target(y: pd.Series | np.ndarray, name: str = "target") -> None:
    """
    Verifies that the target vector contains no NaNs and consists exclusively of binary {0, 1} labels.
    """
    arr = np.asarray(y)
    if arr.size == 0:
        raise ValueError(f"{name} vector is empty.")
    if pd.isna(arr).any():
        raise ValueError(f"{name} contains NaN/missing values.")
    unique_vals = set(np.unique(arr).tolist())
    if not unique_vals.issubset({0, 1}):
        raise ValueError(f"{name} must be strictly binary {{0, 1}}, found values: {sorted(unique_vals)}")


def assert_target_immutable(
    y_before: pd.Series | np.ndarray,
    y_after: pd.Series | np.ndarray,
) -> None:
    """
    Guarantees that a feature engineering or enrichment step did not modify the ground-truth target y.
    Directly prevents the target-overwrite bug identified in prototype waze_feature_enrichment.py.
    """
    arr_before = np.asarray(y_before)
    arr_after = np.asarray(y_after)
    if arr_before.shape != arr_after.shape or not np.array_equal(arr_before, arr_after):
        raise ValueError(
            "Target immutability violation! Ground-truth target vector y was modified during feature processing."
        )


def temporal_train_val_test_split(
    df: pd.DataFrame,
    year_col: str,
    train_end_year: int = 2023,
    val_year: int = 2024,
    test_year: int = 2025,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a dataset strictly chronologically into train, validation, and test sets
    and verifies zero temporal overlap.
    """
    if year_col not in df.columns:
        raise ValueError(f"Year column '{year_col}' not found in DataFrame.")

    train_df = df[df[year_col] <= train_end_year].copy()
    val_df = df[df[year_col] == val_year].copy()
    test_df = df[df[year_col] == test_year].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            f"Temporal split produced an empty partition: "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    max_train = int(train_df[year_col].max())
    min_val = int(val_df[year_col].min())
    max_val = int(val_df[year_col].max())
    min_test = int(test_df[year_col].min())

    if not (max_train < min_val <= max_val < min_test):
        raise ValueError(
            f"Chronological ordering violated: max_train={max_train}, val=[{min_val},{max_val}], min_test={min_test}"
        )

    return train_df, val_df, test_df
