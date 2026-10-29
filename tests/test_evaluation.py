"""
Unit Tests for Model Evaluation & Metrics (`test_evaluation.py`)
================================================================
Tests evaluation metrics, confusion matrix accounting, optimal F1 threshold
selection on validation sets, and bootstrap confidence intervals.
"""

import numpy as np
import pytest

from road_safety.evaluate import (
    bootstrap_metric_ci,
    evaluate_predictions,
    find_optimal_f1_threshold,
)


def test_evaluate_predictions_confusion_matrix_sums_to_n():
    y_test = np.array([0, 0, 0, 0, 1, 1, 0, 1, 0, 1])
    probs = np.array([0.1, 0.2, 0.6, 0.3, 0.8, 0.7, 0.2, 0.4, 0.1, 0.9])
    metrics = evaluate_predictions("TestModel", y_test, probs, threshold=0.5, n_bootstraps=50, seed=42)

    total_cm = metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"]
    assert total_cm == len(y_test)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert metrics["tp"] == 3
    assert metrics["fn"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 5


def test_optimal_f1_threshold_stays_within_bounds():
    y_val = np.array([0, 0, 0, 1, 0, 1, 0, 0, 1, 0])
    val_probs = np.array([0.05, 0.10, 0.12, 0.35, 0.15, 0.42, 0.08, 0.11, 0.55, 0.09])
    thresh = find_optimal_f1_threshold(y_val, val_probs)
    assert 0.05 <= thresh <= 0.95
    # Since all positives have prob >= 0.35 and all negatives <= 0.15, threshold around 0.35 gives F1=1.0
    assert 0.16 <= thresh <= 0.40


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.RandomState(42)
    y = rng.binomial(1, 0.25, size=200)
    probs = np.clip(y * 0.5 + rng.uniform(0, 0.5, size=200), 0.01, 0.99)
    low, high = bootstrap_metric_ci(y, probs, n_bootstraps=100, seed=42)
    assert low <= high
    assert 0.5 <= low <= 1.0


def test_prospective_forecast_and_policy_recommendations():
    from pathlib import Path
    import pandas as pd
    from road_safety.forecast import generate_prospective_12m_forecast

    panel_path = Path(__file__).resolve().parents[1] / "data" / "elstat_regional_monthly_panel_2021_2025.csv"
    df_panel = pd.read_csv(panel_path)
    df_panel = df_panel[df_panel["year"].between(2021, 2025)].copy()

    fc_df = generate_prospective_12m_forecast(df_panel, forecast_year=2026, seed=42)
    # 13 regions x 12 months = 156 prospective predictions
    assert len(fc_df) == 156
    assert (fc_df["predicted_ksi_rate"].between(0.01, 0.99)).all()
    assert (fc_df["severe_regime_prob"].between(0.0, 1.0)).all()
    assert (fc_df["recommended_interventions"].str.len() > 20).all()

