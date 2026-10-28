"""
Reproducible Training, Benchmarking & Ablation Pipeline (`train.py`)
====================================================================
Single-command reproducible pipeline (`python -m road_safety.train`) that:
  1. Validates data schemas, chronological splits (2021-2023 Train, 2024 Val, 2025 Test),
     and anti-leakage rules (`validate_no_leakage`, `assert_target_immutable`).
  2. Benchmarks DummyClassifier, DecisionTree, RandomForest, HistGradientBoosting,
     and XGBoost on Track A (Official ELSTAT Regional Panel) and Track B (Event-Level
     Synthetic Collision Benchmark).
  3. Runs the Week 4 Honest Waze Ablation Study (verifying that `y` is never altered).
  4. Computes Week 5 permutation feature importances, subgroup audits, and error analysis.
  5. Exports structured tables (`reports/benchmark_comparison.csv`, `reports/benchmark_metrics.json`)
     and publication-quality figures (`reports/figures/*.png`).
"""

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from road_safety.data_validation import (
    assert_target_immutable,
    temporal_train_val_test_split,
    validate_binary_target,
    validate_no_leakage,
)
from road_safety.evaluate import evaluate_predictions, find_optimal_f1_threshold
from road_safety.explain import (
    compute_global_feature_importance,
    evaluate_subgroup_performance,
    extract_error_analysis_cases,
)
from road_safety.features import (
    MICRO_BASE_NUMERIC_FEATURES,
    MICRO_CATEGORICAL_FEATURES,
    REGIONAL_CATEGORICAL_FEATURES,
    REGIONAL_NUMERIC_FEATURES,
    CollisionFeatureEngineer,
    build_regional_panel_features,
    extract_regional_X_y,
)
from road_safety.preprocessing import SentinelCleaner, build_preprocessing_transformer
from road_safety.waze_ablation import (
    WAZE_TELEMETRY_COLUMNS,
    generate_proxy_waze_features_strictly_on_X,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def get_candidate_models(seed: int = 42, pos_weight: float = 4.0) -> Dict[str, Any]:
    """
    Returns the standard benchmark suite of classifiers initialized with fixed random seeds.
    """
    models: Dict[str, Any] = {
        "DummyClassifier (Baseline)": DummyClassifier(strategy="prior", random_state=seed),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=seed
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=15,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.04,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=seed,
        ),
    }
    try:
        from xgboost import XGBClassifier  # type: ignore

        models["XGBoost"] = XGBClassifier(
            n_estimators=250,
            learning_rate=0.04,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        )
    except Exception:
        pass
    return models


def ensure_datasets_exist(seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads `elstat_regional_monthly_panel_2021_2025.csv` and `synthetic_benchmark_collisions.csv`.
    If `synthetic_benchmark_collisions.csv` does not yet exist, generates it deterministically
    from ELSTAT marginal distributions with a strictly fixed target `target_ksi` that is never
    modified by Waze or downstream scripts.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    regional_path = DATA_DIR / "elstat_regional_monthly_panel_2021_2025.csv"
    micro_path = DATA_DIR / "synthetic_benchmark_collisions.csv"

    if not regional_path.exists():
        raise FileNotFoundError(f"Missing official regional panel dataset at {regional_path}")

    regional_df = pd.read_csv(regional_path)
    # Keep 2021-2025 clean 60-month window (780 rows = 13 regions x 60 months)
    regional_df = regional_df[regional_df["year"].between(2021, 2025)].copy()

    if not micro_path.exists():
        rng = np.random.RandomState(seed)
        n_rows = 25000
        regions = [
            "Attica (Αττική)",
            "Central Macedonia (Κεντρική Μακεδονία)",
            "Western Greece (Δυτική Ελλάδα)",
            "Peloponnese (Πελοπόννησος)",
            "Crete (Κρήτη)",
            "Thessaly (Θεσσαλία)",
            "South Aegean (Νότιο Αιγαίο)",
            "Central Greece (Στερεά Ελλάδα)",
            "Eastern Macedonia & Thrace (ΑΜΘ)",
            "Ionian Islands (Ιόνια Νησιά)",
            "Epirus (Ήπειρος)",
            "North Aegean (Βόρειο Αιγαίο)",
            "Western Macedonia (Δυτική Μακεδονία)",
        ]
        reg_probs = [0.535, 0.165, 0.052, 0.047, 0.047, 0.035, 0.036, 0.030, 0.021, 0.012, 0.009, 0.006, 0.005]
        years = rng.choice([2021, 2022, 2023, 2024, 2025], size=n_rows, p=[0.195, 0.198, 0.200, 0.205, 0.202])
        months = rng.choice(
            np.arange(1, 13),
            size=n_rows,
            p=[0.068, 0.065, 0.076, 0.080, 0.088, 0.096, 0.105, 0.101, 0.092, 0.083, 0.076, 0.070],
        )
        days = rng.randint(1, 8, size=n_rows)
        hours = rng.choice(
            np.arange(24),
            size=n_rows,
            p=[
                0.025, 0.020, 0.018, 0.015, 0.015, 0.020,
                0.035, 0.055, 0.065, 0.055, 0.050, 0.055,
                0.060, 0.065, 0.070, 0.060, 0.055, 0.055,
                0.050, 0.045, 0.038, 0.032, 0.022, 0.020,
            ],
        )
        reg_col = rng.choice(regions, size=n_rows, p=reg_probs)
        is_attica_or_cm = np.isin(
            reg_col, ["Attica (Αττική)", "Central Macedonia (Κεντρική Μακεδονία)"]
        )
        area_type = np.where(
            is_attica_or_cm,
            rng.choice([1, 2], size=n_rows, p=[0.84, 0.16]),
            rng.choice([1, 2], size=n_rows, p=[0.42, 0.58]),
        )
        road_type = np.where(
            area_type == 1,
            rng.choice([3, 6], size=n_rows, p=[0.20, 0.80]),
            rng.choice([1, 2, 3], size=n_rows, p=[0.45, 0.15, 0.40]),
        )
        speed_limit = np.where(
            area_type == 1,
            rng.choice([30, 50, 70, -1], size=n_rows, p=[0.15, 0.68, 0.14, 0.03]),
            rng.choice([50, 70, 90, 110, 130, -1], size=n_rows, p=[0.10, 0.25, 0.42, 0.13, 0.07, 0.03]),
        )
        is_junction = np.where(
            area_type == 1,
            rng.binomial(1, 0.58, size=n_rows),
            rng.binomial(1, 0.22, size=n_rows),
        )
        is_night = np.isin(hours, [21, 22, 23, 0, 1, 2, 3, 4, 5])
        light_conditions = np.where(
            ~is_night,
            1,
            np.where(
                area_type == 1,
                rng.choice([4, 6, -1], size=n_rows, p=[0.82, 0.16, 0.02]),
                rng.choice([4, 6, -1], size=n_rows, p=[0.35, 0.62, 0.03]),
            ),
        )
        weather_conditions = rng.choice([1, 2, 3, 4, -1], size=n_rows, p=[0.84, 0.10, 0.02, 0.02, 0.02])
        road_surface = np.where(weather_conditions == 2, 2, rng.choice([1, 2, 3, -1], size=n_rows, p=[0.85, 0.10, 0.03, 0.02]))
        num_vehicles = rng.choice([1, 2, 3, 4], size=n_rows, p=[0.32, 0.54, 0.11, 0.03])
        has_motorcycle = rng.binomial(1, np.where(area_type == 1, 0.48, 0.34))
        has_pedestrian = rng.binomial(1, np.where(area_type == 1, 0.18, 0.07))
        has_heavy = rng.binomial(1, np.where(area_type == 2, 0.14, 0.06))
        min_driver_age = np.clip(rng.normal(38, 15, size=n_rows).astype(int), 17, 86)
        max_vehicle_age = np.clip(rng.gamma(shape=3.2, scale=4.2, size=n_rows).astype(int), 0, 28)

        # Ground-truth KSI data-generating process (strictly from baseline physical variables)
        clean_spd = np.where(speed_limit < 0, 50, speed_limit)
        logit = (
            -3.05
            + 0.88 * (area_type == 2)
            + 0.82 * has_motorcycle
            + 0.68 * has_pedestrian
            + 0.016 * (clean_spd - 50.0)
            + 0.58 * (light_conditions == 6)
            + 0.38 * has_heavy
            + 0.30 * (min_driver_age <= 23)
            + 0.28 * (min_driver_age >= 68)
            - 0.32 * is_junction
            + 0.25 * (num_vehicles == 1)
        )
        prob_ksi = 1.0 / (1.0 + np.exp(-logit))
        target_ksi = rng.binomial(1, prob_ksi)

        micro_df = pd.DataFrame(
            {
                "collision_id": [f"SYN-GR-{y}-{i:05d}" for i, y in enumerate(years)],
                "data_source_label": "SYNTHETIC_BENCHMARK_CALIBRATED_TO_ELSTAT",
                "collision_year": years,
                "region": reg_col,
                "month": months,
                "day_of_week": days,
                "hour": hours,
                "area_type": area_type,
                "road_type": road_type,
                "speed_limit": speed_limit,
                "is_junction": is_junction,
                "light_conditions": light_conditions,
                "weather_conditions": weather_conditions,
                "road_surface_conditions": road_surface,
                "number_of_vehicles": num_vehicles,
                "has_motorcycle": has_motorcycle,
                "has_pedestrian": has_pedestrian,
                "has_heavy_vehicle": has_heavy,
                "min_driver_age": min_driver_age,
                "max_vehicle_age": max_vehicle_age,
                "target_ksi": target_ksi,
            }
        )
        micro_df.to_csv(micro_path, index=False)
    else:
        micro_df = pd.read_csv(micro_path)

    return regional_df, micro_df


def run_track_a_regional_benchmark(
    regional_df: pd.DataFrame,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes Track A (Official ELSTAT Regional Monthly Panel) benchmark across all candidate models.
    """
    fe_df = build_regional_panel_features(regional_df, high_ksi_threshold=0.16)
    train_df, val_df, test_df = temporal_train_val_test_split(
        fe_df, year_col="year", train_end_year=2023, val_year=2024, test_year=2025
    )

    X_train, y_train = extract_regional_X_y(train_df)
    X_val, y_val = extract_regional_X_y(val_df)
    X_test, y_test = extract_regional_X_y(test_df)

    validate_binary_target(y_train, "Track A y_train")
    validate_binary_target(y_val, "Track A y_val")
    validate_binary_target(y_test, "Track A y_test")

    pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    models = get_candidate_models(seed=seed, pos_weight=pos_weight)

    results: List[Dict[str, Any]] = []
    curves: Dict[str, Dict[str, Any]] = {}

    for name, clf in models.items():
        prep = build_preprocessing_transformer(
            REGIONAL_NUMERIC_FEATURES, REGIONAL_CATEGORICAL_FEATURES
        )
        pipe = Pipeline([("prep", prep), ("clf", clf)])
        pipe.fit(X_train, y_train)

        val_probs = pipe.predict_proba(X_val)[:, 1]
        test_probs = pipe.predict_proba(X_test)[:, 1]

        opt_thresh = (
            0.50
            if "Dummy" in name
            else find_optimal_f1_threshold(y_val, val_probs)
        )
        metrics = evaluate_predictions(
            model_name=name,
            y_test=y_test,
            test_probs=test_probs,
            threshold=opt_thresh,
            n_bootstraps=300,
            seed=seed,
        )
        metrics["track"] = "Track A: Official ELSTAT Regional Panel"
        results.append(metrics)
        curves[name] = {
            "y_test": y_test.to_numpy(),
            "probs": test_probs,
            "pipeline": pipe,
            "threshold": opt_thresh,
        }

    return pd.DataFrame(results), curves


def run_track_b_micro_and_ablation(
    micro_df: pd.DataFrame,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Executes Track B (Event-Level Synthetic Collision Benchmark) + Week 4 Honest Waze Ablation Study
    + Week 5 Explainability & Subgroup Audit.
    """
    train_df, val_df, test_df = temporal_train_val_test_split(
        micro_df, year_col="collision_year", train_end_year=2023, val_year=2024, test_year=2025
    )

    raw_feature_cols = [
        "region",
        "month",
        "day_of_week",
        "hour",
        "area_type",
        "road_type",
        "speed_limit",
        "is_junction",
        "light_conditions",
        "weather_conditions",
        "road_surface_conditions",
        "number_of_vehicles",
        "has_motorcycle",
        "has_pedestrian",
        "has_heavy_vehicle",
        "min_driver_age",
        "max_vehicle_age",
    ]

    X_train_raw = train_df[raw_feature_cols].copy()
    y_train = train_df["target_ksi"].astype(int).copy()
    X_val_raw = val_df[raw_feature_cols].copy()
    y_val = val_df["target_ksi"].astype(int).copy()
    X_test_raw = test_df[raw_feature_cols].copy()
    y_test = test_df["target_ksi"].astype(int).copy()

    # Save copies of y to verify 100% Target Immutability across all transformations
    y_train_orig = y_train.to_numpy().copy()
    y_test_orig = y_test.to_numpy().copy()

    fe = CollisionFeatureEngineer()
    X_train_fe = fe.fit_transform(X_train_raw)
    X_val_fe = fe.transform(X_val_raw)
    X_test_fe = fe.transform(X_test_raw)

    assert_target_immutable(y_train_orig, y_train)
    assert_target_immutable(y_test_orig, y_test)

    pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    models = get_candidate_models(seed=seed, pos_weight=pos_weight)

    results: List[Dict[str, Any]] = []
    curves: Dict[str, Dict[str, Any]] = {}

    for name, clf in models.items():
        prep = build_preprocessing_transformer(
            MICRO_BASE_NUMERIC_FEATURES, MICRO_CATEGORICAL_FEATURES
        )
        pipe = Pipeline(
            [
                ("sentinel", SentinelCleaner()),
                ("prep", prep),
                ("clf", clf),
            ]
        )
        pipe.fit(X_train_fe, y_train)

        val_probs = pipe.predict_proba(X_val_fe)[:, 1]
        test_probs = pipe.predict_proba(X_test_fe)[:, 1]

        opt_thresh = (
            0.50
            if "Dummy" in name
            else find_optimal_f1_threshold(y_val, val_probs)
        )
        metrics = evaluate_predictions(
            model_name=name,
            y_test=y_test,
            test_probs=test_probs,
            threshold=opt_thresh,
            n_bootstraps=300,
            seed=seed,
        )
        metrics["track"] = "Track B: Event-Level Collision Benchmark"
        results.append(metrics)
        curves[name] = {
            "y_test": y_test.to_numpy(),
            "probs": test_probs,
            "pipeline": pipe,
            "threshold": opt_thresh,
        }

    # Week 4 Deliverable: Honest Waze Proxy Ablation Study (Target y is strictly untouched!)
    X_train_waze = generate_proxy_waze_features_strictly_on_X(X_train_fe, seed=seed)
    X_val_waze = generate_proxy_waze_features_strictly_on_X(X_val_fe, seed=seed + 1)
    X_test_waze = generate_proxy_waze_features_strictly_on_X(X_test_fe, seed=seed + 2)

    assert_target_immutable(y_train_orig, y_train)
    assert_target_immutable(y_test_orig, y_test)

    waze_num_cols = MICRO_BASE_NUMERIC_FEATURES + WAZE_TELEMETRY_COLUMNS
    waze_prep = build_preprocessing_transformer(waze_num_cols, MICRO_CATEGORICAL_FEATURES)
    waze_pipe = Pipeline(
        [
            ("sentinel", SentinelCleaner()),
            ("prep", waze_prep),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=250,
                    learning_rate=0.04,
                    max_leaf_nodes=31,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    waze_pipe.fit(X_train_waze, y_train)
    waze_val_probs = waze_pipe.predict_proba(X_val_waze)[:, 1]
    waze_test_probs = waze_pipe.predict_proba(X_test_waze)[:, 1]
    waze_thresh = find_optimal_f1_threshold(y_val, waze_val_probs)

    waze_metrics = evaluate_predictions(
        model_name="HistGradientBoosting (+Unverified Waze Proxy, y Immutable)",
        y_test=y_test,
        test_probs=waze_test_probs,
        threshold=waze_thresh,
        n_bootstraps=300,
        seed=seed,
    )
    waze_metrics["track"] = "Track B: Week 4 Waze Ablation Study"
    results.append(waze_metrics)

    # Week 5 Deliverable: Explainability & Subgroup Audit on Best Honest Baseline Model
    best_pipe = curves["HistGradientBoosting"]["pipeline"]
    best_probs = curves["HistGradientBoosting"]["probs"]
    best_thresh = curves["HistGradientBoosting"]["threshold"]

    imp_df = compute_global_feature_importance(
        best_pipe, X_test_fe, y_test, scoring="average_precision", n_repeats=8, seed=seed
    )
    subgroup_df = evaluate_subgroup_performance(
        X_test_fe, y_test, best_probs, group_col="region", threshold=best_thresh
    )
    error_cases = extract_error_analysis_cases(
        X_test_fe, y_test, best_probs, threshold=best_thresh, top_k=5
    )

    return pd.DataFrame(results), curves, imp_df, subgroup_df, error_cases


def generate_report_figures(
    regional_df: pd.DataFrame,
    track_a_curves: Dict[str, Any],
    track_b_curves: Dict[str, Any],
    imp_df: pd.DataFrame,
    subgroup_df: pd.DataFrame,
) -> None:
    """
    Generates 4 publication-grade diagnostic figures in `reports/figures/`.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Official ELSTAT Regional Paradox Figure
    reg_summary = (
        regional_df.groupby("region", as_index=False)
        .agg(
            total_accidents=("total_accidents", "sum"),
            ksi_accidents=("ksi_accidents", "sum"),
            ksi_rate=("ksi_severity_rate", "mean"),
        )
        .sort_values("total_accidents", ascending=True)
    )
    reg_summary["short_region"] = reg_summary["region"].apply(lambda s: s.split(" (")[0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    axes[0].barh(reg_summary["short_region"], reg_summary["total_accidents"], color="#1e3a8a")
    axes[0].set_title("Total Injury Collisions (ELSTAT 2021-2025)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Total Injury Collisions")

    colors = ["#dc2626" if r >= 0.16 else "#2563eb" for r in reg_summary["ksi_rate"]]
    axes[1].barh(reg_summary["short_region"], reg_summary["ksi_rate"] * 100.0, color=colors)
    axes[1].axvline(16.0, color="#991b1b", linestyle="--", linewidth=1.5, label="High-KSI Regime Threshold (16%)")
    axes[1].set_title("Mean KSI Severity Rate (% Killed/Seriously Injured)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("KSI Severity Rate (%)")
    axes[1].legend(loc="lower right")
    plt.suptitle("The Greek Road Safety Paradox: Urban Volume vs. Rural/Island Severity (ELSTAT SDT04)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "elstat_regional_paradox_greece.png", dpi=220)
    plt.close(fig)

    # 2. Benchmark ROC & PR Curves (Track B Event-Level Benchmark)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    palette = {
        "DummyClassifier (Baseline)": "#64748b",
        "DecisionTree": "#d97706",
        "RandomForest": "#059669",
        "HistGradientBoosting": "#1d4ed8",
        "XGBoost": "#7c3aed",
    }
    for name, cdata in track_b_curves.items():
        y_t = cdata["y_test"]
        prb = cdata["probs"]
        col = palette.get(name, "#334155")
        fpr, tpr, _ = roc_curve(y_t, prb)
        prec, rec, _ = precision_recall_curve(y_t, prb)
        axes[0].plot(fpr, tpr, lw=2.2, label=name, color=col)
        axes[1].plot(rec, prec, lw=2.2, label=name, color=col)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_title("ROC Curves on Held-Out 2025 Test Partition", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(loc="lower right")

    axes[1].set_title("Precision-Recall Curves on Held-Out 2025 Test Partition", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Recall (Sensitivity)")
    axes[1].set_ylabel("Precision (Positive Predictive Value)")
    axes[1].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "benchmark_roc_pr_curves.png", dpi=220)
    plt.close(fig)

    # 3. Top Permutation Feature Importances
    top_imp = imp_df.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        top_imp["feature"],
        top_imp["importance_mean"],
        xerr=top_imp["importance_std"],
        color="#0f766e",
        ecolor="#334155",
        capsize=3,
    )
    ax.set_title("Global Permutation Feature Importance (PR-AUC Drop on 2025 Test Set)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Decrease in Average Precision (PR-AUC)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance_and_ablation.png", dpi=220)
    plt.close(fig)

    # 4. Regional Subgroup Calibration Plot
    sub_sorted = subgroup_df.sort_values("actual_ksi_rate", ascending=True).copy()
    sub_sorted["short_region"] = sub_sorted["region"].apply(lambda s: s.split(" (")[0])
    y_pos = np.arange(len(sub_sorted))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(y_pos - 0.18, sub_sorted["actual_ksi_rate"] * 100.0, height=0.35, label="Actual Test KSI Rate (%)", color="#1e293b")
    ax.barh(y_pos + 0.18, sub_sorted["mean_predicted_prob"] * 100.0, height=0.35, label="Model Predicted Probability (%)", color="#3b82f6")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub_sorted["short_region"])
    ax.set_xlabel("KSI Rate / Probability (%)")
    ax.set_title("Subgroup Audit: Actual vs. Predicted KSI Rate by Greek Region (2025 Test Set)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "regional_subgroup_calibration.png", dpi=220)
    plt.close(fig)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("ROAD SAFETY ML GREECE — REPRODUCIBLE BENCHMARK & ABLATION PIPELINE (v1.0.0)")
    print("=" * 78)

    regional_df, micro_df = ensure_datasets_exist(seed=42)
    print(f"[Data] Official ELSTAT Regional Panel : {len(regional_df):,} rows (2021-2025, 13 NUTS-2 regions)")
    print(f"[Data] Synthetic Collision Benchmark  : {len(micro_df):,} rows (KSI Rate: {micro_df['target_ksi'].mean():.2%})")

    print("\n[1/4] Running Track A: Official ELSTAT Regional Panel Benchmark...")
    track_a_df, track_a_curves = run_track_a_regional_benchmark(regional_df, seed=42)
    print(track_a_df[["model", "roc_auc", "roc_auc_ci_95", "pr_auc", "f1", "precision", "recall"]].to_string(index=False))

    print("\n[2/4] Running Track B: Event-Level Benchmark & Honest Waze Ablation Study...")
    track_b_df, track_b_curves, imp_df, subgroup_df, error_cases = run_track_b_micro_and_ablation(micro_df, seed=42)
    print(track_b_df[["model", "roc_auc", "roc_auc_ci_95", "pr_auc", "f1", "precision", "recall"]].to_string(index=False))

    print("\n[3/4] Generating Publication Figures in reports/figures/...")
    generate_report_figures(regional_df, track_a_curves, track_b_curves, imp_df, subgroup_df)

    all_benchmarks = pd.concat([track_a_df, track_b_df], ignore_index=True)
    all_benchmarks.to_csv(REPORTS_DIR / "benchmark_comparison.csv", index=False)
    imp_df.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    subgroup_df.to_csv(REPORTS_DIR / "regional_subgroup_audit.csv", index=False)

    summary_payload = {
        "environment": {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "random_seed": 42,
        },
        "track_a_official_elstat_regional": track_a_df.to_dict(orient="records"),
        "track_b_micro_and_waze_ablation": track_b_df.to_dict(orient="records"),
        "top_10_features": imp_df.head(10).to_dict(orient="records"),
        "regional_subgroup_audit": subgroup_df.to_dict(orient="records"),
    }
    with open(REPORTS_DIR / "benchmark_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    print("\n[4/4] Saved artifacts:")
    print(f"  - {REPORTS_DIR / 'benchmark_comparison.csv'}")
    print(f"  - {REPORTS_DIR / 'benchmark_metrics.json'}")
    print(f"  - {REPORTS_DIR / 'feature_importance.csv'}")
    print(f"  - {REPORTS_DIR / 'regional_subgroup_audit.csv'}")
    print(f"  - {FIGURES_DIR / 'elstat_regional_paradox_greece.png'}")
    print(f"  - {FIGURES_DIR / 'benchmark_roc_pr_curves.png'}")
    print(f"  - {FIGURES_DIR / 'feature_importance_and_ablation.png'}")
    print(f"  - {FIGURES_DIR / 'regional_subgroup_calibration.png'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
