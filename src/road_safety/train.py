"""
Reproducible Training, Benchmarking & Feature Ablation Pipeline (`train.py`)
============================================================================
Single-command reproducible pipeline (`python run.py` or `python -m road_safety.train`)
operating exclusively on verified Hellenic Statistical Authority (ELSTAT SDT04) open data:
  1. Validates data schemas, chronological splits (2021-2023 Train, 2024 Val, 2025 Test),
     and anti-leakage rules (`validate_no_leakage`, `assert_target_immutable`).
  2. Benchmarks DummyClassifier, DecisionTree, RandomForest, HistGradientBoosting,
     and XGBoost on the Official ELSTAT Regional Monthly Panel (780 observations,
     13 Greek NUTS-2 regions, 2021-2025).
  3. Executes the Week 4 Empirical Feature Group Ablation Study (comparing Calendar-only,
     Regional, Structural Exposure, and Autoregressive Lag feature groups on the 2025 test set).
  4. Computes Week 5 permutation feature importances, subgroup audits across all 13 Greek
     regions, and error analysis cases (TP, FP, FN).
  5. Exports structured tables (`reports/benchmark_comparison.csv`, `reports/ablation_study.csv`,
     `reports/benchmark_metrics.json`) and publication-quality figures (`reports/figures/*.png`).
"""

import json
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
)
from road_safety.evaluate import evaluate_predictions, find_optimal_f1_threshold
from road_safety.explain import (
    compute_global_feature_importance,
    evaluate_subgroup_performance,
    extract_error_analysis_cases,
)
from road_safety.features import (
    ABLATION_FEATURE_GROUPS,
    REGIONAL_CATEGORICAL_FEATURES,
    REGIONAL_NUMERIC_FEATURES,
    build_regional_panel_features,
    extract_regional_X_y,
)
from road_safety.preprocessing import build_preprocessing_transformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def get_candidate_models(seed: int = 42, pos_weight: float = 1.0) -> Dict[str, Any]:
    """
    Returns the standard benchmark suite of classifiers initialized with fixed random seeds.
    """
    models: Dict[str, Any] = {
        "DummyClassifier (Baseline)": DummyClassifier(strategy="prior", random_state=seed),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=15, class_weight="balanced", random_state=seed
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=15,
            class_weight="balanced",
            random_state=seed,
        ),
    }
    try:
        from xgboost import XGBClassifier  # type: ignore

        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.04,
            max_depth=4,
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


def load_official_elstat_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the verified ELSTAT national annual dataset (2015-2024) and the official
    regional monthly panel (2021-2025, 13 Greek NUTS-2 regions).
    Zero synthetic data files are used.
    """
    nat_path = DATA_DIR / "elstat_national_annual_2015_2024.csv"
    reg_path = DATA_DIR / "elstat_regional_monthly_panel_2021_2025.csv"

    if not nat_path.exists():
        raise FileNotFoundError(f"Missing official ELSTAT national dataset at {nat_path}")
    if not reg_path.exists():
        raise FileNotFoundError(f"Missing official ELSTAT regional panel dataset at {reg_path}")

    nat_df = pd.read_csv(nat_path)
    reg_df = pd.read_csv(reg_path)
    reg_df = reg_df[reg_df["year"].between(2021, 2025)].copy()
    return nat_df, reg_df


def run_elstat_model_benchmark(
    fe_df: pd.DataFrame,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, pd.Series]:
    """
    Benchmarks all candidate models on the official ELSTAT regional panel using a
    strict chronological split: 2021-2023 Train, 2024 Validation, 2025 Test.
    """
    train_df, val_df, test_df = temporal_train_val_test_split(
        fe_df, year_col="year", train_end_year=2023, val_year=2024, test_year=2025
    )

    X_train, y_train = extract_regional_X_y(train_df)
    X_val, y_val = extract_regional_X_y(val_df)
    X_test, y_test = extract_regional_X_y(test_df)

    y_train_snapshot = y_train.to_numpy().copy()
    y_test_snapshot = y_test.to_numpy().copy()

    validate_binary_target(y_train, "ELSTAT y_train")
    validate_binary_target(y_val, "ELSTAT y_val")
    validate_binary_target(y_test, "ELSTAT y_test")

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

        assert_target_immutable(y_train_snapshot, y_train)
        assert_target_immutable(y_test_snapshot, y_test)

        val_probs = pipe.predict_proba(X_val)[:, 1]
        test_probs = pipe.predict_proba(X_test)[:, 1]

        opt_thresh = 0.50 if "Dummy" in name else find_optimal_f1_threshold(y_val, val_probs)
        metrics = evaluate_predictions(
            model_name=name,
            y_test=y_test,
            test_probs=test_probs,
            threshold=opt_thresh,
            n_bootstraps=400,
            seed=seed,
        )
        results.append(metrics)
        curves[name] = {
            "y_test": y_test.to_numpy(),
            "probs": test_probs,
            "pipeline": pipe,
            "threshold": opt_thresh,
        }

    return pd.DataFrame(results), curves, X_test, y_test


def run_empirical_feature_ablation(
    fe_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Week 4 Deliverable: Empirical Feature Group Ablation Study on Official ELSTAT Data.
    Evaluates the exact out-of-time contribution of each real feature group on the 2025 test set
    while guaranteeing 100% target immutability (`assert_target_immutable`).
    """
    train_df, val_df, test_df = temporal_train_val_test_split(
        fe_df, year_col="year", train_end_year=2023, val_year=2024, test_year=2025
    )

    ablation_rows: List[Dict[str, Any]] = []
    for group_label, (num_cols, cat_cols) in ABLATION_FEATURE_GROUPS.items():
        X_tr, y_tr = extract_regional_X_y(train_df, numeric_cols=num_cols, categorical_cols=cat_cols)
        X_va, y_va = extract_regional_X_y(val_df, numeric_cols=num_cols, categorical_cols=cat_cols)
        X_te, y_te = extract_regional_X_y(test_df, numeric_cols=num_cols, categorical_cols=cat_cols)

        y_te_orig = y_te.to_numpy().copy()

        prep = build_preprocessing_transformer(num_cols, cat_cols)
        clf = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=15,
            class_weight="balanced",
            random_state=seed,
        )
        pipe = Pipeline([("prep", prep), ("clf", clf)])
        pipe.fit(X_tr, y_tr)

        assert_target_immutable(y_te_orig, y_te)

        val_probs = pipe.predict_proba(X_va)[:, 1]
        test_probs = pipe.predict_proba(X_te)[:, 1]
        thresh = find_optimal_f1_threshold(y_va, val_probs)

        m = evaluate_predictions(
            model_name=group_label,
            y_test=y_te,
            test_probs=test_probs,
            threshold=thresh,
            n_bootstraps=400,
            seed=seed,
        )
        m["num_features"] = len(num_cols) + len(cat_cols)
        ablation_rows.append(m)

    return pd.DataFrame(ablation_rows)


def generate_report_figures(
    regional_df: pd.DataFrame,
    curves: Dict[str, Any],
    ablation_df: pd.DataFrame,
    imp_df: pd.DataFrame,
    subgroup_df: pd.DataFrame,
) -> None:
    """
    Generates 4 publication-grade diagnostic figures in `reports/figures/` strictly from ELSTAT data.
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

    colors = ["#dc2626" if r >= 0.21 else "#2563eb" for r in reg_summary["ksi_rate"]]
    axes[1].barh(reg_summary["short_region"], reg_summary["ksi_rate"] * 100.0, color=colors)
    axes[1].axvline(21.0, color="#991b1b", linestyle="--", linewidth=1.5, label="Severe KSI Regime Threshold (21%)")
    axes[1].set_title("Mean KSI Severity Rate (% Killed/Seriously Injured)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("KSI Severity Rate (%)")
    axes[1].legend(loc="lower right")
    plt.suptitle("The Greek Road Safety Paradox: Urban Volume vs. Rural/Island Severity (ELSTAT SDT04)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "elstat_regional_paradox_greece.png", dpi=220)
    plt.close(fig)

    # 2. Benchmark ROC & PR Curves on Held-Out 2025 ELSTAT Test Set
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    palette = {
        "DummyClassifier (Baseline)": "#64748b",
        "DecisionTree": "#d97706",
        "RandomForest": "#059669",
        "HistGradientBoosting": "#1d4ed8",
        "XGBoost": "#7c3aed",
    }
    for name, cdata in curves.items():
        y_t = cdata["y_test"]
        prb = cdata["probs"]
        col = palette.get(name, "#334155")
        fpr, tpr, _ = roc_curve(y_t, prb)
        prec, rec, _ = precision_recall_curve(y_t, prb)
        axes[0].plot(fpr, tpr, lw=2.2, label=name, color=col)
        axes[1].plot(rec, prec, lw=2.2, label=name, color=col)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_title("ROC Curves on Held-Out 2025 ELSTAT Test Set", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(loc="lower right")

    axes[1].set_title("Precision-Recall Curves on Held-Out 2025 ELSTAT Test Set", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Recall (Sensitivity)")
    axes[1].set_ylabel("Precision (Positive Predictive Value)")
    axes[1].legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "benchmark_roc_pr_curves.png", dpi=220)
    plt.close(fig)

    # 3. Permutation Importance & Empirical Feature Ablation
    top_imp = imp_df.head(10).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].barh(
        top_imp["feature"],
        top_imp["importance_mean"],
        xerr=top_imp["importance_std"],
        color="#0f766e",
        ecolor="#334155",
        capsize=3,
    )
    axes[0].set_title("Permutation Importance (PR-AUC Drop on 2025 Test Set)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Mean Decrease in Average Precision (PR-AUC)")

    abl_rev = ablation_df.iloc[::-1]
    axes[1].barh(abl_rev["model"], abl_rev["roc_auc"], color="#1d4ed8", alpha=0.85, label="ROC-AUC")
    axes[1].axvline(0.5, color="#dc2626", linestyle="--", lw=1.2, label="Random Baseline (0.50)")
    axes[1].set_xlim(0.4, 1.02)
    axes[1].set_title("Empirical Feature Group Ablation (2025 Test ROC-AUC)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Test Set ROC-AUC")
    axes[1].legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance_and_ablation.png", dpi=220)
    plt.close(fig)

    # 4. Regional Subgroup Calibration Plot (All 13 Greek NUTS-2 Regions)
    sub_sorted = subgroup_df.sort_values("actual_ksi_rate", ascending=True).copy()
    sub_sorted["short_region"] = sub_sorted["region"].apply(lambda s: s.split(" (")[0])
    y_pos = np.arange(len(sub_sorted))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(y_pos - 0.18, sub_sorted["actual_ksi_rate"] * 100.0, height=0.35, label="Actual 2025 High-KSI Regime Rate (%)", color="#1e293b")
    ax.barh(y_pos + 0.18, sub_sorted["mean_predicted_prob"] * 100.0, height=0.35, label="Model Predicted Probability (%)", color="#3b82f6")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub_sorted["short_region"])
    ax.set_xlabel("High-KSI Regime Rate / Predicted Probability (%)")
    ax.set_title("Subgroup Audit across All 13 Greek Regions (2025 Held-Out Test Set)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "regional_subgroup_calibration.png", dpi=220)
    plt.close(fig)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ROAD SAFETY ML GREECE — OFFICIAL ELSTAT EMPIRICAL PIPELINE (v1.1.0, ZERO SYNTHETIC)")
    print("=" * 80)

    nat_df, reg_df = load_official_elstat_datasets()
    fe_df = build_regional_panel_features(reg_df, high_ksi_threshold=0.21)

    print(f"[Data] Official ELSTAT National Series : {len(nat_df)} years (2015-2024)")
    print(f"[Data] Official ELSTAT Regional Panel  : {len(fe_df):,} rows (2021-2025, 13 NUTS-2 regions)")
    print(f"[Data] Severe KSI Regime Rate (>=21%)  : {fe_df['target_high_ksi'].mean():.2%}")

    print("\n[1/4] Running 5-Model Benchmark on Held-Out 2025 ELSTAT Test Set...")
    bench_df, curves, X_test, y_test = run_elstat_model_benchmark(fe_df, seed=42)
    print(
        bench_df[
            ["model", "roc_auc", "roc_auc_ci_95", "pr_auc", "f1", "precision", "recall", "brier_score"]
        ].to_string(index=False)
    )

    print("\n[2/4] Running Week 4 Empirical Feature Group Ablation Study...")
    ablation_df = run_empirical_feature_ablation(fe_df, seed=42)
    print(
        ablation_df[
            ["model", "num_features", "roc_auc", "roc_auc_ci_95", "pr_auc", "f1", "recall"]
        ].to_string(index=False)
    )

    print("\n[3/4] Running Week 5 Explainable AI & 13-Region Subgroup Audit...")
    best_model_name = "RandomForest"
    best_pipe = curves[best_model_name]["pipeline"]
    best_probs = curves[best_model_name]["probs"]
    best_thresh = curves[best_model_name]["threshold"]

    imp_df = compute_global_feature_importance(
        best_pipe, X_test, y_test, scoring="average_precision", n_repeats=15, seed=42
    )
    subgroup_df = evaluate_subgroup_performance(
        X_test, y_test, best_probs, group_col="region", threshold=best_thresh
    )
    error_cases = extract_error_analysis_cases(
        X_test, y_test, best_probs, threshold=best_thresh, top_k=5
    )

    print("\nTop 8 Features by Permutation Importance (2025 Test Set):")
    print(imp_df.head(8).to_string(index=False))

    print("\n[4/4] Generating Publication Figures & Saving Reports...")
    generate_report_figures(reg_df, curves, ablation_df, imp_df, subgroup_df)

    bench_df.to_csv(REPORTS_DIR / "benchmark_comparison.csv", index=False)
    ablation_df.to_csv(REPORTS_DIR / "ablation_study.csv", index=False)
    imp_df.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    subgroup_df.to_csv(REPORTS_DIR / "regional_subgroup_audit.csv", index=False)

    summary_payload = {
        "data_provenance": "100% Official ELSTAT SDT04 Aggregated Tables (Zero Synthetic Data)",
        "environment": {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "random_seed": 42,
        },
        "benchmark_models_2025_test": bench_df.to_dict(orient="records"),
        "feature_group_ablation_2025_test": ablation_df.to_dict(orient="records"),
        "permutation_feature_importance": imp_df.to_dict(orient="records"),
        "regional_subgroup_audit": subgroup_df.to_dict(orient="records"),
        "error_analysis_counts": {
            "true_positives_sample": len(error_cases["true_positives"]),
            "false_positives_sample": len(error_cases["false_positives"]),
            "false_negatives_sample": len(error_cases["false_negatives"]),
        },
    }
    with open(REPORTS_DIR / "benchmark_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    print("=" * 80)


if __name__ == "__main__":
    main()
