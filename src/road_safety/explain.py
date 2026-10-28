"""
Explainable AI & Subgroup Error Analysis (`explain.py`)
=======================================================
Implements Week 5 Deliverables:
  1. Global feature attribution via Permutation Importance & Tree SHAP (when installed).
  2. Subgroup performance & calibration audit across Greek NUTS-2 Regions and Urban vs. Rural roads.
  3. Concrete False Positive (FP) and False Negative (FN) case inspection.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score


def compute_global_feature_importance(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    scoring: str = "average_precision",
    n_repeats: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Computes model-agnostic permutation feature importance on the held-out test set
    with respect to PR-AUC (average_precision) or ROC-AUC.
    """
    perm = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=1,
    )
    df_imp = pd.DataFrame(
        {
            "feature": list(X_test.columns),
            "importance_mean": np.round(perm.importances_mean, 5),
            "importance_std": np.round(perm.importances_std, 5),
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df_imp


def compute_shap_summary_if_available(
    pipeline,
    X_sample: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """
    Computes exact TreeExplainer SHAP mean(|SHAP|) values if the `shap` library is installed,
    returning None gracefully otherwise.
    """
    try:
        import shap  # type: ignore

        prep = pipeline.named_steps.get("prep")
        clf = pipeline.named_steps.get("clf")
        if prep is None or clf is None:
            return None

        # Transform features through pipeline steps preceding 'clf'
        X_trans = X_sample.copy()
        for step_name, step_transformer in pipeline.steps[:-1]:
            X_trans = step_transformer.transform(X_trans)

        feature_names = prep.get_feature_names_out()
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_trans)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        return (
            pd.DataFrame({"transformed_feature": feature_names, "mean_abs_shap": mean_abs_shap})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
    except Exception:
        return None


def evaluate_subgroup_performance(
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    test_probs: np.ndarray,
    group_col: str = "region",
    threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Audits model performance across geographic or road-type subgroups (e.g., Greek NUTS-2 regions)
    to detect regional underperformance or calibration drift.
    """
    df_eval = X_test[[group_col]].copy()
    df_eval["y_true"] = np.asarray(y_test, dtype=int)
    df_eval["y_prob"] = np.asarray(test_probs, dtype=float)
    df_eval["y_pred"] = (df_eval["y_prob"] >= threshold).astype(int)

    rows: List[Dict[str, Any]] = []
    for grp_val, sub in df_eval.groupby(group_col):
        n = len(sub)
        pos_rate = float(sub["y_true"].mean())
        mean_prob = float(sub["y_prob"].mean())
        if len(np.unique(sub["y_true"])) >= 2:
            roc = round(float(roc_auc_score(sub["y_true"], sub["y_prob"])), 4)
            pr = round(float(average_precision_score(sub["y_true"], sub["y_prob"])), 4)
        else:
            roc = np.nan
            pr = np.nan
        rec = round(float(recall_score(sub["y_true"], sub["y_pred"], zero_division=0)), 4)
        f1 = round(float(f1_score(sub["y_true"], sub["y_pred"], zero_division=0)), 4)

        rows.append(
            {
                group_col: grp_val,
                "samples": n,
                "actual_ksi_rate": round(pos_rate, 4),
                "mean_predicted_prob": round(mean_prob, 4),
                "calibration_gap": round(mean_prob - pos_rate, 4),
                "roc_auc": roc,
                "pr_auc": pr,
                "recall": rec,
                "f1": f1,
            }
        )

    return pd.DataFrame(rows).sort_values("samples", ascending=False).reset_index(drop=True)


def extract_error_analysis_cases(
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    test_probs: np.ndarray,
    threshold: float = 0.50,
    top_k: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Extracts representative True Positives (TP), False Positives (FP), and False Negatives (FN)
    to understand physical failure modes of the classifier.
    """
    df = X_test.copy()
    df["actual_ksi"] = np.asarray(y_test, dtype=int)
    df["predicted_prob"] = np.round(np.asarray(test_probs, dtype=float), 4)
    df["predicted_label"] = (df["predicted_prob"] >= threshold).astype(int)

    tp_df = (
        df[(df["actual_ksi"] == 1) & (df["predicted_label"] == 1)]
        .sort_values("predicted_prob", ascending=False)
        .head(top_k)
    )
    fp_df = (
        df[(df["actual_ksi"] == 0) & (df["predicted_label"] == 1)]
        .sort_values("predicted_prob", ascending=False)
        .head(top_k)
    )
    fn_df = (
        df[(df["actual_ksi"] == 1) & (df["predicted_label"] == 0)]
        .sort_values("predicted_prob", ascending=True)
        .head(top_k)
    )

    return {
        "true_positives": tp_df,
        "false_positives": fp_df,
        "false_negatives": fn_df,
    }
