"""
Model Evaluation & Uncertainty Quantification (`evaluate.py`)
=============================================================
Computes threshold-free ranking metrics (ROC-AUC, PR-AUC, Brier Score),
bootstrap 95% confidence intervals, validation-calibrated optimal thresholds,
and confusion matrix breakdowns on identical held-out test partitions.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_optimal_f1_threshold(
    y_val: np.ndarray | pd.Series,
    val_probs: np.ndarray,
) -> float:
    """
    Finds the probability threshold on the validation set that maximizes F1-score.
    Never uses test-set labels to select the decision threshold.
    """
    y_arr = np.asarray(y_val, dtype=int)
    probs = np.asarray(val_probs, dtype=float)
    precisions, recalls, thresholds = precision_recall_curve(y_arr, probs)
    if len(thresholds) == 0:
        return 0.50
    f1_scores = (2.0 * precisions[:-1] * recalls[:-1]) / np.maximum(
        precisions[:-1] + recalls[:-1], 1e-9
    )
    best_idx = int(np.argmax(f1_scores))
    return float(np.clip(thresholds[best_idx], 0.05, 0.95))


def bootstrap_metric_ci(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    metric_fn=roc_auc_score,
    n_bootstraps: int = 300,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Computes a 95% non-parametric bootstrap confidence interval for a ranking metric.
    """
    rng = np.random.RandomState(seed)
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(y_prob, dtype=float)
    n = len(y_arr)
    scores: List[float] = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_arr[idx])) < 2:
            continue
        scores.append(float(metric_fn(y_arr[idx], p_arr[idx])))

    if not scores:
        val = float(metric_fn(y_arr, p_arr))
        return val, val

    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def evaluate_predictions(
    model_name: str,
    y_test: np.ndarray | pd.Series,
    test_probs: np.ndarray,
    threshold: float = 0.50,
    n_bootstraps: int = 300,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates predicted probabilities on the held-out test set and returns a complete
    dictionary of discrimination, calibration, and classification metrics.
    """
    y_arr = np.asarray(y_test, dtype=int)
    p_arr = np.asarray(test_probs, dtype=float)

    if len(y_arr) != len(p_arr):
        raise ValueError(f"Length mismatch: y_test ({len(y_arr)}) vs test_probs ({len(p_arr)})")

    roc_auc = float(roc_auc_score(y_arr, p_arr))
    pr_auc = float(average_precision_score(y_arr, p_arr))
    brier = float(brier_score_loss(y_arr, p_arr))

    roc_ci_low, roc_ci_high = bootstrap_metric_ci(
        y_arr, p_arr, metric_fn=roc_auc_score, n_bootstraps=n_bootstraps, seed=seed
    )
    pr_ci_low, pr_ci_high = bootstrap_metric_ci(
        y_arr, p_arr, metric_fn=average_precision_score, n_bootstraps=n_bootstraps, seed=seed
    )

    preds = (p_arr >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_arr, preds, labels=[0, 1]).ravel()

    precision = float(precision_score(y_arr, preds, zero_division=0))
    recall = float(recall_score(y_arr, preds, zero_division=0))
    f1 = float(f1_score(y_arr, preds, zero_division=0))
    specificity = float(tn / max(1, (tn + fp)))

    return {
        "model": model_name,
        "test_samples": int(len(y_arr)),
        "positive_rate": round(float(y_arr.mean()), 4),
        "roc_auc": round(roc_auc, 4),
        "roc_auc_ci_95": f"[{roc_ci_low:.4f}, {roc_ci_high:.4f}]",
        "pr_auc": round(pr_auc, 4),
        "pr_auc_ci_95": f"[{pr_ci_low:.4f}, {pr_ci_high:.4f}]",
        "brier_score": round(brier, 4),
        "threshold": round(float(threshold), 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
