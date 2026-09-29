"""Metrics for binary classification problem statements (e.g. oral cancer screening).

Output keys are deliberately named to match what
TANUH-processingtee/enclave_manager_new.py:_leaderboard_metrics() reads for the
oral_cancer vertical (dataset_id == 2): sensitivity, specificity, accuracy,
ppv, npv, f2 — note "f2", not "f2_score".
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

from .common import f_beta, safe_div


def compute_binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Optional[Sequence[float]] = None,
) -> dict:
    """Compute binary classification metrics from 0/1-coded labels.

    Args:
        y_true: ground truth labels, already coerced to 0 (negative) / 1 (positive).
        y_pred: predicted labels, same coercion as y_true.
        y_score: optional predicted probability/score for the positive class,
            same length as y_true/y_pred. Enables AUC.

    Returns:
        A flat dict of metrics. All ratio metrics default to 0.0 when their
        denominator is 0 (e.g. no actual positives in the sample).
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must be the same length "
            f"(got {y_true_arr.shape[0]} vs {y_pred_arr.shape[0]})"
        )
    if not set(np.unique(y_true_arr)) <= {0, 1} or not set(np.unique(y_pred_arr)) <= {0, 1}:
        raise ValueError("compute_binary_metrics expects labels coerced to {0, 1}")

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    sensitivity = safe_div(tp, tp + fn)  # a.k.a. recall / TPR
    specificity = safe_div(tn, tn + fp)  # a.k.a. TNR
    ppv = safe_div(tp, tp + fp)  # a.k.a. precision
    npv = safe_div(tn, tn + fn)
    f1 = f_beta(ppv, sensitivity, beta=1.0)
    f2 = f_beta(ppv, sensitivity, beta=2.0)

    metrics = {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "f2": f2,
        "fpr": safe_div(fp, fp + tn),
        "fnr": safe_div(fn, fn + tp),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "num_samples": int(y_true_arr.shape[0]),
    }

    if y_score is not None:
        try:
            metrics["auc"] = float(roc_auc_score(y_true_arr, np.asarray(y_score, dtype=float)))
        except ValueError:
            # Only one class present in y_true, or scores are degenerate.
            metrics["auc"] = None

    return metrics
