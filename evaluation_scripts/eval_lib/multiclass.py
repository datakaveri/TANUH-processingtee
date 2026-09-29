"""Metrics for multiclass / ordinal classification problem statements
(e.g. breast tissue density grading).

Output keys are deliberately named to match what
TANUH-processingtee/enclave_manager_new.py:_leaderboard_metrics() reads for the
breast_cancer vertical (dataset_id == 1): accuracy, macro_f2, weighted_f2,
macro_f1, macro_recall, qwk, macro_specificity, macro_npv, macro_ppv,
confusion_matrix, auc, and per-class TP/FN/precision/recall (used to derive
weighted_f2 when it isn't computed directly).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score

from .common import f_beta, safe_div, unique_sorted_labels


def compute_multiclass_metrics(
    y_true: Sequence,
    y_pred: Sequence,
    y_score: Optional[Sequence[Sequence[float]]] = None,
    labels: Optional[Sequence] = None,
) -> dict:
    """Compute multiclass/ordinal classification metrics.

    Args:
        y_true: ground truth class labels (any hashable, ordered values —
            e.g. BI-RADS density 1..4, or a 1..5 rating scale).
        y_pred: predicted class labels, same label space as y_true.
        y_score: optional per-class probability matrix, shape
            (n_samples, len(labels)), with columns in the same order as
            `labels`. Enables macro one-vs-rest AUC.
        labels: the ordered set of valid class values. Required for QWK to be
            meaningful (it relies on numeric distance between labels) and for
            the confusion matrix to include classes that never appear in this
            sample. Inferred from y_true/y_pred if omitted.

    Returns:
        A flat dict of aggregate metrics plus a `per_class` breakdown keyed by
        stringified label.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must be the same length "
            f"(got {y_true_arr.shape[0]} vs {y_pred_arr.shape[0]})"
        )

    label_list = list(labels) if labels is not None else unique_sorted_labels(y_true_arr.tolist(), y_pred_arr.tolist())
    if len(label_list) < 2:
        raise ValueError(f"Need at least 2 distinct classes, got labels={label_list}")

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=label_list)
    total = int(cm.sum())

    per_class = {}
    precisions, recalls, specificities, npvs, f1s, f2s, supports = [], [], [], [], [], [], []
    for idx, label in enumerate(label_list):
        tp = int(cm[idx, idx])
        fn = int(cm[idx, :].sum() - tp)
        fp = int(cm[:, idx].sum() - tp)
        tn = int(total - tp - fn - fp)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        specificity = safe_div(tn, tn + fp)
        npv = safe_div(tn, tn + fn)
        f1 = f_beta(precision, recall, beta=1.0)
        f2 = f_beta(precision, recall, beta=2.0)
        support = tp + fn

        per_class[str(label)] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "npv": npv,
            "f1": f1,
            "f2": f2,
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        specificities.append(specificity)
        npvs.append(npv)
        f1s.append(f1)
        f2s.append(f2)
        supports.append(support)

    def macro(values: list) -> float:
        return float(np.mean(values)) if values else 0.0

    def weighted(values: list, weights: list) -> float:
        total_weight = sum(weights)
        return float(sum(v * w for v, w in zip(values, weights)) / total_weight) if total_weight else 0.0

    accuracy = safe_div(int(np.trace(cm)), total)

    metrics = {
        "accuracy": accuracy,
        "macro_ppv": macro(precisions),
        "macro_recall": macro(recalls),
        "macro_specificity": macro(specificities),
        "macro_npv": macro(npvs),
        "macro_f1": macro(f1s),
        "macro_f2": macro(f2s),
        "weighted_f1": weighted(f1s, supports),
        "weighted_f2": weighted(f2s, supports),
        "qwk": float(cohen_kappa_score(y_true_arr, y_pred_arr, weights="quadratic", labels=label_list)),
        "confusion_matrix": cm.tolist(),
        "labels": list(label_list),
        "per_class": per_class,
        "num_classes": len(label_list),
        "num_samples": total,
    }

    if y_score is not None:
        try:
            y_score_arr = np.asarray(y_score, dtype=float)
            metrics["auc"] = float(
                roc_auc_score(y_true_arr, y_score_arr, multi_class="ovr", average="macro", labels=label_list)
            )
        except ValueError:
            # e.g. a class missing from y_true, or score columns don't line up with labels.
            metrics["auc"] = None

    return metrics
