"""Top-level entry points: dataset name + raw labels/files in, metrics dict out."""

from __future__ import annotations

from typing import Optional, Sequence

from .binary import compute_binary_metrics
from .datasets import coerce_labels, get_dataset_config
from .io_utils import load_paired_csvs, load_single_csv
from .multiclass import compute_multiclass_metrics


def evaluate(
    dataset_name: str,
    y_true_raw: Sequence,
    y_pred_raw: Sequence,
    y_score: Optional[Sequence] = None,
) -> dict:
    """Compute metrics for one dataset's problem statement from raw label values.

    Raw values are coerced through the dataset's label_map first (e.g. the
    oral_cancer dataset maps "Suspicious"/"Non-Suspicious" onto 1/0), so
    callers can pass whatever label spelling their CSV/inference output uses.
    """
    cfg = get_dataset_config(dataset_name)
    y_true = coerce_labels(y_true_raw, cfg)
    y_pred = coerce_labels(y_pred_raw, cfg)

    if cfg.problem_type == "binary":
        metrics = compute_binary_metrics(y_true, y_pred, y_score)
    else:
        metrics = compute_multiclass_metrics(y_true, y_pred, y_score, labels=cfg.class_values)

    metrics["dataset"] = cfg.name
    metrics["problem_type"] = cfg.problem_type
    return metrics


def evaluate_from_files(
    dataset_name: str,
    *,
    input_csv: Optional[str] = None,
    ground_truth_csv: Optional[str] = None,
    predictions_csv: Optional[str] = None,
    id_col: Optional[str] = None,
    gt_col: Optional[str] = None,
    pred_col: Optional[str] = None,
    score_col: Optional[str] = None,
) -> dict:
    """Load model outputs + ground truth from CSV and compute metrics.

    Either pass `input_csv` (a single file with both a ground-truth and a
    predictions column, per notes.md), or pass `ground_truth_csv` +
    `predictions_csv` (two files joined by an id column — the shape you get
    when predictions come out of a separate model-execution step).
    """
    if input_csv is not None:
        if ground_truth_csv or predictions_csv:
            raise ValueError("Pass either input_csv, or ground_truth_csv+predictions_csv — not both")
        _ids, y_true_raw, y_pred_raw, y_score = load_single_csv(
            input_csv, id_col=id_col, gt_col=gt_col, pred_col=pred_col, score_col=score_col
        )
    elif ground_truth_csv is not None and predictions_csv is not None:
        _ids, y_true_raw, y_pred_raw, y_score = load_paired_csvs(
            ground_truth_csv, predictions_csv, id_col=id_col, gt_col=gt_col, pred_col=pred_col, score_col=score_col
        )
    else:
        raise ValueError("Must pass either input_csv or both ground_truth_csv and predictions_csv")

    return evaluate(dataset_name, y_true_raw, y_pred_raw, y_score)
