#!/usr/bin/env python3
"""
Bucket evaluator: binary_classification.

Scores a model's predictions against a dataset's ground truth. Owned by the
platform; one script serves every binary-classification dataset. It never
touches the model or the raw data — only two CSV tables.

    python3 evaluate.py --predictions P --ground-truth G --spec S --results R

  predictions.csv   file,label,score   (label ∈ {0,1}; score = P(class 1) in [0,1])
  ground_truth.csv  file,label         (label ∈ {0,1})
  dataset_spec.json {"class_names": [...], ...}   (exactly 2 class names)

Every ground-truth row must have exactly one prediction, so every model is
scored on the same samples. Exit codes:
  0   results.json written
  12  predictions invalid (missing / duplicate / unknown file, bad label or
      score) — the model provider's fault
  other non-zero — evaluator/platform error

Metric definitions follow the previous platform evaluator for this vertical
(evaluate_model_OCS.py compute_metrics), plus fnr/fpr. Positive class = 1.
A ratio whose denominator is 0 is reported as 0.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    roc_auc_score,
)

EXIT_PREDICTIONS_INVALID = 12


class PredictionsInvalid(Exception):
    """The adaptor's output does not satisfy the bucket contract."""


def log(msg):
    print(f"[evaluator:binary] {msg}", flush=True)


def read_table(path, required):
    """Read a CSV into a list of dicts, checking the header has `required`."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = [h.strip() for h in (reader.fieldnames or [])]
        missing = [c for c in required if c not in header]
        if missing:
            raise ValueError(f"{Path(path).name}: missing column(s) {missing}; header={header}")
        rows = []
        for raw in reader:
            rows.append({(k or "").strip(): (v or "").strip() for k, v in raw.items()})
        return rows


def load_ground_truth(path):
    gt = {}
    for i, row in enumerate(read_table(path, ["file", "label"]), start=2):
        f = row["file"]
        if not f:
            raise ValueError(f"ground_truth.csv line {i}: empty file")
        if f in gt:
            raise ValueError(f"ground_truth.csv line {i}: duplicate file {f!r}")
        if row["label"] not in ("0", "1"):
            raise ValueError(f"ground_truth.csv line {i}: label must be 0 or 1, got {row['label']!r}")
        gt[f] = int(row["label"])
    if not gt:
        raise ValueError("ground_truth.csv has no rows")
    return gt


def load_predictions(path, gt):
    """Return {file: (label, score)}; raise PredictionsInvalid on any contract breach."""
    try:
        rows = read_table(path, ["file", "label", "score"])
    except ValueError as e:
        raise PredictionsInvalid(str(e))
    preds = {}
    for i, row in enumerate(rows, start=2):
        f = row["file"]
        if f not in gt:
            raise PredictionsInvalid(f"predictions.csv line {i}: unknown file {f!r}")
        if f in preds:
            raise PredictionsInvalid(f"predictions.csv line {i}: duplicate file {f!r}")
        if row["label"] not in ("0", "1"):
            raise PredictionsInvalid(f"predictions.csv line {i}: label must be 0 or 1, got {row['label']!r}")
        try:
            score = float(row["score"])
        except ValueError:
            raise PredictionsInvalid(f"predictions.csv line {i}: score is not a number: {row['score']!r}")
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise PredictionsInvalid(f"predictions.csv line {i}: score must be in [0,1], got {score}")
        preds[f] = (int(row["label"]), score)
    missing = [f for f in gt if f not in preds]
    if missing:
        raise PredictionsInvalid(f"{len(missing)} ground-truth file(s) have no prediction, e.g. {missing[:3]}")
    return preds


def ratio(num, den):
    return float(num) / float(den) if den > 0 else 0.0


def finite_or_none(x):
    x = float(x)
    return x if math.isfinite(x) else None


def compute_metrics(y_true, y_pred, y_score):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    try:
        auc = finite_or_none(roc_auc_score(y_true, y_score))
    except ValueError:  # only one class present in the ground truth
        auc = None
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "ppv": ratio(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
        "fnr": ratio(fn, fn + tp),
        "fpr": ratio(fp, fp + tn),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "auc": auc,
        "confusion_matrix": cm.tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description="binary_classification bucket evaluator")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    class_names = spec.get("class_names") or []
    if len(class_names) != 2:
        raise ValueError(f"binary_classification needs exactly 2 class_names, got {class_names}")

    gt = load_ground_truth(args.ground_truth)
    try:
        preds = load_predictions(args.predictions, gt)
    except PredictionsInvalid as e:
        log(f"PREDICTIONS INVALID: {e}")
        return EXIT_PREDICTIONS_INVALID

    files = sorted(gt)  # fixed order: scores never depend on the adaptor's row order
    y_true = np.array([gt[f] for f in files], dtype=np.int64)
    y_pred = np.array([preds[f][0] for f in files], dtype=np.int64)
    y_score = np.array([preds[f][1] for f in files], dtype=np.float64)

    metrics = compute_metrics(y_true, y_pred, y_score)
    result = {
        "status": "success",
        "task_type": "binary_classification",
        "num_samples": int(len(files)),
        "class_names": class_names,
        "metrics": metrics,
        "confusion_matrix": metrics["confusion_matrix"],
        "label_distribution": {class_names[c]: int((y_true == c).sum()) for c in (0, 1)},
        "prediction_distribution": {class_names[c]: int((y_pred == c).sum()) for c in (0, 1)},
    }
    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"scored {len(files)} samples: " + ", ".join(
        f"{k}={metrics[k]:.4f}" for k in ("accuracy", "sensitivity", "specificity", "ppv", "npv", "f2")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
