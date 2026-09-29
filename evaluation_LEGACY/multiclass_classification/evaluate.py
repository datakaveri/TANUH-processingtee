#!/usr/bin/env python3
"""
Bucket evaluator: multiclass_classification.

Scores a model's predictions against a dataset's ground truth. Owned by the
platform; one script serves every multi-class dataset (ordinal ones included —
QWK is reported for all, and is meaningful when class_names are ordered).

    python3 evaluate.py --predictions P --ground-truth G --spec S --results R

  predictions.csv   file,label,prob_0,...,prob_{C-1}
                    (label ∈ 0..C-1; probs in [0,1] summing to ~1)
  ground_truth.csv  file,label   (label ∈ 0..C-1)
  dataset_spec.json {"class_names": [...], ...}   (C >= 2 names, in label order)

Every ground-truth row must have exactly one prediction. Exit codes:
  0   results.json written
  12  predictions invalid (missing / duplicate / unknown file, bad label,
      bad or non-normalised probabilities) — the model provider's fault
  other non-zero — evaluator/platform error

Metric definitions follow the previous platform evaluator for this vertical
(evaluate_model_breastcancer.py) and the dataset provider's metrics.py:
sensitivity = macro recall, ppv = macro precision, specificity / npv = mean of
the one-vs-rest per-class values. A ratio whose denominator is 0 counts as 0.
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
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

EXIT_PREDICTIONS_INVALID = 12
PROB_SUM_TOLERANCE = 1e-3


class PredictionsInvalid(Exception):
    """The adaptor's output does not satisfy the bucket contract."""


def log(msg):
    print(f"[evaluator:multiclass] {msg}", flush=True)


def read_table(path, required):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = [h.strip() for h in (reader.fieldnames or [])]
        missing = [c for c in required if c not in header]
        if missing:
            raise ValueError(f"{Path(path).name}: missing column(s) {missing}; header={header}")
        return [{(k or "").strip(): (v or "").strip() for k, v in raw.items()} for raw in reader]


def parse_label(raw, num_classes):
    if not raw.isdigit() or int(raw) >= num_classes:
        return None
    return int(raw)


def load_ground_truth(path, num_classes):
    gt = {}
    for i, row in enumerate(read_table(path, ["file", "label"]), start=2):
        f = row["file"]
        if not f:
            raise ValueError(f"ground_truth.csv line {i}: empty file")
        if f in gt:
            raise ValueError(f"ground_truth.csv line {i}: duplicate file {f!r}")
        label = parse_label(row["label"], num_classes)
        if label is None:
            raise ValueError(f"ground_truth.csv line {i}: label must be 0..{num_classes - 1}, got {row['label']!r}")
        gt[f] = label
    if not gt:
        raise ValueError("ground_truth.csv has no rows")
    return gt


def load_predictions(path, gt, num_classes):
    """Return {file: (label, probs)}; raise PredictionsInvalid on any contract breach."""
    prob_cols = [f"prob_{i}" for i in range(num_classes)]
    try:
        rows = read_table(path, ["file", "label"] + prob_cols)
    except ValueError as e:
        raise PredictionsInvalid(str(e))
    preds = {}
    for i, row in enumerate(rows, start=2):
        f = row["file"]
        if f not in gt:
            raise PredictionsInvalid(f"predictions.csv line {i}: unknown file {f!r}")
        if f in preds:
            raise PredictionsInvalid(f"predictions.csv line {i}: duplicate file {f!r}")
        label = parse_label(row["label"], num_classes)
        if label is None:
            raise PredictionsInvalid(f"predictions.csv line {i}: label must be 0..{num_classes - 1}, got {row['label']!r}")
        try:
            probs = [float(row[c]) for c in prob_cols]
        except ValueError:
            raise PredictionsInvalid(f"predictions.csv line {i}: a probability is not a number")
        if any(not math.isfinite(p) or p < 0.0 or p > 1.0 for p in probs):
            raise PredictionsInvalid(f"predictions.csv line {i}: probabilities must be in [0,1]")
        if abs(sum(probs) - 1.0) > PROB_SUM_TOLERANCE:
            raise PredictionsInvalid(f"predictions.csv line {i}: probabilities sum to {sum(probs):.6f}, expected 1")
        preds[f] = (label, probs)
    missing = [f for f in gt if f not in preds]
    if missing:
        raise PredictionsInvalid(f"{len(missing)} ground-truth file(s) have no prediction, e.g. {missing[:3]}")
    return preds


def ratio(num, den):
    return float(num) / float(den) if den > 0 else 0.0


def finite_or_none(x):
    x = float(x)
    return x if math.isfinite(x) else None


def compute_metrics(y_true, y_pred, y_prob, class_names):
    c = len(class_names)
    labels = np.arange(c)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "macro_f2": float(fbeta_score(y_true, y_pred, beta=2, labels=labels, average="macro", zero_division=0)),
        "weighted_f2": float(fbeta_score(y_true, y_pred, beta=2, labels=labels, average="weighted", zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "ppv": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }
    try:
        m["qwk"] = finite_or_none(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except ValueError:
        m["qwk"] = None
    try:
        m["auc"] = finite_or_none(roc_auc_score(y_true, y_prob, labels=labels, multi_class="ovr", average="macro"))
    except ValueError:  # e.g. a class absent from the ground truth
        m["auc"] = None

    per_class, specs, npvs = {}, [], []
    total = int(cm.sum())
    for i in range(c):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum()) - tp
        fp = int(cm[:, i].sum()) - tp
        tn = total - tp - fn - fp
        spec, npv = ratio(tn, tn + fp), ratio(tn, tn + fn)
        specs.append(spec)
        npvs.append(npv)
        per_class[class_names[i]] = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn),
            "specificity": spec,
            "npv": npv,
        }
    m["specificity"] = float(np.mean(specs))
    m["npv"] = float(np.mean(npvs))
    m["confusion_matrix"] = cm.tolist()
    m["per_class"] = per_class
    return m


def main():
    ap = argparse.ArgumentParser(description="multiclass_classification bucket evaluator")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    class_names = spec.get("class_names") or []
    if len(class_names) < 2:
        raise ValueError(f"multiclass_classification needs >= 2 class_names, got {class_names}")
    num_classes = len(class_names)

    gt = load_ground_truth(args.ground_truth, num_classes)
    try:
        preds = load_predictions(args.predictions, gt, num_classes)
    except PredictionsInvalid as e:
        log(f"PREDICTIONS INVALID: {e}")
        return EXIT_PREDICTIONS_INVALID

    files = sorted(gt)
    y_true = np.array([gt[f] for f in files], dtype=np.int64)
    y_pred = np.array([preds[f][0] for f in files], dtype=np.int64)
    y_prob = np.array([preds[f][1] for f in files], dtype=np.float64)

    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
    result = {
        "status": "success",
        "task_type": "multiclass_classification",
        "num_samples": int(len(files)),
        "class_names": class_names,
        "metrics": metrics,
        "confusion_matrix": metrics["confusion_matrix"],
        "label_distribution": {class_names[c]: int((y_true == c).sum()) for c in range(num_classes)},
        "prediction_distribution": {class_names[c]: int((y_pred == c).sum()) for c in range(num_classes)},
    }
    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"scored {len(files)} samples: accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} qwk={metrics['qwk']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
