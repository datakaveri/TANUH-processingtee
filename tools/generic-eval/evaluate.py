#!/usr/bin/env python3
"""
Generic TANUH evaluation engine.
================================

This is the PLATFORM-OWNED evaluator. It is NOT written per-dataset and it is
NOT supplied by a data provider. One engine scores every dataset, selecting its
behaviour from a small declaration the dataset carries (task_type, modality,
class_names) and a metric REGISTRY that wraps the standard scikit-learn metrics.

Adding a dataset needs NO code. Adding a new metric or task type = one small,
reviewed addition to the registry below (see METRIC REGISTRY).

Called by the Processing TEE with the frozen contract:

    python3 evaluate.py \
        --model         /path/to/model.onnx \
        --dataset       /path/to/dataset_dir \
        --results       /path/to/results.json \
        [--preprocessing /path/to/preprocessing.py]

Self-test (no ONNX / no data — exercises the metric registry on synthetic
predictions AND the DATA-zip file-resolver on temporary zips):

    python3 evaluate.py --selftest

THE DATASET DIR (what the TEE lays out per job)
-----------------------------------------------
The dataset UUID folder holds exactly TWO encrypted objects: the DATA (always a
single ZIP) and the GROUND TRUTH (a labels-only table). The TEE decrypts both and
also fetches the DECLARATION from the catalogue (cat/item?id=<uuid>). It lays out:

  dataset_spec.json        the declaration (from the catalogue; NOT in the folder)
  ground_truth.(json|csv)  the answers only: file -> label  (two columns)
  <something>.zip          the DATA: one zip of the raw samples (any layout)

dataset_spec.json — the declaration the engine needs to run:
  {
    "task_type":   "binary_classification" | "multiclass_classification"
                 | "ordinal_classification" | "multilabel_classification"
                 | "regression",
    "modality":    "image" | "dicom" | "tabular",        # -> which decoder
    "num_classes": 4,                                     # classification only
    "class_names": ["A","B","C","D"],                     # classification only
    "input":       { "size": 224, "normalize": "imagenet" },  # optional hints
    "data_file":   "features.csv"                          # tabular only (optional)
  }

ground_truth — labels ONLY, two columns, NO declaration:
  CSV : header "file,label", one row per sample.
  JSON: [ {"file":"x1.dcm","label":2}, ... ]  or  { "x1.dcm": 2, ... }
  tabular: a "label"-only column / {"labels":[...]} aligned to the feature rows.
  label = int in [0,num_classes-1] (single-label), list (multilabel), number
  (regression).

DATA zip: the internal layout does NOT matter. The engine extracts it (zip-slip
safe, junk skipped), strips a single wrapper folder if present, and resolves each
ground-truth "file" by exact path then unique basename — failing closed on a
missing file or an ambiguous duplicate basename. Unreferenced files are ignored.

WHO OWNS WHAT
  * decode (file -> pixels)        -> PLATFORM (one decoder per modality)
  * preprocess (pixels -> tensor)  -> MODEL provider (baked into ONNX, or the
                                      optional --preprocessing hook); the built-in
                                      default here is a plain, documented baseline
  * read output + metrics          -> PLATFORM, chosen by task_type
  * raw data (zip) + ground-truth labels -> DATA provider (no code)
  * declaration + metric list      -> catalogue (CAT API); the engine never calls
                                      it. WHICH metrics count is enforced by the
                                      leaderboard; this engine computes the FULL
                                      task-type suite.

Exit codes (contract shared with the Go pipeline / eval.go):
  10 = user preprocessing.py failed to load/run
  11 = CUDA/GPU environment failure
  other non-zero = engine error
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    fbeta_score,
    hamming_loss,
    log_loss,
    matthews_corrcoef,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

# onnx / onnxruntime / cv2 / pydicom are imported LAZILY (inside the functions
# that need them) so the metric registry can be imported and self-tested with
# only numpy + scikit-learn present.

DEFAULT_BATCH = 8
DEFAULT_INPUT_SIZE = 224
BINARY_THRESHOLD = 0.5
MULTILABEL_THRESHOLD = 0.5

SINGLE_LABEL = {"binary_classification", "multiclass_classification", "ordinal_classification"}
MULTILABEL = {"multilabel_classification"}
REGRESSION = {"regression"}
CLASSIFICATION = SINGLE_LABEL | MULTILABEL
ALL_TASKS = SINGLE_LABEL | MULTILABEL | REGRESSION
FILE_MODALITIES = {"image", "dicom"}


def log(msg: str) -> None:
    print(f"[generic_eval] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ════════════════════════════════════════════════════════════════════════════
# METRIC REGISTRY
# A metric = stable key + metadata + a tiny function over an evaluation context.
# The function wraps scikit-learn; no metric math is reimplemented here.
# To add a metric: append one reg(...) line. To add a task type: add a scorer
# family (a context builder) + register metrics against it.
# ════════════════════════════════════════════════════════════════════════════

REGISTRY = {}  # key -> SimpleNamespace(key,label,task_types,higher_is_better,range,fn)


def reg(key, label, task_types, higher_is_better, fn, value_range=(0.0, 1.0)):
    REGISTRY[key] = SimpleNamespace(
        key=key, label=label, task_types=set(task_types),
        higher_is_better=higher_is_better, range=value_range, fn=fn,
    )


def score(ctx, task_type) -> dict:
    """Run every metric registered for task_type over ctx. Failures -> None."""
    out = {}
    for key, spec in REGISTRY.items():
        if task_type not in spec.task_types:
            continue
        try:
            v = spec.fn(ctx)
        except Exception as exc:  # a metric that can't be computed on this data
            log(f"  metric {key} skipped: {exc}")
            v = None
        if isinstance(v, (np.floating, np.integer)):
            v = float(v)
        elif isinstance(v, float):
            v = round(v, 6)
        out[key] = v
    return out


# ── single-label classification metrics (binary / multiclass / ordinal) ──────

def _labels(c):
    return list(range(c.num_classes))


def _auc(c):
    if c.num_classes == 2 and c.prob_pos is not None:
        return roc_auc_score(c.y_true, c.prob_pos)
    return roc_auc_score(c.y_true, c.prob, multi_class="ovr", average="macro", labels=_labels(c))


def _auprc(c):
    if c.num_classes == 2 and c.prob_pos is not None:
        return average_precision_score(c.y_true, c.prob_pos)
    Y = label_binarize(c.y_true, classes=_labels(c))
    return average_precision_score(Y, c.prob, average="macro")


reg("accuracy", "Accuracy", SINGLE_LABEL, True,
    lambda c: accuracy_score(c.y_true, c.y_pred))
reg("balanced_accuracy", "Balanced Accuracy", SINGLE_LABEL, True,
    lambda c: balanced_accuracy_score(c.y_true, c.y_pred))
reg("macro_precision", "Macro Precision", SINGLE_LABEL, True,
    lambda c: precision_score(c.y_true, c.y_pred, average="macro", zero_division=0))
reg("macro_recall", "Macro Recall (Sensitivity)", SINGLE_LABEL, True,
    lambda c: recall_score(c.y_true, c.y_pred, average="macro", zero_division=0))
reg("macro_f1", "Macro F1", SINGLE_LABEL, True,
    lambda c: f1_score(c.y_true, c.y_pred, average="macro", zero_division=0))
reg("macro_f2", "Macro F2", SINGLE_LABEL, True,
    lambda c: fbeta_score(c.y_true, c.y_pred, beta=2, average="macro", zero_division=0))
reg("weighted_precision", "Weighted Precision", SINGLE_LABEL, True,
    lambda c: precision_score(c.y_true, c.y_pred, average="weighted", zero_division=0))
reg("weighted_recall", "Weighted Recall", SINGLE_LABEL, True,
    lambda c: recall_score(c.y_true, c.y_pred, average="weighted", zero_division=0))
reg("weighted_f1", "Weighted F1", SINGLE_LABEL, True,
    lambda c: f1_score(c.y_true, c.y_pred, average="weighted", zero_division=0))
reg("weighted_f2", "Weighted F2", SINGLE_LABEL, True,
    lambda c: fbeta_score(c.y_true, c.y_pred, beta=2, average="weighted", zero_division=0))
reg("micro_f1", "Micro F1", SINGLE_LABEL, True,
    lambda c: f1_score(c.y_true, c.y_pred, average="micro", zero_division=0))
reg("mcc", "Matthews Correlation Coefficient", SINGLE_LABEL, True,
    lambda c: matthews_corrcoef(c.y_true, c.y_pred), value_range=(-1.0, 1.0))
reg("cohen_kappa", "Cohen's Kappa", SINGLE_LABEL, True,
    lambda c: cohen_kappa_score(c.y_true, c.y_pred), value_range=(-1.0, 1.0))
reg("qwk", "Quadratic Weighted Kappa", {"ordinal_classification"}, True,
    lambda c: cohen_kappa_score(c.y_true, c.y_pred, weights="quadratic"), value_range=(-1.0, 1.0))
reg("macro_specificity", "Macro Specificity", SINGLE_LABEL, True,
    lambda c: float(np.mean(c.per_class_spec)))
reg("macro_npv", "Macro NPV", SINGLE_LABEL, True,
    lambda c: float(np.mean(c.per_class_npv)))
reg("macro_ppv", "Macro PPV", SINGLE_LABEL, True,
    lambda c: precision_score(c.y_true, c.y_pred, average="macro", zero_division=0))
reg("auc", "AUC-ROC", SINGLE_LABEL, True, _auc)
reg("auprc", "AUC-PR (Average Precision)", SINGLE_LABEL, True, _auprc)
reg("log_loss", "Log Loss (Cross-Entropy)", SINGLE_LABEL, False,
    lambda c: log_loss(c.y_true, c.prob, labels=_labels(c)), value_range=(0.0, None))

# binary-only convenience keys (positive class = index 1)
reg("sensitivity", "Sensitivity (Recall)", {"binary_classification"}, True,
    lambda c: recall_score(c.y_true, c.y_pred, pos_label=1, zero_division=0))
reg("specificity", "Specificity (TNR)", {"binary_classification"}, True,
    lambda c: recall_score(c.y_true, c.y_pred, pos_label=0, zero_division=0))
reg("ppv", "PPV (Precision)", {"binary_classification"}, True,
    lambda c: precision_score(c.y_true, c.y_pred, pos_label=1, zero_division=0))
reg("npv", "NPV", {"binary_classification"}, True,
    lambda c: precision_score(c.y_true, c.y_pred, pos_label=0, zero_division=0))
reg("f1", "F1 (positive class)", {"binary_classification"}, True,
    lambda c: f1_score(c.y_true, c.y_pred, pos_label=1, zero_division=0))
reg("f2", "F2 (positive class)", {"binary_classification"}, True,
    lambda c: fbeta_score(c.y_true, c.y_pred, beta=2, pos_label=1, zero_division=0))
reg("brier", "Brier Score", {"binary_classification"}, False,
    lambda c: brier_score_loss(c.y_true, c.prob_pos))

# ── multilabel classification metrics ────────────────────────────────────────

reg("subset_accuracy", "Subset Accuracy (Exact Match)", MULTILABEL, True,
    lambda c: accuracy_score(c.y_true, c.y_pred))
reg("hamming_loss", "Hamming Loss", MULTILABEL, False,
    lambda c: hamming_loss(c.y_true, c.y_pred), value_range=(0.0, 1.0))
reg("ml_macro_precision", "Macro Precision", MULTILABEL, True,
    lambda c: precision_score(c.y_true, c.y_pred, average="macro", zero_division=0))
reg("ml_macro_recall", "Macro Recall", MULTILABEL, True,
    lambda c: recall_score(c.y_true, c.y_pred, average="macro", zero_division=0))
reg("ml_macro_f1", "Macro F1", MULTILABEL, True,
    lambda c: f1_score(c.y_true, c.y_pred, average="macro", zero_division=0))
reg("ml_micro_f1", "Micro F1", MULTILABEL, True,
    lambda c: f1_score(c.y_true, c.y_pred, average="micro", zero_division=0))
reg("ml_weighted_f1", "Weighted F1", MULTILABEL, True,
    lambda c: f1_score(c.y_true, c.y_pred, average="weighted", zero_division=0))
reg("ml_macro_auc", "Macro AUC-ROC", MULTILABEL, True,
    lambda c: roc_auc_score(c.y_true, c.prob, average="macro"))
reg("ml_macro_ap", "Macro Average Precision (mAP)", MULTILABEL, True,
    lambda c: average_precision_score(c.y_true, c.prob, average="macro"))

# ── regression metrics ───────────────────────────────────────────────────────

def _rmse(c):
    return float(np.sqrt(mean_squared_error(c.y_true, c.y_pred)))


def _msle(c):
    if np.any(np.asarray(c.y_true) < 0) or np.any(np.asarray(c.y_pred) < 0):
        return None
    return float(np.mean((np.log1p(c.y_pred) - np.log1p(c.y_true)) ** 2))


reg("mae", "Mean Absolute Error", REGRESSION, False,
    lambda c: mean_absolute_error(c.y_true, c.y_pred), value_range=(0.0, None))
reg("mse", "Mean Squared Error", REGRESSION, False,
    lambda c: mean_squared_error(c.y_true, c.y_pred), value_range=(0.0, None))
reg("rmse", "Root Mean Squared Error", REGRESSION, False, _rmse, value_range=(0.0, None))
reg("mape", "Mean Absolute Percentage Error", REGRESSION, False,
    lambda c: mean_absolute_percentage_error(c.y_true, c.y_pred), value_range=(0.0, None))
reg("r2", "R-squared", REGRESSION, True,
    lambda c: r2_score(c.y_true, c.y_pred), value_range=(None, 1.0))
reg("explained_variance", "Explained Variance", REGRESSION, True,
    lambda c: explained_variance_score(c.y_true, c.y_pred), value_range=(None, 1.0))
reg("median_ae", "Median Absolute Error", REGRESSION, False,
    lambda c: median_absolute_error(c.y_true, c.y_pred), value_range=(0.0, None))
reg("max_error", "Max Error", REGRESSION, False,
    lambda c: max_error(c.y_true, c.y_pred), value_range=(0.0, None))
reg("msle", "Mean Squared Log Error", REGRESSION, False, _msle, value_range=(0.0, None))


# ── context builders (normalise raw model output into a scorer context) ──────

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def interpret_outputs(raw, task_type, num_classes):
    """Map raw model output to predictions/probabilities per task type."""
    raw = np.asarray(raw, dtype=np.float64)

    if task_type in REGRESSION:
        pred = raw.reshape(raw.shape[0], -1)
        return {"y_pred": pred[:, 0] if pred.shape[1] == 1 else pred}

    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)

    if task_type in MULTILABEL:
        prob = _sigmoid(raw)
        return {"prob": prob, "y_pred": (prob >= MULTILABEL_THRESHOLD).astype(int)}

    # single-label
    if raw.shape[1] == 1:  # single-logit binary head
        prob_pos = _sigmoid(raw[:, 0])
        prob = np.stack([1.0 - prob_pos, prob_pos], axis=1)
        return {"prob": prob, "prob_pos": prob_pos,
                "y_pred": (prob_pos > BINARY_THRESHOLD).astype(int)}
    prob = _softmax(raw)
    return {"prob": prob,
            "prob_pos": prob[:, 1] if num_classes == 2 else None,
            "y_pred": prob.argmax(axis=1)}


def build_context(y_true, interp, task_type, num_classes):
    if task_type in REGRESSION:
        return SimpleNamespace(y_true=np.asarray(y_true, dtype=np.float64),
                               y_pred=np.asarray(interp["y_pred"], dtype=np.float64))
    if task_type in MULTILABEL:
        return SimpleNamespace(y_true=np.asarray(y_true, dtype=int),
                               y_pred=np.asarray(interp["y_pred"], dtype=int),
                               prob=np.asarray(interp["prob"], dtype=np.float64),
                               num_classes=num_classes)
    # single-label: precompute confusion matrix + per-class specificity/npv
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(interp["y_pred"], dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    spec, npv = [], []
    total = int(cm.sum())
    for i in range(num_classes):
        tp = int(cm[i, i]); fn = int(cm[i, :].sum()) - tp
        fp = int(cm[:, i].sum()) - tp; tn = total - tp - fn - fp
        spec.append(tn / (tn + fp) if (tn + fp) else 0.0)
        npv.append(tn / (tn + fn) if (tn + fn) else 0.0)
    return SimpleNamespace(y_true=y_true, y_pred=y_pred,
                           prob=np.asarray(interp["prob"], dtype=np.float64),
                           prob_pos=interp.get("prob_pos"),
                           num_classes=num_classes, cm=cm,
                           per_class_spec=np.array(spec), per_class_npv=np.array(npv))


# ════════════════════════════════════════════════════════════════════════════
# ONNX MODEL LOADING (external-data safe, filename-invariant)
# Ported from the per-vertical scripts — robustly resolves the external weights
# blob regardless of how the .onnx / .onnx.data are named.
# ════════════════════════════════════════════════════════════════════════════

def _discover_weights_file(model_path, weights_hint, model_dir):
    model_path = model_path.resolve(); model_dir = model_dir.resolve()
    if weights_hint:
        hint = Path(weights_hint)
        if not hint.is_absolute():
            hint = model_dir / hint.name
        if hint.exists() and hint.resolve() != model_path:
            return hint.resolve()
    siblings = [p.resolve() for p in model_dir.iterdir()
                if p.is_file() and p.resolve() != model_path]
    if not siblings:
        return None
    if len(siblings) == 1:
        return siblings[0]
    weighty = [p for p in siblings
               if p.suffix.lower() in (".data", ".bin", ".weights")
               or p.name.lower().endswith(".onnx.data")]
    pool = weighty or [p for p in siblings if p.suffix.lower() != ".onnx"] or siblings
    return max(pool, key=lambda p: p.stat().st_size)


def load_onnx_model_bytes(model_path, weights_path):
    import onnx
    from onnx.external_data_helper import _get_all_tensors, load_external_data_for_model
    model_path = Path(model_path).resolve()
    model_dir = model_path.parent
    model = onnx.load(str(model_path), load_external_data=False)
    ext = [t for t in _get_all_tensors(model)
           if t.HasField("data_location") and t.data_location == onnx.TensorProto.EXTERNAL]
    if not ext:
        log("  ONNX model is self-contained (no external data)")
        return model_path.read_bytes()
    weights_file = _discover_weights_file(model_path, weights_path, model_dir)
    if weights_file is None or not weights_file.exists():
        raise FileNotFoundError(f"external weights declared but not found in {model_dir}")
    for t in ext:
        for entry in t.external_data:
            if entry.key == "location":
                entry.value = weights_file.name
    log(f"  external weights -> '{weights_file.name}' ({weights_file.stat().st_size} bytes)")
    load_external_data_for_model(model, str(model_dir))
    return model.SerializeToString()


# ════════════════════════════════════════════════════════════════════════════
# DECODERS (PLATFORM-owned; one per modality). Return uint8 (H,W,3) RGB.
# Add a modality here — this is the extension point.
# ════════════════════════════════════════════════════════════════════════════

def _decode_image(path):
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"cv2 could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _decode_dicom(path):
    import pydicom
    from pydicom.uid import ImplicitVRLittleEndian
    ds = pydicom.dcmread(str(path), force=True)
    if not hasattr(ds, "file_meta"):
        ds.file_meta = pydicom.dataset.FileMetaDataset()
    if getattr(ds.file_meta, "TransferSyntaxUID", None) is None:
        ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] > 1 and arr.shape[-1] != 3:
        arr = arr[arr.shape[0] // 2]
    if "MONOCHROME1" in str(getattr(ds, "PhotometricInterpretation", "")).upper():
        arr = arr.max() - arr
    arr -= arr.min()
    mx = arr.max()
    if mx > 0:
        arr /= mx
    arr = (arr * 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr[..., :3]


DECODERS = {"image": _decode_image, "dicom": _decode_dicom}


# ════════════════════════════════════════════════════════════════════════════
# PREPROCESSING (MODEL provider's responsibility; built-in default is a baseline)
# Contract for a user --preprocessing script:
#   def preprocess(image: np.ndarray) -> np.ndarray
#       image: uint8 (H,W,3) RGB; returns float32 (C,H,W)
# ════════════════════════════════════════════════════════════════════════════

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def make_default_preprocess(input_hints, model_input_size):
    import cv2
    size = int(input_hints.get("size") or model_input_size or DEFAULT_INPUT_SIZE)
    normalize = str(input_hints.get("normalize") or "").lower()

    def _fn(image):
        img = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        arr = img.astype(np.float32) / 255.0
        if normalize == "imagenet":
            arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        return arr.transpose(2, 0, 1)

    return _fn


def load_user_preprocess(script_path):
    spec = importlib.util.spec_from_file_location("user_preprocessing", str(script_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "preprocess", None)
    if fn is None or not callable(fn):
        raise ValueError("preprocessing.py must define a callable named 'preprocess'")
    return fn


# ════════════════════════════════════════════════════════════════════════════
# DECLARATION (dataset_spec.json, from the catalogue) + GROUND TRUTH (labels only)
# + DATA-zip extraction and layout-independent file resolution
# ════════════════════════════════════════════════════════════════════════════

DATA_JUNK_NAMES = {".DS_Store", "desktop.ini", "Thumbs.db"}


def _is_junk(name: str) -> bool:
    base = Path(name).name
    return (base in DATA_JUNK_NAMES or base.startswith("._")
            or name.startswith("__MACOSX/") or "/__MACOSX/" in name)


def load_declaration(dataset_dir):
    """Read dataset_spec.json — the declaration the TEE fetched from the catalogue."""
    spec_path = dataset_dir / "dataset_spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"dataset_spec.json not found in {dataset_dir}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    task_type = str(spec.get("task_type", "")).strip()
    if task_type not in ALL_TASKS:
        raise ValueError(f"unsupported task_type {task_type!r}; supported: {sorted(ALL_TASKS)}")
    modality = str(spec.get("modality", "image")).strip().lower()

    num_classes, class_names = 0, []
    if task_type in CLASSIFICATION:
        class_names = [str(c) for c in (spec.get("class_names") or [])]
        num_classes = int(spec.get("num_classes") or len(class_names))
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2 for classification")
        if not class_names:
            class_names = [str(i) for i in range(num_classes)]
        if len(class_names) != num_classes:
            raise ValueError(f"class_names ({len(class_names)}) != num_classes ({num_classes})")

    return SimpleNamespace(
        task_type=task_type, modality=modality,
        num_classes=num_classes, class_names=class_names,
        input=spec.get("input") or {}, data_file=spec.get("data_file"),
    )


def extract_data_zip(dataset_dir):
    """Extract the single DATA zip (zip-slip safe, junk skipped) and return the
    effective data root — stripping one wrapper folder if the zip nests everything
    under a single dir. If there is no zip, fall back to dataset_dir."""
    import zipfile
    zips = sorted(dataset_dir.glob("*.zip"))
    if not zips:
        return dataset_dir
    dest = (dataset_dir / "_data").resolve()
    dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(zips[0]) as zf:
        for info in zf.infolist():
            if _is_junk(info.filename):
                continue
            target = (dest / info.filename).resolve()
            if target != dest and not str(target).startswith(str(dest) + os.sep):
                raise ValueError(f"zip entry escapes destination: {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                out.write(src.read())
    entries = [p for p in dest.iterdir() if not _is_junk(p.name)]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]          # strip a single wrapper folder
    return dest


def build_file_index(root):
    """basename -> [resolved paths] over the whole extracted tree (junk excluded)."""
    index = {}
    for p in root.rglob("*"):
        if p.is_file() and not _is_junk(str(p.relative_to(root))):
            index.setdefault(p.name, []).append(p.resolve())
    return index


def resolve_file(rel, root, index):
    """Three-tier resolve: exact relative path -> unique basename. Fail closed."""
    exact = root / rel
    if exact.is_file():
        return exact.resolve()
    hits = index.get(Path(rel).name, [])
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise FileNotFoundError(f"referenced file not found: {rel}")
    raise ValueError(f"ambiguous basename {Path(rel).name!r}: {len(hits)} matches — "
                     f"use a full relative path in ground truth")


def _normalize_label(raw, task_type, num_classes):
    if task_type in REGRESSION:
        return float(raw)
    if task_type in MULTILABEL:
        vec = np.zeros(num_classes, dtype=int)
        if isinstance(raw, (list, tuple)):
            arr = list(raw)
            if len(arr) == num_classes and set(int(x) for x in arr) <= {0, 1}:
                vec = np.array([int(x) for x in arr], dtype=int)    # 0/1 per class
            else:
                for idx in arr:                                     # active indices
                    vec[int(idx)] = 1
        return vec
    label = int(raw)                                                # single-label
    if not (0 <= label < num_classes):
        raise ValueError(f"label {label} out of range [0,{num_classes-1}]")
    return label


def load_ground_truth(dataset_dir, decl):
    """Read the labels-only table. Returns (refs, raw_labels): refs is a list of
    file references (file modalities) or None (tabular)."""
    gt_json = dataset_dir / "ground_truth.json"
    gt_csv = dataset_dir / "ground_truth.csv"
    if gt_json.exists():
        refs, labels = _parse_gt_json(json.loads(gt_json.read_text(encoding="utf-8")))
    elif gt_csv.exists():
        refs, labels = _parse_gt_csv(gt_csv)
    else:
        raise FileNotFoundError(
            f"ground_truth.json / ground_truth.csv not found in {dataset_dir}")
    if not labels:
        raise ValueError("ground truth contained no rows")
    return refs, labels


def _parse_gt_json(gt):
    if isinstance(gt, dict) and "labels" in gt:            # tabular {"labels":[...]}
        return None, list(gt["labels"])
    if isinstance(gt, dict):                               # map {file: label}
        return list(gt.keys()), list(gt.values())
    if isinstance(gt, list):
        if gt and isinstance(gt[0], dict):                 # [{file,label}] / [{label}]
            if "file" in gt[0]:
                return [str(r["file"]) for r in gt], [r["label"] for r in gt]
            return None, [r["label"] for r in gt]
        return None, list(gt)                              # [label, ...] tabular
    raise ValueError("unrecognised ground_truth.json shape")


def _parse_gt_csv(path):
    import csv as _csv
    with open(path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        cols = reader.fieldnames or []
        if "label" not in cols:
            raise ValueError("ground_truth.csv must have a 'label' column")
        has_file = "file" in cols
        refs, labels = [], []
        for row in reader:
            labels.append(row["label"])
            if has_file:
                refs.append(str(row["file"]).strip())
    return (refs if has_file else None), labels


def load_features(data_root, decl):
    """tabular: load the features matrix from the extracted DATA-zip contents."""
    if decl.data_file:
        path = data_root / decl.data_file
        if not path.exists():
            hits = build_file_index(data_root).get(Path(decl.data_file).name, [])
            if len(hits) != 1:
                raise FileNotFoundError(f"tabular data_file not found: {decl.data_file}")
            path = hits[0]
    else:
        cand = [p for p in data_root.rglob("*")
                if p.is_file() and not _is_junk(p.name)
                and p.suffix.lower() in (".csv", ".npy")]
        if len(cand) != 1:
            raise ValueError("tabular: set data_file (couldn't pick a single features file)")
        path = cand[0]
    features = np.load(path) if path.suffix.lower() == ".npy" else \
        np.loadtxt(path, delimiter=",", dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    return features.reshape(1, -1) if features.ndim == 1 else features


# ════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ════════════════════════════════════════════════════════════════════════════

def _fixed_batch(session):
    ishape = session.get_inputs()[0].shape
    return ishape[0] if isinstance(ishape[0], int) and ishape[0] > 1 else DEFAULT_BATCH


def _run_batched(session, tensors):
    """tensors: list of np arrays already in model-input layout. Returns raw [N,*]."""
    iname = session.get_inputs()[0].name
    onames = [o.name for o in session.get_outputs()]
    fb = _fixed_batch(session)
    parts, timing = [], []
    for off in range(0, len(tensors), fb):
        chunk = tensors[off: off + fb]
        actual = len(chunk)
        if actual < fb:
            chunk = chunk + [np.zeros_like(chunk[0])] * (fb - actual)
        batch = np.stack(chunk, axis=0).astype(np.float32)
        t0 = time.perf_counter()
        out = np.asarray(session.run(onames, {iname: batch})[0])
        timing.extend([(time.perf_counter() - t0) / fb] * actual)
        parts.append(out[:actual])
    return np.concatenate(parts, axis=0), timing


def infer_files(session, decode_fn, preprocess_fn, files, labels):
    tensors, kept_labels, failed = [], [], 0
    expected = None
    for i, path in enumerate(files):
        if i % 50 == 0:
            log(f"  loading {i}/{len(files)}…")
        try:
            t = np.asarray(preprocess_fn(decode_fn(path)), dtype=np.float32)
            if t.ndim != 3:
                raise ValueError(f"preprocess returned {t.ndim}D, expected 3D (C,H,W)")
            if expected is None:
                expected = t.shape
                log(f"  preprocessed tensor shape: {expected}")
            elif t.shape != expected:
                raise ValueError(f"inconsistent tensor shape {t.shape} != {expected}")
            tensors.append(t)
            kept_labels.append(labels[i])
        except Exception as exc:
            log(f"  [SKIP] {Path(path).name}: {exc}")
            failed += 1
    if not tensors:
        raise RuntimeError("no samples could be decoded/preprocessed")
    raw, timing = _run_batched(session, tensors)
    return np.array(kept_labels), raw, timing, failed


def infer_tabular(session, features, labels):
    raw, timing = _run_batched(session, [row for row in features.astype(np.float32)])
    return labels, raw, timing, 0


# ════════════════════════════════════════════════════════════════════════════
# SELF-TEST (synthetic; proves the whole registry computes)
# ════════════════════════════════════════════════════════════════════════════

def selftest():
    rng = np.random.default_rng(0)
    ok = True
    cases = [
        ("binary_classification", 2, 1),
        ("multiclass_classification", 4, 4),
        ("ordinal_classification", 4, 4),
        ("multilabel_classification", 3, 3),
        ("regression", 0, 1),
    ]
    for task_type, num_classes, out_dim in cases:
        n = 40
        if task_type in REGRESSION:
            y_true = rng.normal(5, 2, size=n)
            raw = y_true + rng.normal(0, 1, size=n)
        elif task_type in MULTILABEL:
            y_true = rng.integers(0, 2, size=(n, num_classes))
            raw = rng.normal(0, 1, size=(n, num_classes)) + (y_true * 2 - 1)
        else:
            y_true = rng.integers(0, num_classes, size=n)
            if out_dim == 1:
                raw = (y_true * 2 - 1) + rng.normal(0, 1, size=n)
            else:
                raw = rng.normal(0, 1, size=(n, num_classes))
                raw[np.arange(n), y_true] += 2.5
        interp = interpret_outputs(raw, task_type, num_classes)
        ctx = build_context(y_true, interp, task_type, num_classes)
        metrics = score(ctx, task_type)
        computed = {k: v for k, v in metrics.items() if v is not None}
        missing = [k for k, v in metrics.items() if v is None]
        print(f"\n{task_type}: {len(computed)} metrics computed"
              + (f", {len(missing)} skipped: {missing}" if missing else ""))
        for k in sorted(computed):
            print(f"    {k:22s} = {computed[k]}")
        if not computed:
            ok = False
    print("\n-- resolver tests --")
    if not _selftest_resolver():
        ok = False
    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _selftest_resolver():
    """Prove the DATA-zip extraction + three-tier resolver on temp zips."""
    import tempfile
    import zipfile
    ok = True

    def make_zip(zip_path, entries):
        with zipfile.ZipFile(zip_path, "w") as zf:
            for name in entries:
                zf.writestr(name, b"x")

    layouts = {
        "flat":    ["img001.jpg", "img002.jpg"],
        "wrapped": ["data/img001.jpg", "data/img002.jpg"],
        "nested":  ["a/b/img001.jpg", "c/img002.jpg"],
    }
    for name, entries in layouts.items():
        with tempfile.TemporaryDirectory() as d:
            dd = Path(d)
            make_zip(dd / "data.zip", entries + ["__MACOSX/x", ".DS_Store"])
            root = extract_data_zip(dd)
            index = build_file_index(root)
            try:
                p1 = resolve_file("img001.jpg", root, index)
                p2 = resolve_file("img002.jpg", root, index)
                assert p1.is_file() and p2.is_file()
                print(f"  resolver[{name}]: OK")
            except Exception as exc:
                print(f"  resolver[{name}]: FAIL {exc}")
                ok = False

    with tempfile.TemporaryDirectory() as d:
        dd = Path(d)
        make_zip(dd / "data.zip", ["a/dup.jpg", "b/dup.jpg"])
        root = extract_data_zip(dd)
        index = build_file_index(root)
        try:
            resolve_file("dup.jpg", root, index)
            print("  resolver[ambiguous]: FAIL (no error)")
            ok = False
        except ValueError:
            print("  resolver[ambiguous]: OK (raised)")

    with tempfile.TemporaryDirectory() as d:
        dd = Path(d)
        make_zip(dd / "data.zip", ["img001.jpg"])
        root = extract_data_zip(dd)
        index = build_file_index(root)
        try:
            resolve_file("nope.jpg", root, index)
            print("  resolver[missing]: FAIL (no error)")
            ok = False
        except FileNotFoundError:
            print("  resolver[missing]: OK (raised)")

    return ok


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--results")
    parser.add_argument("--preprocessing", default=None)
    parser.add_argument("--selftest", action="store_true")
    args, _ = parser.parse_known_args()

    if args.selftest:
        sys.exit(selftest())
    if not (args.model and args.dataset and args.results):
        log("ERROR: --model, --dataset and --results are required (or use --selftest)")
        sys.exit(2)

    import onnxruntime as ort

    wall_start = time.perf_counter()
    dataset_dir = Path(args.dataset)
    model_path = Path(args.model)
    weights_path = Path(str(args.model) + ".data")
    if not weights_path.exists():
        weights_path = None

    log("=" * 60)
    log("GENERIC EVALUATION ENGINE")
    log("=" * 60)

    # 1. Declaration (from the catalogue) + ground truth (labels) + DATA zip.
    log("[1/5] Reading declaration + ground truth + extracting DATA zip")
    decl = load_declaration(dataset_dir)
    data_root = extract_data_zip(dataset_dir)
    refs, raw_labels = load_ground_truth(dataset_dir, decl)
    log(f"  task={decl.task_type} modality={decl.modality} "
        f"classes={decl.num_classes} class_names={decl.class_names}")

    # 2. ONNX session.
    log(f"[2/5] Loading ONNX model: {model_path.name}")
    try:
        model_bytes = load_onnx_model_bytes(model_path, weights_path)
        available = set(ort.get_available_providers())
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in available] or ["CPUExecutionProvider"]
        log(f"  providers={providers}")
        session = ort.InferenceSession(model_bytes, providers=providers)
    except Exception as exc:
        if any(k in str(exc).lower() for k in ("cuda", "gpu", "nvidia", "cudnn", "out of memory", "oom")):
            log(f"  CUDA/GPU error: {exc}")
            sys.exit(11)
        raise
    inp0 = session.get_inputs()[0]
    model_input_size = (int(inp0.shape[-1]) if len(inp0.shape) == 4
                        and isinstance(inp0.shape[-1], int) else DEFAULT_INPUT_SIZE)

    # 3. Resolve data + preprocessing, then inference.
    log("[3/5] Resolving data")
    if decl.modality in FILE_MODALITIES:
        if decl.modality not in DECODERS:
            raise ValueError(f"no decoder for modality {decl.modality!r}")
        if refs is None:
            raise ValueError(f"{decl.modality} modality needs a 'file' column in ground truth")
        index = build_file_index(data_root)
        files = [resolve_file(r, data_root, index) for r in refs]   # fail-closed
        labels = np.array([_normalize_label(l, decl.task_type, decl.num_classes)
                           for l in raw_labels])
        log(f"  {len(files)} samples resolved (modality={decl.modality})")
        if args.preprocessing:
            try:
                preprocess_fn = load_user_preprocess(Path(args.preprocessing))
                log("  using user-supplied preprocessing.py")
            except Exception as exc:
                log(f"  ERROR loading preprocessing.py: {exc}")
                sys.exit(10)
        else:
            preprocess_fn = make_default_preprocess(decl.input, model_input_size)
            log("  using built-in default preprocessing")
        log("[4/5] Running inference")
        y_true, raw, timing, failed = infer_files(
            session, DECODERS[decl.modality], preprocess_fn, files, labels)
    elif decl.modality == "tabular":
        features = load_features(data_root, decl)
        labels = np.array([_normalize_label(l, decl.task_type, decl.num_classes)
                           for l in raw_labels])
        if len(labels) != len(features):
            raise ValueError(f"labels ({len(labels)}) != feature rows ({len(features)})")
        log(f"  {len(features)} rows x {features.shape[1]} features (tabular)")
        log("[4/5] Running inference")
        y_true, raw, timing, failed = infer_tabular(session, features, labels)
    else:
        raise ValueError(f"unknown modality {decl.modality!r}")

    # 5. Interpret + score.
    log("[5/5] Interpreting outputs + computing metrics")
    interp = interpret_outputs(raw, decl.task_type, decl.num_classes)
    ctx = build_context(y_true, interp, decl.task_type, decl.num_classes)
    metrics = score(ctx, decl.task_type)

    wall_elapsed = time.perf_counter() - wall_start
    result = {
        "status": "success",
        "task_type": decl.task_type,
        "modality": decl.modality,
        "num_classes": decl.num_classes,
        "class_names": decl.class_names,
        "num_samples": int(len(y_true)),
        "num_failed_load": int(failed),
        "elapsed_seconds": round(wall_elapsed, 4),
        "model_sha256": sha256_file(model_path),
        "onnx_runtime_providers": session.get_providers(),
        "onnx_input_name": inp0.name,
        "onnx_output_names": [o.name for o in session.get_outputs()],
        "output_shape": list(np.asarray(raw).shape),
        "metrics": metrics,
    }
    if decl.task_type in SINGLE_LABEL:
        result["confusion_matrix"] = ctx.cm.tolist()
        result["prediction_distribution"] = {
            str(i): int((ctx.y_pred == i).sum()) for i in range(decl.num_classes)}
        result["label_distribution"] = {
            str(i): int((ctx.y_true == i).sum()) for i in range(decl.num_classes)}

    out_path = Path(args.results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    primary = {k: v for k, v in metrics.items() if v is not None}
    log("=" * 60)
    log(f"  samples={result['num_samples']} failed={failed} "
        f"metrics_computed={len(primary)}")
    log(f"  results -> {out_path}")
    log("=" * 60)


if __name__ == "__main__":
    main()
