"""CSV loading for the two shapes documented in notes.md:

  Option A/B, single file — one row per sample, a ground-truth column and a
  predictions column side by side (id,ground_truth,predictions or
  sample_id,true_rating,predicted_rating):

      load_single_csv(path)

  Two separate files — "model outputs" produced by inference, and a
  separate ground-truth file, joined by an id column:

      load_paired_csvs(ground_truth_path, predictions_path)

Column names are auto-detected from common aliases; pass explicit column
names when a file doesn't match those aliases.
"""

from __future__ import annotations

import csv
from typing import Optional

ID_COLUMN_ALIASES = ["id", "sample_id", "uid", "row_id"]
GT_COLUMN_ALIASES = ["ground_truth", "true_rating", "y_true", "label", "true_label", "truth"]
PRED_COLUMN_ALIASES = ["predictions", "predicted_rating", "y_pred", "prediction", "predicted_label"]
SCORE_COLUMN_ALIASES = ["score", "y_score", "probability", "prob", "confidence"]


class CsvFormatError(ValueError):
    pass


def _read_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise CsvFormatError(f"{path}: no data rows found")
    return rows


def _detect_column(fieldnames: list[str], aliases: list[str], explicit: Optional[str], role: str, path: str) -> str:
    if explicit is not None:
        if explicit not in fieldnames:
            raise CsvFormatError(f"{path}: column {explicit!r} not found; available columns: {fieldnames}")
        return explicit
    lower_map = {name.lower(): name for name in fieldnames}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    raise CsvFormatError(
        f"{path}: could not auto-detect the {role} column among {fieldnames}. "
        f"Expected one of {aliases}, or pass the column name explicitly."
    )


def load_single_csv(
    path: str,
    id_col: Optional[str] = None,
    gt_col: Optional[str] = None,
    pred_col: Optional[str] = None,
    score_col: Optional[str] = None,
):
    """Load a single CSV containing both ground truth and predictions.

    Returns (ids, y_true_raw, y_pred_raw, y_score) where y_score is None if no
    score column was found/requested. ids fall back to 1-based row numbers
    when no id column is present.
    """
    rows = _read_rows(path)
    fieldnames = list(rows[0].keys())

    gt_field = _detect_column(fieldnames, GT_COLUMN_ALIASES, gt_col, "ground truth", path)
    pred_field = _detect_column(fieldnames, PRED_COLUMN_ALIASES, pred_col, "predictions", path)

    id_field = id_col
    if id_field is None:
        lower_map = {name.lower(): name for name in fieldnames}
        for alias in ID_COLUMN_ALIASES:
            if alias in lower_map:
                id_field = lower_map[alias]
                break

    score_field = score_col
    if score_field is None:
        lower_map = {name.lower(): name for name in fieldnames}
        for alias in SCORE_COLUMN_ALIASES:
            if alias in lower_map:
                score_field = lower_map[alias]
                break

    ids, y_true_raw, y_pred_raw, y_score = [], [], [], []
    for row_num, row in enumerate(rows, start=2):  # header is row 1
        ids.append(row[id_field] if id_field else str(row_num))
        gt_value = row.get(gt_field, "")
        pred_value = row.get(pred_field, "")
        if gt_value is None or gt_value.strip() == "":
            raise CsvFormatError(f"{path}: row {row_num} has an empty {gt_field!r} value")
        if pred_value is None or pred_value.strip() == "":
            raise CsvFormatError(f"{path}: row {row_num} has an empty {pred_field!r} value")
        y_true_raw.append(gt_value)
        y_pred_raw.append(pred_value)
        y_score.append(float(row[score_field]) if score_field and row.get(score_field, "") != "" else None)

    if any(s is None for s in y_score):
        y_score = None

    return ids, y_true_raw, y_pred_raw, y_score


def load_paired_csvs(
    ground_truth_path: str,
    predictions_path: str,
    id_col: Optional[str] = None,
    gt_col: Optional[str] = None,
    pred_col: Optional[str] = None,
    score_col: Optional[str] = None,
):
    """Load ground truth and predictions from two separate files, joined by id.

    This is the shape used when predictions come straight out of a model-
    execution run (e.g. an inference job's output CSV) and ground truth is a
    fixed reference file for the dataset.
    """
    gt_rows = _read_rows(ground_truth_path)
    pred_rows = _read_rows(predictions_path)

    gt_fields = list(gt_rows[0].keys())
    pred_fields = list(pred_rows[0].keys())

    gt_id_field = _detect_column(gt_fields, ID_COLUMN_ALIASES, id_col, "id", ground_truth_path)
    pred_id_field = _detect_column(pred_fields, ID_COLUMN_ALIASES, id_col, "id", predictions_path)
    gt_field = _detect_column(gt_fields, GT_COLUMN_ALIASES, gt_col, "ground truth", ground_truth_path)
    pred_field = _detect_column(pred_fields, PRED_COLUMN_ALIASES, pred_col, "predictions", predictions_path)

    score_field = score_col
    if score_field is None:
        lower_map = {name.lower(): name for name in pred_fields}
        for alias in SCORE_COLUMN_ALIASES:
            if alias in lower_map:
                score_field = lower_map[alias]
                break

    gt_by_id = {row[gt_id_field]: row[gt_field] for row in gt_rows}
    pred_by_id = {row[pred_id_field]: row[pred_field] for row in pred_rows}
    score_by_id = (
        {row[pred_id_field]: row[score_field] for row in pred_rows if row.get(score_field, "") != ""}
        if score_field
        else {}
    )

    missing_predictions = gt_by_id.keys() - pred_by_id.keys()
    missing_ground_truth = pred_by_id.keys() - gt_by_id.keys()
    if missing_predictions:
        raise CsvFormatError(
            f"{predictions_path}: missing predictions for {len(missing_predictions)} id(s) present in "
            f"{ground_truth_path}, e.g. {sorted(missing_predictions)[:5]}"
        )
    if missing_ground_truth:
        raise CsvFormatError(
            f"{ground_truth_path}: missing ground truth for {len(missing_ground_truth)} id(s) present in "
            f"{predictions_path}, e.g. {sorted(missing_ground_truth)[:5]}"
        )

    ids = list(gt_by_id.keys())
    y_true_raw = [gt_by_id[i] for i in ids]
    y_pred_raw = [pred_by_id[i] for i in ids]
    y_score = [float(score_by_id[i]) for i in ids] if score_field and len(score_by_id) == len(ids) else None

    return ids, y_true_raw, y_pred_raw, y_score
