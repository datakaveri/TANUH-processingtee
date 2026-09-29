import numpy as np
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from eval_lib.binary import compute_binary_metrics


def test_matches_sklearn_precision_recall_f1():
    y_true = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 0, 0, 1, 1, 1, 0, 1, 1]

    metrics = compute_binary_metrics(y_true, y_pred)

    assert metrics["ppv"] == pytest.approx(precision_score(y_true, y_pred))
    assert metrics["sensitivity"] == pytest.approx(recall_score(y_true, y_pred))
    assert metrics["f1"] == pytest.approx(f1_score(y_true, y_pred))


def test_confusion_matrix_and_npv_specificity_by_hand():
    # true=1 at idx 0,2,4,5,8 (5 positives); true=0 at idx 1,3,6,7,9 (5 negatives)
    y_true = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
    # pred flips idx 2 (FN) and idx 6 (FP) relative to y_true
    y_pred = [1, 0, 0, 0, 1, 1, 1, 0, 1, 0]

    metrics = compute_binary_metrics(y_true, y_pred)

    # TP=4 (0,4,5,8), FN=1 (2), TN=4 (1,3,7,9), FP=1 (6)
    assert metrics["confusion_matrix"] == [[4, 1], [1, 4]]
    assert metrics["sensitivity"] == pytest.approx(4 / 5)
    assert metrics["specificity"] == pytest.approx(4 / 5)
    assert metrics["ppv"] == pytest.approx(4 / 5)
    assert metrics["npv"] == pytest.approx(4 / 5)
    assert metrics["accuracy"] == pytest.approx(8 / 10)


def test_f2_weighs_recall_more_than_precision():
    # Perfect recall, imperfect precision: predicts positive too often.
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 1, 1, 1, 0]

    metrics = compute_binary_metrics(y_true, y_pred)

    assert metrics["sensitivity"] == pytest.approx(1.0)
    assert metrics["ppv"] == pytest.approx(3 / 5)
    # F2 should sit closer to recall (1.0) than F1 does.
    assert metrics["f2"] > metrics["f1"]


def test_degenerate_denominators_return_zero_not_nan():
    # No actual negatives at all -> specificity/npv undefined -> must be 0.0, not NaN.
    y_true = [1, 1, 1, 1]
    y_pred = [1, 0, 1, 1]

    metrics = compute_binary_metrics(y_true, y_pred)

    assert metrics["specificity"] == 0.0
    assert metrics["npv"] == 0.0
    assert not any(isinstance(v, float) and v != v for v in metrics.values())  # no NaNs anywhere


def test_auc_matches_sklearn_when_scores_given():
    y_true = [0, 0, 1, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8, 0.9]
    y_pred = [0, 0, 0, 1, 1]

    metrics = compute_binary_metrics(y_true, y_pred, y_score=y_score)

    assert metrics["auc"] == pytest.approx(roc_auc_score(y_true, y_score))


def test_rejects_labels_outside_zero_one():
    with pytest.raises(ValueError):
        compute_binary_metrics([0, 1, 2], [0, 1, 1])


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        compute_binary_metrics([0, 1], [0, 1, 1])
