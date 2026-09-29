import pytest
from sklearn.metrics import cohen_kappa_score

from eval_lib.multiclass import compute_multiclass_metrics


def test_qwk_matches_sklearn_for_ordinal_ratings():
    # notes.md Option B example values.
    y_true = [5, 2, 1, 4]
    y_pred = [4, 2, 3, 5]

    metrics = compute_multiclass_metrics(y_true, y_pred, labels=[1, 2, 3, 4, 5])

    expected_qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=[1, 2, 3, 4, 5])
    assert metrics["qwk"] == pytest.approx(expected_qwk)


def test_qwk_penalizes_distant_errors_more_than_close_ones():
    # Same number of "misses" but one set is off-by-one, the other off-by-far.
    close_miss = compute_multiclass_metrics([1, 2, 3, 4], [2, 2, 3, 3], labels=[1, 2, 3, 4])
    far_miss = compute_multiclass_metrics([1, 2, 3, 4], [4, 2, 3, 1], labels=[1, 2, 3, 4])

    assert close_miss["qwk"] > far_miss["qwk"]


def test_perfect_predictions_give_perfect_scores():
    y_true = [1, 2, 3, 4, 1, 2, 3, 4]
    y_pred = [1, 2, 3, 4, 1, 2, 3, 4]

    metrics = compute_multiclass_metrics(y_true, y_pred, labels=[1, 2, 3, 4])

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["macro_f2"] == pytest.approx(1.0)
    assert metrics["qwk"] == pytest.approx(1.0)
    assert metrics["macro_ppv"] == pytest.approx(1.0)
    assert metrics["macro_recall"] == pytest.approx(1.0)
    assert metrics["macro_specificity"] == pytest.approx(1.0)
    assert metrics["macro_npv"] == pytest.approx(1.0)


def test_per_class_breakdown_has_keys_leaderboard_mapper_relies_on():
    # enclave_manager_new.py::_leaderboard_metrics derives weighted_f2 from
    # per_class[...]["TP"], ["FN"], ["precision"], ["recall"] when weighted_f2
    # isn't already present, so those exact keys must exist.
    y_true = [1, 2, 3, 4, 2, 3]
    y_pred = [1, 2, 2, 4, 2, 3]

    metrics = compute_multiclass_metrics(y_true, y_pred, labels=[1, 2, 3, 4])

    for label in ("1", "2", "3", "4"):
        entry = metrics["per_class"][label]
        for key in ("TP", "FP", "FN", "TN", "precision", "recall"):
            assert key in entry


def test_weighted_f2_weighs_by_support():
    # Class 1 has far more support than class 2 and is predicted perfectly;
    # class 2 is predicted badly. Weighted F2 should sit much closer to 1.0
    # than macro F2 does, since macro treats both classes equally.
    y_true = [1] * 90 + [2] * 10
    y_pred = [1] * 90 + [1] * 10  # every class-2 sample misclassified as 1

    metrics = compute_multiclass_metrics(y_true, y_pred, labels=[1, 2])

    assert metrics["weighted_f2"] > metrics["macro_f2"]


def test_confusion_matrix_includes_absent_labels():
    # label 4 never appears in this sample, but the confusion matrix and
    # per-class breakdown must still include it (e.g. for a leaderboard that
    # always expects a 4x4 matrix for the 4-category breast-density dataset).
    y_true = [1, 2, 3, 1, 2]
    y_pred = [1, 2, 3, 1, 1]

    metrics = compute_multiclass_metrics(y_true, y_pred, labels=[1, 2, 3, 4])

    assert metrics["num_classes"] == 4
    assert len(metrics["confusion_matrix"]) == 4
    assert metrics["per_class"]["4"]["support"] == 0


def test_rejects_fewer_than_two_classes():
    with pytest.raises(ValueError):
        compute_multiclass_metrics([1, 1, 1], [1, 1, 1], labels=[1])
