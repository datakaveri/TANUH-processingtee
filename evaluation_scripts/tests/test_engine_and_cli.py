import json
import subprocess
import sys
from pathlib import Path

import pytest

from eval_lib import evaluate, evaluate_from_files
from eval_lib.datasets import get_dataset_config

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).parent.parent


def test_evaluate_coerces_oral_cancer_string_labels():
    metrics = evaluate(
        "oral_cancer",
        y_true_raw=["Suspicious", "Non-Suspicious", "Suspicious"],
        y_pred_raw=["Suspicious", "Non-Suspicious", "Non-Suspicious"],
    )
    assert metrics["dataset"] == "oral_cancer"
    assert metrics["problem_type"] == "binary"
    assert metrics["accuracy"] == pytest.approx(2 / 3)


def test_evaluate_from_single_csv_oral_cancer():
    metrics = evaluate_from_files("oral_cancer", input_csv=str(FIXTURES / "oral_cancer_single.csv"))
    # 6 rows, 2 disagreements (rows 103 and 104) -> accuracy 4/6
    assert metrics["num_samples"] == 6
    assert metrics["accuracy"] == pytest.approx(4 / 6)


def test_evaluate_from_single_csv_breast_cancer_ordinal():
    metrics = evaluate_from_files("breast_cancer", input_csv=str(FIXTURES / "breast_cancer_single.csv"))
    assert metrics["problem_type"] == "multiclass"
    assert metrics["num_classes"] == 4
    assert 0.0 <= metrics["qwk"] <= 1.0


def test_evaluate_from_paired_csvs_joins_by_id_and_computes_auc():
    metrics = evaluate_from_files(
        "oral_cancer",
        ground_truth_csv=str(FIXTURES / "oral_cancer_ground_truth.csv"),
        predictions_csv=str(FIXTURES / "oral_cancer_model_outputs.csv"),
    )
    assert metrics["num_samples"] == 5
    assert "auc" in metrics
    assert metrics["auc"] is not None


def test_cli_end_to_end_writes_output_json(tmp_path):
    output_path = tmp_path / "results.json"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "cli.py"),
            "--dataset",
            "oral_cancer",
            "--input",
            str(FIXTURES / "oral_cancer_single.csv"),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text())
    assert payload["dataset"] == "oral_cancer"
    assert payload["accuracy"] == pytest.approx(4 / 6)


def test_cli_rejects_conflicting_input_flags():
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "cli.py"),
            "--dataset",
            "oral_cancer",
            "--input",
            str(FIXTURES / "oral_cancer_single.csv"),
            "--ground-truth",
            str(FIXTURES / "oral_cancer_ground_truth.csv"),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 2
    assert "either" in result.stderr


def _f2_from_precision_recall(p: float, r: float) -> float:
    """Mirrors TANUH-processingtee/enclave_manager_new.py::_f2_from_precision_recall exactly."""
    denom = 4.0 * p + r
    return (5.0 * p * r / denom) if denom else 0.0


def test_per_class_output_is_compatible_with_enclave_managers_weighted_f2_fallback():
    """
    enclave_manager_new.py::_leaderboard_metrics only trusts our weighted_f2 if
    we emit it; if we didn't, it recomputes it from
    per_class[...]["TP"]/["FN"]/["precision"]/["recall"]. This guards that our
    per_class dict stays shaped so that fallback reproduces our own weighted_f2
    (i.e. the two code paths can't silently drift apart).
    """
    metrics = evaluate(
        "breast_cancer",
        y_true_raw=["1", "1", "1", "2", "2", "3", "3", "3", "3", "4"],
        y_pred_raw=["1", "1", "2", "2", "3", "3", "3", "4", "3", "4"],
    )

    per_class = metrics["per_class"]
    total = 0
    acc = 0.0
    for entry in per_class.values():
        support = int(entry["TP"]) + int(entry["FN"])
        total += support
        acc += support * _f2_from_precision_recall(entry["precision"], entry["recall"])
    recomputed_weighted_f2 = (acc / total) if total else None

    assert recomputed_weighted_f2 == pytest.approx(metrics["weighted_f2"])


def test_get_dataset_config_rejects_unknown_dataset():
    with pytest.raises(ValueError):
        get_dataset_config("glaucoma_not_yet_supported")
