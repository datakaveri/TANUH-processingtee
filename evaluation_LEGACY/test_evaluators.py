"""
Contract tests for the bucket evaluators (stdlib unittest; needs numpy + scikit-learn).

    python3 -m unittest evaluation_LEGACY/test_evaluators.py

Each evaluator is run as a subprocess exactly as the Processing TEE runs it,
so the tests pin the CLI, the exit codes and the results.json shape.
"""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
BINARY = HERE / "binary_classification" / "evaluate.py"
MULTI = HERE / "multiclass_classification" / "evaluate.py"


def run_eval(script, gt_rows, pred_header, pred_rows, class_names):
    d = Path(tempfile.mkdtemp())
    with open(d / "ground_truth.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label"])
        w.writerows(gt_rows)
    with open(d / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(pred_header)
        w.writerows(pred_rows)
    (d / "spec.json").write_text(json.dumps({"class_names": class_names}))
    proc = subprocess.run(
        [sys.executable, str(script),
         "--predictions", str(d / "predictions.csv"),
         "--ground-truth", str(d / "ground_truth.csv"),
         "--spec", str(d / "spec.json"),
         "--results", str(d / "results.json")],
        capture_output=True, text=True)
    results = json.loads((d / "results.json").read_text()) if proc.returncode == 0 else None
    return proc.returncode, results, proc.stdout + proc.stderr


BIN_GT = [["a.jpg", 0], ["b.jpg", 1], ["c.jpg", 1], ["d.jpg", 0]]
BIN_HDR = ["file", "label", "score"]
BIN_OK = [["a.jpg", 0, 0.1], ["b.jpg", 1, 0.9], ["c.jpg", 0, 0.4], ["d.jpg", 1, 0.6]]
BIN_CLASSES = ["Non-Suspicious", "Suspicious"]


class BinaryEvaluator(unittest.TestCase):
    def test_happy_path(self):
        code, res, out = run_eval(BINARY, BIN_GT, BIN_HDR, BIN_OK, BIN_CLASSES)
        self.assertEqual(code, 0, out)
        m = res["metrics"]
        self.assertEqual((m["tp"], m["tn"], m["fp"], m["fn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(m["sensitivity"], 0.5)
        self.assertAlmostEqual(m["fnr"], 0.5)
        self.assertAlmostEqual(m["fpr"], 0.5)
        self.assertEqual(res["num_samples"], 4)
        self.assertEqual(m["confusion_matrix"], [[1, 1], [1, 1]])

    def test_row_order_does_not_matter(self):
        _, a, _ = run_eval(BINARY, BIN_GT, BIN_HDR, BIN_OK, BIN_CLASSES)
        _, b, _ = run_eval(BINARY, BIN_GT, BIN_HDR, list(reversed(BIN_OK)), BIN_CLASSES)
        self.assertEqual(a["metrics"], b["metrics"])

    def test_single_class_ground_truth_gives_null_auc(self):
        gt = [["a.jpg", 1], ["b.jpg", 1]]
        preds = [["a.jpg", 1, 0.9], ["b.jpg", 0, 0.2]]
        code, res, out = run_eval(BINARY, gt, BIN_HDR, preds, BIN_CLASSES)
        self.assertEqual(code, 0, out)
        self.assertIsNone(res["metrics"]["auc"])
        self.assertEqual(res["metrics"]["specificity"], 0.0)  # 0/0 counts as 0

    def test_invalid_predictions_exit_12(self):
        cases = {
            "missing row": BIN_OK[:3],
            "duplicate row": BIN_OK + [["a.jpg", 0, 0.1]],
            "unknown file": BIN_OK + [["zzz.jpg", 0, 0.1]],
            "label 2": [["a.jpg", 2, 0.1]] + BIN_OK[1:],
            "score > 1": [["a.jpg", 0, 1.5]] + BIN_OK[1:],
            "score nan": [["a.jpg", 0, "nan"]] + BIN_OK[1:],
            "score text": [["a.jpg", 0, "high"]] + BIN_OK[1:],
        }
        for name, rows in cases.items():
            with self.subTest(name):
                code, _, out = run_eval(BINARY, BIN_GT, BIN_HDR, rows, BIN_CLASSES)
                self.assertEqual(code, 12, out)

    def test_missing_column_exit_12(self):
        code, _, out = run_eval(BINARY, BIN_GT, ["file", "label"], [r[:2] for r in BIN_OK], BIN_CLASSES)
        self.assertEqual(code, 12, out)

    def test_bad_spec_is_not_blamed_on_the_model(self):
        code, _, _ = run_eval(BINARY, BIN_GT, BIN_HDR, BIN_OK, ["only-one"])
        self.assertNotIn(code, (0, 12))


MC_CLASSES = ["A", "B", "C", "D"]
MC_HDR = ["file", "label", "prob_0", "prob_1", "prob_2", "prob_3"]
MC_GT = [["s1.dcm", 0], ["s2.dcm", 1], ["s3.dcm", 2], ["s4.dcm", 3], ["s5.dcm", 3]]
MC_OK = [
    ["s1.dcm", 0, 0.7, 0.1, 0.1, 0.1],
    ["s2.dcm", 1, 0.1, 0.7, 0.1, 0.1],
    ["s3.dcm", 3, 0.1, 0.1, 0.3, 0.5],
    ["s4.dcm", 3, 0.1, 0.1, 0.1, 0.7],
    ["s5.dcm", 2, 0.1, 0.1, 0.5, 0.3],
]


class MulticlassEvaluator(unittest.TestCase):
    def test_happy_path(self):
        code, res, out = run_eval(MULTI, MC_GT, MC_HDR, MC_OK, MC_CLASSES)
        self.assertEqual(code, 0, out)
        m = res["metrics"]
        self.assertAlmostEqual(m["accuracy"], 3 / 5)
        for key in ("macro_f1", "weighted_f1", "macro_f2", "weighted_f2", "sensitivity",
                    "specificity", "ppv", "npv", "qwk", "auc"):
            self.assertIn(key, m)
        self.assertEqual(len(m["confusion_matrix"]), 4)
        self.assertEqual(set(m["per_class"]), set(MC_CLASSES))

    def test_qwk_can_be_negative(self):
        gt = [["a", 0], ["b", 1], ["c", 2], ["d", 3]]
        rev = [["a", 3, 0, 0, 0, 1], ["b", 2, 0, 0, 1, 0], ["c", 1, 0, 1, 0, 0], ["d", 0, 1, 0, 0, 0]]
        code, res, out = run_eval(MULTI, gt, MC_HDR, rev, MC_CLASSES)
        self.assertEqual(code, 0, out)
        self.assertLess(res["metrics"]["qwk"], 0)

    def test_absent_class_gives_null_auc(self):
        gt = [["a", 0], ["b", 1], ["c", 2]]  # class D never appears
        preds = [["a", 0, 0.7, 0.1, 0.1, 0.1], ["b", 1, 0.1, 0.7, 0.1, 0.1], ["c", 2, 0.1, 0.1, 0.7, 0.1]]
        code, res, out = run_eval(MULTI, gt, MC_HDR, preds, MC_CLASSES)
        self.assertEqual(code, 0, out)
        self.assertIsNone(res["metrics"]["auc"])

    def test_invalid_predictions_exit_12(self):
        cases = {
            "missing row": MC_OK[:4],
            "duplicate row": MC_OK + [MC_OK[0]],
            "unknown file": MC_OK + [["x.dcm", 0, 1, 0, 0, 0]],
            "label out of range": [["s1.dcm", 4, 0.7, 0.1, 0.1, 0.1]] + MC_OK[1:],
            "probs do not sum to 1": [["s1.dcm", 0, 0.9, 0.9, 0.1, 0.1]] + MC_OK[1:],
            "negative prob": [["s1.dcm", 0, 1.2, -0.2, 0.0, 0.0]] + MC_OK[1:],
        }
        for name, rows in cases.items():
            with self.subTest(name):
                code, _, out = run_eval(MULTI, MC_GT, MC_HDR, rows, MC_CLASSES)
                self.assertEqual(code, 12, out)

    def test_missing_prob_column_exit_12(self):
        hdr = MC_HDR[:-1]
        code, _, out = run_eval(MULTI, MC_GT, hdr, [r[:-1] for r in MC_OK], MC_CLASSES)
        self.assertEqual(code, 12, out)


if __name__ == "__main__":
    unittest.main()
