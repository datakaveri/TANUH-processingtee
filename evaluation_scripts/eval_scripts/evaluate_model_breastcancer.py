#!/usr/bin/env python3
"""Template for the breast-cancer (dataset_id=1) eval script.

See eval_scripts/README.md — this is a template, not the deployed script.
Everything from `eval_lib.evaluate(...)` down is the real, tested
integration point; `run_inference()` is a placeholder for the actual
DICOM-preprocessing + ONNX inference logic, which isn't available in this
repo.

CLI contract matches TANUH-processingtee/enclave_manager_new.py's
`run_evaluation_script_from_paths()`:
    evaluate_model_breastcancer.py --model M --dataset D --results R [--preprocessing P]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_lib import evaluate  # noqa: E402

DATASET_NAME = "breast_cancer"


def debug(message: str) -> None:
    print(f"[evaluate_model_breastcancer] {message}", flush=True)


def run_inference(session: ort.InferenceSession, dicom_paths: list[str]) -> tuple[list[int], list[list[float]]]:
    """Placeholder: replace with real DICOM loading/windowing + batched inference.

    Must return (predictions, scores) where predictions are class values in
    {1, 2, 3, 4} (matching eval_lib.datasets.BREAST_CANCER.class_values) and
    scores is a per-sample list of 4 class probabilities in that same [1,2,3,4]
    column order (pass scores=None to skip AUC if the model has no calibrated
    probability output).
    """
    raise NotImplementedError(
        "Wire up real DICOM preprocessing + ONNX inference here — see eval_scripts/README.md"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--preprocessing", required=False)
    args = parser.parse_args()

    started = time.time()
    model_path = Path(args.model)
    dataset_path = Path(args.dataset)
    results_path = Path(args.results)

    debug(f"Loading decrypted dataset from {dataset_path}")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dicom_paths = dataset["dicom_paths"]
    labels = dataset["labels"]
    debug(f"dataset_id={dataset.get('dataset_id')} num_samples={len(labels)}")

    session = ort.InferenceSession(str(model_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    predictions, scores = run_inference(session, dicom_paths)

    metrics = evaluate(DATASET_NAME, y_true_raw=labels, y_pred_raw=predictions, y_score=scores)

    results = {
        "status": "success",
        "dataset_id": dataset.get("dataset_id"),
        "dataset_description": dataset.get("description"),
        "num_samples": len(labels),
        "num_classes": dataset.get("num_classes"),
        "metrics": metrics,
        "onnx_runtime_providers": session.get_providers(),
        "elapsed_seconds": round(time.time() - started, 4),
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    debug(f"Results written to {results_path}")


if __name__ == "__main__":
    main()
