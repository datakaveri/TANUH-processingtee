#!/usr/bin/env python3
"""Template for the oral-cancer-screening (dataset_id=2) eval script.

See eval_scripts/README.md — this is a template, not the deployed script.
Everything from `eval_lib.evaluate(...)` down is the real, tested
integration point; `run_inference()` is a placeholder for the actual image
preprocessing + ONNX inference logic, which isn't available in this repo.
This dataset also supports an optional user-submitted `preprocessing.py`
(see enclave_manager_new.py's `preprocessing_path` handling) — load and
apply it before inference if present.

CLI contract matches TANUH-processingtee/enclave_manager_new.py's
`run_evaluation_script_from_paths()`:
    evaluate_model_OCS.py --model M --dataset D --results R [--preprocessing P]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_lib import evaluate  # noqa: E402

DATASET_NAME = "oral_cancer"


def debug(message: str) -> None:
    print(f"[evaluate_model_OCS] {message}", flush=True)


def run_inference(
    session: ort.InferenceSession, image_paths: list[str], preprocessing_path: str | None
) -> tuple[list[int], list[float]]:
    """Placeholder: replace with real image loading + batched inference.

    Must return (predictions, scores) where predictions are 0 (Non-Suspicious)
    / 1 (Suspicious) and scores is the per-sample predicted probability of the
    positive (Suspicious) class (pass scores=None to skip AUC).
    """
    raise NotImplementedError(
        "Wire up real image preprocessing + ONNX inference here — see eval_scripts/README.md"
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
    image_paths = dataset["image_paths"]
    labels = dataset["labels"]
    debug(f"dataset_id={dataset.get('dataset_id')} num_samples={len(labels)}")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    predictions, scores = run_inference(session, image_paths, args.preprocessing)

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
