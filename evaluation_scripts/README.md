# TANUH Evaluation Scripts

Computes benchmark metrics for TANUH model submissions from model outputs
(predictions) + ground truth. Evaluation is **dataset-specific and
problem-statement-specific**: `eval_lib/datasets.py` is a small registry
mapping each dataset to its problem type (binary or multiclass) and its
label space, so the same metrics engine (`eval_lib/binary.py`,
`eval_lib/multiclass.py`) produces the right metric set for each.

Currently registered:

| Dataset | Problem statement | Classes | Notes |
|---|---|---|---|
| `breast_cancer` | multiclass (ordinal) | 1–4 (BI-RADS tissue density) | QWK is meaningful since classes are ordered |
| `oral_cancer` | binary | Suspicious / Non-Suspicious | dataset_id=2, "OCS" |

## Layout

```
eval_lib/
  common.py      # safe_div, f_beta — shared numeric helpers
  binary.py       # compute_binary_metrics(y_true, y_pred, y_score=None)
  multiclass.py    # compute_multiclass_metrics(y_true, y_pred, y_score=None, labels=None)
  datasets.py       # DatasetConfig registry + label coercion (e.g. "Suspicious" -> 1)
  io_utils.py        # CSV loading (single-file or paired ground-truth/predictions)
  engine.py           # evaluate() / evaluate_from_files() — the top-level API
cli.py                 # standalone CLI wrapping engine.evaluate_from_files
eval_scripts/           # templates showing how the TEE-deployed per-dataset
                        # scripts should call into eval_lib (see its README)
tests/                   # pytest suite, cross-checked against sklearn and
                         # against enclave_manager_new.py's own fallback logic
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Single CSV with both ground-truth and predictions columns (notes.md format)
python cli.py --dataset oral_cancer --input predictions.csv

# Model outputs and ground truth as two separate files, joined by id
python cli.py --dataset breast_cancer \
  --ground-truth ground_truth.csv --predictions model_outputs.csv \
  --output results.json

python -m pytest tests/ -v
```

Programmatic use:

```python
from eval_lib import evaluate

metrics = evaluate(
    "oral_cancer",
    y_true_raw=["Suspicious", "Non-Suspicious", "Suspicious"],
    y_pred_raw=["Suspicious", "Non-Suspicious", "Non-Suspicious"],
)
```

## CSV formats

Matches `notes.md`. Either:

- **One file**, with a ground-truth column and a predictions column
  side by side (`id,ground_truth,predictions` or
  `sample_id,true_rating,predicted_rating`) — use `--input`.
- **Two files** — a ground-truth reference file and a model-outputs file
  from a separate inference run — joined by an `id` column. Use
  `--ground-truth` + `--predictions`.

Column names are auto-detected from common aliases (`ground_truth`,
`true_rating`, `label`, ... / `predictions`, `predicted_rating`,
`prediction`, ...); pass `--gt-col`/`--pred-col`/`--id-col`/`--score-col`
explicitly if a file uses something else. An optional score/probability
column enables AUC.

String labels (e.g. `"Suspicious"`/`"Non-Suspicious"`) are coerced to the
dataset's canonical label space via `DatasetConfig.label_map` — see
`eval_lib/datasets.py`.

## Output contract

The metrics dict is shaped to match exactly what
`TANUH-processingtee/enclave_manager_new.py`'s `_leaderboard_metrics()`
reads out of `results["metrics"]` per dataset:

- **binary** (`oral_cancer`, dataset_id=2): `sensitivity`, `specificity`,
  `accuracy`, `ppv`, `npv`, `f2` (not `f2_score` — that renaming happens on
  the leaderboard side).
- **multiclass** (`breast_cancer`, dataset_id=1): `accuracy`, `macro_f2`,
  `weighted_f2`, `macro_f1`, `macro_recall`, `qwk`, `macro_specificity`,
  `macro_npv`, `macro_ppv`, `confusion_matrix`, `auc`, and a `per_class`
  breakdown (keyed by stringified class value) with `TP`/`FP`/`FN`/`TN`/
  `precision`/`recall` — the enclave manager falls back to deriving
  `weighted_f2` from exactly those `per_class` keys if it's ever missing, so
  don't rename them without updating that fallback too.

`tests/test_engine_and_cli.py::test_per_class_output_is_compatible_with_enclave_managers_weighted_f2_fallback`
guards this cross-repo contract by reimplementing that fallback and checking
it agrees with our own `weighted_f2`.

## Adding a new dataset or problem statement

1. Add a `DatasetConfig` to `eval_lib/datasets.py` (`problem_type`,
   `label_map`, `class_values` for multiclass).
2. Register it in `DATASET_REGISTRY`.
3. If it needs a new problem statement beyond binary/multiclass, add a
   `compute_*_metrics()` function alongside `binary.py`/`multiclass.py` and
   branch to it in `eval_lib/engine.py::evaluate()`.

No changes needed to `cli.py` or `io_utils.py` — both are dataset-agnostic.

## Relationship to TANUH-processingtee

This describes the older, dataset-registry-keyed architecture
(`evaluate_model_*.py --model M --dataset D --results R`, scripts fetched
from `gs://tanuh-eval-scripts/`, results read by an `enclave_manager_new.py`)
that predates the current Go-based Processing TEE and is **not** what it
runs today. The live pipeline instead runs a bucket-generic (not
dataset-specific) `evaluate.py` per problem type
(`binary_classification`/`multiclass_classification`), fetched from
`gs://tanuh-evaluators/<bucket>/evaluate.py` and invoked as
`evaluate.py --predictions P --ground-truth G --spec S --results R` — see
`internal/evaluator/`, `internal/eval/` and `internal/pipeline/pipeline.go`
in this repo, and `docs/benchmarking-flow.md`. That current bucket evaluator
lives in [`../evaluation_LEGACY/`](../evaluation_LEGACY/) (kept, unused by
this directory's code, purely so it isn't deleted).

This directory (`eval_lib`, `cli.py`, `eval_scripts/`, `tests/`) is kept as
standalone tooling — e.g. for offline metric computation from a CSV of
predictions vs. ground truth — and is not wired into the Go pipeline.
