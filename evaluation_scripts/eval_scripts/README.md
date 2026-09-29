# Integration templates

These two files are **templates**, not the scripts actually deployed to
`gs://tanuh-eval-scripts/`. The real `evaluate_model_breastcancer.py` and
`evaluate_model_OCS.py` include model-specific inference code (DICOM
windowing, image resizing/normalization, the exact ONNX input tensor shape
each submitted model expects) that lives only on the machine that originally
built those scripts — none of that is checked into this repo, so it can't be
reproduced here without guessing.

What *is* fully specified, and what these templates show, is everything
downstream of inference: given `y_true`/`y_pred` (and optionally per-sample
scores) for a dataset, compute the metrics dict and write it into
`results.json` in exactly the shape an `enclave_manager_new.py`-style
`_leaderboard_metrics()` (reads specific keys out of `results["metrics"]`
per dataset_id) would expect — see `run_evaluation_script_from_paths()`
(invokes this script as `--model --dataset --results [--preprocessing]`).

Note: this `--model`/`--dataset`/`enclave_manager_new.py` architecture
predates the current Go-based Processing TEE in this repo, which instead
runs a bucket-generic `evaluate.py` (see `../../evaluation_LEGACY/` and
`../README.md`'s "Relationship to TANUH-processingtee" section). These
templates are kept as reference/standalone tooling, not as something wired
into the live pipeline.

To turn a template into the real deployed script:

1. Keep the CLI contract (`--model`, `--dataset`, `--results`,
   `--preprocessing`) and the `results.json` shape as-is.
2. Replace `run_inference(...)` with real ONNX Runtime inference against the
   decrypted dataset JSON (paths + labels — see
   `TANUH-processingtee/tools/prepare_and_upload_datasets.py` for the
   `dicom_paths`/`image_paths` + `labels` shape), including whatever
   preprocessing the submitted model needs.
3. Leave everything from `eval_lib.evaluate(...)` onward untouched — that's
   where dataset-specific metric selection (binary vs multiclass) and the
   exact output key names are enforced.
4. Upload the finished script to `gs://tanuh-eval-scripts/` under the object
   name configured in `TANUH-processingtee/config.yml`'s `dataset_map`.
