# TANUH Processing TEE

GCP Confidential Space VM that receives secure jobs from the Buffer TEE over RA-TLS, runs a submitted model against an encrypted dataset inside the TEE, scores it with the dataset's problem-bucket evaluator, submits results to the external leaderboard, notifies the buffer, and deallocates itself. One codebase builds both the GPU and CPU images.

The runtime is a **single Go binary** (`processing-tee`). Python runs three stages of each job as subprocesses (below).

## Architecture

```
Buffer TEE
  │  secure job payload v2 over RA-TLS  (POST /api/load-model, :443)
  ▼
processing-tee (Go, single process)
  ├─ startup: network-policy attestation (policy hash bound into eat_nonce)
  ├─ RA-TLS server  (/ratls/connect: OIDC token, EKM channel binding)
  ├─ payload v2 materialisation  (SHA-256 fail-closed; model laid out by format)
  ├─ catalogue entry (cat/item)  → problem bucket, class_names, required metrics
  ├─ evaluator fetch  gs://tanuh-evaluators/<bucket>/evaluate.py  (sha256 + generation recorded)
  ├─ adaptor dependency install  (dep_scanner.py + uv)
  ├─ dataset: one tanuh-enc-dataset-v1 object per file → KMS unwrap + AES-GCM decrypt, 16 in parallel
  ├─ stages:  infer.py → adaptor.py → evaluate.py
  ├─ metrics mapped to the catalogue's keys → leaderboard POST (Bearer = caller's Keycloak JWT)
  ├─ completion callback → buffer  POST {buffer_job_url}/complete
  └─ self-deallocation  (Compute API stop; after job + idle timeout)
```

There is no Flask layer, no localhost hop, and no app-layer payload
encryption: confidentiality in transit is the RA-TLS channel; integrity is
the browser's SHA-256 commitment, re-verified by the buffer and again here.

## Job stages

A model can be **ONNX** (`model.onnx` + optional `model.onnx.data`),
**TorchScript** (`model.pt` carrying `extra/tanuh.json`) or **Hugging Face**
(a zip of a `save_pretrained()` folder with safetensors weights). The platform
prepares every input to the model's declared shape, so there is no
preprocessing script; model-specific normalisation lives inside the model.
`tools/check_model.py` runs the same checks locally.

```
python3 /app/infer.py --format F --model-dir M --inputs inputs.txt --output-dir raw/
python3 adaptor.py    --raw-dir raw/ --spec dataset_spec.json --output-dir predictions/
python3 evaluate.py   --predictions predictions/predictions.csv --ground-truth ground_truth.csv \
                      --spec dataset_spec.json --results results.json
```

| Exit | Stage | Meaning | Leaderboard error |
|---|---|---|---|
| 11 | infer | CUDA/GPU runtime failure | 3 `CudaError` |
| 13 | infer | model failed to load/run, or unsupported input | 2 `ModelError` |
| 14 | infer | a dataset file could not be decoded | 1 `DatasetDecodeError` |
| non-zero | adaptor | adaptor failed | 1 `AdaptorError` |
| 12 | evaluator | predictions.csv breaks the bucket format | 1 `PredictionsInvalidError` |
| other | evaluator | evaluator failed | 3 `EvaluatorError` |

Leaderboard error codes: 1 = data loading / pre- or post-processing,
2 = model loading, 3 = container/runtime. A result whose metrics cannot
satisfy the dataset's catalogue definitions is submitted as failed
(`MetricMappingError` / `MetricOutOfRangeError`) rather than rejected.

## Layout

| Path | Role |
|---|---|
| `cmd/processing-tee/` | main: policy attestation → RA-TLS server → serve |
| `internal/ratls/` | RA-TLS server, EKM nonce (dependency-free audit surface) |
| `internal/pipeline/` | job lifecycle: materialise → catalogue → evaluator → dataset → stages → report → dealloc |
| `internal/catalogue/` | catalogue entry: bucket, class_names, required leaderboard metrics |
| `internal/evaluator/` | fetch a bucket's evaluate.py (GCS now, image later) |
| `internal/modelpkg/` | lay out + validate a model by format (ONNX / TorchScript / Hugging Face) |
| `internal/groundtruth/` | parse ground_truth.csv, match it to the data files |
| `internal/eval/` | Python stage runner + adaptor dependency install |
| `internal/gcp/` | stdlib REST: metadata tokens, GCS, KMS, Secret Manager, Compute stop |
| `internal/crypto/` | dataset AES-256-GCM formats (tanuh-enc-dataset-v1 + legacy) |
| `internal/leaderboard/` | catalogue-keyed metrics, uuid5 job ids, error classification, submit |
| `internal/attest/` | CS launcher token fetch + claims decode |
| `internal/policy/` | startup network-policy attestation (Python-compatible hash) |
| `policy/network_policy.json` | the attested network policy (data, ships in image) |
| `tools/infer/infer.py` | platform inference (stage 1) — ships as `/app/infer.py` |
| `evaluation_LEGACY/<bucket>/evaluate.py` | bucket evaluators — uploaded to `gs://tanuh-evaluators/<bucket>/` |
| `tools/check_model.py`, `tools/reference/` | model-provider check, reference adaptors, model wrappers (not shipped) |

go.mod has **zero external dependencies**.

## Environment Variables

All runtime vars are passed via Confidential Space metadata with the `tee-env-` prefix.

| Variable | Default | Description |
|---|---|---|
| `RATLS_AUDIENCE` | `ratls-buffer-tee` | Audience for this TEE's attestation tokens |
| `LISTEN_ADDR` | `:443` | RA-TLS listen address |
| `PROCESSING_IDLE_TIMEOUT_SECONDS` | `300` | Idle seconds before self-deallocation |
| `PROCESSING_DEALLOCATE_AFTER_JOB` | `1` | Deallocate after each job (success or failure) |
| `PROCESSING_INFER_TIMEOUT_SECONDS` | `3600` | Cap on the inference stage (0 disables) |
| `PROCESSING_ADAPTOR_TIMEOUT_SECONDS` | `600` | Cap on the adaptor stage |
| `PROCESSING_EVALUATOR_TIMEOUT_SECONDS` | `600` | Cap on the evaluator stage |
| `PROCESSING_DEPS_TIMEOUT_SECONDS` | `600` | Cap on the adaptor dependency install |
| `PROJECT` / `ZONE` / `INSTANCE` | sandbox/us-central1-a/gpu-cs-tdx-h100 | Self-stop target — **set INSTANCE per VM** (`cpu-cs-tdx` on the CPU VM) |
| `LEADERBOARD_SUBMIT_URL` | benchmark.tanuh.ai/leaderboard/submit-solution | Leaderboard endpoint |

## Build & Deploy

One codebase → two images via the `ONNXRUNTIME_PKG` build arg:

```bash
# GPU image (default: onnxruntime-gpu)
IMAGE=us-central1-docker.pkg.dev/p3dx-depa-sandbox/ratls/gpu-cs
docker build -t $IMAGE:<tag> .
docker push $IMAGE:<tag>

# CPU image (CPU-only onnxruntime — cannot segfault on a GPU-less VM)
IMAGE=us-central1-docker.pkg.dev/p3dx-depa-sandbox/ratls/cpu-cs
docker build --build-arg ONNXRUNTIME_PKG=onnxruntime==1.19.2 -t $IMAGE:<tag> .
docker push $IMAGE:<tag>
```

Point the VMs at the new digests and update the Buffer TEE's expected digests
(RA-TLS pins them):

```bash
gcloud compute instances add-metadata cpu-cs-tdx --zone=us-central1-a \
  --metadata tee-image-reference=<cpu image@sha256:digest>
gcloud compute instances add-metadata gpu-cs-tdx-h100 --zone=us-central1-a \
  --metadata tee-image-reference=<gpu image@sha256:digest>
gcloud compute instances add-metadata buffer-tee-vm-tdx --zone=us-east1-c \
  --metadata tee-env-CPU_CS_IMAGE_DIGEST=sha256:<cpu>,tee-env-GPU_CS_IMAGE_DIGEST=sha256:<gpu>
```

## Debug

Serial/container logs include:
- `pipeline:` — job lifecycle, stage timings and state transitions
- `eval:` — stage start/finish; `[infer]`, `[adaptor]`, `[evaluator:*]` — stage output
- `ratls/server:` — RA-TLS connect events
- `policy:` / `POLICY STARTUP ATTESTATION` — boot-time policy evidence

Runtime state persists under `/app/cvm_workflow/`:
- `secure_jobs/:job_id/artifacts/model/` — the model, laid out by format; `artifacts/scripts/adaptor.py`
- `secure_jobs/:job_id/runtime/` — `inputs.txt`, `dataset_spec.json`, `raw/`, `predictions/`, `results.json`
  (`runtime/dataset/` holds the decrypted dataset and is deleted when the job ends)
- `runtime/secure_runtime_state.json` — current status (waiting_for_job / running / complete / error / deallocation_requested)
