# TANUH Processing TEE

GCP Confidential Space VM that receives encrypted jobs from the Buffer TEE, runs ONNX inference inside the TEE, and submits results to GCS and the leaderboard. Two variants exist — GPU and CPU — built from separate directories but sharing the same `enclave_manager_new.py` pipeline.

## Variants

| Variant | Directory | ONNX Runtime | Base Image | GCP VM |
|---|---|---|---|---|
| **GPU** | `Processing_TEE/` | `onnxruntime-gpu==1.19.2` (CUDAExecutionProvider) | `nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04` | `gpu-cs-tdx-h100` |
| **CPU** | `Processing_TEE_CPU/` | `onnxruntime==1.19.2` (CPUExecutionProvider only) | `nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04` | `cpu-cs-tdx` |

The CPU variant sets `NVIDIA_VISIBLE_DEVICES=none` and `CUDA_VISIBLE_DEVICES=-1` so the CUDA runtime is never initialised, avoiding the segfault that `onnxruntime-gpu` triggers on machines without a physical GPU.

The GPU variant uses `NVIDIA_VISIBLE_DEVICES=all` and links against `libcudart.so.12` / `libcudnn.so.9` from the CUDA 12.3 + cuDNN 9 base image.

## Architecture

```
Buffer TEE
  │  Encrypted job payload over RA-TLS  (:443)
  ▼
Go RA-TLS server  (b2p-ratls/cmd/gpu-cs/)
  │  Verifies Buffer TEE attestation + decrypts payload
  ▼
Flask enclave manager  :4000  (enclave_manager_new.py)
  │  Materializes model/weights/preprocessing artifacts
  │  Fetches + decrypts dataset from GCS
  │  Fetches eval script from GCS
  │  Runs ONNX inference
  │
  ├──▶  GCS  gs://p3dx-tanuh-results/results/:job_id/results.json
  └──▶  Leaderboard  POST https://benchmark.tanuh.ai/leaderboard/submit-solution
            (Bearer = caller's Keycloak JWT)
  │
  └──▶  stop-processing-vm.sh  (self-deallocates after job completes)
```

## Job Pipeline

1. Buffer TEE dispatches encrypted payload over RA-TLS → enclave manager accepts it on a background thread
2. Job artifacts (model, weights, optional preprocessing script) materialized under `/app/cvm_workflow/secure_jobs/:job_id/artifacts/`
3. Dataset fetched and decrypted from GCS → written to `runtime/dataset_<id>_decrypted.json`
4. Eval script fetched from GCS (`evaluation_script_<dataset_id>.py`) — **never modified by this repo**
5. ONNX model loaded; inference runs against all dataset samples
6. Results JSON written to `runtime/results.json`
7. Results uploaded to GCS and POSTed to the leaderboard with the user's Keycloak JWT as the Bearer token
8. `stop-processing-vm.sh` called → VM deallocates itself

## Components

| Path | Role |
|---|---|
| `enclave_manager_new.py` | Main pipeline — job materialisation, dataset fetch, inference, GCS upload, leaderboard submit |
| `Fetch_data/fetch_data.py` | GCS dataset and eval script download helpers |
| `Fetch_data/secrets.py` | GCP Secret Manager access + OAuth token helpers (with retry/backoff) |
| `b2p-ratls/` | RA-TLS server submodule (Go) |
| `stop-processing-vm.sh` | Self-deallocation script called after job completion or idle timeout |
| `entrypoint.sh` | Container entrypoint — starts Flask + RA-TLS server |

## Environment Variables

All runtime vars must be passed via GCP Confidential Space metadata with the `tee-env-` prefix.

| Variable | Default | Description |
|---|---|---|
| `RATLS_AUDIENCE` | `ratls-buffer-tee` | Expected audience in the Buffer TEE's OIDC token |
| `LISTEN_ADDR` | `:443` | Address the RA-TLS server listens on |
| `INTERNAL_ADDR` | `127.0.0.1:8081` | Internal address used by the RA-TLS server |
| `PROCESSING_MANAGER_JOB_URL` | `http://127.0.0.1:4000/enclave/cvm/secure-job` | Flask endpoint the RA-TLS server forwards jobs to |
| `PROCESSING_IDLE_TIMEOUT_SECONDS` | `300` | Seconds of inactivity before self-deallocation |
| `PROCESSING_DEALLOCATE_AFTER_JOB` | `1` | Set to `1` to deallocate immediately after each job |
| `PROJECT` | — | GCP project ID (used by stop script) |
| `ZONE` | — | GCP zone of this VM (used by stop script) |
| `INSTANCE` | — | GCP instance name of this VM (used by stop script) |

## Build & Deploy

### GPU variant

```bash
IMAGE=us-central1-docker.pkg.dev/p3dx-depa-sandbox/ratls/gpu-cs

cd Processing_TEE/
docker build -t $IMAGE:processing-tee-v<N> .
docker push $IMAGE:processing-tee-v<N>
```

### CPU variant

```bash
IMAGE=us-central1-docker.pkg.dev/p3dx-depa-sandbox/ratls/cpu-cs

cd Processing_TEE_CPU/
docker build -t $IMAGE:latest .
docker push $IMAGE:latest
```

Update the Buffer TEE's VM metadata so it dispatches to the new digest:
```bash
# CPU TEE
gcloud compute instances add-metadata cpu-cs-tdx --zone=us-central1-a \
  --metadata tee-image-reference=$IMAGE@sha256:<digest>

# GPU TEE
gcloud compute instances add-metadata gpu-cs-tdx-h100 --zone=us-central1-a \
  --metadata tee-image-reference=us-central1-docker.pkg.dev/p3dx-depa-sandbox/ratls/gpu-cs:processing-tee-v<N>
```

Also update `GPU_CS_IMAGE_DIGEST` / `CPU_CS_IMAGE_DIGEST` in the Buffer TEE VM metadata so the RA-TLS verification passes.

## Key Differences: GPU vs CPU

| | GPU (`Processing_TEE/`) | CPU (`Processing_TEE_CPU/`) |
|---|---|---|
| `requirements.txt` | `onnxruntime-gpu==1.19.2` + `torch==2.2.2` | `onnxruntime==1.19.2` (CPU-only) |
| `NVIDIA_VISIBLE_DEVICES` | `all` | `none` |
| `CUDA_VISIBLE_DEVICES` | *(unset)* | `-1` |
| ONNX provider | `CUDAExecutionProvider` → `CPUExecutionProvider` | `CPUExecutionProvider` only |
| `allow_env_override` | 6 vars | 9 vars (adds `PROJECT`, `ZONE`, `INSTANCE`) |
| GitHub branch | `main` | `cpu_tee` |

## Debug

Serial/container logs include:
- `[Google-CVM workflow]` — enclave manager lifecycle and job events
- `[evaluation_script]` — output from the GCS eval script during inference
- `gpu-cs:` — RA-TLS server events

Runtime state is persisted at `/app/cvm_workflow/` inside the container:
- `secure_jobs/:job_id/artifacts/` — model, weights, preprocessing script
- `secure_jobs/:job_id/runtime/results.json` — inference results
- `runtime_state.json` — current VM status (idle / running / complete / deallocation_requested)
