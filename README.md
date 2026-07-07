# TANUH Processing TEE

GCP Confidential Space VM that receives secure jobs from the Buffer TEE over RA-TLS, runs ONNX inference inside the TEE, submits results to the external leaderboard, notifies the buffer, and deallocates itself. One codebase builds both the GPU and CPU images.

The runtime is a **single Go binary** (`processing-tee`). Python exists in the image only for the evaluation scripts fetched from `gs://tanuh-eval-scripts` at job time, executed as a subprocess.

## Architecture

```
Buffer TEE
  │  secure job payload over RA-TLS  (POST /api/load-model, :443)
  ▼
processing-tee (Go, single process)
  ├─ startup: network-policy attestation (policy hash bound into eat_nonce)
  ├─ RA-TLS server  (/ratls/connect: OIDC token, EKM channel binding)
  ├─ payload materialisation  (base64 + SHA-256 fail-closed verification)
  ├─ dataset fetch (GCS) + AES-256-GCM decrypt (key from Secret Manager)
  ├─ eval script fetch (GCS) → python3 subprocess   ← the only Python
  ├─ leaderboard POST  (Bearer = caller's Keycloak JWT; attestation claims in body)
  ├─ completion callback → buffer  POST {buffer_job_url}/complete
  └─ self-deallocation  (Compute API stop; after job + idle timeout)
```

There is no Flask layer, no localhost hop, and no app-layer payload
encryption: confidentiality in transit is the RA-TLS channel; integrity is
the browser's SHA-256 commitment re-verified before eval.

## Eval contract (frozen)

```
python3 evaluation_script_<dataset_id>.py --model M --dataset D --results R [--preprocessing P]
```

Exit codes: `10` = user preprocessing failure, `11` = CUDA/GPU environment
failure, other non-zero = script error. This is the versioned API between the
Go pipeline and the Python eval world.

## Layout

| Path | Role |
|---|---|
| `cmd/processing-tee/` | main: policy attestation → RA-TLS server → serve |
| `internal/ratls/` | RA-TLS server, EKM nonce (dependency-free audit surface) |
| `internal/pipeline/` | job lifecycle: materialise → dataset → eval → report → dealloc |
| `internal/gcp/` | stdlib REST: metadata tokens, GCS, Secret Manager, Compute stop |
| `internal/crypto/` | dataset AES-256-GCM formats (small blob + chunked stream) |
| `internal/leaderboard/` | metrics mapping, uuid5 job ids, error classification, submit |
| `internal/attest/` | CS launcher token fetch + claims decode |
| `internal/policy/` | startup network-policy attestation (Python-compatible hash) |
| `internal/eval/` | evaluation subprocess runner |
| `policy/network_policy.json` | the attested network policy (data, ships in image) |
| `tools/` | dataset preparation/upload (dev-only, not shipped) |

go.mod has **zero external dependencies**.

## Environment Variables

All runtime vars are passed via Confidential Space metadata with the `tee-env-` prefix.

| Variable | Default | Description |
|---|---|---|
| `RATLS_AUDIENCE` | `ratls-buffer-tee` | Audience for this TEE's attestation tokens |
| `LISTEN_ADDR` | `:443` | RA-TLS listen address |
| `PROCESSING_IDLE_TIMEOUT_SECONDS` | `300` | Idle seconds before self-deallocation |
| `PROCESSING_DEALLOCATE_AFTER_JOB` | `1` | Deallocate after each job (success or failure) |
| `PROCESSING_EVAL_TIMEOUT_SECONDS` | `3600` | Hard cap on the eval subprocess (0 disables) |
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
- `pipeline:` — job lifecycle and state transitions
- `[evaluation_script]` — eval subprocess output
- `ratls/server:` — RA-TLS connect events
- `policy:` / `POLICY STARTUP ATTESTATION` — boot-time policy evidence

Runtime state persists under `/app/cvm_workflow/`:
- `secure_jobs/:job_id/artifacts/` — model, weights, preprocessing script
- `secure_jobs/:job_id/runtime/results.json` — inference results
- `runtime/secure_runtime_state.json` — current status (waiting_for_job / running / complete / error / deallocation_requested)
