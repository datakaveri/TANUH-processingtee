#!/bin/sh
set -eu

export BASE_DIR="${BASE_DIR:-/app}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export RATLS_AUDIENCE="${RATLS_AUDIENCE:-ratls-buffer-tee}"
export LISTEN_ADDR="${LISTEN_ADDR:-:443}"
export PROCESSING_MANAGER_JOB_URL="${PROCESSING_MANAGER_JOB_URL:-http://127.0.0.1:4000/enclave/cvm/secure-job}"

mkdir -p \
  "$BASE_DIR/cvm_workflow/artifacts" \
  "$BASE_DIR/cvm_workflow/incoming" \
  "$BASE_DIR/cvm_workflow/runtime" \
  "$BASE_DIR/cvm_workflow/secure_jobs"

if [ -f /app/stop-processing-vm.sh ]; then
  chmod +x /app/stop-processing-vm.sh
fi

/usr/local/bin/gpu-cs &
GPU_CS_PID=$!

cleanup() {
  kill "$GPU_CS_PID" 2>/dev/null || true
}

trap cleanup INT TERM EXIT

cd /app

python policy/startup_policy_attestation.py

exec python enclave_manager_new.py
