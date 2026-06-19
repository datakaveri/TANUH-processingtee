#!/usr/bin/env python3
"""
Local end-to-end test for the Processing TEE pipeline.

Reads model.onnx + model.onnx.data from a known location, computes SHA-256,
base64-encodes them, and POSTs a synthetic secure-job payload to the running
enclave_manager_new.py Flask server (must be started separately).

Usage:
    # In one terminal: python enclave_manager_new.py
    # In another:      python test_local_e2e.py
"""

import base64
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

import requests

MODEL_PATH = Path("/home/azureuser/TANUH machines/Buffer TEE/cvm_workflow/buffer/shared_artifacts/model.onnx")
WEIGHTS_PATH = Path("/home/azureuser/TANUH machines/Buffer TEE/cvm_workflow/buffer/shared_artifacts/model.onnx.data")
ENCLAVE_URL = "http://127.0.0.1:4000"
DATASET_ID = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def b64_file(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: model not found at {MODEL_PATH}")
        sys.exit(1)
    if not WEIGHTS_PATH.exists():
        print(f"ERROR: weights not found at {WEIGHTS_PATH}")
        sys.exit(1)

    print(f"Reading model  ({MODEL_PATH.stat().st_size / 1024:.1f} KB) ...")
    print(f"Reading weights ({WEIGHTS_PATH.stat().st_size / 1024:.1f} KB) ...")

    model_sha256   = sha256_file(MODEL_PATH)
    weights_sha256 = sha256_file(WEIGHTS_PATH)
    model_b64      = b64_file(MODEL_PATH)
    weights_b64    = b64_file(WEIGHTS_PATH)

    job_id = f"test-{uuid.uuid4().hex[:8]}"
    payload = {
        "job_id": job_id,
        "dataset_id": DATASET_ID,
        "model_onnx_base64": model_b64,
        "model_weights_base64": weights_b64,
        "model_sha256": model_sha256,
        "weights_sha256": weights_sha256,
        # Local test: callback to a stub that just accepts POSTs (httpbin or similar).
        # Replace with real Buffer TEE callback URL when testing against live infra.
        "buffer_results_callback_url": f"{ENCLAVE_URL}/enclave/cvm/results-stub",
    }

    print(f"\nJob ID        : {job_id}")
    print(f"model_sha256  : {model_sha256}")
    print(f"weights_sha256: {weights_sha256}")
    print(f"\nPOSTing to {ENCLAVE_URL}/enclave/cvm/secure-job ...")

    resp = requests.post(
        f"{ENCLAVE_URL}/enclave/cvm/secure-job",
        json=payload,
        timeout=30,
    )
    print(f"Response: HTTP {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))

    if resp.status_code not in (200, 202):
        print("\nFAILED — unexpected status code")
        sys.exit(1)

    print("\nJob accepted. Polling runtime state...")
    for _ in range(120):
        time.sleep(3)
        state_resp = requests.get(f"{ENCLAVE_URL}/enclave/cvm/runtime-state", timeout=10)
        state = state_resp.json().get("runtime_state", {})
        status = state.get("status", "?")
        print(f"  status={status}", end="\r", flush=True)
        if status in ("complete", "error", "deallocation_requested", "deallocation_pending"):
            print()
            break
    else:
        print("\nTimeout waiting for job completion")
        sys.exit(1)

    if state.get("status") == "complete":
        print("\n=== Job complete ===")
        results_resp = requests.get(f"{ENCLAVE_URL}/enclave/cvm/results", timeout=10)
        print(json.dumps(results_resp.json(), indent=2))
    else:
        print(f"\nJob ended with status: {state.get('status')} error: {state.get('last_error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
