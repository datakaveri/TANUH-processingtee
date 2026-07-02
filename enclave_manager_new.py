import base64
import hashlib
import http.client
import os
import json
import socket
import uuid
import platform
import shutil
import sys
import subprocess
import time
import logging
import threading
import traceback
from pathlib import Path

_LOCAL_VENDOR = Path(__file__).resolve().parent / ".vendor"
if _LOCAL_VENDOR.exists() and str(_LOCAL_VENDOR) not in sys.path:
    sys.path.append(str(_LOCAL_VENDOR))

from flask import Flask, jsonify, Response, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import requests
from lib.config import config
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from Fetch_data.secrets import fetch_secret, get_oidc_token
from Fetch_data.gcs_fetch import download_gcs_object


app = Flask(__name__)

# Enable CORS for all routes with configuration from config.yml
CORS(app,
     resources={
         r"/*": {
             "origins": config.cors.origins,
             "methods": config.cors.methods,
             "allow_headers": config.cors.allow_headers,
             "expose_headers": config.cors.expose_headers,
             "supports_credentials": config.cors.supports_credentials,
             "max_age": config.cors.max_age
         }
     },
     supports_credentials=True)


# Default state when application is not running
state = {
    "step": 0,
    "maxSteps": 11,
    "title": "Inactive",
    "description": "Inactive",
}


# Flag to track if application is running
is_app_running = False

# Google Confidential VM placeholder workflow paths.
CVM_WORKFLOW_DIR = Path(config.base_dir) / "cvm_workflow"
CVM_ARTIFACTS_DIR = CVM_WORKFLOW_DIR / "artifacts"
CVM_INCOMING_DIR = CVM_WORKFLOW_DIR / "incoming"
CVM_RUNTIME_DIR = CVM_WORKFLOW_DIR / "runtime"
CVM_EVALUATION_SCRIPT = CVM_WORKFLOW_DIR / "evaluation_script.py"
CVM_SECURE_JOBS_DIR = CVM_WORKFLOW_DIR / "secure_jobs"
CVM_SECURE_STATE_PATH = CVM_RUNTIME_DIR / "secure_runtime_state.json"
MODEL_FILE_NAME = "model.onnx"
WEIGHTS_FILE_NAME = "model.onnx.data"
PROCESSING_IDLE_TIMEOUT_SECONDS = int(os.getenv("PROCESSING_IDLE_TIMEOUT_SECONDS", "300"))
PROCESSING_VM_STOP_SCRIPT = os.getenv(
    "PROCESSING_VM_STOP_SCRIPT",
    str(Path(config.base_dir) / "stop-processing-vm.sh"),
)
PROCESSING_VM_STOP_COMMAND = os.getenv("PROCESSING_VM_STOP_COMMAND", "")
PROCESSING_DEALLOCATE_AFTER_JOB = os.getenv("PROCESSING_DEALLOCATE_AFTER_JOB", "1") == "1"
SECURE_JOB_LOCK = threading.Lock()
SECURE_JOB_STATE = {
    "status": "waiting_for_job",
    "last_activity_unix": int(time.time()),
    "current_job_id": "",
    "last_job_id": "",
    "last_error": "",
    "deallocation_requested": False,
}

# Placeholder verifier endpoint. In production this should be the remote verifier
# endpoint that receives the SEV-SNP attestation report from this Google CVM.
ATTESTATION_VERIFIER_ENDPOINT_PLACEHOLDER = "https://verifier.placeholder.example/attestation/report"

# Real confirmation receiver should POST the confirmation payload here:
# POST http://<cvm-enclave-manager-host>:4000/enclave/cvm/confirmation
CONFIRMATION_FILE_PLACEHOLDER = CVM_INCOMING_DIR / "confirmation.json"


def cvm_debug(message):
    """Emit a flush-safe debug line for the Google CVM workflow."""
    print(f"[Google-CVM workflow] {message}", flush=True)


def _ensure_vendor_path():
    """Prefer local runtime wheels installed under .vendor for ONNX testing."""
    vendor_path = Path(config.base_dir) / ".vendor"
    if vendor_path.exists() and str(vendor_path) not in sys.path:
        sys.path.append(str(vendor_path))


def _json_dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def _json_load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_cvm_dirs():
    for directory in (
        CVM_WORKFLOW_DIR,
        CVM_ARTIFACTS_DIR,
        CVM_INCOMING_DIR,
        CVM_RUNTIME_DIR,
        CVM_SECURE_JOBS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def _secure_runtime_state():
    if CVM_SECURE_STATE_PATH.exists():
        return _json_load(CVM_SECURE_STATE_PATH)
    return dict(SECURE_JOB_STATE)


def _write_secure_runtime_state():
    _json_dump(CVM_SECURE_STATE_PATH, SECURE_JOB_STATE)


def _set_secure_job_state(**updates):
    SECURE_JOB_STATE.update(updates)
    SECURE_JOB_STATE["last_activity_unix"] = int(time.time())
    _write_secure_runtime_state()
    cvm_debug(f"Secure runtime state updated: {SECURE_JOB_STATE}")
    return dict(SECURE_JOB_STATE)


def _deallocation_command():
    if PROCESSING_VM_STOP_COMMAND.strip():
        return PROCESSING_VM_STOP_COMMAND.strip(), "command"

    candidate = Path(PROCESSING_VM_STOP_SCRIPT)
    if candidate.exists():
        return str(candidate), "script"

    return "", ""


def request_vm_deallocation(reason):
    if SECURE_JOB_STATE.get("deallocation_requested"):
        cvm_debug(f"VM deallocation already requested earlier; skipping duplicate request ({reason})")
        return False

    target, mode = _deallocation_command()
    if not target:
        cvm_debug(
            f"No VM stop command/script configured; would deallocate Processing TEE now because: {reason}"
        )
        _set_secure_job_state(
            status="deallocation_pending",
            deallocation_requested=True,
            last_error=f"stop command missing: {reason}",
        )
        return False

    cvm_debug(f"Requesting Processing TEE deallocation via {mode}: {target} ({reason})")
    try:
        if mode == "command":
            completed = subprocess.run(
                target,
                cwd=config.base_dir,
                capture_output=True,
                text=True,
                timeout=60,
                shell=True,
                check=False,
            )
        else:
            completed = subprocess.run(
                [target],
                cwd=config.base_dir,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
    except Exception as exc:
        cvm_debug(f"Processing VM stop invocation failed: {exc}")
        _set_secure_job_state(
            status="deallocation_failed",
            deallocation_requested=True,
            last_error=f"stop invocation failed: {exc}",
        )
        return False

    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, end="", flush=True)

    if completed.returncode != 0:
        _set_secure_job_state(
            status="deallocation_failed",
            deallocation_requested=True,
            last_error=f"stop command exited with {completed.returncode}",
        )
        return False

    _set_secure_job_state(
        status="deallocation_requested",
        deallocation_requested=True,
        last_error="",
    )
    return True


def _idle_deallocator_loop():
    while True:
        time.sleep(5)
        try:
            if SECURE_JOB_STATE.get("deallocation_requested"):
                continue
            if SECURE_JOB_STATE.get("status") == "running":
                continue
            idle_for = int(time.time()) - int(SECURE_JOB_STATE.get("last_activity_unix", int(time.time())))
            if idle_for >= PROCESSING_IDLE_TIMEOUT_SECONDS:
                cvm_debug(
                    f"No secure job payload received for {idle_for}s; requesting VM deallocation"
                )
                request_vm_deallocation(
                    f"idle timeout exceeded ({idle_for}s >= {PROCESSING_IDLE_TIMEOUT_SECONDS}s)"
                )
        except Exception as exc:
            cvm_debug(f"Idle deallocator loop error: {exc}")


def start_idle_deallocator_thread():
    _ensure_cvm_dirs()
    _write_secure_runtime_state()
    thread = threading.Thread(
        target=_idle_deallocator_loop,
        name="processing-idle-deallocator",
        daemon=True,
    )
    thread.start()
    cvm_debug(f"Started idle deallocator thread (tid={thread.ident})")
    return thread


def _aesgcm_decrypt_bytes(key: bytes, enc_data: bytes) -> bytes:
    """
    Decrypt small AES-GCM encrypted blob.
    Wire format: [4B nonce_len][nonce][ciphertext+tag]
    """
    import struct
    nonce_len = struct.unpack(">I", enc_data[:4])[0]
    nonce     = enc_data[4: 4 + nonce_len]
    ct        = enc_data[4 + nonce_len:]
    return AESGCM(key).decrypt(nonce, ct, None)


def _aesgcm_decrypt_file_streaming(key: bytes, enc_path: Path, out_path: Path) -> None:
    """
    Decrypt a large AES-GCM encrypted file written in 64MB chunks.
    Wire format:
      [4B: num_chunks]
      repeated: [4B: len(nonce+ct)][nonce][ciphertext+tag]
    """
    import struct
    aesgcm = AESGCM(key)
    with open(enc_path, "rb") as src, open(out_path, "wb") as dst:
        num_chunks = struct.unpack(">I", src.read(4))[0]
        cvm_debug(f"  decrypting {num_chunks} chunks...")
        for i in range(num_chunks):
            chunk_len  = struct.unpack(">I", src.read(4))[0]
            chunk_data = src.read(chunk_len)
            nonce      = chunk_data[:12]
            ct         = chunk_data[12:]
            dst.write(aesgcm.decrypt(nonce, ct, None))
            if (i + 1) % 10 == 0:
                cvm_debug(f"  decrypted chunk {i+1}/{num_chunks}")


def decrypt_selected_dataset(dataset_id):
    cvm_debug(f"Fetching dataset_id={dataset_id} from GCS + Secret Manager")

    dataset_cfg = config.get_dataset_gcp_config(dataset_id)
    secret_id   = dataset_cfg["secret_id"]
    gcs_object  = dataset_cfg["dataset_json_object"]
    bucket      = config.gcp.datasets_bucket
    project_id  = config.gcp.project_id

    # Fetch AES-256 key from Secret Manager (stored as hex string)
    cvm_debug(f"Fetching AES-256 key from Secret Manager: {secret_id}")
    key_hex = fetch_secret(project_id, secret_id)
    key     = bytes.fromhex(key_hex.decode("utf-8").strip())

    # Download encrypted dataset JSON from GCS
    enc_path = CVM_ARTIFACTS_DIR / f"dataset_{dataset_id}.json.enc"
    cvm_debug(f"Downloading encrypted dataset JSON from gs://{bucket}/{gcs_object}")
    download_gcs_object(bucket, gcs_object, str(enc_path))

    # Decrypt in memory (JSON is small)
    with open(enc_path, "rb") as f:
        enc_data = f.read()
    enc_path.unlink()

    plaintext = _aesgcm_decrypt_bytes(key, enc_data)
    dataset   = json.loads(plaintext.decode("utf-8"))

    selected_dataset_path = CVM_RUNTIME_DIR / f"dataset_{dataset_id}_decrypted.json"
    _json_dump(selected_dataset_path, dataset)
    cvm_debug(f"Dataset {dataset_id} decrypted and saved to {selected_dataset_path}")
    return selected_dataset_path


def fetch_and_extract_images(dataset_id):
    """
    Download the encrypted image zip for dataset_id from GCS,
    decrypt it (streaming AES-GCM chunks), and extract to the
    TEE data directory defined in config.
    """
    import zipfile, struct

    dataset_cfg    = config.get_dataset_gcp_config(dataset_id)
    secret_id      = dataset_cfg["secret_id"]
    images_object  = dataset_cfg["images_object"]
    extract_dir    = Path(dataset_cfg["images_extract_dir"])
    bucket         = config.gcp.datasets_bucket
    project_id     = config.gcp.project_id

    # Fetch AES-256 key (same key as dataset JSON)
    cvm_debug(f"Fetching AES-256 key from Secret Manager: {secret_id}")
    key_hex = fetch_secret(project_id, secret_id)
    key     = bytes.fromhex(key_hex.decode("utf-8").strip())

    # Download encrypted zip from GCS
    enc_zip_path = CVM_ARTIFACTS_DIR / f"dataset_{dataset_id}_images.zip.enc"
    cvm_debug(f"Downloading encrypted images from gs://{bucket}/{images_object}")
    download_gcs_object(bucket, images_object, str(enc_zip_path))

    # Decrypt to a temp zip file
    zip_path = CVM_ARTIFACTS_DIR / f"dataset_{dataset_id}_images.zip"
    cvm_debug(f"Decrypting image zip...")
    _aesgcm_decrypt_file_streaming(key, enc_zip_path, zip_path)
    enc_zip_path.unlink()

    # Extract zip to data directory
    extract_dir.mkdir(parents=True, exist_ok=True)
    cvm_debug(f"Extracting images to {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    zip_path.unlink()

    cvm_debug(f"Images extracted to {extract_dir}")
    return extract_dir


def fetch_evaluation_script(dataset_id: int) -> Path:
    """
    Pull the correct evaluation script from GCS for the given dataset_id.
    dataset 1 = breast cancer  →  evaluate_model_breastcancer.py
    dataset 2 = OCS            →  evaluate_model_OCS.py
    Returns the local path where the script was written.
    """
    _ensure_cvm_dirs()
    dataset_cfg = config.get_dataset_gcp_config(dataset_id)
    gcs_object  = dataset_cfg["eval_script_object"]
    bucket      = config.gcp.eval_scripts_bucket
    script_path = CVM_WORKFLOW_DIR / f"evaluation_script_{dataset_id}.py"

    cvm_debug(f"Fetching eval script for dataset_id={dataset_id} from gs://{bucket}/{gcs_object}")
    download_gcs_object(bucket, gcs_object, str(script_path))
    cvm_debug(f"Eval script written to {script_path}")
    return script_path


def run_evaluation_script_from_paths(model_path, dataset_path, results_path, eval_script_path,
                                     preprocessing_path=None):
    model_path       = Path(model_path)
    dataset_path     = Path(dataset_path)
    results_path     = Path(results_path)
    eval_script_path = Path(eval_script_path)
    weights_path     = model_path.parent / WEIGHTS_FILE_NAME

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"External ONNX weights file not found: {weights_path}")
    if not eval_script_path.exists():
        raise FileNotFoundError(f"Evaluation script not found: {eval_script_path}")

    cvm_debug(f"Running evaluation script: {eval_script_path}")
    cvm_debug(f"Model path: {model_path}")
    cvm_debug(f"Weights path: {weights_path}")
    cvm_debug(f"Dataset path: {dataset_path}")
    if preprocessing_path:
        cvm_debug(f"Preprocessing script: {preprocessing_path}")

    env = os.environ.copy()
    vendor_path = str(Path(config.base_dir) / ".vendor")
    env["PYTHONPATH"] = vendor_path + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        str(eval_script_path),
        "--model",   str(model_path),
        "--dataset", str(dataset_path),
        "--results", str(results_path),
    ]
    if preprocessing_path and Path(preprocessing_path).exists():
        cmd += ["--preprocessing", str(preprocessing_path)]

    result = subprocess.run(
        cmd,
        cwd=config.base_dir,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"Evaluation script failed with exit code {result.returncode}")

    results = _json_load(results_path)
    cvm_debug(f"results.json populated: {results}")
    return results_path, results


def _job_dir(job_id):
    return CVM_SECURE_JOBS_DIR / job_id


# ── Key store helpers ─────────────────────────────────────────────────────────

_KEY_STORE_URL = "http://127.0.0.1:8081/api/get-key"


def _fetch_model_key(job_id: str):
    """Return raw 32-byte AES-256 key from the in-process key store, or None if absent."""
    try:
        resp = requests.get(_KEY_STORE_URL, params={"job_id": job_id}, timeout=5)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        key_b64 = resp.json()["key_b64"]
        return base64.b64decode(key_b64)
    except Exception as exc:
        cvm_debug(f"Key store lookup failed for job_id={job_id}: {exc}")
        return None


def _delete_model_key(job_id: str) -> None:
    """Zero and remove the AES key from the key store after use."""
    try:
        requests.delete(_KEY_STORE_URL, params={"job_id": job_id}, timeout=5)
    except Exception as exc:
        cvm_debug(f"Key store delete failed for job_id={job_id}: {exc}")


def _decrypt_aes_gcm(key_bytes: bytes, encrypted_b64: str, aad: bytes) -> bytes:
    """
    Decrypt AES-256-GCM ciphertext.

    Wire format (Buffer TEE → Processing TEE):
        base64( nonce[12] || ciphertext )
    AAD is the job_id bytes — prevents ciphertext from being reused across jobs.
    """
    raw = base64.b64decode(encrypted_b64)
    if len(raw) < 12 + 16:  # nonce + min GCM tag
        raise ValueError(f"Encrypted blob too short: {len(raw)} bytes")
    nonce = raw[:12]
    ciphertext = raw[12:]
    aesgcm = AESGCM(key_bytes)
    return aesgcm.decrypt(nonce, ciphertext, aad)


# ── Attestation JWT helpers ────────────────────────────────────────────────────

_ATTESTATION_AUDIENCE = "https://tanuh-processing-tee"
# Confidential Space launcher token server (unix socket). This is the ONLY
# source of the real attestation token carrying hwmodel/swname/image_digest/
# secboot. The metadata-server identity token (get_oidc_token) does NOT contain
# these claims.
_CS_TEESERVER_SOCKET = "/run/container_launcher/teeserver.sock"
# Fixed nonce (10–74 chars) — we read claims, not channel-bind, so any valid
# nonce works; it just appears as eat_nonce in the token.
_CS_ATTESTATION_NONCE = "tanuh-leaderboard-attestation-nonce"


def _decode_jwt_payload(token: str) -> dict:
    """Decode the payload of a JWT without verifying the signature."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return {}
        padded = parts[1] + "=" * (4 - len(parts[1]) % 4)
        return json.loads(base64.urlsafe_b64decode(padded))
    except Exception:
        return {}


class _UnixHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection over a unix-domain socket (for the CS teeserver)."""

    def __init__(self, socket_path: str, timeout: int = 10):
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._socket_path)
        self.sock = sock


def _fetch_cs_attestation_token() -> str:
    """
    Fetch the Confidential Space attestation OIDC token from the launcher's
    token server over its unix socket:
        POST /v1/token  {audience, nonces:[...], token_type:"OIDC"}
    The response body is the raw JWT. Raises on failure (caller falls back).
    """
    body = json.dumps({
        "audience": _ATTESTATION_AUDIENCE,
        "nonces": [_CS_ATTESTATION_NONCE],
        "token_type": "OIDC",
    })
    conn = _UnixHTTPConnection(_CS_TEESERVER_SOCKET, timeout=10)
    try:
        conn.request("POST", "/v1/token", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = resp.read().decode("utf-8", errors="replace").strip()
        if resp.status != 200:
            raise RuntimeError(f"teeserver /v1/token returned {resp.status}: {data[:200]}")
        return data
    finally:
        conn.close()


def _fetch_attestation_claims() -> dict:
    """
    Extract Confidential Space attestation claims (hwmodel, swname,
    image_digest, secboot, iss) from the CS launcher's attestation token.

    Falls back to the metadata-server identity token if the CS socket is
    unavailable (e.g. local/non-TEE testing) — that token lacks the CS claims,
    so those fields come back blank, which is the best we can do off-TEE.
    Returns an empty dict if no token can be obtained.
    """
    token = None
    try:
        token = _fetch_cs_attestation_token()
    except Exception as exc:
        cvm_debug(f"CS attestation token unavailable ({exc}); falling back to identity token")
        try:
            token = get_oidc_token(_ATTESTATION_AUDIENCE)
        except Exception as exc2:
            cvm_debug(f"Could not fetch any attestation token: {exc2}")
            return {}

    claims = _decode_jwt_payload(token)
    if not claims:
        cvm_debug("Attestation token could not be decoded")
        return {}
    container = claims.get("submods", {}).get("container", {})
    return {
        "hwmodel": claims.get("hwmodel", ""),
        "swname": claims.get("swname", ""),
        "swversion": claims.get("swversion", []),
        "image_digest": container.get("image_digest", ""),
        "image_reference": container.get("image_reference", ""),
        "instance_id": claims.get("sub", ""),
        "iat": claims.get("iat", 0),
        "secboot": claims.get("secboot", False),
        "iss": claims.get("iss", ""),
    }


# ── Leaderboard ────────────────────────────────────────────────────────────────

def _sanitize_error_msg(msg: str) -> str:
    """Strip filesystem paths from an error message before external reporting."""
    import re as _re
    return _re.sub(r"/[^\s\"']+", "<path>", msg)


def _classify_leaderboard_error(exc: Exception) -> tuple:
    """
    Returns (error_code, error_type, error_message).

    error_code:
      1 = user-supplied code (preprocessing script / model loading)
      2 = our eval scripts / dataloader / model population
      3 = environment (CUDA / GPU / GCS / Secret Manager / network)
    """
    import re as _re
    msg  = str(exc)
    name = type(exc).__name__

    # Map eval subprocess exit codes to error categories.
    m = _re.search(r"Evaluation script failed with exit code (\d+)", msg)
    if m:
        code = int(m.group(1))
        if code == 10:
            return 1, "EvalUserCodeError", "Preprocessing script failed to load or execute."
        if code == 11:
            return 3, "EvalEnvironmentError", "CUDA/GPU runtime error in evaluation script."
        return 2, "EvalScriptError", f"Evaluation script exited with code {code}."

    # Environment / infrastructure indicators.
    env_keywords = ("cuda", "gpu", "nvidia", "cudnn", "onnxruntime",
                    "gcs", "secret manager", "connection", "timeout",
                    "no space", "out of memory", "oom")
    if any(kw in msg.lower() for kw in env_keywords):
        return 3, name, _sanitize_error_msg(msg)

    if name in ("FileNotFoundError", "PermissionError", "OSError"):
        return 3, name, _sanitize_error_msg(msg)

    # Default: our pipeline.
    return 2, name, _sanitize_error_msg(msg)


def _safe_error_stage(exc: Exception) -> str:
    """
    Return a safe, user-facing error description that contains no TEE-internal
    paths, no dataset content, and no stack frames. The full traceback is logged
    to the serial port (cvm_debug / traceback.print_exc) which is only accessible
    to the TEE operator, never returned to the submitting user.
    """
    name = type(exc).__name__
    msg  = str(exc)

    # Eval script subprocess failures only carry the exit code — safe to forward.
    if "Evaluation script failed with exit code" in msg:
        return msg

    # Everything else: classify by exception type, never expose the message body.
    stage_map = {
        "FileNotFoundError":  "Pipeline error: required file not found during setup.",
        "ValueError":         "Pipeline error: invalid value encountered during setup.",
        "RuntimeError":       "Pipeline error: runtime error during evaluation.",
        "TimeoutError":       "Pipeline error: operation timed out.",
        "PermissionError":    "Pipeline error: permission denied during setup.",
        "OSError":            "Pipeline error: OS error during setup.",
    }
    return stage_map.get(name, "Pipeline error: evaluation could not be completed.")


# ── External leaderboard (/submit-solution) ───────────────────────────────────
# POST endpoint for the benchmark leaderboard. The real path is under
# /leaderboard/ (the bare /submit-solution returns 405). Override via env.
LEADERBOARD_SUBMIT_URL = os.getenv(
    "LEADERBOARD_SUBMIT_URL",
    "https://benchmark.tanuh.ai/leaderboard/submit-solution",
)
# Internal dataset_id → leaderboard vertical slug.
_LEADERBOARD_DATASET_SLUG = {1: "breast_cancer", 2: "oral_cancer", 3: "glaucoma"}


def _f2_from_precision_recall(p: float, r: float) -> float:
    """F-beta=2 from precision/recall: 5pr / (4p + r). 0 when undefined."""
    denom = 4.0 * p + r
    return (5.0 * p * r / denom) if denom else 0.0


def _leaderboard_metrics(dataset_id: int, metrics: dict) -> dict:
    """
    Map our internal metrics dict onto the leaderboard's per-vertical schema.
    Fields we can't produce are omitted (left blank) rather than guessed.
    """
    m = metrics or {}
    if dataset_id == 2:  # oral_cancer → OralCancerMetrics
        out = {
            "sensitivity": m.get("sensitivity"),
            "specificity": m.get("specificity"),
            "accuracy":    m.get("accuracy"),
            "ppv":         m.get("ppv"),
            "npv":         m.get("npv"),
            "f2_score":    m.get("f2"),
        }
    elif dataset_id == 1:  # breast_cancer → BreastCancerMetrics
        # weighted_f2 isn't emitted directly; derive it from per-class
        # precision/recall weighted by class support when available.
        weighted_f2 = m.get("weighted_f2")
        per_class = m.get("per_class") or {}
        if weighted_f2 is None and per_class:
            total = 0
            acc = 0.0
            for c in per_class.values():
                support = int(c.get("TP", 0)) + int(c.get("FN", 0))
                total += support
                acc += support * _f2_from_precision_recall(
                    c.get("precision", 0.0), c.get("recall", 0.0)
                )
            weighted_f2 = (acc / total) if total else None
        out = {
            "accuracy":         m.get("accuracy"),
            "macro_f2":         m.get("macro_f2"),
            "weighted_f2":      weighted_f2,
            "macro_f1":         m.get("macro_f1"),
            "sensitivity":      m.get("macro_recall"),
            "qwk":              m.get("qwk"),
            "specificity":      m.get("macro_specificity"),
            "npv":              m.get("macro_npv"),
            "ppv":              m.get("macro_ppv"),
            "confusion_matrix": m.get("confusion_matrix"),
            "auc":              m.get("auc"),
        }
    else:
        out = dict(m)
    return {k: v for k, v in out.items() if v is not None}


def _submit_to_leaderboard(job_id: str, dataset_id: int, attestation: dict,
                           status: str = "succeeded",
                           results: dict = None,
                           error: dict = None,
                           submitted_by: str = "") -> None:
    """
    POST one evaluation result to the external leaderboard /submit-solution.

    status="succeeded": sends full metrics payload.
    status="failed":    sends error payload (no metrics).

    Best-effort: never raises — the pipeline's success does not depend on it.
    """
    slug = _LEADERBOARD_DATASET_SLUG.get(dataset_id)
    if not slug:
        cvm_debug(f"Leaderboard: no vertical for dataset_id={dataset_id}; skipping submit")
        return

    lb_job_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"tanuh:{job_id}"))
    attestation_body = {
        "hwmodel":      attestation.get("hwmodel", ""),
        "swname":       attestation.get("swname", ""),
        "image_digest": attestation.get("image_digest", ""),
        "secboot":      bool(attestation.get("secboot", False)),
        "iss":          attestation.get("iss", ""),
    }

    if status == "succeeded" and results:
        body = {
            "job_id":                 lb_job_id,
            "dataset_id":             slug,
            "status":                 "succeeded",
            "num_samples":            results.get("num_samples"),
            "elapsed_seconds":        results.get("elapsed_seconds"),
            "model_sha256":           results.get("model_sha256"),
            "onnx_runtime_providers": results.get("onnx_runtime_providers", []),
            "attestation":            attestation_body,
            "metrics":                _leaderboard_metrics(dataset_id, results.get("metrics", {})),
        }
    else:
        body = {
            "job_id":      lb_job_id,
            "dataset_id":  slug,
            "status":      "failed",
            "attestation": attestation_body,
        }
        if error:
            body["error"] = error

    try:
        token = get_oidc_token(_ATTESTATION_AUDIENCE)
    except Exception as exc:
        cvm_debug(f"Leaderboard: could not obtain Bearer token: {exc}; skipping submit")
        return

    try:
        resp = requests.post(
            LEADERBOARD_SUBMIT_URL,
            json=body,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if resp.status_code in (200, 201):
            cvm_debug(f"Leaderboard: {status} submitted for job {job_id} ({slug}) → {resp.status_code}")
        else:
            cvm_debug(
                f"Leaderboard: submit for job {job_id} returned "
                f"{resp.status_code}: {resp.text[:300]}"
            )
    except Exception as exc:
        cvm_debug(f"Leaderboard: submit failed for job {job_id} (non-fatal): {exc}")


def _materialize_secure_job_payload(payload):
    required_fields = {
        "job_id",
        "dataset_id",
        "model_onnx_base64",
        "model_weights_base64",
        "model_sha256",
        "weights_sha256",
    }
    missing = required_fields - set(payload.keys())
    if missing:
        raise ValueError(f"Missing fields in secure job payload: {sorted(missing)}")

    job_id = payload["job_id"]
    job_dir = _job_dir(job_id)
    artifacts_dir = job_dir / "artifacts"
    runtime_dir = job_dir / "runtime"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / payload.get("model_file", MODEL_FILE_NAME)
    weights_path = artifacts_dir / payload.get("weights_file", WEIGHTS_FILE_NAME)

    # Try to retrieve the per-job AES-256 key deposited by the Buffer TEE via RA-TLS.
    # If no key is present (e.g. during local testing), treat the base64 fields as plaintext.
    aes_key = _fetch_model_key(job_id)
    if aes_key is not None:
        cvm_debug(f"AES-256 key found for job_id={job_id}; decrypting model bytes")
        aad = job_id.encode("utf-8")
        model_bytes = _decrypt_aes_gcm(aes_key, payload["model_onnx_base64"], aad)
        weights_bytes = _decrypt_aes_gcm(aes_key, payload["model_weights_base64"], aad)
        _delete_model_key(job_id)
    else:
        cvm_debug(f"No AES key in store for job_id={job_id}; treating model fields as plaintext base64")
        model_bytes = base64.b64decode(payload["model_onnx_base64"])
        weights_bytes = base64.b64decode(payload["model_weights_base64"])

    if hashlib.sha256(model_bytes).hexdigest() != payload["model_sha256"]:
        raise ValueError("model.onnx SHA256 mismatch in secure payload")
    if hashlib.sha256(weights_bytes).hexdigest() != payload["weights_sha256"]:
        raise ValueError("model weights SHA256 mismatch in secure payload")

    model_path.write_bytes(model_bytes)
    weights_path.write_bytes(weights_bytes)

    # Extract user preprocessing script if provided (Oral Cancer jobs)
    preprocessing_path = None
    preprocessing_b64 = payload.get("preprocessing_script_base64")
    if preprocessing_b64:
        preprocessing_bytes = base64.b64decode(preprocessing_b64)
        preprocessing_sha256_expected = payload.get("preprocessing_sha256", "")
        if preprocessing_sha256_expected and hashlib.sha256(preprocessing_bytes).hexdigest() != preprocessing_sha256_expected:
            raise ValueError("preprocessing.py SHA256 mismatch in secure payload")
        preprocessing_path = artifacts_dir / "preprocessing.py"
        preprocessing_path.write_bytes(preprocessing_bytes)
        cvm_debug(f"preprocessing.py written to {preprocessing_path} ({len(preprocessing_bytes)} bytes)")

    _json_dump(job_dir / "incoming_payload.json", payload)
    cvm_debug(f"Secure job payload materialized under {job_dir}")
    return {
        "job_id": job_id,
        "job_dir": job_dir,
        "runtime_dir": runtime_dir,
        "model_path": model_path,
        "weights_path": weights_path,
        "preprocessing_path": preprocessing_path,
        "dataset_id": int(payload["dataset_id"]),
        "hyperparameters": payload.get("hyperparameters", {}),
        "submitted_by": payload.get("submitted_by", ""),
    }


def run_secure_job_pipeline(payload):
    _ensure_cvm_dirs()
    job_materialized = _materialize_secure_job_payload(payload)
    job_id = job_materialized["job_id"]
    runtime_dir = job_materialized["runtime_dir"]
    dataset_id = int(job_materialized["dataset_id"])
    submitted_by = job_materialized.get("submitted_by", "")

    _set_secure_job_state(
        status="running",
        current_job_id=job_id,
        last_job_id=job_id,
        last_error="",
    )
    cvm_debug(f"Secure job {job_id}: starting pipeline for dataset_id={dataset_id}")

    # Step 1: fetch encrypted dataset JSON from GCS and decrypt
    dataset_path = decrypt_selected_dataset(dataset_id)
    job_dataset_path = runtime_dir / dataset_path.name
    shutil.copy2(dataset_path, job_dataset_path)
    cvm_debug(f"Secure job {job_id}: dataset decrypted and copied to {job_dataset_path}")

    # Step 2: fetch encrypted image zip from GCS, decrypt, and extract
    cvm_debug(f"Secure job {job_id}: fetching and extracting images for dataset_id={dataset_id}")
    fetch_and_extract_images(dataset_id)

    # Step 3: pull the correct evaluation script from GCS for this dataset
    eval_script_path = fetch_evaluation_script(dataset_id)
    cvm_debug(f"Secure job {job_id}: eval script fetched to {eval_script_path}")

    # Step 4: run evaluation
    results_path = runtime_dir / "results.json"
    cvm_debug(f"Secure job {job_id}: evaluating ONNX model with runtime results at {results_path}")
    _, results = run_evaluation_script_from_paths(
        model_path=job_materialized["model_path"],
        dataset_path=job_dataset_path,
        results_path=results_path,
        eval_script_path=eval_script_path,
        preprocessing_path=job_materialized.get("preprocessing_path"),
    )
    results["job_id"] = job_id
    results["model_sha256"] = _sha256_file(job_materialized["model_path"])
    results["weights_sha256"] = _sha256_file(job_materialized["weights_path"])
    results["dataset_id"] = dataset_id
    _json_dump(results_path, results)
    cvm_debug(f"Secure job {job_id}: local results saved to {results_path}")

    # Fetch attestation claims from the Confidential Space OIDC token.
    cvm_debug(f"Secure job {job_id}: fetching attestation claims for leaderboard")
    attestation = _fetch_attestation_claims()

    try:
        _submit_to_leaderboard(job_id, dataset_id, attestation,
                               status="succeeded", results=results,
                               submitted_by=submitted_by)
    except Exception as exc:
        cvm_debug(f"Secure job {job_id}: leaderboard submit failed (non-fatal): {exc}")

    _set_secure_job_state(
        status="complete",
        current_job_id="",
        last_job_id=job_id,
        last_error="",
    )
    if PROCESSING_DEALLOCATE_AFTER_JOB:
        request_vm_deallocation(f"job {job_id} finished successfully")
    return {
        "status": "success",
        "job_id": job_id,
        "results_path": str(results_path),
        "attestation": attestation,
        "results": results,
    }


@app.route("/enclave/cvm/secure-job", methods=["POST"])
def receive_secure_job():
    content = request.json
    if not content:
        return jsonify({"status": "error", "message": "Missing JSON job payload"}), 400

    with SECURE_JOB_LOCK:
        if SECURE_JOB_STATE.get("status") == "running":
            return jsonify(
                {
                    "status": "busy",
                    "message": f"Already running job {SECURE_JOB_STATE.get('current_job_id')}",
                }
            ), 409

        try:
            _ensure_cvm_dirs()
            job_id = content.get("job_id", "")
            _set_secure_job_state(
                status="job_received",
                current_job_id=job_id,
                last_job_id=job_id,
                last_error="",
                deallocation_requested=False,
            )
            _json_dump(CVM_INCOMING_DIR / f"{job_id}-secure-job.json", content)
        except Exception as exc:
            _set_secure_job_state(status="error", last_error=str(exc))
            return jsonify({"status": "error", "message": str(exc)}), 400

    def _runner():
        job_id = content.get("job_id", "")
        try:
            run_secure_job_pipeline(content)
        except Exception as exc:
            # Log full traceback to serial port only — never sent outside the TEE.
            traceback.print_exc()
            cvm_debug(f"Secure job {job_id} failed: {exc}")
            _set_secure_job_state(
                status="error",
                current_job_id="",
                last_job_id=job_id,
                last_error=str(exc),
            )
            # Submit failure to leaderboard.
            dataset_id = int(content.get("dataset_id", 0))
            error_code, error_type, error_message = _classify_leaderboard_error(exc)
            try:
                attestation = _fetch_attestation_claims()
            except Exception:
                attestation = {}
            _submit_to_leaderboard(
                job_id, dataset_id, attestation,
                status="failed",
                submitted_by=content.get("submitted_by", ""),
                error={
                    "error_code":    error_code,
                    "error_type":    error_type,
                    "error_message": error_message,
                },
            )
            if PROCESSING_DEALLOCATE_AFTER_JOB:
                request_vm_deallocation(f"job {job_id} failed: {exc}")

    thread = threading.Thread(
        target=_runner,
        name=f"secure-job-{content.get('job_id', 'unknown')}",
        daemon=True,
    )
    thread.start()
    cvm_debug(
        f"Secure job {content.get('job_id')} accepted from RA-TLS bridge and started on thread {thread.ident}"
    )
    return jsonify(
        {
            "status": "accepted",
            "job_id": content.get("job_id"),
            "thread_name": thread.name,
        }
    ), 202


@app.route("/enclave/cvm/runtime-state", methods=["GET"])
def get_secure_runtime_state():
    return jsonify({"status": "success", "runtime_state": _secure_runtime_state()}), 200


@app.route("/enclave/cvm/results", methods=["GET"])
def get_cvm_results():
    results_path = CVM_RUNTIME_DIR / "results.json"
    if not results_path.exists():
        last_job_id = SECURE_JOB_STATE.get("last_job_id", "")
        if last_job_id:
            secure_results_path = _job_dir(last_job_id) / "runtime" / "results.json"
            if secure_results_path.exists():
                return jsonify(_json_load(secure_results_path)), 200
        return jsonify({"status": "processing", "message": "results.json is not available yet"}), 404
    return jsonify(_json_load(results_path)), 200



# Error handler for critical errors that require service restart
@app.errorhandler(Exception)
def handle_critical_error(e):
    """Handle critical errors by restarting the service.

    Excludes:
    - HTTP exceptions (404, 400, etc.) - normal routing errors
    - PermissionError - file permission issues, should be handled in routes
    - OSError/IOError - file system errors, usually recoverable

    Only actual application crashes and unhandled exceptions trigger service restart.
    """
    # Skip HTTP exceptions - these are normal routing errors, not critical failures
    if isinstance(e, HTTPException):
        # Return proper JSON response with CORS headers for HTTP errors
        response = jsonify({
            "title": "Error",
            "description": f"{e.code} {e.name}: {e.description}"
        })
        response.status_code = e.code
        return response

    # Skip file permission and I/O errors - these are recoverable and should be handled in routes
    if isinstance(e, (PermissionError, OSError, IOError)):
        print(f"File system error (non-critical): {str(e)}")
        traceback.print_exc()
        response = jsonify({
            "title": "Error",
            "description": f"File system error: {str(e)}. Please check file permissions and try again."
        })
        response.status_code = 500
        return response

    # Only handle actual critical errors (unhandled exceptions, crashes, etc.)
    # In Confidential Space the container runtime handles restarts; just log + 500.
    print(f"Critical error in manager: {str(e)}")
    traceback.print_exc()

    response = jsonify({
        "title": "Error",
        "description": f"Critical error occurred: {str(e)}"
    })
    response.status_code = 500
    return response


if __name__ == "__main__":
    print("=" * 60)
    print("Starting Enclave Manager")
    print(f"Port: {config.service.port}")
    print("Endpoints available:")
    print("  - POST /enclave/cvm/secure-job")
    print("  - GET  /enclave/cvm/runtime-state")
    print("  - GET  /enclave/cvm/results")
    print("=" * 60)
    _ensure_cvm_dirs()
    start_idle_deallocator_thread()
    app.run(host=config.service.host, port=config.service.port, debug=True, use_reloader=False)
