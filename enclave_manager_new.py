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
import P3DX_SDK
import requests
from lib.config import config
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from Fetch_data.secrets import fetch_secret, get_oidc_token
from Fetch_data.gcs_fetch import download_gcs_object, download_gcs_object_bytes, upload_gcs_object


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


def _hardware_evidence():
    """Collect best-effort hardware evidence for a Google AMD SEV-SNP CVM."""
    cvm_debug("Collecting hardware evidence from OS-visible CVM interfaces")
    cpuinfo_path = Path("/proc/cpuinfo")
    cpuinfo = cpuinfo_path.read_text(errors="ignore") if cpuinfo_path.exists() else ""
    sev_guest_candidates = [
        Path("/dev/sev-guest"),
        Path("/sys/firmware/sev/guest"),
        Path("/sys/kernel/security/secrets/coco"),
    ]
    evidence = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "amd_cpu_detected": "AuthenticAMD" in cpuinfo or "AMD" in cpuinfo or "AMD" in platform.processor(),
        "sev_snp_interface_detected": any(candidate.exists() for candidate in sev_guest_candidates),
        "sev_snp_interface_candidates": [str(candidate) for candidate in sev_guest_candidates],
        "google_cloud_hint": Path("/sys/class/dmi/id/product_name").read_text(errors="ignore").strip()
        if Path("/sys/class/dmi/id/product_name").exists()
        else None,
    }
    cvm_debug(f"Hardware evidence collected: {evidence}")
    return evidence


def _software_evidence():
    """Collect measurements/config hashes that are safe to share with a verifier."""
    cvm_debug("Collecting software evidence: manager code hash, config hash, PCR files if present")
    evidence = {
        "enclave_manager_code_sha256": P3DX_SDK.hash_enclave_manager_code(config.base_dir),
        "config_yml_sha256": _sha256_file(Path(config.base_dir) / "config.yml"),
    }

    pcr_path = Path(config.get_path("pcr_values"))
    image_hash_path = Path(config.get_path("image_hash"))
    if pcr_path.exists():
        evidence["pcr_values"] = _json_load(pcr_path)
    if image_hash_path.exists():
        evidence["docker_image_sha256"] = image_hash_path.read_text(encoding="utf-8").strip()

    cvm_debug("Software evidence collected")
    return evidence


def build_attestation_report():
    """
    Build a placeholder attestation report.

    Production Google SEV-SNP integration should replace the placeholder section
    with an SNP_GET_REPORT ioctl through /dev/sev-guest, or the Google-supported
    attestation report retrieval path for the chosen Confidential VM product.
    """
    cvm_debug("Building attestation report with nonce, hardware evidence, and software measurements")
    nonce = P3DX_SDK.generate_nonce()
    report = {
        "format": "google-cvm-amd-sev-snp-placeholder-v1",
        "nonce": nonce,
        "created_at_unix": int(time.time()),
        "hardware": _hardware_evidence(),
        "software": _software_evidence(),
        "placeholder_note": (
            "Replace this object with the raw AMD SEV-SNP report and certificate chain "
            "before wiring to the production verifier."
        ),
    }
    report_path = CVM_RUNTIME_DIR / "attestation_report.json"
    _json_dump(report_path, report)
    cvm_debug(f"Attestation report written to {report_path}")
    return report


def send_attestation_report_to_verifier(report):
    """
    Send attestation to verifier, with a local placeholder approval path.

    Set CVM_ATTESTATION_ENDPOINT to use a real verifier. When it is unset, this
    waits 5 seconds and returns approved=True to keep local development moving.
    """
    endpoint = os.getenv("CVM_ATTESTATION_ENDPOINT", ATTESTATION_VERIFIER_ENDPOINT_PLACEHOLDER)
    cvm_debug(f"Prepared attestation report for verifier endpoint: {endpoint}")

    if endpoint == ATTESTATION_VERIFIER_ENDPOINT_PLACEHOLDER:
        cvm_debug("Placeholder verifier active; waiting 5 seconds before approving attestation")
        time.sleep(5)
        return {
            "approved": True,
            "verifier": "placeholder",
            "message": "Placeholder verifier approved after 5 second wait",
        }

    import requests

    cvm_debug("Sending attestation report to configured verifier")
    response = requests.post(endpoint, json=report, timeout=30)
    response.raise_for_status()
    verdict = response.json()
    cvm_debug(f"Verifier response received: {verdict}")
    return verdict


def _resnet34_hyperparameters():
    return {
        "architecture": "resnet34",
        "weights": "random-placeholder",
        "input_shape": [1, 3, 64, 64],
        "num_classes": 3,
        "opset_version": 17,
        "normalization": "placeholder datasets are already scaled to 0..1",
        "batch_size": 8,
    }


def _make_placeholder_dataset(dataset_id, sample_count=18):
    """Fabricate tiny numeric image-like datasets for ONNX Runtime evaluation."""
    import numpy as np

    rng = np.random.default_rng(7000 + dataset_id)
    features = rng.normal(0.18, 0.03, size=(sample_count, 3, 64, 64)).astype("float32")

    if dataset_id == 1:
        labels = np.asarray([idx % 2 for idx in range(sample_count)], dtype="int64")
        for idx, label in enumerate(labels):
            features[idx, :, 24:40, 24:40] += 0.55 if label == 1 else 0.05
        description = "Binary bright-center numerical image dataset"
        num_classes = 2
    elif dataset_id == 2:
        labels = np.asarray([idx % 3 for idx in range(sample_count)], dtype="int64")
        for idx, label in enumerate(labels):
            if label == 0:
                features[idx, 0, 10:18, :] += 0.45
            elif label == 1:
                features[idx, 1, :, 28:36] += 0.45
            else:
                features[idx, 2, 44:54, :] += 0.45
        description = "Three-class stripe-position numerical image dataset"
        num_classes = 3
    elif dataset_id == 3:
        labels = np.asarray([(idx // 2) % 2 for idx in range(sample_count)], dtype="int64")
        for idx, label in enumerate(labels):
            diagonal = np.eye(64, dtype="float32")
            if label == 1:
                features[idx, :, :, :] += diagonal * 0.5
            else:
                features[idx, :, :, :] += np.fliplr(diagonal) * 0.5
        description = "Binary diagonal-pattern numerical image dataset"
        num_classes = 2
    else:
        raise ValueError(f"Unsupported placeholder dataset id: {dataset_id}")

    features = np.clip(features, 0.0, 1.0)
    return {
        "dataset_id": dataset_id,
        "description": description,
        "num_classes": num_classes,
        "features": features.tolist(),
        "labels": labels.tolist(),
    }


def create_placeholder_encrypted_datasets(force=False):
    encrypted_path = CVM_ARTIFACTS_DIR / "encrypted_dataset.json"
    keys_path = CVM_ARTIFACTS_DIR / "dataset_keys.json"
    if encrypted_path.exists() and keys_path.exists() and not force:
        cvm_debug(f"Encrypted placeholder datasets already exist at {encrypted_path}")
        return encrypted_path, keys_path

    cvm_debug("Creating encrypted_dataset.json and dataset_keys.json placeholder files")
    encrypted_payload = {
        "format": "fernet-placeholder-v1",
        "note": "Production flow should fetch only the selected dataset key from Secret Manager.",
        "datasets": {},
    }
    key_payload = {
        "format": "fernet-placeholder-keys-v1",
        "note": "PLACEHOLDER: replace this file with Google Secret Manager lookups.",
        "keys": {},
    }

    for dataset_id in (1, 2, 3):
        key = Fernet.generate_key()
        cipher = Fernet(key)
        dataset = _make_placeholder_dataset(dataset_id)
        plaintext = json.dumps(dataset).encode("utf-8")
        encrypted_payload["datasets"][str(dataset_id)] = {
            "ciphertext": cipher.encrypt(plaintext).decode("utf-8"),
            "algorithm": "Fernet-AES128-CBC-HMACSHA256",
        }
        key_payload["keys"][str(dataset_id)] = key.decode("utf-8")
        cvm_debug(f"Encrypted placeholder dataset {dataset_id}")

    _json_dump(encrypted_path, encrypted_payload)
    _json_dump(keys_path, key_payload)
    return encrypted_path, keys_path


def create_placeholder_resnet34_onnx(force=False):
    model_path = CVM_ARTIFACTS_DIR / MODEL_FILE_NAME
    weights_path = CVM_ARTIFACTS_DIR / WEIGHTS_FILE_NAME
    if model_path.exists() and weights_path.exists() and not force:
        cvm_debug(f"Placeholder ONNX model already exists at {model_path}")
        return model_path, weights_path

    cvm_debug("Exporting placeholder ResNet34 with random weights to ONNX")
    _ensure_vendor_path()
    import torch
    from torchvision.models import resnet34
    import onnx

    CVM_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_model_path = CVM_ARTIFACTS_DIR / "model.inline.onnx"
    if weights_path.exists():
        weights_path.unlink()

    model = resnet34(weights=None, num_classes=3)
    model.eval()
    dummy_input = torch.randn(1, 3, 64, 64)

    torch.onnx.export(
        model,
        dummy_input,
        str(temp_model_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    onnx_model = onnx.load(str(temp_model_path))
    onnx.save_model(
        onnx_model,
        str(model_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=weights_path.name,
        size_threshold=0,
    )
    temp_model_path.unlink(missing_ok=True)
    cvm_debug(f"ONNX model written to {model_path}")
    cvm_debug(f"External ONNX weights written to {weights_path}")
    return model_path, weights_path


def create_placeholder_confirmation(force=False, dataset_id=1, ensure_artifacts=True):
    confirmation_path = CONFIRMATION_FILE_PLACEHOLDER
    if confirmation_path.exists() and not force:
        cvm_debug(f"Placeholder confirmation already exists at {confirmation_path}")
        return confirmation_path

    cvm_debug("Creating placeholder confirmation.json from verifier payload")
    if ensure_artifacts:
        model_path, weights_path = create_placeholder_resnet34_onnx(force=force)
        create_placeholder_encrypted_datasets(force=force)
    else:
        model_path = CVM_ARTIFACTS_DIR / MODEL_FILE_NAME
        weights_path = CVM_ARTIFACTS_DIR / WEIGHTS_FILE_NAME
    confirmation = {
        "model": model_path.name,
        "weights": weights_path.name,
        "dataset_id": dataset_id,
        "hyperparameters": _resnet34_hyperparameters(),
        "placeholder_note": (
            "This file stands in for the payload that the remote attestation verifier "
            "will send after approving the CVM."
        ),
    }
    _json_dump(confirmation_path, confirmation)
    cvm_debug(f"Placeholder confirmation written to {confirmation_path}")
    return confirmation_path


def prepare_cvm_placeholder_fixtures(force=False, dataset_id=1):
    """Generate local placeholder inputs outside the attested runtime pipeline."""
    cvm_debug("Preparing CVM placeholder fixtures outside the runtime pipeline")
    model_path, weights_path = create_placeholder_resnet34_onnx(force=force)
    encrypted_path, keys_path = create_placeholder_encrypted_datasets(force=force)
    confirmation_path = create_placeholder_confirmation(
        force=force,
        dataset_id=dataset_id,
        ensure_artifacts=False,
    )
    return {
        "status": "success",
        "model_path": str(model_path),
        "weights_path": str(weights_path),
        "encrypted_dataset_path": str(encrypted_path),
        "dataset_keys_path": str(keys_path),
        "confirmation_path": str(confirmation_path),
        "note": "These are local placeholder fixtures. The CVM workflow only consumes them.",
    }


def wait_for_confirmation_payload(timeout_seconds=30):
    cvm_debug(f"Waiting up to {timeout_seconds}s for confirmation payload at {CONFIRMATION_FILE_PLACEHOLDER}")
    start = time.time()
    while time.time() - start < timeout_seconds:
        if CONFIRMATION_FILE_PLACEHOLDER.exists():
            confirmation = _json_load(CONFIRMATION_FILE_PLACEHOLDER)
            cvm_debug(f"Confirmation payload received: {confirmation}")
            return confirmation
        time.sleep(1)

    cvm_debug("Confirmation payload timeout reached")
    print("Shutting down without actually deallocating this placeholder VM.", flush=True)
    raise TimeoutError(f"confirmation.json not received within {timeout_seconds} seconds")


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


def run_evaluation_script(confirmation, dataset_path, eval_script_path):
    model_path = CVM_ARTIFACTS_DIR / confirmation["model"]
    results_path = CVM_RUNTIME_DIR / "results.json"
    return run_evaluation_script_from_paths(model_path, dataset_path, results_path, eval_script_path)


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


# ── GCS results + leaderboard ─────────────────────────────────────────────────

def _upload_results_to_gcs(job_id: str, results: dict) -> str:
    """Upload results.json to gs://<results_bucket>/results/<job_id>/results.json."""
    bucket = config.gcp.results_bucket
    object_path = f"results/{job_id}/results.json"
    payload = json.dumps(results, indent=2).encode("utf-8")
    upload_gcs_object(bucket, object_path, payload)
    gcs_uri = f"gs://{bucket}/{object_path}"
    cvm_debug(f"Results uploaded to {gcs_uri}")
    return gcs_uri


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


def _upload_error_to_gcs(job_id: str, error_message: str, stage: str = "") -> str:
    """
    Publish a job failure to the SAME GCS location the Buffer TEE polls for
    results (results/<job_id>/results.json). The Buffer TEE's existing
    GET /buffer/jobs/<job_id>/results endpoint fetches this object from GCS and
    relays it to the browser, so the error reaches the UI over the exact channel
    used for successful results — no extra callback path is introduced.

    Best-effort: never raises, so it cannot mask the original failure or block
    shutdown.
    """
    if not job_id:
        cvm_debug("Cannot upload error to GCS: empty job_id")
        return ""
    error_payload = {
        "status": "error",
        "job_id": job_id,
    }
    try:
        bucket = config.gcp.results_bucket
        object_path = f"results/{job_id}/results.json"
        upload_gcs_object(bucket, object_path,
                          json.dumps(error_payload, indent=2).encode("utf-8"))
        gcs_uri = f"gs://{bucket}/{object_path}"
        cvm_debug(f"Error report uploaded to {gcs_uri} for Buffer TEE/UI to fetch")
        return gcs_uri
    except Exception as exc:
        cvm_debug(f"Failed to upload error report to GCS for {job_id}: {exc}")
        return ""


def _update_leaderboard(job_id: str, results: dict, attestation: dict) -> None:
    """
    Read-modify-write the leaderboard JSON at gs://<results_bucket>/leaderboard.json.
    Appends one entry combining evaluation results with attestation claims.
    """
    bucket = config.gcp.results_bucket
    lb_path = "leaderboard.json"

    try:
        existing = json.loads(download_gcs_object_bytes(bucket, lb_path))
        entries = existing if isinstance(existing, list) else []
    except Exception:
        entries = []

    entry = {
        "job_id": job_id,
        "submitted_at_unix": int(time.time()),
        "dataset_id": results.get("dataset_id"),
        "num_samples": results.get("num_samples"),
        "accuracy": results.get("accuracy"),
        "prediction_distribution": results.get("prediction_distribution"),
        "model_sha256": results.get("model_sha256"),
        "weights_sha256": results.get("weights_sha256"),
        "elapsed_seconds": results.get("elapsed_seconds"),
        "attestation": attestation,
    }
    entries.append(entry)

    upload_gcs_object(bucket, lb_path, json.dumps(entries, indent=2).encode("utf-8"))
    cvm_debug(f"Leaderboard updated in gs://{bucket}/{lb_path} ({len(entries)} entries total)")


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


def _submit_to_leaderboard(job_id: str, results: dict, attestation: dict) -> None:
    """
    POST one evaluation result to the external leaderboard /submit-solution.

    The leaderboard requires job_id to be a UUID and reads the caller identity
    from the Bearer JWT's `sub` claim (signature not verified). We use the
    Processing TEE's Confidential Space OIDC token as the Bearer.

    Best-effort: never raises — the pipeline's success does not depend on it.
    """
    dataset_id = int(results.get("dataset_id") or 0)
    slug = _LEADERBOARD_DATASET_SLUG.get(dataset_id)
    if not slug:
        cvm_debug(f"Leaderboard: no vertical for dataset_id={dataset_id}; skipping submit")
        return

    # Stable UUID derived from our internal job_id (endpoint requires a UUID).
    lb_job_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"tanuh:{job_id}"))

    body = {
        "job_id": lb_job_id,
        "dataset_id": slug,
        "num_samples": results.get("num_samples"),
        "elapsed_seconds": results.get("elapsed_seconds"),
        "model_sha256": results.get("model_sha256"),
        "onnx_runtime_providers": results.get("onnx_runtime_providers", []),
        "attestation": {
            "hwmodel": attestation.get("hwmodel", ""),
            "swname": attestation.get("swname", ""),
            "image_digest": attestation.get("image_digest", ""),
            "secboot": bool(attestation.get("secboot", False)),
            "iss": attestation.get("iss", ""),
        },
        "metrics": _leaderboard_metrics(dataset_id, results.get("metrics", {})),
    }

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
            cvm_debug(f"Leaderboard: submitted job {job_id} ({slug}) → {resp.status_code}")
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

        # Scan preprocessing script for missing deps and install via UV.
        # Raises RuntimeError (caught by run_secure_job_pipeline) on install failure,
        # so the job is marked error with the exact uv output rather than a
        # confusing ImportError later inside the eval subprocess.
        from dep_scanner import install_missing_deps
        cvm_debug("Scanning preprocessing.py for missing dependencies...")
        installed, skipped = install_missing_deps(preprocessing_path)
        if installed:
            cvm_debug(f"UV installed: {installed}")
        cvm_debug(f"UV skipped (stdlib/image/present): {len(skipped)} packages")

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
    }


def run_secure_job_pipeline(payload):
    _ensure_cvm_dirs()
    job_materialized = _materialize_secure_job_payload(payload)
    job_id = job_materialized["job_id"]
    runtime_dir = job_materialized["runtime_dir"]
    dataset_id = int(job_materialized["dataset_id"])

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

    # Upload results to GCS. Buffer TEE will poll GCS directly — no callback needed.
    gcs_results_uri = ""
    try:
        gcs_results_uri = _upload_results_to_gcs(job_id, results)
        cvm_debug(f"Secure job {job_id}: results uploaded to {gcs_results_uri}")
    except Exception as exc:
        cvm_debug(f"Secure job {job_id}: GCS results upload failed (non-fatal): {exc}")

    try:
        _submit_to_leaderboard(job_id, results, attestation)
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
        "gcs_results_uri": gcs_results_uri,
        "attestation": attestation,
        "results": results,
    }


def run_google_cvm_workflow(confirmation_timeout=30, clear_runtime=False):
    """Run the requested Google AMD SEV-SNP CVM placeholder workflow end to end."""
    global state, is_app_running

    is_app_running = True
    state = {
        "step": 1,
        "maxSteps": 4,
        "title": "Google CVM Attestation",
        "description": "Building and sending attestation report",
    }

    _ensure_cvm_dirs()

    if clear_runtime and CVM_RUNTIME_DIR.exists():
        cvm_debug(f"Clearing old runtime files from {CVM_RUNTIME_DIR}")
        for item in CVM_RUNTIME_DIR.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    try:
        cvm_debug("Step 1/4: Verify hardware/software evidence and send attestation report")
        report = build_attestation_report()
        verdict = send_attestation_report_to_verifier(report)
        if not verdict.get("approved"):
            raise PermissionError(f"Attestation verifier did not approve this CVM: {verdict}")
        cvm_debug("Attestation approved; continuing workflow")

        state = {
            "step": 2,
            "maxSteps": 4,
            "title": "Waiting for Confirmation Payload",
            "description": "Polling placeholder confirmation.json",
        }
        cvm_debug("Step 2/4: Receive confirmation payload")
        confirmation = wait_for_confirmation_payload(timeout_seconds=confirmation_timeout)

        state = {
            "step": 3,
            "maxSteps": 4,
            "title": "Decrypting Selected Dataset",
            "description": "Using placeholder dataset_keys.json instead of Secret Manager",
        }
        cvm_debug("Step 3/4: Pull encrypted dataset JSON, images, and eval script")
        dataset_id   = int(confirmation.get("dataset_id", 1))
        dataset_path = decrypt_selected_dataset(dataset_id)
        fetch_and_extract_images(dataset_id)

        state = {
            "step": 4,
            "maxSteps": 4,
            "title": "Evaluating ONNX Model",
            "description": "Running ONNX Runtime evaluation script",
        }
        cvm_debug("Step 4/4: Compile ONNX model, load external weights, evaluate, and write results")
        eval_script_path = fetch_evaluation_script(dataset_id)
        results_path, results = run_evaluation_script(confirmation, dataset_path, eval_script_path)

        state = {
            "step": 4,
            "maxSteps": 4,
            "title": "Secure Evaluation Complete",
            "description": f"Results written to {results_path}",
        }
        cvm_debug("Google CVM placeholder workflow complete")
        return {
            "status": "success",
            "attestation": verdict,
            "confirmation_path": str(CONFIRMATION_FILE_PLACEHOLDER),
            "results_path": str(results_path),
            "results": results,
        }
    finally:
        is_app_running = False


# Removed after_request handler - flask-cors already handles CORS headers
# Adding duplicate headers causes "multiple values" error



# DEPLOY: Deploys the TEE enclave
@app.route("/enclave/deploy", methods=["POST"])
def deploy_enclave():
    jwt_file_path = config.get_path('jwt_response')
    subprocess.run(["sudo", "rm", "-rf", jwt_file_path], check=False, capture_output=True)
    
    print("STARTING deploy")
    global is_app_running, stored_bundle
    
    if is_app_running:
        print("Previous deployment detected. Restarting service to reset state...")
        try:
            P3DX_SDK.restart_enclave_manager()

            time.sleep(3)
            is_app_running = False
            stored_bundle = None
        except Exception as e:
            print(f"Warning: Failed to restart service: {str(e)}")
            response = {
                "title": "Error",
                "description": f"Previous deployment detected but failed to restart service: {str(e)}"
            }
            return jsonify(response), 500
    
    stored_bundle = None

    global state
    state = {
        "step": 1,
        "maxSteps": 11,
        "title": "Spawning Trusted Execution Environment (TEE)",
        "description": "Step 1"
    }
    
    content = request.json if request.json else {}
    compose_url = content.get("compose_url")

    if not compose_url:
        return jsonify({
            "title": "Error",
            "description": "compose_url is required in request payload"
        }), 400

    try:
        cmd = f"python3 -u deploy_enclave.py {repr(compose_url)} 2>&1 | systemd-cat -t tee-deployment"
        subprocess.Popen(
            ["sudo", "sh", "-c", cmd],
            cwd=config.base_dir
        )
        
        is_app_running = True
        response = {
            "title": "Success",
            "description": "Application execution has started."
        }
        return jsonify(response), 200
        
    except Exception as e:
        response = {
            "title": "Error",
            "description": f"Failed to start application: {str(e)}"
        }
        return jsonify(response), 500



stored_bundle = None

@app.route("/enclave/jwt", methods=["POST"])
def receive_jwt():
    try:
        content = request.json
        if not content or 'jwt' not in content:
            return jsonify({"error": "Missing jwt in request"}), 400
        return jsonify({"status": "success", "message": "JWT stored"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/enclave/jwt", methods=["GET"])
def get_jwt():
    print("Fetching JWT token...")
    jwt_file_path = config.get_path('jwt_response')
    
    if not os.path.exists(jwt_file_path):
        response = {
            "title": "Error: JWT not found",
            "description": "JWT token not available yet. Deployment in progress..."
        }
        return jsonify(response), 404
    
    try:
        result = subprocess.run(
            ['sudo', 'chmod', '644', jwt_file_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        with open(jwt_file_path, "r") as f:
            jwt_token = f.read().strip()
        
        if not jwt_token:
            response = {
                "title": "Error: Empty JWT",
                "description": "JWT token file is empty."
            }
            return jsonify(response), 404
        
        print(f"JWT token retrieved successfully (length: {len(jwt_token)})")
        
        response = {
            "title": "Success",
            "jwt": jwt_token
        }
        return jsonify(response), 200
        
    except subprocess.CalledProcessError as e:
        response = {
            "title": "Error: Permission denied",
            "description": f"Failed to set file permissions: {e.stderr.decode() if e.stderr else str(e)}"
        }
        return jsonify(response), 500
        
    except Exception as e:
        response = {
            "title": "Error reading JWT",
            "description": f"Failed to read JWT token: {str(e)}"
        }
        return jsonify(response), 500



# GET FRESH JWT: Returns a fresh JWT token
@app.route("/enclave/jwt/fresh", methods=["GET"])
def get_fresh_jwt():
    """Generate a fresh JWT token by deleting old JWT and executing guest attestation.
    
    Returns:
        JSON response with newly generated JWT token or error details.
    """
    print("Generating fresh JWT token...")
    
    jwt_file_path = config.get_path('jwt_response')
    private_key_path = config.get_path('private_key')
    public_key_path = config.get_path('public_key')
    keys_dir = config.paths.keys_dir
    
    original_cwd = os.getcwd()
    
    try:
        os.chdir(config.base_dir)
        os.makedirs(keys_dir, exist_ok=True)
        
        subprocess.run(
            ["sudo", "chown", "-R", f"{config.user}:{config.user}", keys_dir],
            check=False,
            capture_output=True
        )
        subprocess.run(
            ["sudo", "chmod", "-R", "755", keys_dir],
            check=False,
            capture_output=True
        )
        
        if os.path.exists(jwt_file_path):
            subprocess.run(
                ["sudo", "rm", "-rf", jwt_file_path],
                check=False,
                capture_output=True
            )
            print("Old JWT file deleted")
        
        if not os.path.exists(private_key_path) or not os.path.exists(public_key_path):
            print("Keys not found. Generating new key pair...")
            P3DX_SDK.generate_and_save_key_pair()
            print("Key pair generated successfully")

        try:
            # Measure enclave manager code 
            P3DX_SDK.measure_enclave_manager_code_vtpm()
            print("Enclave manager code hash measured successfully")
            
            # Measure Docker image
            # link = P3DX_SDK.extract_docker_image_from_compose()
            # P3DX_SDK.measureDockervTPM(link)
            # print("Application image hash measured successfully")
        except Exception as e:
            print(f"Warning: Failed to measure code/image: {str(e)}")
        
        # new nonce generated every time a fresh endpoint is hit
        print("Generating fresh deployment nonce...")
        nonce = P3DX_SDK.generate_nonce()                  
        P3DX_SDK.save_nonce(nonce)
        P3DX_SDK.extend_nonce_to_pcr8(nonce)
        print(f"Generated deployment nonce: {nonce}")

        print("Executing guest attestation to generate new JWT...")
        P3DX_SDK.execute_guest_attestation()
        
        subprocess.run(
            ["sudo", "chown", f"{config.user}:{config.user}", jwt_file_path],
            check=False,
            capture_output=True
        )
        subprocess.run(
            ['sudo', 'chmod', '644', jwt_file_path],
            check=False,
            capture_output=True
        )
        
        with open(jwt_file_path, "r") as f:
            jwt_token = f.read().strip()
        
        if not jwt_token:
            response = {
                "title": "Error: Empty JWT",
                "description": "JWT token file is empty after generation."
            }
            return jsonify(response), 500
        
        print(f"Fresh JWT token generated successfully (length: {len(jwt_token)})")
        
        response = {
            "title": "Success",
            "jwt": jwt_token
        }
        return jsonify(response), 200
        
    except RuntimeError as e:
        response = {
            "title": "Error: JWT generation failed",
            "description": str(e)
        }
        return jsonify(response), 500
        
    except Exception as e:
        print(f"Unexpected error generating JWT: {str(e)}")
        response = {
            "title": "Error: JWT generation failed",
            "description": f"Failed to generate JWT token: {str(e)}"
        }
        return jsonify(response), 500
        
    finally:
        os.chdir(original_cwd)

# GET BUNDLE: Returns the encrypted bundle for polling
@app.route("/enclave/bundle", methods=["GET"])
def get_bundle():
    global stored_bundle
    
    bundle_file = config.get_path('encrypted_bundle')
    if os.path.exists(bundle_file):
        try:
            with open(bundle_file, 'r') as f:
                stored_bundle = json.load(f)
        except Exception:
            pass
    
    if stored_bundle:
        return jsonify({"bundle": stored_bundle}), 200
    else:
        return jsonify({"error": "Bundle not found"}), 404


@app.route("/enclave/bundle/upload", methods=["POST"])
def upload_encrypted_bundle():
    print("Receiving encrypted bundle...")
    
    try:
        content = request.json
        
        if not content:
            response = {
                "title": "Error",
                "description": "No data received"
            }
            return jsonify(response), 400
        
        bundle_dir = config.paths.bundle_dir
        os.makedirs(bundle_dir, exist_ok=True)
        
        global stored_bundle
        stored_bundle = content
        
        output_file = config.get_path('encrypted_bundle')
        
        with open(output_file, 'w') as f:
            json.dump(content, f, indent=2)
        
        os.chmod(output_file, 0o644)
        
        print(f"Encrypted bundle saved to {output_file}")
        print(f"File size: {os.path.getsize(output_file)} bytes")
        
        response = {
            "title": "Success",
            "description": f"Encrypted bundle saved successfully",
            "file_path": output_file,
            "file_size": os.path.getsize(output_file)
        }
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error saving encrypted bundle: {str(e)}")
        response = {
            "title": "Error",
            "description": f"Failed to save encrypted bundle: {str(e)}"
        }
        return jsonify(response), 500


@app.route("/enclave/cvm/run", methods=["POST"])
def run_cvm_workflow_endpoint():
    """Run the Google AMD SEV-SNP CVM placeholder attestation/evaluation workflow."""
    content = request.json if request.json else {}
    try:
        result = run_google_cvm_workflow(
            confirmation_timeout=int(content.get("confirmation_timeout", 30)),
            clear_runtime=content.get("clear_runtime", False),
        )
        return jsonify(result), 200
    except TimeoutError as e:
        return jsonify({
            "status": "timeout",
            "message": str(e),
            "placeholder_action": "Shutting down without actually deallocating this placeholder VM.",
        }), 408
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/enclave/cvm/prepare-fixtures", methods=["POST"])
def prepare_cvm_fixtures_endpoint():
    """Generate local placeholder artifacts outside the attested runtime pipeline."""
    content = request.json if request.json else {}
    try:
        result = prepare_cvm_placeholder_fixtures(
            force=content.get("force", False),
            dataset_id=int(content.get("dataset_id", 1)),
        )
        return jsonify(result), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/enclave/cvm/confirmation", methods=["POST"])
def receive_cvm_confirmation():
    """
    Receive confirmation payload after remote verifier approval.

    Production verifier should POST confirmation.json here:
    POST http://<cvm-enclave-manager-host>:4000/enclave/cvm/confirmation
    """
    content = request.json
    if not content:
        return jsonify({"status": "error", "message": "Missing JSON confirmation payload"}), 400

    required_fields = {"model", "weights", "dataset_id", "hyperparameters"}
    missing = required_fields - set(content.keys())
    if missing:
        return jsonify({"status": "error", "message": f"Missing fields: {sorted(missing)}"}), 400

    _json_dump(CONFIRMATION_FILE_PLACEHOLDER, content)
    cvm_debug(f"Confirmation payload received over HTTP and saved to {CONFIRMATION_FILE_PLACEHOLDER}")
    return jsonify({
        "status": "success",
        "confirmation_path": str(CONFIRMATION_FILE_PLACEHOLDER),
    }), 200


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
            # Publish a sanitized failure notice to GCS.
            # Tracebacks and raw exception messages are intentionally excluded:
            # they may contain TEE-internal paths or, in the case of a malicious
            # preprocessing script, dataset content embedded in exception text.
            safe_stage = _safe_error_stage(exc)
            _upload_error_to_gcs(
                job_id,
                error_message=safe_stage,
                stage="secure_job_pipeline",
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


@app.route("/enclave/cvm/results-stub", methods=["POST"])
def results_stub():
    """Test callback endpoint — accepts results POSTed by run_secure_job_pipeline during local testing."""
    content = request.json or {}
    cvm_debug(f"results-stub received callback for job_id={content.get('job_id')} accuracy={content.get('accuracy')}")
    return jsonify({"status": "received", "job_id": content.get("job_id")}), 200


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



# INFERENCE: Returns the inference as a JSON object
@app.route("/enclave/inference", methods=["GET"])
def get_inference():
    print("Fetching inference...")
    logger = logging.getLogger()
    logging.debug('STARTING INFERENCE')
    
    global state
    
    if state["step"] != 5:
        response = {
            "title": "Error: App execution incomplete",
            "description": "No inference output found. Current step: " + str(state["step"])
        }
        return jsonify(response), 403


    output_file = config.get_path('status')
    
    if os.path.exists(output_file):
        try:
            result = subprocess.run(
                ['sudo', 'chmod', '644', output_file], 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            if result.returncode == 0:
                print(f"Successfully set permissions on file: {output_file}")
            else:
                print(f"Failed to set permissions. Error: {result.stderr.decode()}")
                
        except subprocess.CalledProcessError as e:
            print(f"Error executing sudo chmod: {e.stderr.decode()}")
    else:
        print(f"File not found: {output_file}")


    if os.path.isfile(output_file):
        with open(output_file, "r") as f:
            content = f.read()
        
        print(f"Inference file read successfully (size: {len(content)} bytes)")
        
        response = app.response_class(
            response=content,
            mimetype="application/json"
        )
        return response
    else:
        response = {
            "title": "Error: No Inference Output",
            "description": "Inference file does not exist at " + output_file
        }
        return jsonify(response), 403



# SETSTATE: Sets the state of the enclave
@app.route("/enclave/setstate", methods=["POST"])
def setState():
    global state
    global is_app_running
    print("In /enclave/setstate...")
    
    content = request.json
    if not content or "state" not in content:
        return jsonify({"status": "error", "message": "Missing 'state' in request body"}), 400
    
    state = content["state"]
    
    print(f"State updated - Step {state['step']}/{state['maxSteps']}: {state['title']}")
    
    if state["step"] == 11:
        is_app_running = False
        print("Deployment completed, resetting is_app_running flag")
    
    response = app.response_class(
        response='{"status": "ok"}', 
        status=200, 
        mimetype="application/json"
    )
    return response



# STATE: Returns the current state of the enclave
@app.route("/enclave/state", methods=["GET"])
def get_state():
    global state
    response = {
        "step": state.get("step", 0),
        "maxSteps": state.get("maxSteps", 11),
        "title": state.get("title", "Inactive"),
        "description": state.get("description", "Inactive"),
    }
    print(f"State requested - Step {response['step']}/{response['maxSteps']}")
    return jsonify(response)


# STATUS: Returns application status
@app.route("/enclave/status", methods=["GET"])
def get_app_status_endpoint():
    """Poll endpoint for application status.
    
    Returns status.json content
    """
    print("Fetching application status...")
    
    try:
        status_response = P3DX_SDK.get_app_status()
        return jsonify(status_response), 200
            
    except Exception as e:
        print(f"Error fetching status: {str(e)}")
        return jsonify({
            "status": "error",
            "error": {
                "code": "ENDPOINT_ERROR",
                "message": "Failed to fetch status",
                "details": str(e)
            }
        }), 500


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
    print(f"Critical error in manager: {str(e)}")
    traceback.print_exc()
    
    # Restart service on critical errors only
    # Use a flag to prevent infinite restart loops
    restart_attempted = False
    try:
        P3DX_SDK.restart_enclave_manager()
        restart_attempted = True
        print("Service restart initiated successfully")
    except Exception as restart_error:
        error_msg = str(restart_error) if restart_error else "Unknown error"
        print(f"Failed to restart service: {error_msg}")
    
    response = jsonify({
        "title": "Error",
        "description": f"Critical error occurred. {'Service restarting' if restart_attempted else 'Service restart failed'}: {str(e)}"
    })
    response.status_code = 500
    return response


if __name__ == "__main__":
    if "--prepare-cvm-fixtures" in sys.argv:
        force = "--force" in sys.argv
        dataset_id = 1
        for arg in sys.argv:
            if arg.startswith("--dataset-id="):
                dataset_id = int(arg.split("=", 1)[1])
        outcome = prepare_cvm_placeholder_fixtures(force=force, dataset_id=dataset_id)
        print(json.dumps(outcome, indent=2), flush=True)
        sys.exit(0)

    if "--run-cvm-pipeline" in sys.argv:
        clear_runtime = "--clear-runtime" in sys.argv
        timeout = 30
        for arg in sys.argv:
            if arg.startswith("--confirmation-timeout="):
                timeout = int(arg.split("=", 1)[1])
        outcome = run_google_cvm_workflow(
            confirmation_timeout=timeout,
            clear_runtime=clear_runtime,
        )
        print(json.dumps(outcome, indent=2), flush=True)
        sys.exit(0)

    print("=" * 60)
    print("Starting Enclave Manager")
    print(f"Port: {config.service.port}")
    print("Endpoints available:")
    print("  - POST /enclave/deploy")
    print("  - POST /enclave/cvm/run")
    print("  - POST /enclave/cvm/prepare-fixtures")
    print("  - POST /enclave/cvm/confirmation")
    print("  - POST /enclave/cvm/secure-job")
    print("  - GET  /enclave/cvm/runtime-state")
    print("  - GET  /enclave/cvm/results")
    print("  - GET  /enclave/jwt")
    print("  - GET  /enclave/state")
    print("  - POST /enclave/setstate")
    print("  - GET  /enclave/inference")
    print("  - GET  /enclave/status")
    print("=" * 60)
    _ensure_cvm_dirs()
    start_idle_deallocator_thread()
    app.run(host=config.service.host, port=config.service.port, debug=True, use_reloader=False)
