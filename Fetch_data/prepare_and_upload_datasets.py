#!/usr/bin/env python3
"""
One-time setup script: zip, AES-GCM encrypt, and upload datasets + eval scripts to GCS.
Also stores AES-256 keys in Secret Manager.

Run this on any machine with GCS + Secret Manager access before deploying the TEE.

What it does per dataset:
  1. Builds the dataset JSON (file paths as they will exist in the TEE + labels)
  2. Zips the image directory
  3. Generates a random AES-256 key
  4. Encrypts the dataset JSON with that key (AES-GCM)
  5. Encrypts the zip with the same key (AES-GCM, streaming 64MB chunks)
  6. Uploads both encrypted files to gs://tanuh-datasets/
  7. Stores the hex key in Secret Manager
  8. Uploads the eval script to gs://tanuh-eval-scripts/

Requires: google-cloud-storage, google-cloud-secret-manager, cryptography
"""

import csv
import json
import os
import struct
import sys
import zipfile
from pathlib import Path

GCP_PROJECT         = "p3dx-depa-sandbox"
DATASETS_BUCKET     = "tanuh-datasets"
EVAL_SCRIPTS_BUCKET = "tanuh-eval-scripts"

BREAST_CANCER_CSV   = Path("/home/azureuser/Eval Scripts/evaluator/dataset.csv")
BREAST_CANCER_DIR   = Path("/home/azureuser/Eval Scripts/breast-cancer")
OCS_DIR             = Path("/home/azureuser/Eval Scripts/OCS/OCS_Benchmarking")
EVAL_SCRIPT_1       = Path("/home/azureuser/Eval Scripts/evaluator/evaluate_model_breastcancer.py")
EVAL_SCRIPT_2       = Path("/home/azureuser/Eval Scripts/OCS/evaluate_model_OCS.py")

# Absolute paths as they will exist inside the TEE container after zip extraction
TEE_BREAST_CANCER_DATA = "/app/cvm_workflow/data/breast-cancer"
TEE_OCS_DATA           = "/app/cvm_workflow/data/ocs"

OLD_PREFIX = "/media/miglab/DATA_20TB1/Inference_Density/"

CHUNK_SIZE = 64 * 1024 * 1024  # 64 MB per encryption chunk


# ── AES-GCM encryption ────────────────────────────────────────────────────────

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError:
    print("ERROR: pip install cryptography")
    sys.exit(1)


def generate_key() -> bytes:
    return os.urandom(32)  # AES-256


def encrypt_bytes(key: bytes, plaintext: bytes) -> bytes:
    """Encrypt small in-memory data. Wire format: [4B nonce_len][nonce][ct+tag]"""
    aesgcm = AESGCM(key)
    nonce  = os.urandom(12)
    ct     = aesgcm.encrypt(nonce, plaintext, None)
    return struct.pack(">I", len(nonce)) + nonce + ct


def encrypt_file_streaming(key: bytes, src_path: Path, dst_path: Path) -> None:
    """
    Encrypt a large file in 64MB chunks.
    Wire format:
      [4B: num_chunks]
      repeated: [4B: len(nonce+ct)][nonce][ct+tag]
    """
    aesgcm = AESGCM(key)
    chunks = []
    with open(src_path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            nonce = os.urandom(12)
            ct    = aesgcm.encrypt(nonce, chunk, None)
            chunks.append((nonce, ct))
            print(f"    encrypted chunk {len(chunks)} "
                  f"({len(chunk)/1024/1024:.1f} MB)", flush=True)

    with open(dst_path, "wb") as out:
        out.write(struct.pack(">I", len(chunks)))
        for nonce, ct in chunks:
            out.write(struct.pack(">I", len(nonce) + len(ct)))
            out.write(nonce)
            out.write(ct)
    print(f"    encrypted file: {dst_path.stat().st_size/1024/1024:.1f} MB")


# ── GCS helpers ───────────────────────────────────────────────────────────────

def upload_file(bucket_name: str, object_path: str, local_path: Path,
                content_type: str = "application/octet-stream") -> None:
    from google.cloud import storage
    client = storage.Client()
    blob   = client.bucket(bucket_name).blob(object_path)
    size   = local_path.stat().st_size / 1024 / 1024
    print(f"  uploading {local_path.name} ({size:.1f} MB) "
          f"→ gs://{bucket_name}/{object_path}", flush=True)
    blob.upload_from_filename(str(local_path), content_type=content_type)
    print(f"  upload complete.")


def upload_bytes(bucket_name: str, object_path: str, data: bytes,
                 content_type: str = "application/octet-stream") -> None:
    from google.cloud import storage
    client = storage.Client()
    blob   = client.bucket(bucket_name).blob(object_path)
    blob.upload_from_string(data, content_type=content_type)
    print(f"  uploaded {len(data)/1024:.1f} KB → gs://{bucket_name}/{object_path}")


# ── Secret Manager ────────────────────────────────────────────────────────────

def store_secret(project: str, secret_id: str, key_bytes: bytes) -> None:
    from google.cloud import secretmanager
    client      = secretmanager.SecretManagerServiceClient()
    parent      = f"projects/{project}"
    secret_path = f"{parent}/secrets/{secret_id}"
    try:
        client.create_secret(request={
            "parent":    parent,
            "secret_id": secret_id,
            "secret":    {"replication": {"automatic": {}}},
        })
        print(f"  created secret: {secret_id}")
    except Exception:
        print(f"  secret {secret_id} already exists — adding new version")
    client.add_secret_version(request={
        "parent":  secret_path,
        "payload": {"data": key_bytes},
    })
    print(f"  key stored in Secret Manager: {secret_id}")


# ── Dataset JSON builders ─────────────────────────────────────────────────────

def build_breast_cancer_dataset() -> dict:
    """
    Reads dataset.csv. Remaps old-machine paths to TEE-absolute paths
    so the eval script can find files after zip extraction.
    """
    paths, labels = [], []
    with open(BREAST_CANCER_CSV, newline="") as f:
        for row in csv.DictReader(f):
            raw = row["new_file_path"].strip()
            if OLD_PREFIX in raw:
                rel = raw.split(OLD_PREFIX, 1)[1]
            else:
                rel = "/".join(Path(raw).parts[-3:])
            tee_path = TEE_BREAST_CANCER_DATA + "/" + rel
            paths.append(tee_path)
            labels.append(int(row["tissueden"]))
    return {
        "dataset_id":  1,
        "description": "Breast cancer density (DICOM mammography)",
        "num_classes": 4,
        "dicom_paths": paths,
        "labels":      labels,
    }


def build_ocs_dataset() -> dict:
    """
    Derives paths and labels from folder structure (Suspicious=1, Non-Suspicious=0).
    Stores TEE-absolute paths matching zip extraction layout.
    """
    paths, labels = [], []
    for label, folder in [(0, "Non-Suspicious"), (1, "Suspicious")]:
        d = OCS_DIR / folder
        for p in sorted(d.glob("*.jpg")):
            paths.append(f"{TEE_OCS_DATA}/{folder}/{p.name}")
            labels.append(label)
    return {
        "dataset_id":  2,
        "description": "OCS (Oral Cancer Screening)",
        "num_classes": 2,
        "image_paths": paths,
        "labels":      labels,
    }


# ── Zip helper ────────────────────────────────────────────────────────────────

def zip_directory(src_dir: Path, zip_path: Path) -> None:
    print(f"  zipping {src_dir} ...", flush=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        files = [p for p in sorted(src_dir.rglob("*"))
                 if p.is_file() and p.name != "desktop.ini"]
        for i, p in enumerate(files):
            zf.write(p, p.relative_to(src_dir))
            if (i + 1) % 100 == 0:
                print(f"    zipped {i+1}/{len(files)} files...", flush=True)
    print(f"  zip done: {zip_path.stat().st_size/1024/1024:.1f} MB")


# ── Per-dataset pipeline ──────────────────────────────────────────────────────

def process_dataset(name, dataset_dict, src_dir,
                    json_gcs_object, images_gcs_object,
                    secret_id, tmp_dir):
    print(f"\n{'='*60}")
    print(f"DATASET: {name}")
    print(f"{'='*60}")

    key = generate_key()
    print(f"  generated AES-256 key")

    # Encrypt dataset JSON
    json_bytes    = json.dumps(dataset_dict).encode("utf-8")
    enc_json      = encrypt_bytes(key, json_bytes)
    enc_json_path = tmp_dir / f"{name}_dataset.json.enc"
    enc_json_path.write_bytes(enc_json)
    print(f"  dataset JSON: {len(json_bytes)/1024:.1f} KB "
          f"→ {len(enc_json)/1024:.1f} KB encrypted")

    # Zip images
    zip_path = tmp_dir / f"{name}_images.zip"
    zip_directory(src_dir, zip_path)

    # Encrypt zip (streaming)
    enc_zip_path = tmp_dir / f"{name}_images.zip.enc"
    print(f"  encrypting zip in 64MB chunks...")
    encrypt_file_streaming(key, zip_path, enc_zip_path)
    zip_path.unlink()

    # Upload to GCS
    print("  uploading encrypted files to GCS...")
    upload_file(DATASETS_BUCKET, json_gcs_object, enc_json_path)
    upload_file(DATASETS_BUCKET, images_gcs_object, enc_zip_path)
    enc_json_path.unlink()
    enc_zip_path.unlink()

    # Store key in Secret Manager as hex string
    store_secret(GCP_PROJECT, secret_id, key.hex().encode("utf-8"))
    print(f"  {name} complete.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    tmp_dir = Path("/tmp/tanuh_upload")
    tmp_dir.mkdir(exist_ok=True)

    # Dataset 1: breast cancer
    ds1 = build_breast_cancer_dataset()
    print(f"Breast cancer: {len(ds1['dicom_paths'])} samples")
    process_dataset(
        name              = "breast-cancer",
        dataset_dict      = ds1,
        src_dir           = BREAST_CANCER_DIR,
        json_gcs_object   = "breast-cancer/dataset.json.enc",
        images_gcs_object = "breast-cancer/images.zip.enc",
        secret_id         = "tanuh-dataset-key-1",
        tmp_dir           = tmp_dir,
    )

    # Dataset 2: OCS
    ds2 = build_ocs_dataset()
    print(f"\nOCS: {len(ds2['image_paths'])} samples")
    process_dataset(
        name              = "ocs",
        dataset_dict      = ds2,
        src_dir           = OCS_DIR,
        json_gcs_object   = "ocs/dataset.json.enc",
        images_gcs_object = "ocs/images.zip.enc",
        secret_id         = "tanuh-dataset-key-2",
        tmp_dir           = tmp_dir,
    )

    # Eval scripts
    print(f"\n{'='*60}")
    print("EVAL SCRIPTS")
    print(f"{'='*60}")
    upload_bytes(EVAL_SCRIPTS_BUCKET, "evaluate_model_breastcancer.py",
                 EVAL_SCRIPT_1.read_bytes(), content_type="text/x-python")
    upload_bytes(EVAL_SCRIPTS_BUCKET, "evaluate_model_OCS.py",
                 EVAL_SCRIPT_2.read_bytes(), content_type="text/x-python")

    print(f"\n{'='*60}")
    print("ALL DONE")
    print(f"{'='*60}")
    print(f"\ngs://{DATASETS_BUCKET}/breast-cancer/dataset.json.enc")
    print(f"gs://{DATASETS_BUCKET}/breast-cancer/images.zip.enc")
    print(f"gs://{DATASETS_BUCKET}/ocs/dataset.json.enc")
    print(f"gs://{DATASETS_BUCKET}/ocs/images.zip.enc")
    print(f"gs://{EVAL_SCRIPTS_BUCKET}/evaluate_model_breastcancer.py")
    print(f"gs://{EVAL_SCRIPTS_BUCKET}/evaluate_model_OCS.py")
    print(f"\nSecret Manager: tanuh-dataset-key-1, tanuh-dataset-key-2")


if __name__ == "__main__":
    main()
