#!/usr/bin/env python3
"""Local stand-in for starting/stopping the Processing TEE VM."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PID_FILE = BASE_DIR / "cvm_workflow" / "local_vm_processes.json"


def _creationflags() -> int:
    if os.name != "nt":
        return 0
    return subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS


def _log_name(command) -> str:
    executable = Path(command[0]).name or "process"
    role = "manager" if "enclave_manager" in " ".join(command) else "ratls"
    return f"{role}-{executable}.log"


def _start_process(command, workdir: Path, env: dict[str, str]) -> subprocess.Popen:
    stdout_path = BASE_DIR / "cvm_workflow" / "runtime" / _log_name(command)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    handle = stdout_path.open("a", encoding="utf-8")
    return subprocess.Popen(
        command,
        cwd=str(workdir),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        creationflags=_creationflags(),
    )


def start() -> None:
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    if PID_FILE.exists():
        print(f"PID file already exists at {PID_FILE}; run --stop first if needed")
        return

    env = os.environ.copy()
    env.setdefault("BASE_DIR", str(BASE_DIR))
    env.setdefault("PROCESSING_MANAGER_JOB_URL", "http://127.0.0.1:4000/enclave/cvm/secure-job")
    env.setdefault("RATLS_AUDIENCE", "ratls-buffer-tee")
    env.setdefault("LISTEN_ADDR", "127.0.0.1:8443")
    env.setdefault("RATLS_TEST_MODE", "true")
    env.setdefault("RATLS_TEST_IMAGE_DIGEST", env.get("GPU_CS_IMAGE_DIGEST", "local-test-image-digest"))
    env.setdefault("PROCESSING_VM_STOP_COMMAND", f"{sys.executable} {BASE_DIR / 'local_processing_vm.py'} --stop")
    env.setdefault("GOTELEMETRY", "off")

    manager = _start_process([sys.executable, "enclave_manager_new.py"], BASE_DIR, env)
    ratls = _start_process(["go", "run", "./cmd/gpu-cs"], BASE_DIR / "b2p-ratls", env)

    payload = {
        "manager_pid": manager.pid,
        "ratls_pid": ratls.pid,
    }
    PID_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def stop() -> None:
    if not PID_FILE.exists():
        print("No local processing VM pid file found")
        return

    payload = json.loads(PID_FILE.read_text(encoding="utf-8"))
    for key in ("ratls_pid", "manager_pid"):
        pid = int(payload.get(key, 0) or 0)
        if pid <= 0:
            continue
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True)
            else:
                os.kill(pid, signal.SIGTERM)
        except Exception as exc:
            print(f"Could not stop {key}={pid}: {exc}")

    PID_FILE.unlink(missing_ok=True)
    print("Local processing VM stopped")


def status() -> None:
    if not PID_FILE.exists():
        print(json.dumps({"running": False}, indent=2))
        return
    print(PID_FILE.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.stop:
        stop()
    elif args.status:
        status()
    else:
        start()


if __name__ == "__main__":
    main()
