# Processing TEE Image

This directory is a self-contained Docker build context for the Processing TEE / GPU CS side.

The image:
- builds the RA-TLS GPU CS server from `b2p-ratls`
- starts GPU CS on `LISTEN_ADDR`, default `:443`
- starts `enclave_manager_new.py` on port `4000`
- accepts `/api/load-model` over the verified RA-TLS channel
- materializes the JSON job payload into `model.onnx`, `model.onnx.data`, and dataset ID
- creates placeholder encrypted datasets and keys at runtime
- decrypts the selected dataset from `encrypted_dataset.json` using `dataset_keys.json`
- writes and runs `evaluation_script.py` with ONNX Runtime
- posts results back to the Buffer callback URL
- deallocates itself through `stop-processing-vm.sh` after 300s idle or after job completion

Important runtime env vars, supplied through Confidential Space metadata as `tee-env-*`:
- `LISTEN_ADDR=:443`
- `RATLS_AUDIENCE=ratls-buffer-tee`
- `PROCESSING_IDLE_TIMEOUT_SECONDS=300`
- `PROCESSING_DEALLOCATE_AFTER_JOB=1`

Debug trail:
- VM serial/container logs include `gpu-cs:`, `[Google-CVM workflow]`, and `[evaluation_script]` lines.
- Runtime state and per-job results are written under `/app/cvm_workflow` inside the container.
