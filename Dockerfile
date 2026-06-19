FROM ghcr.io/astral-sh/uv:0.7.0 AS uv-source

FROM golang:1.22-bullseye AS ratls-builder
WORKDIR /src
COPY b2p-ratls/go.mod b2p-ratls/go.sum ./b2p-ratls/
WORKDIR /src/b2p-ratls
RUN go mod download
COPY b2p-ratls/ ./
RUN CGO_ENABLED=0 GOOS=linux go build -o /out/gpu-cs ./cmd/gpu-cs/

# CUDA 12.3 + cuDNN 9 runtime — provides libcudart.so.12, libcublas.so.12,
# libcudnn.so.9 that onnxruntime-gpu 1.19.x links against. The host NVIDIA
# driver (libcuda.so) is mounted by Confidential Space at /usr/local/nvidia.
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    BASE_DIR=/app \
    DEBIAN_FRONTEND=noninteractive \
    RATLS_AUDIENCE=ratls-buffer-tee \
    LISTEN_ADDR=:443 \
    INTERNAL_ADDR=127.0.0.1:8081 \
    PROCESSING_MANAGER_JOB_URL=http://127.0.0.1:4000/enclave/cvm/secure-job \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib

LABEL "tee.launch_policy.allow_env_override"="RATLS_AUDIENCE,LISTEN_ADDR,INTERNAL_ADDR,PROCESSING_MANAGER_JOB_URL,PROCESSING_IDLE_TIMEOUT_SECONDS,PROCESSING_DEALLOCATE_AFTER_JOB"
LABEL "tee.launch_policy.allow_cmd_override"="false"

WORKDIR /app

# python3.11 from deadsnakes to match the prior runtime (3.11) on Ubuntu 22.04,
# plus build deps for native wheels (pydicom/pylibjpeg/opencv).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        build-essential \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# UV binary — used at runtime to install user preprocessing script dependencies
COPY --from=uv-source /uv /bin/uv

COPY requirements.txt /tmp/requirements.txt
# Step 1: install everything except JAX and flash-attn.
# torch/torchvision/torchaudio use the cu121 index; the rest come from PyPI.
# --ignore-installed blinker: Ubuntu base ships a distutils-installed blinker 1.4
# that pip cannot cleanly remove.
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        --ignore-installed blinker \
        -r /tmp/requirements.txt

# Step 2: install JAX with CUDA 12 via the official JAX wheel index.
# pip's resolver cannot handle jax[cuda12] from PyPI due to a circular plugin
# dependency; the JAX team's own index is the supported install path.
RUN python -m pip install \
        --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
        "jax[cuda12]==0.4.35"

# flash-attn requires nvcc (CUDA devel toolkit) to compile — not available in
# the runtime base image. Left for UV to install at job time if needed.

COPY . /app
COPY --from=ratls-builder /out/gpu-cs /usr/local/bin/gpu-cs

RUN chmod +x /app/entrypoint.sh /usr/local/bin/gpu-cs \
    && mkdir -p /app/cvm_workflow/artifacts /app/cvm_workflow/incoming /app/cvm_workflow/runtime /app/cvm_workflow/secure_jobs

EXPOSE 4000 443

ENTRYPOINT ["/app/entrypoint.sh"]
