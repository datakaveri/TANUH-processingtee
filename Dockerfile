# TANUH Processing TEE — single Go binary + Python eval runtime.
#
# The Go binary (processing-tee) owns the whole pipeline: RA-TLS intake,
# dataset fetch/decrypt, leaderboard submission, buffer completion callback,
# self-deallocation. Python exists in this image ONLY for the evaluation
# scripts fetched from gs://tanuh-eval-scripts at job time (onnxruntime and
# friends), executed as a subprocess.
FROM golang:1.26-bookworm AS go-builder
WORKDIR /src
COPY go.mod ./
COPY cmd/ ./cmd/
COPY internal/ ./internal/
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" \
    -o /out/processing-tee ./cmd/processing-tee/

# CUDA 12.3 + cuDNN 9 runtime — provides libcudart.so.12, libcublas.so.12,
# libcudnn.so.9 that onnxruntime-gpu 1.19.x links against. The host NVIDIA
# driver (libcuda.so) is mounted by Confidential Space at /usr/local/nvidia.
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    BASE_DIR=/app \
    DEBIAN_FRONTEND=noninteractive \
    RATLS_AUDIENCE=ratls-buffer-tee \
    LISTEN_ADDR=:443 \
    NVIDIA_VISIBLE_DEVICES=none \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_VISIBLE_DEVICES=-1 \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib

LABEL "tee.launch_policy.allow_env_override"="RATLS_AUDIENCE,LISTEN_ADDR,PROCESSING_IDLE_TIMEOUT_SECONDS,PROCESSING_DEALLOCATE_AFTER_JOB,PROCESSING_EVAL_TIMEOUT_SECONDS,PROCESSING_DEPS_TIMEOUT_SECONDS,PROJECT,ZONE,INSTANCE,LEADERBOARD_SUBMIT_URL"
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

COPY requirements.txt /tmp/requirements.txt
# onnxruntime flavor is chosen at build time from a single codebase → two images:
#   default (GPU/H100):  onnxruntime-gpu==1.19.2  (links libcudart.so.12 / libcudnn.so.9)
#   CPU fallback image:  build with --build-arg ONNXRUNTIME_PKG=onnxruntime==1.19.2
#                        (no CUDA provider .so → cannot segfault on a GPU-less VM)
ARG ONNXRUNTIME_PKG=onnxruntime-gpu==1.19.2
# torch from the CUDA 12.1 index (compatible with the 12.3 runtime); the rest from
# PyPI. --ignore-installed blinker: the Ubuntu base ships a distutils-installed
# blinker 1.4 that pip cannot cleanly remove; this lets pinned versions install
# over it without a partial-uninstall error.
RUN python -m pip install --upgrade pip \
    && python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
        --ignore-installed blinker \
        -r /tmp/requirements.txt \
        "$ONNXRUNTIME_PKG"

# UV — used at job time by dep_scanner.py to install any third-party
# packages a user preprocessing script imports that this image doesn't ship
# (pypi.org / files.pythonhosted.org are download-only in the network policy).
COPY --from=ghcr.io/astral-sh/uv:0.7.0 /uv /bin/uv

# Runtime payload: the Go binary, the network policy it attests at boot, and
# the dependency scanner the pipeline runs before an eval with user
# preprocessing. Eval scripts themselves arrive from GCS at job time.
COPY policy/network_policy.json /app/policy/network_policy.json
COPY dep_scanner.py /app/dep_scanner.py
COPY --from=go-builder /out/processing-tee /usr/local/bin/processing-tee

EXPOSE 443

ENTRYPOINT ["/usr/local/bin/processing-tee"]
