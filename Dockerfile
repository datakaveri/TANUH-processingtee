FROM golang:1.22-bullseye AS ratls-builder
WORKDIR /src
COPY b2p-ratls/go.mod b2p-ratls/go.sum ./b2p-ratls/
WORKDIR /src/b2p-ratls
RUN go mod download
COPY b2p-ratls/ ./
RUN CGO_ENABLED=0 GOOS=linux go build -o /out/gpu-cs ./cmd/gpu-cs/

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    BASE_DIR=/app \
    DEBIAN_FRONTEND=noninteractive \
    RATLS_AUDIENCE=ratls-buffer-tee \
    LISTEN_ADDR=:443 \
    PROCESSING_MANAGER_JOB_URL=http://127.0.0.1:4000/enclave/cvm/secure-job

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements.txt

COPY . /app
COPY --from=ratls-builder /out/gpu-cs /usr/local/bin/gpu-cs

RUN chmod +x /app/entrypoint.sh /usr/local/bin/gpu-cs \
    && mkdir -p /app/cvm_workflow/artifacts /app/cvm_workflow/incoming /app/cvm_workflow/runtime /app/cvm_workflow/secure_jobs

EXPOSE 4000 443

ENTRYPOINT ["/app/entrypoint.sh"]
