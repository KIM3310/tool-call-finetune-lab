# syntax=docker/dockerfile:1.4
# Multi-stage Dockerfile for tool-call-finetune-lab
#
# Stages:
#   base      — shared CUDA + Python base
#   training  — full training dependencies (torch, peft, trl, bitsandbytes)
#   serving   — vLLM serving (lighter footprint)

ARG CUDA_VERSION=12.1.1
ARG UBUNTU_VERSION=22.04
ARG PYTHON_VERSION=3.11
# NOTE: Requires Python 3.10+ (project uses modern type hints)

# ---------------------------------------------------------------------------
# Stage 1: base — CUDA + Python + common system deps
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${UBUNTU_VERSION} AS base

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    git \
    curl \
    wget \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

RUN pip install --upgrade pip setuptools wheel

WORKDIR /app

# ---------------------------------------------------------------------------
# Stage 2: training — everything needed for QLoRA fine-tuning
# ---------------------------------------------------------------------------
FROM base AS training

COPY pyproject.toml ./
COPY src/ ./src/

# Install all training dependencies
RUN pip install -e ".[dev]" \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Copy the rest of the project
COPY . .

# Create data/output directories
RUN mkdir -p data/raw data/processed outputs/lora-adapter outputs/merged-model results

# Default: run the full pipeline
CMD ["bash", "scripts/full_pipeline.sh"]

# ---------------------------------------------------------------------------
# Stage 3: serving — minimal image for vLLM inference
# ---------------------------------------------------------------------------
FROM base AS serving

# Only install vLLM and OpenAI client for serving
RUN pip install \
    vllm>=0.6.0 \
    openai>=1.50.0 \
    httpx>=0.27.0 \
    huggingface-hub>=0.26.0

COPY src/ ./src/
COPY pyproject.toml ./

# Install only the package itself (no heavy training deps)
RUN pip install --no-deps -e .

# Model will be mounted at /model
ENV MODEL_PATH=/model \
    VLLM_PORT=8000 \
    SERVED_MODEL_NAME=qwen2.5-7b-tool-call \
    TENSOR_PARALLEL=1 \
    MAX_MODEL_LEN=4096 \
    GPU_MEMORY_UTILIZATION=0.90

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:${VLLM_PORT}/health || exit 1

CMD python3 -m tool_call_finetune_lab.serve.vllm_launcher \
    --model "${MODEL_PATH}" \
    --port "${VLLM_PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel "${TENSOR_PARALLEL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
