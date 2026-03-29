#!/usr/bin/env bash
# full_pipeline.sh — Run the complete tool-call fine-tuning pipeline:
#   data → train → eval (base) → quantize → serve smoke-test → eval (fine-tuned)
#
# Usage:
#   bash scripts/full_pipeline.sh
#   BASE_MODEL=Qwen/Qwen2.5-7B-Instruct bash scripts/full_pipeline.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — override via environment variables
# ---------------------------------------------------------------------------
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
ADAPTER_DIR="${OUTPUT_DIR}/lora-adapter"
MERGED_DIR="${OUTPUT_DIR}/merged-model"
AWQ_DIR="${OUTPUT_DIR}/awq-model"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-qwen2.5-7b-tool-call}"
LOG_DIR="${OUTPUT_DIR}/logs"
RUN_SERVE="${RUN_SERVE:-true}"
SKIP_BASE_EVAL="${SKIP_BASE_EVAL:-false}"

PYTHON="${PYTHON:-python3}"

mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

check_env() {
    if [[ -z "${HF_TOKEN:-}" ]]; then
        die "HF_TOKEN is not set. Export it or add it to .env before running."
    fi
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        log "WARNING: WANDB_API_KEY not set — W&B logging will be disabled."
    fi
}

wait_for_vllm() {
    local port="${1:-$VLLM_PORT}"
    local max_wait=120
    local waited=0
    log "Waiting for vLLM server on port ${port}..."
    until curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; do
        sleep 2
        waited=$((waited + 2))
        if [[ $waited -ge $max_wait ]]; then
            die "vLLM server did not start within ${max_wait}s"
        fi
    done
    log "vLLM server is up."
}

# Load .env if present
if [[ -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
    log "Loaded .env"
fi

# ---------------------------------------------------------------------------
# Step 0: Pre-flight checks
# ---------------------------------------------------------------------------
log "=== STEP 0: Pre-flight checks ==="
check_env

# Verify GPU is available
if ! $PYTHON -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU found'" 2>/dev/null; then
    die "No CUDA GPU available. This pipeline requires at least one GPU."
fi
GPU_NAME=$($PYTHON -c "import torch; print(torch.cuda.get_device_name(0))")
log "GPU: ${GPU_NAME}"

# ---------------------------------------------------------------------------
# Step 1: Download base model (skip if already present)
# ---------------------------------------------------------------------------
log "=== STEP 1: Download base model ==="
if [[ ! -f "models/base/config.json" ]]; then
    $PYTHON scripts/download_base_model.py \
        --model "${BASE_MODEL}" \
        --dest models/base \
        2>&1 | tee "${LOG_DIR}/download.log"
else
    log "Base model already present at models/base — skipping download."
fi

# ---------------------------------------------------------------------------
# Step 2: Prepare data
# ---------------------------------------------------------------------------
log "=== STEP 2: Prepare data ==="
$PYTHON -m tool_call_finetune_lab.data.prepare_bfcl \
    2>&1 | tee "${LOG_DIR}/prepare_bfcl.log"

$PYTHON -m tool_call_finetune_lab.data.prepare_glaive \
    2>&1 | tee "${LOG_DIR}/prepare_glaive.log"

$PYTHON -m tool_call_finetune_lab.data.merge_and_split \
    2>&1 | tee "${LOG_DIR}/merge_split.log"

# ---------------------------------------------------------------------------
# Step 3: (Optional) Evaluate base model for comparison
# ---------------------------------------------------------------------------
if [[ "${SKIP_BASE_EVAL}" != "true" ]]; then
    log "=== STEP 3: Evaluate base model ==="
    $PYTHON -m tool_call_finetune_lab.eval.bfcl_runner \
        --mode local \
        --model-path "models/base" \
        --results-file "results/bfcl_base_results.json" \
        --max-examples 200 \
        2>&1 | tee "${LOG_DIR}/eval_base.log" || log "Base eval failed — continuing"
else
    log "=== STEP 3: Skipping base model eval (SKIP_BASE_EVAL=true) ==="
fi

# ---------------------------------------------------------------------------
# Step 4: Fine-tune with QLoRA
# ---------------------------------------------------------------------------
log "=== STEP 4: QLoRA fine-tuning ==="
$PYTHON -m tool_call_finetune_lab.train.lora_trainer \
    --base-model "${BASE_MODEL}" \
    --output-dir "${ADAPTER_DIR}" \
    2>&1 | tee "${LOG_DIR}/train.log"

# ---------------------------------------------------------------------------
# Step 5: Merge adapter into base model
# ---------------------------------------------------------------------------
log "=== STEP 5: Merge LoRA adapter ==="
$PYTHON -m tool_call_finetune_lab.train.merge_adapter \
    --base-model "${BASE_MODEL}" \
    --adapter-path "${ADAPTER_DIR}" \
    --output-path "${MERGED_DIR}" \
    2>&1 | tee "${LOG_DIR}/merge.log"

# ---------------------------------------------------------------------------
# Step 6: AWQ quantization
# ---------------------------------------------------------------------------
log "=== STEP 6: AWQ quantization ==="
$PYTHON -m tool_call_finetune_lab.quantize.awq_quantize \
    --model-path "${MERGED_DIR}" \
    --output-path "${AWQ_DIR}" \
    --calib-data "data/processed/train.jsonl" \
    2>&1 | tee "${LOG_DIR}/quantize.log"

# ---------------------------------------------------------------------------
# Step 7: Start vLLM server and run evals
# ---------------------------------------------------------------------------
if [[ "${RUN_SERVE}" == "true" ]]; then
    log "=== STEP 7: Start vLLM server ==="
    $PYTHON -m tool_call_finetune_lab.serve.vllm_launcher \
        --model "${AWQ_DIR}" \
        --port "${VLLM_PORT}" \
        --served-model-name "${VLLM_MODEL_NAME}" \
        2>&1 | tee "${LOG_DIR}/vllm.log" &
    VLLM_PID=$!
    log "vLLM server PID: ${VLLM_PID}"

    wait_for_vllm "${VLLM_PORT}"

    # Smoke test
    log "=== STEP 7a: OpenAI-compat smoke test ==="
    $PYTHON -m tool_call_finetune_lab.serve.openai_compat_test \
        --url "http://localhost:${VLLM_PORT}/v1" \
        --model "${VLLM_MODEL_NAME}" \
        2>&1 | tee "${LOG_DIR}/smoke_test.log" || log "Smoke test had failures — check log"

    # BFCL eval against running server
    log "=== STEP 7b: BFCL eval (fine-tuned via vLLM) ==="
    $PYTHON -m tool_call_finetune_lab.eval.bfcl_runner \
        --mode vllm \
        --vllm-url "http://localhost:${VLLM_PORT}/v1" \
        --model-name "${VLLM_MODEL_NAME}" \
        --results-file "results/bfcl_results.json" \
        2>&1 | tee "${LOG_DIR}/eval_finetuned.log"

    # Stage-pilot bridge eval
    log "=== STEP 7c: Stage-pilot bridge eval ==="
    $PYTHON -m tool_call_finetune_lab.eval.stage_pilot_bridge \
        --vllm-url "http://localhost:${VLLM_PORT}/v1" \
        --vllm-model "${VLLM_MODEL_NAME}" \
        2>&1 | tee "${LOG_DIR}/bridge_eval.log"

    # Inference benchmark
    log "=== STEP 7d: Inference benchmark ==="
    $PYTHON -m tool_call_finetune_lab.quantize.benchmark_inference \
        --url "http://localhost:${VLLM_PORT}/v1" \
        --model "${VLLM_MODEL_NAME}" \
        --n-requests 50 \
        2>&1 | tee "${LOG_DIR}/benchmark.log"

    # Stop vLLM
    log "Stopping vLLM server..."
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Step 8: Generate comparison table
# ---------------------------------------------------------------------------
log "=== STEP 8: Generate comparison table ==="
$PYTHON -m tool_call_finetune_lab.eval.compare \
    --finetuned-results "results/bfcl_results.json" \
    --base-results "results/bfcl_base_results.json" \
    --output "results/comparison.md" \
    2>&1 | tee "${LOG_DIR}/compare.log"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
log "=== Pipeline complete ==="
log ""
log "Outputs:"
log "  Adapter:     ${ADAPTER_DIR}"
log "  Merged:      ${MERGED_DIR}"
log "  AWQ model:   ${AWQ_DIR}"
log "  BFCL results: results/bfcl_results.json"
log "  Comparison:  results/comparison.md"
log ""
log "To push to HuggingFace Hub:"
log "  python scripts/push_to_hub.py --model-path ${MERGED_DIR} --repo-id KIM3310/qwen2.5-7b-tool-calling-lora"
log "  python scripts/push_to_hub.py --model-path ${AWQ_DIR} --repo-id KIM3310/qwen2.5-7b-tool-calling-awq"
