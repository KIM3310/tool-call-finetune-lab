[![CI](https://github.com/KIM3310/tool-call-finetune-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/KIM3310/tool-call-finetune-lab/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)

# Tool-Call Fine-Tune Lab

LoRA fine-tuning of **Qwen2.5-7B-Instruct** for reliable tool-calling, evaluated against the [Berkeley Function-Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html), served with vLLM.

## Hiring Fit And Proof Boundary

- **Best fit roles:** applied AI engineer, LLM systems engineer, eval engineer, post-training engineer
- **Strongest public proof:** checked-in Kaggle-ready notebook source, BFCL runner code, QLoRA training pipeline, and vLLM serving path
- **What is real here:** the data pipeline, LoRA training flow, BFCL evaluation harness, quantization path, and OpenAI-compatible serving
- **What is bounded here:** this is a single-node open-model post-training project, not a frontier-scale distributed training stack

## Latest Verified Snapshot

- **Verified on:** 2026-03-29
- **Command:** `make PYTHON=.venv/bin/python verify` plus a fresh `make data`
- **Outcome:** passed locally; 108 tests and Ruff checks completed, and the BFCL + Glaive data pipeline rebuilt to the documented 29,647-example split
- **Notes:** training, quantization, and vLLM serving remain optional heavier paths; the public proof path centers on data, config, eval correctness, and honest release status

## Why This Exists

[stage-pilot](https://github.com/KIM3310/stage-pilot) proves that middleware can lift tool-calling success from 25% to 90% through parser recovery and bounded retries. This project asks: **can we close the remaining 10% gap by teaching the model itself to produce better tool calls?**

We fine-tune an open model on real tool-calling data, validate against a recognized community benchmark (not a self-built test suite), and serve it through an OpenAI-compatible endpoint that plugs directly into stage-pilot's middleware.

## Status

**Training pipeline and public-release prep are checked in locally.** The Kaggle-ready notebook source is checked in locally:

- notebook source: [`notebooks/kaggle_full_pipeline.ipynb`](notebooks/kaggle_full_pipeline.ipynb)
- Kaggle kernel metadata: [`notebooks/kaggle-kernel/kernel-metadata.json`](notebooks/kaggle-kernel/kernel-metadata.json)
- public-sync helper: [`scripts/sync_kaggle_kernel.py`](scripts/sync_kaggle_kernel.py)

The checked-in kernel metadata is now configured for a **public** kernel (`is_private: false`), the attached dataset page is public, and the Kaggle kernel metadata no longer depends on a Kaggle-hosted model attachment. A fresh Kaggle republish attempt succeeded on `2026-03-29`, the public notebook page is live, and the latest remote kernel execution reached `COMPLETE` by taking the documented smoke fallback path on an unsupported accelerator.

## Current evidence

- The repo contains a full QLoRA training pipeline, adapter merge path, BFCL runner, AWQ/vLLM path, and 108 local tests.
- A fresh local data rebuild on 2026-03-29 reproduced the documented split: 23,716 train / 2,962 val / 2,969 test from 29,647 deduplicated examples.
- The latest authenticated Kaggle push reached **kernel version 21** and the public remote run reached `COMPLETE` on `2026-03-29`.
- The checked-in Kaggle notebook demonstrates a **first-100-example BFCL smoke eval with loose function-name matching**, not a full strict benchmark artifact.
- On unsupported Kaggle accelerators, the notebook now completes via a smoke fallback that validates data loading, tokenizer setup, formatting, tokenization, and results emission without entering the incompatible 4-bit training path.
- The repo now mirrors the completed public Kaggle smoke output:
  - [`results/kaggle_public_smoke_bfcl_results.json`](results/kaggle_public_smoke_bfcl_results.json)
- The repo now includes a checked-in evaluator smoke artifact and release-status ledger:
  - [`results/eval_harness_smoke.json`](results/eval_harness_smoke.json)
  - [`results/eval_harness_smoke.md`](results/eval_harness_smoke.md)
  - [`results/public_release_status.json`](results/public_release_status.json)
  - [`results/public_release_status.md`](results/public_release_status.md)
- `results/bfcl_results.json` is still **pending a re-run with actual fine-tuned weights**, so it should not be treated as public proof yet.

### Public-proof gaps still open

- The public Kaggle notebook page is live and the latest remote execution completed, but that public completion used the smoke fallback path rather than a full QLoRA training run.
- A Hugging Face token plus a reachable model directory is required before the LoRA or AWQ artifacts can become public links.
- A full strict BFCL artifact remains pending until the fine-tuned model weights are available again for re-evaluation.

### Verified locally on 2026-03-29

- BFCL v4 and Glaive v2 were downloaded again and rebuilt into the documented train/val/test split.
- The Kaggle kernel folder can now be re-synced locally with public visibility defaults via `python scripts/sync_kaggle_kernel.py --public`.
- The release ledger confirms the current external blockers instead of assuming those links are public.

## Training Details

| Item | Detail |
|------|--------|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Method | QLoRA — 4-bit NF4 quantized base + fp16 LoRA adapters |
| LoRA config | rank=16, alpha=32, dropout=0.05, targets: q/k/v/o_proj |
| Data | 29,647 examples (BFCL v4 + Glaive-function-calling-v2), 80/10/10 split |
| Hardware | Kaggle T4 16GB (QLoRA), also runnable on A100 80GB |
| Training | 1 epoch, lr=2e-4, batch=1 x grad_accum=8, gradient checkpointing |
| Tracking | Weights & Biases (optional) |

## Data Pipeline

```
BFCL v4 (GitHub)          Glaive v2 (HuggingFace)
  2,501 examples              63,218 examples
        \                        /
         \                      /
          merge + dedup + split
                  |
        29,647 total examples
     train: 23,716 (80%)
     val:    2,962 (10%)
     test:   2,969 (10%)
```

- **BFCL**: 10 categories (simple, multiple, parallel, live variants) with ground-truth answers from the official `possible_answer/` files
- **Glaive**: Multi-turn function-calling conversations with tool definitions
- Deduplication by SHA-256 content hash, stratified split by (source, category)

## Architecture

```
stage-pilot copilot
        │
        ▼
   OpenAI-compatible
   tool-call API
        │
   ┌────┴────┐
   │  vLLM   │  ← AWQ INT4 quantized
   │ server  │
   └────┬────┘
        │
   Qwen2.5-7B-Instruct
   + LoRA adapters
   (merged or loaded via PEFT)
```

The fine-tuned model serves through vLLM's OpenAI-compatible endpoint, so stage-pilot's `@ai-sdk-tool/parser` middleware works against it with zero code changes.

## Quick Start

```bash
# Clone and install (lightweight — GPU deps are optional extras)
git clone https://github.com/KIM3310/tool-call-finetune-lab
cd tool-call-finetune-lab
pip install -e ".[dev]"

# Prepare data (runs on any machine, no GPU needed)
make data

# Train on GPU machine
pip install -e ".[gpu]"
make train

# Evaluate
make eval

# Quantize + serve
make quantize
make serve
```

Or run the full pipeline from the checked-in Kaggle notebook source:

```bash
python scripts/sync_kaggle_kernel.py --public
```

## Published Artifacts

| Artifact | Link | Status |
|----------|------|--------|
| Kaggle kernel | [kaggle.com/code/doeonkim00/tool-call-fine-tune-lab-qlora-pipeline](https://www.kaggle.com/code/doeonkim00/tool-call-fine-tune-lab-qlora-pipeline) | Public page live; kernel version 21 completed on 2026-03-29 via the documented smoke fallback path |
| Kaggle smoke result mirror | [`results/kaggle_public_smoke_bfcl_results.json`](results/kaggle_public_smoke_bfcl_results.json) | Checked in from the completed public Kaggle run; smoke-only fallback artifact, not a full benchmark |
| LoRA adapter | [huggingface.co/KIM3310/qwen2.5-7b-tool-calling-lora](https://huggingface.co/KIM3310/qwen2.5-7b-tool-calling-lora) | Not publicly reachable on 2026-03-29; push config is public-ready |
| AWQ quantized | [huggingface.co/KIM3310/qwen2.5-7b-tool-calling-awq](https://huggingface.co/KIM3310/qwen2.5-7b-tool-calling-awq) | Not publicly reachable on 2026-03-29; push config is public-ready |
| Eval harness smoke | [`results/eval_harness_smoke.json`](results/eval_harness_smoke.json) | Checked in |
| Release ledger | [`results/public_release_status.md`](results/public_release_status.md) | Checked in |
| BFCL full results | [`results/bfcl_results.json`](results/bfcl_results.json) | Pending re-run with actual fine-tuned weights |
| W&B training log | [Weights & Biases run](https://wandb.ai/KIM3310/tool-call-finetune-lab) | External link present; public access should be reverified |

## Repository Layout

```
src/tool_call_finetune_lab/
  config.py              # All hyperparameters as dataclasses
  data/                  # BFCL + Glaive download, parse, merge, split
  train/                 # QLoRA trainer + adapter merge
  eval/                  # BFCL runner, stage-pilot bridge, comparison table
  quantize/              # AWQ quantization + inference benchmark
  serve/                 # vLLM launcher + OpenAI-compat smoke test
notebooks/
  kaggle_full_pipeline.ipynb   # Self-contained Kaggle T4 notebook
scripts/                # download base model, push to hub, full pipeline
tests/                  # 108 tests covering data, config, eval harness
```

## Related Projects

This project is part of a two-pronged approach to tool-calling reliability. [stage-pilot](https://github.com/KIM3310/stage-pilot) handles reliability at the middleware layer (25% to 90% success rate through parser recovery and bounded retries). This repo explores closing the remaining gap through model fine-tuning, teaching the model to produce well-formed tool calls natively so the middleware has less recovery work to do.

## License

Apache-2.0 — see [LICENSE](LICENSE).
