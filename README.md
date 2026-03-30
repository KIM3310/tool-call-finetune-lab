[![CI](https://github.com/KIM3310/tool-call-finetune-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/KIM3310/tool-call-finetune-lab/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/KIM3310/tool-call-finetune-lab/branch/main/graph/badge.svg)](https://codecov.io/gh/KIM3310/tool-call-finetune-lab)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)

# Tool-Call Fine-Tune Lab

LoRA fine-tuning of **Qwen2.5-7B-Instruct** for reliable tool-calling, evaluated against the [Berkeley Function-Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html), served with vLLM.

## Why This Exists

[stage-pilot](https://github.com/KIM3310/stage-pilot) proves that middleware can lift tool-calling success from 25% to 90% through parser recovery and bounded retries. This project asks: **can we close the remaining 10% gap by teaching the model itself to produce better tool calls?**

We fine-tune an open model on real tool-calling data, validate against a recognized community benchmark (not a self-built test suite), and serve it through an OpenAI-compatible endpoint that plugs directly into stage-pilot's middleware.

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

### Installation

```bash
# Clone and install (lightweight — GPU deps are optional extras)
git clone https://github.com/KIM3310/tool-call-finetune-lab
cd tool-call-finetune-lab
pip install -e ".[dev]"

# Or with GPU dependencies for training/serving:
pip install -e ".[gpu,dev]"
```

### Data Preparation (no GPU needed)

```bash
make data
```

This downloads BFCL v4 from GitHub and Glaive v2 from HuggingFace, merges, deduplicates, and splits into train/val/test.

### Training (requires GPU)

```bash
make train                  # QLoRA fine-tuning
make merge                  # Merge LoRA adapter into base model
```

### Evaluation

```bash
make eval                   # Run BFCL evaluation + comparison table
```

### Quantization & Serving

```bash
make quantize               # AWQ INT4 quantization
make serve                  # Launch vLLM server
make smoke-test             # Smoke test the running server
```

### Full Pipeline

```bash
make pipeline               # data -> train -> merge -> eval -> quantize
```

### Docker Deployment

```bash
docker compose up vllm-server          # Serve the quantized model
docker compose run smoke-test          # Run smoke tests
```

### Run on Kaggle

Or run the full pipeline on Kaggle: [notebook link](https://www.kaggle.com/code/doeonkim00/tool-call-fine-tune-lab-qlora-pipeline)

## Development

```bash
make check                  # Run all checks (lint + typecheck + test)
make test                   # Unit tests only
make test-cov               # Tests with coverage
make lint                   # Ruff linter
make format                 # Auto-format code
make typecheck              # mypy type checking
make help                   # Show all available commands
```

## Published Artifacts

| Artifact | Link |
|----------|------|
| Kaggle kernel | [kaggle.com/code/doeonkim00/tool-call-fine-tune-lab-qlora-pipeline](https://www.kaggle.com/code/doeonkim00/tool-call-fine-tune-lab-qlora-pipeline) |
| LoRA adapter | [huggingface.co/KIM3310/qwen2.5-7b-tool-calling-lora](https://huggingface.co/KIM3310/qwen2.5-7b-tool-calling-lora) |
| AWQ quantized | [huggingface.co/KIM3310/qwen2.5-7b-tool-calling-awq](https://huggingface.co/KIM3310/qwen2.5-7b-tool-calling-awq) |
| W&B training log | [Weights & Biases run](https://wandb.ai/KIM3310/tool-call-finetune-lab) |

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
scripts/                # download base model, push to hub
tests/                  # Unit tests covering data, config, eval harness
```

## Requirements

- **Python 3.10+** (3.11 recommended)
- **CPU-only** for data preparation, evaluation code review, and tests
- **GPU (16+ GB VRAM)** for training (T4 minimum, A100 recommended)
- **GPU (8+ GB VRAM)** for AWQ-quantized inference via vLLM

## Related Projects

This project is part of a two-pronged approach to tool-calling reliability. [stage-pilot](https://github.com/KIM3310/stage-pilot) handles reliability at the middleware layer (25% to 90% success rate through parser recovery and bounded retries). This repo explores closing the remaining gap through model fine-tuning, teaching the model to produce well-formed tool calls natively so the middleware has less recovery work to do.

## License

Apache-2.0 — see [LICENSE](LICENSE).
