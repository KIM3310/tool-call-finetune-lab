[![CI](https://github.com/KIM3310/tool-call-finetune-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/KIM3310/tool-call-finetune-lab/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/KIM3310/tool-call-finetune-lab/branch/main/graph/badge.svg)](https://codecov.io/gh/KIM3310/tool-call-finetune-lab)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)

# Tool-Call Fine-Tune Lab

QLoRA fine-tuning of **Qwen2.5-7B-Instruct** for tool-calling, evaluated on [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html), served with vLLM.

## Product and Review Surface

A tool-call adaptation lab that connects training, benchmark evaluation, and serving notes into one model-readiness surface.

| Lens | Definition |
|---|---|
| Buyer or user | AI platform teams, model-evaluation teams, applied ML engineers, and developer-tool builders. |
| Commercial route | Sell model adaptation studies, BFCL-style evaluation packs, QLoRA training setup, and serving-readiness reviews. |
| Review signal | Open-weight training notes, benchmark evaluation, QLoRA/PyTorch framing, vLLM serving notes, and tool-calling datasets. |
| Safety boundary | Fine-tuned models need dataset licensing checks, eval coverage, and deployment controls before production use. |
| Fast proof | Run the available evaluation scripts and inspect training config, benchmark reports, and serving notes. |

## Reviewer Fast Path

- **First minute:** Read the BFCL setup, dataset split, and serving notes before any GPU-heavy path.
- **Local demo:** Run `make install` and inspect configs; GPU paths are optional and explicit.
- **Verification:** Run `make verify`; run `make eval` only when the required model/eval assets are available.
- **Commercial read:** Sell it as model adaptation, benchmark evaluation, and serving-readiness work for tool-calling systems.

## Commercialization Playbook

- [Monetization and GTM playbook](docs/monetization-playbook.md) maps the repository to buyer segments, offer ladder, pricing hypotheses, proof gates, and risk boundaries.

## Review Notes

- [Review guide](docs/reviewer-evidence-map.md) summarizes the project angle, first files to inspect, verification commands, and known boundaries.
- [Quality notes](docs/quality-gate.md) lists the local checks, CI surface, and release expectations for this repository.
- [Revenue growth model](docs/revenue-growth-model.md) maps the project to an ethical revenue path, activation loop, pricing logic, and growth experiments.
- [Enterprise readiness notes](docs/enterprise-readiness.md) outlines security, data, operations, integration, and handoff expectations.
- [Conversion UX model](docs/conversion-ux-model.md) maps the buyer path, behavioral design, UI/UX direction, pricing frame, and ethical conversion guardrails.

## Motivation

[stage-pilot](https://github.com/KIM3310/stage-pilot) gets tool-calling success from 25% to 90% via middleware (parser recovery + retries). This repo tries to close the remaining gap by fine-tuning the model itself to produce better tool calls in the first place.

We use BFCL as the eval benchmark (not a homebrew test suite) and serve through an OpenAI-compatible endpoint that plugs into stage-pilot with zero code changes.

## Training

| Item | Detail |
|------|--------|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Method | QLoRA (4-bit NF4 + fp16 LoRA adapters) |
| LoRA config | rank=16, alpha=32, dropout=0.05, targets: q/k/v/o_proj |
| Data | 29,647 examples (BFCL v4 + Glaive-function-calling-v2), 80/10/10 split |
| Hardware | Kaggle T4 16GB (QLoRA), also works on A100 80GB |
| Training | 1 epoch, lr=2e-4, batch=1 x grad_accum=8, gradient checkpointing |
| Tracking | W&B (optional) |

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

- **BFCL**: 10 categories (simple, multiple, parallel, live variants) with ground-truth from `possible_answer/`
- **Glaive**: multi-turn function-calling conversations with tool definitions
- Dedup by SHA-256 hash, stratified split by (source, category)

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

## Quick Start

```bash
# install
git clone https://github.com/KIM3310/tool-call-finetune-lab
cd tool-call-finetune-lab
pip install -e ".[dev]"

# with GPU deps:
pip install -e ".[gpu,dev]"
```

```bash
make data                   # download + merge + split (no GPU)
make train                  # QLoRA fine-tuning
make merge                  # merge LoRA into base
make eval                   # BFCL eval + comparison table
make quantize               # AWQ INT4
make serve                  # vLLM server
make smoke-test             # hit the server
make pipeline               # data -> train -> merge -> eval -> quantize
```

### Docker

```bash
docker compose up vllm-server
docker compose run smoke-test
```

### Kaggle

Full pipeline notebook: [kaggle.com/code/doeonkim00/tool-call-fine-tune-lab-qlora-pipeline](https://www.kaggle.com/code/doeonkim00/tool-call-fine-tune-lab-qlora-pipeline)

## Dev

```bash
make check                  # lint + typecheck + test
make test                   # pytest
make test-cov               # with coverage
make lint                   # ruff
make format                 # auto-format
make typecheck              # mypy
make help                   # all targets
```

## Artifacts

| Artifact | Link |
|----------|------|
| Kaggle kernel | [kaggle.com/code/doeonkim00/tool-call-fine-tune-lab-qlora-pipeline](https://www.kaggle.com/code/doeonkim00/tool-call-fine-tune-lab-qlora-pipeline) |
| LoRA adapter | [huggingface.co/KIM3310/qwen2.5-7b-tool-calling-lora](https://huggingface.co/KIM3310/qwen2.5-7b-tool-calling-lora) |
| AWQ quantized | [huggingface.co/KIM3310/qwen2.5-7b-tool-calling-awq](https://huggingface.co/KIM3310/qwen2.5-7b-tool-calling-awq) |
| W&B run | [wandb.ai/KIM3310/tool-call-finetune-lab](https://wandb.ai/KIM3310/tool-call-finetune-lab) |

## Layout

```
src/tool_call_finetune_lab/
  config.py              # hyperparams (dataclasses)
  data/                  # BFCL + Glaive download, parse, merge, split
  train/                 # QLoRA trainer + adapter merge
  eval/                  # BFCL runner, stage-pilot bridge, comparison
  quantize/              # AWQ quantization + inference bench
  serve/                 # vLLM launcher + smoke test
notebooks/
  kaggle_full_pipeline.ipynb
scripts/                 # download base model, push to hub
tests/
```

## Requirements

- Python 3.10+ (3.11 recommended)
- CPU-only for data prep, eval code, tests
- GPU 16+ GB VRAM for training (T4 minimum)
- GPU 8+ GB VRAM for AWQ inference via vLLM

## See Also

[stage-pilot](https://github.com/KIM3310/stage-pilot) -- middleware-level tool-calling reliability (25% -> 90% via parser recovery and retries). This repo is the model-level complement: teach the model to produce well-formed tool calls so middleware has less work to do.

## License

Apache-2.0 -- see [LICENSE](LICENSE).

## Cloud + AI Architecture

This repository includes a neutral cloud and AI engineering blueprint that maps the current proof surface to runtime boundaries, data contracts, model-risk controls, deployment posture, and validation hooks.

- [Cloud + AI architecture blueprint](docs/cloud-ai-architecture.md)
- [Machine-readable architecture manifest](docs/architecture/blueprint.json)
- Validation command: `python3 scripts/validate_architecture_blueprint.py`
