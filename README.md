# Tool-Call Fine-Tune Lab

## Agent Reliability Audit Surface

- [Open the public GitHub Pages demo](https://kim3310.github.io/tool-call-finetune-lab/)
- Scope: credential-free, synthetic-data preview of the dataset, evaluation, and serving workflow.
- Private CTA: request a fixed-scope [Agent Reliability Audit](https://kim3310-doeon-kim-portfolio.pages.dev/?offer=tool-call-finetune-lab&inquiry=agent-reliability-audit#private-inquiry) using the central private inquiry lane.
- Demo boundary: the public page shows synthetic tool-call eval examples and implementation notes. It is not a hosted fine-tuning service, customer dataset review, production deployment claim, or benchmark guarantee.

[![CI](https://github.com/KIM3310/tool-call-finetune-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/KIM3310/tool-call-finetune-lab/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/KIM3310/tool-call-finetune-lab/branch/main/graph/badge.svg)](https://codecov.io/gh/KIM3310/tool-call-finetune-lab)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)

QLoRA fine-tuning of **Qwen2.5-7B-Instruct** for tool-calling, evaluated on [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html), served with vLLM.

## System Overview

A tool-call adaptation lab that connects training, benchmark evaluation, and serving notes into one model-readiness surface.

| Area | Details |
|---|---|
| Users | AI platform teams, model-evaluation teams, applied ML engineers, and developer-tool builders. |
| Technical path | Start with the demo, then inspect the training configuration, evaluation pipeline, architecture notes, and quality gate. |
| System scope | Open-weight training notes, benchmark evaluation, QLoRA/PyTorch framing, vLLM serving notes, and tool-calling datasets. |
| Operating boundary | Fine-tuned models need dataset licensing checks, eval coverage, and deployment controls before production use. |
| Evaluation path | Run the available evaluation scripts and inspect the training config, benchmark reports, and serving notes. |

## Evaluation Path

- **Start here:** Read the BFCL setup, dataset split, and serving notes before any GPU-heavy path.
- **Local demo:** Run `make install` and inspect configs; GPU paths are optional and explicit.
- **Checks:** Run `make verify`; run `make eval` only when the required model/eval assets are available.

## Service Launch Playbook

- [Service launch playbook](docs/service-launch-playbook.md) maps the repository to its product scope, operating gates, deployment boundaries, and risk controls.

## Architecture Notes

- [Architecture guide](docs/architecture-evidence-map.md) summarizes the system scope, first files to inspect, runtime commands, and known boundaries.
- [Quality notes](docs/quality-gate.md) lists the local checks, CI surface, and release expectations for this repository.
- [Enterprise readiness notes](docs/enterprise-readiness.md) outlines security, data, operations, integration, and handoff expectations.

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

Security/reproducibility defaults:

- Hugging Face model loads use pinned commits and default to `trust_remote_code=False`; enabling remote code requires an explicit flag and immutable `--code-revision`.
- Glaive preparation downloads a pinned dataset file through Hugging Face Hub and verifies its SHA-256 digest before parsing.
- BFCL raw downloads use a pinned Git commit, validate the final raw GitHub host/path, and require SHA-256 values for every expected shard.
- Data preparation fails closed on download errors. Synthetic fixtures require `--allow-synthetic-fixtures` and are marked in JSONL provenance sidecars.
- vLLM binds to `127.0.0.1` by default. Shared-network hosts such as `0.0.0.0` require `--allow-public-bind` and should be placed behind auth, firewall, or reverse proxy controls.

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

Default compose publishes the host port on `127.0.0.1` only. The container-internal
vLLM process may still bind `0.0.0.0` so the smoke-test service can reach it on the
Docker network.

Public host publishing uses the standalone production Compose file and requires
an API key with at least 16 characters:

```bash
export VLLM_API_KEY="$(openssl rand -hex 32)"
VLLM_PRODUCTION_HOST_BIND=0.0.0.0 docker compose \
  -f docker-compose.production.yml up vllm-server
```

The API key protects the OpenAI-compatible routes. Keep the service behind a
reverse proxy/TLS boundary and firewall rules that restrict endpoint access.

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

- [Cloud + AI architecture blueprint](docs/cloud-ai-architecture.md)
- [Machine-readable architecture manifest](docs/architecture/blueprint.json)
- Validation command: `python3 scripts/validate_architecture_blueprint.py`

## Enterprise Productization

- [Product operating model](docs/product-operating-model.md) defines the product scope, trust boundary, operating checks, and service path for this repository.

## System Architecture

- [System architecture](docs/system-architecture.md) maps the runtime boundary, data/control flow, cloud or local deployment surface, and operating assumptions for this repository.

## Service Architecture

- [Service architecture](docs/service-architecture.md) defines the cloud resources, account information, cost controls, and production guardrails needed to turn this repo into a scoped service without publishing public financial assumptions.

<!-- search-growth-readme:start -->

## Search And Service Surface

- Public entry: free open-source pipeline and sample eval reports
- Paid boundary: fixed-scope Agent Reliability Audit for private scenario suites, trace review, failure taxonomy, provider scorecard, and prioritized remediation plan
- Canonical URL: https://kim3310.github.io/tool-call-finetune-lab/
- Lead capture: https://kim3310-doeon-kim-portfolio.pages.dev/?offer=tool-call-finetune-lab&inquiry=agent-reliability-audit#private-inquiry
- Resource route: https://kim3310-doeon-kim-portfolio.pages.dev/resources/tool-call-finetune-lab/
- Commercial route: https://kim3310-doeon-kim-portfolio.pages.dev/?offer=tool-call-finetune-lab#service-offers
- Machine-readable offer: [docs/service-offer.json](docs/service-offer.json)
- Search growth implementation: [docs/search-growth-implementation.md](docs/search-growth-implementation.md)
- Revenue architecture: [docs/revenue-architecture.md](docs/revenue-architecture.md)

<!-- search-growth-readme:end -->

<!-- KIM3310:AD-DATA-PIVOT:START -->
## Free Resource, Advertising, and Aggregate Data

- [Public utility and architecture checklist](https://kim3310-doeon-kim-portfolio.pages.dev/resources/tool-call-finetune-lab/)
- Revenue model: contextual advertising on the policy-eligible central resource page.
- Aggregate value: anonymous aggregate tool-call dataset-control interest and checklist usage counts
- Boundary: ads allowed only on public dataset-boundary resources; raw examples, prompts, eval outputs, and dashboards are ad-free
- Consent defaults off, DNT/GPC fail closed, and personal or sensitive data is never sold.
<!-- KIM3310:AD-DATA-PIVOT:END -->
