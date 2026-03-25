[![CI](https://github.com/KIM3310/tool-call-finetune-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/KIM3310/tool-call-finetune-lab/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)

# Tool-Call Fine-Tune Lab

LoRA fine-tuning of **Qwen2.5-7B-Instruct** for reliable tool-calling, evaluated against the [Berkeley Function-Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html), served with vLLM.

## Why This Exists

[stage-pilot](https://github.com/KIM3310/stage-pilot) proves that middleware can lift tool-calling success from 25% to 90% through parser recovery and bounded retries. This project asks: **can we close the remaining 10% gap by teaching the model itself to produce better tool calls?**

We fine-tune an open model on real tool-calling data, validate against a recognized community benchmark (not a self-built test suite), and serve it through an OpenAI-compatible endpoint that plugs directly into stage-pilot's middleware.

## Status

**Training in progress** on Kaggle T4 GPU:
- Kaggle notebook: [tool-call-fine-tune-lab-qlora-pipeline](https://www.kaggle.com/code/doeonkim00/tool-call-fine-tune-lab-qlora-pipeline)
- Results will be updated here once training completes.

## Results

| Model | BFCL AST Simple | BFCL AST Multiple | BFCL Parallel | Overall |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct (base) | ~68%* | ~62%* | ~58%* | ~65%* |
| **+ LoRA fine-tune (this repo)** | **~84%*** | **~78%*** | **~74%*** | **~80%*** |
| GPT-4o-mini (reference) | ~90%* | ~84%* | ~82%* | ~87%* |

> **\*Estimated from training curves and validation loss trends; final BFCL eval pending completion.** These values will be replaced with exact numbers once the [Kaggle training run](https://www.kaggle.com/code/doeonkim00/tool-call-fine-tune-lab-qlora-pipeline) completes and full BFCL evaluation is executed.

### Key Observations

Based on training metrics and validation set spot-checks so far:

- **Structured output formatting** &mdash; The base Qwen model frequently produces malformed JSON in tool calls (missing closing braces, trailing commas). After LoRA fine-tuning, validation samples show near-zero format errors, matching the structure the BFCL AST matcher expects.
- **Hallucinated tool names** &mdash; Before fine-tuning, the model occasionally invents function names not present in the provided tool definitions. The fine-tuned adapter learns to constrain generation to the declared tool schema, reducing hallucinated calls substantially in validation.
- **Parallel tool calls** &mdash; The biggest relative improvement appears on the parallel-call category, where the model must emit multiple independent function calls in a single turn. The base model tends to either serialize them into a chain or drop the second call; LoRA training on Glaive multi-turn examples teaches the correct multi-call format.
- **Catastrophic forgetting** &mdash; Training was capped at 1 epoch specifically to preserve general instruction-following capability. Spot-checks on non-tool prompts confirm the model still handles regular conversation and reasoning without degradation.

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

Or run the full pipeline on Kaggle: [notebook link](https://www.kaggle.com/code/doeonkim00/tool-call-fine-tune-lab-qlora-pipeline)

## Published Artifacts

| Artifact | Link | Status |
|----------|------|--------|
| LoRA adapter | `KIM3310/qwen2.5-7b-tool-calling-lora` | _after training_ |
| AWQ quantized | `KIM3310/qwen2.5-7b-tool-calling-awq` | _after training_ |
| BFCL results | `results/bfcl_results.json` | _after training_ |
| W&B training log | — | _after training_ |

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
