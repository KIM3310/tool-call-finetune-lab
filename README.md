# Tool-Call Fine-Tune Lab

A lab for preparing tool-call training data and testing evaluation correctness. It includes optional QLoRA training, adapter merging, quantization, and vLLM serving paths. The public reproduction path runs entirely on a CPU and makes its synthetic evidence explicit.

## Start with the proof

```bash
make install
make verify
make proof
```

Requires Python 3.10 or newer. [`evidence/evaluation-contract.json`](evidence/evaluation-contract.json) is generated from real execution of the scorer and splitter against synthetic cases, with source-file fingerprints. `make verify` reproduces that artifact exactly.

| Engineering decision | Implementation and regression evidence |
|---|---|
| Exclude target answers and future tool results while preserving earlier conversation context | [`eval/bfcl_runner.py`](src/tool_call_finetune_lab/eval/bfcl_runner.py), [`tests/test_evaluation_integrity.py`](tests/test_evaluation_integrity.py) |
| Compare complete JSON arguments without coercing strings or booleans | The same scorer rejects extra arguments, changed identifiers, wrong types, and malformed calls |
| Preserve distinct labels while keeping matching requests in one partition | [`data/merge_and_split.py`](src/tool_call_finetune_lab/data/merge_and_split.py) groups matching request/tool inputs across sources |
| Refuse incompatible comparisons | [`eval/compare.py`](src/tool_call_finetune_lab/eval/compare.py) checks dataset and scorer fingerprints; legacy results are marked unverified |
| Make skipped samples and truncated failure logs visible | Evaluation metadata records input, evaluated, skipped, and failed counts |

## Evaluation contract

This is a strict local normalized-tool-call scorer. It does **not** implement the complete [official BFCL AST/execution/multi-turn evaluator](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html), and its percentages must not be presented as official BFCL leaderboard scores. Input examples without a tool-call target are outside this scorer's evaluation scope.

The evaluator targets the last assistant tool-call turn, keeps the earlier conversation, and excludes that target and everything after it. Function names and string values are case-sensitive; complete argument objects must match. Object key order is irrelevant, list order matters, and equivalent JSON numbers may match. Invalid arguments do not become an empty object.

## Data integrity

Deduplication uses the complete conversation and tool definitions, retaining different labels that the former prompt/function-name key could discard. Splitting groups matching system/user requests and tool schemas into a single partition across sources. The fixed-seed assignment is independent of input order. Group constraints take priority over exact source/category proportions.

Missing files and malformed JSONL fail explicitly. Matching input hashes reduce exact-request leakage; they do not detect paraphrases, semantic duplicates, or exposure in a base model's pretraining data.

## Optional model pipeline

The configured path uses Qwen2.5-7B-Instruct with QLoRA, followed by optional adapter merge, AWQ quantization, and vLLM serving. See [the implementation reference](REFERENCE.md#training), [`config.py`](src/tool_call_finetune_lab/config.py), and [`train/lora_trainer.py`](src/tool_call_finetune_lab/train/lora_trainer.py).

```bash
make install-gpu
make pipeline
make serve
```

These paths require compatible GPU hardware, model/data assets, and any necessary credentials. Pipeline stages run sequentially, including under `make -j`. The vLLM evaluator honors its timeout and `VLLM_API_KEY`.

Container serving binds to loopback by default. The explicit [`docker-compose.production.yml`](docker-compose.production.yml) override requires auth, a configured reverse proxy, and firewall restrictions before public exposure. See [the serving reference](REFERENCE.md#docker).

No GPU training, checkpoint validation, or live-provider quality measurement was performed in this upgrade. Historical private model artifacts are not evidence available through this public repository. The checked-in contract report proves evaluation/data-processing behavior, not a model-quality improvement.

[Engineering notes](docs/engineering-notes.md) · [Full historical setup reference](REFERENCE.md) · [Source](https://github.com/KIM3310/tool-call-finetune-lab)

[Cloud architecture](docs/cloud-ai-architecture.md) · [Blueprint](docs/architecture/blueprint.json) · [Blueprint validator](scripts/validate_architecture_blueprint.py)
