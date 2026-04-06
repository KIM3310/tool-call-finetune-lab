# Evaluation Methodology

How we evaluate the fine-tuned model and why we use BFCL.

## Why BFCL

Tool-calling eval is tricky: unlike normal NLP benchmarks there's no single correct answer. A tool call can be "correct" with different arg orderings, equivalent types, optional params present or absent. Rolling our own eval would need to handle all this, and the numbers wouldn't be comparable to anything.

BFCL works because:

1. **Recognized benchmark**: maintained by the Gorilla project (UC Berkeley), used to rank open and closed models side by side. Our numbers are directly comparable to GPT-4o, Llama, etc. on the public leaderboard.
2. **AST-based matching**: not string comparison. Parses tool calls into ASTs so arg reordering, whitespace, type coercion are handled automatically.
3. **Reproducible**: eval data, ground truth, and scoring scripts are all public.

## BFCL Categories

### Simple Function Calling
One query, one correct function call. Tests basic arg extraction and function selection.

### Multiple Function Calling
One query, multiple tools available, model picks the right one. Tests tool selection.

### Parallel Function Calling
Model has to emit multiple independent calls in one turn. Hardest category -- model needs to recognize multiple actions are needed and not serialize them into a chain.

### Other Categories
Java/JS function signatures, REST API calls, relevance detection (when NOT to call a tool), live API schemas. We focus on Simple/Multiple/Parallel since those are most relevant for stage-pilot.

## AST Matching

String comparison would fail `get_weather(city="SF", unit="f")` vs `get_weather(unit="f", city="SF")` even though they're identical. BFCL handles this by:

1. Parsing both output and ground truth with `ast.parse`
2. Normalizing: sort kwargs alphabetically, resolve type aliases, standardize numbers
3. Comparing normalized trees

Handles: arg ordering, whitespace, type representations (`True`/`true`, `1.0`/`1`), quoting style.

## Eval Pipeline

```
BFCL test prompts
      |
  vLLM endpoint (fine-tuned model)
      |
  raw model outputs
      |
  BFCL AST evaluator (parse, normalize, compare)
      |
  per-category accuracy
      |
  comparison table (vs base model, vs GPT-4o-mini)
```

Three configs evaluated:
1. **Qwen2.5-7B-Instruct (base)**: pre-training baseline
2. **Qwen2.5-7B + LoRA**: fine-tuned, shows the improvement
3. **GPT-4o-mini (reference)**: upper-bound calibration point, not a direct competitor

## Reading the Results

- **AST Simple**: basic competence. Most instruction-tuned models do OK here.
- **AST Multiple**: tool selection. Errors = picked the wrong function.
- **Parallel**: where open models struggle most, and where LoRA fine-tuning helps the most (largest relative gain).
- **Overall**: weighted average.

80%+ overall is competitive with commercial APIs for practical deployments, especially with stage-pilot's recovery layer on top.

## Limitations

- BFCL is single-turn only. Multi-turn (incorporate tool results, make follow-up calls) isn't covered.
- AST matching is Python-centric. Java/JS categories exist but matching is less robust.
- Benchmark tests exact-match. In production, partial correctness (right function, one wrong arg) is often recoverable via stage-pilot retries.
