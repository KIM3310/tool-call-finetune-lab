# Evaluation Methodology

This document explains how we evaluate the fine-tuned model, why we chose the Berkeley Function-Calling Leaderboard (BFCL) as our primary benchmark, and what the results mean.

## Why BFCL

Evaluating tool-calling models presents a unique challenge: unlike standard NLP benchmarks where there is a single correct answer, a tool call can be "correct" in multiple ways (different argument orderings, equivalent type representations, optional parameters present or absent). A custom eval suite would need to handle all of this, and the results would not be comparable to anything external.

BFCL was chosen for three reasons:

1. **Community benchmark**: BFCL is maintained by the Gorilla project at UC Berkeley and is used to rank both open and closed models on the same evaluation. This means our results are directly comparable to GPT-4o, Claude, Llama, and other models on the public leaderboard.
2. **AST-based matching**: Instead of string comparison, BFCL uses Abstract Syntax Tree (AST) matching to determine if a generated tool call is functionally equivalent to the ground truth. This handles argument reordering, whitespace differences, and type coercion automatically.
3. **Reproducibility**: The evaluation data, ground-truth answers, and evaluation scripts are all publicly available. Anyone can re-run our evaluation and verify the numbers.

## BFCL Categories

BFCL v4 evaluates tool-calling across multiple categories that test different aspects of function-calling ability:

### Simple Function Calling
The model receives a single user query and a set of tool definitions, and must produce exactly one correct function call. This tests basic argument extraction and function selection.

Example: User asks "What's the weather in San Francisco?" with a `get_weather(city: str, unit: str)` tool available. The model must output the correct function name and extract "San Francisco" as the city argument.

### Multiple Function Calling
The model receives a query that requires calling one function selected from multiple available tools. This tests the model's ability to choose the right tool when several are available, not just extract arguments.

Example: User asks "Book a flight to Tokyo" when `book_flight`, `book_hotel`, and `get_weather` are all available. The model must select `book_flight` and ignore the others.

### Parallel Function Calling
The model must produce multiple independent function calls in a single response turn. This is the most challenging category because the model needs to:
- Recognize that multiple actions are needed
- Emit all calls in the correct structured format
- Not serialize them into a dependent chain

Example: User asks "Check the weather in NYC and SF" with a `get_weather` tool. The model should emit two parallel calls, one for each city.

### Additional Categories
BFCL also includes categories for:
- **Java/JavaScript function calls**: Non-Python function signatures
- **REST API calls**: HTTP endpoint invocations
- **Relevance detection**: Determining when no tool call is appropriate
- **Live variants**: Real-world API schemas from active services

We focus primarily on Simple, Multiple, and Parallel as they represent the core tool-calling capabilities relevant to stage-pilot integration.

## How AST Matching Works

Traditional string-based evaluation would mark `get_weather(city="San Francisco", unit="fahrenheit")` as incorrect if the ground truth is `get_weather(unit="fahrenheit", city="San Francisco")`, even though they are functionally identical.

BFCL's AST matching resolves this by:

1. **Parsing** both the generated output and ground truth into Abstract Syntax Trees (Python's `ast.parse`).
2. **Normalizing** the trees: sorting keyword arguments alphabetically, resolving type aliases (e.g., `str` vs `string`), and standardizing numeric representations.
3. **Comparing** the normalized trees for structural equivalence.

This means evaluation is robust to:
- Argument order differences
- Whitespace and formatting variations
- Equivalent type representations (`True` vs `true`, `1.0` vs `1`)
- String quoting style (`'value'` vs `"value"`)

A call is marked correct only if the function name matches exactly AND all required arguments are present with correct values after normalization.

## Evaluation Pipeline

Our eval pipeline (`src/tool_call_finetune_lab/eval/`) works as follows:

```
BFCL test prompts (from bfcl_v4 dataset)
         |
         v
  vLLM serving endpoint
  (fine-tuned model, OpenAI-compat API)
         |
         v
  Raw model outputs collected
         |
         v
  BFCL AST evaluator
  (parse, normalize, compare)
         |
         v
  Per-category accuracy scores
         |
         v
  Comparison table (vs. base model, vs. GPT-4o-mini)
```

We evaluate three model configurations:
1. **Qwen2.5-7B-Instruct (base)**: The unmodified base model, establishing the pre-training baseline.
2. **Qwen2.5-7B-Instruct + LoRA**: The fine-tuned model with merged LoRA adapters, showing the improvement from training.
3. **GPT-4o-mini (reference)**: A closed-model reference point for calibration. This is not meant as a direct competitor but as an upper-bound reference from a model optimized for tool calling.

## Interpreting Results

When reading the results table in the README:

- **AST Simple** accuracy reflects basic function-calling competence. Most modern instruction-tuned models score reasonably well here.
- **AST Multiple** tests tool selection. Errors here usually mean the model picked the wrong function from the available set.
- **Parallel** is where open models typically struggle the most. The model must understand that multiple independent actions are needed and emit them in the correct multi-call format. This is also where LoRA fine-tuning shows the largest relative improvement.
- **Overall** is a weighted average across all evaluated categories.

Accuracy above 80% overall is competitive with commercial API models for practical tool-calling deployments, especially when combined with stage-pilot's middleware recovery layer.

## Limitations

- BFCL evaluates single-turn tool calling. Multi-turn conversations where the model must incorporate tool results and make follow-up calls are not covered by the current benchmark.
- AST matching is Python-centric. While BFCL includes Java/JS categories, the matching logic is most robust for Python function signatures.
- The benchmark tests exact-match correctness. In production, partial correctness (e.g., calling the right function with one wrong argument) may still be recoverable by stage-pilot's retry logic.
