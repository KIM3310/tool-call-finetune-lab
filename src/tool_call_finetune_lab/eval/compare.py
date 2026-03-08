"""Generate a comparison markdown table: base model vs fine-tuned vs GPT-4o-mini.

Usage:
    python -m tool_call_finetune_lab.eval.compare \
        --finetuned-results results/bfcl_results.json \
        --base-results results/bfcl_base_results.json \
        --output results/comparison.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# BFCL categories to show in the table (key → display name)
CATEGORIES = {
    "gorilla_openfunctions_v1_test_simple": "AST Simple",
    "gorilla_openfunctions_v1_test_multiple_function": "AST Multiple",
    "gorilla_openfunctions_v1_test_parallel_function": "AST Parallel",
    "gorilla_openfunctions_v1_test_parallel_multiple_function": "AST Parallel+Multi",
    "simple": "AST Simple",
    "multiple": "AST Multiple",
    "parallel": "AST Parallel",
    "_overall": "Overall",
}


def _load_results(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load a results JSON file. Returns None if path is missing."""
    if not path or not Path(path).exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _get_accuracy(results: Optional[Dict[str, Any]], category: str) -> str:
    """Extract accuracy string for a given category key."""
    if results is None:
        return "—"
    cats = results.get("categories", {})
    if category in cats:
        return f"{cats[category]['accuracy']:.1f}%"
    # Try partial key match
    for key, stats in cats.items():
        if category.lower() in key.lower():
            return f"{stats['accuracy']:.1f}%"
    return "—"


def run_gpt4o_mini_eval(
    test_file: str,
    max_examples: int = 100,
    output_file: str = "results/bfcl_gpt4omini_results.json",
) -> Optional[Dict[str, Any]]:
    """Run GPT-4o-mini against the test set for reference comparison.

    Requires OPENAI_API_KEY to be set. Skips gracefully if not available.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — skipping GPT-4o-mini evaluation")
        return None

    from openai import OpenAI

    from tool_call_finetune_lab.eval.bfcl_runner import VLLMBackend, evaluate, load_test_data

    logger.info("Running GPT-4o-mini evaluation (max %d examples)...", max_examples)

    client = OpenAI(api_key=api_key)

    class GPT4oMiniBackend:
        def __init__(self) -> None:
            self.client = client

        def predict(
            self,
            messages: List[Dict[str, Any]],
            tools: List[Dict[str, Any]],
        ) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
            kwargs: Dict[str, Any] = {
                "model": "gpt-4o-mini",
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.0,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            msg = choice.message

            if msg.tool_calls:
                calls = [
                    {
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments)
                        if tc.function.arguments
                        else {},
                    }
                    for tc in msg.tool_calls
                ]
                return msg.content or "", calls

            return msg.content or "", None

    try:
        test_examples = load_test_data(test_file)[:max_examples]
        backend = GPT4oMiniBackend()
        results = evaluate(backend, test_examples)

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("GPT-4o-mini results saved to %s", output_file)
        return results
    except Exception as e:
        logger.error("GPT-4o-mini evaluation failed: %s", e)
        return None


def generate_comparison_table(
    finetuned_results: Optional[Dict[str, Any]],
    base_results: Optional[Dict[str, Any]],
    gpt4omini_results: Optional[Dict[str, Any]],
    output_file: str,
) -> str:
    """Generate a markdown comparison table and write it to output_file."""
    lines: List[str] = [
        "# BFCL Evaluation Comparison",
        "",
        "| Category | Base (Qwen2.5-7B) | Fine-tuned (LoRA) | GPT-4o-mini |",
        "|---|---|---|---|",
    ]

    category_keys = list(CATEGORIES.keys())

    for cat_key in category_keys:
        display = CATEGORIES[cat_key]
        base_acc = _get_accuracy(base_results, cat_key)
        ft_acc = _get_accuracy(finetuned_results, cat_key)
        gpt_acc = _get_accuracy(gpt4omini_results, cat_key)

        # Skip if all are unknown
        if base_acc == "—" and ft_acc == "—" and gpt_acc == "—":
            continue

        lines.append(f"| {display} | {base_acc} | {ft_acc} | {gpt_acc} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Base: `Qwen/Qwen2.5-7B-Instruct` without fine-tuning",
            "- Fine-tuned: QLoRA rank=16 on BFCL + Glaive-v2, 3 epochs",
            "- GPT-4o-mini: OpenAI API reference (gpt-4o-mini-2024-07-18)",
            "- All accuracies are exact-match tool name + argument comparison",
            "",
            "_Generated by `tool_call_finetune_lab.eval.compare`_",
        ]
    )

    md = "\n".join(lines)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md)

    logger.info("Comparison table written to %s", output_file)
    return md


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BFCL comparison table")
    parser.add_argument("--finetuned-results", default="results/bfcl_results.json")
    parser.add_argument("--base-results", default="results/bfcl_base_results.json")
    parser.add_argument("--gpt4omini-results", default="results/bfcl_gpt4omini_results.json")
    parser.add_argument("--test-file", default="data/processed/test.jsonl")
    parser.add_argument("--output", default="results/comparison.md")
    parser.add_argument(
        "--run-gpt4omini",
        action="store_true",
        help="Run GPT-4o-mini evaluation if OPENAI_API_KEY is set",
    )
    parser.add_argument("--max-gpt-examples", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    finetuned = _load_results(args.finetuned_results)
    base = _load_results(args.base_results)

    gpt4omini = _load_results(args.gpt4omini_results)
    if gpt4omini is None and args.run_gpt4omini:
        gpt4omini = run_gpt4o_mini_eval(
            test_file=args.test_file,
            max_examples=args.max_gpt_examples,
        )

    md = generate_comparison_table(finetuned, base, gpt4omini, args.output)
    print("\n" + md)


if __name__ == "__main__":
    main()
