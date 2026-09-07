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
    "gorilla_openfunctions_v1_test_simple": "Simple",
    "gorilla_openfunctions_v1_test_multiple_function": "Multiple",
    "gorilla_openfunctions_v1_test_parallel_function": "Parallel",
    "gorilla_openfunctions_v1_test_parallel_multiple_function": "Parallel+Multi",
    "simple": "Simple",
    "multiple": "Multiple",
    "parallel": "Parallel",
    "_overall": "Overall",
}


def _load_results(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load a results JSON file. Returns None if path is missing."""
    if not path or not Path(path).exists():
        return None
    with open(path, encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
        return data


def _get_accuracy(results: Optional[Dict[str, Any]], category: str) -> str:
    """Extract accuracy string for a given category key."""
    if results is None:
        return "—"
    cats = results.get("categories", {})
    if category in cats:
        return f"{cats[category]['accuracy']:.1f}%"
    return "—"


def run_gpt4o_mini_eval(
    test_file: str,
    max_examples: int = 100,
    output_file: str = "results/bfcl_gpt4omini_results.json",
) -> Optional[Dict[str, Any]]:
    """Run an OpenRouter/OpenAI-compatible model against the test set.

    Requires OPENROUTER_API_KEY or OPENAI_API_KEY. Skips gracefully if not available.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning(
            "OPENROUTER_API_KEY / OPENAI_API_KEY not set — skipping reference evaluation"
        )
        return None
    base_url = os.environ.get("OPENROUTER_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    model = (
        os.environ.get("OPENROUTER_MODEL") or os.environ.get("OPENAI_MODEL") or "qwen/qwen3-coder"
    )

    from openai import OpenAI

    from tool_call_finetune_lab.eval.bfcl_runner import evaluate, load_test_data

    logger.info(
        "Running reference model evaluation with %s (max %d examples)...", model, max_examples
    )

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    class GPT4oMiniBackend:
        def __init__(self) -> None:
            self.client = client
            self.model_name = model

        def predict(
            self,
            messages: List[Dict[str, Any]],
            tools: List[Dict[str, Any]],
        ) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
            kwargs: Dict[str, Any] = {
                "model": model,
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
    available = [
        result for result in (base_results, finetuned_results, gpt4omini_results) if result
    ]
    metadata = [result.get("metadata", {}) for result in available]
    complete_metadata = bool(metadata) and all(
        meta.get("scoring_contract") and meta.get("dataset_sha256") for meta in metadata
    )
    if (
        complete_metadata
        and len({(meta["scoring_contract"], meta["dataset_sha256"]) for meta in metadata}) > 1
    ):
        raise ValueError("Cannot compare different scoring contracts or evaluation datasets")
    lines: List[str] = [
        "# Tool-call Evaluation Comparison",
        "",
        "| Category | Base | Adapted | Reference |",
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
            "- This local normalized-call scorer does not produce official BFCL leaderboard scores.",
            "- Matching dataset/scorer metadata verified."
            if complete_metadata
            else "- Legacy results lack dataset/scorer metadata; comparability is unverified.",
            f"- Base model: {(base_results or {}).get('metadata', {}).get('model', 'unspecified')}",
            f"- Adapted model: {(finetuned_results or {}).get('metadata', {}).get('model', 'unspecified')}",
            f"- Reference model: {(gpt4omini_results or {}).get('metadata', {}).get('model', 'unspecified')}",
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
