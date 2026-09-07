"""BFCL evaluation runner.

Loads the fine-tuned model (via vLLM OpenAI-compatible endpoint or locally),
runs BFCL test cases, computes accuracy per category, and saves results.

Usage:
    # Against a running vLLM server:
    python -m tool_call_finetune_lab.eval.bfcl_runner --mode vllm

    # Against the local HF model:
    python -m tool_call_finetune_lab.eval.bfcl_runner --mode local \
        --model-path outputs/merged-model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tool_call_finetune_lab.config import (
    QWEN_25_7B_INSTRUCT_PINNED_REVISION,
    validate_immutable_revision,
    validate_remote_code_policy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool-call extraction helpers
# ---------------------------------------------------------------------------


def _extract_tool_calls_from_response(response_text: str) -> List[Dict[str, Any]]:
    """Parse tool calls from a model response string.

    Handles both:
    - <tool_call>{"name": ..., "arguments": ...}</tool_call> format (Qwen2.5)
    - OpenAI-style JSON function call strings
    """
    import re

    calls: List[Dict[str, Any]] = []

    # Qwen2.5 <tool_call> format
    tool_call_re = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    for match in tool_call_re.finditer(response_text):
        raw = match.group(1).strip()
        try:
            obj = json.loads(raw)
            if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or not obj["name"]:
                return []
            calls.append(
                {
                    "name": obj.get("name", ""),
                    "arguments": obj.get("arguments", {}),
                }
            )
        except (ValueError, RecursionError):
            return []

    if calls:
        return calls

    # Fallback: bare JSON object with "name" + "arguments"
    try:
        obj = json.loads(response_text.strip())
        if isinstance(obj, dict) and "name" in obj:
            calls.append(
                {
                    "name": obj["name"],
                    "arguments": obj.get("arguments", {}),
                }
            )
            return calls
    except (json.JSONDecodeError, ValueError):
        pass

    return calls


def _normalize_arguments(args: Any) -> Optional[Dict[str, Any]]:
    """Decode a JSON object; invalid values never become an empty argument set."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (ValueError, RecursionError):
            return None
    return args if isinstance(args, dict) else None


def _json_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without coercing strings, booleans, or object keys."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return (
            (not isinstance(left, float) or math.isfinite(left))
            and (not isinstance(right, float) or math.isfinite(right))
            and left == right
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            isinstance(key, str) and _json_equal(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return isinstance(left, (str, type(None))) and left == right


def _tool_call_matches(predicted: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    """Strict local scorer: exact function identity and complete JSON arguments.

    This normalized-call contract is not the official BFCL AST/execution evaluator.
    """
    name = predicted.get("name")
    if not isinstance(name, str) or not name or name != expected.get("name"):
        return False
    pred_args = _normalize_arguments(predicted.get("arguments", {}))
    exp_args = _normalize_arguments(expected.get("arguments", {}))
    return pred_args is not None and exp_args is not None and _json_equal(pred_args, exp_args)


# ---------------------------------------------------------------------------
# Model inference backends
# ---------------------------------------------------------------------------


class VLLMBackend:
    """Query a running vLLM server via the OpenAI-compatible API."""

    def __init__(self, base_url: str, model_name: str, timeout: int = 60) -> None:
        from openai import OpenAI

        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        self.client = OpenAI(
            base_url=base_url, api_key=os.environ.get("VLLM_API_KEY") or "EMPTY", timeout=timeout
        )
        self.model_name = model_name
        self.timeout = timeout

    def predict(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        """Return (response_text, tool_calls_list)."""
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.0,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        # Extract structured tool calls if returned by API
        if msg.tool_calls:
            api_calls = [
                {
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments) if tc.function.arguments else {},
                }
                for tc in msg.tool_calls
            ]
            return msg.content or "", api_calls

        # Fall back to parsing text
        text = msg.content or ""
        parsed = _extract_tool_calls_from_response(text)
        return text, parsed if parsed else None


class LocalHFBackend:
    """Run inference locally with a HuggingFace model."""

    def __init__(
        self,
        model_path: str,
        max_seq_length: int = 4096,
        model_revision: str = QWEN_25_7B_INSTRUCT_PINNED_REVISION,
        code_revision: str = QWEN_25_7B_INSTRUCT_PINNED_REVISION,
        trust_remote_code: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        validate_immutable_revision(model_revision, "model_revision")
        validate_immutable_revision(code_revision, "code_revision")
        validate_remote_code_policy(trust_remote_code, code_revision)

        logger.info("Loading model from %s for local inference...", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            revision=model_revision,
            code_revision=code_revision,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=trust_remote_code,
            revision=model_revision,
            code_revision=code_revision,
        )
        self.max_seq_length = max_seq_length
        self.model_name = model_path

    def predict(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        import torch

        text = self.tokenizer.apply_chat_template(
            messages,
            tools=tools if tools else None,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=None,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=False)

        parsed = _extract_tool_calls_from_response(response)
        return response, parsed if parsed else None


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate(
    backend: Any,
    test_examples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run evaluation over test examples and return per-category accuracy dict."""
    category_correct: Dict[str, int] = defaultdict(int)
    category_total: Dict[str, int] = defaultdict(int)
    category_latency: Dict[str, List[float]] = defaultdict(list)
    failures: List[Dict[str, Any]] = []

    for i, ex in enumerate(test_examples):
        category = ex.get("category", "unknown")
        messages = ex.get("messages", [])
        tools = ex.get("tools", [])

        # Evaluate the last tool-call turn, preserving earlier context and excluding
        # the target answer and every subsequent tool result or user message.
        targets = [
            index
            for index, message in enumerate(messages)
            if message.get("role") == "assistant" and message.get("tool_calls")
        ]
        if not targets:
            continue
        target_index = targets[-1]
        inference_msgs = messages[:target_index]
        expected_calls: List[Dict[str, Any]] = []
        for call in messages[target_index]["tool_calls"]:
            function = call.get("function", call)
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                raise ValueError(f"Malformed target tool call in example {i}")
            expected_calls.append(
                {"name": function["name"], "arguments": function.get("arguments", {})}
            )

        try:
            t0 = time.perf_counter()
            _, predicted_calls = backend.predict(inference_msgs, tools)
            latency = time.perf_counter() - t0
        except Exception as e:
            logger.warning("Inference error on example %d: %s", i, e)
            category_total[category] += 1
            failures.append({"index": i, "error": str(e), "category": category})
            continue

        category_latency[category].append(latency)
        category_total[category] += 1

        # BFCL-style correctness requires a one-to-one match. Extra calls can
        # trigger unintended side effects and therefore count as failures.
        if predicted_calls and len(predicted_calls) == len(expected_calls):
            # For each expected call, find a match in predicted
            matched = 0
            pred_remaining = list(predicted_calls)
            for exp in expected_calls:
                for j, pred in enumerate(pred_remaining):
                    if _tool_call_matches(pred, exp):
                        matched += 1
                        pred_remaining.pop(j)
                        break
            if matched == len(expected_calls):
                category_correct[category] += 1
            else:
                failures.append(
                    {
                        "index": i,
                        "category": category,
                        "expected": expected_calls,
                        "predicted": predicted_calls,
                    }
                )
        else:
            failures.append(
                {
                    "index": i,
                    "category": category,
                    "expected": expected_calls,
                    "predicted": predicted_calls,
                }
            )

        if (i + 1) % 50 == 0:
            logger.info("Evaluated %d / %d examples", i + 1, len(test_examples))

    # Compile results
    results: Dict[str, Any] = {}
    all_correct = 0
    all_total = 0

    for cat in sorted(category_total.keys()):
        correct = category_correct[cat]
        total = category_total[cat]
        latencies = category_latency[cat]
        accuracy = correct / total if total > 0 else 0.0
        results[cat] = {
            "correct": correct,
            "total": total,
            "accuracy": round(accuracy * 100, 2),
            "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        }
        all_correct += correct
        all_total += total

    overall_accuracy = all_correct / all_total if all_total > 0 else 0.0
    results["_overall"] = {
        "correct": all_correct,
        "total": all_total,
        "accuracy": round(overall_accuracy * 100, 2),
    }

    logger.info(
        "Overall accuracy: %d / %d = %.2f%%",
        all_correct,
        all_total,
        overall_accuracy * 100,
    )

    return {
        "categories": results,
        "failures": failures[:50],
        "metadata": {
            "scoring_contract": "normalized-tool-call-exact-v2",
            "official_bfcl_score": False,
            "model": getattr(backend, "model_name", "unspecified"),
            "dataset_sha256": hashlib.sha256(
                json.dumps(test_examples, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "input_examples": len(test_examples),
            "evaluated_examples": all_total,
            "skipped_examples": len(test_examples) - all_total,
            "failure_count": len(failures),
            "failure_log_truncated": len(failures) > 50,
        },
    }


def load_test_data(test_file: str) -> List[Dict[str, Any]]:
    """Load test JSONL and filter to examples with tool calls."""
    examples: List[Dict[str, Any]] = []
    path = Path(test_file)
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                # Only keep examples with at least one assistant tool call
                has_call = any(
                    m.get("role") == "assistant" and m.get("tool_calls")
                    for m in ex.get("messages", [])
                )
                if has_call:
                    examples.append(ex)

    logger.info("Loaded %d test examples with tool calls", len(examples))
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BFCL evaluation runner")
    parser.add_argument(
        "--mode",
        choices=["vllm", "local"],
        default="vllm",
        help="Inference backend: 'vllm' (HTTP) or 'local' (HF model)",
    )
    parser.add_argument("--model-path", default="outputs/merged-model")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--model-name", default="qwen2.5-7b-tool-call")
    parser.add_argument("--test-file", default="data/processed/test.jsonl")
    parser.add_argument("--results-file", default="results/bfcl_results.json")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--model-revision", default=QWEN_25_7B_INSTRUCT_PINNED_REVISION)
    parser.add_argument("--code-revision", default=QWEN_25_7B_INSTRUCT_PINNED_REVISION)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Explicitly allow audited remote model code. Use with pinned --code-revision.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from tool_call_finetune_lab.config import EvalConfig

    cfg = EvalConfig(
        bfcl_results_file=args.results_file,
        vllm_base_url=args.vllm_url,
        model_name=args.model_name,
    )

    # Load test data
    test_examples = load_test_data(args.test_file)
    if args.max_examples:
        test_examples = test_examples[: args.max_examples]
        logger.info("Capped to %d examples for evaluation", args.max_examples)

    # Build backend
    if args.mode == "vllm":
        logger.info("Using vLLM backend at %s", cfg.vllm_base_url)
        backend = VLLMBackend(cfg.vllm_base_url, cfg.model_name)
    else:
        logger.info("Using local HF backend from %s", args.model_path)
        backend = LocalHFBackend(  # type: ignore[assignment]
            args.model_path,
            model_revision=args.model_revision,
            code_revision=args.code_revision,
            trust_remote_code=args.trust_remote_code,
        )

    # Run evaluation
    results = evaluate(backend, test_examples)

    # Save results
    results_path = Path(cfg.bfcl_results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to %s", results_path)

    # Print summary
    print("\nBFCL Evaluation Results:")
    print("-" * 50)
    for cat, stats in results["categories"].items():
        print(f"  {cat:<40} {stats['accuracy']:>6.2f}%  ({stats['correct']}/{stats['total']})")


if __name__ == "__main__":
    main()
