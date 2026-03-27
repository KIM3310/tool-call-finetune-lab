"""Stage-pilot bridge evaluation.

Tests the fine-tuned model against stage-pilot's tool-calling benchmark by
routing requests through the OpenAI-compatible vLLM endpoint with and without
the stage-pilot middleware layer.

Usage:
    python -m tool_call_finetune_lab.eval.stage_pilot_bridge \
        --vllm-url http://localhost:8000/v1 \
        --stage-pilot-url http://localhost:9000/v1
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Representative tool-calling test cases that mirror stage-pilot's use cases
STAGE_PILOT_TEST_CASES: List[Dict[str, Any]] = [
    {
        "name": "simple_single_tool",
        "messages": [
            {"role": "user", "content": "What is the weather in Tokyo?"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "default": "celsius",
                            },
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "expected_tool_name": "get_weather",
        "expected_args_subset": {"city": "Tokyo"},
    },
    {
        "name": "parallel_tool_calls",
        "messages": [
            {
                "role": "user",
                "content": "Get weather for both Tokyo and London simultaneously.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "expected_tool_name": "get_weather",
        "expected_call_count": 2,
    },
    {
        "name": "multi_tool_selection",
        "messages": [
            {
                "role": "user",
                "content": "Search for 'machine learning' papers and then summarize the top result.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search_papers",
                    "description": "Search academic papers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer", "default": 5},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "summarize_text",
                    "description": "Summarize a piece of text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "max_length": {"type": "integer", "default": 200},
                        },
                        "required": ["text"],
                    },
                },
            },
        ],
        "expected_tool_name": "search_papers",
        "expected_args_subset": {"query": "machine learning"},
    },
    {
        "name": "no_tool_needed",
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            }
        ],
        "expected_tool_name": None,  # Model should answer directly
    },
    {
        "name": "nested_argument_types",
        "messages": [
            {
                "role": "user",
                "content": "Create a calendar event titled 'Team Standup' on 2025-01-15 at 09:00 for 30 minutes.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "create_calendar_event",
                    "description": "Create a calendar event",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
                            "time": {"type": "string", "description": "HH:MM format"},
                            "duration_minutes": {"type": "integer"},
                        },
                        "required": ["title", "date", "time"],
                    },
                },
            }
        ],
        "expected_tool_name": "create_calendar_event",
        "expected_args_subset": {"title": "Team Standup", "date": "2025-01-15"},
    },
]


def _call_endpoint(
    client: Any,
    model_name: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    timeout: int = 60,
) -> Tuple[Optional[List[Dict[str, Any]]], float, str]:
    """Call an OpenAI-compatible endpoint and return (tool_calls, latency_s, raw_text)."""
    t0 = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=512,
            temperature=0.0,
            timeout=timeout,
        )
    except Exception as e:
        latency = time.perf_counter() - t0
        logger.error("API call failed: %s", e)
        return None, latency, str(e)

    latency = time.perf_counter() - t0
    choice = response.choices[0]
    msg = choice.message
    text = msg.content or ""

    if msg.tool_calls:
        calls = [
            {
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments) if tc.function.arguments else {},
            }
            for tc in msg.tool_calls
        ]
        return calls, latency, text

    return None, latency, text


def _score_test_case(
    test_case: Dict[str, Any],
    tool_calls: Optional[List[Dict[str, Any]]],
) -> Tuple[bool, str]:
    """Return (passed, reason) for a test case given the model's tool_calls."""
    expected_name = test_case.get("expected_tool_name")
    expected_count = test_case.get("expected_call_count")
    expected_args_subset = test_case.get("expected_args_subset", {})

    if expected_name is None:
        # Expect no tool call
        if not tool_calls:
            return True, "Correctly chose not to call a tool"
        return False, f"Unexpected tool call(s): {[c['name'] for c in tool_calls]}"

    if not tool_calls:
        return False, f"Expected tool '{expected_name}' but got no tool call"

    if expected_count:
        if len(tool_calls) < expected_count:
            return False, f"Expected {expected_count} calls, got {len(tool_calls)}"

    # Find a call matching the expected name
    matching = [c for c in tool_calls if c["name"] == expected_name]
    if not matching:
        return False, f"Expected tool '{expected_name}', got {[c['name'] for c in tool_calls]}"

    call = matching[0]
    args = call.get("arguments", {})

    # Check subset of expected arguments
    for key, exp_val in expected_args_subset.items():
        act_val = args.get(key)
        if act_val is None:
            return False, f"Missing argument '{key}'"
        if str(act_val).lower() != str(exp_val).lower():
            return False, f"Argument '{key}': expected '{exp_val}', got '{act_val}'"

    return True, "Pass"


def run_bridge_eval(
    vllm_url: str,
    vllm_model: str,
    stage_pilot_url: Optional[str] = None,
    stage_pilot_model: Optional[str] = None,
    output_file: str = "results/stage_pilot_bridge.json",
) -> Dict[str, Any]:
    """Run test cases against vLLM (direct) and optionally stage-pilot middleware."""
    from openai import OpenAI

    direct_client = OpenAI(base_url=vllm_url, api_key="EMPTY")
    sp_client = OpenAI(base_url=stage_pilot_url, api_key="EMPTY") if stage_pilot_url else None

    results: List[Dict[str, Any]] = []

    for tc in STAGE_PILOT_TEST_CASES:
        logger.info("Running test case: %s", tc["name"])
        row: Dict[str, Any] = {"test": tc["name"]}

        # Direct vLLM
        calls, latency, text = _call_endpoint(
            direct_client, vllm_model, tc["messages"], tc["tools"]
        )
        passed, reason = _score_test_case(tc, calls)
        row["direct"] = {
            "passed": passed,
            "reason": reason,
            "latency_s": round(latency, 3),
            "tool_calls": calls,
            "raw_text": text[:200] if text else "",
        }
        logger.info("  Direct:  %s — %s (%.3fs)", "PASS" if passed else "FAIL", reason, latency)

        # Stage-pilot (if configured)
        if sp_client and stage_pilot_model:
            sp_calls, sp_latency, sp_text = _call_endpoint(
                sp_client, stage_pilot_model, tc["messages"], tc["tools"]
            )
            sp_passed, sp_reason = _score_test_case(tc, sp_calls)
            row["stage_pilot"] = {
                "passed": sp_passed,
                "reason": sp_reason,
                "latency_s": round(sp_latency, 3),
                "tool_calls": sp_calls,
                "raw_text": sp_text[:200] if sp_text else "",
            }
            logger.info(
                "  Stage-pilot: %s — %s (%.3fs)",
                "PASS" if sp_passed else "FAIL",
                sp_reason,
                sp_latency,
            )

        results.append(row)

    # Summary
    direct_pass = sum(1 for r in results if r["direct"]["passed"])
    total = len(results)
    summary = {
        "direct_pass_rate": round(direct_pass / total * 100, 1),
        "direct_passed": direct_pass,
        "total": total,
    }

    if sp_client:
        sp_pass = sum(1 for r in results if r.get("stage_pilot", {}).get("passed", False))
        summary["stage_pilot_pass_rate"] = round(sp_pass / total * 100, 1)
        summary["stage_pilot_passed"] = sp_pass

    output: Dict[str, Any] = {"summary": summary, "results": results}

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info("Bridge evaluation complete. Saved to %s", output_file)
    logger.info("Direct pass rate: %d/%d (%.1f%%)", direct_pass, total, summary["direct_pass_rate"])

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-pilot bridge evaluation")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--vllm-model", default="qwen2.5-7b-tool-call")
    parser.add_argument("--stage-pilot-url", default=None)
    parser.add_argument("--stage-pilot-model", default=None)
    parser.add_argument("--output-file", default="results/stage_pilot_bridge.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_bridge_eval(
        vllm_url=args.vllm_url,
        vllm_model=args.vllm_model,
        stage_pilot_url=args.stage_pilot_url,
        stage_pilot_model=args.stage_pilot_model,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
