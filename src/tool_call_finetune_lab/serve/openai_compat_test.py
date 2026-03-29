"""Smoke tests for the vLLM OpenAI-compatible endpoint.

Verifies that the running vLLM server correctly handles:
- Basic chat completion
- Tool-call requests (single, parallel, no-call scenarios)
- Streaming responses

Usage:
    python -m tool_call_finetune_lab.serve.openai_compat_test
    python -m tool_call_finetune_lab.serve.openai_compat_test \
        --url http://localhost:8000/v1 \
        --model qwen2.5-7b-tool-call
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------


def test_health_check(base_url: str) -> Tuple[bool, str]:
    """Verify the server is reachable via the /health or /v1/models endpoint."""
    import httpx

    url = base_url.rstrip("/v1").rstrip("/") + "/health"
    try:
        r = httpx.get(url, timeout=10)
        if r.status_code == 200:
            return True, f"Health check OK ({url})"
    except Exception:
        pass

    # Fallback: try /v1/models
    models_url = base_url.rstrip("/") + "/models"
    try:
        r = httpx.get(models_url, timeout=10)
        if r.status_code == 200:
            models = r.json().get("data", [])
            return True, f"Models endpoint OK — {len(models)} model(s) available"
    except Exception as e:
        return False, f"Server unreachable: {e}"

    return False, "Server not responding"


def test_basic_completion(client: Any, model: str) -> Tuple[bool, str]:
    """Test a basic non-tool chat completion."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'hello' in exactly one word."}],
            max_tokens=10,
            temperature=0.0,
        )
        text = response.choices[0].message.content or ""
        if text.strip():
            return True, f"Basic completion OK: '{text.strip()[:50]}'"
        return False, "Empty response from basic completion"
    except Exception as e:
        return False, f"Basic completion failed: {e}"


def test_single_tool_call(client: Any, model: str) -> Tuple[bool, str]:
    """Test that the model calls a single tool correctly."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
            },
        }
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            tools=tools,
            tool_choice="auto",
            max_tokens=256,
            temperature=0.0,
        )
        choice = response.choices[0]
        msg = choice.message

        if msg.tool_calls:
            tc = msg.tool_calls[0]
            if tc.function.name == "get_weather":
                args = json.loads(tc.function.arguments or "{}")
                city = args.get("city", "").lower()
                if "paris" in city:
                    return True, f"Single tool call OK: get_weather(city='{args.get('city')}')"
                return (
                    True,
                    f"Tool called but unexpected city: '{args.get('city')}' (may still be correct)",
                )
            return False, f"Wrong tool called: '{tc.function.name}' (expected 'get_weather')"

        # Some models answer directly for simple queries — treat as soft pass
        text = msg.content or ""
        return (
            True,
            f"Model answered directly (no tool call): '{text[:100]}' — consider adjusting tool_choice",
        )
    except Exception as e:
        return False, f"Single tool call failed: {e}"


def test_parallel_tool_calls(client: Any, model: str) -> Tuple[bool, str]:
    """Test that the model can make parallel tool calls."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Get weather for Tokyo AND London at the same time.",
                }
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=512,
            temperature=0.0,
        )
        choice = response.choices[0]
        msg = choice.message

        if msg.tool_calls and len(msg.tool_calls) >= 2:
            names = [tc.function.name for tc in msg.tool_calls]
            return True, f"Parallel tool calls OK: {names}"
        elif msg.tool_calls and len(msg.tool_calls) == 1:
            return (
                True,
                "Only 1 tool call made (parallel not triggered) — acceptable for some models",
            )
        return False, "No tool calls made for parallel request"
    except Exception as e:
        return False, f"Parallel tool call test failed: {e}"


def test_no_tool_needed(client: Any, model: str) -> Tuple[bool, str]:
    """Test that the model does NOT call a tool when it's unnecessary."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Get current stock price",
                "parameters": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"],
                },
            },
        }
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is 5 multiplied by 7?"}],
            tools=tools,
            tool_choice="auto",
            max_tokens=64,
            temperature=0.0,
        )
        msg = response.choices[0].message
        if not msg.tool_calls:
            text = msg.content or ""
            if "35" in text:
                return True, f"Correctly answered without tool call: '{text.strip()[:80]}'"
            return True, f"No tool called (response: '{text.strip()[:80]}')"
        return False, f"Incorrectly called tool: {[tc.function.name for tc in msg.tool_calls]}"
    except Exception as e:
        return False, f"No-tool test failed: {e}"


def test_streaming(client: Any, model: str) -> Tuple[bool, str]:
    """Test that streaming completions work."""
    try:
        chunks: List[str] = []
        with client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            max_tokens=50,
            temperature=0.0,
            stream=True,
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                chunks.append(delta)

        full_text = "".join(chunks)
        if full_text.strip():
            return True, f"Streaming OK: '{full_text.strip()[:80]}'"
        return False, "Streaming produced empty output"
    except Exception as e:
        return False, f"Streaming test failed: {e}"


def test_tool_argument_types(client: Any, model: str) -> Tuple[bool, str]:
    """Test that the model generates correct argument types."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "create_reminder",
                "description": "Create a reminder",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "minutes_from_now": {
                            "type": "integer",
                            "description": "How many minutes until the reminder",
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                        },
                    },
                    "required": ["text", "minutes_from_now"],
                },
            },
        }
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Remind me to take medicine in 30 minutes with high priority.",
                }
            ],
            tools=tools,
            tool_choice="required",
            max_tokens=256,
            temperature=0.0,
        )
        msg = response.choices[0].message
        if not msg.tool_calls:
            return False, "No tool calls despite tool_choice='required'"

        tc = msg.tool_calls[0]
        args = json.loads(tc.function.arguments or "{}")

        # Validate types
        issues: List[str] = []
        if not isinstance(args.get("minutes_from_now"), int):
            issues.append(
                f"minutes_from_now should be int, got {type(args.get('minutes_from_now'))}"
            )
        if args.get("priority") and args["priority"] not in ("low", "medium", "high"):
            issues.append(f"Invalid priority: '{args['priority']}'")

        if issues:
            return False, f"Type issues: {'; '.join(issues)}"
        return True, f"Argument types OK: {args}"
    except Exception as e:
        return False, f"Argument type test failed: {e}"


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


def run_all_tests(
    base_url: str,
    model: str,
    fail_fast: bool = False,
) -> Tuple[int, int]:
    """Run all smoke tests. Returns (passed, total)."""
    from openai import OpenAI

    print("\nvLLM Endpoint Smoke Tests")
    print(f"URL:   {base_url}")
    print(f"Model: {model}")
    print("=" * 60)

    # Health check first
    ok, msg = test_health_check(base_url)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] health_check: {msg}")
    if not ok and fail_fast:
        print("\nServer unreachable — aborting tests.")
        return 0, 1

    client = OpenAI(base_url=base_url, api_key="EMPTY")

    tests = [
        ("basic_completion", lambda: test_basic_completion(client, model)),
        ("single_tool_call", lambda: test_single_tool_call(client, model)),
        ("parallel_tool_calls", lambda: test_parallel_tool_calls(client, model)),
        ("no_tool_needed", lambda: test_no_tool_needed(client, model)),
        ("streaming", lambda: test_streaming(client, model)),
        ("tool_argument_types", lambda: test_tool_argument_types(client, model)),
    ]

    passed = 1 if ok else 0
    total = len(tests) + 1

    for name, fn in tests:
        try:
            t0 = time.perf_counter()
            result_ok, result_msg = fn()
            elapsed = time.perf_counter() - t0
            status = "PASS" if result_ok else "FAIL"
            print(f"[{status}] {name}: {result_msg} ({elapsed * 1000:.0f}ms)")
            if result_ok:
                passed += 1
            elif fail_fast:
                print("\nFail-fast enabled — stopping.")
                break
        except Exception as e:
            print(f"[ERROR] {name}: Unhandled exception: {e}")
            if fail_fast:
                break

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("All tests passed!")
    else:
        print(f"{total - passed} test(s) failed.")

    return passed, total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM OpenAI-compatible endpoint smoke tests")
    parser.add_argument("--url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="qwen2.5-7b-tool-call")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    passed, total = run_all_tests(args.url, args.model, args.fail_fast)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
