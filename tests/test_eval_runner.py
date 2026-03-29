"""Tests for the BFCL evaluation harness.

All model calls are mocked — these tests verify the scoring logic,
tool-call extraction, and result formatting without requiring a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

# ---------------------------------------------------------------------------
# Tool-call extraction tests
# ---------------------------------------------------------------------------


class TestExtractToolCalls:
    def test_extract_from_tool_call_tags(self) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import _extract_tool_calls_from_response

        text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>'
        calls = _extract_tool_calls_from_response(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["arguments"]["city"] == "Tokyo"

    def test_extract_multiple_tool_calls(self) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import _extract_tool_calls_from_response

        text = (
            '<tool_call>\n{"name": "fn_a", "arguments": {"x": 1}}\n</tool_call>\n'
            '<tool_call>\n{"name": "fn_b", "arguments": {"y": 2}}\n</tool_call>'
        )
        calls = _extract_tool_calls_from_response(text)
        assert len(calls) == 2
        assert calls[0]["name"] == "fn_a"
        assert calls[1]["name"] == "fn_b"

    def test_extract_from_bare_json(self) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import _extract_tool_calls_from_response

        text = '{"name": "my_func", "arguments": {"arg1": "value1"}}'
        calls = _extract_tool_calls_from_response(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "my_func"

    def test_extract_empty_text(self) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import _extract_tool_calls_from_response

        calls = _extract_tool_calls_from_response("")
        assert calls == []

    def test_extract_plain_text_no_calls(self) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import _extract_tool_calls_from_response

        calls = _extract_tool_calls_from_response("The answer is 42.")
        assert calls == []

    def test_extract_malformed_json_in_tag(self) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import _extract_tool_calls_from_response

        text = "<tool_call>not valid json</tool_call>"
        calls = _extract_tool_calls_from_response(text)
        # Should not crash, returns empty
        assert calls == []


# ---------------------------------------------------------------------------
# Tool-call matching tests
# ---------------------------------------------------------------------------


class TestToolCallMatching:
    def test_exact_match(self) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import _tool_call_matches

        pred = {"name": "get_weather", "arguments": {"city": "Tokyo"}}
        exp = {"name": "get_weather", "arguments": {"city": "Tokyo"}}
        assert _tool_call_matches(pred, exp) is True

    def test_name_mismatch(self) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import _tool_call_matches

        pred = {"name": "wrong_tool", "arguments": {"city": "Tokyo"}}
        exp = {"name": "get_weather", "arguments": {"city": "Tokyo"}}
        assert _tool_call_matches(pred, exp) is False

    def test_missing_argument(self) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import _tool_call_matches

        pred = {"name": "get_weather", "arguments": {}}
        exp = {"name": "get_weather", "arguments": {"city": "Tokyo"}}
        assert _tool_call_matches(pred, exp) is False

    def test_extra_arguments_are_ok(self) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import _tool_call_matches

        # Prediction has extra args beyond what's expected — still matches
        pred = {"name": "get_weather", "arguments": {"city": "Tokyo", "unit": "celsius"}}
        exp = {"name": "get_weather", "arguments": {"city": "Tokyo"}}
        assert _tool_call_matches(pred, exp) is True

    def test_case_insensitive_values(self) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import _tool_call_matches

        pred = {"name": "get_weather", "arguments": {"city": "TOKYO"}}
        exp = {"name": "get_weather", "arguments": {"city": "tokyo"}}
        assert _tool_call_matches(pred, exp) is True

    def test_arguments_as_json_string(self) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import _tool_call_matches

        pred = {"name": "get_weather", "arguments": '{"city": "Tokyo"}'}
        exp = {"name": "get_weather", "arguments": {"city": "Tokyo"}}
        assert _tool_call_matches(pred, exp) is True

    def test_empty_expected_arguments(self) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import _tool_call_matches

        pred = {"name": "fn", "arguments": {"x": 1}}
        exp = {"name": "fn", "arguments": {}}
        assert _tool_call_matches(pred, exp) is True


# ---------------------------------------------------------------------------
# Normalize arguments tests
# ---------------------------------------------------------------------------


class TestNormalizeArguments:
    def test_dict_passthrough(self) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import _normalize_arguments

        d = {"a": 1, "b": "two"}
        assert _normalize_arguments(d) == d

    def test_json_string(self) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import _normalize_arguments

        result = _normalize_arguments('{"x": 42}')
        assert result == {"x": 42}

    def test_invalid_json_string(self) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import _normalize_arguments

        result = _normalize_arguments("not json")
        assert "_raw" in result

    def test_non_dict_non_string(self) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import _normalize_arguments

        result = _normalize_arguments(None)
        assert result == {}


# ---------------------------------------------------------------------------
# Mock backend + evaluate() tests
# ---------------------------------------------------------------------------


class MockBackend:
    """A mock inference backend for testing the evaluate() loop."""

    def __init__(self, responses: List[Tuple[str, Optional[List[Dict[str, Any]]]]]) -> None:
        self._responses = responses
        self._call_count = 0

    def predict(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        resp = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return resp


def _make_test_example(
    category: str = "simple",
    tool_name: str = "get_weather",
    arg_key: str = "city",
    arg_val: str = "Tokyo",
) -> Dict[str, Any]:
    return {
        "source": "bfcl",
        "category": category,
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps({arg_key: arg_val}),
                        },
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": "Test tool",
                    "parameters": {
                        "type": "object",
                        "properties": {arg_key: {"type": "string"}},
                        "required": [arg_key],
                    },
                },
            }
        ],
    }


class TestEvaluate:
    def test_all_correct(self) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import evaluate

        examples = [_make_test_example() for _ in range(5)]
        # Mock backend always returns the correct tool call
        backend = MockBackend([("", [{"name": "get_weather", "arguments": {"city": "Tokyo"}}])])
        results = evaluate(backend, examples)
        assert results["categories"]["simple"]["correct"] == 5
        assert results["categories"]["simple"]["total"] == 5
        assert results["categories"]["simple"]["accuracy"] == 100.0
        assert results["categories"]["_overall"]["accuracy"] == 100.0

    def test_all_wrong_name(self) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import evaluate

        examples = [_make_test_example() for _ in range(3)]
        # Always calls the wrong tool
        backend = MockBackend([("", [{"name": "wrong_tool", "arguments": {"city": "Tokyo"}}])])
        results = evaluate(backend, examples)
        assert results["categories"]["simple"]["correct"] == 0
        assert results["categories"]["simple"]["accuracy"] == 0.0

    def test_no_tool_calls(self) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import evaluate

        examples = [_make_test_example() for _ in range(4)]
        # Backend returns no tool calls
        backend = MockBackend([("Just some text", None)])
        results = evaluate(backend, examples)
        assert results["categories"]["simple"]["correct"] == 0

    def test_mixed_correct_incorrect(self) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import evaluate

        examples = [_make_test_example() for _ in range(4)]
        # Alternate correct/incorrect
        backend = MockBackend(
            [
                ("", [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]),
                ("", [{"name": "wrong", "arguments": {}}]),
            ]
        )
        results = evaluate(backend, examples)
        # 2 out of 4 correct
        assert results["categories"]["simple"]["correct"] == 2
        assert results["categories"]["simple"]["accuracy"] == 50.0

    def test_multiple_categories(self) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import evaluate

        examples = [_make_test_example(category="simple") for _ in range(3)] + [
            _make_test_example(category="multiple") for _ in range(2)
        ]
        backend = MockBackend([("", [{"name": "get_weather", "arguments": {"city": "Tokyo"}}])])
        results = evaluate(backend, examples)
        assert "simple" in results["categories"]
        assert "multiple" in results["categories"]
        assert results["categories"]["_overall"]["total"] == 5

    def test_overall_accuracy_computed(self) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import evaluate

        examples = [_make_test_example() for _ in range(10)]
        backend = MockBackend([("", [{"name": "get_weather", "arguments": {"city": "Tokyo"}}])])
        results = evaluate(backend, examples)
        overall = results["categories"]["_overall"]
        assert overall["correct"] == 10
        assert overall["total"] == 10
        assert overall["accuracy"] == 100.0

    def test_failures_are_logged(self) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import evaluate

        examples = [_make_test_example() for _ in range(3)]
        # All fail
        backend = MockBackend([("", [{"name": "bad_tool", "arguments": {}}])])
        results = evaluate(backend, examples)
        assert len(results["failures"]) == 3
        assert results["failures"][0]["category"] == "simple"

    def test_empty_examples(self) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import evaluate

        backend = MockBackend([("", None)])
        results = evaluate(backend, [])
        assert results["categories"]["_overall"]["total"] == 0
        assert results["categories"]["_overall"]["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# load_test_data tests
# ---------------------------------------------------------------------------


class TestLoadTestData:
    def test_loads_examples_with_tool_calls(self, tmp_path: Path) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import load_test_data

        ex = _make_test_example()
        p = tmp_path / "test.jsonl"
        p.write_text(json.dumps(ex) + "\n")
        examples = load_test_data(str(p))
        assert len(examples) == 1

    def test_filters_examples_without_tool_calls(self, tmp_path: Path) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import load_test_data

        no_call = {
            "source": "bfcl",
            "category": "simple",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "tools": [],
        }
        p = tmp_path / "test.jsonl"
        p.write_text(json.dumps(no_call) + "\n")
        examples = load_test_data(str(p))
        assert len(examples) == 0

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import load_test_data

        with pytest.raises(FileNotFoundError):
            load_test_data(str(tmp_path / "missing.jsonl"))

    def test_mixed_examples(self, tmp_path: Path) -> None:
        from tool_call_finetune_lab.eval.bfcl_runner import load_test_data

        ex_with_call = _make_test_example()
        ex_without_call = {
            "source": "bfcl",
            "category": "simple",
            "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}],
            "tools": [],
        }
        p = tmp_path / "test.jsonl"
        lines = json.dumps(ex_with_call) + "\n" + json.dumps(ex_without_call) + "\n"
        p.write_text(lines)
        examples = load_test_data(str(p))
        assert len(examples) == 1


# ---------------------------------------------------------------------------
# Stage-pilot bridge tests
# ---------------------------------------------------------------------------


class TestStagePilotBridge:
    def test_score_test_case_pass(self) -> None:
        from tool_call_finetune_lab.eval.stage_pilot_bridge import _score_test_case

        tc = {
            "name": "simple",
            "expected_tool_name": "get_weather",
            "expected_args_subset": {"city": "Tokyo"},
        }
        calls = [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
        passed, _reason = _score_test_case(tc, calls)
        assert passed is True

    def test_score_test_case_wrong_name(self) -> None:
        from tool_call_finetune_lab.eval.stage_pilot_bridge import _score_test_case

        tc = {"name": "x", "expected_tool_name": "get_weather"}
        calls = [{"name": "bad_tool", "arguments": {}}]
        passed, reason = _score_test_case(tc, calls)
        assert passed is False
        assert "expected tool" in reason.lower() or "got" in reason.lower()

    def test_score_test_case_no_call_when_expected_none(self) -> None:
        from tool_call_finetune_lab.eval.stage_pilot_bridge import _score_test_case

        tc = {"name": "no_tool", "expected_tool_name": None}
        passed, _reason = _score_test_case(tc, None)
        assert passed is True

    def test_score_test_case_unexpected_tool_call(self) -> None:
        from tool_call_finetune_lab.eval.stage_pilot_bridge import _score_test_case

        tc = {"name": "no_tool", "expected_tool_name": None}
        calls = [{"name": "surprise_tool", "arguments": {}}]
        passed, _reason = _score_test_case(tc, calls)
        assert passed is False

    def test_score_test_case_missing_arg(self) -> None:
        from tool_call_finetune_lab.eval.stage_pilot_bridge import _score_test_case

        tc = {
            "name": "x",
            "expected_tool_name": "get_weather",
            "expected_args_subset": {"city": "Tokyo"},
        }
        calls = [{"name": "get_weather", "arguments": {"unit": "celsius"}}]
        passed, reason = _score_test_case(tc, calls)
        assert passed is False
        assert "city" in reason

    def test_score_test_case_wrong_arg_value(self) -> None:
        from tool_call_finetune_lab.eval.stage_pilot_bridge import _score_test_case

        tc = {
            "name": "x",
            "expected_tool_name": "get_weather",
            "expected_args_subset": {"city": "Tokyo"},
        }
        calls = [{"name": "get_weather", "arguments": {"city": "London"}}]
        passed, _reason = _score_test_case(tc, calls)
        assert passed is False

    def test_score_test_case_expected_count(self) -> None:
        from tool_call_finetune_lab.eval.stage_pilot_bridge import _score_test_case

        tc = {
            "name": "parallel",
            "expected_tool_name": "get_weather",
            "expected_call_count": 2,
        }
        calls = [{"name": "get_weather", "arguments": {"city": "A"}}]  # Only 1
        passed, _reason = _score_test_case(tc, calls)
        assert passed is False

    def test_score_test_case_enough_calls(self) -> None:
        from tool_call_finetune_lab.eval.stage_pilot_bridge import _score_test_case

        tc = {
            "name": "parallel",
            "expected_tool_name": "get_weather",
            "expected_call_count": 2,
        }
        calls = [
            {"name": "get_weather", "arguments": {"city": "A"}},
            {"name": "get_weather", "arguments": {"city": "B"}},
        ]
        passed, _reason = _score_test_case(tc, calls)
        assert passed is True
