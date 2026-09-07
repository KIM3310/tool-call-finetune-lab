from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from tool_call_finetune_lab.data.merge_and_split import (
    _input_hash,
    deduplicate,
    load_jsonl,
    stratified_split,
)
from tool_call_finetune_lab.eval.bfcl_runner import (
    VLLMBackend,
    _extract_tool_calls_from_response,
    _tool_call_matches,
    evaluate,
)
from tool_call_finetune_lab.eval.compare import generate_comparison_table


def example(question: str = "question", answer: Any = 1, source: str = "fixture") -> dict[str, Any]:
    return {
        "source": source,
        "category": "simple",
        "messages": [
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "lookup", "arguments": {"value": answer}}},
                ],
            },
        ],
        "tools": [{"function": {"name": "lookup"}}],
    }


@pytest.mark.parametrize(
    ("predicted", "expected", "matches"),
    [
        ({"value": "1"}, {"value": 1}, False),
        ({"value": True}, {"value": 1}, False),
        ({"value": "SECRET"}, {"value": "secret"}, False),
        ({"value": 1, "extra": True}, {"value": 1}, False),
        ({"value": [1, 2]}, {"value": [2, 1]}, False),
        ({"value": float("nan")}, {"value": float("nan")}, False),
        ({"value": float("inf")}, {"value": float("inf")}, False),
        ({"value": 1.0}, {"value": 1}, True),
        ({"value": {"a": 1, "b": 2}}, {"value": {"b": 2, "a": 1}}, True),
        ("not-json", {}, False),
        (None, {}, False),
    ],
)
def test_json_argument_matching_preserves_meaning(
    predicted: Any, expected: Any, matches: bool
) -> None:
    assert (
        _tool_call_matches(
            {"name": "lookup", "arguments": predicted}, {"name": "lookup", "arguments": expected}
        )
        is matches
    )


@pytest.mark.parametrize("invalid", ["[]", "null", "3", "not-json"])
def test_invalid_tag_does_not_disappear_beside_a_valid_call(invalid: str) -> None:
    response = '<tool_call>{"name":"lookup","arguments":{}}</tool_call>'
    response += f"<tool_call>{invalid}</tool_call>"
    assert _extract_tool_calls_from_response(response) == []


def test_multiturn_evaluation_preserves_history_and_excludes_future_answers() -> None:
    row = example()
    row["messages"] += [
        {"role": "tool", "content": "earlier observation"},
        {"role": "user", "content": "next question"},
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "lookup", "arguments": {"value": 2}}},
            ],
        },
        {"role": "tool", "content": "future answer must not reach the model"},
    ]
    backend = Mock()
    backend.model_name = "fixture:no-model"
    backend.predict.return_value = ("", [{"name": "lookup", "arguments": {"value": 2}}])
    results = evaluate(backend, [row])
    assert backend.predict.call_args.args[0] == row["messages"][:4]
    assert results["categories"]["_overall"]["correct"] == 1
    assert results["metadata"]["official_bfcl_score"] is False


def test_denominator_and_truncated_failures_are_explicit() -> None:
    backend = Mock()
    backend.model_name = "fixture:no-model"
    backend.predict.return_value = ("", [])
    rows = [example(str(i)) for i in range(51)] + [{"messages": []}]
    result = evaluate(backend, rows)
    assert result["metadata"]["input_examples"] == 52
    assert result["metadata"]["evaluated_examples"] == 51
    assert result["metadata"]["skipped_examples"] == 1
    assert result["metadata"]["failure_count"] == 51
    assert result["metadata"]["failure_log_truncated"] is True
    assert len(result["failures"]) == 50


def test_distinct_labels_are_retained_but_matching_requests_never_cross_splits() -> None:
    rows = [
        example(f"request-{index}", answer, source)
        for index in range(20)
        for answer, source in [(1, "source-a"), (2, "source-b")]
    ]
    assert len(deduplicate(rows + [copy.deepcopy(rows[0])])) == len(rows)
    parts = stratified_split(rows, 0.6, 0.2, seed=7)
    assert parts == stratified_split(list(reversed(rows)), 0.6, 0.2, seed=7)
    inputs = [{_input_hash(row) for row in part} for part in parts]
    assert all(inputs)
    assert inputs[0].isdisjoint(inputs[1])
    assert inputs[0].isdisjoint(inputs[2])
    assert inputs[1].isdisjoint(inputs[2])
    assert sum(map(len, parts)) == len(rows)


@pytest.mark.parametrize(("train", "validation"), [(float("nan"), 0.1), (0.8, 0.3), (-0.1, 0.1)])
def test_invalid_partition_ratios_fail_explicitly(train: float, validation: float) -> None:
    with pytest.raises(ValueError, match="split ratios"):
        stratified_split([example()], train, validation, seed=0)


def test_corrupt_dataset_does_not_silently_drop_rows(tmp_path: Path) -> None:
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps(example()) + "\ninvalid-json\n")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_jsonl(str(path))


def test_comparison_rejects_incompatible_dataset_fingerprints(tmp_path: Path) -> None:
    base = {"metadata": {"scoring_contract": "v2", "dataset_sha256": "a", "model": "base"}}
    adapted = {"metadata": {"scoring_contract": "v2", "dataset_sha256": "b", "model": "adapted"}}
    with pytest.raises(ValueError, match="different scoring contracts or evaluation datasets"):
        generate_comparison_table(adapted, base, None, str(tmp_path / "comparison.md"))
    assert not (tmp_path / "comparison.md").exists()


def test_vllm_backend_applies_timeout_and_configured_key(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    factory = Mock()
    monkeypatch.setattr(openai, "OpenAI", factory)
    monkeypatch.setenv("VLLM_API_KEY", "fixture-only-key")
    VLLMBackend("http://localhost:8000/v1", "fixture-model", timeout=7)
    factory.assert_called_once_with(
        base_url="http://localhost:8000/v1", api_key="fixture-only-key", timeout=7
    )


def test_empty_comparison_does_not_claim_verified_metadata(tmp_path: Path) -> None:
    report = generate_comparison_table(None, None, None, str(tmp_path / "empty.md"))
    assert "comparability is unverified" in report
    assert "Matching dataset/scorer metadata verified" not in report


def test_comparison_does_not_relabel_partially_matching_categories(tmp_path: Path) -> None:
    result = {"categories": {"parallel_multiple_function": {"accuracy": 12.5}}}
    report = generate_comparison_table(None, result, None, str(tmp_path / "category.md"))
    assert "| Parallel | 12.5%" not in report
    assert "| Multiple | 12.5%" not in report
