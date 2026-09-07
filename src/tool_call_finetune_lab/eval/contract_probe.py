"""Reproduce evaluator and split invariants with explicit synthetic fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tool_call_finetune_lab.data.merge_and_split import _input_hash, deduplicate, stratified_split
from tool_call_finetune_lab.eval.bfcl_runner import _tool_call_matches, evaluate


def build_contract_report() -> dict[str, Any]:
    def example(question: str, answer: int, source: str) -> dict[str, Any]:
        return {
            "source": source,
            "category": "synthetic",
            "messages": [
                {"role": "user", "content": question},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"function": {"name": "lookup", "arguments": {"value": answer}}}
                    ],
                },
            ],
            "tools": [{"function": {"name": "lookup"}}],
        }

    cases: list[tuple[str, Any, Any, bool]] = [
        ("exact object", {"value": 1}, {"value": 1}, True),
        ("numeric string", {"value": "1"}, {"value": 1}, False),
        ("boolean as number", {"value": True}, {"value": 1}, False),
        ("extra argument", {"value": 1, "other": 2}, {"value": 1}, False),
        ("case-sensitive identifier", {"value": "Ab"}, {"value": "ab"}, False),
        ("JSON object key ordering", {"b": 2, "a": 1}, {"a": 1, "b": 2}, True),
    ]
    scoring = []
    for name, predicted, expected, wanted in cases:
        actual = _tool_call_matches(
            {"name": "lookup", "arguments": predicted}, {"name": "lookup", "arguments": expected}
        )
        assert actual is wanted, name
        scoring.append({"case": name, "accepted": actual, "expected": wanted})

    rows = [
        example(f"request-{i}", answer, source)
        for i in range(20)
        for answer, source in [(1, "source-a"), (2, "source-b")]
    ]
    unique = deduplicate(rows + [rows[0]])
    assert len(unique) == 40
    partitions = stratified_split(unique, 0.6, 0.2, seed=7)
    groups = [{_input_hash(row) for row in partition} for partition in partitions]
    assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i))
    assert partitions == stratified_split(list(reversed(unique)), 0.6, 0.2, seed=7)

    class FixtureBackend:
        model_name = "fixture:no-model"

        def predict(self, messages: Any, tools: Any) -> tuple[str, Any]:
            assert messages == [{"role": "user", "content": "request-0"}]
            return "", [{"name": "lookup", "arguments": {"value": 1}}]

    row = example("request-0", 1, "source-a")
    row["messages"].append({"role": "tool", "content": "future observation"})
    evaluated = evaluate(FixtureBackend(), [row])
    assert evaluated["categories"]["_overall"]["correct"] == 1
    package = Path(__file__).resolve().parents[1]
    source_paths = ["eval/bfcl_runner.py", "eval/compare.py", "data/merge_and_split.py"]
    return {
        "schema_version": 1,
        "scope": "Synthetic evaluator-contract checks; no model inference or training performed",
        "official_bfcl_score": False,
        "scoring_cases": scoring,
        "target_answer_and_future_context_excluded": True,
        "split": {
            "unique_rows": len(unique),
            "rows": list(map(len, partitions)),
            "request_groups": list(map(len, groups)),
            "cross_partition_request_overlap": 0,
            "input_order_invariant": True,
            "seed": 7,
        },
        "source_sha256": {
            path: hashlib.sha256((package / path).read_bytes()).hexdigest() for path in source_paths
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="evidence/evaluation-contract.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    destination = Path(args.output)
    payload = json.dumps(build_contract_report(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not destination.exists() or destination.read_text() != payload:
            raise SystemExit("Contract evidence differs; run make proof and review the change.")
        print("Evaluator contract evidence reproduced exactly.")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload)
        print(f"Contract evidence written: {destination}")


if __name__ == "__main__":
    main()
