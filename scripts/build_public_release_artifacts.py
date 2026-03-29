"""Build public-release status and evaluator smoke artifacts.

This script does not fabricate model benchmark numbers. It creates:
1. A small evaluator smoke artifact proving the BFCL scoring harness runs.
2. A release-status ledger showing current external publish blockers.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import requests

from tool_call_finetune_lab.eval.bfcl_runner import evaluate


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
TODAY = date.today().isoformat()


def _make_example(
    *,
    category: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source": "smoke",
        "category": category,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant with access to tools."},
            {"role": "user", "content": f"Call {tool_name} with the right arguments."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(arguments),
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
                    "description": "Smoke-test tool",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            key: {"type": "string"} for key in arguments.keys()
                        },
                        "required": list(arguments.keys()),
                    },
                },
            }
        ],
    }


class ReplayBackend:
    def __init__(self) -> None:
        self._responses = [
            ("", [{"name": "get_weather", "arguments": {"city": "Seoul"}}]),
            ("", [{"name": "search_flights", "arguments": {"destination": "Tokyo"}}]),
            ("", [{"name": "search_flights", "arguments": {"destination": "Busan"}}]),
            ("", [{"name": "search_hotels", "arguments": {"city": "Paris"}}]),
            ("", [{"name": "get_weather", "arguments": {"city": "New York"}}]),
        ]
        self._idx = 0

    def predict(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> tuple[str, Any]:
        response = self._responses[self._idx]
        self._idx += 1
        return response


def build_smoke_artifact() -> dict[str, Any]:
    examples = [
        _make_example(category="simple", tool_name="get_weather", arguments={"city": "Seoul"}),
        _make_example(category="multiple", tool_name="search_flights", arguments={"destination": "Tokyo"}),
        _make_example(category="multiple", tool_name="search_hotels", arguments={"city": "Busan"}),
        _make_example(category="parallel", tool_name="search_hotels", arguments={"city": "Paris"}),
        _make_example(category="simple", tool_name="get_weather", arguments={"city": "London"}),
    ]

    results = evaluate(ReplayBackend(), examples)
    artifact = {
        "artifact_type": "eval-harness-smoke",
        "generated_on": TODAY,
        "note": "Synthetic evaluator smoke artifact. This validates scoring/output shape only and is not a model benchmark.",
        "results": results,
    }
    return artifact


def _line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _probe_url(url: str) -> dict[str, Any]:
    try:
        response = requests.get(url, timeout=20, allow_redirects=True)
        return {"url": url, "status_code": response.status_code}
    except Exception as exc:  # pragma: no cover - network best effort
        return {"url": url, "error": f"{type(exc).__name__}: {exc}"}


def _probe_kaggle() -> dict[str, Any]:
    dataset_page = _probe_url("https://www.kaggle.com/datasets/doeonkim00/tool-call-finetune-data")
    kernel_page = _probe_url("https://www.kaggle.com/code/doeonkim00/tool-call-fine-tune-lab-qlora-pipeline")
    return {
        "public_dataset_page_probe": dataset_page,
        "kernel_page_probe": kernel_page,
        "last_authenticated_public_save_attempt": {
            "verified_on": TODAY,
            "status_code": 403,
            "message": "Phone verification is required to make a notebook public.",
        },
    }


def build_release_status() -> dict[str, Any]:
    train_file = ROOT / "data" / "processed" / "train.jsonl"
    val_file = ROOT / "data" / "processed" / "val.jsonl"
    test_file = ROOT / "data" / "processed" / "test.jsonl"

    return {
        "generated_on": TODAY,
        "data_snapshot": {
            "train_examples": _line_count(train_file) if train_file.exists() else 0,
            "val_examples": _line_count(val_file) if val_file.exists() else 0,
            "test_examples": _line_count(test_file) if test_file.exists() else 0,
        },
        "kaggle": _probe_kaggle(),
        "hugging_face": {
            "lora": _probe_url("https://huggingface.co/KIM3310/qwen2.5-7b-tool-calling-lora"),
            "awq": _probe_url("https://huggingface.co/KIM3310/qwen2.5-7b-tool-calling-awq"),
        },
    }


def write_markdown(smoke: dict[str, Any], status: dict[str, Any]) -> None:
    smoke_md = RESULTS_DIR / "eval_harness_smoke.md"
    release_md = RESULTS_DIR / "public_release_status.md"

    smoke_results = smoke["results"]["categories"]
    smoke_lines = [
        "# Eval Harness Smoke Artifact",
        "",
        f"- Generated on: `{TODAY}`",
        "- Purpose: prove the BFCL evaluator and JSON output path run locally",
        "- Boundary: synthetic replay backend, not a fine-tuned model benchmark",
        "",
        "| Category | Accuracy | Correct | Total |",
        "|---|---|---|---|",
    ]
    for category, stats in smoke_results.items():
        smoke_lines.append(
            f"| {category} | {stats['accuracy']}% | {stats['correct']} | {stats['total']} |"
        )
    smoke_md.write_text("\n".join(smoke_lines) + "\n", encoding="utf-8")

    kaggle = status["kaggle"]
    hf = status["hugging_face"]
    release_lines = [
        "# Public Release Status",
        "",
        f"- Generated on: `{TODAY}`",
        f"- Data snapshot: `{status['data_snapshot']['train_examples']}` train / `{status['data_snapshot']['val_examples']}` val / `{status['data_snapshot']['test_examples']}` test",
        "",
        "## Kaggle",
        "",
        f"- Public dataset page probe: `{kaggle.get('public_dataset_page_probe', {}).get('status_code', 'n/a')}`",
        f"- Kernel page probe: `{kaggle.get('kernel_page_probe', {}).get('status_code', 'n/a')}`",
        f"- Last authenticated public save attempt: `{kaggle.get('last_authenticated_public_save_attempt', {}).get('status_code', 'n/a')}`",
        "",
        "## Hugging Face",
        "",
        f"- LoRA repo probe: `{hf['lora'].get('status_code', 'n/a')}`",
        f"- AWQ repo probe: `{hf['awq'].get('status_code', 'n/a')}`",
        "",
        "## Interpretation",
        "",
        "- The attached Kaggle dataset page is public, but the notebook page is still unavailable.",
        "- The latest authenticated Kaggle save attempt failed with: `Phone verification is required to make a notebook public.`",
        "- The Hugging Face artifact URLs are not publicly reachable from this environment right now.",
        "- A full `results/bfcl_results.json` still requires the actual fine-tuned weights to be available again.",
    ]
    release_md.write_text("\n".join(release_lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    smoke = build_smoke_artifact()
    status = build_release_status()

    (RESULTS_DIR / "eval_harness_smoke.json").write_text(
        json.dumps(smoke, indent=2) + "\n",
        encoding="utf-8",
    )
    (RESULTS_DIR / "public_release_status.json").write_text(
        json.dumps(status, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(smoke, status)

    print("Wrote:")
    print(" - results/eval_harness_smoke.json")
    print(" - results/eval_harness_smoke.md")
    print(" - results/public_release_status.json")
    print(" - results/public_release_status.md")


if __name__ == "__main__":
    main()
