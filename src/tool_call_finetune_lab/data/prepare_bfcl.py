"""Download and preprocess the Berkeley Function-Calling Leaderboard (v4) dataset.

Downloads question files and their matching ground-truth answer files from the
BFCL GitHub repo, joins them by ID, and converts to a standard chat format with
tool definitions and expected tool calls. Saves to data/raw/bfcl.jsonl.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tool_call_finetune_lab.config import DataConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BFCL_GITHUB_BASE = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla"
    "/main/berkeley-function-call-leaderboard/bfcl_eval/data"
)

# (question_file, answer_file) pairs
BFCL_FILE_PAIRS = [
    ("BFCL_v4_simple_python.json", "possible_answer/BFCL_v4_simple_python.json"),
    ("BFCL_v4_multiple.json", "possible_answer/BFCL_v4_multiple.json"),
    ("BFCL_v4_parallel.json", "possible_answer/BFCL_v4_parallel.json"),
    ("BFCL_v4_parallel_multiple.json", "possible_answer/BFCL_v4_parallel_multiple.json"),
    ("BFCL_v4_simple_java.json", "possible_answer/BFCL_v4_simple_java.json"),
    ("BFCL_v4_simple_javascript.json", "possible_answer/BFCL_v4_simple_javascript.json"),
    ("BFCL_v4_live_simple.json", "possible_answer/BFCL_v4_live_simple.json"),
    ("BFCL_v4_live_multiple.json", "possible_answer/BFCL_v4_live_multiple.json"),
    ("BFCL_v4_live_parallel.json", "possible_answer/BFCL_v4_live_parallel.json"),
    ("BFCL_v4_live_parallel_multiple.json", "possible_answer/BFCL_v4_live_parallel_multiple.json"),
]

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "When the user's request requires a tool call, respond ONLY with the tool call(s). "
    "Do not add explanation before or after tool calls."
)


def _download_jsonl(url: str) -> List[Dict[str, Any]]:
    """Download a JSONL file and return parsed rows."""
    import httpx

    resp = httpx.get(url, timeout=60, follow_redirects=True)
    resp.raise_for_status()
    rows = []
    for line in resp.text.strip().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _fix_param_type(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert BFCL's 'type': 'dict' to standard JSON Schema 'type': 'object'."""
    if isinstance(params, dict):
        result = {}
        for k, v in params.items():
            if k == "type" and v == "dict":
                result[k] = "object"
            elif isinstance(v, dict):
                result[k] = _fix_param_type(v)
            elif isinstance(v, list):
                result[k] = [_fix_param_type(i) if isinstance(i, dict) else i for i in v]
            else:
                result[k] = v
        return result
    return params


def _normalize_tools(functions_raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert BFCL function defs to OpenAI-style tool definitions."""
    tools = []
    for fn in functions_raw:
        if not isinstance(fn, dict) or "name" not in fn:
            continue
        params = _fix_param_type(fn.get("parameters", {}))
        tools.append({
            "type": "function",
            "function": {
                "name": fn["name"],
                "description": fn.get("description", f"Function {fn['name']}"),
                "parameters": params,
            },
        })
    return tools


def _normalize_ground_truth(gt: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert BFCL ground_truth format to OpenAI-style tool_calls.

    BFCL format: [{"func_name": {"arg1": [val1, val2], "arg2": [val]}}]
    Each arg has a list of acceptable values; we take the first.
    """
    tool_calls = []
    for entry in gt:
        if not isinstance(entry, dict):
            continue
        for func_name, args in entry.items():
            arguments = {}
            if isinstance(args, dict):
                for arg_name, acceptable_values in args.items():
                    if isinstance(acceptable_values, list) and acceptable_values:
                        # Take first acceptable value
                        val = acceptable_values[0]
                        # Skip empty string alternatives
                        if val == "" and len(acceptable_values) > 1:
                            val = acceptable_values[0]
                        arguments[arg_name] = val
                    else:
                        arguments[arg_name] = acceptable_values
            tool_calls.append({
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            })
    return tool_calls


def _extract_user_content(question: Any) -> Optional[str]:
    """Extract user message from BFCL's nested question format.

    BFCL v4 format: [[{"role": "user", "content": "..."}]]
    """
    if isinstance(question, list):
        # Unwrap nested lists
        while isinstance(question, list) and len(question) > 0 and isinstance(question[0], list):
            question = question[0]
        # Now should be a list of message dicts
        if isinstance(question, list):
            for msg in question:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return str(msg.get("content", ""))
    elif isinstance(question, str):
        return question
    return None


def _build_example(
    row: Dict[str, Any],
    answer: Dict[str, Any],
    category: str,
) -> Optional[Dict[str, Any]]:
    """Build a training example from a question row and its ground truth answer."""
    user_content = _extract_user_content(row.get("question"))
    if not user_content:
        return None

    functions_raw = row.get("function", [])
    if not isinstance(functions_raw, list):
        functions_raw = [functions_raw]

    tools = _normalize_tools(functions_raw)
    if not tools:
        return None

    gt = answer.get("ground_truth", [])
    if not isinstance(gt, list):
        gt = [gt]

    tool_calls = _normalize_ground_truth(gt)
    if not tool_calls:
        return None

    return {
        "source": "bfcl",
        "category": category,
        "id": row.get("id", ""),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
        ],
        "tools": tools,
    }


def download_and_convert(config: DataConfig) -> List[Dict[str, Any]]:
    """Download BFCL v4 from GitHub, join with answers, convert to training format."""
    examples: List[Dict[str, Any]] = []

    logger.info("Downloading BFCL v4 dataset from GitHub ...")

    for q_file, a_file in BFCL_FILE_PAIRS:
        category = q_file.replace(".json", "")
        q_url = f"{BFCL_GITHUB_BASE}/{q_file}"
        a_url = f"{BFCL_GITHUB_BASE}/{a_file}"

        try:
            logger.info("  %s ...", q_file)
            q_rows = _download_jsonl(q_url)
            a_rows = _download_jsonl(a_url)

            # Index answers by id
            answer_by_id = {r["id"]: r for r in a_rows if "id" in r}

            converted = 0
            for row in q_rows:
                row_id = row.get("id", "")
                answer = answer_by_id.get(row_id)
                if not answer:
                    continue
                ex = _build_example(row, answer, category)
                if ex:
                    examples.append(ex)
                    converted += 1

            logger.info("    -> %d/%d rows converted from %s", converted, len(q_rows), category)
        except Exception as e:
            logger.warning("  Failed %s: %s", q_file, e)

    if not examples:
        logger.warning("No BFCL examples converted. Using synthetic fallback.")
        examples = _create_synthetic_examples()

    if config.max_samples_bfcl and len(examples) > config.max_samples_bfcl:
        import random
        random.seed(config.seed)
        examples = random.sample(examples, config.max_samples_bfcl)

    logger.info("Total BFCL examples: %d", len(examples))
    return examples


def _create_synthetic_examples() -> List[Dict[str, Any]]:
    """Minimal synthetic examples for testing when download fails."""
    return [
        {
            "source": "bfcl",
            "category": "simple",
            "id": "synthetic_0",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "What's the weather in San Francisco?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "San Francisco", "unit": "celsius"}',
                            },
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                            },
                            "required": ["location"],
                        },
                    },
                }
            ],
        }
    ]


def save_jsonl(examples: List[Dict[str, Any]], output_path: str) -> None:
    """Write examples to a JSONL file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info("Saved %d examples to %s", len(examples), path)


def main() -> None:
    config = DataConfig()
    examples = download_and_convert(config)
    save_jsonl(examples, config.bfcl_output)
    logger.info("BFCL preparation complete.")


if __name__ == "__main__":
    main()
