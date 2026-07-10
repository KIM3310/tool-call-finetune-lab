"""Download and preprocess the Berkeley Function-Calling Leaderboard (v4) dataset.

Downloads question files and their matching ground-truth answer files from the
BFCL GitHub repo, joins them by ID, and converts to a standard chat format with
tool definitions and expected tool calls. Saves to data/raw/bfcl.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from tool_call_finetune_lab.config import DataConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BFCL_GITHUB_BASE = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla"
    "/{revision}/berkeley-function-call-leaderboard/bfcl_eval/data"
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


def _build_bfcl_raw_url(relative_path: str, revision: str) -> str:
    """Build a pinned BFCL raw GitHub URL and reject unsafe paths."""
    if relative_path.startswith("/") or ".." in Path(relative_path).parts:
        raise ValueError(f"Unsafe BFCL path: {relative_path}")
    return f"{BFCL_GITHUB_BASE.format(revision=revision)}/{relative_path}"


def _expected_bfcl_files() -> set[str]:
    return {path for pair in BFCL_FILE_PAIRS for path in pair}


def _require_bfcl_checksums(checksums: dict[str, str]) -> None:
    missing = sorted(_expected_bfcl_files() - set(checksums))
    if missing:
        raise RuntimeError(f"Missing BFCL checksums for expected shards: {', '.join(missing)}")


def _validate_bfcl_download_url(url: str, revision: str, relative_path: str) -> None:
    """Ensure redirects stay on the expected raw GitHub repository path and revision."""
    parsed = urlparse(url)
    expected_path = (
        f"/ShishirPatil/gorilla/{revision}/"
        f"berkeley-function-call-leaderboard/bfcl_eval/data/{relative_path}"
    )
    if parsed.scheme != "https" or parsed.netloc != "raw.githubusercontent.com":
        raise ValueError(f"Unexpected BFCL download host: {url}")
    if parsed.path != expected_path:
        raise ValueError(
            "Unexpected BFCL download path or revision: "
            f"expected {expected_path}, got {parsed.path}"
        )


def _download_jsonl(
    url: str,
    expected_sha256: str,
    revision: str,
    relative_path: str,
) -> List[Dict[str, Any]]:
    """Download a JSONL file and return parsed rows."""
    import httpx

    resp = httpx.get(url, timeout=60, follow_redirects=True)
    resp.raise_for_status()
    final_url = str(resp.url)
    _validate_bfcl_download_url(final_url, revision, relative_path)
    content = resp.text
    actual_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"SHA-256 mismatch for {final_url}: expected {expected_sha256}, got {actual_sha256}"
        )
    rows = []
    for line in content.strip().splitlines():
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
        result: Dict[str, Any] = {}
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
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": fn["name"],
                    "description": fn.get("description", f"Function {fn['name']}"),
                    "parameters": params,
                },
            }
        )
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
                        # Take first acceptable value, skipping empty strings
                        val = acceptable_values[0]
                        if val == "" and len(acceptable_values) > 1:
                            val = acceptable_values[1]
                        arguments[arg_name] = val
                    else:
                        arguments[arg_name] = acceptable_values
            tool_calls.append(
                {
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            )
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
    downloaded_rows: dict[str, List[Dict[str, Any]]] = {}

    logger.info("Downloading BFCL v4 dataset from GitHub revision %s ...", config.bfcl_revision)
    _require_bfcl_checksums(config.bfcl_checksums)

    try:
        for relative_path in sorted(_expected_bfcl_files()):
            url = _build_bfcl_raw_url(relative_path, config.bfcl_revision)
            logger.info("  %s ...", relative_path)
            downloaded_rows[relative_path] = _download_jsonl(
                url,
                config.bfcl_checksums[relative_path],
                config.bfcl_revision,
                relative_path,
            )
    except Exception as e:
        raise RuntimeError(f"BFCL download failed; no partial BFCL data will be used: {e}") from e

    for q_file, a_file in BFCL_FILE_PAIRS:
        category = q_file.replace(".json", "")
        q_rows = downloaded_rows[q_file]
        a_rows = downloaded_rows[a_file]

        logger.info("  %s ...", q_file)
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
                ex["provenance"] = {
                    "source_revision": config.bfcl_revision,
                    "source_files": [q_file, a_file],
                    "synthetic_fixture": False,
                }
                examples.append(ex)
                converted += 1

        logger.info("    -> %d/%d rows converted from %s", converted, len(q_rows), category)

    if not examples and not config.allow_synthetic_fixtures:
        raise RuntimeError(
            "No BFCL examples converted. Re-run with allow_synthetic_fixtures=True "
            "or --allow-synthetic-fixtures only for explicit fixture generation."
        )

    if not examples:
        logger.warning("No BFCL examples converted. Using explicitly allowed synthetic fixtures.")
        examples = _create_synthetic_examples()
        for ex in examples:
            ex["provenance"] = {
                "source_revision": config.bfcl_revision,
                "source_files": [],
                "synthetic_fixture": True,
            }

    if config.max_samples_bfcl and len(examples) > config.max_samples_bfcl:
        import random

        # Deterministic sample cap, not cryptographic use.
        random.seed(config.seed)
        examples = random.sample(examples, config.max_samples_bfcl)  # nosec B311

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
    _write_provenance(path, examples)
    logger.info("Saved %d examples to %s", len(examples), path)


def _write_provenance(path: Path, examples: List[Dict[str, Any]]) -> None:
    content = path.read_bytes()
    source_revisions = sorted(
        {
            str(ex.get("provenance", {}).get("source_revision"))
            for ex in examples
            if ex.get("provenance", {}).get("source_revision")
        }
    )
    provenance = {
        "artifact": str(path),
        "row_count": len(examples),
        "sha256": hashlib.sha256(content).hexdigest(),
        "source_revisions": source_revisions,
        "synthetic_fixture": any(
            bool(ex.get("provenance", {}).get("synthetic_fixture")) for ex in examples
        ),
    }
    provenance_path = path.with_suffix(path.suffix + ".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare pinned BFCL data")
    parser.add_argument("--allow-synthetic-fixtures", action="store_true")
    parser.add_argument("--bfcl-revision", default=None)
    args = parser.parse_args()

    kwargs: Dict[str, Any] = {"allow_synthetic_fixtures": args.allow_synthetic_fixtures}
    if args.bfcl_revision:
        kwargs["bfcl_revision"] = args.bfcl_revision
    config = DataConfig(**kwargs)
    examples = download_and_convert(config)
    save_jsonl(examples, config.bfcl_output)
    logger.info("BFCL preparation complete.")


if __name__ == "__main__":
    main()
