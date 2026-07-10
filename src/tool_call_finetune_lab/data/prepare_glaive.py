"""Download and preprocess Glaive Function-Calling v2 dataset.

Converts conversational Glaive examples to the standard tool-call training
format used across this project, saving to data/raw/glaive.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tool_call_finetune_lab.config import DataConfig

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent
    load_dataset = None  # type: ignore[assignment]

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent
    hf_hub_download = None  # type: ignore[assignment]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Glaive conversation role tags
SYSTEM_TAG = "SYSTEM:"
USER_TAG = "USER:"
ASSISTANT_TAG = "ASSISTANT:"
FUNCTION_TAG = "FUNCTION RESPONSE:"

# Glaive tool-call markup
TOOL_CALL_RE = re.compile(r"<functioncall>\s*(.*?)\s*(?:</functioncall>|$)", re.DOTALL)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_verified_source(config: DataConfig, token: str | None) -> Path:
    if hf_hub_download is None:
        raise RuntimeError("huggingface-hub is not installed")
    source_path = Path(
        hf_hub_download(
            repo_id=config.glaive_repo,
            filename=config.glaive_filename,
            repo_type="dataset",
            revision=config.dataset_revision,
            token=token,
        )
    )
    actual_sha256 = _sha256_file(source_path)
    if actual_sha256.lower() != config.glaive_sha256.lower():
        raise ValueError(
            f"Glaive source SHA-256 mismatch: expected {config.glaive_sha256}, got {actual_sha256}"
        )
    return source_path


def _parse_system_block(system_str: str) -> tuple[str, List[Dict[str, Any]]]:
    """Extract system prompt text and tool definitions from a Glaive SYSTEM block.

    Glaive stores tool definitions as JSON arrays embedded in the system prompt,
    preceded by 'You have access to the following functions:\n\n'.
    """
    tools: List[Dict[str, Any]] = []

    # Try to extract the JSON array of function definitions
    json_match = re.search(r"\[.*\]", system_str, re.DOTALL)
    if json_match:
        try:
            raw_tools = json.loads(json_match.group(0))
            for t in raw_tools:
                if isinstance(t, dict):
                    if "type" not in t:
                        tools.append({"type": "function", "function": t})
                    else:
                        tools.append(t)
        except json.JSONDecodeError:
            pass

    # Strip the function definitions block for the clean system prompt
    clean = re.sub(r"You have access to the following functions.*", "", system_str, flags=re.DOTALL)
    clean = clean.strip() or (
        "You are a helpful assistant with access to tools. "
        "Use them when appropriate to answer the user's request."
    )

    return clean, tools


def _parse_tool_call_content(content: str) -> Optional[Dict[str, Any]]:
    """Extract a structured tool call from an assistant message containing <functioncall>."""
    match = TOOL_CALL_RE.search(content)
    if not match:
        return None

    raw = match.group(1).strip()
    try:
        call_obj = json.loads(raw)
        name = call_obj.get("name", "")
        arguments = call_obj.get("arguments") or call_obj.get("parameters") or {}
        if not name:
            return None
        return {
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments)
                if isinstance(arguments, dict)
                else str(arguments),
            },
        }
    except json.JSONDecodeError:
        # Try regex extraction of name/arguments
        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', raw)
        args_match = re.search(r'"arguments"\s*:\s*(\{.*\})', raw, re.DOTALL)
        if name_match:
            return {
                "type": "function",
                "function": {
                    "name": name_match.group(1),
                    "arguments": args_match.group(1) if args_match else "{}",
                },
            }
        return None


def _parse_glaive_conversation(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a single Glaive row into the standard training example format.

    Glaive rows have two columns:
      - system: the system prompt with embedded tool definitions
      - chat:   the full conversation as a single string with role tags
    """
    system_raw = row.get("system", "") or ""
    chat_raw = row.get("chat", "") or ""

    if not chat_raw.strip():
        return None

    system_prompt, tools = _parse_system_block(system_raw)

    # Split chat into segments by role tags
    segments = _split_chat(chat_raw)
    if not segments:
        return None

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    for role, content in segments:
        content = content.strip()
        if not content:
            continue

        if role == "user":
            messages.append({"role": "user", "content": content})

        elif role == "assistant":
            tool_call = _parse_tool_call_content(content)
            if tool_call:
                # Strip the <functioncall>...</functioncall> markup from text content
                clean_content = TOOL_CALL_RE.sub("", content).strip()
                msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": clean_content,
                    "tool_calls": [tool_call],
                }
            else:
                msg = {"role": "assistant", "content": content}
            messages.append(msg)

        elif role == "tool":
            # Function response — map to "tool" role
            messages.append({"role": "tool", "content": content})

    # Filter: must have at least one user + one assistant with tool_calls
    has_tool_call = any(m.get("role") == "assistant" and m.get("tool_calls") for m in messages)
    has_user = any(m.get("role") == "user" for m in messages)

    if not has_tool_call or not has_user:
        return None

    return {
        "source": "glaive",
        "category": "function_calling",
        "messages": messages,
        "tools": tools,
    }


def _split_chat(chat: str) -> List[tuple[str, str]]:
    """Split a raw Glaive chat string into (role, content) pairs."""
    # Pattern: role tags are USER:, ASSISTANT:, FUNCTION RESPONSE:
    pattern = re.compile(
        r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)",
        re.IGNORECASE,
    )
    parts = pattern.split(chat)

    segments: List[tuple[str, str]] = []
    i = 1  # parts[0] is text before first tag (usually empty)
    while i < len(parts) - 1:
        tag = parts[i].strip().upper().rstrip(":")
        content = parts[i + 1]
        if tag == "USER":
            segments.append(("user", content))
        elif tag == "ASSISTANT":
            segments.append(("assistant", content))
        elif tag in ("FUNCTION RESPONSE", "FUNCTION_RESPONSE"):
            segments.append(("tool", content))
        i += 2

    return segments


def download_and_convert(config: DataConfig) -> List[Dict[str, Any]]:
    """Download Glaive v2 from HuggingFace and convert to standard format."""
    examples: List[Dict[str, Any]] = []

    logger.info(
        "Loading Glaive dataset from %s revision %s ...",
        config.glaive_repo,
        config.dataset_revision,
    )

    import os

    hf_token = os.environ.get("HF_TOKEN")

    try:
        if load_dataset is None:
            raise RuntimeError("datasets is not installed")
        source_path = _download_verified_source(config, hf_token)
        # The Hub source is immutable and checksum-verified before local parsing.
        ds = load_dataset(  # nosec B615
            "json",
            data_files=str(source_path),
            split="train",
        )
        logger.info("Loaded Glaive dataset: %d rows", len(ds))

        for row in ds:
            ex = _parse_glaive_conversation(dict(row))
            if ex:
                ex["provenance"] = {
                    "source_revision": config.dataset_revision,
                    "source_repo": config.glaive_repo,
                    "source_filename": config.glaive_filename,
                    "source_sha256": config.glaive_sha256,
                    "synthetic_fixture": False,
                }
                examples.append(ex)
            if config.max_samples_glaive and len(examples) >= config.max_samples_glaive:
                break

    except Exception as e:
        if not config.allow_synthetic_fixtures:
            raise RuntimeError(f"Failed to download Glaive dataset: {e}") from e
        logger.warning("Failed to download Glaive dataset: %s", e)
        logger.info("Creating explicitly allowed minimal synthetic Glaive fixtures...")
        examples = _create_synthetic_examples()
        for ex in examples:
            ex["provenance"] = {
                "source_revision": config.dataset_revision,
                "source_repo": config.glaive_repo,
                "source_filename": config.glaive_filename,
                "source_sha256": None,
                "synthetic_fixture": True,
            }

    logger.info("Total Glaive examples after conversion: %d", len(examples))
    return examples


def _create_synthetic_examples() -> List[Dict[str, Any]]:
    """Create minimal synthetic Glaive-style examples for testing."""
    return [
        {
            "source": "glaive",
            "category": "function_calling",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant with access to tools.",
                },
                {"role": "user", "content": "Book a flight from NYC to LA for tomorrow."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "search_flights",
                                "arguments": '{"origin": "NYC", "destination": "LA", "date": "tomorrow"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"flights": [{"id": "AA123", "price": 299, "departure": "08:00"}]}',
                },
                {
                    "role": "assistant",
                    "content": "I found a flight AA123 departing at 8:00 AM for $299. Would you like to book it?",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_flights",
                        "description": "Search for available flights",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "origin": {"type": "string", "description": "Origin city/airport"},
                                "destination": {
                                    "type": "string",
                                    "description": "Destination city/airport",
                                },
                                "date": {"type": "string", "description": "Travel date"},
                            },
                            "required": ["origin", "destination", "date"],
                        },
                    },
                }
            ],
        },
        {
            "source": "glaive",
            "category": "function_calling",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant with access to tools.",
                },
                {"role": "user", "content": "What movies are showing in Seattle tonight?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_movies",
                                "arguments": '{"city": "Seattle", "date": "tonight"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"movies": [{"title": "Inception", "time": "7:30 PM"}, {"title": "Dune Part 2", "time": "9:00 PM"}]}',
                },
                {
                    "role": "assistant",
                    "content": "Tonight in Seattle you can see Inception at 7:30 PM or Dune Part 2 at 9:00 PM.",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_movies",
                        "description": "Get movies playing in a city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "date": {"type": "string"},
                            },
                            "required": ["city", "date"],
                        },
                    },
                }
            ],
        },
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

    parser = argparse.ArgumentParser(description="Prepare pinned Glaive data")
    parser.add_argument("--allow-synthetic-fixtures", action="store_true")
    parser.add_argument("--dataset-revision", default=None)
    args = parser.parse_args()

    kwargs: Dict[str, Any] = {"allow_synthetic_fixtures": args.allow_synthetic_fixtures}
    if args.dataset_revision:
        kwargs["dataset_revision"] = args.dataset_revision
    config = DataConfig(**kwargs)
    examples = download_and_convert(config)
    save_jsonl(examples, config.glaive_output)
    logger.info("Glaive preparation complete.")


if __name__ == "__main__":
    main()
