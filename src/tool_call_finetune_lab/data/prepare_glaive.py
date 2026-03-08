"""Download and preprocess Glaive Function-Calling v2 dataset.

Converts conversational Glaive examples to the standard tool-call training
format used across this project, saving to data/raw/glaive.jsonl.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tool_call_finetune_lab.config import DataConfig

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
    has_tool_call = any(
        m.get("role") == "assistant" and m.get("tool_calls") for m in messages
    )
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
    from datasets import load_dataset

    examples: List[Dict[str, Any]] = []

    logger.info("Loading Glaive dataset from %s ...", config.glaive_repo)

    import os

    load_kwargs: Dict[str, Any] = {}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        load_kwargs["token"] = hf_token

    try:
        ds = load_dataset(config.glaive_repo, split="train", **load_kwargs)
        logger.info("Loaded Glaive dataset: %d rows", len(ds))

        for row in ds:
            ex = _parse_glaive_conversation(dict(row))
            if ex:
                examples.append(ex)
            if config.max_samples_glaive and len(examples) >= config.max_samples_glaive:
                break

    except Exception as e:
        logger.error("Failed to download Glaive dataset: %s", e)
        logger.info("Creating minimal synthetic Glaive examples for testing...")
        examples = _create_synthetic_examples()

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
    logger.info("Saved %d examples to %s", len(examples), path)


def main() -> None:
    config = DataConfig()
    examples = download_and_convert(config)
    save_jsonl(examples, config.glaive_output)
    logger.info("Glaive preparation complete.")


if __name__ == "__main__":
    main()
