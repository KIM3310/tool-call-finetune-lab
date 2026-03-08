"""Utility: convert standard tool-call examples to Qwen2.5 ChatML format.

Qwen2.5-Instruct uses ChatML with special tool-call markup:
  <tool_call>
  {"name": "...", "arguments": {...}}
  </tool_call>

This module formats training examples for the tokenizer's apply_chat_template
as well as producing raw ChatML strings for inspection.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Qwen2.5 special tokens
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"


def format_tool_definition(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a tool definition to the format Qwen2.5 expects.

    Qwen2.5 expects tools as:
    {
      "type": "function",
      "function": {
        "name": str,
        "description": str,
        "parameters": { "type": "object", "properties": {...}, "required": [...] }
      }
    }
    """
    if "function" not in tool:
        # Flat format — wrap it
        return {
            "type": "function",
            "function": {
                "name": tool.get("name", "unknown"),
                "description": tool.get("description", ""),
                "parameters": tool.get(
                    "parameters", {"type": "object", "properties": {}, "required": []}
                ),
            },
        }
    return tool


def _tool_call_to_chatml(tool_call: Dict[str, Any]) -> str:
    """Render a single tool call as a ChatML <tool_call> block."""
    fn = tool_call.get("function", tool_call)
    name = fn.get("name", "")
    args_raw = fn.get("arguments", "{}")

    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            args = {"_raw": args_raw}
    else:
        args = args_raw

    payload = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
    return f"{TOOL_CALL_OPEN}\n{payload}\n{TOOL_CALL_CLOSE}"


def _tool_response_to_chatml(content: str) -> str:
    """Render a tool response as a ChatML <tool_response> block."""
    return f"{TOOL_RESPONSE_OPEN}\n{content}\n{TOOL_RESPONSE_CLOSE}"


def example_to_chatml(
    example: Dict[str, Any],
    add_generation_prompt: bool = False,
) -> str:
    """Convert a standard training example dict to a raw ChatML string.

    This is primarily for debugging / inspection. For actual training, use
    ``example_to_hf_messages`` + ``tokenizer.apply_chat_template``.
    """
    messages = example.get("messages", [])
    tools = example.get("tools", [])
    parts: List[str] = []

    # If there are tools, prepend them to the system message
    tools_block: Optional[str] = None
    if tools:
        normalized = [format_tool_definition(t) for t in tools]
        tools_json = json.dumps(normalized, ensure_ascii=False, indent=2)
        tools_block = f"# Tools\n\nYou have access to the following tools:\n\n{tools_json}"

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        if role == "system":
            if tools_block:
                content = tools_block + ("\n\n" + content if content.strip() else "")
            parts.append(f"{IM_START}system\n{content}{IM_END}")

        elif role == "user":
            parts.append(f"{IM_START}user\n{content}{IM_END}")

        elif role == "assistant":
            if tool_calls:
                call_blocks = "\n".join(_tool_call_to_chatml(tc) for tc in tool_calls)
                # Prepend any text content before tool calls
                full_content = (content + "\n" + call_blocks).strip() if content.strip() else call_blocks
                parts.append(f"{IM_START}assistant\n{full_content}{IM_END}")
            else:
                parts.append(f"{IM_START}assistant\n{content}{IM_END}")

        elif role == "tool":
            # Tool/function responses
            response_block = _tool_response_to_chatml(content)
            parts.append(f"{IM_START}tool\n{response_block}{IM_END}")

    if add_generation_prompt:
        parts.append(f"{IM_START}assistant\n")

    return "\n".join(parts)


def example_to_hf_messages(
    example: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert a standard example to a HuggingFace-compatible messages list.

    This format is passed to ``tokenizer.apply_chat_template(messages, tools=tools)``.
    Qwen2.5 tokenizer handles tool injection automatically when tools is provided.
    """
    messages = example.get("messages", [])
    hf_messages: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        if role in ("system", "user"):
            hf_messages.append({"role": role, "content": content})

        elif role == "assistant":
            if tool_calls:
                # Format tool calls for HF chat template
                hf_tool_calls = []
                for tc in tool_calls:
                    fn = tc.get("function", tc)
                    name = fn.get("name", "")
                    args_raw = fn.get("arguments", "{}")
                    if isinstance(args_raw, str):
                        try:
                            args = json.loads(args_raw)
                        except json.JSONDecodeError:
                            args = {"_raw": args_raw}
                    else:
                        args = args_raw
                    hf_tool_calls.append(
                        {"type": "function", "function": {"name": name, "arguments": args}}
                    )
                hf_messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": hf_tool_calls,
                    }
                )
            else:
                hf_messages.append({"role": "assistant", "content": content})

        elif role == "tool":
            hf_messages.append({"role": "tool", "content": content})

    return hf_messages


def format_for_training(
    example: Dict[str, Any],
    tokenizer: Any,
    max_length: int = 4096,
) -> Optional[Dict[str, Any]]:
    """Apply the Qwen2.5 chat template and return tokenized input/label tensors.

    Returns None if the resulting sequence exceeds max_length.
    """
    tools = [format_tool_definition(t) for t in example.get("tools", [])]
    messages = example_to_hf_messages(example)

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tools=tools if tools else None,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as e:
        logger.warning("apply_chat_template failed: %s", e)
        # Fallback to raw ChatML
        text = example_to_chatml(example)

    tokens = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )

    if len(tokens["input_ids"]) >= max_length:
        return None

    return {"text": text, **tokens}
