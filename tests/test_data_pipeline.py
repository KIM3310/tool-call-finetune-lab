"""Tests for data preparation functions.

These tests run without GPU and without downloading datasets —
they exercise parsing, formatting, deduplication, and splitting logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_bfcl_row() -> Dict[str, Any]:
    return {
        "question": "What is the weather in San Francisco?",
        "function": [
            {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            }
        ],
        "answer": [{"name": "get_weather", "arguments": {"location": "San Francisco"}}],
        "category": "simple",
    }


@pytest.fixture
def sample_glaive_row() -> Dict[str, Any]:
    return {
        "system": (
            "SYSTEM: You are a helpful assistant with access to the following functions:\n\n"
            '[{"type": "function", "function": {"name": "search", "description": "Search the web",'
            ' "parameters": {"type": "object", "properties": {"query": {"type": "string"}},'
            ' "required": ["query"]}}}]\n\n'
            "If you choose to call a function respond with JSON."
        ),
        "chat": (
            "USER: Search for Python tutorials\n"
            'ASSISTANT: <functioncall> {"name": "search", "arguments": {"query": "Python tutorials"}} </functioncall>\n'
            'FUNCTION RESPONSE: {"results": [{"title": "Python Tutorial", "url": "https://example.com"}]}\n'
            "ASSISTANT: I found some Python tutorials for you."
        ),
    }


@pytest.fixture
def sample_examples() -> List[Dict[str, Any]]:
    """A list of standard training examples for testing merge/split logic."""
    return [
        {
            "source": "bfcl",
            "category": "simple",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": f"User question {i}"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "tool_a", "arguments": f'{{"arg": "{i}"}}'},
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "tool_a",
                        "description": "Tool A",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                }
            ],
        }
        for i in range(20)
    ] + [
        {
            "source": "glaive",
            "category": "function_calling",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": f"Glaive question {i}"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "tool_b", "arguments": f'{{"x": "{i}"}}'},
                        }
                    ],
                },
            ],
            "tools": [],
        }
        for i in range(10)
    ]


# ---------------------------------------------------------------------------
# BFCL parser tests
# ---------------------------------------------------------------------------


class TestBFCLParser:
    def test_bfcl_download_fails_closed_without_synthetic_opt_in(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from tool_call_finetune_lab.config import DataConfig
        from tool_call_finetune_lab.data import prepare_bfcl

        def fail_download(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
            raise RuntimeError(f"blocked {args} {kwargs}")

        monkeypatch.setattr(prepare_bfcl, "_download_jsonl", fail_download)
        cfg = DataConfig(raw_dir=str(tmp_path / "raw"), processed_dir=str(tmp_path / "processed"))

        with pytest.raises(RuntimeError, match="BFCL download failed"):
            prepare_bfcl.download_and_convert(cfg)

    def test_bfcl_requires_checksum_for_every_expected_shard(self, tmp_path: Path) -> None:
        from tool_call_finetune_lab.config import DataConfig
        from tool_call_finetune_lab.data import prepare_bfcl

        cfg = DataConfig(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
            bfcl_checksums={"BFCL_v4_simple_python.json": "0" * 64},
        )

        with pytest.raises(RuntimeError, match="Missing BFCL checksums"):
            prepare_bfcl.download_and_convert(cfg)

    def test_bfcl_later_shard_failure_discards_earlier_success(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from tool_call_finetune_lab.config import DataConfig
        from tool_call_finetune_lab.data import prepare_bfcl

        downloaded: list[str] = []
        fail_after = 2

        def fake_download(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
            url = str(args[0])
            downloaded.append(url)
            if len(downloaded) <= fail_after:
                return []
            raise RuntimeError("later shard unavailable")

        monkeypatch.setattr(prepare_bfcl, "_download_jsonl", fake_download)
        cfg = DataConfig(raw_dir=str(tmp_path / "raw"), processed_dir=str(tmp_path / "processed"))

        with pytest.raises(RuntimeError, match="BFCL download failed"):
            prepare_bfcl.download_and_convert(cfg)

        assert len(downloaded) > 2

    def test_bfcl_synthetic_opt_in_marks_provenance(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from tool_call_finetune_lab.config import DataConfig
        from tool_call_finetune_lab.data import prepare_bfcl

        def empty_download(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
            return []

        monkeypatch.setattr(prepare_bfcl, "_download_jsonl", empty_download)
        cfg = DataConfig(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
            allow_synthetic_fixtures=True,
        )

        examples = prepare_bfcl.download_and_convert(cfg)

        assert examples
        assert examples[0]["provenance"]["synthetic_fixture"] is True
        assert examples[0]["provenance"]["source_revision"] == cfg.bfcl_revision

    def test_bfcl_raw_url_uses_pinned_revision_and_validates_path(self) -> None:
        from tool_call_finetune_lab.data.prepare_bfcl import _build_bfcl_raw_url

        revision = "0123456789abcdef0123456789abcdef01234567"
        url = _build_bfcl_raw_url("BFCL_v4_simple_python.json", revision)

        assert "/main/" not in url
        assert f"/{revision}/berkeley-function-call-leaderboard/bfcl_eval/data/" in url

        with pytest.raises(ValueError, match="Unsafe BFCL path"):
            _build_bfcl_raw_url("../secret.json", revision)

    def test_download_jsonl_validates_checksum(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tool_call_finetune_lab.data.prepare_bfcl import _download_jsonl

        class Response:
            text = '{"id": "ok"}\n'
            url = (
                "https://raw.githubusercontent.com/ShishirPatil/gorilla/"
                "0123456789abcdef0123456789abcdef01234567/"
                "berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_simple_python.json"
            )

            def raise_for_status(self) -> None:
                return None

        def fake_get(*args: Any, **kwargs: Any) -> Response:
            return Response()

        import httpx

        monkeypatch.setattr(httpx, "get", fake_get)

        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            _download_jsonl(
                Response.url,
                "0" * 64,
                "0123456789abcdef0123456789abcdef01234567",
                "BFCL_v4_simple_python.json",
            )

    def test_download_jsonl_rejects_same_repo_redirect_to_main(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tool_call_finetune_lab.config import BFCL_PINNED_REVISION
        from tool_call_finetune_lab.data.prepare_bfcl import _download_jsonl

        class Response:
            text = '{"id": "ok"}\n'
            url = (
                "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
                "berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_simple_python.json"
            )

            def raise_for_status(self) -> None:
                return None

        def fake_get(*args: Any, **kwargs: Any) -> Response:
            return Response()

        import httpx

        monkeypatch.setattr(httpx, "get", fake_get)

        with pytest.raises(ValueError, match="revision"):
            _download_jsonl(
                (
                    "https://raw.githubusercontent.com/ShishirPatil/gorilla/"
                    f"{BFCL_PINNED_REVISION}/berkeley-function-call-leaderboard/"
                    "bfcl_eval/data/BFCL_v4_simple_python.json"
                ),
                "0" * 64,
                BFCL_PINNED_REVISION,
                "BFCL_v4_simple_python.json",
            )

    def test_build_example_basic(self) -> None:
        from tool_call_finetune_lab.data.prepare_bfcl import _build_example

        row = {
            "id": "test_0",
            "question": [[{"role": "user", "content": "Get weather in NYC"}]],
            "function": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "dict", "properties": {"location": {"type": "string"}}},
                }
            ],
        }
        answer = {"id": "test_0", "ground_truth": [{"get_weather": {"location": ["NYC"]}}]}
        ex = _build_example(row, answer, "simple")
        assert ex is not None
        assert ex["source"] == "bfcl"
        assert ex["category"] == "simple"

    def test_build_example_has_messages(self) -> None:
        from tool_call_finetune_lab.data.prepare_bfcl import _build_example

        row = {
            "id": "test_1",
            "question": [[{"role": "user", "content": "Get weather"}]],
            "function": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "dict", "properties": {"loc": {"type": "string"}}},
                }
            ],
        }
        answer = {"id": "test_1", "ground_truth": [{"get_weather": {"loc": ["SF"]}}]}
        ex = _build_example(row, answer, "simple")
        assert ex is not None
        roles = [m["role"] for m in ex["messages"]]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles

    def test_build_example_no_question_returns_none(self) -> None:
        from tool_call_finetune_lab.data.prepare_bfcl import _build_example

        row = {
            "id": "x",
            "question": [[]],
            "function": [{"name": "f", "description": "d", "parameters": {}}],
        }
        answer = {"id": "x", "ground_truth": [{"f": {}}]}
        assert _build_example(row, answer, "simple") is None

    def test_fix_param_type(self) -> None:
        from tool_call_finetune_lab.data.prepare_bfcl import _fix_param_type

        result = _fix_param_type({"type": "dict", "properties": {"a": {"type": "dict"}}})
        assert result["type"] == "object"
        assert result["properties"]["a"]["type"] == "object"

    def test_normalize_ground_truth(self) -> None:
        from tool_call_finetune_lab.data.prepare_bfcl import _normalize_ground_truth

        gt = [{"get_weather": {"location": ["NYC", "New York"], "unit": ["celsius"]}}]
        result = _normalize_ground_truth(gt)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"

    def test_normalize_ground_truth_skips_empty_string(self) -> None:
        """Regression test: empty string first value should use second value."""
        from tool_call_finetune_lab.data.prepare_bfcl import _normalize_ground_truth

        gt = [{"set_config": {"mode": ["", "auto"], "value": ["42"]}}]
        result = _normalize_ground_truth(gt)
        assert len(result) == 1
        args = json.loads(result[0]["function"]["arguments"])
        assert args["mode"] == "auto"  # Should skip "" and use "auto"
        assert args["value"] == "42"

    def test_extract_user_content(self) -> None:
        from tool_call_finetune_lab.data.prepare_bfcl import _extract_user_content

        assert _extract_user_content([[{"role": "user", "content": "hello"}]]) == "hello"
        assert _extract_user_content("plain string") == "plain string"
        assert _extract_user_content([[]]) is None

    def test_save_jsonl(self, tmp_path: Path) -> None:
        from tool_call_finetune_lab.data.prepare_bfcl import save_jsonl

        examples = [{"a": 1}, {"b": 2}]
        out = tmp_path / "test.jsonl"
        save_jsonl(examples, str(out))
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"a": 1}

    def test_synthetic_examples_are_valid(self) -> None:
        from tool_call_finetune_lab.data.prepare_bfcl import _create_synthetic_examples

        examples = _create_synthetic_examples()
        assert len(examples) >= 1
        for ex in examples:
            assert "messages" in ex
            assert "tools" in ex
            assert any(m["role"] == "user" for m in ex["messages"])


# ---------------------------------------------------------------------------
# Glaive parser tests
# ---------------------------------------------------------------------------


class TestGlaiveParser:
    def test_glaive_download_uses_pinned_revision_and_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from tool_call_finetune_lab.config import DataConfig
        from tool_call_finetune_lab.data import prepare_glaive

        captured: Dict[str, Any] = {}

        def fake_hf_hub_download(*args: Any, **kwargs: Any) -> Any:
            captured["args"] = args
            captured["kwargs"] = kwargs
            raise RuntimeError("network blocked")

        monkeypatch.setattr(prepare_glaive, "hf_hub_download", fake_hf_hub_download)
        cfg = DataConfig(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
        )

        with pytest.raises(RuntimeError, match="Failed to download Glaive dataset"):
            prepare_glaive.download_and_convert(cfg)

        assert captured["kwargs"]["revision"] == cfg.dataset_revision
        assert captured["kwargs"]["filename"] == cfg.glaive_filename
        assert captured["kwargs"]["repo_type"] == "dataset"

    def test_glaive_synthetic_opt_in_marks_provenance(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from tool_call_finetune_lab.config import DataConfig
        from tool_call_finetune_lab.data import prepare_glaive

        def fake_hf_hub_download(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("network blocked")

        monkeypatch.setattr(prepare_glaive, "hf_hub_download", fake_hf_hub_download)
        cfg = DataConfig(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
            allow_synthetic_fixtures=True,
        )

        examples = prepare_glaive.download_and_convert(cfg)

        assert examples
        assert examples[0]["provenance"]["synthetic_fixture"] is True
        assert examples[0]["provenance"]["source_revision"] == cfg.dataset_revision
        assert examples[0]["provenance"]["source_sha256"] is None

    def test_glaive_source_checksum_is_verified_before_parsing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        sample_glaive_row: Dict[str, Any],
    ) -> None:
        import hashlib

        from tool_call_finetune_lab.config import DataConfig
        from tool_call_finetune_lab.data import prepare_glaive

        source = tmp_path / "glaive.json"
        source.write_text("verified source", encoding="utf-8")
        expected_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        captured: Dict[str, Any] = {}

        monkeypatch.setattr(prepare_glaive, "hf_hub_download", lambda **_kwargs: str(source))

        def fake_load_dataset(*args: Any, **kwargs: Any) -> Any:
            captured["args"] = args
            captured["kwargs"] = kwargs
            return [sample_glaive_row]

        monkeypatch.setattr(prepare_glaive, "load_dataset", fake_load_dataset)
        cfg = DataConfig(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
            glaive_sha256=expected_sha256,
        )

        examples = prepare_glaive.download_and_convert(cfg)

        assert captured == {
            "args": ("json",),
            "kwargs": {"data_files": str(source), "split": "train"},
        }
        assert examples[0]["provenance"]["source_sha256"] == expected_sha256

    def test_glaive_source_checksum_mismatch_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from tool_call_finetune_lab.config import DataConfig
        from tool_call_finetune_lab.data import prepare_glaive

        source = tmp_path / "glaive.json"
        source.write_text("unexpected source", encoding="utf-8")
        monkeypatch.setattr(prepare_glaive, "hf_hub_download", lambda **_kwargs: str(source))
        monkeypatch.setattr(
            prepare_glaive,
            "load_dataset",
            lambda *_args, **_kwargs: pytest.fail("checksum must be verified before parsing"),
        )
        cfg = DataConfig(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
            glaive_sha256="0" * 64,
        )

        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            prepare_glaive.download_and_convert(cfg)

    def test_parse_glaive_conversation_basic(self, sample_glaive_row: Dict[str, Any]) -> None:
        from tool_call_finetune_lab.data.prepare_glaive import _parse_glaive_conversation

        ex = _parse_glaive_conversation(sample_glaive_row)
        assert ex is not None
        assert ex["source"] == "glaive"

    def test_parse_glaive_has_tool_call(self, sample_glaive_row: Dict[str, Any]) -> None:
        from tool_call_finetune_lab.data.prepare_glaive import _parse_glaive_conversation

        ex = _parse_glaive_conversation(sample_glaive_row)
        assert ex is not None
        assistant_msgs = [m for m in ex["messages"] if m.get("role") == "assistant"]
        tool_call_msgs = [m for m in assistant_msgs if m.get("tool_calls")]
        assert len(tool_call_msgs) >= 1

    def test_parse_glaive_tool_name(self, sample_glaive_row: Dict[str, Any]) -> None:
        from tool_call_finetune_lab.data.prepare_glaive import _parse_glaive_conversation

        ex = _parse_glaive_conversation(sample_glaive_row)
        assert ex is not None
        assistant_msgs = [m for m in ex["messages"] if m.get("tool_calls")]
        tc = assistant_msgs[0]["tool_calls"][0]
        assert tc["function"]["name"] == "search"

    def test_parse_glaive_empty_chat_returns_none(self) -> None:
        from tool_call_finetune_lab.data.prepare_glaive import _parse_glaive_conversation

        row: Dict[str, Any] = {"system": "", "chat": ""}
        assert _parse_glaive_conversation(row) is None

    def test_parse_glaive_no_tool_call_returns_none(self) -> None:
        from tool_call_finetune_lab.data.prepare_glaive import _parse_glaive_conversation

        row: Dict[str, Any] = {
            "system": "",
            "chat": "USER: Hello\nASSISTANT: Hi there!",
        }
        # No tool call — should return None (filtered out)
        assert _parse_glaive_conversation(row) is None

    def test_parse_system_block_extracts_tools(self) -> None:
        from tool_call_finetune_lab.data.prepare_glaive import _parse_system_block

        system = (
            "You have access to the following functions:\n\n"
            '[{"type": "function", "function": {"name": "foo", "description": "bar",'
            ' "parameters": {"type": "object", "properties": {}}}}]'
        )
        _prompt, tools = _parse_system_block(system)
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "foo"

    def test_split_chat(self) -> None:
        from tool_call_finetune_lab.data.prepare_glaive import _split_chat

        chat = "USER: Hello\nASSISTANT: Hi\nUSER: What time is it?\nASSISTANT: 3pm"
        segments = _split_chat(chat)
        assert len(segments) == 4
        assert segments[0] == ("user", " Hello\n")
        assert segments[1][0] == "assistant"

    def test_parse_tool_call_content(self) -> None:
        from tool_call_finetune_lab.data.prepare_glaive import _parse_tool_call_content

        content = (
            '<functioncall> {"name": "get_weather", "arguments": {"city": "Tokyo"}} </functioncall>'
        )
        result = _parse_tool_call_content(content)
        assert result is not None
        assert result["function"]["name"] == "get_weather"

    def test_synthetic_examples(self) -> None:
        from tool_call_finetune_lab.data.prepare_glaive import _create_synthetic_examples

        examples = _create_synthetic_examples()
        assert len(examples) >= 1
        for ex in examples:
            has_tc = any(
                m.get("role") == "assistant" and m.get("tool_calls") for m in ex["messages"]
            )
            assert has_tc


# ---------------------------------------------------------------------------
# Merge and split tests
# ---------------------------------------------------------------------------


class TestMergeAndSplit:
    def test_content_hash_is_deterministic(self, sample_examples: List[Dict[str, Any]]) -> None:
        from tool_call_finetune_lab.data.merge_and_split import _content_hash

        ex = sample_examples[0]
        h1 = _content_hash(ex)
        h2 = _content_hash(ex)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_content_hash_differs_for_different_examples(
        self, sample_examples: List[Dict[str, Any]]
    ) -> None:
        from tool_call_finetune_lab.data.merge_and_split import _content_hash

        h1 = _content_hash(sample_examples[0])
        h2 = _content_hash(sample_examples[1])
        assert h1 != h2

    def test_deduplicate_removes_duplicates(self, sample_examples: List[Dict[str, Any]]) -> None:
        from tool_call_finetune_lab.data.merge_and_split import deduplicate

        # Duplicate first example
        duped = sample_examples + [sample_examples[0]]
        result = deduplicate(duped)
        assert len(result) == len(sample_examples)

    def test_deduplicate_preserves_unique(self, sample_examples: List[Dict[str, Any]]) -> None:
        from tool_call_finetune_lab.data.merge_and_split import deduplicate

        result = deduplicate(sample_examples)
        assert len(result) == len(sample_examples)

    def test_stratified_split_sizes(self, sample_examples: List[Dict[str, Any]]) -> None:
        from tool_call_finetune_lab.data.merge_and_split import stratified_split

        train, val, test = stratified_split(sample_examples, 0.8, 0.1, seed=42)
        total = len(train) + len(val) + len(test)
        assert total == len(sample_examples)

    def test_stratified_split_no_overlap(self, sample_examples: List[Dict[str, Any]]) -> None:
        from tool_call_finetune_lab.data.merge_and_split import _content_hash, stratified_split

        train, val, test = stratified_split(sample_examples, 0.8, 0.1, seed=42)
        train_hashes = {_content_hash(e) for e in train}
        val_hashes = {_content_hash(e) for e in val}
        test_hashes = {_content_hash(e) for e in test}
        assert train_hashes.isdisjoint(val_hashes)
        assert train_hashes.isdisjoint(test_hashes)
        assert val_hashes.isdisjoint(test_hashes)

    def test_stratified_split_train_is_largest(self, sample_examples: List[Dict[str, Any]]) -> None:
        from tool_call_finetune_lab.data.merge_and_split import stratified_split

        train, val, test = stratified_split(sample_examples, 0.8, 0.1, seed=42)
        assert len(train) > len(val)
        assert len(train) > len(test)

    def test_load_jsonl(self, tmp_path: Path) -> None:
        from tool_call_finetune_lab.data.merge_and_split import load_jsonl

        p = tmp_path / "test.jsonl"
        p.write_text('{"a": 1}\n{"b": 2}\n')
        result = load_jsonl(str(p))
        assert result == [{"a": 1}, {"b": 2}]

    def test_load_jsonl_missing_file(self, tmp_path: Path) -> None:
        from tool_call_finetune_lab.data.merge_and_split import load_jsonl

        result = load_jsonl(str(tmp_path / "nonexistent.jsonl"))
        assert result == []

    def test_save_jsonl_roundtrip(
        self, tmp_path: Path, sample_examples: List[Dict[str, Any]]
    ) -> None:
        from tool_call_finetune_lab.data.merge_and_split import load_jsonl, save_jsonl

        out = tmp_path / "out.jsonl"
        save_jsonl(sample_examples[:5], str(out))
        loaded = load_jsonl(str(out))
        assert len(loaded) == 5
        provenance = json.loads((tmp_path / "out.jsonl.provenance.json").read_text())
        assert provenance["row_count"] == 5
        assert len(provenance["sha256"]) == 64
        assert provenance["synthetic_fixture"] is False


# ---------------------------------------------------------------------------
# Chat template formatting tests
# ---------------------------------------------------------------------------


class TestFormatChatTemplate:
    def test_example_to_chatml_has_roles(self) -> None:
        from tool_call_finetune_lab.data.format_chat_template import example_to_chatml

        ex = {
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
                                "name": "get_weather",
                                "arguments": '{"city": "Tokyo"}',
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
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                }
            ],
        }
        result = example_to_chatml(ex)
        assert "<|im_start|>system" in result
        assert "<|im_start|>user" in result
        assert "<|im_start|>assistant" in result

    def test_example_to_chatml_includes_tool_call(self) -> None:
        from tool_call_finetune_lab.data.format_chat_template import example_to_chatml

        ex = {
            "messages": [
                {"role": "user", "content": "Q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "my_tool", "arguments": '{"x": 1}'},
                        }
                    ],
                },
            ],
            "tools": [],
        }
        result = example_to_chatml(ex)
        assert "<tool_call>" in result
        assert "my_tool" in result

    def test_example_to_chatml_add_generation_prompt(self) -> None:
        from tool_call_finetune_lab.data.format_chat_template import example_to_chatml

        ex = {
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [],
        }
        result = example_to_chatml(ex, add_generation_prompt=True)
        assert result.endswith("<|im_start|>assistant\n")

    def test_example_to_hf_messages_structure(self) -> None:
        from tool_call_finetune_lab.data.format_chat_template import example_to_hf_messages

        ex = {
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "greet", "arguments": '{"name": "world"}'},
                        }
                    ],
                },
            ],
            "tools": [],
        }
        messages = example_to_hf_messages(ex)
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert "tool_calls" in messages[2]
        assert messages[2]["tool_calls"][0]["function"]["name"] == "greet"

    def test_format_tool_definition_flat_format(self) -> None:
        from tool_call_finetune_lab.data.format_chat_template import format_tool_definition

        flat = {
            "name": "my_fn",
            "description": "Does something",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
        result = format_tool_definition(flat)
        assert result["type"] == "function"
        assert result["function"]["name"] == "my_fn"

    def test_format_tool_definition_already_correct(self) -> None:
        from tool_call_finetune_lab.data.format_chat_template import format_tool_definition

        correct = {
            "type": "function",
            "function": {
                "name": "foo",
                "description": "bar",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
        result = format_tool_definition(correct)
        assert result == correct
