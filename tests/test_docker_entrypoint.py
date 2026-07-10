from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import yaml


def load_compose_yaml(path: str) -> dict[str, Any]:
    return cast(dict[str, Any], yaml.safe_load(Path(path).read_text()))


def test_entrypoint_execs_vllm_as_pid_one_with_exact_argv_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tool_call_finetune_lab.config import QWEN_25_7B_INSTRUCT_PINNED_REVISION
    from tool_call_finetune_lab.serve import docker_entrypoint

    captured: dict[str, Any] = {}
    environ = {
        "MODEL_PATH": "/models/tool",
        "VLLM_HOST": "0.0.0.0",
        "VLLM_ALLOW_PUBLIC_BIND": "1",
        "VLLM_PORT": "9000",
        "TENSOR_PARALLEL": "2",
        "MAX_MODEL_LEN": "8192",
        "GPU_MEMORY_UTILIZATION": "0.75",
        "SERVED_MODEL_NAME": "tool-model",
        "HF_TOKEN": "hf_test",
    }

    def fake_execvpe(file: str, argv: list[str], env: os._Environ[str]) -> None:
        captured["file"] = file
        captured["argv"] = argv
        captured["env"] = env
        raise SystemExit(0)

    monkeypatch.setattr(os, "environ", environ)
    monkeypatch.setattr(os, "execvpe", fake_execvpe)
    monkeypatch.setattr(sys, "argv", ["docker_entrypoint.py", "--disable-log-requests"])

    with pytest.raises(SystemExit):
        docker_entrypoint.main()

    expected_argv = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "/models/tool",
        "--host",
        "0.0.0.0",
        "--port",
        "9000",
        "--tensor-parallel-size",
        "2",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.75",
        "--dtype",
        "float16",
        "--served-model-name",
        "tool-model",
        "--max-num-seqs",
        "256",
        "--revision",
        QWEN_25_7B_INSTRUCT_PINNED_REVISION,
        "--code-revision",
        QWEN_25_7B_INSTRUCT_PINNED_REVISION,
        "--enable-prefix-caching",
        "--tool-call-parser",
        "hermes",
        "--disable-log-requests",
    ]
    assert captured == {
        "file": sys.executable,
        "argv": expected_argv,
        "env": environ,
    }


@pytest.mark.parametrize("value", [None, "", "0", "false", "FALSE"])
def test_entrypoint_false_public_bind_blocks_public_vllm_before_exec(
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
) -> None:
    from tool_call_finetune_lab.serve import docker_entrypoint

    environ = {
        "MODEL_PATH": "/models/tool",
        "VLLM_HOST": "0.0.0.0",
        "VLLM_PORT": "8000",
    }
    if value is not None:
        environ["VLLM_ALLOW_PUBLIC_BIND"] = value

    monkeypatch.setattr(os, "environ", environ)
    monkeypatch.setattr(
        os,
        "execvpe",
        lambda *_args: pytest.fail("entrypoint must validate public bind before exec"),
    )

    with pytest.raises(ValueError, match="allow-public-bind"):
        docker_entrypoint.main()


@pytest.mark.parametrize("value", ["2", "maybe", " false ", " true "])
def test_entrypoint_invalid_public_bind_blocks_before_exec(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    from tool_call_finetune_lab.serve import docker_entrypoint

    monkeypatch.setattr(
        os,
        "environ",
        {
            "MODEL_PATH": "/models/tool",
            "VLLM_HOST": "0.0.0.0",
            "VLLM_ALLOW_PUBLIC_BIND": value,
        },
    )
    monkeypatch.setattr(
        os,
        "execvpe",
        lambda *_args: pytest.fail("entrypoint must parse public bind before exec"),
    )

    with pytest.raises(ValueError, match="VLLM_ALLOW_PUBLIC_BIND"):
        docker_entrypoint.main()


def test_default_compose_host_publication_is_loopback_only() -> None:
    compose = load_compose_yaml("docker-compose.yml")
    ports = compose["services"]["vllm-server"]["ports"]

    assert ports == ["127.0.0.1:${VLLM_PORT:-8000}:8000"]


def test_production_compose_override_documents_explicit_public_bind() -> None:
    override_path = Path("docker-compose.production.yml")
    override = load_compose_yaml(str(override_path))
    ports = override["services"]["vllm-server"]["ports"]
    environment = override["services"]["vllm-server"]["environment"]
    docs = Path("README.md").read_text()

    assert ports == ["${VLLM_PRODUCTION_HOST_BIND:?Set explicit production bind host}:8000:8000"]
    assert environment["VLLM_API_KEY"].startswith("${VLLM_API_KEY:?")
    assert "!override" not in override_path.read_text()
    assert "docker-compose.production.yml" in docs
    assert "auth" in docs
    assert "reverse proxy" in docs
    assert "firewall" in docs


def test_dockerfile_uses_entrypoint_exec_wrapper() -> None:
    dockerfile = Path("Dockerfile").read_text()

    assert (
        'ENTRYPOINT ["python3", "-m", "tool_call_finetune_lab.serve.docker_entrypoint"]'
        in dockerfile
    )
    assert "CMD []" in dockerfile


def test_entrypoint_module_uses_execvpe_static_check() -> None:
    module = importlib.import_module("tool_call_finetune_lab.serve.docker_entrypoint")
    assert module.__file__ is not None
    source = Path(module.__file__).read_text()

    assert "os.execvpe(" in source
    assert "subprocess" not in source


@pytest.mark.parametrize(
    "forwarded",
    [
        ["--host", "0.0.0.0"],
        ["--host=0.0.0.0"],
        ["--ho", "0.0.0.0"],
        ["--trust-remote-code"],
        ["--trust_remote_code"],
        ["--revision", "main"],
        ["--code-revision", "main"],
        ["--config", "/tmp/vllm.yaml"],
    ],
)
def test_entrypoint_blocks_forwarded_managed_options_before_exec(
    monkeypatch: pytest.MonkeyPatch,
    forwarded: list[str],
) -> None:
    from tool_call_finetune_lab.serve import docker_entrypoint

    monkeypatch.setattr(os, "environ", {"MODEL_PATH": "/models/tool"})
    monkeypatch.setattr(sys, "argv", ["docker_entrypoint.py", *forwarded])
    monkeypatch.setattr(
        os,
        "execvpe",
        lambda *_args: pytest.fail("managed forwarded options must be blocked before exec"),
    )

    with pytest.raises(ValueError, match="managed by this launcher"):
        docker_entrypoint.main()


def test_entrypoint_passes_valid_api_key_to_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tool_call_finetune_lab.serve import docker_entrypoint

    captured: dict[str, Any] = {}
    api_key = "test-key-0123456789abcdef"

    def fake_execvpe(file: str, argv: list[str], env: os._Environ[str]) -> None:
        captured["argv"] = argv
        raise SystemExit(0)

    monkeypatch.setattr(
        os,
        "environ",
        {"MODEL_PATH": "/models/tool", "VLLM_API_KEY": api_key},
    )
    monkeypatch.setattr(sys, "argv", ["docker_entrypoint.py"])
    monkeypatch.setattr(os, "execvpe", fake_execvpe)

    with pytest.raises(SystemExit):
        docker_entrypoint.main()

    argv = captured["argv"]
    assert argv[argv.index("--api-key") + 1] == api_key


@pytest.mark.parametrize("api_key", ["", "short", "contains whitespace"])
def test_build_command_rejects_invalid_api_key(api_key: str) -> None:
    from tool_call_finetune_lab.serve.vllm_launcher import build_vllm_command

    if not api_key:
        assert "--api-key" not in build_vllm_command("/model", api_key=None)
        return
    with pytest.raises(ValueError, match="VLLM_API_KEY"):
        build_vllm_command("/model", api_key=api_key)
