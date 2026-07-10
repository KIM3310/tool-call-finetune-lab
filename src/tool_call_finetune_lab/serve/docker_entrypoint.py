"""Docker entrypoint for vLLM serving with strict environment parsing."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping

from tool_call_finetune_lab.serve.vllm_launcher import _detect_quantization, build_vllm_command

TRUE_PUBLIC_BIND_VALUES = {"1", "true", "yes", "on"}
FALSE_PUBLIC_BIND_VALUES = {"", "0", "false"}
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def parse_public_bind_env(value: str | None) -> bool:
    """Parse VLLM_ALLOW_PUBLIC_BIND without shell-style truthiness."""
    if value is None:
        return False
    normalized = value.lower()
    if normalized in TRUE_PUBLIC_BIND_VALUES:
        return True
    if normalized in FALSE_PUBLIC_BIND_VALUES:
        return False
    raise ValueError(
        "VLLM_ALLOW_PUBLIC_BIND must be empty, 0, false, or one of: "
        f"{', '.join(sorted(TRUE_PUBLIC_BIND_VALUES))}"
    )


def _env_int(environ: Mapping[str, str], name: str, default: int) -> int:
    return int(environ.get(name, str(default)))


def _env_float(environ: Mapping[str, str], name: str, default: float) -> float:
    return float(environ.get(name, str(default)))


def main() -> None:
    environ = os.environ
    model_path = environ.get("MODEL_PATH", "/model")
    host = environ.get("VLLM_HOST", "127.0.0.1")
    allow_public_bind = parse_public_bind_env(environ.get("VLLM_ALLOW_PUBLIC_BIND"))
    if host not in LOOPBACK_HOSTS and not allow_public_bind:
        raise ValueError("Non-loopback vLLM host requires --allow-public-bind.")

    argv = build_vllm_command(
        model_path=model_path,
        host=host,
        port=_env_int(environ, "VLLM_PORT", 8000),
        tensor_parallel=_env_int(environ, "TENSOR_PARALLEL", 1),
        max_model_len=_env_int(environ, "MAX_MODEL_LEN", 4096),
        gpu_memory_utilization=_env_float(environ, "GPU_MEMORY_UTILIZATION", 0.90),
        quantization=_detect_quantization(model_path),
        served_model_name=environ.get("SERVED_MODEL_NAME", "qwen2.5-7b-tool-call"),
        api_key=environ.get("VLLM_API_KEY") or None,
        extra_args=sys.argv[1:],
    )
    os.execvpe(argv[0], argv, environ)  # nosec B606


if __name__ == "__main__":
    main()
