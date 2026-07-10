"""vLLM launcher for the AWQ-quantized tool-call model.

Launches vLLM with the OpenAI-compatible API server. Can be run directly
or via Docker. Supports both AWQ-quantized and full-precision models.

Usage:
    python -m tool_call_finetune_lab.serve.vllm_launcher
    python -m tool_call_finetune_lab.serve.vllm_launcher \
        --model outputs/awq-model \
        --port 8000 \
        --tensor-parallel 1
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess  # nosec B404
import sys
from pathlib import Path
from typing import List, Optional

from tool_call_finetune_lab.config import (
    QWEN_25_7B_INSTRUCT_PINNED_REVISION,
    validate_immutable_revision,
    validate_remote_code_policy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MANAGED_FORWARD_OPTIONS = frozenset(
    {
        "--api-key",
        "--code-revision",
        "--config",
        "--dtype",
        "--enable-prefix-caching",
        "--gpu-memory-utilization",
        "--host",
        "--max-model-len",
        "--max-num-seqs",
        "--model",
        "--port",
        "--quantization",
        "--revision",
        "--served-model-name",
        "--tensor-parallel-size",
        "--tool-call-parser",
        "--trust-remote-code",
    }
)


def validate_forwarded_args(extra_args: List[str]) -> None:
    """Prevent appended vLLM options from overriding validated settings."""
    for argument in extra_args:
        if not argument.startswith("--"):
            continue
        option = argument.split("=", 1)[0].replace("_", "-")
        if option in MANAGED_FORWARD_OPTIONS or any(
            managed.startswith(option) for managed in MANAGED_FORWARD_OPTIONS
        ):
            raise ValueError(f"Forwarded vLLM option is managed by this launcher: {option}")


def validate_api_key(api_key: str | None) -> None:
    """Reject empty or trivially weak API keys when authentication is enabled."""
    if api_key is None:
        return
    if len(api_key) < 16 or any(character.isspace() for character in api_key):
        raise ValueError("VLLM_API_KEY must be at least 16 non-whitespace characters.")


def redact_command(command: List[str]) -> List[str]:
    """Return a display-safe command with API credentials removed."""
    redacted = list(command)
    for index, argument in enumerate(redacted):
        if argument == "--api-key" and index + 1 < len(redacted):
            redacted[index + 1] = "[REDACTED]"
        elif argument.startswith("--api-key="):
            redacted[index] = "--api-key=[REDACTED]"
    return redacted


def build_vllm_command(
    model_path: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    tensor_parallel: int = 1,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.90,
    quantization: Optional[str] = "awq",
    dtype: str = "float16",
    served_model_name: str = "qwen2.5-7b-tool-call",
    max_num_seqs: int = 256,
    enable_prefix_caching: bool = True,
    trust_remote_code: bool = False,
    model_revision: str = QWEN_25_7B_INSTRUCT_PINNED_REVISION,
    code_revision: str = QWEN_25_7B_INSTRUCT_PINNED_REVISION,
    api_key: str | None = None,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """Build the vLLM server command list.

    Args:
        model_path: Path to the model (AWQ-quantized or full-precision).
        host: Bind host address.
        port: Port to listen on.
        tensor_parallel: Number of GPUs for tensor parallelism.
        max_model_len: Maximum sequence length.
        gpu_memory_utilization: Fraction of GPU memory to use.
        quantization: Quantization method ('awq', 'gptq', or None).
        dtype: Compute dtype ('float16', 'bfloat16', 'auto').
        served_model_name: Model name exposed via the API.
        max_num_seqs: Maximum number of concurrent sequences.
        enable_prefix_caching: Enable KV cache prefix sharing.
        trust_remote_code: Pass --trust-remote-code to vLLM.
        api_key: Optional bearer token for the OpenAI-compatible API.
        extra_args: Additional CLI arguments to append.

    Returns:
        Command as a list of strings, ready for subprocess.
    """
    validate_immutable_revision(model_revision, "model_revision")
    validate_immutable_revision(code_revision, "code_revision")
    validate_remote_code_policy(trust_remote_code, code_revision)
    validate_api_key(api_key)
    if extra_args:
        validate_forwarded_args(extra_args)

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--dtype",
        dtype,
        "--served-model-name",
        served_model_name,
        "--max-num-seqs",
        str(max_num_seqs),
        "--revision",
        model_revision,
        "--code-revision",
        code_revision,
    ]

    if quantization:
        cmd.extend(["--quantization", quantization])

    if enable_prefix_caching:
        cmd.append("--enable-prefix-caching")

    if trust_remote_code:
        cmd.append("--trust-remote-code")

    if api_key:
        cmd.extend(["--api-key", api_key])

    # Tool-call specific: enable structured outputs / tool call parser
    cmd.extend(["--tool-call-parser", "hermes"])

    if extra_args:
        cmd.extend(extra_args)

    return cmd


def _detect_quantization(model_path: str) -> Optional[str]:
    """Auto-detect if a model is AWQ-quantized by checking for quant_config.json."""
    quant_config = Path(model_path) / "quant_config.json"
    awq_meta = Path(model_path) / "awq_metadata.json"

    if awq_meta.exists() or quant_config.exists():
        try:
            import json

            cfg_path = quant_config if quant_config.exists() else awq_meta
            with open(cfg_path) as f:
                cfg = json.load(f)
            if cfg.get("quant_type", "").lower() == "awq" or "w_bit" in cfg.get("quant_config", {}):
                return "awq"
        except Exception:
            return "awq"

    return None


def launch(
    model_path: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    tensor_parallel: int = 1,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.90,
    quantization: Optional[str] = None,
    dtype: str = "float16",
    served_model_name: str = "qwen2.5-7b-tool-call",
    max_num_seqs: int = 256,
    allow_public_bind: bool = False,
    trust_remote_code: bool = False,
    model_revision: str = QWEN_25_7B_INSTRUCT_PINNED_REVISION,
    code_revision: str = QWEN_25_7B_INSTRUCT_PINNED_REVISION,
    api_key: str | None = None,
    dry_run: bool = False,
) -> Optional[subprocess.Popen]:  # type: ignore[type-arg]
    """Launch the vLLM server process.

    Args:
        dry_run: If True, print the command but don't execute it.

    Returns:
        The Popen process handle (or None if dry_run).
    """
    if host not in {"127.0.0.1", "localhost", "::1"} and not allow_public_bind:
        raise ValueError("Non-loopback vLLM host requires --allow-public-bind.")

    if quantization is None:
        quantization = _detect_quantization(model_path)
        if quantization:
            logger.info("Auto-detected quantization: %s", quantization)
        else:
            logger.info("No quantization detected — loading full-precision model")

    cmd = build_vllm_command(
        model_path=model_path,
        host=host,
        port=port,
        tensor_parallel=tensor_parallel,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        quantization=quantization,
        dtype=dtype,
        served_model_name=served_model_name,
        max_num_seqs=max_num_seqs,
        trust_remote_code=trust_remote_code,
        model_revision=model_revision,
        code_revision=code_revision,
        api_key=api_key,
    )

    display_command = redact_command(cmd)
    logger.info("vLLM command: %s", " ".join(display_command))
    logger.info("Model will be served at http://%s:%d/v1", host, port)

    if dry_run:
        print("DRY RUN — would execute:")
        print(" ".join(display_command))
        return None

    # Set HF token if available
    env = os.environ.copy()
    hf_token = env.get("HF_TOKEN")
    if hf_token:
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token

    # Intentional vLLM launch with an argv list and no shell.
    process = subprocess.Popen(cmd, env=env)  # nosec B603
    logger.info("vLLM server started (PID %d). Press Ctrl+C to stop.", process.pid)

    try:
        process.wait()
    except KeyboardInterrupt:
        logger.info("Shutting down vLLM server...")
        process.terminate()
        process.wait(timeout=10)

    return process


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch vLLM server for tool-call model")
    parser.add_argument("--model", default="outputs/awq-model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--quantization",
        default=None,
        choices=["awq", "gptq", "none"],
        help="Quantization type (auto-detected if not set)",
    )
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "auto"])
    parser.add_argument("--served-model-name", default="qwen2.5-7b-tool-call")
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument(
        "--allow-public-bind",
        action="store_true",
        help=(
            "Allow non-loopback hosts such as 0.0.0.0. Use only behind auth, "
            "firewall, or reverse proxy controls."
        ),
    )
    parser.add_argument("--model-revision", default=QWEN_25_7B_INSTRUCT_PINNED_REVISION)
    parser.add_argument("--code-revision", default=QWEN_25_7B_INSTRUCT_PINNED_REVISION)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Explicitly allow audited remote model code. Use with pinned --code-revision.",
    )
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quantization = None if args.quantization == "none" else args.quantization
    launch(
        model_path=args.model,
        host=args.host,
        port=args.port,
        tensor_parallel=args.tensor_parallel,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=quantization,
        dtype=args.dtype,
        served_model_name=args.served_model_name,
        max_num_seqs=args.max_num_seqs,
        allow_public_bind=args.allow_public_bind,
        trust_remote_code=args.trust_remote_code,
        model_revision=args.model_revision,
        code_revision=args.code_revision,
        api_key=args.api_key,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
