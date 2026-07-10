"""Centralized configuration dataclasses for all pipeline stages."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BFCL_PINNED_REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
QWEN_25_7B_INSTRUCT_PINNED_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
GLAIVE_PINNED_REVISION = "e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac"
GLAIVE_DATA_FILENAME = "glaive-function-calling-v2.json"
GLAIVE_DATA_SHA256 = "e9b5d671812b5ca2fbd7b625a37d5c99a19576c37252cdc806defe256aea6dad"
DEFAULT_BFCL_CHECKSUMS = {
    "BFCL_v4_simple_python.json": "82dd63ba502eb2520c6b5d1d9a5c4b590e03ff261565175561f6228a367d1991",
    "possible_answer/BFCL_v4_simple_python.json": "90cd5bc653690ee8e459b5b3f3fc9458606f7f3fcbf795bb51b7dc581f8c86dc",
    "BFCL_v4_multiple.json": "aef168155ebd74b7ac2401198b201343bc7d16d7a3d7e0d4e6d8ee82c6969b2a",
    "possible_answer/BFCL_v4_multiple.json": "244e00ce9395df948bcafc7bee64e8f9c87ef70887587d83cae45b13699f3047",
    "BFCL_v4_parallel.json": "19f51a82eff42e5d62541aa500115a056eb78f437c2ba1f10415fd7c8e5dda84",
    "possible_answer/BFCL_v4_parallel.json": "8a6aa19c1adddc6a5a2f7e40f9dbf30cc7e95815e7b830c90589ab318229e0f0",
    "BFCL_v4_parallel_multiple.json": "8863ea8433239f55c5f016154cf0830853c89f693c6ea270396a2fa121960579",
    "possible_answer/BFCL_v4_parallel_multiple.json": "5ebf24f458c1f16300c05505d83d6f0a1b68b79be273a033febd0d4f840507e3",
    "BFCL_v4_simple_java.json": "13d2303a125b08754f0e41995b9273b5005fa8ed8ebfaa24ef53b4d83c4b5c6e",
    "possible_answer/BFCL_v4_simple_java.json": "78f25616084044fa05bbfcee68e03f6ececb222bdd5cb3b7783a675fb3366e35",
    "BFCL_v4_simple_javascript.json": "329e67fedf79a6243d93dbda4b388d12bd2d31f1f2163d92cb6ef676d1764f44",
    "possible_answer/BFCL_v4_simple_javascript.json": "e2f9f2e51d88e0c8056ffbf1a3dd3d02eb032532d2b5d98c9cc9003385bdd56b",
    "BFCL_v4_live_simple.json": "1af2ac87dca47556db7b7e37e51e28b459a38b594e3c7b3c792b4903598ca0c4",
    "possible_answer/BFCL_v4_live_simple.json": "fec9cfa9744a936f9126981e85a2023da1e63e273eafebc81923a1162fad70ce",
    "BFCL_v4_live_multiple.json": "fd8ccfad4d911420d0e3341dbe2fff77d1d341da934248b9bb2bda24ab3a10c8",
    "possible_answer/BFCL_v4_live_multiple.json": "97e90d59c5bd76c55a2920ce93e5566e9046307d3f558578f085f9d3a56c3084",
    "BFCL_v4_live_parallel.json": "6c26e9fdc3350cf596e6d1ea9c179cbff834761bccf562f4141ed29a839ca421",
    "possible_answer/BFCL_v4_live_parallel.json": "8a9f189ff0e832ebbbbdade1fd95a7dbcc67406e9177df3f0aad76f59ab00350",
    "BFCL_v4_live_parallel_multiple.json": "21d4b9319c1faac431e22757b367ea28917fe467364c3a4b17f16ec06d4f6e79",
    "possible_answer/BFCL_v4_live_parallel_multiple.json": "f5b5f360556c5feb51db46fb9f56ee4b304f4b45b161599bbb14161c98a2873f",
}


def _is_immutable_revision(revision: str | None) -> bool:
    """Return True for full Git commit hashes."""
    if revision is None:
        return False
    return len(revision) == 40 and all(c in "0123456789abcdefABCDEF" for c in revision)


def validate_immutable_revision(revision: str, field_name: str) -> None:
    """Reject mutable branches and tags for reproducible remote artifacts."""
    if not _is_immutable_revision(revision):
        raise ValueError(f"{field_name} must be an immutable 40-character Git commit hash.")


def validate_remote_code_policy(trust_remote_code: bool, code_revision: str) -> None:
    """Require pinned code when remote model code execution is explicitly enabled."""
    if trust_remote_code:
        validate_immutable_revision(code_revision, "code_revision")


@dataclass
class ModelConfig:
    """Base model selection and sequence parameters."""

    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    max_seq_length: int = 4096
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    trust_remote_code: bool = False
    model_revision: str = QWEN_25_7B_INSTRUCT_PINNED_REVISION
    code_revision: str = QWEN_25_7B_INSTRUCT_PINNED_REVISION

    def __post_init__(self) -> None:
        if self.max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive, got {self.max_seq_length}")
        valid_dtypes = ("float16", "bfloat16", "float32")
        if self.torch_dtype not in valid_dtypes:
            raise ValueError(
                f"Unsupported torch_dtype: {self.torch_dtype}. Must be one of {valid_dtypes}"
            )
        validate_immutable_revision(self.model_revision, "model_revision")
        validate_immutable_revision(self.code_revision, "code_revision")
        validate_remote_code_policy(self.trust_remote_code, self.code_revision)


@dataclass
class LoraConfig:
    """LoRA adapter hyperparameters."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.rank}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {self.alpha}")


@dataclass
class TrainingConfig:
    """SFTTrainer / training loop hyperparameters."""

    output_dir: str = "outputs/lora-adapter"
    epochs: int = 3
    lr: float = 2e-4
    batch_size: int = 4
    gradient_accumulation: int = 4
    warmup_ratio: float = 0.1
    bf16: bool = True
    fp16: bool = False
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    report_to: str = "wandb"
    run_name: str = "qwen2.5-7b-tool-call-lora"
    dataloader_num_workers: int = 4
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.bf16 and self.fp16:
            raise ValueError("Cannot enable both bf16 and fp16 simultaneously.")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.lr <= 0:
            raise ValueError(f"learning rate must be positive, got {self.lr}")

    @property
    def effective_batch_size(self) -> int:
        """Return the effective batch size (per_device * gradient_accumulation)."""
        return self.batch_size * self.gradient_accumulation


@dataclass
class DataConfig:
    """Dataset sources and split ratios."""

    bfcl_repo: str = "gorilla-llm/berkeley-function-call-leaderboard"
    glaive_repo: str = "glaiveai/glaive-function-calling-v2"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    bfcl_output: str = "data/raw/bfcl.jsonl"
    glaive_output: str = "data/raw/glaive.jsonl"
    train_file: str = "data/processed/train.jsonl"
    val_file: str = "data/processed/val.jsonl"
    test_file: str = "data/processed/test.jsonl"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    max_samples_bfcl: int | None = None
    max_samples_glaive: int | None = None
    dataset_revision: str = GLAIVE_PINNED_REVISION
    glaive_filename: str = GLAIVE_DATA_FILENAME
    glaive_sha256: str = GLAIVE_DATA_SHA256
    bfcl_revision: str = BFCL_PINNED_REVISION
    bfcl_checksums: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_BFCL_CHECKSUMS))
    allow_synthetic_fixtures: bool = False

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total:.4f}"
            )
        if not _is_immutable_revision(self.bfcl_revision):
            raise ValueError("bfcl_revision must be an immutable 40-character Git commit hash.")
        validate_immutable_revision(self.dataset_revision, "dataset_revision")
        if len(self.glaive_sha256) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in self.glaive_sha256
        ):
            raise ValueError("glaive_sha256 must be a 64-character SHA-256 digest.")
        if not self.glaive_filename or Path(self.glaive_filename).name != self.glaive_filename:
            raise ValueError("glaive_filename must be a plain filename without path components.")
        Path(self.raw_dir).mkdir(parents=True, exist_ok=True)
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class ServeConfig:
    """vLLM serving parameters."""

    vllm_model_path: str = "outputs/awq-model"
    tensor_parallel: int = 1
    max_model_len: int = 4096
    port: int = 8000
    host: str = "127.0.0.1"
    allow_public_bind: bool = False
    gpu_memory_utilization: float = 0.90
    quantization: str = "awq"
    dtype: str = "float16"
    served_model_name: str = "qwen2.5-7b-tool-call"
    max_num_seqs: int = 256

    def __post_init__(self) -> None:
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1], got {self.gpu_memory_utilization}"
            )
        if self.port <= 0 or self.port > 65535:
            raise ValueError(f"Invalid port: {self.port}")
        if self.host not in {"127.0.0.1", "localhost", "::1"} and not self.allow_public_bind:
            raise ValueError("Non-loopback vLLM host requires allow_public_bind=True.")


@dataclass
class EvalConfig:
    """Evaluation parameters."""

    results_dir: str = "results"
    bfcl_results_file: str = "results/bfcl_results.json"
    compare_output_file: str = "results/comparison.md"
    vllm_base_url: str = "http://localhost:8000/v1"
    model_name: str = "qwen2.5-7b-tool-call"
    max_tokens: int = 512
    temperature: float = 0.0
    timeout_seconds: int = 60
    openai_model: str = "qwen/qwen3-coder"

    def __post_init__(self) -> None:
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)


def get_hf_token() -> str | None:
    """Retrieve HuggingFace token from environment."""
    return os.environ.get("HF_TOKEN")


def get_wandb_key() -> str | None:
    """Retrieve W&B API key from environment."""
    return os.environ.get("WANDB_API_KEY")
