"""Centralized configuration dataclasses for all pipeline stages."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Base model selection and sequence parameters."""

    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    max_seq_length: int = 4096
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    trust_remote_code: bool = True

    def __post_init__(self) -> None:
        if self.max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive, got {self.max_seq_length}")
        valid_dtypes = ("float16", "bfloat16", "float32")
        if self.torch_dtype not in valid_dtypes:
            raise ValueError(
                f"Unsupported torch_dtype: {self.torch_dtype}. Must be one of {valid_dtypes}"
            )


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

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total:.4f}"
            )
        Path(self.raw_dir).mkdir(parents=True, exist_ok=True)
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class ServeConfig:
    """vLLM serving parameters."""

    vllm_model_path: str = "outputs/awq-model"
    tensor_parallel: int = 1
    max_model_len: int = 4096
    port: int = 8000
    host: str = "0.0.0.0"
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
    openai_model: str = "gpt-4o-mini"

    def __post_init__(self) -> None:
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)


def get_hf_token() -> str | None:
    """Retrieve HuggingFace token from environment."""
    return os.environ.get("HF_TOKEN")


def get_wandb_key() -> str | None:
    """Retrieve W&B API key from environment."""
    return os.environ.get("WANDB_API_KEY")
