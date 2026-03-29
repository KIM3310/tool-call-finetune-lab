"""Tests for configuration dataclasses and validation logic."""

from __future__ import annotations

import pytest


class TestModelConfig:
    def test_defaults(self) -> None:
        from tool_call_finetune_lab.config import ModelConfig

        cfg = ModelConfig()
        assert cfg.base_model == "Qwen/Qwen2.5-7B-Instruct"
        assert cfg.max_seq_length == 4096
        assert cfg.torch_dtype == "bfloat16"

    def test_invalid_max_seq_length(self) -> None:
        from tool_call_finetune_lab.config import ModelConfig

        with pytest.raises(ValueError, match="max_seq_length must be positive"):
            ModelConfig(max_seq_length=0)

    def test_invalid_max_seq_length_negative(self) -> None:
        from tool_call_finetune_lab.config import ModelConfig

        with pytest.raises(ValueError, match="max_seq_length must be positive"):
            ModelConfig(max_seq_length=-1)

    def test_invalid_torch_dtype(self) -> None:
        from tool_call_finetune_lab.config import ModelConfig

        with pytest.raises(ValueError, match="Unsupported torch_dtype"):
            ModelConfig(torch_dtype="int8")

    def test_valid_torch_dtypes(self) -> None:
        from tool_call_finetune_lab.config import ModelConfig

        for dtype in ("float16", "bfloat16", "float32"):
            cfg = ModelConfig(torch_dtype=dtype)
            assert cfg.torch_dtype == dtype

    def test_custom_base_model(self) -> None:
        from tool_call_finetune_lab.config import ModelConfig

        cfg = ModelConfig(base_model="meta-llama/Llama-3-8B")
        assert cfg.base_model == "meta-llama/Llama-3-8B"


class TestLoraConfig:
    def test_defaults(self) -> None:
        from tool_call_finetune_lab.config import LoraConfig

        cfg = LoraConfig()
        assert cfg.rank == 16
        assert cfg.alpha == 32
        assert cfg.dropout == 0.05
        assert "q_proj" in cfg.target_modules

    def test_invalid_rank(self) -> None:
        from tool_call_finetune_lab.config import LoraConfig

        with pytest.raises(ValueError, match="LoRA rank must be positive"):
            LoraConfig(rank=0)

    def test_invalid_alpha(self) -> None:
        from tool_call_finetune_lab.config import LoraConfig

        with pytest.raises(ValueError, match="LoRA alpha must be positive"):
            LoraConfig(alpha=-1)

    def test_invalid_dropout_too_high(self) -> None:
        from tool_call_finetune_lab.config import LoraConfig

        with pytest.raises(ValueError, match="dropout must be in"):
            LoraConfig(dropout=1.0)

    def test_invalid_dropout_negative(self) -> None:
        from tool_call_finetune_lab.config import LoraConfig

        with pytest.raises(ValueError, match="dropout must be in"):
            LoraConfig(dropout=-0.1)

    def test_valid_zero_dropout(self) -> None:
        from tool_call_finetune_lab.config import LoraConfig

        cfg = LoraConfig(dropout=0.0)
        assert cfg.dropout == 0.0

    def test_custom_target_modules(self) -> None:
        from tool_call_finetune_lab.config import LoraConfig

        cfg = LoraConfig(target_modules=["q_proj", "v_proj"])
        assert cfg.target_modules == ["q_proj", "v_proj"]

    def test_alpha_equals_twice_rank_is_common(self) -> None:
        from tool_call_finetune_lab.config import LoraConfig

        cfg = LoraConfig(rank=32, alpha=64)
        assert cfg.alpha / cfg.rank == 2.0


class TestTrainingConfig:
    def test_defaults(self) -> None:
        from tool_call_finetune_lab.config import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.epochs == 3
        assert cfg.lr == 2e-4
        assert cfg.batch_size == 4
        assert cfg.bf16 is True
        assert cfg.fp16 is False

    def test_bf16_and_fp16_mutual_exclusion(self) -> None:
        from tool_call_finetune_lab.config import TrainingConfig

        with pytest.raises(ValueError, match="Cannot enable both bf16 and fp16"):
            TrainingConfig(bf16=True, fp16=True)

    def test_invalid_epochs(self) -> None:
        from tool_call_finetune_lab.config import TrainingConfig

        with pytest.raises(ValueError, match="epochs must be positive"):
            TrainingConfig(epochs=0)

    def test_invalid_lr(self) -> None:
        from tool_call_finetune_lab.config import TrainingConfig

        with pytest.raises(ValueError, match="learning rate must be positive"):
            TrainingConfig(lr=0.0)

    def test_negative_lr(self) -> None:
        from tool_call_finetune_lab.config import TrainingConfig

        with pytest.raises(ValueError, match="learning rate must be positive"):
            TrainingConfig(lr=-1e-4)

    def test_custom_output_dir(self) -> None:
        from tool_call_finetune_lab.config import TrainingConfig

        cfg = TrainingConfig(output_dir="/tmp/test-model")
        assert cfg.output_dir == "/tmp/test-model"

    def test_effective_batch_size(self) -> None:
        from tool_call_finetune_lab.config import TrainingConfig

        cfg = TrainingConfig(batch_size=4, gradient_accumulation=4)
        effective = cfg.batch_size * cfg.gradient_accumulation
        assert effective == 16


class TestDataConfig:
    def test_defaults(self) -> None:
        from tool_call_finetune_lab.config import DataConfig

        cfg = DataConfig()
        assert cfg.train_ratio == 0.8
        assert cfg.val_ratio == 0.1
        assert cfg.test_ratio == 0.1
        assert cfg.seed == 42

    def test_ratios_sum_to_one(self) -> None:
        from tool_call_finetune_lab.config import DataConfig

        cfg = DataConfig()
        total = cfg.train_ratio + cfg.val_ratio + cfg.test_ratio
        assert abs(total - 1.0) < 1e-6

    def test_invalid_ratios(self) -> None:
        from tool_call_finetune_lab.config import DataConfig

        with pytest.raises(ValueError, match=r"must equal 1\.0"):
            DataConfig(train_ratio=0.7, val_ratio=0.1, test_ratio=0.1)

    def test_custom_ratios_valid(self) -> None:
        from tool_call_finetune_lab.config import DataConfig

        cfg = DataConfig(train_ratio=0.9, val_ratio=0.05, test_ratio=0.05)
        assert abs(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio - 1.0) < 1e-6

    def test_max_samples_default_none(self) -> None:
        from tool_call_finetune_lab.config import DataConfig

        cfg = DataConfig()
        assert cfg.max_samples_bfcl is None
        assert cfg.max_samples_glaive is None


class TestServeConfig:
    def test_defaults(self) -> None:
        from tool_call_finetune_lab.config import ServeConfig

        cfg = ServeConfig()
        assert cfg.port == 8000
        assert cfg.tensor_parallel == 1
        assert cfg.max_model_len == 4096
        assert cfg.quantization == "awq"

    def test_invalid_gpu_memory_utilization_zero(self) -> None:
        from tool_call_finetune_lab.config import ServeConfig

        with pytest.raises(ValueError, match="gpu_memory_utilization must be in"):
            ServeConfig(gpu_memory_utilization=0.0)

    def test_invalid_gpu_memory_utilization_over_one(self) -> None:
        from tool_call_finetune_lab.config import ServeConfig

        with pytest.raises(ValueError, match="gpu_memory_utilization must be in"):
            ServeConfig(gpu_memory_utilization=1.1)

    def test_invalid_port(self) -> None:
        from tool_call_finetune_lab.config import ServeConfig

        with pytest.raises(ValueError, match="Invalid port"):
            ServeConfig(port=0)

    def test_invalid_port_too_high(self) -> None:
        from tool_call_finetune_lab.config import ServeConfig

        with pytest.raises(ValueError, match="Invalid port"):
            ServeConfig(port=99999)

    def test_valid_config(self) -> None:
        from tool_call_finetune_lab.config import ServeConfig

        cfg = ServeConfig(
            vllm_model_path="/tmp/model",
            port=9000,
            gpu_memory_utilization=0.85,
        )
        assert cfg.port == 9000
        assert cfg.gpu_memory_utilization == 0.85


class TestEvalConfig:
    def test_defaults(self) -> None:
        from tool_call_finetune_lab.config import EvalConfig

        cfg = EvalConfig()
        assert "localhost:8000" in cfg.vllm_base_url
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 512
        assert cfg.timeout_seconds == 60

    def test_creates_results_dir(self, tmp_path: object) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            from tool_call_finetune_lab.config import EvalConfig

            cfg = EvalConfig(results_dir=f"{d}/results")
            import os

            assert os.path.isdir(cfg.results_dir)


class TestEnvHelpers:
    def test_get_hf_token_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tool_call_finetune_lab.config import get_hf_token

        monkeypatch.delenv("HF_TOKEN", raising=False)
        assert get_hf_token() is None

    def test_get_hf_token_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tool_call_finetune_lab.config import get_hf_token

        monkeypatch.setenv("HF_TOKEN", "test-token-123")
        assert get_hf_token() == "test-token-123"

    def test_get_wandb_key_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tool_call_finetune_lab.config import get_wandb_key

        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        assert get_wandb_key() is None

    def test_get_wandb_key_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tool_call_finetune_lab.config import get_wandb_key

        monkeypatch.setenv("WANDB_API_KEY", "wandb-abc-xyz")
        assert get_wandb_key() == "wandb-abc-xyz"
