"""QLoRA fine-tuning script using TRL SFTTrainer + PEFT LoRA adapters.

Usage:
    python -m tool_call_finetune_lab.train.lora_trainer
    # or with custom config overrides:
    python -m tool_call_finetune_lab.train.lora_trainer \
        --epochs 5 --lr 1e-4 --rank 32
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    examples: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _make_hf_dataset(examples: List[Dict[str, Any]], tokenizer: Any, max_seq_length: int) -> Any:
    """Convert list of training examples to a HuggingFace Dataset with 'text' column."""
    from datasets import Dataset

    from tool_call_finetune_lab.data.format_chat_template import (
        example_to_hf_messages,
        format_tool_definition,
    )

    texts: List[str] = []
    skipped = 0

    for ex in examples:
        tools = [format_tool_definition(t) for t in ex.get("tools", [])]
        messages = example_to_hf_messages(ex)

        try:
            text = tokenizer.apply_chat_template(
                messages,
                tools=tools if tools else None,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            logger.debug("apply_chat_template failed (%s), using fallback", e)
            from tool_call_finetune_lab.data.format_chat_template import example_to_chatml

            text = example_to_chatml(ex)

        # Pre-check length
        token_len = len(tokenizer.encode(text))
        if token_len > max_seq_length:
            skipped += 1
            continue

        texts.append(text)

    if skipped:
        logger.info("Skipped %d examples exceeding max_seq_length=%d", skipped, max_seq_length)

    logger.info("Dataset prepared: %d examples", len(texts))
    return Dataset.from_dict({"text": texts})


def build_model_and_tokenizer(
    model_config: Any,
    lora_cfg: Any,
) -> tuple:
    """Load base model with 4-bit quantization and apply LoRA adapters."""
    import torch
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info("Loading tokenizer from %s", model_config.base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.base_model,
        trust_remote_code=model_config.trust_remote_code,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model with 4-bit NF4 quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    logger.info(
        "Applying LoRA: rank=%d, alpha=%d, target_modules=%s",
        lora_cfg.rank,
        lora_cfg.alpha,
        lora_cfg.target_modules,
    )
    peft_lora_config = PeftLoraConfig(
        r=lora_cfg.rank,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=lora_cfg.target_modules,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
    )
    model = get_peft_model(model, peft_lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train(
    model_config: Optional[Any] = None,
    lora_cfg: Optional[Any] = None,
    training_cfg: Optional[Any] = None,
    data_cfg: Optional[Any] = None,
) -> str:
    """Run QLoRA fine-tuning. Returns path to saved adapter.

    Args:
        model_config: ModelConfig instance (uses defaults if None)
        lora_cfg: LoraConfig instance (uses defaults if None)
        training_cfg: TrainingConfig instance (uses defaults if None)
        data_cfg: DataConfig instance (uses defaults if None)

    Returns:
        Output directory path where adapter weights are saved.
    """
    from trl import SFTConfig as TrainingArguments
    from trl import SFTTrainer

    from tool_call_finetune_lab.config import (
        DataConfig,
        LoraConfig,
        ModelConfig,
        TrainingConfig,
        get_wandb_key,
    )

    model_config = model_config or ModelConfig()
    lora_cfg = lora_cfg or LoraConfig()
    training_cfg = training_cfg or TrainingConfig()
    data_cfg = data_cfg or DataConfig()

    # W&B setup
    wandb_key = get_wandb_key()
    if wandb_key:
        import wandb

        wandb.login(key=wandb_key)
        os.environ["WANDB_PROJECT"] = "tool-call-finetune-lab"
    else:
        os.environ["WANDB_DISABLED"] = "true"
        training_cfg.report_to = "none"
        logger.warning("WANDB_API_KEY not set — disabling W&B logging")

    # Check data files exist
    train_path = Path(data_cfg.train_file)
    val_path = Path(data_cfg.val_file)
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}. Run 'make data' first.")

    model, tokenizer = build_model_and_tokenizer(model_config, lora_cfg)

    logger.info("Loading training data from %s", train_path)
    train_examples = _load_jsonl(str(train_path))
    train_dataset = _make_hf_dataset(train_examples, tokenizer, model_config.max_seq_length)

    eval_dataset = None
    if val_path.exists():
        logger.info("Loading validation data from %s", val_path)
        val_examples = _load_jsonl(str(val_path))
        eval_dataset = _make_hf_dataset(val_examples, tokenizer, model_config.max_seq_length)

    output_dir = Path(training_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        dataset_text_field="text",
        max_seq_length=model_config.max_seq_length,
        output_dir=str(output_dir),
        num_train_epochs=training_cfg.epochs,
        per_device_train_batch_size=training_cfg.batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation,
        learning_rate=training_cfg.lr,
        lr_scheduler_type=training_cfg.lr_scheduler_type,
        warmup_ratio=training_cfg.warmup_ratio,
        bf16=training_cfg.bf16,
        fp16=training_cfg.fp16,
        logging_steps=training_cfg.logging_steps,
        save_steps=training_cfg.save_steps,
        eval_steps=training_cfg.eval_steps if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_total_limit=training_cfg.save_total_limit,
        load_best_model_at_end=training_cfg.load_best_model_at_end if eval_dataset else False,
        report_to=training_cfg.report_to,
        run_name=training_cfg.run_name,
        dataloader_num_workers=training_cfg.dataloader_num_workers,
        optim=training_cfg.optim,
        max_grad_norm=training_cfg.max_grad_norm,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        packing=False,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving adapter to %s", output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logger.info("Training complete. Adapter saved to: %s", output_dir)
    return str(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for tool-calling")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs/lora-adapter")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from tool_call_finetune_lab.config import DataConfig, LoraConfig, ModelConfig, TrainingConfig

    model_config = ModelConfig(
        base_model=args.base_model,
        max_seq_length=args.max_seq_length,
    )
    lora_cfg = LoraConfig(rank=args.rank, alpha=args.alpha)
    training_cfg = TrainingConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation=args.grad_accum,
        output_dir=args.output_dir,
    )
    data_cfg = DataConfig()

    train(model_config, lora_cfg, training_cfg, data_cfg)


if __name__ == "__main__":
    main()
