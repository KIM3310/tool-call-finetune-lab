"""Merge LoRA adapter weights back into the base model.

Produces a single, self-contained model directory that can be loaded
with AutoModelForCausalLM.from_pretrained() without PEFT installed.

Usage:
    python -m tool_call_finetune_lab.train.merge_adapter \
        --adapter-path outputs/lora-adapter \
        --output-path outputs/merged-model
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def merge_and_save(
    base_model: str,
    adapter_path: str,
    output_path: str,
    torch_dtype: str = "bfloat16",
    safe_serialization: bool = True,
) -> str:
    """Load base model + LoRA adapter, merge weights, and save merged model.

    Args:
        base_model: HuggingFace model ID or local path for the base model.
        adapter_path: Path to the directory containing LoRA adapter weights.
        output_path: Destination directory for the merged model.
        torch_dtype: Torch dtype string for loading the base model.
        safe_serialization: Whether to save in safetensors format.

    Returns:
        The output path where the merged model was saved.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading base model: %s", base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info("Loading LoRA adapter from: %s", adapter_path)
    model = PeftModel.from_pretrained(base, adapter_path)

    logger.info("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    logger.info("Saving merged model to: %s", output_dir)
    model.save_pretrained(
        str(output_dir),
        safe_serialization=safe_serialization,
    )

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_dir))

    # Write a small metadata file
    import json

    metadata = {
        "base_model": base_model,
        "adapter_path": adapter_path,
        "merge_dtype": torch_dtype,
    }
    with open(output_dir / "merge_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Merge complete. Model saved to: %s", output_dir)
    return str(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model ID or path",
    )
    parser.add_argument(
        "--adapter-path",
        default="outputs/lora-adapter",
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/merged-model",
        help="Output directory for merged model",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for base model loading",
    )
    parser.add_argument(
        "--no-safe-serialization",
        action="store_true",
        help="Use .bin instead of .safetensors format",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_and_save(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        torch_dtype=args.dtype,
        safe_serialization=not args.no_safe_serialization,
    )


if __name__ == "__main__":
    main()
