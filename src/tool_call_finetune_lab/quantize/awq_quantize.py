"""AWQ INT4 quantization using the autoawq library.

Quantizes the merged (full-weight) model to 4-bit AWQ for efficient vLLM serving.

Usage:
    python -m tool_call_finetune_lab.quantize.awq_quantize \
        --model-path outputs/merged-model \
        --output-path outputs/awq-model \
        --calib-data data/processed/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# AWQ quantization defaults
AWQ_DEFAULTS: Dict[str, Any] = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}


def load_calibration_data(
    calib_file: str,
    tokenizer: Any,
    n_samples: int = 512,
    max_seq_len: int = 512,
    seed: int = 42,
) -> List[str]:
    """Load and prepare calibration data for AWQ quantization.

    AWQ needs a representative sample of the target distribution to
    compute activation-aware weight clipping thresholds.

    Args:
        calib_file: Path to JSONL file of training examples.
        tokenizer: HuggingFace tokenizer for the model.
        n_samples: Number of calibration samples to use.
        max_seq_len: Maximum token length per sample.
        seed: Random seed for sampling.

    Returns:
        List of raw text strings for calibration.
    """
    from tool_call_finetune_lab.data.format_chat_template import (
        example_to_hf_messages,
        format_tool_definition,
    )

    examples: List[Dict[str, Any]] = []
    path = Path(calib_file)

    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
    else:
        logger.warning("Calibration file not found: %s — using synthetic data", calib_file)

    if not examples:
        # Synthetic fallback
        return [
            "What is the weather in Tokyo?",
            "Search for Python tutorials.",
            "Book a flight from NYC to LA.",
            "What movies are playing tonight?",
            "Calculate 15% tip on $47.50.",
        ] * max(1, n_samples // 5)

    rng = random.Random(seed)
    rng.shuffle(examples)
    selected = examples[:n_samples]

    texts: List[str] = []
    for ex in selected:
        tools = [format_tool_definition(t) for t in ex.get("tools", [])]
        messages = example_to_hf_messages(ex)

        try:
            text = tokenizer.apply_chat_template(
                messages,
                tools=tools if tools else None,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            # Fallback: just use user messages concatenated
            text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")

        # Truncate to max_seq_len tokens
        token_ids = tokenizer.encode(text)
        if len(token_ids) > max_seq_len:
            token_ids = token_ids[:max_seq_len]
            text = tokenizer.decode(token_ids, skip_special_tokens=False)

        texts.append(text)

    logger.info("Prepared %d calibration samples", len(texts))
    return texts


def quantize_awq(
    model_path: str,
    output_path: str,
    calib_data_file: str = "data/processed/train.jsonl",
    quant_config: Optional[Dict[str, Any]] = None,
    n_calib_samples: int = 512,
    calib_seq_len: int = 512,
) -> str:
    """Run AWQ quantization on the merged model.

    Args:
        model_path: Path to the merged full-precision model.
        output_path: Where to save the AWQ-quantized model.
        calib_data_file: JSONL file for calibration data.
        quant_config: AWQ quantization config dict (uses defaults if None).
        n_calib_samples: Number of calibration samples.
        calib_seq_len: Max sequence length for calibration samples.

    Returns:
        Path to the saved AWQ model.
    """
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    if quant_config is None:
        quant_config = AWQ_DEFAULTS.copy()

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model for AWQ quantization from: %s", model_path)
    logger.info("AWQ config: %s", quant_config)

    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        safetensors=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    logger.info("Loading calibration data...")
    calib_texts = load_calibration_data(
        calib_data_file,
        tokenizer,
        n_samples=n_calib_samples,
        max_seq_len=calib_seq_len,
    )

    logger.info("Starting AWQ quantization (this will take 15-60 minutes on GPU)...")
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_texts)

    logger.info("Saving AWQ quantized model to: %s", output_dir)
    model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Write quantization config metadata
    meta = {
        "base_model_path": model_path,
        "quant_config": quant_config,
        "n_calib_samples": n_calib_samples,
        "calib_seq_len": calib_seq_len,
    }
    with open(output_dir / "awq_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("AWQ quantization complete. Model saved to: %s", output_dir)
    return str(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AWQ INT4 quantization")
    parser.add_argument("--model-path", default="outputs/merged-model")
    parser.add_argument("--output-path", default="outputs/awq-model")
    parser.add_argument("--calib-data", default="data/processed/train.jsonl")
    parser.add_argument("--n-calib-samples", type=int, default=512)
    parser.add_argument("--calib-seq-len", type=int, default=512)
    parser.add_argument("--w-bit", type=int, default=4, choices=[4, 8])
    parser.add_argument("--q-group-size", type=int, default=128)
    parser.add_argument(
        "--version", default="GEMM", choices=["GEMM", "GEMV"], help="AWQ kernel version"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quant_config = {
        "zero_point": True,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": args.version,
    }
    quantize_awq(
        model_path=args.model_path,
        output_path=args.output_path,
        calib_data_file=args.calib_data,
        quant_config=quant_config,
        n_calib_samples=args.n_calib_samples,
        calib_seq_len=args.calib_seq_len,
    )


if __name__ == "__main__":
    main()
