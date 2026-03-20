"""Download Qwen2.5-7B-Instruct from HuggingFace Hub.

Usage:
    python scripts/download_base_model.py
    python scripts/download_base_model.py --model Qwen/Qwen2.5-7B-Instruct --dest models/base
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def download_model(
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    dest_dir: str = "models/base",
    revision: str = "main",
) -> str:
    """Download a model from HuggingFace Hub to a local directory.

    Args:
        model_id: HuggingFace model repository ID.
        dest_dir: Local directory to save the model.
        revision: Git revision (branch, tag, or commit hash).

    Returns:
        Absolute path to the downloaded model directory.
    """
    from huggingface_hub import snapshot_download

    dest = Path(dest_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning(
            "HF_TOKEN not set — download may fail for gated models. "
            "Set it in .env or export HF_TOKEN=your_token"
        )

    logger.info("Downloading %s (revision=%s) to %s ...", model_id, revision, dest)

    local_path = snapshot_download(
        repo_id=model_id,
        local_dir=str(dest),
        revision=revision,
        token=hf_token,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )

    logger.info("Download complete: %s", local_path)

    # Quick sanity check
    config_file = Path(local_path) / "config.json"
    if config_file.exists():
        import json

        with open(config_file) as f:
            cfg = json.load(f)
        logger.info(
            "Model type: %s, hidden_size: %s, num_layers: %s",
            cfg.get("model_type"),
            cfg.get("hidden_size"),
            cfg.get("num_hidden_layers"),
        )
    else:
        logger.warning("config.json not found — download may be incomplete")

    return local_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download base model from HuggingFace")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dest", default="models/base")
    parser.add_argument("--revision", default="main")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_model(args.model, args.dest, args.revision)


if __name__ == "__main__":
    main()
