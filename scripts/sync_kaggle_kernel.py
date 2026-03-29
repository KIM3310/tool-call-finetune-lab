"""Sync the checked-in Kaggle notebook into the kernel folder and set visibility.

Usage:
    python scripts/sync_kaggle_kernel.py --public
    python scripts/sync_kaggle_kernel.py --private
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync Kaggle notebook + metadata")
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--public", action="store_true", help="Set kernel-metadata.json to public")
    visibility.add_argument("--private", action="store_true", help="Set kernel-metadata.json to private")
    parser.add_argument(
        "--source-notebook",
        default="notebooks/kaggle_full_pipeline.ipynb",
        help="Source notebook path",
    )
    parser.add_argument(
        "--kernel-dir",
        default="notebooks/kaggle-kernel",
        help="Destination Kaggle kernel directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    source_notebook = repo_root / args.source_notebook
    kernel_dir = repo_root / args.kernel_dir
    dest_notebook = kernel_dir / source_notebook.name
    metadata_path = kernel_dir / "kernel-metadata.json"

    kernel_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_notebook, dest_notebook)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["is_private"] = bool(args.private) and not bool(args.public)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    visibility = "private" if metadata["is_private"] else "public"
    print(f"Notebook synced to {dest_notebook}")
    print(f"Kernel metadata updated: {metadata_path} ({visibility})")


if __name__ == "__main__":
    main()
