"""Merge BFCL and Glaive JSONL files, deduplicate, and split into train/val/test.

Outputs:
  data/processed/train.jsonl
  data/processed/val.jsonl
  data/processed/test.jsonl
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tool_call_finetune_lab.config import DataConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _content_hash(example: Dict[str, Any]) -> str:
    """Compute a deterministic hash of the messages content for deduplication."""
    # Use the user message content + first tool call name as dedup key
    messages = example.get("messages", [])
    user_texts = [m["content"] for m in messages if m.get("role") == "user"]
    tool_calls = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                tool_calls.append(fn.get("name", ""))

    key = json.dumps({"user": user_texts, "calls": tool_calls}, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    examples: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        logger.warning("File not found: %s — skipping", path)
        return examples
    with open(p, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("JSON parse error at %s line %d: %s", path, line_no, e)
    logger.info("Loaded %d examples from %s", len(examples), path)
    return examples


def deduplicate(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate examples by content hash."""
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    for ex in examples:
        h = _content_hash(ex)
        if h not in seen:
            seen.add(h)
            unique.append(ex)
    removed = len(examples) - len(unique)
    logger.info(
        "Deduplication: %d → %d examples (removed %d duplicates)",
        len(examples),
        len(unique),
        removed,
    )
    return unique


def stratified_split(
    examples: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Stratified split by source (bfcl / glaive) and category.

    Returns (train, val, test) lists.
    """
    rng = random.Random(seed)

    # Group by (source, category)
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        key = (ex.get("source", "unknown"), ex.get("category", "unknown"))
        groups[key].append(ex)

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []

    for (source, cat), group in groups.items():
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, int(n * train_ratio))
        n_val = max(0, int(n * val_ratio))
        # remainder goes to test
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])
        logger.debug(
            "  (%s, %s): %d total → %d train / %d val / %d test",
            source,
            cat,
            n,
            len(group[:n_train]),
            len(group[n_train : n_train + n_val]),
            len(group[n_train + n_val :]),
        )

    # Shuffle each split
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def save_jsonl(examples: List[Dict[str, Any]], output_path: str) -> None:
    """Write examples to a JSONL file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info("Saved %d examples to %s", len(examples), path)


def print_statistics(
    all_examples: List[Dict[str, Any]],
    train: List[Dict[str, Any]],
    val: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
) -> None:
    """Print dataset statistics to stdout."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total examples (after dedup): {len(all_examples)}")
    print(f"  Train: {len(train)} ({100 * len(train) / max(1, len(all_examples)):.1f}%)")
    print(f"  Val:   {len(val)} ({100 * len(val) / max(1, len(all_examples)):.1f}%)")
    print(f"  Test:  {len(test)} ({100 * len(test) / max(1, len(all_examples)):.1f}%)")
    print()

    # Source breakdown
    source_counts: Dict[str, int] = defaultdict(int)
    for ex in all_examples:
        source_counts[ex.get("source", "unknown")] += 1
    print("Source breakdown:")
    for src, count in sorted(source_counts.items()):
        print(f"  {src}: {count}")

    # Category breakdown
    cat_counts: Dict[str, int] = defaultdict(int)
    for ex in all_examples:
        cat_counts[ex.get("category", "unknown")] += 1
    print("\nTop categories:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cat}: {count}")

    # Avg tools per example
    total_tools = sum(len(ex.get("tools", [])) for ex in all_examples)
    avg_tools = total_tools / max(1, len(all_examples))
    print(f"\nAvg tools per example: {avg_tools:.1f}")

    # Multi-turn examples
    multi_turn = sum(
        1
        for ex in all_examples
        if sum(1 for m in ex.get("messages", []) if m["role"] == "user") > 1
    )
    print(
        f"Multi-turn examples: {multi_turn} ({100 * multi_turn / max(1, len(all_examples)):.1f}%)"
    )
    print("=" * 60 + "\n")


def main() -> None:
    config = DataConfig()

    # Load both sources
    bfcl_examples = load_jsonl(config.bfcl_output)
    glaive_examples = load_jsonl(config.glaive_output)

    all_examples = bfcl_examples + glaive_examples
    if not all_examples:
        logger.error("No examples found. Run prepare_bfcl.py and prepare_glaive.py first.")
        return

    # Deduplicate
    all_examples = deduplicate(all_examples)

    # Stratified split
    train, val, test = stratified_split(
        all_examples,
        config.train_ratio,
        config.val_ratio,
        config.seed,
    )

    # Save splits
    save_jsonl(train, config.train_file)
    save_jsonl(val, config.val_file)
    save_jsonl(test, config.test_file)

    # Print stats
    print_statistics(all_examples, train, val, test)
    logger.info("Merge and split complete.")


if __name__ == "__main__":
    main()
