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
import math
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


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _content_hash(example: Dict[str, Any]) -> str:
    """Deduplicate full conversation/tool content, preserving distinct labels."""
    return _canonical_hash(
        {"messages": example.get("messages", []), "tools": example.get("tools", [])}
    )


def _input_hash(example: Dict[str, Any]) -> str:
    """Conservatively keep matching requests and tool schemas in one partition."""
    return _canonical_hash(
        {
            "messages": [
                message
                for message in example.get("messages", [])
                if message.get("role") in {"system", "user"}
            ],
            "tools": example.get("tools", []),
        }
    )


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    examples: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    with open(p, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                if not isinstance(example, dict):
                    raise ValueError(f"Expected a JSON object at {path}:{line_no}")
                examples.append(example)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
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
    """Split request groups, keeping matching inputs together across sources.

    Source/category balance is approximate when groups span multiple strata.
    Input ordering does not change the partition assigned with a fixed seed.
    """
    if (
        any(
            isinstance(value, bool) or not math.isfinite(value) or value < 0 or value > 1
            for value in (train_ratio, val_ratio)
        )
        or train_ratio + val_ratio > 1
    ):
        raise ValueError("split ratios must be finite fractions with a sum at most one")
    rng = random.Random(seed)  # nosec B311
    request_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for example in examples:
        request_groups[_input_hash(example)].append(example)
    strata: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for key, group in request_groups.items():
        stratum = min(
            (str(ex.get("source", "unknown")), str(ex.get("category", "unknown"))) for ex in group
        )
        strata[stratum].append(key)
    partitions: List[List[Dict[str, Any]]] = [[], [], []]
    for stratum in sorted(strata):
        keys = sorted(strata[stratum])
        rng.shuffle(keys)
        n = len(keys)
        n_train = max(1, int(n * train_ratio)) if train_ratio > 0 else 0
        n_val = int(n * val_ratio)
        for index, key in enumerate(keys):
            destination = 0 if index < n_train else 1 if index < n_train + n_val else 2
            partitions[destination].extend(
                sorted(request_groups[key], key=lambda ex: (_content_hash(ex), _canonical_hash(ex)))
            )
    for partition in partitions:
        rng.shuffle(partition)
    return partitions[0], partitions[1], partitions[2]


def save_jsonl(examples: List[Dict[str, Any]], output_path: str) -> None:
    """Write examples to a JSONL file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    _write_provenance(path, examples)
    logger.info("Saved %d examples to %s", len(examples), path)


def _write_provenance(path: Path, examples: List[Dict[str, Any]]) -> None:
    content = path.read_bytes()
    source_revisions = sorted(
        {
            str(ex.get("provenance", {}).get("source_revision"))
            for ex in examples
            if ex.get("provenance", {}).get("source_revision")
        }
    )
    provenance = {
        "artifact": str(path),
        "row_count": len(examples),
        "sha256": hashlib.sha256(content).hexdigest(),
        "source_revisions": source_revisions,
        "synthetic_fixture": any(
            bool(ex.get("provenance", {}).get("synthetic_fixture")) for ex in examples
        ),
    }
    path.with_suffix(path.suffix + ".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )


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
        raise ValueError("No examples found. Run prepare_bfcl.py and prepare_glaive.py first.")

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
