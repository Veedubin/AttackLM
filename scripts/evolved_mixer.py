#!/usr/bin/env python3
"""Evolved-pair mixer for AttackLM.

Loads evolved (longer, more complex) training pairs from filtered JSONL files
and mixes them into fine-tuning datasets at a configurable ratio.

Evolved file layout:
    data/datasets/evolved/<source>_<method>_filtered.jsonl

Each JSONL record must have a ``messages`` array of {role, content} dicts.
Provenance fields (``source``, ``source_uri``, ``license``, etc.) are
preserved as-is.

Usage from train_all.py:
    --evolved-ratio 0.3 --evolved-dir data/datasets/evolved

Or programmatically:
    from evolved_mixer import discover_evolved_files, mix_evolved
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Optional

# Base directory: one level up from this script
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EVOLVED_DIR = BASE_DIR / "data" / "datasets" / "evolved"
CACHE_DIR = BASE_DIR / "data" / "datasets" / "combined"


def discover_evolved_files(evolved_dir: Path) -> dict[str, list[Path]]:
    """Scan an evolved directory for ``*_filtered.jsonl`` files.

    Looks for files matching ``<evolved_dir>/*_filtered.jsonl`` and groups
    them by source name (the stem with ``_filtered`` stripped).

    Args:
        evolved_dir: Path to the evolved data directory
            (e.g. ``data/datasets/evolved/``).

    Returns:
        Dict mapping source name to list of matching file paths.
        Empty dict if the directory does not exist or has no evolved files.
    """
    if not evolved_dir.is_dir():
        return {}

    sources: dict[str, list[Path]] = {}
    for f in sorted(evolved_dir.glob("*_filtered.jsonl")):
        # Extract source name from "<source>_<method>_filtered.jsonl"
        stem = f.stem  # e.g. "metasploit-framework_multi_turn_filtered"
        if stem.endswith("_filtered"):
            source_name = stem[: -len("_filtered")]
        else:
            # Fallback: use full stem
            source_name = stem
        sources.setdefault(source_name, []).append(f)
    return sources


def load_evolved_file(
    file_path: Path, max_examples: Optional[int] = None
) -> list[dict]:
    """Load JSONL records from a single evolved file.

    Args:
        file_path: Path to a ``*_filtered.jsonl`` file.
        max_examples: Cap on number of records to load (None = all).

    Returns:
        List of record dicts (preserving all original fields).
    """
    records: list[dict] = []
    try:
        with open(file_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
                if max_examples and len(records) >= max_examples:
                    break
    except (OSError, json.JSONDecodeError) as e:
        print(f"  WARNING: error reading {file_path}: {e}")
        return []
    return records


def _compute_cache_key(
    target_path: Path,
    evolved_sources: list[str],
    ratio: float,
    seed: int,
) -> str:
    """Compute a stable cache key for the mixed dataset.

    The key depends on the target file content hash, the evolved source
    names, the ratio, and the seed so that different configurations
    produce different cache files.
    """
    target_hash = _file_hash(target_path)
    payload = json.dumps(
        {
            "target_hash": target_hash,
            "evolved_sources": sorted(evolved_sources),
            "ratio": ratio,
            "seed": seed,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _file_hash(path: Path) -> str:
    """SHA-256 hash of a file (first 64 hex chars)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def mix_evolved(
    target_path: Path,
    evolved_dir: Path,
    ratio: float,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> tuple[Path, dict]:
    """Mix evolved examples into a target dataset.

    Loads target data, discovers and loads evolved data from the evolved
    directory, samples according to ``ratio``, then writes a combined
    shuffled JSONL.

    Args:
        target_path: Path to the target (combined) JSONL dataset.
        evolved_dir: Path to directory containing ``*_filtered.jsonl`` files.
        ratio: Fraction of target dataset size to sample as evolved pairs
            (e.g. 0.3 means 30% evolved). Clamped to [0.0, 1.0].
        seed: Random seed for reproducible shuffling and sampling.
        output_dir: Directory for the output cache file
            (default: ``data/datasets/combined/``).

    Returns:
        Tuple of (output_path, composition_dict).
        The composition dict includes:
            - ``target_examples``: number of target records
            - ``evolved_examples``: number of evolved records included
            - ``evolved_ratio``: effective ratio achieved
            - ``evolved_sources``: per-source record counts
    """
    # Clamp ratio to [0.0, 1.0]
    ratio = max(0.0, min(1.0, ratio))

    if ratio <= 0.0:
        return target_path, {
            "target_examples": 0,
            "evolved_examples": 0,
            "evolved_ratio": 0.0,
            "evolved_sources": {},
        }

    rng = random.Random(seed)

    # ------------------------------------------------------------------ Load target
    target_records: list[dict] = []
    with open(target_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                target_records.append(json.loads(line))

    n_target = len(target_records)
    if n_target == 0:
        raise ValueError(f"Target dataset is empty: {target_path}")

    # ------------------------------------------------------------------ Discover & load evolved
    sources = discover_evolved_files(evolved_dir)
    if not sources:
        print(
            f"  WARNING: no *_filtered.jsonl files found in {evolved_dir}; "
            "returning target dataset unchanged"
        )
        return target_path, {
            "target_examples": n_target,
            "evolved_examples": 0,
            "evolved_ratio": 0.0,
            "evolved_sources": {},
        }

    all_evolved: list[dict] = []
    source_counts: dict[str, int] = {}

    for source_name, files in sources.items():
        src_total = 0
        for fp in files:
            records = load_evolved_file(fp)
            if not records:
                print(f"  WARNING: {fp.name} is empty, skipping")
                continue
            all_evolved.extend(records)
            src_total += len(records)
        if src_total > 0:
            source_counts[source_name] = src_total

    n_available = len(all_evolved)
    if n_available == 0:
        print(
            "  WARNING: no evolved examples loaded across all sources; "
            "returning target dataset unchanged"
        )
        return target_path, {
            "target_examples": n_target,
            "evolved_examples": 0,
            "evolved_ratio": 0.0,
            "evolved_sources": source_counts,
        }

    # ------------------------------------------------------------------ Compute budget
    budget = int(ratio * n_target)
    budget = min(budget, n_available)

    if budget <= 0:
        return target_path, {
            "target_examples": n_target,
            "evolved_examples": 0,
            "evolved_ratio": 0.0,
            "evolved_sources": source_counts,
        }

    # ------------------------------------------------------------------ Sample evolved (stratified by source)
    sampled = _stratified_sample(all_evolved, source_counts, budget, rng)

    # ------------------------------------------------------------------ Compute actual composition
    actual_source_counts: dict[str, int] = {}
    for rec in sampled:
        src = rec.get("source", "unknown")
        actual_source_counts[src] = actual_source_counts.get(src, 0) + 1

    composition = {
        "target_examples": n_target,
        "evolved_examples": len(sampled),
        "evolved_ratio": len(sampled) / n_target if n_target else 0.0,
        "evolved_sources": actual_source_counts,
    }

    # ------------------------------------------------------------------ Write combined
    cache_key = _compute_cache_key(
        target_path,
        list(source_counts.keys()),
        ratio,
        seed,
    )

    if output_dir is None:
        output_dir = CACHE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"evolved_{cache_key}.jsonl"

    # Check cache
    if out_path.exists():
        cached_count = sum(1 for _ in open(out_path))
        print(f"  [evolved cache HIT] {out_path.name} ({cached_count:,} pairs)")
        return out_path, composition

    combined = target_records + sampled
    rng.shuffle(combined)

    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in combined:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(
        f"  [evolved cache WRITE] {out_path.name} "
        f"({len(combined):,} pairs, {file_size_mb:.1f} MB, "
        f"evolved={len(sampled)}/{n_target}={len(sampled) / n_target:.1%})"
    )
    return out_path, composition


def _stratified_sample(
    all_evolved: list[dict],
    source_counts: dict[str, int],
    budget: int,
    rng: random.Random,
) -> list[dict]:
    """Sample evolved records proportionally across sources.

    Distributes the budget across sources proportional to their available
    records. This ensures smaller sources still contribute rather than being
    overwhelmed by larger ones.
    """
    # Group records by source
    source_records: dict[str, list[dict]] = {}
    for rec in all_evolved:
        src = rec.get("source", "unknown")
        source_records.setdefault(src, []).append(rec)

    if not source_records:
        # Fallback: uniform sample
        return rng.sample(all_evolved, min(budget, len(all_evolved)))

    # Compute per-source allocation (proportional to source size)
    total_available = sum(source_counts.values())
    sampled: list[dict] = []
    remaining_budget = budget

    sorted_sources = sorted(source_records.keys())
    for i, src in enumerate(sorted_sources):
        if i == len(sorted_sources) - 1:
            # Last source gets the remainder
            n_src = remaining_budget
        else:
            # Proportional allocation
            proportion = (
                source_counts.get(src, 0) / total_available
                if total_available > 0
                else 0
            )
            n_src = max(0, round(budget * proportion))
            remaining_budget -= n_src

        available = len(source_records[src])
        n_src = min(n_src, available)
        sampled.extend(rng.sample(source_records[src], n_src))

    return sampled


# --------------------------------------------------------------------------- CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Mix evolved pairs into an AttackLM training dataset",
    )
    parser.add_argument(
        "target",
        type=Path,
        help="Path to target combined JSONL dataset",
    )
    parser.add_argument(
        "--evolved-dir",
        type=Path,
        default=DEFAULT_EVOLVED_DIR,
        help=f"Directory containing evolved JSONL files (default: {DEFAULT_EVOLVED_DIR})",
    )
    parser.add_argument(
        "--evolved-ratio",
        type=float,
        default=0.3,
        help="Fraction of target size to sample as evolved pairs (default: 0.3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for combined file (default: data/datasets/combined/)",
    )

    args = parser.parse_args()

    out_path, comp = mix_evolved(
        target_path=args.target,
        evolved_dir=args.evolved_dir,
        ratio=args.evolved_ratio,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print(f"\nOutput: {out_path}")
    print(f"Composition: {json.dumps(comp, indent=2)}")
