#!/usr/bin/env python3
"""Experience-replay / mixed-corpus mixer for AttackLM.

Loads general-domain "replay" examples from one or more replay sources and
mixes them into fine-tuning batches to reduce catastrophic forgetting.

Replay source layout (mirrors per-source convention):
    data/datasets/buckets/sources/<source>/base/replay/data_<domain>.jsonl

Each JSONL record must have a `messages` array of {role, content} dicts.
Provenance fields (`source`, `source_uri`, `license`, `license_uri`,
`rights_contact`) are preserved as-is.

Usage from train_all.py:
    --replay-source replay-general/ --replay-ratio 0.07

Or programmatically:
    from replay_mixer import discover_replay_files, mix_replay
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Optional

# Base directory: one level up from this script
BASE_DIR = Path(__file__).resolve().parent.parent
SOURCES_DIR = BASE_DIR / "data" / "datasets" / "buckets" / "sources"
CACHE_DIR = BASE_DIR / "data" / "datasets" / "combined"


def discover_replay_files(source_dir: Path) -> dict[str, list[Path]]:
    """Scan a replay source directory for data_<domain>.jsonl files.

    Looks for files matching ``<source_dir>/base/replay/data_<domain>.jsonl``
    and groups them by domain (the stem portion after ``data_``).

    Args:
        source_dir: Path to a replay source directory
            (e.g. ``sources/replay-general/``).

    Returns:
        Dict mapping domain name to list of matching file paths.
        Empty dict if the source does not exist or has no replay files.
    """
    replay_dir = source_dir / "base" / "replay"
    if not replay_dir.is_dir():
        return {}

    domains: dict[str, list[Path]] = {}
    for f in sorted(replay_dir.glob("data_*.jsonl")):
        # Extract domain from "data_<domain>.jsonl"
        stem = f.stem  # e.g. "data_code"
        if stem.startswith("data_"):
            domain = stem[5:]  # strip "data_" prefix
        else:
            continue
        domains.setdefault(domain, []).append(f)
    return domains


def load_replay_domain(
    file_path: Path, max_examples: Optional[int] = None
) -> list[dict]:
    """Load JSONL records from a single replay domain file.

    Args:
        file_path: Path to a ``data_<domain>.jsonl`` file.
        max_examples: Cap on number of records to load (None = all).

    Returns:
        List of record dicts (preserving all original fields).
    """
    records: list[dict] = []
    with open(file_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_examples and len(records) >= max_examples:
                break
    return records


def _compute_cache_key(
    target_path: Path,
    replay_sources: list[str],
    ratio: float,
    max_examples: int,
    stratify: bool,
    domain_ratios: Optional[dict[str, float]],
    seed: int,
) -> str:
    """Compute a stable cache key for the mixed dataset.

    The key depends on the target file content hash, the replay source names,
    and all mixing parameters so that different configurations produce
    different cache files.
    """
    # Hash the target file content
    target_hash = _file_hash(target_path)
    payload = json.dumps(
        {
            "target_hash": target_hash,
            "sources": sorted(replay_sources),
            "ratio": ratio,
            "max_examples": max_examples,
            "stratify": stratify,
            "domain_ratios": domain_ratios,
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


def mix_replay(
    target_path: Path,
    replay_sources: list[Path],
    ratio: float,
    max_examples: int = 0,
    stratify: bool = True,
    domain_ratios: Optional[dict[str, float]] = None,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> tuple[Path, dict]:
    """Mix replay examples into a target dataset.

    Loads target data, discovers and loads replay data from one or more
    replay sources, samples according to ``ratio`` and stratification
    settings, then writes a combined shuffled JSONL.

    Args:
        target_path: Path to the target (combined) JSONL dataset.
        replay_sources: List of paths to replay source directories
            (e.g. ``[Path("data/datasets/buckets/sources/replay-general")]``).
        ratio: Fraction of target dataset size to sample as replay
            (e.g. 0.07 means 7% replay).
        max_examples: Hard cap on total replay examples (0 = use ratio).
        stratify: If True, split budget across domains proportionally;
            if False, sample uniformly from all replay records.
        domain_ratios: Optional override for per-domain weights
            (e.g. ``{"code": 0.3, "conversation": 0.25}``).
            Keys must match domain names discovered from file stems.
            Missing domains get equal share of remaining budget.
        seed: Random seed for reproducible shuffling and sampling.
        output_dir: Directory for the output cache file
            (default: ``data/datasets/combined/``).

    Returns:
        Tuple of (output_path, composition_dict).
        The composition dict includes:
            - ``target_examples``: number of target records
            - ``replay_examples``: number of replay records included
            - ``replay_ratio``: effective ratio achieved
            - ``replay_sources``: per-source record counts
            - ``replay_domains``: per-domain record counts
    """
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

    # ------------------------------------------------------------------ Discover & load replay
    all_replay: list[dict] = []
    source_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}

    for src_path in replay_sources:
        src_name = src_path.name if src_path.is_dir() else str(src_path)
        domains = discover_replay_files(src_path)
        if not domains:
            print(
                f"  WARNING: replay source '{src_name}' has no replay files, skipping"
            )
            source_counts[src_name] = 0
            continue

        src_total = 0
        for domain, files in domains.items():
            for fp in files:
                records = load_replay_domain(fp)
                all_replay.extend(records)
                domain_counts[domain] = domain_counts.get(domain, 0) + len(records)
                src_total += len(records)
        source_counts[src_name] = src_total

    n_available = len(all_replay)
    if n_available == 0:
        print(
            "  WARNING: no replay examples found across all sources; "
            "returning target dataset unchanged"
        )
        return target_path, {
            "target_examples": n_target,
            "replay_examples": 0,
            "replay_ratio": 0.0,
            "replay_sources": source_counts,
            "replay_domains": domain_counts,
        }

    # ------------------------------------------------------------------ Compute budget
    budget = int(ratio * n_target)
    if max_examples > 0:
        budget = min(budget, max_examples)
    budget = min(budget, n_available)

    # Short-circuit: if budget is 0, no replay to add — return target unchanged
    if budget <= 0:
        return target_path, {
            "target_examples": n_target,
            "replay_examples": 0,
            "replay_ratio": 0.0,
            "replay_sources": source_counts,
            "replay_domains": domain_counts,
        }

    # ------------------------------------------------------------------ Sample replay
    sampled: list[dict]
    if stratify and domain_counts:
        # Stratified sampling: split budget across domains
        sampled = _stratified_sample(
            all_replay, domain_counts, budget, domain_ratios, rng
        )
    else:
        # Uniform sampling across all replay records
        sampled = rng.sample(all_replay, budget)

    # ------------------------------------------------------------------ Compute actual composition
    actual_domain_counts: dict[str, int] = {}
    actual_source_counts: dict[str, int] = {}
    for rec in sampled:
        dom = _infer_domain(rec)
        if dom:
            actual_domain_counts[dom] = actual_domain_counts.get(dom, 0) + 1
        src = rec.get("source", "unknown")
        actual_source_counts[src] = actual_source_counts.get(src, 0) + 1

    composition = {
        "target_examples": n_target,
        "replay_examples": len(sampled),
        "replay_ratio": len(sampled) / n_target if n_target else 0.0,
        "replay_sources": actual_source_counts,
        "replay_domains": actual_domain_counts,
    }

    # ------------------------------------------------------------------ Write combined
    cache_key = _compute_cache_key(
        target_path,
        [str(s) for s in replay_sources],
        ratio,
        max_examples,
        stratify,
        domain_ratios,
        seed,
    )

    if output_dir is None:
        output_dir = CACHE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"replay_{cache_key}.jsonl"

    # Check cache
    if out_path.exists():
        cached_count = sum(1 for _ in open(out_path))
        print(f"  [replay cache HIT] {out_path.name} ({cached_count:,} pairs)")
        return out_path, composition

    combined = target_records + sampled
    rng.shuffle(combined)

    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in combined:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(
        f"  [replay cache WRITE] {out_path.name} "
        f"({len(combined):,} pairs, {file_size_mb:.1f} MB, "
        f"replay={len(sampled)}/{n_target}={len(sampled) / n_target:.1%})"
    )
    return out_path, composition


def _stratified_sample(
    all_replay: list[dict],
    domain_counts: dict[str, int],
    budget: int,
    domain_ratios: Optional[dict[str, float]],
    rng: random.Random,
) -> list[dict]:
    """Sample replay records with stratification across domains.

    If ``domain_ratios`` is provided, use those weights; otherwise
    distribute proportionally to file counts per domain.
    """
    # Group records by domain (using provenance or file-based assignment)
    domain_records: dict[str, list[dict]] = {}
    for rec in all_replay:
        dom = _infer_domain(rec)
        if dom:
            domain_records.setdefault(dom, []).append(rec)

    if not domain_records:
        # Fallback: uniform sample if no domain info
        return rng.sample(all_replay, min(budget, len(all_replay)))

    # Compute per-domain allocation
    if domain_ratios:
        # User-provided ratios: normalize and fill missing domains
        total_weight = sum(domain_ratios.values())
        known_domains = set(domain_records.keys())
        specified_domains = set(domain_ratios.keys())

        # Normalize specified ratios
        weights = {d: domain_ratios.get(d, 0.0) / total_weight for d in known_domains}

        # Distribute unspecified domains' budget equally
        unspecified = known_domains - specified_domains
        if unspecified:
            leftover = 1.0 - sum(weights.values())
            per_unspec = leftover / len(unspecified) if unspecified else 0.0
            for d in unspecified:
                weights[d] = per_unspec
    else:
        # Proportional to file counts
        total = sum(domain_records[d].__len__() for d in domain_records)
        weights = {d: len(domain_records[d]) / total for d in domain_records}

    # Allocate budget per domain
    sampled: list[dict] = []
    remaining_budget = budget

    # Sort domains for deterministic allocation
    sorted_domains = sorted(domain_records.keys())
    for i, dom in enumerate(sorted_domains):
        if i == len(sorted_domains) - 1:
            # Last domain gets the remainder to avoid rounding gaps
            n_dom = remaining_budget
        else:
            n_dom = max(0, round(budget * weights.get(dom, 0.0)))
            remaining_budget -= n_dom

        available = len(domain_records[dom])
        n_dom = min(n_dom, available)
        sampled.extend(rng.sample(domain_records[dom], n_dom))

    return sampled


def _infer_domain(rec: dict) -> str:
    """Infer the domain tag from a replay record.

    Checks for a ``domain`` field first, then falls back to extracting
    from the ``source`` field (e.g. ``"replay-general/code"`` → ``"code"``).
    Returns empty string if no domain can be inferred.
    """
    if "domain" in rec and rec["domain"]:
        return str(rec["domain"])
    # Fallback: extract from source field
    src = rec.get("source", "")
    if "/" in src:
        # e.g. "replay-general/code" → "code"
        return src.rsplit("/", 1)[-1]
    return ""


# --------------------------------------------------------------------------- CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Mix replay data into an AttackLM training dataset",
    )
    parser.add_argument(
        "target",
        type=Path,
        help="Path to target combined JSONL dataset",
    )
    parser.add_argument(
        "--replay-source",
        type=Path,
        nargs="+",
        required=True,
        help="One or more replay source directories",
    )
    parser.add_argument(
        "--replay-ratio",
        type=float,
        default=0.07,
        help="Fraction of target size to sample as replay (default: 0.07)",
    )
    parser.add_argument(
        "--replay-max-examples",
        type=int,
        default=0,
        help="Hard cap on total replay examples (0 = ratio-based budget)",
    )
    parser.add_argument(
        "--replay-stratify",
        dest="replay_stratify",
        action="store_true",
        default=True,
        help="Stratified sampling across domains (default: True)",
    )
    parser.add_argument(
        "--no-replay-stratify",
        dest="replay_stratify",
        action="store_false",
        help="Disable stratified sampling (uniform instead)",
    )
    parser.add_argument(
        "--replay-domain-ratios",
        type=str,
        default=None,
        help='JSON dict of domain weights, e.g. \'{"code":0.3,"conversation":0.25}\'',
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

    domain_ratios = None
    if args.replay_domain_ratios:
        domain_ratios = json.loads(args.replay_domain_ratios)

    out_path, comp = mix_replay(
        target_path=args.target,
        replay_sources=args.replay_source,
        ratio=args.replay_ratio,
        max_examples=args.replay_max_examples,
        stratify=args.replay_stratify,
        domain_ratios=domain_ratios,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print(f"\nOutput: {out_path}")
    print(f"Composition: {json.dumps(comp, indent=2)}")
