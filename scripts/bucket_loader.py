#!/usr/bin/env python3
"""Bucket loader for AttackLM.

Buckets are organized as:
    data/datasets/buckets/
        manifest.json
        <bucket_name>/              # top-level buckets (tactics, orchestrator)
            data.jsonl
            metadata.json
        ai-models/                  # AI/ML red team category
            prompt-injection/
            jailbreaking/
        tools/                      # External tool data
            metasploit/
            infection_monkey/
            rta/

Bucket names are paths relative to BUCKETS_DIR using forward slashes:
    "collection", "ai-models/prompt-injection", "tools/metasploit"

This module provides:
    - list_buckets(category=None) — enumerate all buckets from manifest
    - get_bucket(name) — get metadata for a specific bucket (by path)
    - build_combined(bucket_names, flags, seed=42) — concatenate buckets
      into a single shuffled JSONL with a content-hash for cache invalidation
    - cache_key(bucket_names, flags) — stable hash for the combined dataset
    - get_tactic_buckets() — MITRE tactic buckets
    - get_ai_model_buckets() — ai-models/* buckets (prompt-injection, jailbreaking)
    - get_tool_buckets() — tools/* buckets (metasploit, infection_monkey, rta)
    - get_default_train_buckets() — buckets trained by default (tactics + orchestrator)

The combined dataset is cached at:
    data/datasets/combined/<cache_key>.jsonl

If a new run with the same bucket_names + flags is launched, the cached file
is detected and reused. If the buckets or flags change, a new cache key is
computed and a fresh file is built.
"""

import hashlib
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

BUCKETS_DIR = Path("data/datasets/buckets")
CACHE_DIR = Path("data/datasets/combined")


def list_buckets(category: Optional[str] = None) -> list[dict]:
    """Enumerate all buckets, optionally filtered by category.

    Returns a list of bucket metadata dicts sorted by path. Each bucket
    has a 'path' field (e.g. "collection" or "ai-models/prompt-injection")
    that is used as the bucket identifier.
    """
    manifest_path = BUCKETS_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            f"Run `python scripts/setup_buckets.py` first."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)
    buckets = manifest["buckets"]
    if category:
        buckets = [b for b in buckets if b.get("category") == category]
    return sorted(buckets, key=lambda b: b["path"])


def get_bucket(name: str) -> Optional[dict]:
    """Get metadata for a single bucket by path (e.g. 'collection' or
    'ai-models/prompt-injection')."""
    for b in list_buckets():
        if b["path"] == name:
            return b
    return None


def cache_key(bucket_names: list[str], flags: dict) -> str:
    """Compute a stable short hash for a set of buckets + flags.

    The hash is used as the cache filename. Same buckets + same flags =
    same cache file (reused). Different buckets or flags = different file
    (rebuilt).
    """
    # Sort bucket names for stable ordering
    sorted_names = sorted(bucket_names)
    payload = json.dumps(
        {"buckets": sorted_names, "flags": flags},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def build_combined(
    bucket_names: list[str],
    flags: Optional[dict] = None,
    seed: int = 42,
    shuffle: bool = True,
) -> Path:
    """Concatenate the given buckets into a single shuffled JSONL file.

    `bucket_names` is a list of bucket paths (e.g. "collection",
    "ai-models/prompt-injection", "tools/metasploit").

    Returns the path to the cached combined file. If a cached file with the
    same key already exists, it is returned unchanged (idempotent re-runs).
    """
    flags = flags or {}
    key = cache_key(bucket_names, flags)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"combined_{key}.jsonl"

    # If cache exists, reuse it
    if cache_path.exists():
        count = sum(1 for _ in open(cache_path))
        print(f"  [cache HIT] Reusing {cache_path.name} ({count:,} pairs, key={key})")
        return cache_path

    # Build from source buckets
    print(f"  [cache MISS] Building combined dataset (key={key})...")
    all_pairs: list[dict] = []
    for name in bucket_names:
        b = get_bucket(name)
        if not b:
            print(f"    WARNING: bucket '{name}' not found in manifest, skipping")
            continue
        # data.jsonl lives at BUCKETS_DIR / <path> / data.jsonl
        # For nested paths like "ai-models/prompt-injection", the path component
        # is preserved as a relative path
        data_path = BUCKETS_DIR / name / "data.jsonl"
        if not data_path.exists():
            print(f"    WARNING: bucket '{name}' has no data.jsonl, skipping")
            continue
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_pairs.append(json.loads(line))
        print(f"    + {name:40s} {b['count']:>6,d} pairs")

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(all_pairs)
        print(f"  Shuffled {len(all_pairs):,} pairs (seed={seed})")

    # Write to cache
    with open(cache_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    file_size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(
        f"  [cache WRITE] {cache_path.name} ({len(all_pairs):,} pairs, {file_size_mb:.1f} MB)"
    )
    return cache_path


def clean_cache() -> int:
    """Remove all cached combined datasets. Returns the number of files removed."""
    if not CACHE_DIR.exists():
        return 0
    count = 0
    for f in CACHE_DIR.glob("combined_*.jsonl"):
        f.unlink()
        count += 1
    return count


def get_tactic_buckets() -> list[dict]:
    """Return only the MITRE tactic buckets (category='tactic').

    These are the 'core' buckets that always go into a single-model run
    unless explicitly excluded. Excludes orchestrator and prompt_injection
    (those are opt-in via flags).
    """
    return [b for b in list_buckets() if b.get("category") == "tactic"]


def get_ai_model_buckets() -> list[dict]:
    """Return ai-models/* buckets (prompt-injection, jailbreaking).

    These are AI/ML red team data — opt-in via --model-attacks flag.
    """
    return [b for b in list_buckets() if b.get("category") == "ai_redteam"]


def get_tool_buckets() -> list[dict]:
    """Return tools/* buckets (metasploit, infection_monkey, rta).

    These are external red-team tool data — opt-in via --include-tools flag.
    """
    return [b for b in list_buckets() if b.get("category") == "tools"]


def get_orchestrator_bucket() -> dict | None:
    """Return the orchestrator bucket (meta category), or None if not present."""
    for b in list_buckets():
        if b.get("category") == "meta":
            return b
    return None


def get_default_train_buckets() -> list[dict]:
    """Return all buckets that are part of the default training set.

    Default training set = 10 MITRE tactic buckets + 1 orchestrator = 11 total.
    This matches the user's spec: train one model per bucket, 10 tactics
    + orchestrator, then opt-in for ai-models/* and tools/* via flags.
    """
    return [b for b in list_buckets() if b.get("category") in ("tactic", "meta")]


def get_all_train_buckets() -> list[dict]:
    """Return all buckets that are part of the default training set.

    Alias for get_default_train_buckets() — kept for backward compat.
    """
    return get_default_train_buckets()


if __name__ == "__main__":
    # CLI: list buckets or build combined
    import argparse

    parser = argparse.ArgumentParser(description="Bucket loader CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_list = sub.add_parser("list", help="List all buckets")
    p_list.add_argument("--category", help="Filter by category")

    p_build = sub.add_parser("build", help="Build combined dataset")
    p_build.add_argument("buckets", nargs="+", help="Bucket names to combine")
    p_build.add_argument("--seed", type=int, default=42)
    p_build.add_argument("--no-shuffle", action="store_true")

    p_clean = sub.add_parser("clean-cache", help="Remove cached combined files")

    args = parser.parse_args()

    if args.cmd == "list":
        for b in list_buckets(args.category):
            print(
                f"  {b['path']:40s} {b['category']:12s} {b['mitre_tactic']:10s} {b['count']:>6,d} pairs"
            )
    elif args.cmd == "build":
        path = build_combined(args.buckets, seed=args.seed, shuffle=not args.no_shuffle)
        print(f"Output: {path}")
    elif args.cmd == "clean-cache":
        n = clean_cache()
        print(f"Removed {n} cached files")
    else:
        parser.print_help()
