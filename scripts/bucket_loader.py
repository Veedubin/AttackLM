#!/usr/bin/env python3
"""Bucket loader for AttackLM.

Buckets are organized as 4 parents (v0.2.1+):
    data/datasets/buckets/
        manifest.json
        base/                          # 10 MITRE tactic buckets
            collection/
            command_and_control/
            ...
        tools/                         # External tool data (3 sub-buckets)
            metasploit/
            infection_monkey/
            rta/
        ai/                            # AI/ML red team category (2 sub-buckets)
            prompt-injection/
            jailbreaking/
        orchestrator/                  # Single bucket (top-level)

Bucket names are paths relative to BUCKETS_DIR using forward slashes:
    "base/collection", "ai/prompt-injection", "tools/metasploit"

This module provides:
    - list_buckets(category=None) — enumerate all buckets from manifest
    - get_bucket(name) — get metadata for a specific bucket (by path)
    - build_combined(bucket_names, flags, seed=42) — concatenate buckets
      into a single shuffled JSONL with a content-hash for cache invalidation
    - cache_key(bucket_names, flags) — stable hash for the combined dataset
    - get_tactic_buckets() — MITRE tactic buckets (filtered by category)
    - get_ai_model_buckets() — ai/* buckets (prompt-injection, jailbreaking)
    - get_tool_buckets() — tools/* buckets (metasploit, infection_monkey, rta)
    - get_default_train_buckets() — buckets trained by default (tactics + orchestrator)
    - resolve_dataset_spec(spec) / resolve_dataset_specs(specs) — convert
      a user-facing spec ("base/", "tools/metasploit/", "all", etc.) to
      the underlying list of bucket dicts

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
    has a 'path' field (e.g. "base/collection" or "ai/prompt-injection")
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
    """Get metadata for a single bucket by path (e.g. 'base/collection' or
    'ai/prompt-injection')."""
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

    `bucket_names` is a list of bucket paths (e.g. "base/collection",
    "ai/prompt-injection", "tools/metasploit").

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
        # For nested paths like "ai/prompt-injection" or "base/collection",
        # the path component is preserved as a relative path.
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
    """Return ai/* buckets (prompt-injection, jailbreaking).

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
    + orchestrator, then opt-in for ai/* and tools/* via flags.
    """
    return [b for b in list_buckets() if b.get("category") in ("tactic", "meta")]


def get_all_train_buckets() -> list[dict]:
    """Return all buckets that are part of the default training set.

    Alias for get_default_train_buckets() — kept for backward compat.
    """
    return get_default_train_buckets()


# ---------------------------------------------------------------------------
# Dataset spec resolver (v0.1.6+)
# ---------------------------------------------------------------------------
# Users want to pass a list of bucket specs on the CLI and have them
# resolved to actual bucket paths. The syntax is intentionally simple
# and directory-shaped so it mirrors the on-disk layout:
#
#   base/                     → all 10 MITRE tactic buckets
#   tools/                    → all 3 tool buckets (metasploit, infection_monkey, rta)
#   tools/metasploit/         → just metasploit
#   ai/                       → both AI buckets
#   ai/jailbreaking/          → just jailbreaking
#   orchestrator              → the orchestrator bucket
#
# Aliases for common combinations:
#   all                       → base + tools + ai + orchestrator
#   tactics                   → just base/
#   tools-all                 → just tools/ (alias for "tools/")
#
# This makes the CLI natural: `--dataset base/ tools/metasploit/`
# reads as "tactics + just metasploit, no infection_monkey or rta".

# Map from category-name to a function returning the list of buckets
_CATEGORY_RESOLVERS = {
    "base": get_tactic_buckets,
    "tools": get_tool_buckets,
    "ai": get_ai_model_buckets,
    "ai-models": get_ai_model_buckets,  # alias for 'ai'
}

# Map from top-level alias to a list of (resolver, subfilter)
_ALIAS_RESOLVERS = {
    "all": [
        ("base", None),
        ("tools", None),
        ("ai", None),
        ("orchestrator", None),
    ],
    "tactics": [("base", None)],
    "tools-all": [("tools", None)],
}


def _normalize_spec(spec: str) -> str:
    """Normalize a dataset spec: strip trailing slashes, lowercase."""
    s = spec.strip().rstrip("/").lower()
    return s


def resolve_dataset_spec(spec: str) -> list[dict]:
    """Resolve a single dataset spec to a list of bucket dicts.

    Accepts:
        "base/"                    → 10 tactic buckets
        "tools/"                   → 3 tool buckets
        "tools/metasploit/"        → 1 bucket
        "ai/"                      → 2 AI buckets
        "ai/jailbreaking/"         → 1 bucket
        "orchestrator"             → 1 orchestrator bucket
        "all"                      → all 4 categories
        "tactics"                  → just base/
        "tools-all"                → just tools/

    Returns list of bucket dicts (in stable, sorted order). Duplicate
    buckets are deduplicated while preserving first-seen order. If a
    spec doesn't match any known pattern, raises ValueError with a
    helpful message listing what was tried.
    """
    s = _normalize_spec(spec)
    if not s:
        return []

    # Alias?
    if s in _ALIAS_RESOLVERS:
        out: list[dict] = []
        for category, subfilter in _ALIAS_RESOLVERS[s]:
            if category == "orchestrator":
                b = get_orchestrator_bucket()
                if b:
                    out.append(b)
                continue
            resolver = _CATEGORY_RESOLVERS[category]
            for b in resolver():
                if (
                    subfilter
                    and not b["path"].startswith(subfilter + "/")
                    and b["path"] != subfilter
                ):
                    continue
                out.append(b)
        return _dedupe_buckets(out)

    # Category (base, tools, ai) → all buckets in that category.
    # Note: 'ai-models' is also accepted as an alias for 'ai' for
    # backward compat with v0.2.0 scripts that used the old name.
    if s in _CATEGORY_RESOLVERS:
        return _CATEGORY_RESOLVERS[s]()

    # Subpath (e.g. "tools/metasploit")
    # Treat the first segment as a category and the rest as a subfilter
    parts = s.split("/", 1)
    category = parts[0]
    subfilter = parts[1] if len(parts) > 1 else None

    if category in _CATEGORY_RESOLVERS:
        out = []
        for b in _CATEGORY_RESOLVERS[category]():
            if subfilter:
                # b["path"] is like "tools/metasploit", subfilter is "metasploit"
                if b["path"] == f"{category}/{subfilter}" or b["path"].endswith(
                    f"/{subfilter}"
                ):
                    out.append(b)
        if not out and subfilter:
            # Try the full path as a literal bucket name
            for b in _CATEGORY_RESOLVERS[category]():
                if b["path"] == f"{category}/{subfilter}":
                    return [b]
        return out

    # Orchestrator (no subpath — it's a single bucket, not a category)
    if s == "orchestrator":
        b = get_orchestrator_bucket()
        return [b] if b else []

    # Top-level bucket name (e.g. "collection", "defense_evasion")?
    b = get_bucket(s)
    if b:
        return [b]

    # Nothing matched — raise with a helpful list
    available = sorted({b["path"] for b in list_buckets()})
    available.extend(
        ["base/", "tools/", "ai/", "orchestrator", "all", "tactics", "tools-all"]
    )
    raise ValueError(
        f"Unknown dataset spec: {spec!r}\n"
        f"  Tried: alias, category, subpath, top-level bucket.\n"
        f"  Available specs:\n    " + "\n    ".join(available)
    )


def resolve_dataset_specs(specs: list[str]) -> list[dict]:
    """Resolve a list of dataset specs and return the union of buckets.

    Dedupes buckets across specs (preserving first-seen order). The output
    is suitable for passing directly to build_combined().
    """
    out = []
    for spec in specs:
        out.extend(resolve_dataset_spec(spec))
    return _dedupe_buckets(out)


def _dedupe_buckets(buckets: list[dict]) -> list[dict]:
    """Remove duplicate bucket dicts from a list, preserving first-seen order."""
    seen_paths = set()
    out = []
    for b in buckets:
        if b["path"] not in seen_paths:
            seen_paths.add(b["path"])
            out.append(b)
    return out


def format_specs_human(specs: list[str]) -> str:
    """Format a list of specs for human-readable display (e.g. in logs).

    Example: ['base/', 'tools/metasploit/'] → 'base + 1/3 tools (metasploit)'
    """
    parts = []
    for spec in specs:
        try:
            resolved = resolve_dataset_spec(spec)
            n = len(resolved)
            names = [b["path"] for b in resolved]
            if n == 0:
                parts.append(f"{spec} (no buckets)")
            elif n == 1:
                parts.append(f"{names[0]}")
            elif spec.rstrip("/") in _CATEGORY_RESOLVERS:
                # e.g. "tools/" → "3/3 tools"
                cat = spec.rstrip("/").lower()
                total = len(_CATEGORY_RESOLVERS[cat]())
                parts.append(f"{n}/{total} {cat}")
            else:
                parts.append(f"{n} buckets ({', '.join(names)})")
        except ValueError as e:
            parts.append(f"{spec} (invalid)")
    return " + ".join(parts)


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
