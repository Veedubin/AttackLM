#!/usr/bin/env python3
"""Balanced bucket sampling for AttackLM.

Given a target model size + available VRAM, decide how many examples to
draw from each bucket so that no single bucket dominates training.

Why this exists
---------------
The bucket layout is heavily skewed: ``tools/metasploit`` alone can
dominate the dataset, while niche categories like ``ics/`` or
``wireless/`` may have only a handful of examples. If you train on
``--dataset all`` straight, the model overfits to the largest source.
For a 3B model on a 16GB card that's a limited step budget being burned
mostly on a single category.

For larger rigs, you have room to train on more data — but the
balance is still off. This sampler applies per-bucket caps that
shrink the largest buckets down toward a target distribution, while
keeping the small buckets at their full size (no upsampling — the
small buckets are the natural ceiling because we only have that
many real examples to teach from).

Categories (12)
---------------
AttackLM uses 12 attack-vector categories for balanced sampling:

    tactic              MITRE tactic buckets (recon, initial access, etc.)
    tools               External red-team tools (Metasploit, IM, RTA)
    web_app             Web application attacks (SQLi, XSS, CSRF, IDOR, SSRF)
    identity            Identity & access management (cred stuff, MFA bypass)
    cloud               Cloud & container security (AWS/GCP/Azure, K8s)
    social_engineering  Social engineering & OSINT (phishing, vishing)
    supply_chain        Supply chain attacks (dep confusion, typosquatting)
    wireless            Wireless/RF attacks (WPA, deauth, rogue AP)
    ics                 ICS/SCADA & OT (Modbus, PLC exploitation)
    physical            Physical security (tailgating, badge cloning)
    ai_specific         AI-specific security (prompt injection, jailbreaks)
    meta                Orchestrator / meta-level instruction pairs

Profiles
--------
A profile is a named bundle of (per-bucket cap, min-per-bucket).
The defaults below are derived from the bucket size distribution as
of 2026-06-10:

    3b-16gb            3B QLoRA on 16GB card. ~7-9K pairs, ~2-3 hr.
    7b-16gb            7B QLoRA on 16GB card. ~7-9K pairs, ~3-4 hr.
    7b-128gb           7B QLoRA on 128GB rig. ~10-12K pairs, ~4-6 hr.
    14b-128gb          14B QLoRA on 128GB rig. ~10-12K pairs, ~5-7 hr.
    31b-128gb          31B QLoRA on 128GB rig. ~12-15K pairs, ~6-8 hr.
    3b-16gb-balanced   3B balanced across 12 categories. ~10K pairs.
    7b-32gb-balanced   7B balanced across 12 categories. ~15K pairs.
    7b-128gb-balanced  7B balanced across 12 categories. ~25K pairs.
    full               No cap, use all pairs. ~12-16 hr on 128GB rig.

These are sensible defaults; tune with --per-bucket-cap or
--target-total for your own use case.

Usage
-----
Preview (no files written):

    python scripts/balance_buckets.py --profile 7b-128gb --dry-run

Write a balanced JSONL to data/datasets/balanced/:

    python scripts/balance_buckets.py --profile 7b-128gb \\
        --output data/datasets/balanced/balanced_7b-128gb.jsonl

Then pass that file to attacklm train:

    attacklm train --dataset data/datasets/balanced/balanced_7b-128gb.jsonl \\
                   --output models/attacklm-7b-128gb \\
                   --base-model huihui-ai/Qwen2.5-Coder-7B-Instruct-abliterated

Sampling strategies
-------------------
Within a bucket, after applying the per-bucket cap, examples are
selected with --strategy:

    head         First N examples in the file (reproducible, biased to
                 file order, but the data is already reasonably
                 shuffled by the extractors)
    random       Uniform random sample of N (seeded by --seed)
    stratified   Stratified sample: group by (mitre_ids[0], source)
                 when present, then sample proportionally from each
                 group. Default. Slower for large buckets because it
                 does a second pass over the data, but yields better
                 coverage of techniques and sources.

The output JSONL preserves the same schema as the source buckets
(messages, mitre_ids, source, license, etc.).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

# Reuse the existing bucket plumbing.
# When invoked as a console script (`attacklm balance`), the working
# directory may not be the project root, so we need to make sure
# sibling scripts in this directory are importable. Same pattern
# as scripts/train_template.py.
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent))

from bucket_loader import (
    BUCKETS_DIR,
    list_buckets,
    get_bucket,
)


# ---------------------------------------------------------------------------
# Profile definitions
# ---------------------------------------------------------------------------
# Each profile is a dict with:
#   per_bucket_cap: int    - max examples from any single bucket
#   min_per_bucket: int    - small buckets below this are kept whole
#                            (this is a courtesy: buckets smaller than
#                            the cap lose nothing; it's just to make
#                            the math obvious in the dry-run output)
#   description:   str     - human-readable description
#
# The per_bucket_cap applies to ALL buckets uniformly. We don't set
# different caps per bucket from a profile because that would be
# arbitrary; if you want per-bucket tuning, use --per-bucket-cap.
#
# Cap reasoning:
#   - The smallest non-trivial buckets (wireless, physical, ics) sit
#     well below 800, so a 800 cap on a 16GB card doesn't affect them.
#   - The largest bucket (metasploit=8349) gets cut by 90%+ on the
#     small profiles, which is the whole point.
#   - 128GB rigs benefit from slightly higher caps because the
#     larger models have more capacity to learn from the data.
#   - Balanced profiles (3b-16gb-balanced, etc.) use lower caps per
#     bucket but draw from all 12 categories for broader coverage.
# ---------------------------------------------------------------------------

PROFILES: dict[str, dict] = {
    "3b-16gb": {
        "per_bucket_cap": 800,
        "min_per_bucket": 100,
        "description": "3B QLoRA on 16GB card. ~7-9K total pairs, 2-3 hr.",
    },
    "7b-16gb": {
        "per_bucket_cap": 800,
        "min_per_bucket": 100,
        "description": "7B QLoRA on 16GB card. ~7-9K total pairs, 3-4 hr.",
    },
    "7b-128gb": {
        "per_bucket_cap": 1500,
        "min_per_bucket": 100,
        "description": "7B QLoRA on 128GB rig. ~10-12K total pairs, 4-6 hr.",
    },
    "14b-128gb": {
        "per_bucket_cap": 1500,
        "min_per_bucket": 100,
        "description": "14B QLoRA on 128GB rig. ~10-12K total pairs, 5-7 hr.",
    },
    "31b-128gb": {
        "per_bucket_cap": 2000,
        "min_per_bucket": 100,
        "description": "31B QLoRA on 128GB rig. ~12-15K total pairs, 6-8 hr.",
    },
    "3b-16gb-balanced": {
        "per_bucket_cap": 600,
        "min_per_bucket": 50,
        "description": (
            "3B balanced across 12 categories. ~10K total pairs, "
            "uses --target-total with category-balanced shares."
        ),
    },
    "7b-32gb-balanced": {
        "per_bucket_cap": 1000,
        "min_per_bucket": 75,
        "description": (
            "7B balanced across 12 categories. ~15K total pairs, "
            "uses --target-total with category-balanced shares."
        ),
    },
    "7b-128gb-balanced": {
        "per_bucket_cap": 2000,
        "min_per_bucket": 100,
        "description": (
            "7B balanced across 12 categories on 128GB rig. ~25K total pairs, "
            "uses --target-total with category-balanced shares."
        ),
    },
    "full": {
        "per_bucket_cap": 1_000_000,  # effectively unlimited
        "min_per_bucket": 0,
        "description": "No cap. Use all available pairs. ~12-16 hr on 128GB rig.",
    },
    "custom": {
        # --per-bucket-cap or --target-total required
        "per_bucket_cap": None,
        "min_per_bucket": 0,
        "description": "Custom: specify --per-bucket-cap or --target-total.",
    },
}


# ---------------------------------------------------------------------------
# Core sampling
# ---------------------------------------------------------------------------


def _stratify_key(example: dict) -> str:
    """Pick a stratification key for an example.

    Preference order:
        1. First MITRE technique ID (e.g. 'T1001.002')
        2. Source (e.g. 'rapid7/metasploit-framework')
        3. First line of assistant content (e.g. '**Module: exploits/windows/smb/psexec**')
        4. 'unknown'

    The third tier is the workhorse: most AttackLM examples have
    ``mitre_ids`` and ``source`` populated, but the metasploit
    bucket (8,349 examples) does not. In that case the assistant
    response's first line is something like::

        **Module: `exploits/windows/smb/ms17_010_psexec`** — MS17-010 ...

    The first ~80 chars of that line is a stable, content-based
    proxy for "this is about module X" — good enough to give
    stratified sampling real groups to work with even on the
    un-attributed buckets.
    """
    mitre_ids = example.get("mitre_ids") or []
    if mitre_ids and mitre_ids[0]:
        return mitre_ids[0]
    source = example.get("source")
    if source:
        return f"src:{source}"
    # Fall back to first line of the assistant content
    msgs = example.get("messages") or []
    for m in msgs:
        if m.get("role") == "assistant":
            content = m.get("content") or ""
            first_line = content.strip().split("\n", 1)[0].strip()
            if first_line:
                # Truncate to keep keys short
                return first_line[:80]
            break
    return "unknown"


def _sample_stratified(
    examples: list[dict],
    n: int,
    seed: int,
) -> list[dict]:
    """Stratified sample: group by stratification key, then allocate
    n examples across the groups with **at least 1 per group** when
    possible.

    Algorithm:
        1. Build groups by _stratify_key().
        2. If num_groups > n: can't give everyone 1, fall back to
           uniform random (the cap is the binding constraint here).
        3. Give every group a minimum share of 1.
        4. Distribute the remaining n - num_groups proportionally
           to group size (round, not floor).
        5. Cap each group at its own size.
        6. If still under n, the largest group gets the surplus.
        7. Shuffle the final selection so it's not group-by-group.

    Why minimum-1 per group?
        With sparse groups (most metasploit modules have 1-3
        examples), proportional allocation gives most groups 0.
        That defeats the purpose of stratification — we end up
        oversampling one large group instead of covering many
        small ones. Minimum-1 ensures every technique / module
        gets representation in the sample, which is what we
        actually want for "balanced" tactical coverage.

    Falls back to uniform random if:
        - fewer than 3 distinct groups (not worth stratifying)
        - or num_groups > n (cap is binding; min-1-per-group would
                              exceed n)
    """
    rng = random.Random(seed)

    groups: dict[str, list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        groups[_stratify_key(ex)].append(i)

    distinct = len(groups)
    if distinct < 3 or distinct > n:
        # Cap is binding or not worth stratifying
        return _sample_random(examples, n, seed)

    # Step 1: give every group a minimum share of 1
    # (Order groups deterministically by key for reproducibility)
    sorted_keys = sorted(groups.keys())
    allocation: dict[str, int] = {k: 1 for k in sorted_keys}
    running = len(sorted_keys)

    # Step 2: distribute the remaining n - running proportionally
    remaining = n - running
    if remaining > 0:
        total = len(examples)
        # Sort by size descending so the largest groups get the
        # extra allocation first (and the smallest groups don't
        # get capped at 1 unnecessarily)
        by_size = sorted(sorted_keys, key=lambda k: len(groups[k]), reverse=True)
        # Proportional allocation by size, rounded
        extras: dict[str, int] = {}
        for k in by_size:
            cap_room = len(groups[k]) - 1  # already have 1 allocated
            share = round(remaining * len(groups[k]) / total)
            extras[k] = min(cap_room, share)
        # If rounding undercounted, top up the largest group
        extra_running = sum(extras.values())
        deficit = remaining - extra_running
        if deficit > 0:
            for k in by_size:  # largest first
                cap_room = len(groups[k]) - 1 - extras[k]
                if cap_room > 0:
                    take = min(cap_room, deficit)
                    extras[k] += take
                    deficit -= take
                    if deficit <= 0:
                        break
        for k in sorted_keys:
            allocation[k] += extras.get(k, 0)

    # Step 3: actually sample from each group
    selected_indices: list[int] = []
    for k in sorted_keys:
        pool = groups[k]
        count = allocation[k]
        if count >= len(pool):
            selected_indices.extend(pool)
        else:
            selected_indices.extend(rng.sample(pool, count))

    # Step 4: if rounding overshot the target n, drop the surplus
    # from the largest groups (the ones that got the most extra).
    if len(selected_indices) > n:
        # Sort indices by group size descending, drop from the end
        # (which is the largest group)
        rng.shuffle(selected_indices)
        # Just take the first n — the order is already shuffled
        selected_indices = selected_indices[:n]

    # Step 5: shuffle so output isn't group-by-group
    rng.shuffle(selected_indices)
    return [examples[i] for i in selected_indices]


def _sample_random(examples: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if n >= len(examples):
        return list(examples)
    return rng.sample(examples, n)


def _sample_head(examples: list[dict], n: int, seed: int) -> list[dict]:
    # seed is unused but kept for signature symmetry
    return examples[:n]


def _load_bucket(bucket: dict) -> list[dict]:
    """Load all examples from a bucket. In the per-source layout (v0.3.0+),
    a bucket may be split across multiple sources; this aggregates them all."""
    from pathlib import Path
    from bucket_loader import SOURCES_DIR

    bucket_path = bucket["path"]
    candidates: list[Path] = []
    if SOURCES_DIR.exists():
        for src_dir in SOURCES_DIR.iterdir():
            if not src_dir.is_dir() or src_dir.name.startswith("_"):
                continue
            # 2-level: sources/<source>/<bucket>/<tactic>/
            p2 = src_dir / bucket_path
            if p2.is_dir():
                candidates.extend(p2.glob("*.jsonl"))
            # 1-level: sources/<source>/<bucket>/  (e.g. orchestrator)
            p1 = src_dir / bucket_path.split("/")[-1]
            if p1.is_dir() and p1 != p2:
                candidates.extend(p1.glob("*.jsonl"))
    if not candidates:
        print(
            f"    WARNING: no jsonl files for bucket '{bucket_path}', skipping",
            file=sys.stderr,
        )
        return []
    examples: list[dict] = []
    for jsonl in sorted(set(candidates)):
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
    return examples


# ---------------------------------------------------------------------------
# Per-bucket cap resolution
# ---------------------------------------------------------------------------


def _resolve_caps(
    profile_name: str,
    profile: dict,
    args: argparse.Namespace,
    buckets: list[dict],
) -> dict[str, int]:
    """Compute per-bucket caps based on the chosen profile + CLI overrides.

    Returns: {bucket_path: cap} dict.
    """
    # Custom profile: user must specify either --per-bucket-cap or --target-total
    if profile_name == "custom":
        if args.per_bucket_cap:
            # Apply the per-bucket cap. Accepts either a JSON dict of
            # {bucket_path: cap} or a plain integer for uniform cap.
            user_caps = json.loads(args.per_bucket_cap)
            if isinstance(user_caps, int):
                # Uniform cap: same limit for every bucket
                return {b["path"]: min(user_caps, b["count"]) for b in buckets}
            return {
                b["path"]: min(user_caps.get(b["path"], 1_000_000), b["count"])
                for b in buckets
            }
        if args.target_total:
            return _caps_for_target_total(args.target_total, buckets)
        print(
            "ERROR: --profile custom requires either --per-bucket-cap or --target-total",
            file=sys.stderr,
        )
        sys.exit(2)

    base_cap = profile["per_bucket_cap"]
    if base_cap is None:
        print(f"ERROR: profile {profile_name!r} has no per_bucket_cap", file=sys.stderr)
        sys.exit(2)

    return {b["path"]: min(base_cap, b["count"]) for b in buckets}


def _caps_for_target_total(target_total: int, buckets: list[dict]) -> dict[str, int]:
    """Distribute a target total across buckets, **balanced by category**.

    Strategy: first assign each of the 12 parent categories a target
    share of the total, then distribute that share across the buckets
    within each category, capped at each bucket's own size.

    Why category-balanced, not size-balanced?
        A naive "proportional to log(size)" allocation still gives
        the largest bucket (metasploit, 8,349) the largest cap —
        defeating the purpose of balancing. What we actually want
        is "each category contributes roughly equally so the model
        doesn't overfit to one data source."

    Target category shares (configurable via --category-shares):
        tactic              : 15%  (reduced from 50% — most tactic data
                                  is metasploit-heavy)
        tools               : 10%  (reduced from 25%)
        web_app             : 15%  (SQLi, XSS, CSRF, IDOR, SSRF, cmd inj)
        identity            : 10%  (cred stuffing, MFA bypass, Kerberoasting)
        cloud               : 10%  (AWS/GCP/Azure, K8s escapes, IAM privesc)
        social_engineering  :  8%  (phishing, vishing, pretexting)
        supply_chain        :  6%  (dep confusion, typosquatting, CI/CD)
        wireless            :  4%  (WPA attacks, deauth, rogue AP)
        physical            :  4%  (tailgating, badge cloning, USB drops)
        ics                 :  5%  (Modbus, PLC exploitation, industrial ransomware)
        ai_specific         :  8%  (prompt injection, jailbreaks, model extraction)
        meta                :  5%  (orchestrator / meta-level instruction pairs)

    Within a category, share is distributed by log(1 + size) so
    larger buckets get more (but not proportionally so). Tiny
    buckets (<= 100 examples) are kept whole.

    Args:
        target_total: desired total number of output pairs
        buckets: list of bucket dicts from the manifest

    Returns:
        {bucket_path: cap} dict
    """
    # Default category targets (sums to 1.00). Tuned by hand based on
    # the expanded 12-category bucket distribution. Override via
    # --category-shares.
    default_shares = {
        "tactic": 0.15,
        "tools": 0.10,
        "web_app": 0.15,
        "identity": 0.10,
        "cloud": 0.10,
        "social_engineering": 0.08,
        "supply_chain": 0.06,
        "wireless": 0.04,
        "physical": 0.04,
        "ics": 0.05,
        "ai_specific": 0.08,
        "meta": 0.05,
        "defensive": 0.00,  # added by team presets
    }

    # If a custom profile was selected, the user may have passed
    # --category-shares as a JSON string. We don't have access to args
    # here, so we read it from a module-global (set by main()).
    category_shares = _CATEGORY_SHARES_OVERRIDE or default_shares

    # Group buckets by category
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for b in buckets:
        cat = b.get("category", "?")
        by_cat[cat].append(b)

    # Tiny buckets: keep whole (they don't get downsampled regardless)
    tiny: list[dict] = [b for b in buckets if b["count"] <= 100]
    tiny_total = sum(b["count"] for b in tiny)

    # Compute per-category target sizes
    cat_targets: dict[str, int] = {}
    total_remaining = max(0, target_total - tiny_total)
    share_sum = sum(category_shares.values()) or 1.0
    for cat, share in category_shares.items():
        cat_targets[cat] = int(round(total_remaining * share / share_sum))

    # If a category's target exceeds its available data, redistribute
    # the excess to other categories proportionally (by their original share).
    # Loop because one redistribution can overshoot another small category.
    for _ in range(len(category_shares)):
        overshoot = 0
        caps_applied: set[str] = set()
        for cat, target in list(cat_targets.items()):
            avail_in_cat = sum(b["count"] for b in by_cat.get(cat, []))
            if target > avail_in_cat:
                overshoot += target - avail_in_cat
                cat_targets[cat] = avail_in_cat
                caps_applied.add(cat)
        if overshoot <= 0:
            break
        # Distribute the surplus proportionally to non-capped categories
        remaining_share_sum = (
            sum(s for c, s in category_shares.items() if c not in caps_applied) or 1.0
        )
        for cat, share in category_shares.items():
            if cat in caps_applied:
                continue
            avail = sum(b["count"] for b in by_cat.get(cat, []))
            room = avail - cat_targets[cat]
            if room <= 0:
                caps_applied.add(cat)
                continue
            give = min(room, int(round(overshoot * share / remaining_share_sum)))
            cat_targets[cat] += give
        # Check if we still have un-distributed overshoot
        total_room_left = sum(
            sum(b["count"] for b in by_cat.get(cat, [])) - cat_targets[cat]
            for cat in category_shares
        )
        if total_room_left <= 0:
            break  # no more room anywhere

    # Distribute each category's target across its buckets by
    # log-proportional share
    caps: dict[str, int] = {b["path"]: b["count"] for b in tiny}
    for cat, target in cat_targets.items():
        cat_buckets = [b for b in by_cat.get(cat, []) if b["count"] > 100]
        if not cat_buckets:
            continue
        # Sort by size descending so the largest bucket absorbs any
        # rounding remainder
        cat_buckets.sort(key=lambda b: b["count"], reverse=True)
        raw = {b["path"]: math.log1p(b["count"]) for b in cat_buckets}
        total_raw = sum(raw.values())
        running = 0
        for b in cat_buckets:
            share = math.floor(target * raw[b["path"]] / total_raw)
            cap = min(b["count"], share)
            caps[b["path"]] = cap
            running += cap
        # Top up the largest bucket in this category with the deficit
        deficit = target - running
        if deficit > 0:
            largest = cat_buckets[0]
            caps[largest["path"]] = min(
                largest["count"],
                caps[largest["path"]] + deficit,
            )

    # Ensure ALL buckets have a cap entry (categories not in shares get 0)
    for b in buckets:
        if b["path"] not in caps:
            caps[b["path"]] = 0

    return caps


# Module-global: set by main() when --category-shares is passed.
# Lets _caps_for_target_total() pick it up without changing the
# function signature (which is called from balance()).
_CATEGORY_SHARES_OVERRIDE: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# Main balancing routine
# ---------------------------------------------------------------------------


def balance(
    profile_name: str,
    args: argparse.Namespace,
    buckets: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    """Build a balanced sample of the given buckets.

    Returns: (selected_examples, stats_dict)
        selected_examples: list of example dicts ready to be written
        stats_dict: per-bucket counts, totals, dropped counts
    """
    if profile_name not in PROFILES:
        print(f"ERROR: unknown profile {profile_name!r}", file=sys.stderr)
        print(f"  Available: {', '.join(PROFILES.keys())}", file=sys.stderr)
        sys.exit(2)
    profile = PROFILES[profile_name]
    buckets = buckets or list_buckets()

    if not buckets:
        print("ERROR: no buckets found in manifest", file=sys.stderr)
        sys.exit(1)

    caps = _resolve_caps(profile_name, profile, args, buckets)
    sampler = {
        "head": _sample_head,
        "random": _sample_random,
        "stratified": _sample_stratified,
    }[args.strategy]

    selected: list[dict] = []
    stats: dict = {
        "profile": profile_name,
        "strategy": args.strategy,
        "seed": args.seed,
        "per_bucket": [],
        "totals": {
            "available": 0,
            "selected": 0,
            "dropped": 0,
            "buckets_uncapped": 0,
            "buckets_capped": 0,
        },
    }

    for b in buckets:
        cap = caps[b["path"]]
        available = b["count"]
        was_capped = cap < available

        examples = _load_bucket(b)
        if len(examples) != available:
            # Mismatch between manifest and actual file — warn and use
            # the actual count
            print(
                f"    WARNING: {b['path']} manifest says {available} but "
                f"file has {len(examples)}; using file count",
                file=sys.stderr,
            )
            available = len(examples)
            cap = min(cap, available)

        if cap == 0:
            sampled: list[dict] = []
        elif cap >= available:
            sampled = list(examples)  # keep all
        else:
            sampled = sampler(examples, cap, args.seed)

        # Tag each example with its source bucket path so we can
        # trace back where it came from
        for ex in sampled:
            ex.setdefault("_source_bucket", b["path"])

        selected.extend(sampled)
        stats["per_bucket"].append(
            {
                "path": b["path"],
                "category": b.get("category", "?"),
                "available": available,
                "cap": cap,
                "selected": len(sampled),
                "capped": was_capped,
            }
        )
        stats["totals"]["available"] += available
        stats["totals"]["selected"] += len(sampled)
        if was_capped:
            stats["totals"]["buckets_capped"] += 1
        else:
            stats["totals"]["buckets_uncapped"] += 1
    stats["totals"]["dropped"] = (
        stats["totals"]["available"] - stats["totals"]["selected"]
    )

    return selected, stats


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _print_stats_table(stats: dict, profile_desc: str) -> None:
    """Pretty-print the per-bucket balance table."""
    print(f"\nProfile: {stats['profile']}  —  {profile_desc}")
    print(f"Strategy: {stats['strategy']}    Seed: {stats['seed']}")
    print()
    print(
        f"  {'Bucket':40s}  {'Avail':>7s}  {'Cap':>5s}  {'Sel':>5s}  {'%':>5s}  {'Capped':>6s}"
    )
    print(f"  {'-' * 40}  {'-' * 7}  {'-' * 5}  {'-' * 5}  {'-' * 5}  {'-' * 6}")
    for b in stats["per_bucket"]:
        pct = (b["selected"] / b["available"] * 100) if b["available"] else 0
        capped = "yes" if b["capped"] else ""
        print(
            f"  {b['path']:40s}  {b['available']:>7,d}  {b['cap']:>5d}  "
            f"{b['selected']:>5d}  {pct:>4.0f}%  {capped:>6s}"
        )
    t = stats["totals"]
    print(f"  {'-' * 40}  {'-' * 7}  {'-' * 5}  {'-' * 5}  {'-' * 5}  {'-' * 6}")
    pct_total = (t["selected"] / t["available"] * 100) if t["available"] else 0
    print(
        f"  {'TOTAL':40s}  {t['available']:>7,d}  {'':>5s}  "
        f"{t['selected']:>5d}  {pct_total:>4.0f}%  "
        f"({t['buckets_capped']}/{t['buckets_capped'] + t['buckets_uncapped']} capped)"
    )
    print()


def _print_category_breakdown(stats: dict) -> None:
    """Show the category-level distribution so the user can sanity-check balance."""
    by_cat: dict[str, dict[str, int]] = defaultdict(
        lambda: {"available": 0, "selected": 0}
    )
    for b in stats["per_bucket"]:
        cat = b["category"]
        by_cat[cat]["available"] += b["available"]
        by_cat[cat]["selected"] += b["selected"]
    print("Category distribution:")
    print(f"  {'Category':14s}  {'Avail':>7s}  {'Sel':>5s}  {'% of total':>10s}")
    print(f"  {'-' * 14}  {'-' * 7}  {'-' * 5}  {'-' * 10}")
    total_sel = stats["totals"]["selected"]
    for cat in sorted(by_cat):
        sel = by_cat[cat]["selected"]
        avail = by_cat[cat]["available"]
        pct = (sel / total_sel * 100) if total_sel else 0
        print(f"  {cat:14s}  {avail:>7,d}  {sel:>5d}  {pct:>9.1f}%")
    print()


def _write_jsonl(examples: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


# ---------------------------------------------------------------------------
# Preset helpers
# ---------------------------------------------------------------------------

PRESET_DIR = _Path(__file__).resolve().parent.parent / "presets"


def _resolve_preset_path(preset_name: str) -> Path:
    """Resolve a preset name to a file path."""
    builtins = {
        "red-team": PRESET_DIR / "red-team.json",
        "purple-team": PRESET_DIR / "purple-team.json",
        "blue-team": PRESET_DIR / "blue-team.json",
    }
    if preset_name in builtins:
        path = builtins[preset_name]
    else:
        path = Path(preset_name)

    if not path.exists():
        raise FileNotFoundError(f"Preset not found: {path}")
    return path


def _bucket_weights_to_category_shares(weights: dict[str, float]) -> dict[str, float]:
    """Convert bucket weight patterns to category shares for balance_buckets.

    Maps wildcard patterns like 'base/*' to the actual category names used
    in the manifest (tactic, tools, ai_redteam, meta, defensive, etc.).
    """
    # Map bucket patterns to manifest categories
    pattern_to_category = {
        "base/*": "tactic",
        "tools/*": "tools",
        "ai/*": "ai_redteam",
        "orchestrator": "meta",
        "defensive/*": "defensive",
    }

    shares: dict[str, float] = {}
    for pattern, weight in weights.items():
        cat = pattern_to_category.get(pattern)
        if cat:
            if cat in shares:
                shares[cat] += weight
            else:
                shares[cat] = weight

    # Normalize to sum to 1.0
    total = sum(shares.values())
    if total > 0:
        shares = {k: v / total for k, v in shares.items()}

    return shares


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    # Pre-parse: handle --list-profiles (and --help) before argparse
    # validation, so users can run `balance_buckets.py --list-profiles`
    # without specifying --profile.
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if "--list-profiles" in argv_list:
        print("Available profiles:")
        for name, prof in PROFILES.items():
            cap = prof["per_bucket_cap"]
            cap_str = "unlimited" if cap is None or cap >= 1_000_000 else f"cap={cap}"
            print(f"  {name:10s}  {cap_str:10s}  {prof['description']}")
        print()
        print("Profiles determine the per-bucket cap applied uniformly to")
        print("all buckets. The 'custom' profile lets you set --per-bucket-cap")
        print("explicitly or --target-total with category-balanced allocation.")
        return 0
    if "--help" in argv_list or "-h" in argv_list:
        # Fall through to argparse for proper --help rendering
        pass

    parser = argparse.ArgumentParser(
        description="Build a balanced subset of the AttackLM buckets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--profile",
        default=None,
        choices=list(PROFILES.keys()),
        help="Target model + VRAM profile (determines per-bucket cap).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Team preset: 'red-team', 'purple-team', 'blue-team', or path to custom preset JSON. Overrides --profile.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Override system prompt in output pairs (used with --preset).",
    )
    parser.add_argument(
        "--strategy",
        default="stratified",
        choices=["head", "random", "stratified"],
        help="Within-bucket sampling strategy (default: stratified).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for random/stratified sampling (default: 42).",
    )
    parser.add_argument(
        "--per-bucket-cap",
        type=str,
        default=None,
        help=(
            "JSON string of {bucket_path: cap} overrides, e.g. "
            '\'{"tools/metasploit": 1500, "base/discovery": 800}\'. '
            "Only used with --profile custom."
        ),
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=None,
        help=(
            "Target total number of pairs to select across all buckets. "
            "Caps are auto-computed by category-balanced allocation "
            "(see --category-shares). Only used with --profile custom."
        ),
    )
    parser.add_argument(
        "--category-shares",
        type=str,
        default=None,
        help=(
            "JSON string overriding default category target shares. "
            'Example: \'{"tactic": 0.15, "tools": 0.10, "web_app": 0.15, '
            '"identity": 0.10, "cloud": 0.10, "social_engineering": 0.08, '
            '"supply_chain": 0.06, "wireless": 0.04, "physical": 0.04, '
            '"ics": 0.05, "ai_specific": 0.08, "meta": 0.05}\'. '
            "Shares must sum to 1.0. Only used with --profile custom "
            "+ --target-total."
        ),
    )
    parser.add_argument(
        "--buckets",
        nargs="+",
        default=None,
        help=(
            "Restrict to specific bucket paths (default: all buckets). "
            "Example: --buckets base/ tools/metasploit/"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL path. If omitted, runs in --dry-run mode "
            "(prints the stats and exits)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without writing output. Implied if --output omitted.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="Print all profiles and exit. (Handled before argparse validation.)",
    )

    args = parser.parse_args(argv)

    # --- Preset handling ---
    if args.preset:
        preset_path = _resolve_preset_path(args.preset)
        with open(preset_path) as f:
            preset = json.load(f)
        print(
            f"Loaded preset: {preset['name']} — {preset['description']}",
            file=sys.stderr,
        )

        # Override profile with custom
        args.profile = "custom"
        if args.target_total is None:
            args.target_total = preset.get("total_pairs", 16000)

        # Build category shares from bucket weights
        weights = preset.get("bucket_weights", {})
        shares = _bucket_weights_to_category_shares(weights)
        args.category_shares = json.dumps(shares)

        if args.system_prompt is None:
            args.system_prompt = preset.get("system_prompt")

    if not args.profile:
        print("ERROR: --profile or --preset is required", file=sys.stderr)
        sys.exit(2)

    # Wire --category-shares through to the cap allocator via
    # the module-global. (It's the cleanest way to inject a runtime
    # parameter into a function that has multiple call sites.)
    if args.category_shares:
        try:
            parsed = json.loads(args.category_shares)
        except json.JSONDecodeError as e:
            print(f"ERROR: --category-shares must be valid JSON: {e}", file=sys.stderr)
            sys.exit(2)
        total = sum(parsed.values())
        if not (0.99 < total < 1.01):
            print(
                f"ERROR: --category-shares must sum to 1.0, got {total:.3f}",
                file=sys.stderr,
            )
            sys.exit(2)
        global _CATEGORY_SHARES_OVERRIDE
        _CATEGORY_SHARES_OVERRIDE = parsed

    # Filter buckets if --buckets was passed
    if args.buckets:
        from bucket_loader import resolve_dataset_specs

        resolved = resolve_dataset_specs(args.buckets)
        if not resolved:
            print("ERROR: --buckets resolved to no buckets", file=sys.stderr)
            sys.exit(2)
        buckets = resolved
    else:
        buckets = list_buckets()

    selected, stats = balance(args.profile, args, buckets=buckets)

    profile_desc = PROFILES[args.profile]["description"]
    _print_stats_table(stats, profile_desc)
    _print_category_breakdown(stats)

    if args.dry_run or args.output is None:
        print("(dry run — no file written. Pass --output <path> to write.)")
        return 0

    _write_jsonl(selected, args.output)
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Wrote {len(selected):,} pairs ({size_mb:.1f} MB) to {args.output}")
    print(
        f"\nNext step:\n"
        f"  attacklm train --dataset {args.output} \\\n"
        f"                 --output models/attacklm-{args.profile.replace('-', '_')}_<timestamp> \\\n"
        f"                 --base-model huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
