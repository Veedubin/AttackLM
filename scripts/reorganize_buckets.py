#!/usr/bin/env python3
"""Reorganize buckets into the new layout:
    ai-models/prompt-injection/
    ai-models/jailbreaking/
    tools/metasploit/
    tools/infection_monkey/
    tools/rta/

Also merge atomic_red_team + caldera into the 10 MITRE tactic buckets
(since they're ATT&CK-based, each pair gets routed to its tactic bucket
by looking up the technique ID in the user message).

Usage:
    uv run python scripts/reorganize_buckets.py
"""

import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mitre_tactic_lookup import get_tactic_for_technique  # noqa: E402

DATA_DIR = Path("data/datasets")
BUCKETS_DIR = DATA_DIR / "buckets"

# MITRE tactic ID → bucket directory name
TACTIC_ID_TO_BUCKET = {
    "TA0001": "initial-access",  # not currently a bucket
    "TA0002": "execution",
    "TA0003": "persistence",
    "TA0004": "privilege-escalation",
    "TA0005": "defense-evasion",
    "TA0006": "credential-access",
    "TA0007": "discovery",
    "TA0008": "lateral-movement",
    "TA0009": "collection",
    "TA0010": "exfiltration",
    "TA0011": "command-and-control",
    "TA0012": "impact",
    "TA0042": "resource-development",  # not currently a bucket
    "TA0043": "reconnaissance",  # not currently a bucket
}

# Map current tactic bucket directory names to tactic IDs
BUCKET_NAME_TO_TACTIC_ID = {
    "collection": "TA0009",
    "command_and_control": "TA0011",
    "credential_access": "TA0006",
    "defense_evasion": "TA0005",
    "discovery": "TA0007",
    "execution": "TA0002",
    "exfiltration": "TA0010",
    "lateral_movement": "TA0008",
    "persistence": "TA0003",
    "privilege_escalation": "TA0004",
}

# Buckets to move to ai-models/ and tools/
MOVES = {
    # current name → new path (relative to BUCKETS_DIR)
    "prompt_injection": "ai-models/prompt-injection",
    "jailbreak": "ai-models/jailbreaking",
    "metasploit": "tools/metasploit",
    "infection_monkey": "tools/infection_monkey",
    "rta": "tools/rta",
}


# Buckets to merge into MITRE tactic buckets
TO_MERGE = ["atomic_red_team", "caldera_plugins"]


# Regex to extract a MITRE technique ID from a user message
# Matches "T1234" or "T1234.001" with optional (Description) suffix
TECH_ID_RE = re.compile(r"\b(T\d{4}(?:\.\d{3})?)\b")


def extract_technique_id(messages: list[dict]) -> str | None:
    """Extract the MITRE technique ID from a pair's messages.

    Looks first in user message, then assistant, then system.
    Returns the first T-ID found, or None.
    """
    # Priority: user message > assistant > system
    for role in ("user", "assistant", "system"):
        for m in messages:
            if m.get("role") != role:
                continue
            content = m.get("content", "")
            m_id = TECH_ID_RE.search(content)
            if m_id:
                return m_id.group(1)
    return None


def move_bucket(src_name: str, dest_rel: str) -> tuple[Path, Path]:
    """Move a bucket directory. Returns (src, dest)."""
    src = BUCKETS_DIR / src_name
    dest = BUCKETS_DIR / dest_rel
    if not src.exists():
        print(f"  WARNING: source {src_name} does not exist, skipping")
        return src, dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(src), str(dest))
    print(f"  Moved: {src_name}/ → {dest_rel}/")
    return src, dest


def merge_attack_based_buckets(
    bucket_names: list[str], dry_run: bool = False
) -> dict[str, int]:
    """Merge atomic_red_team + caldera pairs into the 10 MITRE tactic buckets.

    For each pair, look up the technique ID in the user message, find its
    tactic, and append the pair to that tactic's bucket.

    Returns a dict of {tactic_bucket: merged_count}.
    """
    merged_counts: dict[str, int] = {}
    unmatched_count = 0
    total_processed = 0

    # First pass: collect all pairs from the source buckets
    all_pairs: list[dict] = []
    for bn in bucket_names:
        bucket_path = BUCKETS_DIR / bn
        if not bucket_path.exists():
            print(f"  WARNING: bucket {bn} not found, skipping")
            continue
        data_path = bucket_path / "data.jsonl"
        if not data_path.exists():
            continue
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_pairs.append(json.loads(line))
        print(f"  Loaded {len(all_pairs) - total_processed} pairs from {bn}/data.jsonl")
        total_processed = len(all_pairs)

    # Second pass: route each pair to its tactic bucket
    routed: dict[str, list[dict]] = {}  # tactic_bucket_name → list of pairs
    for pair in all_pairs:
        msgs = pair.get("messages", [])
        tech_id = extract_technique_id(msgs)
        if not tech_id:
            unmatched_count += 1
            continue
        tactic_id = get_tactic_for_technique(tech_id)
        if not tactic_id or tactic_id not in TACTIC_ID_TO_BUCKET:
            unmatched_count += 1
            continue
        bucket_name = TACTIC_ID_TO_BUCKET[tactic_id]
        # If this tactic doesn't have a dedicated bucket, skip
        if bucket_name not in BUCKET_NAME_TO_TACTIC_ID:
            unmatched_count += 1
            continue
        routed.setdefault(bucket_name, []).append(pair)

    # Third pass: append routed pairs to the destination buckets
    for tactic_bucket, pairs in routed.items():
        dest = BUCKETS_DIR / tactic_bucket / "data.jsonl"
        if dry_run:
            print(
                f"  [DRY RUN] would append {len(pairs):>5d} pairs → {tactic_bucket}/data.jsonl"
            )
        else:
            with open(dest, "a") as f:
                for pair in pairs:
                    f.write(json.dumps(pair) + "\n")
            print(f"  Merged {len(pairs):>5d} pairs → {tactic_bucket}/data.jsonl")
        merged_counts[tactic_bucket] = len(pairs)

    print()
    print(f"  Total processed:  {len(all_pairs):>5d} pairs")
    print(f"  Matched → tactic: {sum(merged_counts.values()):>5d} pairs")
    print(
        f"  Unmatched:        {unmatched_count:>5d} pairs (no T-ID or unknown tactic)"
    )

    return merged_counts


def update_metadata_for_moved_buckets():
    """Update metadata.json for the moved buckets (new path, new category)."""
    updates = {
        "ai-models/prompt-injection": (
            "ai_redteam",
            "TA0040",
            "Prompt injection & AI exploit (was prompt_injection)",
        ),
        "ai-models/jailbreaking": (
            "ai_redteam",
            "TA0040",
            "AI jailbreak patterns (was jailbreak)",
        ),
        "tools/metasploit": (
            "tools",
            "various",
            "Rapid7 Metasploit — 15 module categories consolidated",
        ),
        "tools/infection_monkey": (
            "tools",
            "various",
            "guardicore/infection_monkey — 36 plugin manifests",
        ),
        "tools/rta": ("tools", "various", "endgameinc/RTA — 40 Python TTP scripts"),
    }
    for rel_path, (category, mitre, description) in updates.items():
        meta_path = BUCKETS_DIR / rel_path / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        meta["category"] = category
        meta["mitre_tactic"] = mitre
        meta["description"] = description
        # Update count from data.jsonl
        data_path = BUCKETS_DIR / rel_path / "data.jsonl"
        if data_path.exists():
            meta["count"] = sum(1 for line in open(data_path) if line.strip())
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


def rebuild_manifest():
    """Rebuild manifest.json from the new directory layout."""
    buckets = []
    # Walk the bucket directory and find all metadata.json files
    for meta_path in sorted(BUCKETS_DIR.glob("**/metadata.json")):
        rel_dir = meta_path.parent.relative_to(BUCKETS_DIR)
        with open(meta_path) as f:
            meta = json.load(f)
        # Use forward-slash path as the bucket name (supports nested buckets)
        meta["path"] = str(rel_dir).replace("\\", "/")
        # Recompute count
        data_path = meta_path.parent / "data.jsonl"
        if data_path.exists():
            meta["count"] = sum(1 for line in open(data_path) if line.strip())
        buckets.append(meta)

    # Sort: tactic buckets first (by name), then orchestrator, then ai-models, then tools
    def sort_key(b):
        cat = b.get("category", "")
        if cat == "tactic":
            return (0, b["path"])
        if cat == "meta":
            return (1, b["path"])
        if cat == "ai_redteam":
            return (2, b["path"])
        if cat == "tools":
            return (3, b["path"])
        return (4, b["path"])

    buckets.sort(key=sort_key)

    manifest = {
        "version": 2,
        "layout": "nested (ai-models/, tools/)",
        "created": datetime.now().isoformat(),
        "total_buckets": len(buckets),
        "total_pairs": sum(b["count"] for b in buckets),
        "buckets": buckets,
    }

    with open(BUCKETS_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main():
    print("=" * 70)
    print("REORGANIZING BUCKETS")
    print("=" * 70)
    print()

    # Step 1: Move prompt_injection, jailbreak → ai-models/
    print("Step 1: Moving ai-models/* buckets")
    for src, dest in [
        ("prompt_injection", "ai-models/prompt-injection"),
        ("jailbreak", "ai-models/jailbreaking"),
    ]:
        move_bucket(src, dest)
    print()

    # Step 2: Move metasploit, infection_monkey, rta → tools/
    print("Step 2: Moving tools/* buckets")
    for src, dest in [
        ("metasploit", "tools/metasploit"),
        ("infection_monkey", "tools/infection_monkey"),
        ("rta", "tools/rta"),
    ]:
        move_bucket(src, dest)
    print()

    # Step 3: Merge atomic_red_team + caldera into MITRE tactic buckets
    print("Step 3: Merging atomic_red_team + caldera into MITRE tactic buckets")
    merged = merge_attack_based_buckets(TO_MERGE, dry_run=False)
    print()

    # Step 4: Remove the now-empty source buckets
    print("Step 4: Removing empty source buckets (atomic_red_team, caldera_plugins)")
    for bn in TO_MERGE:
        bp = BUCKETS_DIR / bn
        if bp.exists():
            shutil.rmtree(bp)
            print(f"  Removed: {bn}/")
    print()

    # Step 5: Update metadata for moved buckets
    print("Step 5: Updating metadata for moved buckets")
    update_metadata_for_moved_buckets()
    print()

    # Step 6: Rebuild manifest
    print("Step 6: Rebuilding manifest.json")
    manifest = rebuild_manifest()
    print(f"  Total buckets: {manifest['total_buckets']}")
    print(f"  Total pairs:   {manifest['total_pairs']:,}")
    print()

    # Final summary
    print("=" * 70)
    print("FINAL LAYOUT")
    print("=" * 70)
    by_category = {}
    for b in manifest["buckets"]:
        cat = b.get("category", "?")
        by_category.setdefault(cat, []).append(b)
    for cat, blist in by_category.items():
        n = sum(b["count"] for b in blist)
        print(f"  [{cat}]")
        for b in blist:
            extra = f" ({b['mitre_tactic']})" if b.get("mitre_tactic") else ""
            print(f"    {b['path']:40s} {b['count']:>6,d} pairs{extra}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
