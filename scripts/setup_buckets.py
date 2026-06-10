#!/usr/bin/env python3
"""Organize training data into bucket directories.

Bucket layout:
    data/datasets/buckets/
        manifest.json                       # Master manifest of all buckets
        <bucket_name>/
            data.jsonl                      # All training pairs for this bucket
            metadata.json                   # Per-bucket metadata (tactic, source, count)

Each bucket represents a distinct training domain (one MITRE tactic, the
orchestrator, or prompt injection). This makes the data layout explicit
and lets train_all.py auto-discover what to train on.

Usage:
    uv run python scripts/setup_buckets.py              # Create/migrate buckets
    uv run python scripts/setup_buckets.py --clean       # Remove old flat files after migration
    uv run python scripts/setup_buckets.py --status      # Show current bucket state
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("data/datasets")
BUCKETS_DIR = DATA_DIR / "buckets"

# Mapping of existing flat files → bucket names + metadata
# Format: (source_file, bucket_name, display_name, category, mitre_tactic, description)
BUCKET_DEFINITIONS = [
    # === MITRE ATT&CK Enterprise Tactics (9 buckets) ===
    (
        "collection_dataset.jsonl",
        "collection",
        "Collection",
        "tactic",
        "TA0009",
        "Data collection from target systems (T1005, T1213, T1114, T1119)",
    ),
    (
        "command_and_control_dataset.jsonl",
        "command_and_control",
        "Command and Control",
        "tactic",
        "TA0011",
        "C2 channel establishment (T1071, T1090, T1095, T1572, T1573)",
    ),
    (
        "credential_access_dataset.jsonl",
        "credential_access",
        "Credential Access",
        "tactic",
        "TA0006",
        "Credential theft and brute force (T1003, T1110, T1555, T1552)",
    ),
    (
        "defense_evasion_dataset.jsonl",
        "defense_evasion",
        "Defense Evasion",
        "tactic",
        "TA0005",
        "AV/EDR bypass, obfuscation, masquerading (T1027, T1055, T1562, T1070)",
    ),
    (
        "discovery_dataset.jsonl",
        "discovery",
        "Discovery",
        "tactic",
        "TA0007",
        "System/network enumeration (T1057, T1082, T1087, T1018, T1049)",
    ),
    (
        "execution_dataset.jsonl",
        "execution",
        "Execution",
        "tactic",
        "TA0002",
        "Code execution techniques (T1059, T1106, T1204, T1047)",
    ),
    (
        "exfiltration_dataset.jsonl",
        "exfiltration",
        "Exfiltration",
        "tactic",
        "TA0010",
        "Data exfiltration over C2 (T1041, T1567, T1052, T1011)",
    ),
    (
        "lateral_movement_dataset.jsonl",
        "lateral_movement",
        "Lateral Movement",
        "tactic",
        "TA0008",
        "Remote service exploitation (T1021, T1570, T1080, T1559)",
    ),
    (
        "persistence_dataset.jsonl",
        "persistence",
        "Persistence",
        "tactic",
        "TA0003",
        "Maintain foothold (T1543, T1547, T1136, T1098, T1197)",
    ),
    (
        "privilege_escalation_dataset.jsonl",
        "privilege_escalation",
        "Privilege Escalation",
        "tactic",
        "TA0004",
        "Elevate privileges (T1068, T1548, T1134, T1543)",
    ),
    # === Metasploit (consolidated as one bucket — too many sub-modules) ===
    # Skip individual metasploit_* files; consolidate into one "metasploit" bucket.
    # Will be handled in migrate step by globbing metasploit_*_dataset.jsonl.
    # === AI/ML red team (NEW from this session) ===
    (
        "atomic_red_team_dataset.jsonl",
        "atomic_red_team",
        "Atomic Red Team",
        "tooling",
        "various",
        "Red Canary Atomic Red Team — 2,506 ATT&CK test cases",
    ),
    (
        "caldera_plugins_dataset.jsonl",
        "caldera_plugins",
        "Caldera Plugins",
        "tooling",
        "various",
        "MITRE Caldera Stockpile + Arsenal + Manx + Access — 664 abilities",
    ),
    (
        "rta_dataset.jsonl",
        "rta",
        "RTA",
        "tooling",
        "various",
        "endgameinc/RTA — 40 Python TTP scripts",
    ),
    (
        "infection_monkey_dataset.jsonl",
        "infection_monkey",
        "Infection Monkey",
        "tooling",
        "various",
        "guardicore/infection_monkey — 36 plugin manifests",
    ),
    # === Jailbreak / prompt injection (sandbox for AI-specific) ===
    (
        "jailbreak_dataset.jsonl",
        "jailbreak",
        "Jailbreak",
        "ai_redteam",
        "TA0040",
        "AI jailbreak patterns (custom TA0040 extension)",
    ),
    # === Orchestrator + Prompt Injection (always optional, included via flags) ===
    (
        "orchestrator_dataset.jsonl",
        "orchestrator",
        "Orchestrator Routing",
        "meta",
        "n/a",
        "Agent routing decisions — 380 examples across 6 agents",
    ),
    (
        "prompt_injection_dataset.jsonl",
        "prompt_injection",
        "Prompt Injection",
        "ai_redteam",
        "TA0040",
        "Prompt injection & AI exploit — 687 examples",
    ),
]


def migrate_bucket(
    src: Path,
    bucket_name: str,
    display_name: str,
    category: str,
    mitre_tactic: str,
    description: str,
) -> dict:
    """Migrate a single source file into a bucket directory.

    Returns the bucket metadata dict.
    """
    bucket_dir = BUCKETS_DIR / bucket_name
    bucket_dir.mkdir(parents=True, exist_ok=True)
    dest = bucket_dir / "data.jsonl"

    if not src.exists():
        # Bucket has no data (e.g., orchestrator not yet generated)
        return {
            "name": bucket_name,
            "display_name": display_name,
            "category": category,
            "mitre_tactic": mitre_tactic,
            "description": description,
            "count": 0,
            "source_file": None,
            "created": datetime.now().isoformat(),
        }

    # Copy the file
    shutil.copy2(src, dest)

    # Count examples
    count = sum(1 for line in open(dest) if line.strip())

    return {
        "name": bucket_name,
        "display_name": display_name,
        "category": category,
        "mitre_tactic": mitre_tactic,
        "description": description,
        "count": count,
        "source_file": src.name,
        "created": datetime.now().isoformat(),
    }


def consolidate_metasploit() -> dict:
    """Consolidate all metasploit_*_dataset.jsonl into one 'metasploit' bucket."""
    bucket_dir = BUCKETS_DIR / "metasploit"
    bucket_dir.mkdir(parents=True, exist_ok=True)
    dest = bucket_dir / "data.jsonl"

    metasploit_files = sorted(DATA_DIR.glob("metasploit_*_dataset.jsonl"))
    # Exclude the combined file (it's already a duplicate of the others)
    metasploit_files = [
        f for f in metasploit_files if f.name != "metasploit_combined_dataset.jsonl"
    ]

    if metasploit_files:
        with open(dest, "w") as out_f:
            for f in metasploit_files:
                with open(f) as in_f:
                    out_f.write(in_f.read())
        count = sum(1 for line in open(dest) if line.strip())
    else:
        count = 0

    return {
        "name": "metasploit",
        "display_name": "Metasploit Modules",
        "category": "tooling",
        "mitre_tactic": "various",
        "description": f"Rapid7 Metasploit — {len(metasploit_files)} module categories consolidated",
        "count": count,
        "source_files": [f.name for f in metasploit_files],
        "created": datetime.now().isoformat(),
    }


def show_status():
    """Print current bucket state."""
    manifest_path = BUCKETS_DIR / "manifest.json"
    if not manifest_path.exists():
        print("No manifest.json found. Run setup_buckets.py first.")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    print("=" * 70)
    print("BUCKET STATUS")
    print("=" * 70)
    print(f"{'Bucket':25s} {'Category':12s} {'MITRE':10s} {'Count':>8s}  Data file")
    print("-" * 70)
    total = 0
    for b in manifest["buckets"]:
        path = BUCKETS_DIR / b["name"] / "data.jsonl"
        exists = "OK" if path.exists() and b["count"] > 0 else "EMPTY"
        print(
            f"  {b['name']:23s} {b['category']:12s} {b['mitre_tactic']:10s} {b['count']:>8,d}  [{exists}]"
        )
        total += b["count"]
    print("-" * 70)
    print(f"  {'TOTAL':23s} {'':12s} {'':10s} {total:>8,d}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove old flat files after successful migration",
    )
    parser.add_argument(
        "--status", action="store_true", help="Show current bucket state and exit"
    )
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    print("=" * 70)
    print("MIGRATING DATA INTO BUCKETS")
    print("=" * 70)
    print()

    BUCKETS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Standard tactic + tool buckets
    buckets_meta = []
    for (
        src_name,
        bucket_name,
        display_name,
        category,
        mitre_tactic,
        description,
    ) in BUCKET_DEFINITIONS:
        src = DATA_DIR / src_name
        meta = migrate_bucket(
            src, bucket_name, display_name, category, mitre_tactic, description
        )
        status = (
            f"{meta['count']:>6,d} pairs"
            if meta["count"] > 0
            else "    (no source file)"
        )
        print(f"  [{meta['category']:11s}] {bucket_name:25s} {status}  ← {src_name}")
        buckets_meta.append(meta)

    # 2) Consolidate metasploit modules into one bucket
    ms_meta = consolidate_metasploit()
    print(
        f"  [{ms_meta['category']:11s}] {'metasploit':25s} {ms_meta['count']:>6,d} pairs  ← {len(ms_meta.get('source_files', []))} files"
    )
    buckets_meta.append(ms_meta)

    # 3) Write the master manifest
    manifest = {
        "version": 1,
        "created": datetime.now().isoformat(),
        "total_buckets": len(buckets_meta),
        "total_pairs": sum(b["count"] for b in buckets_meta),
        "buckets": buckets_meta,
    }
    manifest_path = BUCKETS_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print()
    print("=" * 70)
    print("BUCKET SUMMARY")
    print("=" * 70)
    by_category = {}
    for b in buckets_meta:
        by_category.setdefault(b["category"], []).append(b)
    for cat, blist in by_category.items():
        n = sum(b["count"] for b in blist)
        print(f"  {cat:12s} {len(blist):2d} buckets, {n:>6,d} pairs total")
    print(
        f"  {'ALL':12s} {len(buckets_meta):2d} buckets, {manifest['total_pairs']:>6,d} pairs total"
    )
    print()
    print(f"  Manifest: {manifest_path}")
    print(f"  Bucket data: {BUCKETS_DIR}/<bucket_name>/data.jsonl")

    # 4) Optionally clean up old flat files
    if args.clean:
        print()
        print("=" * 70)
        print("CLEANING UP OLD FLAT FILES")
        print("=" * 70)
        removed = 0
        for src_name, *_ in BUCKET_DEFINITIONS:
            src = DATA_DIR / src_name
            if src.exists():
                src.unlink()
                print(f"  Removed: {src_name}")
                removed += 1
        # Remove individual metasploit files (consolidated)
        for f in DATA_DIR.glob("metasploit_*_dataset.jsonl"):
            f.unlink()
            print(f"  Removed: {f.name}")
            removed += 1
        # Remove old combined files (no longer needed — rebuilt on demand)
        for f in DATA_DIR.glob("combined_*.jsonl"):
            f.unlink()
            print(f"  Removed: {f.name}")
            removed += 1
        # Remove old *training_pairs.jsonl (pre-rename artifacts)
        for f in DATA_DIR.glob("*_training_pairs.jsonl"):
            f.unlink()
            print(f"  Removed: {f.name}")
            removed += 1
        print()
        print(f"  Removed {removed} old files. New flat layout: 0 files in {DATA_DIR}")
    else:
        print()
        print("  Tip: run with --clean to remove the old flat files once verified.")


if __name__ == "__main__":
    main()
