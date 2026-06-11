#!/usr/bin/env python3
"""Build HuggingFace-compatible dataset from AttackLM buckets.

Reads all bucket data.jsonl files, enriches each record with bucket-level
metadata (mitre_ids, source, license, bucket path, category), creates a
stratified 90/10 train/test split, and writes the result as JSONL files.

Usage:
    python hf/scripts/build_hf_dataset.py --output hf/data
    python hf/scripts/build_hf_dataset.py --output hf/data --seed 42
    python hf/scripts/build_hf_dataset.py --output hf/data --split-ratio 0.9
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Source-to-license mapping for bucket enrichment
# ---------------------------------------------------------------------------

# Maps bucket paths to their upstream source name(s) and license(s).
# When a bucket has multiple upstream sources, we use the most prominent one
# and note "mixed" in the license field if they differ.

BUCKET_SOURCES: dict[str, dict[str, str]] = {
    # --- MITRE tactic buckets ---
    "base/collection": {"source": "atomic-red-team", "license": "MIT"},
    "base/command_and_control": {"source": "atomic-red-team", "license": "MIT"},
    "base/credential_access": {"source": "atomic-red-team", "license": "MIT"},
    "base/defense_evasion": {"source": "atomic-red-team", "license": "MIT"},
    "base/discovery": {"source": "atomic-red-team", "license": "MIT"},
    "base/execution": {"source": "atomic-red-team", "license": "MIT"},
    "base/exfiltration": {"source": "atomic-red-team", "license": "MIT"},
    "base/lateral_movement": {"source": "atomic-red-team", "license": "MIT"},
    "base/persistence": {"source": "atomic-red-team", "license": "MIT"},
    "base/privilege_escalation": {"source": "atomic-red-team", "license": "MIT"},
    # --- Tool buckets ---
    "tools/metasploit": {"source": "metasploit", "license": "BSD-3-Clause"},
    "tools/rta": {"source": "rta", "license": "AGPL-3.0"},
    "tools/infection_monkey": {"source": "infection_monkey", "license": "GPL-3.0"},
    # --- AI security buckets ---
    "ai/prompt-injection": {"source": "promptfoo", "license": "MIT"},
    "ai/jailbreaking": {"source": "garak", "license": "Apache-2.0"},
    # --- Orchestrator ---
    "orchestrator": {"source": "synthetic", "license": "MIT"},
    # --- Extended category buckets ---
    "attack_tactics/red_team_tactics": {"source": "synthetic", "license": "MIT"},
    "cloud/attacks": {"source": "synthetic", "license": "MIT"},
    "ics/attacks": {"source": "synthetic", "license": "MIT"},
    "wireless/attacks": {"source": "synthetic", "license": "MIT"},
    "supply_chain/attacks": {"source": "synthetic", "license": "MIT"},
    "web_app/attacks": {"source": "synthetic", "license": "MIT"},
    "social_engineering/phishing": {"source": "synthetic", "license": "MIT"},
}

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def find_buckets(buckets_dir: Path) -> list[tuple[Path, Path]]:
    """Discover all bucket directories that contain data.jsonl + metadata.json.

    Returns list of (data_path, metadata_path) tuples.
    """
    result: list[tuple[Path, Path]] = []
    for data_path in sorted(buckets_dir.rglob("data.jsonl")):
        meta_path = data_path.parent / "metadata.json"
        if meta_path.exists():
            result.append((data_path, meta_path))
        else:
            print(f"  [WARN] No metadata.json for {data_path.parent}", file=sys.stderr)
    return result


def load_bucket(data_path: Path, meta_path: Path) -> tuple[str, dict, list[dict]]:
    """Load a single bucket's data and metadata.

    Returns (bucket_path, metadata, records).
    """
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)

    records: list[dict] = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    # Derive the bucket path relative to the buckets directory.
    # e.g., data/datasets/buckets/base/collection/data.jsonl -> base/collection
    # Walk up from data_path until we find the "buckets" parent.
    bucket_path = _derive_bucket_path(data_path)
    return bucket_path, metadata, records


def _derive_bucket_path(data_path: Path) -> str:
    """Derive the canonical bucket path from a data.jsonl file path.

    Handles paths like:
      .../buckets/base/collection/data.jsonl     -> base/collection
      .../buckets/ai/jailbreaking/data.jsonl       -> ai/jailbreaking
      .../buckets/tools/metasploit/data.jsonl     -> tools/metasploit
    """
    # The bucket directory is the parent of data.jsonl
    bucket_dir = data_path.parent
    # Find the "buckets" parent
    parts = bucket_dir.parts
    try:
        buckets_idx = len(parts) - 1 - parts[::-1].index("buckets")
        relative_parts = parts[buckets_idx + 1 :]
        return "/".join(relative_parts) if relative_parts else bucket_dir.name
    except ValueError:
        # No "buckets" in path — just use the directory name
        return bucket_dir.name


def enrich_record(record: dict, bucket_path: str, metadata: dict) -> dict:
    """Enrich a record with bucket-level metadata.

    Adds: mitre_ids, source, license, bucket, category.
    Preserves existing fields if they already exist on the record.
    """
    source_info = BUCKET_SOURCES.get(
        bucket_path, {"source": "unknown", "license": "unknown"}
    )

    enriched = dict(record)  # shallow copy

    # mitre_ids: prefer record-level, fall back to metadata-level
    if "mitre_ids" not in enriched or not enriched["mitre_ids"]:
        enriched["mitre_ids"] = metadata.get("mitre_ids", [])

    # source and license: prefer record-level, fall back to bucket mapping
    enriched.setdefault("source", source_info["source"])
    enriched.setdefault("license", source_info["license"])

    # Always add bucket path and category
    enriched["bucket"] = bucket_path
    enriched["category"] = metadata.get("category", "unknown")

    return enriched


def stratified_split(
    records_by_bucket: dict[str, list[dict]],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Create a stratified train/test split by bucket.

    Each bucket contributes train_ratio of its records to train,
    and (1 - train_ratio) to test. This ensures every category is
    represented in both splits.

    Returns (train_records, test_records).
    """
    rng = random.Random(seed)
    train: list[dict] = []
    test: list[dict] = []

    for bucket_path, records in sorted(records_by_bucket.items()):
        # Shuffle within each bucket for randomness
        shuffled = list(records)
        rng.shuffle(shuffled)

        split_idx = int(len(shuffled) * train_ratio)
        train.extend(shuffled[:split_idx])
        test.extend(shuffled[split_idx:])

    # Shuffle the final splits
    rng.shuffle(train)
    rng.shuffle(test)

    return train, test


def compute_statistics(
    train: list[dict],
    test: list[dict],
    records_by_bucket: dict[str, list[dict]],
) -> dict:
    """Compute dataset statistics for reporting."""
    all_records = train + test
    buckets = defaultdict(lambda: {"train": 0, "test": 0, "total": 0})

    for record in train:
        b = record.get("bucket", "unknown")
        buckets[b]["train"] += 1
        buckets[b]["total"] += 1

    for record in test:
        b = record.get("bucket", "unknown")
        buckets[b]["test"] += 1
        buckets[b]["total"] += 1

    # Count by source
    sources = defaultdict(int)
    for r in all_records:
        sources[r.get("source", "unknown")] += 1

    # Count by license
    licenses = defaultdict(int)
    for r in all_records:
        licenses[r.get("license", "unknown")] += 1

    # Count by category
    categories = defaultdict(int)
    for r in all_records:
        categories[r.get("category", "unknown")] += 1

    # Count unique MITRE technique IDs
    mitre_ids: set[str] = set()
    for r in all_records:
        for mid in r.get("mitre_ids", []):
            mitre_ids.add(mid)

    return {
        "total": len(all_records),
        "train": len(train),
        "test": len(test),
        "train_ratio": len(train) / len(all_records) if all_records else 0,
        "num_buckets": len(buckets),
        "num_mitre_techniques": len(mitre_ids),
        "buckets": dict(buckets),
        "sources": dict(sources),
        "licenses": dict(licenses),
        "categories": dict(categories),
    }


def write_jsonl(records: list[dict], path: Path) -> None:
    """Write records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build HuggingFace-compatible dataset from AttackLM buckets."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hf/data"),
        help="Output directory for train/test JSONL files (default: hf/data)",
    )
    parser.add_argument(
        "--buckets-dir",
        type=Path,
        default=Path("data/datasets/buckets"),
        help="Path to buckets directory (default: data/datasets/buckets)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.9,
        help="Train split ratio (default: 0.9 for 90/10 split)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics without writing files",
    )
    args = parser.parse_args()

    # Resolve paths relative to the project root
    project_root = Path(__file__).resolve().parent.parent.parent
    buckets_dir = (project_root / args.buckets_dir).resolve()
    output_dir = (project_root / args.output).resolve()

    if not buckets_dir.exists():
        print(f"ERROR: Buckets directory not found: {buckets_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover and load all buckets
    print(f"Scanning buckets in {buckets_dir}...")
    bucket_files = find_buckets(buckets_dir)

    if not bucket_files:
        print("ERROR: No bucket directories found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(bucket_files)} bucket(s):\n")

    records_by_bucket: dict[str, list[dict]] = {}

    for data_path, meta_path in bucket_files:
        bucket_path, metadata, records = load_bucket(data_path, meta_path)
        count = metadata.get("count", len(records))
        actual_count = len(records)
        category = metadata.get("category", "unknown")
        mitre_tactic = metadata.get("mitre_tactic", "n/a")

        print(
            f"  {bucket_path:40s}  {actual_count:5d} records  "
            f"(metadata says {count})  "
            f"category={category}  tactic={mitre_tactic}"
        )

        # Warn if count mismatch
        if actual_count != count:
            print(
                f"    [WARN] Metadata count ({count}) != actual records ({actual_count})",
                file=sys.stderr,
            )

        # Enrich each record with bucket metadata
        enriched = [enrich_record(r, bucket_path, metadata) for r in records]
        records_by_bucket[bucket_path] = enriched

    # Create stratified train/test split
    print(
        f"\nCreating {args.split_ratio:.0%}/{1 - args.split_ratio:.0%} "
        f"train/test split (seed={args.seed})..."
    )
    train, test = stratified_split(
        records_by_bucket,
        train_ratio=args.split_ratio,
        seed=args.seed,
    )

    # Compute and print statistics
    stats = compute_statistics(train, test, records_by_bucket)

    print(f"\n{'=' * 60}")
    print(f"AttackLM Dataset Statistics")
    print(f"{'=' * 60}")
    print(f"Total records:     {stats['total']:,}")
    print(f"Train records:     {stats['train']:,} ({stats['train_ratio']:.1%})")
    print(f"Test records:      {stats['test']:,} ({1 - stats['train_ratio']:.1%})")
    print(f"Num buckets:       {stats['num_buckets']}")
    print(f"MITRE techniques:  {stats['num_mitre_techniques']}")

    print(f"\n--- Per-bucket split ---")
    for bucket, counts in sorted(
        stats["buckets"].items(), key=lambda x: -x[1]["total"]
    ):
        print(
            f"  {bucket:40s}  train={counts['train']:5d}  test={counts['test']:4d}  "
            f"total={counts['total']:5d}"
        )

    print(f"\n--- By source ---")
    for source, count in sorted(stats["sources"].items(), key=lambda x: -x[1]):
        print(f"  {source:30s}  {count:5d}")

    print(f"\n--- By license ---")
    for license_name, count in sorted(stats["licenses"].items(), key=lambda x: -x[1]):
        print(f"  {license_name:20s}  {count:5d}")

    print(f"\n--- By category ---")
    for cat, count in sorted(stats["categories"].items(), key=lambda x: -x[1]):
        print(f"  {cat:20s}  {count:5d}")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Write output files
    train_path = output_dir / "attacklm-train.jsonl"
    test_path = output_dir / "attacklm-test.jsonl"

    print(f"\nWriting train split to {train_path}...")
    write_jsonl(train, train_path)

    print(f"Writing test split to {test_path}...")
    write_jsonl(test, test_path)

    # Write a manifest with statistics
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "version": "0.3.0",
        "total_records": stats["total"],
        "train_records": stats["train"],
        "test_records": stats["test"],
        "train_ratio": args.split_ratio,
        "seed": args.seed,
        "num_buckets": stats["num_buckets"],
        "num_mitre_techniques": stats["num_mitre_techniques"],
        "buckets": stats["buckets"],
        "sources": stats["sources"],
        "licenses": stats["licenses"],
        "categories": stats["categories"],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Manifest written to {manifest_path}")

    print(
        f"\nDone! {stats['train']:,} train + {stats['test']:,} test = {stats['total']:,} total records"
    )


if __name__ == "__main__":
    main()
