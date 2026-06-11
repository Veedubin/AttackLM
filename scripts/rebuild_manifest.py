#!/usr/bin/env python3
"""
rebuild_manifest.py — Rebuild the AttackLM bucket manifest from disk.

Walks data/datasets/buckets/ recursively, discovers all metadata.json files,
validates and auto-populates missing fields, sorts buckets by category priority,
and writes a new manifest.json with version 4.

Provenance-aware: discovers per-bucket sub-files (data_human.jsonl,
data_llm.jsonl, data_synth.jsonl) and sums counts. Also supports legacy
data.jsonl buckets (not yet migrated to three-tier provenance).

Usage:
    python scripts/rebuild_manifest.py
    python scripts/rebuild_manifest.py --dry-run   # Preview without writing
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
BUCKETS_DIR = PROJECT_DIR / "data" / "datasets" / "buckets"
MANIFEST_PATH = BUCKETS_DIR / "manifest.json"

# ---------------------------------------------------------------------------
# Required metadata fields
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = [
    "name",
    "display_name",
    "category",
    "count",
    "description",
    "sub_sources",
]

# ---------------------------------------------------------------------------
# Sort priority: tactics first, then tools, then ai, then new categories,
# then orchestrator (meta) last
# ---------------------------------------------------------------------------
CATEGORY_ORDER = [
    "tactic",
    "tools",
    "ai_redteam",
    "attack_tactics",
    "web_app",
    "cloud",
    "social_engineering",
    "supply_chain",
    "ics",
    "wireless",
    "meta",
]

# Map category to sort index (lower = earlier)
CATEGORY_INDEX = {cat: i for i, cat in enumerate(CATEGORY_ORDER)}


def count_jsonl_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    if not path.exists():
        return 0
    count = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def discover_buckets() -> list[dict]:
    """Walk buckets dir, find all metadata.json files, and build bucket entries."""
    buckets: list[dict] = []

    # Sub-file names for three-tier provenance
    SUB_FILES = {
        "human": "data_human.jsonl",
        "llm": "data_llm.jsonl",
        "synth": "data_synth.jsonl",
    }

    for metadata_path in sorted(BUCKETS_DIR.rglob("metadata.json")):
        # Compute relative path from BUCKETS_DIR to the metadata's parent dir
        bucket_dir = metadata_path.parent
        rel_path = bucket_dir.relative_to(BUCKETS_DIR)
        rel_path_str = str(rel_path)

        # Skip if metadata is at the root (shouldn't happen)
        if rel_path_str == ".":
            print(f"  SKIP: metadata.json at root — {metadata_path}", file=sys.stderr)
            continue

        # Load metadata
        try:
            with open(metadata_path, encoding="utf-8") as fh:
                metadata = json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"  ERROR: Invalid JSON in {metadata_path}: {exc}", file=sys.stderr)
            continue

        # Validate and auto-populate missing fields
        bucket_name = metadata.get("name", bucket_dir.name)
        display_name = metadata.get(
            "display_name", bucket_dir.name.replace("_", " ").title()
        )

        # Derive category from parent directory or metadata
        category = metadata.get("category", "")
        if not category:
            # Derive from parent directory structure
            parts = rel_path.parts
            if len(parts) >= 2:
                parent_dir = parts[0]
                category_map = {
                    "base": "tactic",
                    "ai": "ai_redteam",
                    "tools": "tools",
                    "orchestrator": "meta",
                    "attack_tactics": "attack_tactics",
                    "web_app": "web_app",
                    "cloud": "cloud",
                    "social_engineering": "social_engineering",
                    "supply_chain": "supply_chain",
                    "ics": "ics",
                    "wireless": "wireless",
                }
                category = category_map.get(parent_dir, parent_dir)

        # Discover per-tier sub-files and count lines
        sub_sources: dict[str, int] = {"human": 0, "llm": 0, "synth": 0}
        source_files: dict[str, str] = {}
        found_any_sub = False

        for tier, filename in SUB_FILES.items():
            tier_path = bucket_dir / filename
            if tier_path.exists():
                tier_count = count_jsonl_lines(tier_path)
                sub_sources[tier] = tier_count
                source_files[tier] = filename
                if tier_count > 0:
                    found_any_sub = True

        # Also check for legacy data.jsonl (not yet migrated)
        legacy_path = bucket_dir / "data.jsonl"
        legacy_count = count_jsonl_lines(legacy_path) if legacy_path.exists() else 0

        if found_any_sub:
            # Use three-tier sub-files as source of truth
            count = sum(sub_sources.values())
            # Override metadata count with actual line count
            metadata_count = metadata.get("count", 0)
            if metadata_count != count:
                print(
                    f"  COUNT CORRECTED for {rel_path_str}: "
                    f"metadata={metadata_count} -> actual(sub_files)={count}"
                )
        elif legacy_count > 0:
            # Legacy bucket — treat all records as synth for now
            # (migrate_legacy_buckets.py will classify them later)
            count = legacy_count
            sub_sources = {"human": 0, "llm": 0, "synth": legacy_count}
            source_files = {"synth": "data.jsonl"}
            metadata_count = metadata.get("count", 0)
            if metadata_count != count:
                print(
                    f"  COUNT CORRECTED for {rel_path_str}: "
                    f"metadata={metadata_count} -> actual(legacy)={count}"
                )
        else:
            # No data files found
            count = metadata.get("count", 0)
            if count > 0:
                print(
                    f"  WARNING: No data files for {rel_path_str}, using metadata count",
                    file=sys.stderr,
                )

        # Build bucket entry, preserving all existing metadata fields
        entry: dict = {
            "name": bucket_name,
            "display_name": metadata.get("display_name", display_name),
            "category": category,
            "count": count,
            "description": metadata.get("description", ""),
            "path": rel_path_str,
            "sub_sources": sub_sources,
        }

        # Preserve optional fields
        for key in (
            "mitre_tactic",
            "mitre_ids",
            "source_file",
            "source_files",
            "created",
        ):
            if key in metadata:
                entry[key] = metadata[key]

        # Set source_files from discovered sub-files if not already set
        if "source_files" not in entry and source_files:
            entry["source_files"] = source_files

        # Check for missing required fields and report
        missing = [f for f in REQUIRED_FIELDS if not entry.get(f)]
        if missing:
            print(
                f"  WARNING: {rel_path_str} missing required fields: {missing}",
                file=sys.stderr,
            )

        buckets.append(entry)

    return buckets


def sort_buckets(buckets: list[dict]) -> list[dict]:
    """Sort buckets: tactics first, then tools, then ai, then new categories, then orchestrator."""

    def sort_key(bucket: dict) -> tuple[int, str]:
        category = bucket.get("category", "zzz")
        cat_idx = CATEGORY_INDEX.get(category, len(CATEGORY_ORDER))
        name = bucket.get("name", "")
        return (cat_idx, name)

    return sorted(buckets, key=sort_key)


def compute_summary(buckets: list[dict]) -> dict:
    """Compute summary statistics from bucket entries."""
    total_buckets = len(buckets)
    total_pairs = sum(b["count"] for b in buckets)

    # Per-category counts
    category_counts: dict[str, dict] = {}
    for bucket in buckets:
        cat = bucket.get("category", "unknown")
        if cat not in category_counts:
            category_counts[cat] = {"buckets": 0, "pairs": 0}
        category_counts[cat]["buckets"] += 1
        category_counts[cat]["pairs"] += bucket["count"]

    # Per-tier counts (provenance summary)
    tier_totals: dict[str, int] = {"human": 0, "llm": 0, "synth": 0}
    for bucket in buckets:
        sub = bucket.get("sub_sources", {})
        for tier in tier_totals:
            tier_totals[tier] += sub.get(tier, 0)

    return {
        "total_buckets": total_buckets,
        "total_pairs": total_pairs,
        "category_counts": category_counts,
        "tier_totals": tier_totals,
    }


def build_manifest(buckets: list[dict]) -> dict:
    """Build the full manifest structure."""
    summary = compute_summary(buckets)

    return {
        "version": 4,
        "layout": "nested (ai/, attack_tactics/, base/, cloud/, ics/, orchestrator/, social_engineering/, supply_chain/, tools/, web_app/, wireless/)",
        "provenance": "three-tier (human/llm/synth)",
        "created": datetime.now(timezone.utc).isoformat(),
        "total_buckets": summary["total_buckets"],
        "total_pairs": summary["total_pairs"],
        "tier_totals": summary["tier_totals"],
        "buckets": buckets,
    }


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    print("=" * 72)
    print("AttackLM Manifest Rebuilder")
    print("=" * 72)
    print(f"  Buckets dir: {BUCKETS_DIR}")
    print(f"  Manifest:    {MANIFEST_PATH}")
    print(f"  Dry run:     {dry_run}")
    print()

    # Step 1: Discover all buckets
    print("Step 1: Discovering buckets...")
    buckets = discover_buckets()
    print(f"  Found {len(buckets)} buckets")
    print()

    # Step 2: Sort buckets
    print("Step 2: Sorting buckets by category priority...")
    buckets = sort_buckets(buckets)
    print()

    # Step 3: Compute summary
    print("Step 3: Computing summary...")
    summary = compute_summary(buckets)
    print()

    # Step 4: Print per-category breakdown
    print("=" * 72)
    print("BUCKET SUMMARY")
    print("=" * 72)
    print(f"{'Category':<22} {'Buckets':>8} {'Pairs':>10} {'% of Total':>12}")
    print("-" * 72)

    total_pairs = summary["total_pairs"]
    for cat, info in sorted(
        summary["category_counts"].items(), key=lambda x: -x[1]["pairs"]
    ):
        pct = (info["pairs"] / total_pairs * 100) if total_pairs > 0 else 0
        print(f"{cat:<22} {info['buckets']:>8} {info['pairs']:>10} {pct:>11.1f}%")

    print("-" * 72)
    print(
        f"{'TOTAL':<22} {summary['total_buckets']:>8} {summary['total_pairs']:>10} {'100.0%':>12}"
    )
    print()

    # Step 5: Print bucket details
    print("=" * 72)
    print("BUCKET DETAILS")
    print("=" * 72)
    print(f"{'#':<4} {'Path':<38} {'Category':<18} {'Count':>7}")
    print("-" * 72)
    for i, bucket in enumerate(buckets, 1):
        print(
            f"{i:<4} {bucket['path']:<38} {bucket['category']:<18} {bucket['count']:>7}"
        )
    print()

    # Step 6: Validation checks
    print("=" * 72)
    print("VALIDATION CHECKS")
    print("=" * 72)

    errors = []

    # Check total buckets (v3 had 23, v4 may differ after migration)
    print(f"  [INFO] Total buckets: {summary['total_buckets']}")

    # Check total pairs
    print(f"  [INFO] Total pairs: {summary['total_pairs']:,}")
    print(
        f"         human={summary['tier_totals']['human']:,}  "
        f"llm={summary['tier_totals']['llm']:,}  "
        f"synth={summary['tier_totals']['synth']:,}"
    )

    # Check no zero-count buckets
    zero_count = [b for b in buckets if b["count"] == 0]
    if zero_count:
        for b in zero_count:
            errors.append(f"Bucket '{b['path']}' has count=0")
        print(f"  [FAIL] {len(zero_count)} buckets with count=0")
    else:
        print("  [OK] No buckets with count=0")

    # Check each bucket has at least one data file and metadata.json on disk
    missing_files = []
    for bucket in buckets:
        bucket_dir = BUCKETS_DIR / bucket["path"]
        has_sub = any(
            (bucket_dir / f"data_{tier}.jsonl").exists()
            for tier in ("human", "llm", "synth")
        )
        has_legacy = (bucket_dir / "data.jsonl").exists()
        if not has_sub and not has_legacy:
            missing_files.append(f"{bucket['path']}/data_*.jsonl")
        if not (bucket_dir / "metadata.json").exists():
            missing_files.append(f"{bucket['path']}/metadata.json")

    if missing_files:
        for mf in missing_files:
            errors.append(f"Missing file: {mf}")
        print(f"  [FAIL] {len(missing_files)} missing files")
    else:
        print("  [OK] All buckets have data files and metadata.json")

    # Check required fields
    missing_fields = []
    for bucket in buckets:
        for field in REQUIRED_FIELDS:
            if not bucket.get(field):
                missing_fields.append(f"{bucket['path']}.{field}")

    if missing_fields:
        for mf in missing_fields:
            errors.append(f"Missing required field: {mf}")
        print(f"  [FAIL] {len(missing_fields)} missing required fields")
    else:
        print("  [OK] All required fields present")

    print()

    # Step 7: Write manifest
    if errors:
        print("ERRORS DETECTED:")
        for error in errors:
            print(f"  - {error}")
        print()
        if not dry_run:
            print("Writing manifest anyway (with errors)...")
        else:
            print("Dry run — not writing manifest.")

    if not dry_run:
        manifest = build_manifest(buckets)
        with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
        print(f"Manifest written to: {MANIFEST_PATH}")
        print(f"  Version: {manifest['version']}")
        print(f"  Total buckets: {manifest['total_buckets']}")
        print(f"  Total pairs: {manifest['total_pairs']:,}")
    else:
        print("Dry run — manifest NOT written.")

    print()
    print("=" * 72)
    print("BEFORE vs AFTER BALANCE")
    print("=" * 72)
    print("  BEFORE: 16 buckets, 16,982 pairs")
    print("          metasploit = 8,349 (49.2%)")
    print()
    print(
        f"  AFTER:  {summary['total_buckets']} buckets, {summary['total_pairs']:,} pairs"
    )
    metasploit_count = next(
        (b["count"] for b in buckets if b["name"] == "metasploit"), 0
    )
    if metasploit_count and total_pairs:
        metasploit_pct = metasploit_count / total_pairs * 100
        print(f"          metasploit = {metasploit_count:,} ({metasploit_pct:.1f}%)")
    print()
    print("  New categories added:")
    for cat in [
        "attack_tactics",
        "web_app",
        "cloud",
        "social_engineering",
        "supply_chain",
        "ics",
        "wireless",
    ]:
        info = summary["category_counts"].get(cat)
        if info:
            print(f"    {cat}: {info['pairs']} pairs in {info['buckets']} bucket(s)")
        else:
            print(f"    {cat}: NOT FOUND")
    print()

    if errors:
        print("Completed with ERRORS. See above.")
        sys.exit(1)
    else:
        print("Completed successfully.")


if __name__ == "__main__":
    main()
