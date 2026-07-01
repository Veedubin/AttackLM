#!/usr/bin/env python3
"""
package_dataset.py — Package the AttackLM dataset for distribution via GitHub releases.

Creates a tarball of the pre-processed dataset so users can run `attacklm init`
without needing to git clone upstream repos and run extractors.

Includes:
  - All sources under data/datasets/buckets/sources/ (18 public sources)
  - data/datasets/buckets/manifest.json
  - Excludes 3 restricted sources: RTA, infection_monkey, BPL

Usage:
    python scripts/package_dataset.py
    python scripts/package_dataset.py --output /tmp/attacklm-dataset.tar.gz
"""

import argparse
import io
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
BUCKETS_DIR = PROJECT_DIR / "data" / "datasets" / "buckets"
SOURCES_DIR = BUCKETS_DIR / "sources"
MANIFEST_PATH = BUCKETS_DIR / "manifest.json"

DEFAULT_OUTPUT = PROJECT_DIR / "dist" / "attacklm-dataset.tar.gz"

# Restricted sources that must NEVER appear in public releases.
# These live only in archive/restricted-sources/ (gitignored).
RESTRICTED_SOURCES = {"RTA", "infection_monkey", "BPL"}

# Directory/file names under sources/ that are not source directories.
_NON_SOURCE_ENTRIES = {"_index.json", "README.md"}


def count_jsonl_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    count = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def discover_sources() -> list[str]:
    """Return sorted list of public source directory names."""
    if not SOURCES_DIR.exists():
        print(f"ERROR: Sources directory not found: {SOURCES_DIR}", file=sys.stderr)
        sys.exit(1)

    sources = []
    for entry in sorted(SOURCES_DIR.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in RESTRICTED_SOURCES:
            continue
        if entry.name.startswith("_") or entry.name.startswith("."):
            continue
        sources.append(entry.name)
    return sources


def count_records(sources: list[str]) -> dict[str, int]:
    """Count records per source by scanning JSONL files."""
    counts: dict[str, int] = {}
    for src in sources:
        src_dir = SOURCES_DIR / src
        n = 0
        for jsonl in src_dir.rglob("*.jsonl"):
            n += count_jsonl_lines(jsonl)
        counts[src] = n
    return counts


def human_size(size_bytes: int) -> str:
    """Return human-readable file size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024  # type: ignore[assignment]
    return f"{size_bytes:.1f} TB"


def build_tarball(output_path: Path, sources: list[str]) -> dict:
    """Build the tarball and return metadata dict with stats."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_arcname = "data/datasets/buckets/manifest.json"
    total_files = 0
    total_bytes = 0
    source_file_counts: dict[str, int] = {}

    with tarfile.open(output_path, "w:gz") as tar:
        # --- Add manifest.json ---
        if not MANIFEST_PATH.exists():
            print(f"ERROR: manifest.json not found: {MANIFEST_PATH}", file=sys.stderr)
            sys.exit(1)

        tar.add(str(MANIFEST_PATH), arcname=manifest_arcname)
        total_files += 1
        total_bytes += MANIFEST_PATH.stat().st_size
        print(f"  Added: {manifest_arcname}")

        # --- Add each source directory ---
        for src in sources:
            src_dir = SOURCES_DIR / src
            src_prefix = f"data/datasets/buckets/sources/{src}"
            file_count = 0

            for filepath in sorted(src_dir.rglob("*")):
                if filepath.is_dir():
                    continue
                # Skip hidden files and non-data files
                if filepath.name.startswith("."):
                    continue

                arcname = f"{src_prefix}/{filepath.relative_to(src_dir)}"
                tar.add(str(filepath), arcname=arcname)
                total_files += 1
                total_bytes += filepath.stat().st_size
                file_count += 1

            source_file_counts[src] = file_count
            print(f"  Added: {src_prefix}/ ({file_count} files)")

    # Actual tarball size on disk
    tarball_size = output_path.stat().st_size

    return {
        "total_files": total_files,
        "total_source_bytes": total_bytes,
        "tarball_size": tarball_size,
        "source_file_counts": source_file_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package the AttackLM dataset for distribution via GitHub releases."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output tarball path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    output_path = args.output or DEFAULT_OUTPUT
    output_path = output_path.resolve()

    print("=" * 72)
    print("AttackLM Dataset Packager")
    print("=" * 72)
    print(f"  Sources dir:   {SOURCES_DIR}")
    print(f"  Manifest:      {MANIFEST_PATH}")
    print(f"  Output:         {output_path}")
    print(f"  Excluded:       {', '.join(sorted(RESTRICTED_SOURCES))}")
    print()

    # Discover public sources
    print("Step 1: Discovering public sources...")
    sources = discover_sources()
    print(f"  Found {len(sources)} public sources: {', '.join(sources)}")
    print()

    # Count records
    print("Step 2: Counting records per source...")
    record_counts = count_records(sources)
    total_records = sum(record_counts.values())
    for src, count in record_counts.items():
        print(f"  {src:<30s} {count:>8,} records")
    print(f"  {'TOTAL':<30s} {total_records:>8,} records")
    print()

    # Build tarball
    print("Step 3: Building tarball...")
    stats = build_tarball(output_path, sources)
    print()

    # Summary
    print("=" * 72)
    print("PACKAGING SUMMARY")
    print("=" * 72)
    print(f"  Output:           {output_path}")
    print(f"  Tarball size:     {human_size(stats['tarball_size'])}")
    print(f"  Source count:     {len(sources)}")
    print(f"  Total records:    {total_records:,}")
    print(f"  Total files:      {stats['total_files']}")
    print(f"  Uncompressed:     {human_size(stats['total_source_bytes'])}")
    print(
        f"  Compression:     {stats['total_source_bytes'] / stats['tarball_size']:.1f}x"
        if stats["tarball_size"] > 0
        else "  Compression:     N/A"
    )
    print()

    # Verify no restricted sources leaked in
    print("Step 4: Verifying no restricted sources in tarball...")
    restricted_found = []
    with tarfile.open(output_path, "r:gz") as tar:
        for member in tar.getmembers():
            for restricted in RESTRICTED_SOURCES:
                if f"/{restricted}/" in member.name or member.name.endswith(
                    f"/{restricted}"
                ):
                    restricted_found.append(member.name)
                    break
    if restricted_found:
        print(f"  ERROR: Restricted sources found in tarball:", file=sys.stderr)
        for name in restricted_found:
            print(f"    {name}", file=sys.stderr)
        sys.exit(1)
    else:
        print("  OK — no restricted sources in tarball.")
    print()

    # Write a package metadata JSON alongside the tarball
    meta_path = output_path.with_suffix(".tar.gz.meta.json")
    meta = {
        "version": 1,
        "created": datetime.now(timezone.utc).isoformat(),
        "tarball": output_path.name,
        "tarball_size": stats["tarball_size"],
        "source_count": len(sources),
        "total_records": total_records,
        "total_files": stats["total_files"],
        "sources": {
            src: {
                "records": record_counts[src],
                "files": stats["source_file_counts"].get(src, 0),
            }
            for src in sources
        },
        "excluded_sources": sorted(RESTRICTED_SOURCES),
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"  Metadata:         {meta_path}")
    print()

    print("=" * 72)
    print(f"Done! Tarball ready at: {output_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
