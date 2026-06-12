#!/usr/bin/env python3
"""
rebuild_manifest.py — Rebuild the AttackLM bucket manifest from disk.

v5.0.0: Walks BOTH the new per-source layout
(`data/datasets/buckets/sources/<source>/<bucket>/<tactic>/data*.jsonl`)
and the legacy flat layout (`data/datasets/buckets/<bucket>/data*.jsonl`).

The per-source layout is canonical. The flat layout is still present for
backward compatibility with existing training/audit scripts but is no
longer the source of truth.

Buckets in `archive/restricted-sources/` are NEVER included in the
public manifest.

Usage:
    python scripts/rebuild_manifest.py
    python scripts/rebuild_manifest.py --dry-run
"""

import json
import sys
from collections import Counter
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

# ---------------------------------------------------------------------------
# Sub-file names for three-tier provenance
# ---------------------------------------------------------------------------
SUB_FILES = {
    "human": "data_human.jsonl",
    "llm": "data_llm.jsonl",
    "synth": "data_synth.jsonl",
}


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


def discover_from_sources() -> tuple[list[dict], dict]:
    """
    Walk the new per-source layout. Returns (buckets, source_meta).

    buckets: list of bucket entries (tactic-level), each carrying a
             `sources` field with per-source record counts.
    source_meta: per-source summary { name: {license, n_records, buckets: [...] } }
    """
    buckets_map: dict[str, dict] = {}  # bucket_path -> entry
    source_meta: dict[str, dict] = {}

    if not SOURCES_DIR.exists():
        return [], {}

    # Load _index.json if present for license/display info
    index_path = SOURCES_DIR / "_index.json"
    index_data = {}
    if index_path.exists():
        try:
            with index_path.open() as f:
                idx = json.load(f)
            for s in idx.get("sources", []):
                index_data[s["name"]] = s
        except Exception:
            pass

    for src_dir in sorted(SOURCES_DIR.iterdir()):
        if not src_dir.is_dir() or src_dir.name.startswith("_"):
            continue
        source_name = src_dir.name
        meta = index_data.get(source_name, {})
        src_entry = {
            "name": source_name,
            "display": meta.get("display", source_name),
            "license": meta.get("license", "unknown"),
            "license_uri": meta.get("license_uri", ""),
            "upstream_url": meta.get("upstream_url", ""),
            "risk": meta.get("risk", "unknown"),
            "n_records": 0,
            "buckets": [],
        }

        for jsonl in sorted(src_dir.rglob("*.jsonl")):
            parts = jsonl.relative_to(src_dir).parts
            # Layout variants:
            #   <bucket>/<tactic>/<file>.jsonl    (e.g. base/execution/data.jsonl)
            #   <bucket>/<file>.jsonl              (e.g. orchestrator/data.jsonl)
            if len(parts) == 3:
                bucket_path = f"{parts[0]}/{parts[1]}"
            elif len(parts) == 2:
                bucket_path = parts[0]
            else:
                continue
            n = count_jsonl_lines(jsonl)
            src_entry["n_records"] += n
            src_entry["buckets"].append(
                {
                    "bucket": bucket_path,
                    "file": jsonl.name,
                    "records": n,
                }
            )

            # Aggregate into tactic-level bucket
            if bucket_path not in buckets_map:
                buckets_map[bucket_path] = {
                    "name": bucket_path.replace("/", "_"),
                    "path": bucket_path,
                    "sources": {},
                    "files": {},
                }
            entry = buckets_map[bucket_path]
            entry["sources"][source_name] = entry["sources"].get(source_name, 0) + n
            entry["files"][f"{source_name}:{jsonl.name}"] = n

        source_meta[source_name] = src_entry

    buckets = []
    for path, entry in buckets_map.items():
        # Derive category from first path component
        first = path.split("/")[0]
        cat_map = {
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
        category = cat_map.get(first, first)
        # Determine dominant source for this bucket
        dominant_source = max(entry["sources"], key=entry["sources"].get)
        dom_meta = source_meta.get(dominant_source, {})

        # Determine sub_sources tier for the bucket based on filename
        sub_sources = {"human": 0, "llm": 0, "synth": 0}
        for key, n in entry["files"].items():
            src, fname = key.split(":", 1)
            if fname == "data_llm.jsonl":
                sub_sources["llm"] += n
            elif fname == "data_synth.jsonl":
                sub_sources["synth"] += n
            else:
                # data.jsonl is treated as "human" if it comes from a real
                # upstream source, "synth" if from attacklm-synthetic.
                if src == "attacklm-synthetic" or src == "llm-generated":
                    sub_sources["synth"] += n
                else:
                    sub_sources["human"] += n

        buckets.append(
            {
                "name": entry["name"],
                "path": entry["path"],
                "category": category,
                "count": sum(entry["sources"].values()),
                "sub_sources": sub_sources,
                "sources": entry["sources"],
                "dominant_source": dominant_source,
                "license": dom_meta.get("license", "unknown"),
            }
        )

    return buckets, source_meta


def discover_from_flat() -> list[dict]:
    """
    Walk the legacy flat layout (data/datasets/buckets/<bucket>/data*.jsonl).
    Used as a fallback / sanity check.
    """
    buckets: list[dict] = []
    for metadata_path in sorted(BUCKETS_DIR.rglob("metadata.json")):
        # Skip metadata under sources/ (handled by discover_from_sources)
        if "sources/" in str(metadata_path.relative_to(BUCKETS_DIR)):
            continue
        bucket_dir = metadata_path.parent
        rel_path = bucket_dir.relative_to(BUCKETS_DIR)
        # Skip root
        if str(rel_path) == ".":
            continue
        try:
            with open(metadata_path, encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:
            continue
        n = 0
        for fname in SUB_FILES.values():
            n += count_jsonl_lines(bucket_dir / fname)
        if n == 0:
            legacy = bucket_dir / "data.jsonl"
            n = count_jsonl_lines(legacy)
        if n == 0:
            continue
        buckets.append(
            {
                "name": meta.get("name", bucket_dir.name),
                "path": str(rel_path),
                "count": n,
            }
        )
    return buckets


def sort_buckets(buckets: list[dict]) -> list[dict]:
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
    cat_index = {c: i for i, c in enumerate(CATEGORY_ORDER)}

    def key(b: dict) -> tuple[int, str]:
        return (
            cat_index.get(b.get("category", "zzz"), len(CATEGORY_ORDER)),
            b.get("name", ""),
        )

    return sorted(buckets, key=key)


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    print("=" * 72)
    print("AttackLM Manifest Rebuilder v5 (per-source layout)")
    print("=" * 72)
    print(f"  Sources dir:   {SOURCES_DIR}")
    print(f"  Manifest:      {MANIFEST_PATH}")
    print(f"  Dry run:       {dry_run}")
    print()

    print("Step 1: Discovering buckets from per-source layout...")
    buckets, source_meta = discover_from_sources()
    print(
        f"  Found {len(buckets)} tactic-level buckets across {len(source_meta)} sources"
    )
    print()

    print("Step 2: Also scanning flat layout (sanity check)...")
    flat_buckets = discover_from_flat()
    print(f"  Found {len(flat_buckets)} flat-layout buckets (for back-compat)")
    print()

    print("Step 3: Sorting buckets by category priority...")
    buckets = sort_buckets(buckets)
    print()

    # Summary
    total_pairs = sum(b["count"] for b in buckets)
    cat_counts: Counter = Counter()
    for b in buckets:
        cat_counts[b.get("category", "unknown")] += b["count"]

    print("=" * 72)
    print("CATEGORY SUMMARY (per-source layout)")
    print("=" * 72)
    print(f"{'Category':<22} {'Pairs':>10} {'%':>8}")
    print("-" * 42)
    for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        pct = n / total_pairs * 100 if total_pairs else 0
        print(f"{cat:<22} {n:>10,} {pct:>7.1f}%")
    print("-" * 42)
    print(f"{'TOTAL':<22} {total_pairs:>10,} {'100.0%':>8}")
    print()

    print("=" * 72)
    print("SOURCE SUMMARY")
    print("=" * 72)
    print(f"{'Source':<25} {'License':<14} {'Risk':<8} {'Records':>10} {'%':>6}")
    print("-" * 70)
    for src_name, meta in sorted(source_meta.items(), key=lambda x: -x[1]["n_records"]):
        n = meta["n_records"]
        pct = n / total_pairs * 100 if total_pairs else 0
        print(
            f"{src_name:<25} {meta['license']:<14} {meta['risk']:<8} {n:>10,} {pct:>5.1f}%"
        )
    print("-" * 70)
    print(f"{'TOTAL':<25} {'':<14} {'':<8} {total_pairs:>10,} {'100.0%':>6}")
    print()

    print("=" * 72)
    print("TIER TOTALS (provenance)")
    print("=" * 72)
    tier: Counter = Counter()
    for b in buckets:
        for k, v in b.get("sub_sources", {}).items():
            tier[k] += v
    for k in ("human", "llm", "synth"):
        print(f"  {k:<8} {tier.get(k, 0):>10,}")
    print()

    # Validation
    errors = []
    zero_count = [b for b in buckets if b["count"] == 0]
    if zero_count:
        errors.append(f"{len(zero_count)} buckets with count=0")
    if not errors:
        print("Validation: OK")
    else:
        for e in errors:
            print(f"Validation: FAIL — {e}")
    print()

    if not dry_run:
        manifest = {
            "version": 5,
            "layout": "per-source (sources/<source>/<bucket>/<tactic>/data*.jsonl)",
            "legacy_layout": "flat (data/datasets/buckets/<bucket>/data.jsonl) — kept for back-compat",
            "provenance": "per-source + per-record fields (source, source_uri, license, license_uri, rights_contact, plus per-license attribution fields)",
            "excluded_sources": [
                "endgameinc/RTA (AGPL-3.0)",
                "guardicore/infection_monkey (GPL-3.0)",
                "TheBigPromptLibrary (mixed/unclear)",
            ],
            "created": datetime.now(timezone.utc).isoformat(),
            "total_buckets": len(buckets),
            "total_pairs": total_pairs,
            "tier_totals": dict(tier),
            "source_totals": {k: v["n_records"] for k, v in source_meta.items()},
            "sources": source_meta,
            "buckets": buckets,
        }
        with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
        print(f"Manifest written: {MANIFEST_PATH}")
        print(f"  Version: {manifest['version']}")
        print(f"  Total buckets: {len(buckets)}")
        print(f"  Total pairs: {total_pairs:,}")
    else:
        print("Dry run — manifest NOT written.")
    print()


if __name__ == "__main__":
    main()
