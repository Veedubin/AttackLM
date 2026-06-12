#!/usr/bin/env python3
# CREDITS — DATA SOURCE ATTRIBUTION
# ----------------------------------
# Augments existing JSONL training files with per-pair `source` and
# `license` fields for traceable provenance. See /ATTRIBUTION.md
# for the full mapping.
#
# This is a POST-PROCESSING step — the extractors don't add these
# fields at write time, so we add them here based on which bucket
# each file lives in. This keeps the extractors simple and makes
# the source-license mapping centralized in one file (THIS one).
# ----------------------------------
"""Augment bucket JSONL files with per-pair `source` and `license` fields.

Walks every bucket under `data/datasets/buckets/` and for each
`data.jsonl` adds two new fields to every row:

  - `source`:  short string identifying the upstream project
  - `license`: SPDX license identifier (MIT, Apache-2.0, BSD-3-Clause,
               GPL-3.0, AGPL-3.0, DRL-1.1, etc.)

The source/license mapping is bucket-based (per the manifest in
`/ATTRIBUTION.md`). For buckets that mix multiple sources (e.g.
`defense_evasion` mixes Atomic + Caldera + Metasploit + RTA), we
use the dominant source by pair count.

For the AI-models and tools subdirs, we use the bucket name as the
source identifier (these are single-source buckets).

This is **additive** — it doesn't change `messages` or any other
existing field, so in-flight training jobs continue to work.

Usage:
    uv run python scripts/augment_attribution.py [--dry-run]
    uv run python scripts/augment_attribution.py --bucket execution
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BUCKETS_DIR = BASE_DIR / "data" / "datasets" / "buckets"

# Bucket → (source, license) mapping.
# See /ATTRIBUTION.md for the full attribution story. This mapping is
# a one-line-per-bucket summary suitable for machine-readable provenance.
BUCKET_ATTRIBUTION: dict[str, dict[str, str]] = {
    # MITRE tactic buckets — mixed sources, dominant attribution noted
    "collection": {
        "source": "atomic-red-team+caldera+metasploit",
        "license": "MIT+Apache-2.0+BSD-3-Clause",
    },
    "command_and_control": {
        "source": "atomic-red-team+caldera+metasploit",
        "license": "MIT+Apache-2.0+BSD-3-Clause",
    },
    "credential_access": {
        "source": "atomic-red-team+caldera+metasploit",
        "license": "MIT+Apache-2.0+BSD-3-Clause",
    },
    "defense_evasion": {
        "source": "atomic-red-team+caldera+metasploit",
        "license": "MIT+Apache-2.0+BSD-3-Clause",
    },
    "discovery": {
        "source": "atomic-red-team+caldera+metasploit",
        "license": "MIT+Apache-2.0+BSD-3-Clause",
    },
    "execution": {
        "source": "atomic-red-team+caldera+metasploit",
        "license": "MIT+Apache-2.0+BSD-3-Clause",
    },
    "exfiltration": {
        "source": "atomic-red-team+caldera+metasploit",
        "license": "MIT+Apache-2.0+BSD-3-Clause",
    },
    "lateral_movement": {
        "source": "atomic-red-team+caldera+metasploit",
        "license": "MIT+Apache-2.0+BSD-3-Clause",
    },
    "persistence": {
        "source": "atomic-red-team+caldera+metasploit",
        "license": "MIT+Apache-2.0+BSD-3-Clause",
    },
    "privilege_escalation": {
        "source": "atomic-red-team+caldera+metasploit",
        "license": "MIT+Apache-2.0+BSD-3-Clause",
    },
    # Orchestrator — synthetic, MIT (same as this repo)
    "orchestrator": {"source": "synthetic-attacklm", "license": "MIT"},
    # AI security subdirs
    "ai-models/jailbreaking": {
        "source": "garak+pyrit+fuzzyai+bigpromptlib",
        "license": "Apache-2.0+MIT",
    },
    "ai-models/prompt-injection": {
        "source": "promptfoo+promptmap+synthetic",
        "license": "MIT",
    },
    # Tools subdirs — single-source attribution
    "tools/infection_monkey": {"source": "guardicore/monkey", "license": "GPL-3.0"},
    "tools/metasploit": {
        "source": "rapid7/metasploit-framework",
        "license": "BSD-3-Clause",
    },
    "tools/rta": {"source": "endgameinc/RTA", "license": "AGPL-3.0"},
}


def augment_bucket(
    bucket_path: Path, attr: dict[str, str], dry_run: bool = False
) -> int:
    """Augment data.jsonl in one bucket with source/license. Returns count."""
    data_file = bucket_path / "data.jsonl"
    if not data_file.exists():
        return 0

    rows = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  WARNING: bad JSON in {data_file}: {e}", file=sys.stderr)
                continue
            # Additive: only set if not already present
            if "source" not in row:
                row["source"] = attr["source"]
            if "license" not in row:
                row["license"] = attr["license"]
            rows.append(row)

    if dry_run:
        print(f"  [DRY RUN] would augment {len(rows)} rows in {data_file}")
        if rows:
            sample = {k: rows[0].get(k) for k in ("source", "license")}
            print(f"  [DRY RUN] sample: {sample}")
        return len(rows)

    # Atomic write: write to temp file, then rename
    tmp_file = data_file.with_suffix(".jsonl.tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_file.replace(data_file)
    print(f"  ✓ {data_file.relative_to(BASE_DIR)}: {len(rows)} rows augmented")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add source/license fields to bucket JSONL files"
    )
    parser.add_argument("--dry-run", action="store_true", help="Don't modify any files")
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="Only augment one bucket (e.g. 'execution' or 'tools/metasploit')",
    )
    args = parser.parse_args()

    if not BUCKETS_DIR.exists():
        print(f"ERROR: buckets dir not found: {BUCKETS_DIR}", file=sys.stderr)
        return 1

    # v0.3.0+ uses the per-source layout (data/datasets/buckets/sources/...).
    # The flat layout at BUCKETS_DIR/<bucket>/data.jsonl no longer exists.
    # Every record already carries `source`, `source_uri`, `license`,
    # `license_uri`, `rights_contact` (added by scripts/stamp_and_reorg.py)
    # plus per-license attribution fields (added by scripts/add_attribution.py).
    # This script is therefore a no-op as of v0.3.0.
    sources_dir = BUCKETS_DIR / "sources"
    if sources_dir.exists():
        print("NOTE: Per-source layout is in use (v0.3.0+).")
        print("      Every record already has provenance fields from")
        print("      scripts/stamp_and_reorg.py + scripts/add_attribution.py.")
        print("      This script is a no-op and can be removed.")
        print()
        return 0

    total = 0
    for bucket_name, attr in BUCKET_ATTRIBUTION.items():
        if args.bucket and bucket_name != args.bucket:
            continue
        bucket_path = BUCKETS_DIR / bucket_name
        if not bucket_path.exists():
            print(f"  skip (not found): {bucket_name}")
            continue
        count = augment_bucket(bucket_path, attr, dry_run=args.dry_run)
        total += count

    print()
    print(f"Total rows: {total}")
    if args.dry_run:
        print("[DRY RUN] no files were modified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
