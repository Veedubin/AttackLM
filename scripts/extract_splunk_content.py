#!/usr/bin/env python3
# CREDITS — DATA SOURCE ATTRIBUTION
# ----------------------------------
# This script transforms data from: splunk/security_content
# Repository: https://github.com/splunk/security_content
# License:    Apache License 2.0
# Copyright:  (c) Splunk Inc.
#
# The output JSONL is a *transformation* of upstream Splunk detection YAML
# files into OpenAI-style chat triples. See /ATTRIBUTION.md for full
# per-source attribution and re-distribution guidance.
# ----------------------------------
"""Deterministic extraction of Splunk detection rules into AttackLM JSONL training pairs.

Walks ``data/splunk-security-content/detections/`` and parses every ``.yml`` file.
For each detection, generates 1 OpenAI-style message triple covering SPL query,
data sources, notable event configuration, and MITRE ATT&CK mapping.

Output: ``data/datasets/buckets/sources/splunk-content/defensive/detection_engineering/data.jsonl``

Usage:
    python scripts/extract_splunk_content.py
    python scripts/extract_splunk_content.py --dry-run --max-files 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from mitre_tactic_lookup import get_tactic_for_technique, get_tactic_name

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DETECTIONS_DIR = BASE_DIR / "data" / "splunk-security-content" / "detections"
OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "datasets"
    / "buckets"
    / "sources"
    / "splunk-content"
    / "defensive"
    / "detection_engineering"
)
OUTPUT_PATH = OUTPUT_DIR / "data.jsonl"

# ---------------------------------------------------------------------------
# System message
# ---------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are a Splunk detection engineer. Write production-grade SPL detection "
    "queries mapped to MITRE ATT&CK with data source configuration, notable "
    "event setup, and false positive analysis."
)

# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------
ATTRIBUTION = {
    "source": "splunk-content",
    "source_uri": "https://github.com/splunk/security_content",
    "license": "Apache-2.0",
    "license_uri": "https://github.com/splunk/security_content/blob/main/LICENSE",
    "rights_contact": "Splunk Inc.",
    "attribution_text": (
        "Copyright (c) Splunk Inc. Licensed under Apache License 2.0. "
        "See https://github.com/splunk/security_content/blob/main/LICENSE."
    ),
}


# ---------------------------------------------------------------------------
# MITRE ID extraction
# ---------------------------------------------------------------------------
def _extract_mitre_ids(tags: dict[str, Any] | None) -> list[str]:
    """Extract MITRE ATT&CK technique IDs from Splunk detection tags."""
    if not tags:
        return []
    ids = []
    # Common patterns in Splunk security_content
    for field in ["mitre_attack_id", "mitre_attack", "attack_id"]:
        val = tags.get(field, "")
        if isinstance(val, str) and val:
            for m in re.finditer(r"T\d{4}(?:\.\d{3})?", val):
                ids.append(m.group(0))
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    for m in re.finditer(r"T\d{4}(?:\.\d{3})?", item):
                        ids.append(m.group(0))

    # Also check for nested mitre_attack array
    mitre_attack = tags.get("mitre_attack", [])
    if isinstance(mitre_attack, list):
        for item in mitre_attack:
            if isinstance(item, str):
                for m in re.finditer(r"T\d{4}(?:\.\d{3})?", item):
                    ids.append(m.group(0))

    return sorted(set(ids))


# ---------------------------------------------------------------------------
# Parse a single Splunk detection YAML file
# ---------------------------------------------------------------------------
def parse_splunk_file(filepath: Path) -> dict[str, Any] | None:
    """Parse one Splunk detection YAML file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"  [WARN] YAML parse error in {filepath}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  [WARN] Error reading {filepath}: {e}", file=sys.stderr)
        return None

    if not isinstance(data, dict):
        return None

    name = data.get("name", "")
    if not name:
        return None

    return data


# ---------------------------------------------------------------------------
# Build training pair
# ---------------------------------------------------------------------------
def build_pair(detection: dict[str, Any]) -> dict[str, Any] | None:
    """Build one training pair from a Splunk detection."""
    name = detection.get("name", "Untitled Detection")
    description = detection.get("description", "")
    search = detection.get("search", "")
    data_source = detection.get("data_source", [])
    type_name = detection.get("type", "TTP")
    tags = detection.get("tags", {})
    how_to_implement = detection.get("how_to_implement", "")
    known_false_positives = detection.get("known_false_positives", "")

    mitre_ids = _extract_mitre_ids(tags)
    technique_str = (
        ", ".join(mitre_ids) if mitre_ids else "appropriate MITRE ATT&CK technique"
    )

    # Build data sources text
    ds_text = ""
    if data_source:
        if isinstance(data_source, list):
            ds_items = [f"- {ds}" for ds in data_source]
            ds_text = "\n**Data Sources:**\n" + "\n".join(ds_items)
        elif isinstance(data_source, str):
            ds_text = f"\n**Data Sources:** {data_source}"

    # Build false positives
    fp_text = ""
    if known_false_positives:
        fp_text = f"\n**Known False Positives:** {known_false_positives}"

    # Build implementation
    impl_text = ""
    if how_to_implement:
        impl_text = f"\n**Implementation:** {how_to_implement}"

    # Build assistant response
    assistant = (
        f"## {name}\n"
        f"**Type:** {type_name}\n"
        f"**MITRE ATT&CK:** {technique_str}\n\n"
        f"{description.strip()}\n\n"
        f"### SPL Query\n"
        f"```spl\n{search.strip()}\n```"
        f"{ds_text}"
        f"{fp_text}"
        f"{impl_text}"
        f"\n\n**Notable Event Configuration:** Create a notable event in Splunk "
        f"Enterprise Security with the recommended severity and drilldown "
        f"search. Map to the {technique_str} MITRE ATT&CK technique in the "
        f"Enterprise Security framework."
    )

    user = (
        f"Write a Splunk detection for {name}. Map to MITRE ATT&CK "
        f"{technique_str}. Include the SPL query, data sources, notable "
        f"event configuration, and false positive analysis."
    )

    pair = {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "mitre_ids": mitre_ids,
        **ATTRIBUTION,
    }

    # Add tactic info
    if mitre_ids:
        for tech_id in mitre_ids:
            tactic_id = get_tactic_for_technique(tech_id)
            if tactic_id:
                pair["mitre_tactic_id"] = tactic_id
                tactic_name = get_tactic_name(tactic_id)
                if tactic_name:
                    pair["tactic"] = tactic_name
                    pair["kill_chain_phase"] = tactic_name
                break

    return pair


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract Splunk detection rules into AttackLM training pairs"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print pairs without writing"
    )
    parser.add_argument(
        "--max-files", type=int, default=0, help="Max files to process (0=all)"
    )
    args = parser.parse_args()

    if not DETECTIONS_DIR.exists():
        print(
            f"Splunk detections directory not found: {DETECTIONS_DIR}", file=sys.stderr
        )
        print("Run attacklm init --clone-only or attacklm init first.", file=sys.stderr)
        return 1

    yml_files = sorted(DETECTIONS_DIR.rglob("*.yml"))
    if args.max_files > 0:
        yml_files = yml_files[: args.max_files]

    print(f"Processing {len(yml_files)} Splunk detection files...", file=sys.stderr)

    all_pairs: list[dict[str, Any]] = []
    skipped = 0

    for fp in yml_files:
        detection = parse_splunk_file(fp)
        if detection is None:
            skipped += 1
            continue

        pair = build_pair(detection)
        if pair:
            all_pairs.append(pair)

    print(
        f"Generated {len(all_pairs)} pairs ({skipped} files skipped)", file=sys.stderr
    )

    if args.dry_run:
        for pair in all_pairs[:5]:
            print(json.dumps(pair, indent=2))
        print(f"\n... and {len(all_pairs) - 5} more pairs", file=sys.stderr)
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"Wrote {len(all_pairs)} pairs to {OUTPUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
