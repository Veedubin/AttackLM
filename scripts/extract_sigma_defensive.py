#!/usr/bin/env python3
# CREDITS — DATA SOURCE ATTRIBUTION
# ----------------------------------
# This script transforms data from: SigmaHQ/sigma
# Repository: https://github.com/SigmaHQ/sigma
# License:    Detection Rule License (DRL) 1.1
# Copyright:  (c) SigmaHQ contributors. All rights reserved.
#
# The output JSONL is a *transformation* of upstream Sigma detection rule
# YAML files into OpenAI-style chat triples. See /ATTRIBUTION.md for full
# per-source attribution and re-distribution guidance.
# ----------------------------------
"""Deterministic extraction of Sigma detection rules into AttackLM JSONL training pairs.

Walks ``data/sigma/rules/`` and parses every ``.yml`` file. For each rule,
generates 1 OpenAI-style message triple covering detection logic, false positives,
and MITRE ATT&CK mapping.

Output: ``data/datasets/buckets/sources/sigma-hq/defensive/detection_engineering/data.jsonl``

Usage:
    python scripts/extract_sigma_defensive.py
    python scripts/extract_sigma_defensive.py --dry-run --max-files 3
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
RULES_DIR = BASE_DIR / "data" / "sigma" / "rules"
OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "datasets"
    / "buckets"
    / "sources"
    / "sigma-hq"
    / "defensive"
    / "detection_engineering"
)
OUTPUT_PATH = OUTPUT_DIR / "data.jsonl"

# ---------------------------------------------------------------------------
# System message
# ---------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are a Detection Engineering specialist. Write precise Sigma detection "
    "rules mapped to MITRE ATT&CK with detection logic, false positive analysis, "
    "and deployment guidance."
)

# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------
ATTRIBUTION = {
    "source": "sigma-hq",
    "source_uri": "https://github.com/SigmaHQ/sigma",
    "license": "DRL-1.1",
    "license_uri": "https://github.com/SigmaHQ/sigma/blob/master/LICENSE",
    "rights_contact": "SigmaHQ",
    "attribution_text": (
        "Detection Rule License (DRL) 1.1 — Copyright (c) SigmaHQ contributors. "
        "See https://github.com/SigmaHQ/sigma/blob/master/LICENSE for full terms."
    ),
}


# ---------------------------------------------------------------------------
# MITRE tag extraction
# ---------------------------------------------------------------------------
def _extract_mitre_ids(tags: list[str] | None) -> list[str]:
    """Extract MITRE ATT&CK technique IDs from Sigma tags."""
    if not tags:
        return []
    ids = []
    for tag in tags:
        if isinstance(tag, str):
            m = re.match(r"attack\.(t\d+(?:\.\d+)?)", tag, re.IGNORECASE)
            if m:
                ids.append(m.group(1).upper())
    return sorted(set(ids))


# ---------------------------------------------------------------------------
# Parse a single Sigma YAML file
# ---------------------------------------------------------------------------
def parse_sigma_file(filepath: Path) -> dict[str, Any] | None:
    """Parse one Sigma rule YAML file."""
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

    title = data.get("title", "")
    if not title:
        return None

    return data


# ---------------------------------------------------------------------------
# Build training pair
# ---------------------------------------------------------------------------
def build_pair(rule: dict[str, Any]) -> dict[str, Any] | None:
    """Build one training pair from a Sigma rule."""
    title = rule.get("title", "Untitled Rule")
    description = rule.get("description", "")
    status = rule.get("status", "stable")
    level = rule.get("level", "medium")
    falsepositives = rule.get("falsepositives", [])
    tags = rule.get("tags", [])
    detection = rule.get("detection", {})
    references = rule.get("references", [])

    mitre_ids = _extract_mitre_ids(tags)
    technique_str = (
        ", ".join(mitre_ids) if mitre_ids else "appropriate MITRE ATT&CK technique"
    )

    # Build detection summary
    detection_parts = []
    if isinstance(detection, dict):
        condition = detection.get("condition", "")
        selections = {k: v for k, v in detection.items() if k != "condition"}
        if selections:
            detection_parts.append("**Detection Logic:**")
            for sel_name, sel_value in selections.items():
                if isinstance(sel_value, dict):
                    field_str = ", ".join(f"{k}={v}" for k, v in sel_value.items())
                    detection_parts.append(f"- `{sel_name}`: {field_str}")
                elif isinstance(sel_value, list):
                    detection_parts.append(
                        f"- `{sel_name}`: {', '.join(str(v) for v in sel_value)}"
                    )
        if condition:
            detection_parts.append(f"\n**Condition:** `{condition}`")

    # Build false positives
    fp_text = ""
    if falsepositives:
        if isinstance(falsepositives, list):
            fp_items = [f"- {fp}" for fp in falsepositives]
            fp_text = "\n**False Positives:**\n" + "\n".join(fp_items)
        elif isinstance(falsepositives, str):
            fp_text = f"\n**False Positives:** {falsepositives}"

    # Build references
    ref_text = ""
    if references:
        ref_items = [f"- {ref}" for ref in references[:5]]
        ref_text = "\n**References:**\n" + "\n".join(ref_items)

    # Build assistant response
    assistant = (
        f"## {title}\n"
        f"**Status:** {status} | **Level:** {level}\n"
        f"**MITRE ATT&CK:** {technique_str}\n\n"
        f"{description.strip()}\n\n"
        + "\n".join(detection_parts)
        + f"{fp_text}"
        + f"{ref_text}"
        + f"\n\n**Deployment Notes:** This Sigma rule can be converted to "
        f"Elasticsearch (ElastAlert), Splunk, Azure Sentinel, or any SIEM "
        f"supporting the Sigma format using `sigmac` or `sigma-cli`. Test in "
        f"a staging environment before production deployment."
    )

    technique_name = ""
    if mitre_ids:
        technique_name = get_tactic_name(mitre_ids[0]) or ""

    user = (
        f"Write a Sigma detection rule for {title}. Map it to MITRE ATT&CK "
        f"{technique_str}. Include detection logic, false positive scenarios, "
        f"and SIEM deployment notes."
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
        description="Extract Sigma detection rules into AttackLM training pairs"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print pairs without writing"
    )
    parser.add_argument(
        "--max-files", type=int, default=0, help="Max files to process (0=all)"
    )
    args = parser.parse_args()

    if not RULES_DIR.exists():
        print(f"Sigma rules directory not found: {RULES_DIR}", file=sys.stderr)
        print("Run attacklm-clone or attacklm-init first.", file=sys.stderr)
        return 1

    yml_files = sorted(RULES_DIR.rglob("*.yml"))
    if args.max_files > 0:
        yml_files = yml_files[: args.max_files]

    print(f"Processing {len(yml_files)} Sigma rule files...", file=sys.stderr)

    all_pairs: list[dict[str, Any]] = []
    skipped = 0

    for fp in yml_files:
        rule = parse_sigma_file(fp)
        if rule is None:
            skipped += 1
            continue

        pair = build_pair(rule)
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
