#!/usr/bin/env python3
# CREDITS — DATA SOURCE ATTRIBUTION
# ----------------------------------
# This script transforms data from: elastic/detection-rules
# Repository: https://github.com/elastic/detection-rules
# License:    Elastic License 2.0
# Copyright:  Elasticsearch B.V.
#
# The output JSONL is a *transformation* of upstream Elastic detection rule
# TOML files into OpenAI-style chat triples. See /ATTRIBUTION.md for full
# per-source attribution and re-distribution guidance.
# ----------------------------------
"""Deterministic extraction of Elastic detection rules into AttackLM JSONL training pairs.

Walks ``data/elastic-detection-rules/rules/`` and parses every ``.toml`` file.
For each rule, generates 1 OpenAI-style message triple covering EQL/KQL query,
severity, risk score, and MITRE ATT&CK mapping.

Output: ``data/datasets/buckets/sources/elastic-rules/defensive/detection_engineering/data.jsonl``

Usage:
    python scripts/extract_elastic_rules.py
    python scripts/extract_elastic_rules.py --dry-run --max-files 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from mitre_tactic_lookup import get_tactic_for_technique, get_tactic_name

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RULES_DIR = BASE_DIR / "data" / "elastic-detection-rules" / "rules"
OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "datasets"
    / "buckets"
    / "sources"
    / "elastic-rules"
    / "defensive"
    / "detection_engineering"
)
OUTPUT_PATH = OUTPUT_DIR / "data.jsonl"

# ---------------------------------------------------------------------------
# System message
# ---------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are an Elastic Security detection engineer. Write production-grade "
    "detection rules in EQL/KQL mapped to MITRE ATT&CK with severity scoring, "
    "risk assessment, and false positive analysis."
)

# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------
ATTRIBUTION = {
    "source": "elastic-rules",
    "source_uri": "https://github.com/elastic/detection-rules",
    "license": "Elastic-2.0",
    "license_uri": "https://github.com/elastic/detection-rules/blob/main/LICENSE.txt",
    "rights_contact": "Elasticsearch B.V.",
    "attribution_text": (
        "Copyright Elasticsearch B.V. Licensed under Elastic License 2.0. "
        "See https://github.com/elastic/detection-rules/blob/main/LICENSE.txt."
    ),
}


# ---------------------------------------------------------------------------
# Simple TOML parser (no external dependency needed for basic TOML)
# ---------------------------------------------------------------------------
def _parse_simple_toml(filepath: Path) -> dict[str, Any] | None:
    """Parse a basic TOML file without external dependencies.

    Handles the Elastic detection-rules TOML format which uses
    dotted section headers like [rule], [rule.threat], etc.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"  [WARN] Error reading {filepath}: {e}", file=sys.stderr)
        return None

    result: dict[str, Any] = {}
    current_section: list[str] = []

    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Section header
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            current_section = section.split(".")
            continue

        # Key = value
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()

            # Strip quotes
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]

            # Build nested dict path
            target = result
            for part in current_section:
                if part not in target:
                    target[part] = {}
                target = target[part]

            # Handle multi-line arrays (simplified)
            if value == "[":
                target[key] = []
            elif value == "]":
                pass
            else:
                # Try to parse as list
                if value.startswith("[") and value.endswith("]"):
                    inner = value[1:-1]
                    items = [
                        i.strip().strip('"').strip("'")
                        for i in inner.split(",")
                        if i.strip()
                    ]
                    target[key] = items
                else:
                    target[key] = value

    return result


# ---------------------------------------------------------------------------
# Extract MITRE IDs from rule
# ---------------------------------------------------------------------------
def _extract_mitre_ids(rule_data: dict[str, Any]) -> list[str]:
    """Extract MITRE ATT&CK technique IDs from rule threat data."""
    ids = []
    rule = rule_data.get("rule", {})
    threat = rule.get("threat", {})

    # Handle technique array
    technique = threat.get("technique", [])
    if isinstance(technique, dict):
        technique = [technique]

    for tech in technique:
        if isinstance(tech, dict):
            tid = tech.get("id", "")
            if tid:
                ids.append(tid)
            sub = tech.get("subtechnique", [])
            if isinstance(sub, dict):
                sub = [sub]
            for s in sub:
                if isinstance(s, dict):
                    sid = s.get("id", "")
                    if sid:
                        ids.append(sid)

    return sorted(set(ids))


# ---------------------------------------------------------------------------
# Build training pair
# ---------------------------------------------------------------------------
def build_pair(rule_data: dict[str, Any]) -> dict[str, Any] | None:
    """Build one training pair from an Elastic detection rule."""
    rule = rule_data.get("rule", {})
    if not rule:
        return None

    name = rule.get("name", "Untitled Rule")
    description = rule.get("description", "")
    query = rule.get("query", "")
    language = rule.get("language", "eql")
    severity = rule.get("severity", "medium")
    risk_score = rule.get("risk_score", "47")
    note = rule.get("note", "")
    false_positives = rule.get("false_positives", [])

    mitre_ids = _extract_mitre_ids(rule_data)
    technique_str = (
        ", ".join(mitre_ids) if mitre_ids else "appropriate MITRE ATT&CK technique"
    )

    # Build false positives text
    fp_text = ""
    if false_positives:
        if isinstance(false_positives, list):
            fp_items = [f"- {fp}" for fp in false_positives]
            fp_text = "\n**False Positives:**\n" + "\n".join(fp_items)
        elif isinstance(false_positives, str):
            fp_text = f"\n**False Positives:** {false_positives}"

    # Build investigation guide
    investigation = ""
    if note:
        investigation = f"\n**Investigation Guide:**\n{note.strip()}"

    # Build assistant response
    assistant = (
        f"## {name}\n"
        f"**Severity:** {severity} | **Risk Score:** {risk_score}\n"
        f"**Language:** {language.upper()}\n"
        f"**MITRE ATT&CK:** {technique_str}\n\n"
        f"{description.strip()}\n\n"
        f"### Detection Query ({language.upper()})\n"
        f"```{language}\n{query.strip()}\n```"
        f"{fp_text}"
        f"{investigation}"
        f"\n\n**Deployment:** This rule should be deployed in Elastic Security "
        f"with the recommended interval and risk score. Tune false positive "
        f"thresholds based on environment-specific baseline activity."
    )

    user = (
        f"Write an Elastic detection rule for {name}. Map to MITRE ATT&CK "
        f"{technique_str}. Include the {language.upper()} query, severity, "
        f"risk score, and false positive scenarios."
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
        description="Extract Elastic detection rules into AttackLM training pairs"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print pairs without writing"
    )
    parser.add_argument(
        "--max-files", type=int, default=0, help="Max files to process (0=all)"
    )
    args = parser.parse_args()

    if not RULES_DIR.exists():
        print(f"Elastic rules directory not found: {RULES_DIR}", file=sys.stderr)
        print("Run attacklm-clone or attacklm-init first.", file=sys.stderr)
        return 1

    toml_files = sorted(RULES_DIR.rglob("*.toml"))
    if args.max_files > 0:
        toml_files = toml_files[: args.max_files]

    print(f"Processing {len(toml_files)} Elastic rule files...", file=sys.stderr)

    all_pairs: list[dict[str, Any]] = []
    skipped = 0

    for fp in toml_files:
        rule_data = _parse_simple_toml(fp)
        if rule_data is None:
            skipped += 1
            continue

        pair = build_pair(rule_data)
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
