#!/usr/bin/env python3
# CREDITS — DATA SOURCE ATTRIBUTION
# ----------------------------------
# This script transforms data from: OTRF/ThreatHunter-Playbook
# Repository: https://github.com/OTRF/ThreatHunter-Playbook
# License:    Apache License 2.0
# Copyright:  (c) Open Threat Research (OTRF)
# ----------------------------------
"""Deterministic extraction of ThreatHunter-Playbook into AttackLM JSONL training pairs.

Walks ``data/threathunter-playbook/playbooks/`` and parses Markdown playbooks
with KQL queries, data sources, and hunting steps. For each playbook, generates
1 OpenAI-style message triple covering hunting methodology.

Output: ``data/datasets/buckets/sources/threathunter-playbook/defensive/threat_hunting/data.jsonl``

Usage:
    python scripts/extract_threathunter_playbook.py
    python scripts/extract_threathunter_playbook.py --dry-run --max-files 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from mitre_tactic_lookup import get_tactic_for_technique, get_tactic_name

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PLAYBOOKS_DIR = BASE_DIR / "data" / "threathunter-playbook" / "playbooks"
OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "datasets"
    / "buckets"
    / "sources"
    / "threathunter-playbook"
    / "defensive"
    / "threat_hunting"
)
OUTPUT_PATH = OUTPUT_DIR / "data.jsonl"

# ---------------------------------------------------------------------------
# System message
# ---------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are a Threat Hunting methodology specialist. Design detection playbooks "
    "with KQL queries, data source requirements, and step-by-step hunting "
    "procedures mapped to MITRE ATT&CK."
)

# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------
ATTRIBUTION = {
    "source": "threathunter-playbook",
    "source_uri": "https://github.com/OTRF/ThreatHunter-Playbook",
    "license": "Apache-2.0",
    "license_uri": "https://github.com/OTRF/ThreatHunter-Playbook/blob/master/LICENSE",
    "rights_contact": "Open Threat Research (OTRF)",
    "attribution_text": (
        "Copyright (c) Open Threat Research. Licensed under Apache License 2.0. "
        "See https://github.com/OTRF/ThreatHunter-Playbook/blob/master/LICENSE."
    ),
}


# ---------------------------------------------------------------------------
# Parse a Markdown playbook
# ---------------------------------------------------------------------------
def parse_playbook(filepath: Path) -> dict[str, Any] | None:
    """Parse a Markdown playbook file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"  [WARN] Error reading {filepath}: {e}", file=sys.stderr)
        return None

    if not content.strip():
        return None

    # Extract title from first heading
    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else filepath.stem

    # Extract MITRE technique IDs
    mitre_ids = []
    for m in re.finditer(r"T\d{4}(?:\.\d{3})?", content):
        mitre_ids.append(m.group(0))
    mitre_ids = sorted(set(mitre_ids))

    # Extract KQL queries from code blocks
    kql_queries = []
    in_code = False
    current_query = []
    for line in content.split("\n"):
        if line.strip().startswith("```"):
            if in_code:
                kql_queries.append("\n".join(current_query))
                current_query = []
                in_code = False
            else:
                in_code = True
        elif in_code:
            current_query.append(line)

    # Extract data sources
    data_sources = []
    ds_section = False
    for line in content.split("\n"):
        if re.match(r"^##\s+.*[Dd]ata\s*[Ss]ource", line):
            ds_section = True
            continue
        if ds_section and line.startswith("##"):
            ds_section = False
        if ds_section and line.strip().startswith("-"):
            data_sources.append(line.strip()[1:].strip())

    return {
        "title": title,
        "content": content,
        "mitre_ids": mitre_ids,
        "kql_queries": kql_queries,
        "data_sources": data_sources,
    }


# ---------------------------------------------------------------------------
# Build training pair
# ---------------------------------------------------------------------------
def build_pair(playbook: dict[str, Any]) -> dict[str, Any] | None:
    """Build one training pair from a playbook."""
    title = playbook["title"]
    content = playbook["content"]
    mitre_ids = playbook["mitre_ids"]
    kql_queries = playbook["kql_queries"]
    data_sources = playbook["data_sources"]

    technique_str = (
        ", ".join(mitre_ids) if mitre_ids else "the appropriate MITRE ATT&CK technique"
    )

    # Build data sources text
    ds_text = ""
    if data_sources:
        ds_items = [f"- {ds}" for ds in data_sources[:10]]
        ds_text = "\n**Required Data Sources:**\n" + "\n".join(ds_items)

    # Build KQL queries text
    kql_text = ""
    if kql_queries:
        kql_text = "\n**Detection Queries:**\n"
        for i, query in enumerate(kql_queries[:3]):
            kql_text += f"\n```kql\n{query.strip()}\n```\n"

    # Extract methodology from content (first 2000 chars after title)
    body = re.sub(r"^#\s+.+$", "", content, count=1, flags=re.MULTILINE).strip()
    methodology = body[:2000]

    user = (
        f"How would you hunt for {technique_str} in a Windows environment? "
        f"Provide KQL queries, required data sources, and expected artifacts."
    )

    assistant = (
        f"## {title}\n"
        f"**MITRE ATT&CK:** {technique_str}\n\n"
        f"{methodology}"
        f"{ds_text}"
        f"{kql_text}"
        f"\n\n**Expected Artifacts:**\n"
        f"- Process creation events with suspicious command-line arguments\n"
        f"- Network connections to unusual external IPs or ports\n"
        f"- File system modifications in sensitive directories\n"
        f"- Registry key modifications indicating persistence\n\n"
        f"**False Positive Handling:**\n"
        f"- Exclude known administrative accounts and service accounts\n"
        f"- Whitelist approved software deployment processes\n"
        f"- Correlate with change management records for authorized activity"
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
        description="Extract ThreatHunter-Playbook into AttackLM training pairs"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print pairs without writing"
    )
    parser.add_argument(
        "--max-files", type=int, default=0, help="Max files to process (0=all)"
    )
    args = parser.parse_args()

    if not PLAYBOOKS_DIR.exists():
        print(f"Playbooks directory not found: {PLAYBOOKS_DIR}", file=sys.stderr)
        print("Run attacklm-clone or attacklm-init first.", file=sys.stderr)
        return 1

    md_files = sorted(PLAYBOOKS_DIR.rglob("*.md"))
    if args.max_files > 0:
        md_files = md_files[: args.max_files]

    print(f"Processing {len(md_files)} playbook files...", file=sys.stderr)

    all_pairs: list[dict[str, Any]] = []
    skipped = 0

    for fp in md_files:
        playbook = parse_playbook(fp)
        if playbook is None:
            skipped += 1
            continue

        pair = build_pair(playbook)
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
