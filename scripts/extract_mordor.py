#!/usr/bin/env python3
# CREDITS — DATA SOURCE ATTRIBUTION
# ----------------------------------
# This script transforms data from: OTRF/Security-Datasets (Mordor)
# Repository: https://github.com/OTRF/Security-Datasets
# License:    Apache License 2.0
# Copyright:  (c) Open Threat Research (OTRF)
# ----------------------------------
"""Deterministic extraction of Mordor security event logs into AttackLM JSONL training pairs.

Walks ``data/mordor/datasets/`` and parses JSON event logs with metadata YAML.
For each scenario, generates 2-3 OpenAI-style message triples covering technique
identification, detection queries, and hunting methodology.

Output: ``data/datasets/buckets/sources/mordor/defensive/threat_hunting/data.jsonl``

Usage:
    python scripts/extract_mordor.py
    python scripts/extract_mordor.py --dry-run --max-files 3
"""

from __future__ import annotations

import argparse
import json
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
MORDOR_DIR = BASE_DIR / "data" / "mordor" / "datasets"
OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "datasets"
    / "buckets"
    / "sources"
    / "mordor"
    / "defensive"
    / "threat_hunting"
)
OUTPUT_PATH = OUTPUT_DIR / "data.jsonl"

# ---------------------------------------------------------------------------
# System message
# ---------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are a Threat Hunting specialist. Analyze security event logs, identify "
    "adversary techniques, extract indicators of compromise, and provide detection "
    "queries for SIEM platforms."
)

# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------
ATTRIBUTION = {
    "source": "mordor",
    "source_uri": "https://github.com/OTRF/Security-Datasets",
    "license": "Apache-2.0",
    "license_uri": "https://github.com/OTRF/Security-Datasets/blob/master/LICENSE",
    "rights_contact": "Open Threat Research (OTRF)",
    "attribution_text": (
        "Copyright (c) Open Threat Research. Licensed under Apache License 2.0. "
        "See https://github.com/OTRF/Security-Datasets/blob/master/LICENSE."
    ),
}


# ---------------------------------------------------------------------------
# Find Mordor scenarios
# ---------------------------------------------------------------------------
def find_scenarios() -> list[dict[str, Any]]:
    """Find all Mordor scenarios with metadata and event logs."""
    scenarios = []

    if not MORDOR_DIR.exists():
        return scenarios

    for yaml_path in sorted(MORDOR_DIR.rglob("*.yaml")):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                metadata = yaml.safe_load(f)
        except Exception:
            continue

        if not isinstance(metadata, dict):
            continue

        title = metadata.get("title", yaml_path.stem)
        description = metadata.get("description", "")
        technique = metadata.get("attack_technique", "")
        platform = metadata.get("platform", "Windows")
        adversary = metadata.get("adversary", "")

        # Find associated JSON event log
        json_dir = yaml_path.parent
        json_files = sorted(json_dir.glob("*.json"))

        scenarios.append(
            {
                "title": title,
                "description": description,
                "technique": technique,
                "platform": platform,
                "adversary": adversary,
                "yaml_path": yaml_path,
                "json_files": json_files,
            }
        )

    return scenarios


# ---------------------------------------------------------------------------
# Sample event logs for prompt
# ---------------------------------------------------------------------------
def sample_events(json_files: list[Path], max_events: int = 10) -> str:
    """Sample event log entries for inclusion in the prompt."""
    events = []
    for jf in json_files[:3]:  # max 3 files
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if isinstance(data, list):
            events.extend(data[:max_events])
        elif isinstance(data, dict):
            events.append(data)

    if not events:
        return "[No event logs available]"

    # Format a sample
    sample = events[:max_events]
    return json.dumps(sample, indent=2)


# ---------------------------------------------------------------------------
# Build training pairs
# ---------------------------------------------------------------------------
def build_pairs(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    """Build 2-3 training pairs from a Mordor scenario."""
    pairs = []
    title = scenario["title"]
    description = scenario["description"]
    technique = scenario["technique"]
    platform = scenario["platform"]
    adversary = scenario["adversary"]
    event_sample = sample_events(scenario["json_files"])

    mitre_ids = [technique] if technique else []
    technique_str = technique if technique else "the appropriate MITRE ATT&CK technique"

    # Pair 1: Technique identification
    user1 = (
        f"Analyze these {platform} event logs. What MITRE ATT&CK technique is "
        f"being executed? What specific indicators confirm your assessment?\n\n"
        f"```json\n{event_sample[:2000]}\n```"
    )

    assistant1 = (
        f"## Technique Identification: {title}\n"
        f"**MITRE ATT&CK:** {technique_str}\n"
        f"**Platform:** {platform}\n"
        + (f"**Adversary:** {adversary}\n" if adversary else "")
        + f"\n{description.strip()}\n\n"
        f"### Key Indicators\n"
        f"- Event log patterns consistent with {technique_str}\n"
        f"- Process creation events showing suspicious parent-child relationships\n"
        f"- Network connections to unusual external IPs or ports\n"
        f"- File system modifications in sensitive directories\n"
        f"- Registry key modifications indicating persistence mechanisms\n\n"
        f"### Confidence Assessment\n"
        f"High confidence based on multiple correlated indicators across "
        f"process, network, and file system telemetry."
    )

    pairs.append(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": user1},
                {"role": "assistant", "content": assistant1},
            ],
            "mitre_ids": mitre_ids,
            **ATTRIBUTION,
        }
    )

    # Pair 2: Detection query
    user2 = (
        f"Write a {platform} detection query for {technique_str}. Include data "
        f"sources, field mappings, and expected false positive scenarios."
    )

    assistant2 = (
        f"## Detection Query for {technique_str}\n\n"
        f"### Data Sources\n"
        f"- Windows Security Event Logs (Event ID 4688, 5156, 4663)\n"
        f"- Sysmon Event Logs (Event ID 1, 3, 11)\n"
        f"- Endpoint Detection and Response (EDR) telemetry\n\n"
        f"### KQL Query (Microsoft 365 Defender / Azure Sentinel)\n"
        f"```kql\n"
        f"DeviceProcessEvents\n"
        f'| where ProcessCommandLine contains "suspicious_pattern"\n'
        f'| where InitiatingProcessFileName != "expected_parent.exe"\n'
        f"| project Timestamp, DeviceName, ProcessCommandLine, InitiatingProcessFileName\n"
        f"```\n\n"
        f"### Splunk SPL Query\n"
        f"```spl\n"
        f"index=windows EventCode=4688\n"
        f"| search Process_Command_Line=*suspicious*\n"
        f"| table _time, host, Process_Command_Line, Parent_Process_Name\n"
        f"```\n\n"
        f"### Expected False Positives\n"
        f"- Legitimate administrative tools with similar command-line patterns\n"
        f"- Software deployment and update processes\n"
        f"- Security tooling performing similar actions for validation\n\n"
        f"### Tuning Recommendations\n"
        f"Add allowlist entries for known administrative accounts and "
        f"scheduled tasks. Correlate with vulnerability scan results to "
        f"reduce noise from authorized penetration tests."
    )

    pairs.append(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": user2},
                {"role": "assistant", "content": assistant2},
            ],
            "mitre_ids": mitre_ids,
            **ATTRIBUTION,
        }
    )

    # Pair 3: Hunting methodology
    user3 = (
        f"Describe a threat hunting methodology for detecting {technique_str} "
        f"at scale in a {platform} environment."
    )

    assistant3 = (
        f"## Threat Hunting Methodology: {technique_str}\n\n"
        f"### Hypothesis\n"
        f"An adversary is using {technique_str} to achieve their objectives "
        f"within the {platform} environment.\n\n"
        f"### Data Requirements\n"
        f"- Process creation logs with full command-line arguments\n"
        f"- Network connection logs with process correlation\n"
        f"- File system audit logs for sensitive directories\n"
        f"- Registry modification audit logs\n\n"
        f"### Hunting Steps\n"
        f"1. **Baseline Establishment**: Collect 30 days of normal activity "
        f"to establish baseline patterns for process creation, network "
        f"connections, and file modifications.\n"
        f"2. **Anomaly Detection**: Query for deviations from baseline — "
        f"unusual parent-child process relationships, processes spawned "
        f"from temporary directories, network connections to rare IPs.\n"
        f"3. **Stack Counting**: Group by process command-line patterns "
        f"and flag the least frequent (bottom 1%) for investigation.\n"
        f"4. **Time-Based Analysis**: Look for activity outside normal "
        f"business hours or clustered in short time windows.\n"
        f"5. **Cross-Source Correlation**: Correlate process events with "
        f"network connections and file modifications occurring within "
        f"±5 minutes of each other.\n\n"
        f"### Expected Artifacts\n"
        f"- Process creation events with suspicious command-line arguments\n"
        f"- Network connections to known C2 infrastructure or unusual ports\n"
        f"- File modifications in system directories or user profile paths\n"
        f"- Registry modifications to Run keys, services, or scheduled tasks\n\n"
        f"### Escalation Criteria\n"
        f"Escalate to incident response if 3+ indicators are observed on "
        f"a single host within a 1-hour window, or if the same pattern "
        f"appears on 5+ hosts."
    )

    pairs.append(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": user3},
                {"role": "assistant", "content": assistant3},
            ],
            "mitre_ids": mitre_ids,
            **ATTRIBUTION,
        }
    )

    # Add tactic info to all pairs
    for pair in pairs:
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

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract Mordor security event logs into AttackLM training pairs"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print pairs without writing"
    )
    parser.add_argument(
        "--max-files", type=int, default=0, help="Max scenarios to process (0=all)"
    )
    args = parser.parse_args()

    scenarios = find_scenarios()
    if not scenarios:
        print(f"No Mordor scenarios found in {MORDOR_DIR}", file=sys.stderr)
        print("Run attacklm init --clone-only or attacklm init first.", file=sys.stderr)
        return 1

    if args.max_files > 0:
        scenarios = scenarios[: args.max_files]

    print(f"Processing {len(scenarios)} Mordor scenarios...", file=sys.stderr)

    all_pairs: list[dict[str, Any]] = []
    for scenario in scenarios:
        pairs = build_pairs(scenario)
        all_pairs.extend(pairs)

    print(
        f"Generated {len(all_pairs)} pairs from {len(scenarios)} scenarios",
        file=sys.stderr,
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
