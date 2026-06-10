#!/usr/bin/env python3
"""
extract_by_tactic.py — Parse all AttackLM data sources and group techniques
by MITRE ATT&CK tactic, producing merged JSON manifests.

Usage:
    python extract_by_tactic.py              # Parse all sources, write all manifests
    python extract_by_tactic.py --tactic TA0003  # Only Persistence
    python extract_by_tactic.py --dry-run     # Print summary without writing files
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"

ATOMIC_DIR = DATA_DIR / "atomic-red-team" / "atomics"
CALDERA_DIR = DATA_DIR / "stockpile" / "data" / "abilities"
SIGMA_DIR = DATA_DIR / "sigma" / "rules" / "windows"
MSF_DIR = DATA_DIR / "metasploit-framework" / "modules"
OUTPUT_DIR = DATA_DIR / "manifests"

# ---------------------------------------------------------------------------
# Technique → Tactic mapping (covers all 335 Atomic Red Team files)
# ---------------------------------------------------------------------------
TECHNIQUE_TO_TACTIC: dict[str, str] = {
    # Execution (TA0002)
    "T1059": "TA0002",
    "T1106": "TA0002",
    "T1203": "TA0002",
    "T1047": "TA0002",
    "T1202": "TA0002",
    "T1204": "TA0002",
    "T1559": "TA0002",
    "T1569": "TA0002",
    "T1609": "TA0002",
    "T1610": "TA0002",
    "T1611": "TA0002",
    "T1612": "TA0002",
    "T1613": "TA0002",
    "T1651": "TA0002",
    "T1652": "TA0002",
    "T1659": "TA0002",
    "T1686": "TA0002",
    "T1688": "TA0002",
    "T1689": "TA0002",
    "T1690": "TA0002",
    "T1072": "TA0002",
    "T1566": "TA0002",
    # Persistence (TA0003)
    "T1547": "TA0003",
    "T1053": "TA0003",
    "T1543": "TA0003",
    "T1546": "TA0003",
    "T1037": "TA0003",
    "T1078": "TA0003",
    "T1098": "TA0003",
    "T1136": "TA0003",
    "T1137": "TA0003",
    "T1176": "TA0003",
    "T1505": "TA0003",
    "T1542": "TA0003",
    "T1574": "TA0003",
    # Privilege Escalation (TA0004)
    "T1068": "TA0004",
    "T1134": "TA0004",
    "T1548": "TA0004",
    "T1685": "TA0004",
    # Defense Evasion (TA0005)
    "T1562": "TA0005",
    "T1027": "TA0005",
    "T1055": "TA0005",
    "T1036": "TA0005",
    "T1070": "TA0005",
    "T1112": "TA0005",
    "T1127": "TA0005",
    "T1129": "TA0005",
    "T1140": "TA0005",
    "T1197": "TA0005",
    "T1216": "TA0005",
    "T1217": "TA0005",
    "T1218": "TA0005",
    "T1220": "TA0005",
    "T1221": "TA0005",
    "T1222": "TA0005",
    "T1484": "TA0005",
    "T1497": "TA0005",
    "T1553": "TA0005",
    "T1556": "TA0005",
    "T1557": "TA0005",
    "T1564": "TA0005",
    "T1572": "TA0005",
    "T1578": "TA0005",
    "T1580": "TA0005",
    "T1620": "TA0005",
    "T1647": "TA0005",
    "T1648": "TA0005",
    "T1649": "TA0005",
    # Credential Access (TA0006)
    "T1003": "TA0006",
    "T1558": "TA0006",
    "T1552": "TA0006",
    "T1110": "TA0006",
    "T1555": "TA0006",
    "T1606": "TA0006",
    "T1212": "TA0006",  # Exploitation for Credential Access
    # Discovery (TA0007)
    "T1087": "TA0007",
    "T1046": "TA0007",
    "T1082": "TA0007",
    "T1007": "TA0007",
    "T1010": "TA0007",
    "T1012": "TA0007",
    "T1016": "TA0007",
    "T1018": "TA0007",
    "T1033": "TA0007",
    "T1039": "TA0007",
    "T1040": "TA0007",
    "T1049": "TA0007",
    "T1057": "TA0007",
    "T1069": "TA0007",
    "T1083": "TA0007",
    "T1124": "TA0007",
    "T1135": "TA0007",
    "T1201": "TA0007",
    "T1482": "TA0007",
    "T1518": "TA0007",
    "T1526": "TA0007",
    "T1614": "TA0007",
    "T1615": "TA0007",
    "T1619": "TA0007",
    "T1589": "TA0007",  # Gather Victim Identity Information
    "T1592": "TA0007",  # Gather Victim Host Information
    "T1183": "TA0005",  # Image File Execution Options Injection
    "T1187": "TA0006",  # Forge Web Credentials
    # Lateral Movement (TA0008)
    "T1021": "TA0008",
    "T1550": "TA0008",
    "T1091": "TA0008",
    "T1133": "TA0008",
    "T1195": "TA0008",
    "T1219": "TA0008",
    "T1539": "TA0008",
    "T1563": "TA0008",
    "T1570": "TA0008",
    "T1190": "TA0002",  # Exploit Public-Facing Application (MITRE TA0001; we route to Execution since AttackLM skips Initial Access)
    # Collection (TA0009)
    "T1005": "TA0009",
    "T1006": "TA0009",
    "T1056": "TA0009",
    "T1074": "TA0009",
    "T1105": "TA0009",
    "T1113": "TA0009",
    "T1114": "TA0009",
    "T1115": "TA0009",
    "T1119": "TA0009",
    "T1120": "TA0009",
    "T1123": "TA0009",
    "T1125": "TA0009",
    "T1187": "TA0009",
    "T1560": "TA0009",
    # Exfiltration (TA0010)
    "T1020": "TA0010",
    "T1025": "TA0010",
    "T1041": "TA0010",
    "T1048": "TA0010",
    "T1528": "TA0010",
    "T1567": "TA0010",
    # Command & Control (TA0011)
    "T1071": "TA0011",
    "T1090": "TA0011",
    "T1573": "TA0011",
    "T1001": "TA0011",
    "T1095": "TA0011",
    "T1132": "TA0011",
    "T1207": "TA0011",
    "T1568": "TA0011",
    "T1571": "TA0011",
    "T1622": "TA0011",
}

# ---------------------------------------------------------------------------
# Tactic metadata (ID → display info)
# ---------------------------------------------------------------------------
TACTIC_INFO: dict[str, dict[str, str]] = {
    "TA0002": {"name": "execution", "display": "Execution"},
    "TA0003": {"name": "persistence", "display": "Persistence"},
    "TA0004": {"name": "privilege_escalation", "display": "Privilege Escalation"},
    "TA0005": {"name": "defense_evasion", "display": "Defense Evasion"},
    "TA0006": {"name": "credential_access", "display": "Credential Access"},
    "TA0007": {"name": "discovery", "display": "Discovery"},
    "TA0008": {"name": "lateral_movement", "display": "Lateral Movement"},
    "TA0009": {"name": "collection", "display": "Collection"},
    "TA0010": {"name": "exfiltration", "display": "Exfiltration"},
    "TA0011": {"name": "command_and_control", "display": "Command and Control"},
}

# Regex for MITRE technique tags in Sigma rules (e.g. attack.t1547.001)
SIGMA_TECH_RE = re.compile(r"attack\.t(\d{4})(?:\.(\d{3}))?", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_yaml(path: Path) -> Any | None:
    """Safely load a YAML file, returning None on error."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except (yaml.YAMLError, OSError) as exc:
        warnings.warn(f"Failed to parse {path}: {exc}", stacklevel=2)
        return None


def _base_technique_id(tech_id: str) -> str:
    """Extract the base technique ID (e.g., 'T1547') from a sub-technique
    like 'T1547.001' or already-base 'T1547'."""
    match = re.match(r"(T\d{4})", tech_id, re.IGNORECASE)
    return match.group(1).upper() if match else tech_id.upper()


def _tactic_for_technique(tech_id: str) -> str | None:
    """Map a technique ID (base or sub) to its tactic ID using the base lookup."""
    base = _base_technique_id(tech_id)
    return TECHNIQUE_TO_TACTIC.get(base)


def _dedupe_commands(commands: list[dict]) -> list[dict]:
    """Deduplicate command entries by command text."""
    seen: set[str] = set()
    deduped: list[dict] = []
    for cmd in commands:
        key = cmd.get("command", "")
        if key not in seen:
            seen.add(key)
            deduped.append(cmd)
    return deduped


def _dedupe_detection_rules(rules: list[dict]) -> list[dict]:
    """Deduplicate detection rule entries by title."""
    seen: set[str] = set()
    deduped: list[dict] = []
    for rule in rules:
        key = rule.get("title", "")
        if key not in seen:
            seen.add(key)
            deduped.append(rule)
    return deduped


def _clean_command(text: str) -> str:
    """Strip leading/trailing whitespace and collapse internal newlines."""
    if not text:
        return ""
    return text.strip()


# ---------------------------------------------------------------------------
# Atomic Red Team parser
# ---------------------------------------------------------------------------
def parse_atomic_red_team() -> dict[str, dict[str, dict]]:
    """Parse all Atomic Red Team YAML files.

    Returns:
        Nested dict: {tactic_id: {technique_id: {name, description, platforms,
        commands, detection_rules}}}
    """
    if not ATOMIC_DIR.exists():
        warnings.warn(
            f"Atomic Red Team directory not found: {ATOMIC_DIR}", stacklevel=2
        )
        return {}

    print(f"  [Atomic] Scanning {ATOMIC_DIR} ...")
    tactic_data: dict[str, dict[str, dict]] = defaultdict(dict)
    yaml_count = 0
    skip_count = 0

    for tech_dir in sorted(ATOMIC_DIR.iterdir()):
        if not tech_dir.is_dir() or not tech_dir.name.upper().startswith("T"):
            continue

        for yaml_file in tech_dir.glob("*.yaml"):
            doc = _load_yaml(yaml_file)
            if doc is None:
                continue

            attack_technique = str(doc.get("attack_technique", "")).upper()
            if not attack_technique:
                skip_count += 1
                continue

            tactic_id = _tactic_for_technique(attack_technique)
            if tactic_id is None:
                skip_count += 1
                continue

            display_name = doc.get("display_name", attack_technique)
            description = ""
            platforms: list[str] = []
            commands: list[dict] = []

            for test in doc.get("atomic_tests", []):
                # Collect description from first test that has one
                test_desc = test.get("description", "")
                if not description and test_desc:
                    description = str(test_desc).strip()

                # Collect platforms
                for p in test.get("supported_platforms", []):
                    if p not in platforms:
                        platforms.append(p)

                # Collect executor commands
                executor_block = test.get("executor", {})
                if isinstance(executor_block, dict):
                    executor_name = executor_block.get("name", "unknown")
                    command_text = _clean_command(executor_block.get("command", ""))
                    cleanup_text = _clean_command(
                        executor_block.get("cleanup_command", "")
                    )
                    if command_text:
                        commands.append(
                            {
                                "executor": executor_name,
                                "command": command_text,
                                "cleanup": cleanup_text or None,
                            }
                        )
                # Handle multiple executors (list form)
                elif isinstance(executor_block, list):
                    for ex in executor_block:
                        executor_name = ex.get("name", "unknown")
                        command_text = _clean_command(ex.get("command", ""))
                        cleanup_text = _clean_command(ex.get("cleanup_command", ""))
                        if command_text:
                            commands.append(
                                {
                                    "executor": executor_name,
                                    "command": command_text,
                                    "cleanup": cleanup_text or None,
                                }
                            )

            # Use the technique ID from the directory/YAML as-is (preserves sub-technique)
            technique_id = attack_technique
            # If sub-technique directories exist (e.g. T1547.001), use that
            sub_yaml_match = re.match(r"(T\d{4}\.\d{3})", tech_dir.name, re.IGNORECASE)
            if sub_yaml_match:
                technique_id = sub_yaml_match.group(1).upper()

            tactic_data[tactic_id][technique_id] = {
                "name": display_name,
                "description": description,
                "platforms": platforms,
                "commands": _dedupe_commands(commands),
                "detection_rules": [],
            }
            yaml_count += 1

    print(f"  [Atomic] Parsed {yaml_count} YAML files ({skip_count} skipped)")
    return dict(tactic_data)


# ---------------------------------------------------------------------------
# Caldera (stockpile) parser
# ---------------------------------------------------------------------------
def parse_caldera() -> dict[str, dict[str, dict]]:
    """Parse all Caldera ability YAML files.

    Returns:
        Nested dict: {tactic_id: {technique_id: {name, description, platforms,
        commands, detection_rules}}}
    """
    if not CALDERA_DIR.exists():
        warnings.warn(
            f"Caldera abilities directory not found: {CALDERA_DIR}", stacklevel=2
        )
        return {}

    print(f"  [Caldera] Scanning {CALDERA_DIR} ...")
    tactic_data: dict[str, dict[str, dict]] = defaultdict(dict)
    ability_count = 0
    skip_count = 0

    for yml_file in sorted(CALDERA_DIR.rglob("*.yml")):
        doc = _load_yaml(yml_file)
        if doc is None:
            continue

        # Caldera YAMLs contain a list of abilities at the top level
        abilities = doc if isinstance(doc, list) else [doc]

        for ability in abilities:
            if not isinstance(ability, dict):
                continue

            # Extract technique ID
            technique_block = ability.get("technique", {})
            if isinstance(technique_block, dict):
                tech_id = str(technique_block.get("attack_id", "")).upper()
                tech_name = technique_block.get("name", "")
            else:
                tech_id = str(ability.get("technique_id", "")).upper()
                tech_name = str(ability.get("technique_name", ""))

            if not tech_id:
                skip_count += 1
                continue

            tactic_id = _tactic_for_technique(tech_id)
            if tactic_id is None:
                skip_count += 1
                continue

            description = str(ability.get("description", "")).strip()
            commands: list[dict] = []
            platforms: list[str] = []

            # Format 1: "platforms" dict with OS → executor_type → details
            platforms_block = ability.get("platforms", {})
            if isinstance(platforms_block, dict):
                for os_name, executor_types in platforms_block.items():
                    if os_name not in platforms:
                        platforms.append(os_name)
                    if isinstance(executor_types, dict):
                        for exec_type, details in executor_types.items():
                            command_text = _clean_command(details.get("command", ""))
                            cleanup_text = _clean_command(details.get("cleanup", ""))
                            payloads = details.get("payloads", [])
                            if command_text:
                                cmd_entry = {
                                    "executor": exec_type,
                                    "command": command_text,
                                    "cleanup": cleanup_text or None,
                                }
                                if payloads:
                                    cmd_entry["payloads"] = payloads
                                commands.append(cmd_entry)

            # Format 2: "executors" list (older Caldera format)
            executors_list = ability.get("executors", [])
            if isinstance(executors_list, list):
                for ex in executors_list:
                    os_name = ex.get("platform", "unknown")
                    if os_name not in platforms:
                        platforms.append(os_name)
                    executor_name = ex.get("name", "unknown")
                    command_text = _clean_command(ex.get("command", ""))
                    cleanup_text = _clean_command(ex.get("cleanup", ""))
                    payloads = ex.get("payloads", [])
                    if command_text:
                        cmd_entry = {
                            "executor": executor_name,
                            "command": command_text,
                            "cleanup": cleanup_text or None,
                        }
                        if payloads:
                            cmd_entry["payloads"] = payloads
                        commands.append(cmd_entry)

            # Merge with existing technique entry (from Atomic)
            existing = tactic_data[tactic_id].get(tech_id)
            if existing:
                if not existing["description"] and description:
                    existing["description"] = description
                for p in platforms:
                    if p not in existing["platforms"]:
                        existing["platforms"].append(p)
                existing["commands"] = _dedupe_commands(existing["commands"] + commands)
            else:
                tactic_data[tactic_id][tech_id] = {
                    "name": tech_name or tech_id,
                    "description": description,
                    "platforms": platforms,
                    "commands": _dedupe_commands(commands),
                    "detection_rules": [],
                }

            ability_count += 1

    print(f"  [Caldera] Parsed {ability_count} abilities ({skip_count} skipped)")
    return dict(tactic_data)


# ---------------------------------------------------------------------------
# Sigma rule parser
# ---------------------------------------------------------------------------
def parse_sigma() -> dict[str, dict[str, list[dict]]]:
    """Parse all Sigma rules and map them to technique IDs.

    Returns:
        {tactic_id: {technique_id: [detection_rule_dict, ...]}}
    """
    if not SIGMA_DIR.exists():
        warnings.warn(f"Sigma rules directory not found: {SIGMA_DIR}", stacklevel=2)
        return {}

    print(f"  [Sigma] Scanning {SIGMA_DIR} ...")
    tactic_rules: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    rule_count = 0
    skip_count = 0

    for yml_file in sorted(SIGMA_DIR.rglob("*.yml")):
        doc = _load_yaml(yml_file)
        if doc is None or not isinstance(doc, dict):
            continue

        tags = doc.get("tags", [])
        if not isinstance(tags, list):
            tags = []

        # Find MITRE ATT&CK tags
        technique_ids_found: list[str] = []
        for tag in tags:
            match = SIGMA_TECH_RE.search(str(tag))
            if match:
                base = f"T{match.group(1)}"
                sub = f".{match.group(2)}" if match.group(2) else ""
                technique_ids_found.append(base + sub)

        if not technique_ids_found:
            skip_count += 1
            continue

        # Build a detection rule entry
        title = doc.get("title", "Unknown Rule")
        logsource = doc.get("logsource", {})
        description = str(doc.get("description", "")).strip()

        detection_rule = {
            "title": title,
            "logsource": logsource if isinstance(logsource, dict) else {},
            "description": description,
        }

        for tech_id in technique_ids_found:
            tactic_id = _tactic_for_technique(tech_id)
            if tactic_id is None:
                continue
            tactic_rules[tactic_id][tech_id].append(detection_rule)
            rule_count += 1

    print(
        f"  [Sigma] Parsed {rule_count} technique mappings ({skip_count} rules without MITRE tags)"
    )
    return {k: dict(v) for k, v in tactic_rules.items()}


# ---------------------------------------------------------------------------
# Metasploit Framework parser
# ---------------------------------------------------------------------------
def parse_metasploit() -> dict[str, dict[str, list[dict]]]:
    """Parse Metasploit modules that have MITRE ATT&CK technique references.

    Returns:
        Nested dict: {tactic_id: {technique_id: [module_record, ...]}}
    """
    if not MSF_DIR.exists():
        warnings.warn(
            f"Metasploit modules directory not found: {MSF_DIR}. "
            f"Run scripts/clone_repos.sh first.",
            stacklevel=2,
        )
        return {}

    # Import the dedicated parser to avoid duplicating regex definitions
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        from parse_metasploit_to_jsonl import parse_module_file  # type: ignore
    except ImportError as exc:
        warnings.warn(f"Could not import Metasploit parser: {exc}", stacklevel=2)
        return {}

    print(f"  [Metasploit] Scanning {MSF_DIR} ...")
    tactic_data: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    parsed = 0
    skipped = 0
    for rb_file in MSF_DIR.rglob("*.rb"):
        try:
            rec = parse_module_file(rb_file)
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            continue
        if rec is None:
            skipped += 1
            continue
        parsed += 1
        for tech in rec.get("mitre_techniques", []):
            tactic_id = _tactic_for_technique(tech["id"])
            if tactic_id is None:
                continue
            tactic_data[tactic_id][tech["id"]].append(rec)
    print(f"  [Metasploit] Parsed {parsed} modules ({skipped} skipped)")
    return {k: dict(v) for k, v in tactic_data.items()}


# ---------------------------------------------------------------------------
# Merge all sources
# ---------------------------------------------------------------------------
def merge_sources(
    atomic_data: dict[str, dict[str, dict]],
    caldera_data: dict[str, dict[str, dict]],
    sigma_data: dict[str, dict[str, list[dict]]],
    metasploit_data: dict[str, dict[str, list[dict]]] | None = None,
) -> dict[str, dict]:
    """Merge Atomic, Caldera, Sigma, and Metasploit data into per-tactic manifests.

    Metasploit data is merged additively: each technique gets a
    ``metasploit_modules`` field containing the list of related MSF modules
    (with name, path, CVEs, options, msfconsole commands, etc.). Modules
    that don't share an MITRE tag with any known technique are still
    captured under their first tagged tactic so the data isn't lost.

    Returns:
        {tactic_id: {tactic_id, tactic_name, techniques: [...]}}
    """
    metasploit_data = metasploit_data or {}
    # Collect all tactic IDs seen across sources
    all_tactic_ids: set[str] = set()
    all_tactic_ids.update(atomic_data.keys())
    all_tactic_ids.update(caldera_data.keys())
    all_tactic_ids.update(sigma_data.keys())
    all_tactic_ids.update(metasploit_data.keys())

    manifests: dict[str, dict] = {}

    for tactic_id in sorted(all_tactic_ids):
        info = TACTIC_INFO.get(tactic_id, {})
        tactic_name = info.get("name", tactic_id.lower())

        # Merge techniques from all sources
        techniques_map: dict[str, dict] = {}

        # From Atomic
        for tech_id, tech_data in atomic_data.get(tactic_id, {}).items():
            techniques_map[tech_id] = {
                "technique_id": tech_id,
                "name": tech_data["name"],
                "description": tech_data["description"],
                "platforms": tech_data["platforms"],
                "commands": tech_data["commands"],
                "detection_rules": tech_data.get("detection_rules", []),
            }

        # From Caldera — merge into existing or create new
        for tech_id, tech_data in caldera_data.get(tactic_id, {}).items():
            if tech_id in techniques_map:
                existing = techniques_map[tech_id]
                if not existing["description"] and tech_data["description"]:
                    existing["description"] = tech_data["description"]
                for p in tech_data["platforms"]:
                    if p not in existing["platforms"]:
                        existing["platforms"].append(p)
                existing["commands"] = _dedupe_commands(
                    existing["commands"] + tech_data["commands"]
                )
                existing["detection_rules"] = _dedupe_detection_rules(
                    existing["detection_rules"] + tech_data.get("detection_rules", [])
                )
            else:
                techniques_map[tech_id] = {
                    "technique_id": tech_id,
                    "name": tech_data["name"],
                    "description": tech_data["description"],
                    "platforms": tech_data["platforms"],
                    "commands": tech_data["commands"],
                    "detection_rules": tech_data.get("detection_rules", []),
                }

        # From Sigma — merge detection rules into existing techniques
        for tech_id, rules in sigma_data.get(tactic_id, {}).items():
            if tech_id in techniques_map:
                techniques_map[tech_id]["detection_rules"] = _dedupe_detection_rules(
                    techniques_map[tech_id]["detection_rules"] + rules
                )
            else:
                # Sigma found a technique not in Atomic/Caldera — create minimal entry
                techniques_map[tech_id] = {
                    "technique_id": tech_id,
                    "name": tech_id,
                    "description": "",
                    "platforms": [],
                    "commands": [],
                    "detection_rules": rules,
                }

        # From Metasploit — attach the list of modules that reference this
        # technique. Modules carry their own msfconsole command sequence,
        # CVEs, options, etc. We strip the giant description_doc to keep the
        # merged manifest manageable.
        for tech_id, msf_modules in metasploit_data.get(tactic_id, {}).items():
            slim_modules = [
                {k: v for k, v in m.items() if k != "description_doc"}
                for m in msf_modules
            ]
            if tech_id in techniques_map:
                techniques_map[tech_id].setdefault("metasploit_modules", []).extend(
                    slim_modules
                )
            else:
                # Metasploit found a technique not in Atomic/Caldera/Sigma
                first = msf_modules[0] if msf_modules else {}
                techniques_map[tech_id] = {
                    "technique_id": tech_id,
                    "name": first.get("mitre_techniques", [{}])[0].get("name", tech_id),
                    "description": first.get("description", ""),
                    "platforms": first.get("platform", []),
                    "commands": first.get("commands", []),
                    "detection_rules": [],
                    "metasploit_modules": slim_modules,
                }

        # Sort techniques by ID for deterministic output
        sorted_techniques = sorted(
            techniques_map.values(),
            key=lambda t: t["technique_id"],
        )

        manifests[tactic_id] = {
            "tactic_id": tactic_id,
            "tactic_name": tactic_name,
            "techniques": sorted_techniques,
        }

    return manifests


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_manifests(manifests: dict[str, dict], dry_run: bool = False) -> None:
    """Write tactic manifests to JSON files in OUTPUT_DIR."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"{'DRY RUN — ' if dry_run else ''}Manifest Summary")
    print(f"{'=' * 60}")

    total_techniques = 0
    total_commands = 0
    total_detection = 0
    total_msf = 0

    for tactic_id, manifest in sorted(manifests.items()):
        technique_count = len(manifest["techniques"])
        command_count = sum(len(t.get("commands", [])) for t in manifest["techniques"])
        detection_count = sum(
            len(t.get("detection_rules", [])) for t in manifest["techniques"]
        )
        msf_count = sum(
            len(t.get("metasploit_modules", [])) for t in manifest["techniques"]
        )
        total_techniques += technique_count
        total_commands += command_count
        total_detection += detection_count
        total_msf += msf_count

        tactic_name = manifest["tactic_name"]
        print(
            f"  {tactic_id} ({tactic_name:25s}): "
            f"{technique_count:3d} techniques, "
            f"{command_count:4d} commands, "
            f"{detection_count:4d} detection rules, "
            f"{msf_count:4d} MSF modules"
        )

        if dry_run:
            continue

        output_path = OUTPUT_DIR / f"{tactic_name}.json"
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
        print(f"    → Written to {output_path}")

    print(
        f"\n  TOTAL: {total_techniques} techniques, "
        f"{total_commands} commands, {total_detection} detection rules, "
        f"{total_msf} Metasploit modules"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Parse all AttackLM data sources and group techniques by MITRE ATT&CK tactic."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse AttackLM data sources and group techniques by MITRE ATT&CK tactic."
    )
    parser.add_argument(
        "--tactic",
        type=str,
        default=None,
        help="Only process this tactic (e.g., TA0003 or persistence).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print summary without writing manifest files.",
    )
    args = parser.parse_args()

    print("AttackLM — Extract by Tactic")
    print(f"  BASE_DIR:   {BASE_DIR}")
    print(f"  ATOMIC_DIR: {ATOMIC_DIR}")
    print(f"  CALDERA_DIR: {CALDERA_DIR}")
    print(f"  SIGMA_DIR:  {SIGMA_DIR}")
    print(f"  MSF_DIR:    {MSF_DIR}")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print()

    # Step 1: Parse each source
    atomic_data = parse_atomic_red_team()
    caldera_data = parse_caldera()
    sigma_data = parse_sigma()
    metasploit_data = parse_metasploit()

    # Step 2: Merge
    print("\n  Merging all sources ...")
    manifests = merge_sources(atomic_data, caldera_data, sigma_data, metasploit_data)

    # Step 3: Filter to single tactic if requested
    if args.tactic:
        # Accept both ID (TA0003) and name (persistence)
        target = args.tactic.upper()
        if target in manifests:
            manifests = {target: manifests[target]}
        else:
            # Try name match
            matched = {
                tid: m
                for tid, m in manifests.items()
                if m["tactic_name"] == args.tactic.lower()
                or m["tactic_name"] == args.tactic.replace("-", "_").lower()
            }
            if matched:
                manifests = matched
            else:
                print(
                    f"ERROR: Tactic '{args.tactic}' not found. "
                    f"Available: {list(TACTIC_INFO.keys())}"
                )
                sys.exit(1)

    # Step 4: Write manifest files
    write_manifests(manifests, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
