#!/usr/bin/env python3
"""AttackLM Dataset Bias Audit Script.

Scans ALL buckets in data/datasets/buckets/, extracts MITRE technique IDs
from structured fields and text content, and produces a comprehensive bias
report covering technique distribution, coverage gaps, source skew, and
per-tactic heatmaps.

Usage:
    python scripts/audit_dataset.py
    python scripts/audit_dataset.py --output data/audit_report.json
    python scripts/audit_dataset.py --root /path/to/AttackLM
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Regex patterns for extracting MITRE technique IDs from text content.
# Matches T1xxx(.yyy) and AML.Txxxx patterns.
_TECHNIQUE_RE = re.compile(r"\b(T\d{4}(?:\.\d{3})?)\b", re.IGNORECASE)
_ATLAS_RE = re.compile(r"\b(AML\.T\d{4}(?:\.\d{3})?)\b", re.IGNORECASE)
# Pattern for "**Technique: Name (T1234.001)**" style references in markdown
_TECHNIQUE_MARKDOWN_RE = re.compile(
    r"\*\*Technique:[^*]*?\(([Tt]\d{4}(?:\.\d{3})?)\)\*\*"
)

# MITRE ATT&CK Enterprise tactic IDs and names (the canonical 14)
MITRE_TACTICS: dict[str, str] = {
    "TA0042": "Resource Development",
    "TA0043": "Reconnaissance",
    "TA0001": "Initial Access",
    "TA0002": "Execution",
    "TA0003": "Persistence",
    "TA0004": "Privilege Escalation",
    "TA0005": "Defense Evasion",
    "TA0006": "Credential Access",
    "TA0007": "Discovery",
    "TA0008": "Lateral Movement",
    "TA0009": "Collection",
    "TA0010": "Exfiltration",
    "TA0011": "Command and Control",
    "TA0012": "Impact",
    "TA0040": "Prompt Injection",
}

# Technique → primary tactic mapping.
# Imported from the project's own mitre_tactic_lookup.py for consistency.
# This is a curated subset of the most common techniques; techniques not
# listed here will be classified via _FALLBACK_RANGES.
TECHNIQUE_TO_TACTIC: dict[str, str] = {
    "T1078.001": "TA0001",
    "T1078.003": "TA0001",
    "T1078.004": "TA0001",
    "T1566.001": "TA0001",
    "T1566.002": "TA0001",
    "T1566.003": "TA0001",
    "T1195": "TA0001",
    "T1195.002": "TA0001",
    "T1190": "TA0001",
    "T1133": "TA0001",
    "T1200": "TA0001",
    "T1091": "TA0001",
    "T1189": "TA0001",
    "T1199": "TA0001",
    "T1078": "TA0001",
    "T1059": "TA0002",
    "T1059.001": "TA0002",
    "T1059.002": "TA0002",
    "T1059.003": "TA0002",
    "T1059.004": "TA0002",
    "T1059.005": "TA0002",
    "T1059.006": "TA0002",
    "T1059.007": "TA0002",
    "T1059.010": "TA0002",
    "T1053": "TA0002",
    "T1053.002": "TA0002",
    "T1053.003": "TA0002",
    "T1053.005": "TA0002",
    "T1053.006": "TA0002",
    "T1053.007": "TA0002",
    "T1204": "TA0002",
    "T1204.002": "TA0002",
    "T1204.003": "TA0002",
    "T1035": "TA0002",
    "T1121": "TA0002",
    "T1559": "TA0002",
    "T1559.002": "TA0002",
    "T1609": "TA0002",
    "T1610": "TA0002",
    "T1543": "TA0003",
    "T1543.001": "TA0003",
    "T1543.002": "TA0003",
    "T1543.003": "TA0003",
    "T1543.004": "TA0003",
    "T1547": "TA0003",
    "T1547.001": "TA0003",
    "T1136": "TA0003",
    "T1136.001": "TA0003",
    "T1136.002": "TA0003",
    "T1136.003": "TA0003",
    "T1137": "TA0003",
    "T1546": "TA0003",
    "T1574": "TA0003",
    "T1574.001": "TA0003",
    "T1574.006": "TA0003",
    "T1574.008": "TA0003",
    "T1574.009": "TA0003",
    "T1574.010": "TA0003",
    "T1574.011": "TA0003",
    "T1574.012": "TA0003",
    "T1548": "TA0004",
    "T1548.001": "TA0004",
    "T1548.002": "TA0004",
    "T1548.003": "TA0004",
    "T1055": "TA0004",
    "T1055.001": "TA0004",
    "T1055.002": "TA0004",
    "T1055.003": "TA0004",
    "T1055.004": "TA0004",
    "T1055.009": "TA0004",
    "T1055.011": "TA0004",
    "T1055.012": "TA0004",
    "T1055.015": "TA0004",
    "T1068": "TA0004",
    "T1134": "TA0004",
    "T1134.001": "TA0004",
    "T1134.002": "TA0004",
    "T1134.004": "TA0004",
    "T1134.005": "TA0004",
    "T1562": "TA0005",
    "T1562.001": "TA0005",
    "T1562.002": "TA0005",
    "T1562.003": "TA0005",
    "T1562.004": "TA0005",
    "T1562.006": "TA0005",
    "T1562.008": "TA0005",
    "T1564": "TA0005",
    "T1564.001": "TA0005",
    "T1564.002": "TA0005",
    "T1564.003": "TA0005",
    "T1564.004": "TA0005",
    "T1564.006": "TA0005",
    "T1564.008": "TA0005",
    "T1027": "TA0005",
    "T1027.001": "TA0005",
    "T1027.002": "TA0005",
    "T1027.004": "TA0005",
    "T1027.006": "TA0005",
    "T1027.007": "TA0005",
    "T1027.010": "TA0005",
    "T1027.013": "TA0005",
    "T1027.018": "TA0005",
    "T1070": "TA0005",
    "T1070.001": "TA0005",
    "T1070.003": "TA0005",
    "T1070.004": "TA0005",
    "T1070.005": "TA0005",
    "T1070.006": "TA0005",
    "T1070.008": "TA0005",
    "T1003": "TA0006",
    "T1003.001": "TA0006",
    "T1003.002": "TA0006",
    "T1003.003": "TA0006",
    "T1003.004": "TA0006",
    "T1003.005": "TA0006",
    "T1003.006": "TA0006",
    "T1003.007": "TA0006",
    "T1003.008": "TA0006",
    "T1110": "TA0006",
    "T1110.001": "TA0006",
    "T1110.002": "TA0006",
    "T1110.003": "TA0006",
    "T1110.004": "TA0006",
    "T1552": "TA0006",
    "T1552.001": "TA0006",
    "T1552.002": "TA0006",
    "T1552.003": "TA0006",
    "T1552.004": "TA0006",
    "T1552.005": "TA0006",
    "T1552.006": "TA0006",
    "T1552.007": "TA0006",
    "T1558": "TA0006",
    "T1558.001": "TA0006",
    "T1558.002": "TA0006",
    "T1558.003": "TA0006",
    "T1558.004": "TA0006",
    "T1082": "TA0007",
    "T1083": "TA0007",
    "T1046": "TA0007",
    "T1049": "TA0007",
    "T1010": "TA0007",
    "T1012": "TA0007",
    "T1016": "TA0007",
    "T1016.001": "TA0007",
    "T1016.002": "TA0007",
    "T1018": "TA0007",
    "T1033": "TA0007",
    "T1057": "TA0007",
    "T1069": "TA0007",
    "T1069.001": "TA0007",
    "T1069.002": "TA0007",
    "T1135": "TA0007",
    "T1201": "TA0007",
    "T1217": "TA0007",
    "T1482": "TA0007",
    "T1518": "TA0007",
    "T1518.001": "TA0007",
    "T1526": "TA0007",
    "T1613": "TA0007",
    "T1614": "TA0007",
    "T1614.001": "TA0007",
    "T1615": "TA0007",
    "T1622": "TA0007",
    "T1021": "TA0008",
    "T1021.001": "TA0008",
    "T1021.002": "TA0008",
    "T1021.003": "TA0008",
    "T1021.004": "TA0008",
    "T1021.005": "TA0008",
    "T1021.006": "TA0008",
    "T1550": "TA0008",
    "T1550.002": "TA0008",
    "T1550.003": "TA0008",
    "T1569": "TA0008",
    "T1569.001": "TA0008",
    "T1569.002": "TA0008",
    "T1569.003": "TA0008",
    "T1570": "TA0008",
    "T1572": "TA0008",
    "T1573": "TA0008",
    "T1005": "TA0009",
    "T1006": "TA0009",
    "T1039": "TA0009",
    "T1025": "TA0009",
    "T1113": "TA0009",
    "T1114": "TA0009",
    "T1114.001": "TA0009",
    "T1114.002": "TA0009",
    "T1114.003": "TA0009",
    "T1119": "TA0009",
    "T1123": "TA0009",
    "T1125": "TA0009",
    "T1213": "TA0009",
    "T1213.001": "TA0009",
    "T1213.002": "TA0009",
    "T1213.003": "TA0009",
    "T1041": "TA0010",
    "T1048": "TA0010",
    "T1567": "TA0010",
    "T1567.001": "TA0010",
    "T1567.002": "TA0010",
    "T1567.003": "TA0010",
    "T1568": "TA0010",
    "T1568.002": "TA0010",
    "T1020": "TA0010",
    "T1071": "TA0011",
    "T1071.001": "TA0011",
    "T1071.004": "TA0011",
    "T1090": "TA0011",
    "T1090.001": "TA0011",
    "T1090.003": "TA0011",
    "T1095": "TA0011",
    "T1104": "TA0011",
    "T1105": "TA0011",
    "T1132": "TA0011",
    "T1132.001": "TA0011",
    "T1571": "TA0011",
    "T1572": "TA0011",
    "T1573": "TA0011",
    "T1573.001": "TA0011",
    "T1573.002": "TA0011",
    "T1486": "TA0012",
    "T1489": "TA0012",
    "T1490": "TA0012",
    "T1491": "TA0012",
    "T1491.001": "TA0012",
    "T1496": "TA0012",
    "T1496.001": "TA0012",
    "T1498": "TA0012",
    "T1499": "TA0012",
    "AML.T0035": "TA0040",
    "AML.T0037": "TA0040",
}

# Fallback ranges for technique numbers not in the explicit map.
# (low, high, tactic_id)
_FALLBACK_RANGES: list[tuple[int, int, str]] = [
    (1001, 1002, "TA0009"),
    (1003, 1003, "TA0006"),
    (1005, 1006, "TA0009"),
    (1007, 1007, "TA0007"),
    (1010, 1012, "TA0007"),
    (1014, 1014, "TA0005"),
    (1015, 1016, "TA0007"),
    (1018, 1018, "TA0007"),
    (1020, 1020, "TA0010"),
    (1021, 1021, "TA0008"),
    (1025, 1025, "TA0009"),
    (1027, 1027, "TA0005"),
    (1029, 1030, "TA0010"),
    (1033, 1033, "TA0007"),
    (1035, 1037, "TA0002"),
    (1039, 1039, "TA0009"),
    (1040, 1041, "TA0010"),
    (1044, 1044, "TA0002"),
    (1046, 1049, "TA0007"),
    (1053, 1053, "TA0002"),
    (1055, 1055, "TA0004"),
    (1056, 1056, "TA0006"),
    (1057, 1057, "TA0007"),
    (1059, 1059, "TA0002"),
    (1063, 1063, "TA0007"),
    (1064, 1064, "TA0002"),
    (1068, 1068, "TA0004"),
    (1069, 1069, "TA0007"),
    (1070, 1070, "TA0005"),
    (1071, 1071, "TA0011"),
    (1072, 1072, "TA0008"),
    (1074, 1074, "TA0005"),
    (1077, 1078, "TA0001"),
    (1081, 1081, "TA0006"),
    (1082, 1083, "TA0007"),
    (1085, 1086, "TA0002"),
    (1087, 1087, "TA0006"),
    (1088, 1089, "TA0005"),
    (1090, 1090, "TA0011"),
    (1091, 1091, "TA0001"),
    (1093, 1098, "TA0003"),
    (1103, 1103, "TA0003"),
    (1104, 1105, "TA0011"),
    (1107, 1107, "TA0005"),
    (1110, 1110, "TA0006"),
    (1112, 1112, "TA0005"),
    (1113, 1113, "TA0009"),
    (1115, 1115, "TA0009"),
    (1116, 1116, "TA0005"),
    (1117, 1118, "TA0002"),
    (1119, 1119, "TA0009"),
    (1120, 1120, "TA0010"),
    (1121, 1121, "TA0002"),
    (1122, 1122, "TA0003"),
    (1123, 1123, "TA0009"),
    (1124, 1124, "TA0007"),
    (1125, 1125, "TA0009"),
    (1126, 1127, "TA0005"),
    (1129, 1129, "TA0002"),
    (1132, 1132, "TA0011"),
    (1133, 1133, "TA0001"),
    (1134, 1134, "TA0004"),
    (1135, 1135, "TA0007"),
    (1136, 1136, "TA0003"),
    (1137, 1137, "TA0003"),
    (1140, 1140, "TA0005"),
    (1158, 1158, "TA0003"),
    (1170, 1170, "TA0002"),
    (1176, 1176, "TA0003"),
    (1187, 1189, "TA0001"),
    (1190, 1190, "TA0001"),
    (1195, 1197, "TA0005"),
    (1199, 1200, "TA0001"),
    (1201, 1202, "TA0007"),
    (1204, 1204, "TA0002"),
    (1207, 1207, "TA0005"),
    (1210, 1210, "TA0008"),
    (1212, 1215, "TA0006"),
    (1216, 1218, "TA0005"),
    (1219, 1219, "TA0009"),
    (1220, 1222, "TA0005"),
    (1482, 1499, "TA0012"),
    (1505, 1505, "TA0003"),
    (1518, 1518, "TA0007"),
    (1526, 1531, "TA0007"),
    (1537, 1537, "TA0009"),
    (1539, 1539, "TA0006"),
    (1542, 1543, "TA0003"),
    (1546, 1547, "TA0003"),
    (1548, 1548, "TA0004"),
    (1550, 1550, "TA0008"),
    (1552, 1552, "TA0006"),
    (1553, 1553, "TA0005"),
    (1555, 1555, "TA0003"),
    (1556, 1556, "TA0006"),
    (1557, 1557, "TA0006"),
    (1558, 1558, "TA0006"),
    (1559, 1559, "TA0002"),
    (1560, 1560, "TA0009"),
    (1562, 1565, "TA0005"),
    (1566, 1566, "TA0001"),
    (1567, 1568, "TA0010"),
    (1569, 1569, "TA0008"),
    (1570, 1570, "TA0008"),
    (1571, 1571, "TA0011"),
    (1572, 1573, "TA0011"),
    (1574, 1574, "TA0003"),
    (1578, 1578, "TA0005"),
    (1580, 1580, "TA0011"),
    (1583, 1585, "TA0042"),
    (1587, 1588, "TA0042"),
    (1590, 1595, "TA0043"),
    (1596, 1602, "TA0043"),
    (1608, 1609, "TA0042"),
    (1611, 1612, "TA0005"),
    (1613, 1615, "TA0007"),
    (1619, 1619, "TA0009"),
    (1620, 1620, "TA0005"),
    (1622, 1622, "TA0007"),
    (1647, 1654, "TA0007"),
    (1656, 1672, "TA0042"),
    (1685, 1689, "TA0043"),
]

# Reference list of well-known MITRE ATT&CK Enterprise techniques (sub-techniques).
# This is a representative set used for coverage-gap detection.
# In production, load from the official STIX data.
REFERENCE_TECHNIQUES: list[str] = [
    # Initial Access
    "T1078",
    "T1078.001",
    "T1078.002",
    "T1078.003",
    "T1078.004",
    "T1566",
    "T1566.001",
    "T1566.002",
    "T1566.003",
    "T1190",
    "T1133",
    "T1200",
    "T1091",
    "T1189",
    "T1195",
    "T1195.002",
    "T1199",
    # Execution
    "T1059",
    "T1059.001",
    "T1059.002",
    "T1059.003",
    "T1059.004",
    "T1059.005",
    "T1059.006",
    "T1059.007",
    "T1053",
    "T1053.002",
    "T1053.003",
    "T1053.005",
    "T1053.006",
    "T1204",
    "T1204.002",
    "T1204.003",
    "T1035",
    "T1121",
    "T1559",
    "T1559.002",
    "T1609",
    "T1610",
    # Persistence
    "T1543",
    "T1543.001",
    "T1543.002",
    "T1543.003",
    "T1547",
    "T1547.001",
    "T1547.002",
    "T1136",
    "T1136.001",
    "T1136.002",
    "T1137",
    "T1137.001",
    "T1137.002",
    "T1546",
    "T1546.001",
    "T1546.003",
    "T1574",
    "T1574.001",
    "T1574.006",
    # Privilege Escalation
    "T1548",
    "T1548.001",
    "T1548.002",
    "T1548.003",
    "T1055",
    "T1055.001",
    "T1055.002",
    "T1055.003",
    "T1068",
    "T1134",
    "T1134.001",
    "T1134.002",
    # Defense Evasion
    "T1562",
    "T1562.001",
    "T1562.002",
    "T1564",
    "T1564.001",
    "T1564.002",
    "T1027",
    "T1027.001",
    "T1027.002",
    "T1070",
    "T1070.001",
    "T1070.004",
    "T1036",
    "T1036.002",
    "T1036.005",
    "T1218",
    "T1218.001",
    "T1218.011",
    # Credential Access
    "T1003",
    "T1003.001",
    "T1003.002",
    "T1110",
    "T1110.001",
    "T1110.003",
    "T1552",
    "T1552.001",
    "T1552.004",
    "T1558",
    "T1558.001",
    "T1558.002",
    "T1056",
    "T1056.001",
    # Discovery
    "T1082",
    "T1083",
    "T1046",
    "T1049",
    "T1018",
    "T1033",
    "T1057",
    "T1069",
    "T1069.001",
    "T1069.002",
    "T1135",
    "T1201",
    "T1482",
    # Lateral Movement
    "T1021",
    "T1021.001",
    "T1021.002",
    "T1021.004",
    "T1550",
    "T1550.002",
    "T1550.003",
    "T1569",
    "T1570",
    "T1572",
    # Collection
    "T1005",
    "T1039",
    "T1113",
    "T1114",
    "T1119",
    "T1123",
    "T1125",
    "T1213",
    "T1213.001",
    # Exfiltration
    "T1041",
    "T1048",
    "T1567",
    "T1568",
    "T1020",
    # Command and Control
    "T1071",
    "T1071.001",
    "T1071.004",
    "T1090",
    "T1095",
    "T1105",
    "T1571",
    "T1573",
    # Impact
    "T1486",
    "T1489",
    "T1490",
    "T1491",
    "T1496",
    "T1498",
    "T1499",
    # Resource Development
    "T1583",
    "T1587",
    "T1588",
    "T1608",
    # Reconnaissance
    "T1592",
    "T1593",
    "T1595",
    "T1596",
]

# Tactics considered "network exploitation / post-exploitation" for bias scoring
_NETWORK_POST_EXPLOITATION_TACTICS: set[str] = {
    "TA0005",  # Defense Evasion
    "TA0006",  # Credential Access
    "TA0007",  # Discovery
    "TA0008",  # Lateral Movement
    "TA0009",  # Collection
    "TA0010",  # Exfiltration
    "TA0011",  # Command and Control
}

# Sources considered metasploit-heavy for bias scoring
_METASPLOIT_SOURCES: set[str] = {
    "metasploit",
    "metasploit-framework",
    "rapid7/metasploit-framework",
    "metasploit_framework",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_tactic_for_technique(tech_id: str) -> str | None:
    """Map a MITRE technique ID to its primary tactic ID."""
    normalized = tech_id.upper()
    if normalized.startswith("AML."):
        if normalized in TECHNIQUE_TO_TACTIC:
            return TECHNIQUE_TO_TACTIC[normalized]
        return "TA0040"
    if normalized in TECHNIQUE_TO_TACTIC:
        return TECHNIQUE_TO_TACTIC[normalized]
    # Try parent technique for sub-techniques
    if "." in normalized:
        parent = normalized.split(".")[0]
        if parent in TECHNIQUE_TO_TACTIC:
            return TECHNIQUE_TO_TACTIC[parent]
    # Fallback range lookup
    match = re.match(r"T(\d{4})", normalized)
    if match:
        num = int(match.group(1))
        for low, high, tactic in _FALLBACK_RANGES:
            if low <= num <= high:
                return tactic
    return None


def extract_techniques_from_record(record: dict) -> set[str]:
    """Extract all MITRE technique IDs from a JSONL record.

    Looks in:
    1. The `mitre_ids` structured field
    2. Pattern matching in `messages[*].content` (assistant and user roles)
    """
    found: set[str] = set()

    # 1. Structured field
    for tid in record.get("mitre_ids") or []:
        found.add(tid.upper())

    # 2. Content pattern matching
    for msg in record.get("messages", []):
        content = msg.get("content", "")
        if not content:
            continue
        for m in _TECHNIQUE_RE.finditer(content):
            found.add(m.group(1).upper())
        for m in _ATLAS_RE.finditer(content):
            found.add(m.group(1).upper())
        for m in _TECHNIQUE_MARKDOWN_RE.finditer(content):
            found.add(m.group(1).upper())

    return found


def classify_bucket(bucket_path: Path, root: Path) -> str:
    """Classify a bucket into a category: tactic, tools, ai_redteam, or meta."""
    rel = bucket_path.relative_to(root / "data" / "datasets" / "buckets")
    parts = rel.parts
    if parts[0] == "base":
        return "tactic"
    elif parts[0] == "tools":
        return "tools"
    elif parts[0] == "ai":
        return "ai_redteam"
    elif parts[0] == "orchestrator":
        return "meta"
    return "unknown"


def bucket_display_name(bucket_path: Path, root: Path) -> str:
    """Get a human-readable display name for a bucket path."""
    rel = bucket_path.relative_to(root / "data" / "datasets" / "buckets")
    parts = rel.parts
    if parts[0] == "base" and len(parts) >= 2:
        return parts[1]
    elif len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return parts[0]


def avg_msg_length(record: dict) -> float:
    """Compute average character length of messages in a record."""
    msgs = record.get("messages", [])
    if not msgs:
        return 0.0
    total = sum(len(m.get("content", "")) for m in msgs)
    return total / len(msgs)


def format_histogram(counter: Counter, top_n: int = 20) -> list[dict]:
    """Format a Counter as a sorted list of {item, count} dicts."""
    return [
        {"technique": tid, "count": count} for tid, count in counter.most_common(top_n)
    ]


def format_least_common(counter: Counter, bottom_n: int = 20) -> list[dict]:
    """Format least common items from a Counter."""
    items = counter.most_common()
    items.reverse()
    return [{"technique": tid, "count": count} for tid, count in items[:bottom_n]]


# ---------------------------------------------------------------------------
# Main audit logic
# ---------------------------------------------------------------------------


def audit_dataset(root: Path) -> dict[str, Any]:
    """Run the full dataset audit and return the report dict."""
    buckets_dir = root / "data" / "datasets" / "buckets"

    # Discover all data.jsonl files
    jsonl_files = sorted(buckets_dir.rglob("data.jsonl"))
    if not jsonl_files:
        print(f"ERROR: No data.jsonl files found under {buckets_dir}", file=sys.stderr)
        sys.exit(1)

    # --- Per-bucket stats ---
    bucket_reports: list[dict] = []
    all_techniques: Counter = Counter()
    all_sources: Counter = Counter()
    total_records = 0
    total_msg_length = 0.0

    # Category-level aggregations
    category_techniques: dict[str, Counter] = {}
    category_counts: dict[str, int] = {}
    category_sources: dict[str, Counter] = {}

    # Tactic-level aggregation (for heatmap)
    tactic_counter: Counter = Counter()

    # Metasploit-heavy tracking
    metasploit_record_count = 0

    for jsonl_path in jsonl_files:
        bucket_name = bucket_display_name(jsonl_path.parent, root)
        category = classify_bucket(jsonl_path.parent, root)

        bucket_techniques: Counter = Counter()
        bucket_sources: Counter = Counter()
        bucket_count = 0
        bucket_msg_length = 0.0

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                bucket_count += 1
                total_records += 1

                # Extract techniques
                techs = extract_techniques_from_record(record)
                for t in techs:
                    bucket_techniques[t] += 1
                    all_techniques[t] += 1
                    tactic = get_tactic_for_technique(t)
                    if tactic:
                        tactic_counter[tactic] += 1

                # Source tracking
                source = record.get("source") or "unknown"
                bucket_sources[source] += 1
                all_sources[source] += 1

                # Message length
                avg_len = avg_msg_length(record)
                bucket_msg_length += avg_len
                total_msg_length += avg_len

                # Metasploit bias tracking
                source_lower = source.lower()
                if source_lower in _METASPLOIT_SOURCES:
                    metasploit_record_count += 1

                # Also check system prompt for metasploit indicators
                for msg in record.get("messages", []):
                    if (
                        msg.get("role") == "system"
                        and "metasploit" in msg.get("content", "").lower()
                    ):
                        metasploit_record_count += 1
                        break

                # Techniques from network/post-exploitation tactics
                for t in techs:
                    tactic = get_tactic_for_technique(t)
                    if tactic and tactic in _NETWORK_POST_EXPLOITATION_TACTICS:
                        # Already counted above in tactic_counter
                        pass

        avg_len = bucket_msg_length / bucket_count if bucket_count else 0.0

        bucket_reports.append(
            {
                "bucket": bucket_name,
                "category": category,
                "path": str(jsonl_path.relative_to(root)),
                "total_count": bucket_count,
                "unique_techniques": len(bucket_techniques),
                "technique_distribution": format_histogram(bucket_techniques),
                "source_distribution": {s: c for s, c in bucket_sources.most_common()},
                "average_message_length": round(avg_len, 1),
            }
        )

        # Category aggregation
        if category not in category_techniques:
            category_techniques[category] = Counter()
            category_counts[category] = 0
            category_sources[category] = Counter()
        category_techniques[category] += bucket_techniques
        category_counts[category] = bucket_count + category_counts.get(category, 0)
        category_sources[category] += bucket_sources

    # --- Overall report assembly ---
    overall_avg_len = total_msg_length / total_records if total_records else 0.0

    # Category breakdown
    category_breakdown = {}
    for cat in sorted(category_counts):
        category_breakdown[cat] = {
            "total_records": category_counts[cat],
            "unique_techniques": len(category_techniques[cat]),
            "top_sources": dict(category_sources[cat].most_common(10)),
        }

    # Top 20 / Bottom 20 techniques
    top_20 = format_histogram(all_techniques, 20)
    bottom_20 = format_least_common(all_techniques, 20)

    # Zero-coverage techniques
    found_set = set(all_techniques.keys())
    zero_coverage = sorted(t for t in REFERENCE_TECHNIQUES if t not in found_set)

    # Bias score: % of data that is metasploit-heavy (network/post-exploitation)
    metasploit_bias_pct = round(
        (metasploit_record_count / total_records * 100) if total_records else 0.0, 2
    )

    # Per-tactic heatmap
    tactic_heatmap = {}
    for tactic_id, tactic_name in sorted(MITRE_TACTICS.items()):
        count = tactic_counter.get(tactic_id, 0)
        pct = round(count / total_records * 100, 2) if total_records else 0.0
        tactic_heatmap[tactic_id] = {
            "name": tactic_name,
            "technique_hits": count,
            "percentage_of_total": pct,
            "representation": (
                "over-represented"
                if pct > 15
                else "under-represented"
                if pct < 3
                else "balanced"
            ),
        }

    report: dict[str, Any] = {
        "audit_metadata": {
            "total_records": total_records,
            "total_buckets": len(jsonl_files),
            "unique_techniques_found": len(all_techniques),
            "unique_sources": len(all_sources),
            "average_message_length": round(overall_avg_len, 1),
        },
        "category_breakdown": category_breakdown,
        "bucket_reports": bucket_reports,
        "top_20_techniques": top_20,
        "bottom_20_techniques": bottom_20,
        "zero_coverage_techniques": zero_coverage,
        "bias_score": {
            "metasploit_heavy_pct": metasploit_bias_pct,
            "metasploit_records": metasploit_record_count,
            "total_records": total_records,
            "interpretation": (
                f"{metasploit_bias_pct}% of all records come from metasploit-heavy sources. "
                "This may indicate bias toward network exploitation and post-exploitation scenarios."
                if metasploit_bias_pct > 50
                else f"{metasploit_bias_pct}% of records are metasploit-heavy. "
                "Moderate representation from network exploitation sources."
                if metasploit_bias_pct > 20
                else f"Only {metasploit_bias_pct}% of records are metasploit-heavy. "
                "Good diversity across attack domains."
            ),
        },
        "tactic_heatmap": tactic_heatmap,
        "source_distribution": dict(all_sources.most_common()),
    }

    return report


def print_summary(report: dict) -> None:
    """Print a human-readable summary to stdout."""
    meta = report["audit_metadata"]
    print("=" * 72)
    print("  AttackLM Dataset Bias Audit Report")
    print("=" * 72)
    print()
    print(f"  Total Records:          {meta['total_records']:,}")
    print(f"  Total Buckets:           {meta['total_buckets']}")
    print(f"  Unique Techniques:      {meta['unique_techniques_found']}")
    print(f"  Unique Sources:          {meta['unique_sources']}")
    print(f"  Avg Message Length:       {meta['average_message_length']:.1f} chars")
    print()

    # Category breakdown
    print("-" * 72)
    print("  CATEGORY BREAKDOWN")
    print("-" * 72)
    for cat, info in report["category_breakdown"].items():
        print(
            f"  {cat:15s}  {info['total_records']:>6,} records  "
            f"{info['unique_techniques']:>4} unique techniques  "
            f"sources: {', '.join(f'{s}({c})' for s, c in list(info['top_sources'].items())[:5])}"
        )
    print()

    # Per-bucket summary
    print("-" * 72)
    print("  BUCKET SUMMARY")
    print("-" * 72)
    print(f"  {'Bucket':<35s} {'Count':>6s} {'Techs':>6s} {'AvgLen':>8s}")
    print(f"  {'─' * 35} {'─' * 6} {'─' * 6} {'─' * 8}")
    for br in report["bucket_reports"]:
        print(
            f"  {br['bucket']:<35s} {br['total_count']:>6,} "
            f"{br['unique_techniques']:>6} {br['average_message_length']:>8.1f}"
        )
    print()

    # Top 20 techniques
    print("-" * 72)
    print("  TOP 20 TECHNIQUES (most common)")
    print("-" * 72)
    for i, entry in enumerate(report["top_20_techniques"], 1):
        tactic = get_tactic_for_technique(entry["technique"])
        tactic_name = MITRE_TACTICS.get(tactic, "Unknown") if tactic else "Unknown"
        print(
            f"  {i:2d}. {entry['technique']:<15s} {entry['count']:>6,} hits  "
            f"({tactic_name})"
        )
    print()

    # Bottom 20 techniques (rare coverage)
    print("-" * 72)
    print("  BOTTOM 20 TECHNIQUES (rare coverage)")
    print("-" * 72)
    for i, entry in enumerate(report["bottom_20_techniques"], 1):
        tactic = get_tactic_for_technique(entry["technique"])
        tactic_name = MITRE_TACTICS.get(tactic, "Unknown") if tactic else "Unknown"
        print(
            f"  {i:2d}. {entry['technique']:<15s} {entry['count']:>6,} hits  "
            f"({tactic_name})"
        )
    print()

    # Zero coverage
    zero = report["zero_coverage_techniques"]
    print("-" * 72)
    print(f"  ZERO COVERAGE ({len(zero)} reference techniques missing)")
    print("-" * 72)
    if zero:
        # Group by tactic
        by_tactic: dict[str, list[str]] = {}
        for t in zero:
            tactic = get_tactic_for_technique(t) or "Unknown"
            by_tactic.setdefault(tactic, []).append(t)
        for tactic_id in sorted(by_tactic):
            name = MITRE_TACTICS.get(tactic_id, tactic_id)
            techs = ", ".join(sorted(by_tactic[tactic_id]))
            print(f"  {name}:")
            print(f"    {techs}")
    else:
        print("  All reference techniques are covered!")
    print()

    # Bias score
    print("-" * 72)
    print("  BIAS SCORE")
    print("-" * 72)
    bias = report["bias_score"]
    print(
        f"  Metasploit-heavy records:  {bias['metasploit_records']:,} / "
        f"{bias['total_records']:,} ({bias['metasploit_heavy_pct']}%)"
    )
    print(f"  {bias['interpretation']}")
    print()

    # Tactic heatmap
    print("-" * 72)
    print("  TACTIC HEATMAP (technique hit distribution)")
    print("-" * 72)
    print(f"  {'Tactic':<30s} {'Hits':>6s} {'%':>7s} {'Status':>20s}")
    print(f"  {'─' * 30} {'─' * 6} {'─' * 7} {'─' * 20}")
    for tactic_id, info in report["tactic_heatmap"].items():
        label = f"{tactic_id} {info['name']}"
        print(
            f"  {label:<30s} {info['technique_hits']:>6,} "
            f"{info['percentage_of_total']:>6.1f}% "
            f"{info['representation']:>20s}"
        )
    print()

    # Source distribution
    print("-" * 72)
    print("  SOURCE DISTRIBUTION")
    print("-" * 72)
    for source, count in report["source_distribution"].items():
        pct = count / meta["total_records"] * 100
        bar = "█" * int(pct / 2)
        print(f"  {source:<40s} {count:>6,} ({pct:>5.1f}%) {bar}")
    print()

    print("=" * 72)
    print("  End of Report")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit AttackLM dataset for MITRE technique coverage and bias."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Root directory of the AttackLM project (default: parent of scripts/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for JSON report (default: data/audit_report.json)",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_path = args.output or root / "data" / "audit_report.json"

    report = audit_dataset(root)

    # Write JSON report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"JSON report written to {output_path}")

    # Print human-readable summary
    print_summary(report)


if __name__ == "__main__":
    main()
