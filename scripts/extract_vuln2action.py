#!/usr/bin/env python3
# CREDITS — DATA SOURCE ATTRIBUTION
# ----------------------------------
# This script transforms data from: Vuln2Action
# Paper: JISA 2026 — "Vuln2Action: CVE+CWE+CPE+CAPEC+Exploit-DB fused
#         dataset with reproduction steps"
# Dataset: https://github.com/Vuln2Action (or HuggingFace)
# License: Research paper — check dataset repository for license terms
# ----------------------------------
"""Deterministic extraction of Vuln2Action vulnerability data into AttackLM JSONL training pairs.

Processes the Vuln2Action dataset linking CVEs to CWEs, CPEs, CAPECs, and
Exploit-DB entries. For each record, generates instruction-response pairs
covering vulnerability exploitation, impact analysis, and remediation.

Output: ``data/datasets/buckets/sources/vuln2action/vulnerability_analysis/data.jsonl``

Usage:
    python scripts/extract_vuln2action.py --input-dir /path/to/vuln2action
    python scripts/extract_vuln2action.py --dry-run --limit 10
"""

from __future__ import annotations

import argparse
import csv
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
DEFAULT_INPUT_DIR = BASE_DIR / "data" / "vuln2action"
OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "datasets"
    / "buckets"
    / "sources"
    / "vuln2action"
    / "vulnerability_analysis"
)
OUTPUT_PATH = OUTPUT_DIR / "data.jsonl"

# ---------------------------------------------------------------------------
# System message
# ---------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are a Vulnerability Analysis specialist. Provide detailed analysis "
    "of CVE vulnerabilities including exploitation methods, attack patterns "
    "(CAPEC), affected products (CPE), impact assessment, and remediation "
    "guidance mapped to MITRE ATT&CK techniques."
)

# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------
ATTRIBUTION = {
    "source": "vuln2action",
    "source_uri": "https://github.com/Vuln2Action",
    "license": "Research",
    "license_uri": "https://github.com/Vuln2Action",
    "rights_contact": "Vuln2Action authors",
    "attribution_text": (
        "Vuln2Action — JISA 2026 paper dataset. "
        "CVE+CWE+CPE+CAPEC+Exploit-DB fused dataset with reproduction steps. "
        "See the Vuln2Action repository for license terms."
    ),
}

# ---------------------------------------------------------------------------
# CAPEC → MITRE ATT&CK mapping
# ---------------------------------------------------------------------------
# Maps CAPEC attack pattern IDs to relevant MITRE ATT&CK technique IDs.
# Source: MITRE CAPEC-ATT&CK cross-reference and NIST CAPEC catalog.
# This is a representative subset covering the most common attack patterns.
CAPEC_TO_ATTACK: dict[str, list[str]] = {
    # Initial Access
    "CAPEC-1": ["T1190"],  # Accessing Functionality Not Properly Constrained
    "CAPEC-2": ["T1190"],  # Accessing Functionality Not Properly Constrained (alt)
    "CAPEC-6": ["T1190"],  # Argument Injection
    "CAPEC-10": ["T1190"],  # Buffer Overflow via Environment Variables
    "CAPEC-15": ["T1190"],  # Command Delimiters
    "CAPEC-24": ["T1078"],  # Abuse of Functionality
    "CAPEC-34": ["T1190"],  # HTTP Request Splitting
    "CAPEC-49": ["T1190"],  # Password Recovery Exploitation
    "CAPEC-50": ["T1110"],  # Password Recovery Exploitation (alternate)
    "CAPEC-55": ["T1190"],  # Path Traversal
    "CAPEC-59": ["T1190"],  # Session Credential Falsification through Forging
    "CAPEC-62": ["T1190"],  # Cross Site Request Forgery
    "CAPEC-63": ["T1190"],  # Cross-Site Scripting (XSS)
    "CAPEC-66": ["T1190"],  # SQL Injection
    "CAPEC-67": ["T1190"],  # String Format Overflow
    "CAPEC-69": ["T1190"],  # XSS Through HTTP Query
    "CAPEC-71": ["T1190"],  # Using Leading 'Content-type' to Bypass Validation
    "CAPEC-76": ["T1190"],  # Manipulating File Extensions
    "CAPEC-77": ["T1190"],  # Manipulating User-Controlled Variables
    "CAPEC-88": ["T1190"],  # OS Command Injection
    "CAPEC-90": ["T1190"],  # Parameter Injection
    "CAPEC-91": ["T1190"],  # Forced Browse
    "CAPEC-92": ["T1190"],  # Forced Browse (alt)
    "CAPEC-94": ["T1190"],  # Man-in-the-Middle
    "CAPEC-95": ["T1565"],  # WSDL Scraping
    "CAPEC-97": ["T1190"],  # Cryptanalysis
    "CAPEC-98": ["T1110"],  # Cryptanalysis (alt)
    "CAPEC-99": ["T1110"],  # Brute Force
    "CAPEC-100": ["T1190"],  # Overflow Buffers
    "CAPEC-102": ["T1190"],  # Cross-Site Scripting via Encoded Tags
    "CAPEC-104": ["T1190"],  # Overflow Buffers (alt)
    "CAPEC-108": ["T1190"],  # Command Line Execution through File Manipulation
    "CAPEC-112": ["T1110"],  # Brute Force (alt)
    "CAPEC-116": ["T1190"],  # Excavation
    "CAPEC-117": ["T1190"],  # Interception
    "CAPEC-118": ["T1190"],  # Resource Exhaustion
    "CAPEC-121": ["T1190"],  # Exploitation of Trusted Credentials
    "CAPEC-122": ["T1190"],  # Exploit Trusted Relationships
    "CAPEC-123": ["T1190"],  # XML Injection
    "CAPEC-126": ["T1190"],  # Path Traversal (alt)
    "CAPEC-127": ["T1190"],  # Path Traversal (alt2)
    "CAPEC-130": ["T1190"],  # Great, Open and Free
    "CAPEC-131": ["T1190"],  # Data-Structure Attacks
    "CAPEC-136": ["T1110"],  # Content-Based Password Attack
    "CAPEC-137": ["T1110"],  # Parameter Injection (alt)
    "CAPEC-140": ["T1110"],  # Brute Force (alt2)
    "CAPEC-148": ["T1190"],  # Content Spoofing
    "CAPEC-151": ["T1190"],  # Identity Spoofing
    "CAPEC-153": ["T1190"],  # Input Data Manipulation
    "CAPEC-157": ["T1110"],  # Sniffing Attacks
    "CAPEC-160": ["T1190"],  # Exploitation of Privilege/Trust
    "CAPEC-161": ["T1190"],  # Injection of Trusted Elements
    "CAPEC-165": ["T1190"],  # File Content Injection
    "CAPEC-169": ["T1190"],  # Footprinting
    "CAPEC-170": ["T1592"],  # Web Scraping
    "CAPEC-173": ["T1190"],  # Action Spoofing
    "CAPEC-176": ["T1190"],  # Configuration/Environment Manipulation
    "CAPEC-180": ["T1190"],  # Exploiting Incorrectly Configured SSL/TLS
    "CAPEC-182": ["T1190"],  # Flash Injection
    "CAPEC-185": ["T1190"],  # Double Encoding
    "CAPEC-194": ["T1190"],  # Parameter Injection (alt2)
    "CAPEC-195": ["T1190"],  # Principal Spoofing
    "CAPEC-196": ["T1190"],  # XSS Targeting Non-Script Elements
    "CAPEC-197": ["T1190"],  # XSS Targeting Script Elements
    "CAPEC-204": ["T1190"],  # Lifting Sensitive Data Embedded in Cache
    "CAPEC-210": ["T1190"],  # Abuse of Functionality (alt)
    "CAPEC-216": ["T1190"],  # Communication Channel Manipulation
    "CAPEC-221": ["T1190"],  # XSS via Scripts (alt)
    "CAPEC-224": ["T1190"],  # Fuzzing
    "CAPEC-225": ["T1190"],  # XXE
    "CAPEC-230": ["T1190"],  # Path Traversal (alt3)
    "CAPEC-233": ["T1190"],  # Parameter Injection (alt3)
    "CAPEC-235": ["T1190"],  # Exploiting Trust in Update Mechanisms
    "CAPEC-236": ["T1190"],  # Exploiting Trust in Update Mechanisms (alt)
    "CAPEC-242": ["T1190"],  # XSS via DOM (alt)
    "CAPEC-243": ["T1190"],  # XSS via DOM (alt2)
    "CAPEC-247": ["T1190"],  # XSS via DOM (alt3)
    "CAPEC-248": ["T1190"],  # Command Injection (alt)
    "CAPEC-251": ["T1190"],  # Local Inclusion
    "CAPEC-252": ["T1190"],  # XSS via DOM (alt4)
    "CAPEC-253": ["T1190"],  # Remote Inclusion
    "CAPEC-256": ["T1190"],  # SOAP Injection
    "CAPEC-257": ["T1190"],  # LDAP Injection
    "CAPEC-258": ["T1190"],  # XPATH Injection
    "CAPEC-259": ["T1190"],  # XSS via DOM (alt5)
    "CAPEC-261": ["T1190"],  # XSS via DOM (alt6)
    "CAPEC-269": ["T1190"],  # XSS via DOM (alt7)
    "CAPEC-270": ["T1190"],  # XSS via DOM (alt8)
    "CAPEC-272": ["T1190"],  # Protocol Manipulation
    "CAPEC-273": ["T1190"],  # HTTP Response Splitting
    "CAPEC-274": ["T1190"],  # HTTP Response Splitting (alt)
    "CAPEC-279": ["T1190"],  # LDAP Injection (alt)
    # Execution
    "CAPEC-122": ["T1190", "T1200"],  # Exploit Trusted Relationships
    "CAPEC-124": ["T1203"],  # Shared Resource Manipulation
    "CAPEC-125": ["T1203"],  # Path Traversal (Execution variant)
    "CAPEC-150": ["T1059"],  # Command-Line Execution
    "CAPEC-156": ["T1059"],  # Reverse Engineering
    "CAPEC-158": ["T1059"],  # Command-Line Execution (alt)
    "CAPEC-162": ["T1059"],  # Manipulating User-Controlled Variables (Execution)
    "CAPEC-163": ["T1059"],  # Cross-Site Scripting (Execution)
    "CAPEC-164": ["T1059"],  # XSS Through HTTP Query (Execution)
    "CAPEC-167": ["T1059"],  # File System Injection
    "CAPEC-168": ["T1059"],  # Injection of Trusted Elements (Execution)
    # Persistence
    "CAPEC-17": ["T1543"],  # Accessing, Modifying or Executing Executable Files
    "CAPEC-31": ["T1547"],  # Accessing/Modifying System-Level Service
    "CAPEC-35": ["T1543"],  # Leverage Executable Code in Non-Executable Files
    "CAPEC-38": ["T1543"],  # Leverage Executable Code in Non-Executable Files (alt)
    "CAPEC-42": ["T1543"],  # MIME Conversion (alt)
    "CAPEC-43": ["T1543"],  # Exploiting Mixed Content
    "CAPEC-44": ["T1543"],  # Hardware Addition
    # Privilege Escalation
    "CAPEC-1": ["T1068"],  # Accessing Functionality Not Properly Constrained
    "CAPEC-122": ["T1068"],  # Exploit Trusted Relationships (PrivEsc)
    "CAPEC-233": ["T1068"],  # Parameter Injection (PrivEsc)
    "CAPEC-236": ["T1068"],  # Exploiting Trust (PrivEsc)
    # Defense Evasion
    "CAPEC-10": ["T1562"],  # Buffer Overflow via Environment Variables (DefEvas)
    "CAPEC-15": ["T1562"],  # Command Delimiters (DefEvas)
    "CAPEC-93": ["T1070"],  # Log Deletion
    "CAPEC-94": ["T1562"],  # Man-in-the-Middle (DefEvas)
    # Credential Access
    "CAPEC-49": ["T1110"],  # Password Recovery Exploitation
    "CAPEC-50": ["T1110"],  # Password Recovery Exploitation (alt)
    "CAPEC-55": ["T1110.001"],  # Forged Credentials
    "CAPEC-112": ["T1110"],  # Brute Force
    "CAPEC-136": ["T1110"],  # Content-Based Password Attack
    "CAPEC-140": ["T1110"],  # Brute Force (alt)
    "CAPEC-151": ["T1110"],  # Identity Spoofing (CredAccess)
    "CAPEC-157": ["T1110"],  # Sniffing Attacks (CredAccess)
    "CAPEC-160": ["T1110"],  # Exploitation of Privilege/Trust (CredAccess)
    "CAPEC-204": ["T1552"],  # Lifting Sensitive Data Embedded in Cache
    # Discovery
    "CAPEC-169": ["T1082"],  # Footprinting (Discovery)
    "CAPEC-170": ["T1592"],  # Web Scraping (Discovery)
    "CAPEC-224": ["T1082"],  # Fuzzing (Discovery)
    # Lateral Movement
    "CAPEC-94": ["T1021"],  # Man-in-the-Middle (LatMov)
    "CAPEC-121": ["T1021"],  # Exploitation of Trusted Credentials (LatMov)
    "CAPEC-122": ["T1021"],  # Exploit Trusted Relationships (LatMov)
    # Collection
    "CAPEC-117": ["T1560"],  # Interception (Collection)
    "CAPEC-118": ["T1560"],  # Resource Exhaustion (Collection)
    # Impact
    "CAPEC-100": ["T1499"],  # Overflow Buffers (Impact)
    "CAPEC-118": ["T1499"],  # Resource Exhaustion (Impact)
    "CAPEC-130": ["T1499"],  # Great, Open and Free (Impact)
    "CAPEC-210": ["T1499"],  # Abuse of Functionality (Impact)
}

# CWE → CAPEC mapping (most common ones)
# Maps CWE weakness IDs to related CAPEC attack patterns.
CWE_TO_CAPEC: dict[str, list[str]] = {
    "CWE-22": ["CAPEC-126", "CAPEC-127", "CAPEC-230"],  # Path Traversal
    "CWE-78": ["CAPEC-88", "CAPEC-248"],  # OS Command Injection
    "CWE-79": ["CAPEC-63", "CAPEC-196", "CAPEC-197"],  # XSS
    "CWE-89": ["CAPEC-66"],  # SQL Injection
    "CWE-94": ["CAPEC-88", "CAPEC-248"],  # Code Injection
    "CWE-119": ["CAPEC-100", "CAPEC-104"],  # Buffer Overflow
    "CWE-125": ["CAPEC-100"],  # Out-of-bounds Read
    "CWE-190": ["CAPEC-100"],  # Integer Overflow
    "CWE-200": ["CAPEC-116", "CAPEC-169"],  # Info Exposure
    "CWE-250": ["CAPEC-1", "CAPEC-122"],  # Execution with Unnecessary Privileges
    "CWE-269": ["CAPEC-122"],  # Improper Privilege Management
    "CWE-287": ["CAPEC-99", "CAPEC-112"],  # Improper Authentication
    "CWE-306": ["CAPEC-91"],  # Missing Authentication
    "CWE-311": ["CAPEC-180"],  # Missing Encryption
    "CWE-312": ["CAPEC-180"],  # Cleartext Storage
    "CWE-327": ["CAPEC-97"],  # Broken Crypto
    "CWE-352": ["CAPEC-62"],  # CSRF
    "CWE-400": ["CAPEC-118"],  # Uncontrolled Resource Consumption
    "CWE-434": ["CAPEC-88", "CAPEC-248"],  # Unrestricted File Upload
    "CWE-502": ["CAPEC-131"],  # Deserialization
    "CWE-611": ["CAPEC-225"],  # XXE
    "CWE-613": ["CAPEC-91"],  # Insufficient Session Expiration
    "CWE-732": ["CAPEC-1"],  # Incorrect Permission Assignment
    "CWE-770": ["CAPEC-118"],  # Allocation of Resources Without Limits
    "CWE-776": ["CAPEC-118"],  # XML Entity Expansion
    "CWE-862": ["CAPEC-91"],  # Missing Authorization
    "CWE-863": ["CAPEC-91"],  # Incorrect Authorization
    "CWE-918": ["CAPEC-272", "CAPEC-273"],  # SSRF
    "CWE-922": ["CAPEC-180"],  # Insecure Storage
    "CWE-1021": ["CAPEC-63"],  # Improper Render
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _resolve_attack_ids(capec_ids: list[str], cwe_ids: list[str]) -> list[str]:
    """Resolve CAPEC and CWE IDs to MITRE ATT&CK technique IDs.

    First checks direct CAPEC→ATT&CK mapping, then resolves CWE→CAPEC→ATT&CK.
    Returns deduplicated sorted list of ATT&CK technique IDs.
    """
    attack_ids: set[str] = set()

    # Direct CAPEC → ATT&CK mapping
    for capec in capec_ids:
        capec_upper = capec.upper().replace("CAPEC-", "CAPEC-")
        if capec_upper in CAPEC_TO_ATTACK:
            attack_ids.update(CAPEC_TO_ATTACK[capec_upper])

    # CWE → CAPEC → ATT&CK mapping
    for cwe in cwe_ids:
        cwe_upper = cwe.upper().replace("CWE-", "CWE-")
        if cwe_upper in CWE_TO_CAPEC:
            for capec in CWE_TO_CAPEC[cwe_upper]:
                if capec in CAPEC_TO_ATTACK:
                    attack_ids.update(CAPEC_TO_ATTACK[capec])

    return sorted(attack_ids)


def _format_cve_description(
    cve_id: str,
    description: str,
    cwe_ids: list[str],
    cpe_ids: list[str],
    capec_ids: list[str],
    exploitdb_ids: list[str],
    cvss_score: float | None,
    cvss_vector: str | None,
    reproduction_steps: str | None,
) -> str:
    """Format a complete vulnerability analysis response."""
    parts: list[str] = []

    # Header
    parts.append(f"## {cve_id} — Vulnerability Analysis\n")

    # Description
    if description:
        parts.append(f"### Description\n{description.strip()}\n")

    # Severity
    if cvss_score is not None:
        severity = _cvss_severity(cvss_score)
        parts.append(f"### Severity\n**CVSS Score:** {cvss_score} ({severity})")
        if cvss_vector:
            parts.append(f"**CVSS Vector:** `{cvss_vector}`")
        parts.append("")

    # CWE (Weakness)
    if cwe_ids:
        parts.append("### Weakness Classification (CWE)")
        for cwe in cwe_ids:
            cwe_name = _cwe_name(cwe)
            parts.append(f"- **{cwe}**: {cwe_name}")
        parts.append("")

    # CPE (Affected Products)
    if cpe_ids:
        parts.append("### Affected Products (CPE)")
        for cpe in cpe_ids[:10]:  # Limit to 10 for readability
            parts.append(f"- `{cpe}`")
        if len(cpe_ids) > 10:
            parts.append(f"- ... and {len(cpe_ids) - 10} more")
        parts.append("")

    # CAPEC (Attack Patterns)
    if capec_ids:
        parts.append("### Attack Patterns (CAPEC)")
        for capec in capec_ids:
            capec_name = _capec_name(capec)
            parts.append(f"- **{capec}**: {capec_name}")
        parts.append("")

    # MITRE ATT&CK Techniques
    attack_ids = _resolve_attack_ids(capec_ids, cwe_ids)
    if attack_ids:
        parts.append("### MITRE ATT&CK Techniques")
        for tid in attack_ids:
            tactic_id = get_tactic_for_technique(tid)
            tactic_name = get_tactic_name(tactic_id) if tactic_id else "Unknown"
            parts.append(f"- **{tid}** ({tactic_name})")
        parts.append("")

    # Exploit-DB References
    if exploitdb_ids:
        parts.append("### Known Exploits (Exploit-DB)")
        for eid in exploitdb_ids:
            parts.append(f"- Exploit-DB: https://www.exploit-db.com/exploits/{eid}")
        parts.append("")

    # Reproduction Steps
    if reproduction_steps:
        parts.append("### Reproduction Steps\n")
        parts.append(reproduction_steps.strip())
        parts.append("")

    # Remediation Guidance
    parts.append("### Remediation Guidance\n")
    remediation_items = _remediation_guidance(cwe_ids, attack_ids)
    for item in remediation_items:
        parts.append(f"- {item}")
    parts.append("")

    return "\n".join(parts)


def _cvss_severity(score: float) -> str:
    """Return severity label for a CVSS score."""
    if score >= 9.0:
        return "CRITICAL"
    elif score >= 7.0:
        return "HIGH"
    elif score >= 4.0:
        return "MEDIUM"
    elif score > 0.0:
        return "LOW"
    return "INFO"


def _cwe_name(cwe_id: str) -> str:
    """Return a human-readable name for common CWE IDs."""
    cwe_names: dict[str, str] = {
        "CWE-22": "Path Traversal",
        "CWE-78": "OS Command Injection",
        "CWE-79": "Cross-site Scripting (XSS)",
        "CWE-89": "SQL Injection",
        "CWE-94": "Code Injection",
        "CWE-119": "Buffer Overflow",
        "CWE-125": "Out-of-bounds Read",
        "CWE-190": "Integer Overflow or Wraparound",
        "CWE-200": "Information Exposure",
        "CWE-250": "Execution with Unnecessary Privileges",
        "CWE-269": "Improper Privilege Management",
        "CWE-287": "Improper Authentication",
        "CWE-306": "Missing Authentication for Critical Function",
        "CWE-311": "Missing Encryption of Sensitive Data",
        "CWE-312": "Cleartext Storage of Sensitive Information",
        "CWE-327": "Use of Broken or Risky Cryptographic Algorithm",
        "CWE-352": "Cross-Site Request Forgery (CSRF)",
        "CWE-400": "Uncontrolled Resource Consumption",
        "CWE-434": "Unrestricted Upload of File with Dangerous Type",
        "CWE-502": "Deserialization of Untrusted Data",
        "CWE-611": "XML External Entity (XXE) Injection",
        "CWE-613": "Insufficient Session Expiration",
        "CWE-732": "Incorrect Permission Assignment for Critical Resource",
        "CWE-770": "Allocation of Resources Without Limits",
        "CWE-776": "XML Entity Expansion",
        "CWE-862": "Missing Authorization",
        "CWE-863": "Incorrect Authorization",
        "CWE-918": "Server-Side Request Forgery (SSRF)",
        "CWE-922": "Insecure Storage of Sensitive Information",
        "CWE-1021": "Improper Render",
    }
    return cwe_names.get(cwe_id, "See MITRE CWE database for details")


def _capec_name(capec_id: str) -> str:
    """Return a human-readable name for common CAPEC IDs."""
    capec_names: dict[str, str] = {
        "CAPEC-1": "Accessing Functionality Not Properly Constrained",
        "CAPEC-6": "Argument Injection",
        "CAPEC-10": "Buffer Overflow via Environment Variables",
        "CAPEC-15": "Command Delimiters",
        "CAPEC-24": "Abuse of Functionality",
        "CAPEC-31": "Accessing/Modifying System-Level Service",
        "CAPEC-55": "Path Traversal",
        "CAPEC-62": "Cross Site Request Forgery",
        "CAPEC-63": "Cross-Site Scripting (XSS)",
        "CAPEC-66": "SQL Injection",
        "CAPEC-88": "OS Command Injection",
        "CAPEC-90": "Parameter Injection",
        "CAPEC-91": "Forced Browse",
        "CAPEC-94": "Man-in-the-Middle",
        "CAPEC-97": "Cryptanalysis",
        "CAPEC-99": "Brute Force",
        "CAPEC-100": "Overflow Buffers",
        "CAPEC-116": "Excavation",
        "CAPEC-117": "Interception",
        "CAPEC-118": "Resource Exhaustion",
        "CAPEC-121": "Exploitation of Trusted Credentials",
        "CAPEC-122": "Exploit Trusted Relationships",
        "CAPEC-125": "Path Traversal (Execution)",
        "CAPEC-131": "Data-Structure Attacks",
        "CAPEC-150": "Command-Line Execution",
        "CAPEC-169": "Footprinting",
        "CAPEC-170": "Web Scraping",
        "CAPEC-180": "Exploiting Incorrectly Configured SSL/TLS",
        "CAPEC-196": "XSS Targeting Non-Script Elements",
        "CAPEC-197": "XSS Targeting Script Elements",
        "CAPEC-204": "Lifting Sensitive Data Embedded in Cache",
        "CAPEC-225": "XXE",
        "CAPEC-248": "Command Injection",
        "CAPEC-251": "Local File Inclusion",
        "CAPEC-253": "Remote File Inclusion",
        "CAPEC-272": "Protocol Manipulation",
        "CAPEC-273": "HTTP Response Splitting",
    }
    return capec_names.get(capec_id, "See MITRE CAPEC database for details")


def _remediation_guidance(cwe_ids: list[str], attack_ids: list[str]) -> list[str]:
    """Generate remediation guidance based on CWE and ATT&CK IDs."""
    guidance: list[str] = []

    # CWE-based remediation
    cwe_remediations: dict[str, list[str]] = {
        "CWE-22": [
            "Validate and sanitize all file path inputs",
            "Use chroot jails or restricted directories",
            "Implement canonical path verification",
        ],
        "CWE-78": [
            "Use parameterized commands instead of string concatenation",
            "Implement strict input validation with allowlists",
            "Apply principle of least privilege to command execution",
        ],
        "CWE-79": [
            "Encode output based on context (HTML, JavaScript, URL, CSS)",
            "Implement Content Security Policy (CSP) headers",
            "Use modern framework auto-escaping features",
        ],
        "CWE-89": [
            "Use parameterized queries or prepared statements",
            "Implement input validation and allowlisting",
            "Apply least-privilege database permissions",
        ],
        "CWE-94": [
            "Avoid eval() and similar dynamic code execution",
            "Use allowlists for permitted operations",
            "Implement strict input validation",
        ],
        "CWE-119": [
            "Use memory-safe languages or safe string libraries",
            "Implement bounds checking for all buffer operations",
            "Enable compiler protections (ASLR, DEP, stack canaries)",
        ],
        "CWE-200": [
            "Implement proper access controls on error pages",
            "Sanitize error messages to remove sensitive data",
            "Use generic error responses for unauthenticated users",
        ],
        "CWE-287": [
            "Implement multi-factor authentication",
            "Use secure session management",
            "Apply defense-in-depth authentication validation",
        ],
        "CWE-306": [
            "Require authentication for all sensitive functions",
            "Implement proper authorization checks",
            "Use middleware to enforce authentication globally",
        ],
        "CWE-311": [
            "Encrypt sensitive data at rest and in transit",
            "Use TLS 1.2+ for all communications",
            "Implement proper key management",
        ],
        "CWE-352": [
            "Implement anti-CSRF tokens",
            "Validate the Origin and Referer headers",
            "Use SameSite cookie attribute",
        ],
        "CWE-434": [
            "Validate file type by content, not extension",
            "Store uploads outside the web root",
            "Implement file size limits and virus scanning",
        ],
        "CWE-502": [
            "Avoid deserializing untrusted data",
            "Use safe serialization formats (JSON, Protocol Buffers)",
            "Implement integrity checks and type allowlisting",
        ],
        "CWE-611": [
            "Disable DTD processing and external entity resolution",
            "Use JSON instead of XML where possible",
            "Configure XML parsers to disallow external entities",
        ],
        "CWE-918": [
            "Implement allowlists for permitted destinations",
            "Block requests to private/internal IP ranges",
            "Validate and sanitize all URL inputs",
        ],
    }

    for cwe in cwe_ids:
        if cwe in cwe_remediations:
            guidance.extend(cwe_remediations[cwe])

    # ATT&CK-based remediation
    attack_remediations: dict[str, str] = {
        "T1190": "Patch internet-facing applications; implement WAF; segment network",
        "T1059": "Restrict script execution; implement application allowlisting; monitor PowerShell/bash usage",
        "T1078": "Enforce MFA; implement PAM; audit account usage",
        "T1110": "Implement account lockout; enforce password complexity; use MFA",
        "T1543": "Audit services and startup items; monitor for unauthorized service creation",
        "T1547": "Audit startup and logon items; monitor for persistence mechanisms",
        "T1562": "Monitor security tool status; implement tamper protection",
        "T1565": "Verify data integrity; implement checksums; monitor for unauthorized data modification",
        "T1592": "Minimize information disclosure; harden web servers; implement access controls",
        "T1082": "Limit system information exposure; implement access controls on enumeration APIs",
        "T1021": "Restrict lateral movement; implement network segmentation; use jump servers",
        "T1560": "Encrypt data at rest; implement DLP; monitor for data staging",
        "T1499": "Implement rate limiting; use DDoS mitigation services; monitor resource usage",
        "T1552": "Secure credential storage; audit for credentials in files; implement secret management",
        "T1070": "Centralize logging; implement log integrity monitoring; use tamper-proof log storage",
        "T1203": "Keep software updated; implement application allowlisting; sandbox untrusted content",
        "T1068": "Apply least privilege; patch privilege escalation vulnerabilities; monitor for exploit attempts",
    }

    for tid in attack_ids:
        if tid in attack_remediations:
            guidance.append(f"[{tid}] {attack_remediations[tid]}")

    if not guidance:
        guidance.append(
            "Apply vendor patches and follow CVE-specific remediation guidance"
        )
        guidance.append("Implement defense-in-depth security controls")
        guidance.append("Monitor for exploitation attempts using threat intelligence")

    return guidance


# ---------------------------------------------------------------------------
# Data loading functions
# ---------------------------------------------------------------------------
def _find_data_files(input_dir: Path) -> list[Path]:
    """Find Vuln2Action data files in the input directory.

    Supports CSV, JSON, and JSONL formats. Looks for files with names
    containing 'vuln2action', 'vuln', 'cve', or 'dataset'.
    """
    candidates: list[Path] = []

    if not input_dir.exists():
        return candidates

    # Look for data files
    for ext in ("*.csv", "*.json", "*.jsonl", "*.jsonl.gz"):
        for fp in sorted(input_dir.rglob(ext)):
            name_lower = fp.name.lower()
            if any(
                kw in name_lower
                for kw in ("vuln2action", "vuln", "cve", "dataset", "data")
            ):
                candidates.append(fp)

    # If no named matches, try all data files
    if not candidates:
        for ext in ("*.csv", "*.json", "*.jsonl", "*.jsonl.gz"):
            candidates.extend(sorted(input_dir.rglob(ext)))

    return candidates


def _load_csv(filepath: Path) -> list[dict[str, Any]]:
    """Load Vuln2Action data from CSV format."""
    records: list[dict[str, Any]] = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))
    except Exception as e:
        print(f"  [WARN] Error reading CSV {filepath}: {e}", file=sys.stderr)
    return records


def _load_json(filepath: Path) -> list[dict[str, Any]]:
    """Load Vuln2Action data from JSON/JSONL format."""
    records: list[dict[str, Any]] = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception as e:
        print(f"  [WARN] Error reading {filepath}: {e}", file=sys.stderr)
        return records

    # Try JSONL first (one JSON object per line)
    if content.startswith("{") or "\n{" in content:
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if records:
            return records

    # Try as single JSON array
    try:
        data = json.loads(content)
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            # Might be wrapped in a key like "data" or "records"
            for key in ("data", "records", "results", "vulnerabilities", "items"):
                if key in data and isinstance(data[key], list):
                    records = data[key]
                    break
            if not records:
                records = [data]
    except json.JSONDecodeError:
        pass

    return records


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Vuln2Action record to a common schema.

    Handles various field naming conventions found in the dataset:
    - CVE ID: cve_id, cve, CVE, vuln_id
    - Description: description, desc, vulnerability_description
    - CWE: cwe_id, cwe, CWE, weakness
    - CPE: cpe_id, cpe, CPE, affected_products
    - CAPEC: capec_id, capec, CAPEC, attack_pattern
    - Exploit-DB: exploitdb_id, exploit_db, edb_id, exploit
    - CVSS: cvss_score, cvss, severity_score, base_score
    - CVSS Vector: cvss_vector, cvss_v3_vector
    - Reproduction: reproduction_steps, repro_steps, steps
    """

    def _get_first(record: dict, *keys: str) -> str:
        for key in keys:
            val = record.get(key, "")
            if val:
                return str(val).strip()
        return ""

    def _get_list(record: dict, *keys: str) -> list[str]:
        for key in keys:
            val = record.get(key, [])
            if val:
                if isinstance(val, str):
                    # Handle comma-separated, semicolon-separated, or pipe-separated
                    import re

                    items = re.split(r"[,;|]", val)
                    return [v.strip() for v in items if v.strip()]
                elif isinstance(val, list):
                    return [str(v).strip() for v in val if v]
        return []

    cve_id = _get_first(record, "cve_id", "cve", "CVE", "vuln_id", "id")
    description = _get_first(
        record, "description", "desc", "vulnerability_description", "summary"
    )
    cwe_ids = _get_list(record, "cwe_id", "cwe", "CWE", "weakness", "cwe_ids")
    cpe_ids = _get_list(record, "cpe_id", "cpe", "CPE", "affected_products", "cpes")
    capec_ids = _get_list(
        record, "capec_id", "capec", "CAPEC", "attack_pattern", "capec_ids"
    )
    exploitdb_ids = _get_list(
        record, "exploitdb_id", "exploit_db", "edb_id", "exploit", "exploitdb_ids"
    )

    # CVSS score
    cvss_score = None
    cvss_raw = _get_first(record, "cvss_score", "cvss", "severity_score", "base_score")
    if cvss_raw:
        try:
            cvss_score = float(cvss_raw)
        except (ValueError, TypeError):
            pass

    cvss_vector = _get_first(record, "cvss_vector", "cvss_v3_vector", "vector")

    # Reproduction steps
    reproduction_steps = _get_first(
        record, "reproduction_steps", "repro_steps", "steps", "reproduction"
    )

    # Normalize CWE IDs to CWE-XXX format
    normalized_cwe = []
    for cwe in cwe_ids:
        cwe = cwe.strip()
        if cwe.upper().startswith("CWE"):
            if not cwe.upper().startswith("CWE-"):
                # CWE22 → CWE-22
                num = cwe.upper().replace("CWE", "")
                cwe = f"CWE-{num}"
            normalized_cwe.append(cwe.upper())
        elif cwe.isdigit():
            normalized_cwe.append(f"CWE-{cwe}")

    # Normalize CAPEC IDs to CAPEC-XXX format
    normalized_capec = []
    for capec in capec_ids:
        capec = capec.strip()
        if capec.upper().startswith("CAPEC"):
            if not capec.upper().startswith("CAPEC-"):
                num = capec.upper().replace("CAPEC", "")
                capec = f"CAPEC-{num}"
            normalized_capec.append(capec.upper())
        elif capec.isdigit():
            normalized_capec.append(f"CAPEC-{capec}")

    # Normalize CVE IDs to CVE-YYYY-NNNNN format
    cve_id = cve_id.strip().upper()
    if cve_id and not cve_id.startswith("CVE-"):
        import re

        m = re.match(r"CVE(\d{4})[-_]?(\d+)", cve_id)
        if m:
            cve_id = f"CVE-{m.group(1)}-{m.group(2)}"
        elif cve_id.isdigit():
            cve_id = cve_id  # Leave as-is if just a number

    # Normalize Exploit-DB IDs
    normalized_edb = []
    for eid in exploitdb_ids:
        eid = eid.strip()
        # Remove "EDB-" or "EXPLOIT-DB-" prefix if present
        eid = eid.upper().replace("EXPLOIT-DB-", "").replace("EDB-", "")
        if eid.isdigit():
            normalized_edb.append(eid)

    return {
        "cve_id": cve_id,
        "description": description,
        "cwe_ids": normalized_cwe,
        "cpe_ids": cpe_ids,
        "capec_ids": normalized_capec,
        "exploitdb_ids": normalized_edb,
        "cvss_score": cvss_score,
        "cvss_vector": cvss_vector,
        "reproduction_steps": reproduction_steps,
    }


# ---------------------------------------------------------------------------
# Generate training pairs
# ---------------------------------------------------------------------------
def generate_pairs(
    records: list[dict[str, Any]],
    limit: int = 0,
) -> list[dict[str, Any]]:
    """Generate AttackLM training pairs from Vuln2Action records."""
    pairs: list[dict[str, Any]] = []

    for record in records:
        normalized = _normalize_record(record)
        cve_id = normalized["cve_id"]

        if not cve_id:
            continue

        # Resolve ATT&CK technique IDs from CAPEC and CWE mappings
        attack_ids = _resolve_attack_ids(normalized["capec_ids"], normalized["cwe_ids"])

        # Add tactic info
        tactic_id = None
        tactic_name = None
        kill_chain_phase = None
        if attack_ids:
            tactic_id = get_tactic_for_technique(attack_ids[0])
            if tactic_id:
                tactic_name = get_tactic_name(tactic_id)
                kill_chain_phase = tactic_name

        # Format the comprehensive response
        response = _format_cve_description(
            cve_id=cve_id,
            description=normalized["description"],
            cwe_ids=normalized["cwe_ids"],
            cpe_ids=normalized["cpe_ids"],
            capec_ids=normalized["capec_ids"],
            exploitdb_ids=normalized["exploitdb_ids"],
            cvss_score=normalized["cvss_score"],
            cvss_vector=normalized["cvss_vector"],
            reproduction_steps=normalized["reproduction_steps"],
        )

        # Build instruction (user message)
        cwe_str = (
            ", ".join(normalized["cwe_ids"][:3])
            if normalized["cwe_ids"]
            else "unknown weakness"
        )
        instruction = (
            f"Explain the exploitation of {cve_id} and its impact. "
            f"Include the CWE classification ({cwe_str}), "
            f"affected products, attack patterns, and remediation guidance."
        )

        pair: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ],
            "mitre_ids": attack_ids,
            **ATTRIBUTION,
        }

        if tactic_id:
            pair["mitre_tactic_id"] = tactic_id
        if tactic_name:
            pair["tactic"] = tactic_name
        if kill_chain_phase:
            pair["kill_chain_phase"] = kill_chain_phase

        pairs.append(pair)

        if limit > 0 and len(pairs) >= limit:
            break

    return pairs


# ---------------------------------------------------------------------------
# Manual download instructions
# ---------------------------------------------------------------------------
DOWNLOAD_INSTRUCTIONS = """
=============================================================================
Vuln2Action Dataset — Manual Download Instructions
=============================================================================

The Vuln2Action dataset was not found in the expected directory.

Option 1 — Clone from GitHub (if available):
    cd /path/to/AttackLM
    mkdir -p data/vuln2action
    git clone https://github.com/Vuln2Action/Vuln2Action.git data/vuln2action

Option 2 — Download from HuggingFace (if available):
    pip install datasets
    python -c "
    from datasets import load_dataset
    ds = load_dataset('Vuln2Action/vuln2action')
    ds['train'].to_json('data/vuln2action/vuln2action.jsonl')
    "

Option 3 — Use the paper's supplementary materials:
    Download from the JISA 2026 paper page and extract to data/vuln2action/

Option 4 — Use a local copy with --input-dir:
    python scripts/extract_vuln2action.py --input-dir /path/to/local/vuln2action

Expected file formats: CSV, JSON, or JSONL with fields:
    cve_id, description, cwe_id, cpe_id, capec_id, exploitdb_id,
    cvss_score, reproduction_steps
=============================================================================
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract Vuln2Action vulnerability data into AttackLM training pairs"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Custom input directory for Vuln2Action data files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse records and print stats without writing output",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max records to process (0=all)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else DEFAULT_INPUT_DIR

    print("AttackLM — Extract Vuln2Action Training Pairs")
    print(f"  Input dir: {input_dir}")
    print(f"  Output:    {OUTPUT_PATH}")
    print()

    # Find data files
    data_files = _find_data_files(input_dir)

    if not data_files:
        print("ERROR: No Vuln2Action data files found.", file=sys.stderr)
        print(DOWNLOAD_INSTRUCTIONS, file=sys.stderr)
        return 1

    print(f"  Found {len(data_files)} data file(s)")
    for f in data_files:
        print(f"    - {f}")

    # Load all records
    all_records: list[dict[str, Any]] = []
    for filepath in data_files:
        suffix = filepath.suffix.lower()
        if suffix == ".csv":
            records = _load_csv(filepath)
        elif suffix in (".json", ".jsonl"):
            records = _load_json(filepath)
        else:
            # Try JSON for .jsonl.gz or unknown
            records = _load_json(filepath)

        print(f"  [{filepath.name}] Loaded {len(records)} records")
        all_records.extend(records)

    print(f"\n  Total records: {len(all_records)}")

    if not all_records:
        print("ERROR: No valid records found in data files.", file=sys.stderr)
        return 1

    # Generate training pairs
    pairs = generate_pairs(all_records, limit=args.limit)

    if not pairs:
        print("ERROR: No training pairs generated from records.", file=sys.stderr)
        return 1

    # Summary statistics
    total_pairs = len(pairs)
    cve_count = len(
        set(
            p["messages"][1]["content"].split(" ")[3]
            for p in pairs
            if len(p["messages"][1]["content"].split(" ")) > 3
        )
    )
    all_attack_ids: set[str] = set()
    all_cwe: set[str] = set()
    all_capec: set[str] = set()
    has_edb = 0
    has_repro = 0

    for pair in pairs:
        all_attack_ids.update(pair.get("mitre_ids", []))
        # Parse CWE/CAPEC from assistant content
        content = pair["messages"][2]["content"]
        if "Exploit-DB:" in content:
            has_edb += 1
        if "Reproduction Steps" in content:
            has_repro += 1

    print(f"\n{'=' * 60}")
    print(f"  Records processed: {len(all_records)}")
    print(f"  Training pairs generated: {total_pairs}")
    print(f"  Pairs with Exploit-DB refs: {has_edb}")
    print(f"  Pairs with reproduction steps: {has_repro}")
    print(f"  Unique MITRE ATT&CK IDs: {len(all_attack_ids)}")
    if all_attack_ids:
        print(
            f"    {', '.join(sorted(all_attack_ids)[:20])}"
            f"{'...' if len(all_attack_ids) > 20 else ''}"
        )

    if args.dry_run:
        print(f"\n{'=' * 60}")
        print("  DRY RUN — No files written")
        print(f"{'=' * 60}")

        # Show sample pairs
        if pairs:
            print(f"\n  Sample pair (first):\n")
            sample = pairs[0]
            sample_json = json.dumps(sample, indent=2, ensure_ascii=False)
            print(sample_json[:3000])
            if len(sample_json) > 3000:
                print("  ... (truncated)")
        return 0

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n  Output written: {total_pairs} pairs → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
