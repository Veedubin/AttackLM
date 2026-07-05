#!/usr/bin/env python3
# CREDITS — DATA SOURCE ATTRIBUTION
# ----------------------------------
# This script transforms data from: NYU CTF Bench (NYU-LLM-CTF)
# Repository: https://github.com/NYU-LLM-CTF/NYU_CTF_Bench
# License:    GNU General Public License v2.0
# Copyright:  (c) NYU Secure Systems Lab
#
# The output JSONL is a *transformation* of upstream CTF challenge metadata
# and MITRE ATT&CK mappings into OpenAI-style chat triples.
# See /ATTRIBUTION.md for full per-source attribution and re-distribution
# guidance.
# ----------------------------------
"""Deterministic extraction of NYU CTF Bench challenges into AttackLM JSONL training pairs.

Walks the NYU CTF Bench repository (cloned or local) and parses challenge
metadata from ``challenge.json`` files, combined with the official MITRE
ATT&CK mapping from ``mitre_attack_mapping/``. For each challenge, generates
OpenAI-style message triples:

  - **Pair type 1** — Challenge solving approach (always)
  - **Pair type 2** — Detailed solution methodology (always)
  - **Pair type 3** — Detection and defensive guidance (when MITRE mapping exists)

Output: ``data/datasets/buckets/sources/nyu-ctf-bench/ctf_challenges/data.jsonl``

Usage:
    python scripts/extract_nyu_ctf_bench.py
    python scripts/extract_nyu_ctf_bench.py --repo-path ./data/NYU_CTF_Bench
    python scripts/extract_nyu_ctf_bench.py --limit 10 --output-dir /tmp/test_nyuctf
    python scripts/extract_nyu_ctf_bench.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Import shared MITRE tactic lookup
sys.path.insert(0, str(Path(__file__).parent))
from mitre_tactic_lookup import get_tactic_for_technique, get_tactic_name

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_REPO_PATH = BASE_DIR / "data" / "NYU_CTF_Bench"
DEFAULT_OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "datasets"
    / "buckets"
    / "sources"
    / "nyu-ctf-bench"
    / "ctf_challenges"
)

# ---------------------------------------------------------------------------
# System message
# ---------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are an authorized CTF challenge analyst and offensive security specialist. "
    "You provide detailed walkthroughs, exploitation strategies, and defensive "
    "recommendations for Capture The Flag security challenges."
)

# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------
ATTRIBUTION = {
    "source": "nyu-ctf-bench",
    "source_uri": "https://github.com/NYU-LLM-CTF/NYU_CTF_Bench",
    "license": "GPL-2.0",
    "license_uri": "https://github.com/NYU-LLM-CTF/NYU_CTF_Bench/blob/main/LICENSE",
    "rights_contact": "NYU Secure Systems Lab",
    "attribution_text": (
        "Copyright (c) NYU Secure Systems Lab. Licensed under GNU General Public "
        "License v2.0. See https://github.com/NYU-LLM-CTF/NYU_CTF_Bench/blob/main/LICENSE."
    ),
}

# ---------------------------------------------------------------------------
# CTF category → friendly name + default MITRE technique mapping
# ---------------------------------------------------------------------------
CATEGORY_INFO: dict[str, dict[str, Any]] = {
    "rev": {
        "friendly": "reverse engineering",
        "default_techniques": ["T1057"],  # Process Discovery / binary analysis
        "description_prefix": (
            "This is a reverse engineering (rev) CTF challenge. Analyze the binary, "
            "understand its logic, and extract the flag."
        ),
    },
    "pwn": {
        "friendly": "binary exploitation",
        "default_techniques": ["T1068"],  # Exploitation for Privilege Escalation
        "description_prefix": (
            "This is a binary exploitation (pwn) CTF challenge. Find a vulnerability "
            "in the binary and exploit it to gain control."
        ),
    },
    "web": {
        "friendly": "web security",
        "default_techniques": ["T1190"],  # Exploit Public-Facing Application
        "description_prefix": (
            "This is a web security CTF challenge. Identify and exploit vulnerabilities "
            "in the web application to retrieve the flag."
        ),
    },
    "crypto": {
        "friendly": "cryptography",
        "default_techniques": ["T1600"],  # Weaken Encryption
        "description_prefix": (
            "This is a cryptography CTF challenge. Analyze the cryptographic scheme, "
            "find weaknesses, and recover the plaintext or flag."
        ),
    },
    "forensics": {
        "friendly": "digital forensics",
        "default_techniques": ["T1005"],  # Data from Local System
        "description_prefix": (
            "This is a digital forensics CTF challenge. Analyze the provided artifacts "
            "to uncover evidence and extract the flag."
        ),
    },
    "misc": {
        "friendly": "miscellaneous",
        "default_techniques": ["T1204"],  # User Execution
        "description_prefix": (
            "This is a miscellaneous CTF challenge. Apply creative problem-solving "
            "and diverse security skills to find the flag."
        ),
    },
}


# ---------------------------------------------------------------------------
# Clone / locate the repo
# ---------------------------------------------------------------------------
def ensure_repo(repo_path: Path) -> Path:
    """Ensure the NYU CTF Bench repo is available, cloning if necessary.

    Returns the path to the repo root.
    """
    if repo_path.exists() and (repo_path / "test_dataset.json").exists():
        return repo_path

    print(f"Repo not found at {repo_path}")
    print("Cloning NYU CTF Bench repository...")
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/NYU-LLM-CTF/NYU_CTF_Bench.git",
            str(repo_path),
        ],
        check=True,
    )
    return repo_path


# ---------------------------------------------------------------------------
# Load challenge data
# ---------------------------------------------------------------------------
def load_dataset(repo_path: Path, split: str = "test") -> dict[str, dict[str, Any]]:
    """Load the dataset JSON for the given split.

    Returns a dict mapping canonical challenge names to metadata.
    """
    dataset_file = repo_path / f"{split}_dataset.json"
    if not dataset_file.exists():
        print(f"WARNING: Dataset file not found: {dataset_file}", file=sys.stderr)
        return {}

    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def load_mitre_mapping(repo_path: Path, split: str = "test") -> dict[str, Any]:
    """Load the MITRE ATT&CK mapping for challenges.

    Returns a dict with 'mapping' (challenge → technique IDs) and
    'techniques' (technique ID → name).
    """
    mapping_file = repo_path / "mitre_attack_mapping" / f"{split}_mapping.json"
    if not mapping_file.exists():
        print(
            f"WARNING: MITRE mapping file not found: {mapping_file}",
            file=sys.stderr,
        )
        return {"mapping": {}, "techniques": {}}

    with open(mapping_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def load_challenge_json(repo_path: Path, challenge_path: str) -> dict[str, Any] | None:
    """Load a challenge.json file from the repo.

    Returns the parsed JSON dict, or None if the file doesn't exist.
    """
    challenge_json = repo_path / challenge_path / "challenge.json"
    if not challenge_json.exists():
        return None

    try:
        with open(challenge_json, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [WARN] Cannot parse {challenge_json}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Build training pairs
# ---------------------------------------------------------------------------
def build_pairs(
    canonical_name: str,
    challenge_info: dict[str, Any],
    challenge_json: dict[str, Any] | None,
    mitre_techniques: list[str],
    technique_names: dict[str, str],
) -> list[dict[str, Any]]:
    """Build 2-3 training pairs from a single CTF challenge.

    Generates:
      - Pair 1: Challenge solving approach (always)
      - Pair 2: Detailed solution methodology (always)
      - Pair 3: Detection and defensive guidance (if MITRE mapping exists)
    """
    pairs: list[dict[str, Any]] = []

    category = challenge_info.get("category", "misc")
    year = challenge_info.get("year", "unknown")
    event = challenge_info.get("event", "unknown")
    challenge_name = challenge_info.get("challenge", canonical_name)

    cat_info = CATEGORY_INFO.get(category, CATEGORY_INFO["misc"])
    friendly_category = cat_info["friendly"]
    default_techniques = cat_info["default_techniques"]

    # Use MITRE mapping if available, otherwise fall back to category defaults
    mitre_ids = mitre_techniques if mitre_techniques else default_techniques

    # Resolve challenge description
    description = ""
    flag = ""
    files: list[str] = []
    has_server = False
    server_info = ""

    if challenge_json:
        description = challenge_json.get("description", "")
        flag = challenge_json.get("flag", "")
        files = challenge_json.get("files", [])
        box = challenge_json.get("box", "")
        port = challenge_json.get("internal_port", "")
        has_server = bool(box)

        if has_server:
            server_info = f"Server at `{box}:{port}`" if port else f"Server at `{box}`"

        # Resolve {box} and {port} placeholders in description
        if description and (box or port):
            description = description.replace("{box}", box or "N/A")
            description = description.replace("{port}", str(port) if port else "N/A")

    # Build a clean description for the prompt
    challenge_label = f"{year} {event} {friendly_category}: {challenge_name}"

    # Format MITRE technique annotations
    technique_annotations = []
    for tid in mitre_ids:
        tname = technique_names.get(tid, "")
        tactic_id = get_tactic_for_technique(tid)
        tactic_name = get_tactic_name(tactic_id) if tactic_id else ""
        if tname:
            technique_annotations.append(f"{tid} ({tname}) — {tactic_name}")
        else:
            technique_annotations.append(tid)

    techniques_str = (
        ", ".join(technique_annotations)
        if technique_annotations
        else "general security testing"
    )

    # ------------------------------------------------------------------
    # Pair 1: Challenge solving approach
    # ------------------------------------------------------------------
    user_msg_1 = f"Solve this CTF challenge: {challenge_label}"

    if description:
        user_msg_1 += f"\n\n{description}"

    if files:
        file_list = ", ".join(files)
        user_msg_1 += f"\n\nProvided files: {file_list}"

    if has_server:
        user_msg_1 += f"\n\n{server_info}"

    # Build assistant response
    approach_parts: list[str] = []
    approach_parts.append(f"## CTF Challenge Analysis: {challenge_name}")
    approach_parts.append(f"**Category:** {friendly_category}")
    approach_parts.append(f"**Event:** {year} {event}")

    if mitre_ids:
        approach_parts.append(f"**Relevant MITRE ATT&CK Techniques:** {techniques_str}")

    approach_parts.append("")
    approach_parts.append("### Initial Assessment")
    approach_parts.append(
        f"This is a **{friendly_category}** challenge from the {year} {event}."
    )

    if description:
        approach_parts.append("")
        approach_parts.append("### Challenge Description")
        approach_parts.append(description)

    if files:
        approach_parts.append("")
        approach_parts.append("### Available Files")
        for f in files:
            approach_parts.append(f"- `{f}`")

    if has_server:
        approach_parts.append("")
        approach_parts.append("### Network Service")
        approach_parts.append(server_info)

    # Add category-specific approach hints
    approach_parts.append("")
    approach_parts.append("### Recommended Approach")
    approach_hints = {
        "rev": (
            "1. Identify the binary type and architecture (file, strings, checksec)\n"
            "2. Disassemble and analyze the binary logic\n"
            "3. Look for flag comparison routines or encoding schemes\n"
            "4. Trace the flag generation or validation path\n"
            "5. Extract or reverse the flag"
        ),
        "pwn": (
            "1. Identify binary protections (NX, ASLR, canaries, PIE)\n"
            "2. Find vulnerability (buffer overflow, format string, UAF, etc.)\n"
            "3. Determine exploit strategy based on available protections\n"
            "4. Craft exploit payload (ROP chain, shellcode, etc.)\n"
            "5. Test locally, then against the remote server"
        ),
        "web": (
            "1. Enumerate the web application endpoints and functionality\n"
            "2. Check for common vulnerabilities (SQLi, XSS, LFI, RCE, etc.)\n"
            "3. Analyze client-side code for hidden endpoints or logic\n"
            "4. Exploit the identified vulnerability to access the flag\n"
            "5. Verify the flag format matches the expected pattern"
        ),
        "crypto": (
            "1. Identify the cryptographic algorithm and parameters\n"
            "2. Analyze the implementation for weaknesses\n"
            "3. Look for common crypto vulnerabilities (weak RNG, key reuse, etc.)\n"
            "4. Apply appropriate cryptanalysis technique\n"
            "5. Recover the plaintext or forge the flag"
        ),
        "forensics": (
            "1. Identify the artifact type (disk image, memory dump, pcap, etc.)\n"
            "2. Extract metadata and timeline information\n"
            "3. Search for strings, hidden files, or embedded data\n"
            "4. Apply file carving or data recovery techniques\n"
            "5. Correlate findings to extract the flag"
        ),
        "misc": (
            "1. Read the challenge description carefully for hints\n"
            "2. Identify the challenge type (encoding, steganography, trivia, etc.)\n"
            "3. Apply appropriate technique based on the category\n"
            "4. Look for patterns in the provided data\n"
            "5. Validate the flag format and submit"
        ),
    }
    approach_parts.append(approach_hints.get(category, approach_hints["misc"]))

    assistant_msg_1 = "\n".join(approach_parts)

    # Add tactic info
    mitre_tactic_id = None
    tactic_name = None
    kill_chain_phase = None
    if mitre_ids:
        for tid in mitre_ids:
            tid_resolved = get_tactic_for_technique(tid)
            if tid_resolved:
                mitre_tactic_id = tid_resolved
                tactic_name = get_tactic_name(tid_resolved)
                kill_chain_phase = tactic_name
                break

    pair1 = {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg_1},
            {"role": "assistant", "content": assistant_msg_1},
        ],
        "mitre_ids": mitre_ids,
        "category": category,
        "year": str(year),
        "event": event,
        **ATTRIBUTION,
    }
    if mitre_tactic_id:
        pair1["mitre_tactic_id"] = mitre_tactic_id
    if tactic_name:
        pair1["tactic"] = tactic_name
    if kill_chain_phase:
        pair1["kill_chain_phase"] = kill_chain_phase

    pairs.append(pair1)

    # ------------------------------------------------------------------
    # Pair 2: Detailed solution methodology
    # ------------------------------------------------------------------
    user_msg_2 = f"Provide a detailed solution methodology for the {friendly_category} CTF challenge: {challenge_name} ({year} {event})"

    if description:
        user_msg_2 += f"\n\nChallenge description:\n{description}"

    solution_parts: list[str] = []
    solution_parts.append(f"## Solution Methodology: {challenge_name}")
    solution_parts.append(
        f"**Category:** {friendly_category} | **Event:** {year} {event}"
    )
    solution_parts.append("")

    solution_parts.append("### Step-by-Step Approach")
    solution_steps = {
        "rev": [
            "1. **File Identification**: Run `file` command to identify binary type, architecture, and linking",
            "2. **String Analysis**: Use `strings` to extract readable strings, potential flags, and API calls",
            "3. **Security Check**: Run `checksec` to identify protections (NX, canary, PIE, RELRO)",
            "4. **Disassembly**: Load into Ghidra/IDA/r2 for static analysis of key functions",
            "5. **Logic Tracing**: Identify the flag validation or generation routine",
            "6. **Constraint Solving**: Use angr/z3 or manual analysis to solve for the flag",
            "7. **Flag Extraction**: Extract the flag from the solved constraints or patched binary",
        ],
        "pwn": [
            "1. **Binary Analysis**: Identify type, architecture, and security mitigations",
            "2. **Vulnerability Discovery**: Fuzz inputs, analyze unsafe function calls, review boundary checks",
            "3. **Exploit Development**: Craft payload targeting the identified vulnerability",
            "4. **Bypass Protections**: Develop ROP chain, ret2libc, or other bypass for NX/ASLR/canary",
            "5. **Payload Testing**: Test exploit locally with debug environment",
            "6. **Remote Exploitation**: Adapt exploit for remote server with appropriate networking",
            "7. **Flag Capture**: Execute exploit against the remote server and capture the flag",
        ],
        "web": [
            "1. **Reconnaissance**: Enumerate pages, forms, API endpoints, and source code",
            "2. **Vulnerability Scanning**: Test for OWASP Top 10 (SQLi, XSS, LFI, RCE, etc.)",
            "3. **Exploitation**: Apply appropriate payload based on the identified vulnerability",
            "4. **Privilege Escalation**: If needed, escalate from initial access to flag location",
            "5. **Flag Extraction**: Access the flag file or database entry",
            "6. **Verification**: Confirm the flag format matches expected pattern",
        ],
        "crypto": [
            "1. **Algorithm Identification**: Determine the cryptographic scheme (RSA, AES, ECC, etc.)",
            "2. **Parameter Extraction**: Extract keys, moduli, ciphertexts, and other parameters",
            "3. **Vulnerability Analysis**: Check for common weaknesses (small d, low entropy, key reuse)",
            "4. **Mathematical Attack**: Apply appropriate cryptanalysis (factoring, discrete log, etc.)",
            "5. **Decryption/Forgery**: Decrypt ciphertext or forge valid signatures",
            "6. **Flag Recovery**: Extract the flag from the decrypted or forged data",
        ],
        "forensics": [
            "1. **Artifact Identification**: Determine file type and format (disk image, pcap, memory, etc.)",
            "2. **Metadata Extraction**: Extract timestamps, filesystem info, and creation dates",
            "3. **Content Analysis**: Search for strings, hidden files, deleted entries, or steganographic data",
            "4. **Data Recovery**: Use carving tools (foremost, binwalk) or file system analysis",
            "5. **Correlation**: Correlate findings across multiple artifacts if available",
            "6. **Flag Extraction**: Locate and extract the flag from the recovered data",
        ],
        "misc": [
            "1. **Challenge Type Identification**: Determine if it's encoding, steganography, trivia, etc.",
            "2. **Data Collection**: Gather all provided files, links, and textual clues",
            "3. **Pattern Recognition**: Look for patterns, encodings (base64, hex, ROT13), or hidden data",
            "4. **Tool Selection**: Choose appropriate tools based on challenge type",
            "5. **Solution Testing**: Verify the decoded/extracted flag against expected format",
            "6. **Flag Submission**: Format and submit the recovered flag",
        ],
    }
    solution_parts.extend(solution_steps.get(category, solution_steps["misc"]))

    solution_parts.append("")
    solution_parts.append("### Tools Commonly Used")
    tools_by_category = {
        "rev": [
            "Ghidra",
            "IDA Pro",
            "radare2",
            "Binary Ninja",
            "angr",
            "z3",
            "strings",
            "file",
            "checksec",
            "ltrace",
            "strace",
        ],
        "pwn": [
            "pwntools",
            "gdb/pwndbg",
            "ROPgadget",
            "one_gadget",
            "checksec",
            "qemu",
            "socat",
            "ncat",
        ],
        "web": [
            "Burp Suite",
            "curl",
            "sqlmap",
            "dirb/gobuster",
            "nikto",
            "browser devtools",
            "Postman",
        ],
        "crypto": [
            "SageMath",
            "PyCryptodome",
            "RsaCtfTool",
            "CyberChef",
            "factordb",
            "hashcat",
        ],
        "forensics": [
            "Autopsy",
            "Volatility",
            "Wireshark",
            "binwalk",
            "foremost",
            "exiftool",
            "strings",
        ],
        "misc": ["CyberChef", "Python", "stegsolve", "zsteg", "Binwalk", "xxd"],
    }
    tools = tools_by_category.get(category, tools_by_category["misc"])
    solution_parts.append(", ".join(f"`{t}`" for t in tools))

    if flag:
        solution_parts.append("")
        solution_parts.append(f"### Flag")
        solution_parts.append(f"`{flag}`")

    assistant_msg_2 = "\n".join(solution_parts)

    pair2 = {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg_2},
            {"role": "assistant", "content": assistant_msg_2},
        ],
        "mitre_ids": mitre_ids,
        "category": category,
        "year": str(year),
        "event": event,
        **ATTRIBUTION,
    }
    if mitre_tactic_id:
        pair2["mitre_tactic_id"] = mitre_tactic_id
    if tactic_name:
        pair2["tactic"] = tactic_name
    if kill_chain_phase:
        pair2["kill_chain_phase"] = kill_chain_phase

    pairs.append(pair2)

    # ------------------------------------------------------------------
    # Pair 3: Detection and defensive guidance (only if MITRE mapping exists)
    # ------------------------------------------------------------------
    if mitre_techniques:
        user_msg_3 = (
            f"How would you detect and defend against the techniques used in "
            f"the {friendly_category} CTF challenge '{challenge_name}' ({year} {event})?"
        )

        if technique_annotations:
            user_msg_3 += f"\n\nRelevant MITRE ATT&CK techniques: {', '.join(technique_annotations)}"

        defense_parts: list[str] = []
        defense_parts.append(
            f"## Detection & Defense: {challenge_name} ({friendly_category})"
        )
        defense_parts.append(f"**MITRE ATT&CK Techniques:** {techniques_str}")
        defense_parts.append("")

        defense_parts.append("### Detection Strategy")
        for tid in mitre_ids:
            tname = technique_names.get(tid, tid)
            defense_parts.append(f"- **{tid} ({tname})**:")

            # Category-specific detection guidance
            if category == "pwn" or tid in ("T1055", "T1068", "T1203", "T1574"):
                defense_parts.append(
                    "  Monitor for unexpected process injection, privilege escalation, "
                    "and exploitation attempts. Enable audit logging for process creation "
                    "and handle operations."
                )
            elif category == "web" or tid in ("T1190", "T1102", "T1071"):
                defense_parts.append(
                    "  Deploy WAF rules to detect exploitation of public-facing applications. "
                    "Monitor web server logs for anomalous request patterns. Implement rate "
                    "limiting and input validation."
                )
            elif category == "crypto" or tid in ("T1600", "T1552", "T1556"):
                defense_parts.append(
                    "  Enforce strong cryptographic standards, rotate keys regularly, "
                    "monitor for weak cipher usage. Implement certificate pinning and "
                    "validate all crypto implementations."
                )
            elif category == "forensics" or tid in ("T1005", "T1083", "T1040", "T1046"):
                defense_parts.append(
                    "  Implement comprehensive logging (process, network, file access). "
                    "Deploy EDR solutions for endpoint visibility. Centralize logs in "
                    "SIEM for correlation and alerting."
                )
            elif category == "rev" or tid in ("T1057", "T1082", "T1120"):
                defense_parts.append(
                    "  Use anti-debugging and anti-tamper techniques in production binaries. "
                    "Monitor for unexpected debugging or analysis activity on production systems."
                )
            else:
                defense_parts.append(
                    "  Implement defense-in-depth monitoring across network, endpoint, and "
                    "application layers. Correlate alerts across multiple data sources."
                )

        defense_parts.append("")
        defense_parts.append("### Defensive Recommendations")
        defense_parts.append(
            "1. **Principle of Least Privilege**: Restrict user and service permissions to "
            "the minimum required for operation"
        )
        defense_parts.append(
            "2. **Input Validation**: Sanitize and validate all user inputs, especially "
            "on public-facing services"
        )
        defense_parts.append(
            "3. **Security Monitoring**: Deploy comprehensive logging and SIEM correlation "
            "for the techniques identified above"
        )
        defense_parts.append(
            "4. **Patching**: Keep all software and dependencies up to date with security patches"
        )
        defense_parts.append(
            "5. **Network Segmentation**: Isolate critical systems and limit lateral movement paths"
        )

        defense_parts.append("")
        defense_parts.append("### Key Indicators")
        defense_parts.append("- Unexpected process creation or privilege changes")
        defense_parts.append(
            "- Anomalous network connections to unusual ports or hosts"
        )
        defense_parts.append("- File system modifications in sensitive directories")
        defense_parts.append(
            "- Unusual authentication attempts or credential usage patterns"
        )

        assistant_msg_3 = "\n".join(defense_parts)

        pair3 = {
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": user_msg_3},
                {"role": "assistant", "content": assistant_msg_3},
            ],
            "mitre_ids": mitre_ids,
            "category": category,
            "year": str(year),
            "event": event,
            **ATTRIBUTION,
        }
        if mitre_tactic_id:
            pair3["mitre_tactic_id"] = mitre_tactic_id
        if tactic_name:
            pair3["tactic"] = tactic_name
        if kill_chain_phase:
            pair3["kill_chain_phase"] = kill_chain_phase

        pairs.append(pair3)

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract NYU CTF Bench challenges into AttackLM JSONL training pairs.",
    )
    parser.add_argument(
        "--repo-path",
        type=str,
        default=None,
        help="Path to the NYU CTF Bench repo (default: data/NYU_CTF_Bench/). "
        "Will be cloned if not found.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for JSONL (default: data/datasets/buckets/sources/nyu-ctf-bench/ctf_challenges/).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "development", "both"],
        help="Dataset split to process (default: test).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit to N challenges (0 = all). Useful for testing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse challenges and print stats without writing output.",
    )
    args = parser.parse_args()

    # Resolve paths
    repo_path = Path(args.repo_path) if args.repo_path else DEFAULT_REPO_PATH
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_path = output_dir / "data.jsonl"

    # Ensure repo exists
    repo_path = ensure_repo(repo_path)

    print("AttackLM — Extract NYU CTF Bench Training Pairs")
    print(f"  Repo path: {repo_path}")
    print(f"  Output:   {output_path}")
    print(f"  Split:    {args.split}")
    print()

    # Determine which splits to process
    splits = ["test", "development"] if args.split == "both" else [args.split]

    all_pairs: list[dict[str, Any]] = []
    total_challenges = 0
    skipped = 0

    for split in splits:
        # Load dataset
        dataset = load_dataset(repo_path, split)
        if not dataset:
            print(f"  [{split}] No challenges found, skipping.")
            continue

        # Load MITRE mapping
        mitre_data = load_mitre_mapping(repo_path, split)
        mitre_mapping = mitre_data.get("mapping", {})
        technique_names = mitre_data.get("techniques", {})

        print(f"  [{split}] Found {len(dataset)} challenges")

        # Apply limit
        challenge_items = list(dataset.items())
        if args.limit > 0:
            challenge_items = challenge_items[: args.limit]

        for canonical_name, challenge_info in challenge_items:
            challenge_path = challenge_info.get("path", "")
            category = challenge_info.get("category", "misc")

            # Load challenge.json for richer data
            challenge_json = load_challenge_json(repo_path, challenge_path)

            # Get MITRE technique mapping
            mitre_techniques = mitre_mapping.get(canonical_name, [])

            if challenge_json is None and not challenge_path:
                skipped += 1
                continue

            pairs = build_pairs(
                canonical_name=canonical_name,
                challenge_info=challenge_info,
                challenge_json=challenge_json,
                mitre_techniques=mitre_techniques,
                technique_names=technique_names,
            )
            all_pairs.extend(pairs)
            total_challenges += 1

            # Verbose per-challenge output
            challenge_name = challenge_info.get("challenge", canonical_name)
            year = challenge_info.get("year", "?")
            pair_count = len(pairs)
            mitre_str = ", ".join(mitre_techniques) if mitre_techniques else "none"
            print(
                f"  [{split}] {canonical_name}: "
                f"{category}/{challenge_name} ({year}) "
                f"→ {pair_count} pairs, MITRE: [{mitre_str}]"
            )

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"  Challenges processed: {total_challenges}")
    print(f"  Challenges skipped:   {skipped}")
    print(f"  Total training pairs: {len(all_pairs)}")

    # Count by category
    category_counts: dict[str, int] = {}
    for pair in all_pairs:
        cat = pair.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print(f"\n  By category:")
    for cat, count in sorted(category_counts.items()):
        friendly = CATEGORY_INFO.get(cat, {}).get("friendly", cat)
        print(f"    {cat:12s} ({friendly:25s}): {count}")

    # Count by pair type
    type_counts = {"approach": 0, "methodology": 0, "defense": 0}
    for pair in all_pairs:
        msg = pair["messages"][1]["content"]
        if "detect and defend" in msg.lower():
            type_counts["defense"] += 1
        elif "detailed solution methodology" in msg.lower():
            type_counts["methodology"] += 1
        else:
            type_counts["approach"] += 1

    print(f"\n  By pair type:")
    for ptype, count in type_counts.items():
        print(f"    {ptype:15s}: {count}")

    # Unique MITRE IDs
    all_mitre: set[str] = set()
    for pair in all_pairs:
        all_mitre.update(pair.get("mitre_ids", []))
    print(f"\n  Unique MITRE ATT&CK IDs: {len(all_mitre)}")
    if all_mitre:
        mitre_list = sorted(all_mitre)
        print(
            f"    {', '.join(mitre_list[:20])}{'...' if len(mitre_list) > 20 else ''}"
        )

    # Unique events
    all_events: set[str] = set()
    for pair in all_pairs:
        event = pair.get("event", "")
        year = pair.get("year", "")
        if event:
            all_events.add(f"{year} {event}")
    print(f"  Events: {', '.join(sorted(all_events))}")

    if args.dry_run:
        print(f"\n{'=' * 60}")
        print("  DRY RUN — No files written")
        print(f"{'=' * 60}")

        # Show sample pair for verification
        if all_pairs:
            print(f"\n  Sample pair (first):\n")
            sample = all_pairs[0]
            sample_json = json.dumps(sample, indent=2, ensure_ascii=False)
            print(sample_json[:3000])
            if len(sample_json) > 3000:
                print("  ... (truncated)")

        return 0

    # --- Write output ---
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n  Output written: {len(all_pairs)} pairs → {output_path}")

    # --- Write per-source metadata ---
    source_md = output_dir / "SOURCE.md"
    source_md.write_text(
        f"# NYU CTF Bench\n\n"
        f"- **Source**: https://github.com/NYU-LLM-CTF/NYU_CTF_Bench\n"
        f"- **License**: GNU General Public License v2.0\n"
        f"- **License URI**: https://github.com/NYU-LLM-CTF/NYU_CTF_Bench/blob/main/LICENSE\n"
        f"- **Rights Contact**: NYU Secure Systems Lab\n"
        f"- **Challenges**: {total_challenges}\n"
        f"- **Training Pairs**: {len(all_pairs)}\n"
        f"- **Split**: {args.split}\n"
        f"- **Extracted**: {__import__('datetime').datetime.now().isoformat()}\n",
        encoding="utf-8",
    )

    # Copy LICENSE
    license_src = repo_path / "LICENSE"
    license_dst = output_dir / "LICENSE.md"
    if license_src.exists():
        import shutil

        shutil.copy2(license_src, license_dst)
        print(f"  License copied: {license_dst}")

    print(f"  Source metadata: {source_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
