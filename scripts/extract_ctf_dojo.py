#!/usr/bin/env python3
# CREDITS — DATA SOURCE ATTRIBUTION
# ----------------------------------
# This script transforms data from: Amazon CTF-Dojo
# Repository: https://github.com/amazon-science/CTF-Dojo
# License:    CC-BY-NC-4.0 (Creative Commons Attribution-NonCommercial 4.0)
# Copyright:  (c) Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# The output JSONL is a *transformation* of upstream CTF challenge metadata
# into OpenAI-style chat triples. See /ATTRIBUTION.md for full per-source
# attribution and re-distribution guidance.
#
# IMPORTANT: CC-BY-NC-4.0 restricts commercial use. This dataset may only
# be used for non-commercial research and educational purposes.
# ----------------------------------
"""Deterministic extraction of Amazon CTF-Dojo challenges into AttackLM JSONL training pairs.

Walks the CTF-Dojo repo (or a local clone) and parses ``ctf_archive.json`` plus
per-challenge ``DESCRIPTION.md`` / ``challenge.json`` files. For each challenge,
generates 1-2 OpenAI-style message triples:

  - **Pair type 1** — Challenge description (always, if description available)
  - **Pair type 2** — Solution / Docker setup hints (if challenge.json has compose or files)

Output: ``data/datasets/buckets/sources/ctf-dojo/ctf_challenges/data.jsonl``

Usage:
    uv run python scripts/extract_ctf_dojo.py
    uv run python scripts/extract_ctf_dojo.py --limit 5 --output-dir /tmp/test_ctfdojo
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
DEFAULT_REPO_PATH = BASE_DIR / "data" / "CTF-Dojo"
DEFAULT_OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "datasets"
    / "buckets"
    / "sources"
    / "ctf-dojo"
    / "ctf_challenges"
)

# ---------------------------------------------------------------------------
# System messages per category
# ---------------------------------------------------------------------------
SYSTEM_MSGS: dict[str, str] = {
    "pwn": (
        "You are a cybersecurity CTF specialist focused on binary exploitation. "
        "Analyze pwn challenges, identify vulnerabilities (buffer overflows, format strings, "
        "heap exploits, ROP chains), and provide exploitation strategies."
    ),
    "crypto": (
        "You are a cybersecurity CTF specialist focused on cryptography. "
        "Analyze crypto challenges, identify weaknesses in cryptographic implementations "
        "(RSA, AES, ECC, hashing), and provide solution strategies."
    ),
    "rev": (
        "You are a cybersecurity CTF specialist focused on reverse engineering. "
        "Analyze reverse engineering challenges, identify binary analysis techniques "
        "(disassembly, decompilation, anti-debugging bypass), and provide solution strategies."
    ),
    "web": (
        "You are a cybersecurity CTF specialist focused on web security. "
        "Analyze web challenges, identify web vulnerabilities (SQL injection, XSS, SSRF, "
        "authentication bypass), and provide solution strategies."
    ),
    "forensics": (
        "You are a cybersecurity CTF specialist focused on digital forensics. "
        "Analyze forensics challenges, identify evidence recovery techniques "
        "(disk analysis, memory forensics, network traffic analysis), and provide solution strategies."
    ),
    "misc": (
        "You are a cybersecurity CTF specialist focused on miscellaneous challenges. "
        "Analyze miscellaneous CTF challenges (steganography, OSINT, encoding), "
        "and provide solution strategies."
    ),
}

DEFAULT_SYSTEM_MSG = (
    "You are a cybersecurity CTF challenge solver. Analyze the challenge "
    "description, identify the vulnerability category, and provide a "
    "step-by-step solution approach."
)

# ---------------------------------------------------------------------------
# CTF category → MITRE ATT&CK mapping
# ---------------------------------------------------------------------------
CATEGORY_TACTIC_MAP: dict[str, tuple[str, str]] = {
    # category → (tactic_id, tactic_name)
    "pwn": ("TA0002", "Execution"),  # Binary exploitation → Execution
    "crypto": ("TA0006", "Credential Access"),  # Crypto attacks → Credential Access
    "rev": ("TA0007", "Discovery"),  # Reverse engineering → Discovery
    "web": ("TA0001", "Initial Access"),  # Web attacks → Initial Access
    "forensics": ("TA0007", "Discovery"),  # Forensics → Discovery
    "misc": ("TA0007", "Discovery"),  # Misc → Discovery
}

# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------
ATTRIBUTION = {
    "source": "amazon-science/CTF-Dojo",
    "source_uri": "https://github.com/amazon-science/CTF-Dojo",
    "license": "CC-BY-NC-4.0",
    "license_uri": "https://creativecommons.org/licenses/by-nc/4.0/",
    "rights_contact": "Amazon Science",
    "attribution_text": (
        "Copyright (c) Amazon.com, Inc. or its affiliates. Licensed under "
        "CC-BY-NC-4.0. See https://creativecommons.org/licenses/by-nc/4.0/."
    ),
}


# ---------------------------------------------------------------------------
# Clone / download repo
# ---------------------------------------------------------------------------
def ensure_repo(repo_path: Path) -> Path:
    """Ensure the CTF-Dojo repo is available locally.

    If ``repo_path`` exists, use it. Otherwise, clone from GitHub.
    """
    if repo_path.exists():
        # Check it has the expected structure
        archive_file = repo_path / "ctf_archive.json"
        if archive_file.exists():
            print(f"  Using existing repo at: {repo_path}")
            return repo_path
        # Maybe it's the parent — look for ctf-archive subdir
        ctf_archive_subdir = repo_path / "ctf-archive"
        if ctf_archive_subdir.exists():
            # The repo itself is the parent
            return repo_path

    print(f"  Cloning CTF-Dojo repo to {repo_path}...")
    repo_url = "https://github.com/amazon-science/CTF-Dojo.git"
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_path)],
            check=True,
            capture_output=True,
        )
        print(f"  Cloned successfully.")
    except subprocess.CalledProcessError as e:
        print(
            f"  ERROR: Failed to clone: {e.stderr.decode() if e.stderr else e}",
            file=sys.stderr,
        )
        print(
            f"  Please clone manually: git clone {repo_url} {repo_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    return repo_path


# ---------------------------------------------------------------------------
# Load ctf_archive.json
# ---------------------------------------------------------------------------
def load_archive(repo_path: Path) -> dict[str, dict[str, Any]]:
    """Load ctf_archive.json from the repo root.

    Returns a dict keyed by challenge ID, each with
    benchmark, event, challenge, path, category fields.
    """
    archive_path = repo_path / "ctf_archive.json"
    if not archive_path.exists():
        print(f"  ERROR: ctf_archive.json not found at {archive_path}", file=sys.stderr)
        sys.exit(1)

    with open(archive_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# ---------------------------------------------------------------------------
# Read challenge description
# ---------------------------------------------------------------------------
def read_description(challenge_dir: Path) -> str:
    """Read DESCRIPTION.md from a challenge directory."""
    desc_path = challenge_dir / "DESCRIPTION.md"
    if not desc_path.exists():
        return ""
    try:
        return desc_path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return ""


def read_challenge_json(challenge_dir: Path) -> dict[str, Any] | None:
    """Read challenge.json from a challenge directory if it exists."""
    cj_path = challenge_dir / "challenge.json"
    if not cj_path.exists():
        return None
    try:
        with open(cj_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def read_rehost_md(challenge_dir: Path) -> str:
    """Read REHOST.md from a challenge directory if it exists."""
    rehost_path = challenge_dir / "REHOST.md"
    if not rehost_path.exists():
        return ""
    try:
        return rehost_path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Build training pairs
# ---------------------------------------------------------------------------
def build_challenge_description_pair(
    *,
    challenge_id: str,
    event: str,
    challenge_name: str,
    category: str,
    description: str,
    system_msg: str,
) -> dict[str, Any]:
    """Pair type 1: Challenge description with analysis prompt."""
    # Determine MITRE tactic from category
    tactic_id, tactic_name = CATEGORY_TACTIC_MAP.get(category, ("TA0007", "Discovery"))

    user_content = (
        f"Solve the following CTF challenge:\n\n"
        f"**Event**: {event}\n"
        f"**Challenge**: {challenge_name}\n"
        f"**Category**: {category}\n\n"
        f"{description}"
    )

    # Build structured response
    assistant_lines = [
        f"## {challenge_name} ({event} — {category})\n",
        f"**MITRE ATT&CK Tactic**: {tactic_id} ({tactic_name})\n",
        f"**Category**: {category}\n",
    ]

    # Add description context
    if description:
        # Truncate very long descriptions
        desc_preview = description[:2000]
        if len(description) > 2000:
            desc_preview += "\n\n[... description truncated ...]"
        assistant_lines.append(f"\n### Challenge Description\n{desc_preview}")

    # Add analysis guidance based on category
    category_hints = {
        "pwn": "Examine for buffer overflows, format string vulnerabilities, use-after-free, "
        "double-free, heap exploitation techniques. Check protections (NX, ASLR, PIE, "
        "canary) and plan ROP/shellcode strategy.",
        "crypto": "Analyze the cryptographic construction for weaknesses: small key sizes, "
        "padding oracle attacks, mathematical properties of the scheme, "
        "implementation flaws, side-channel information.",
        "rev": "Use static analysis tools (Ghidra, IDA, radare2) to understand program logic. "
        "Look for string comparisons, XOR operations, custom encodings, "
        "anti-debugging tricks, and flag-checking routines.",
        "web": "Test for common web vulnerabilities: SQL injection, XSS, SSRF, path traversal, "
        "authentication bypass, session manipulation, file inclusion.",
        "forensics": "Examine file headers, metadata, embedded data, analyze network captures, "
        "memory dumps, disk images. Use tools like binwalk, strings, file, "
        "Volatility, Wireshark.",
        "misc": "Look for encoding schemes, steganography, OSINT clues, "
        "unusual file formats, and creative puzzle-solving approaches.",
    }
    hint = category_hints.get(category, category_hints["misc"])
    assistant_lines.append(f"\n### Approach\n{hint}")

    assistant_content = "\n".join(assistant_lines)

    pair: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "source": "amazon-science/CTF-Dojo",
        "mitre_ids": [tactic_id],
        "mitre_tactic_id": tactic_id,
        "tactic": tactic_name,
        "kill_chain_phase": tactic_name,
        "ctf_event": event,
        "ctf_category": category,
        "challenge_id": challenge_id,
        "license": "CC-BY-NC-4.0",
        "license_uri": "https://creativecommons.org/licenses/by-nc/4.0/",
    }
    return pair


def build_solution_pair(
    *,
    challenge_id: str,
    event: str,
    challenge_name: str,
    category: str,
    challenge_json: dict[str, Any],
    rehost_md: str,
    system_msg: str,
) -> dict[str, Any] | None:
    """Pair type 2: Solution / Docker setup hints (if challenge.json has compose or files)."""
    tactic_id, tactic_name = CATEGORY_TACTIC_MAP.get(category, ("TA0007", "Discovery"))

    # Check if there's useful information for a solution pair
    files = challenge_json.get("files", [])
    compose = challenge_json.get("compose", False)
    image = challenge_json.get("image", "")
    flag_sha256 = challenge_json.get("sha256_flag", "") or challenge_json.get(
        "flag_sha256", ""
    )

    # Only generate if there's meaningful setup info
    if not files and not compose and not image and not rehost_md:
        return None

    user_content = (
        f"How do I set up and solve the CTF challenge '{challenge_name}' "
        f"from {event} ({category} category)?"
    )

    assistant_lines = [
        f"## {challenge_name} — Setup & Solution Hints\n",
        f"**Event**: {event}\n",
        f"**Category**: {category}\n",
    ]

    # Docker / setup info
    if compose:
        assistant_lines.append(
            "\n### Docker Setup\n"
            "This challenge runs as a Docker container. Use:\n"
            "```bash\n"
            "docker-compose up\n"
            "```\n"
            "to start the challenge environment."
        )
    elif image:
        assistant_lines.append(
            f"\n### Docker Setup\n"
            f"Challenge image: `{image}`\n\n"
            f"Pull and run with:\n"
            f"```bash\n"
            f"docker pull {image}\n"
            f"docker run -p <local_port>:<challenge_port> {image}\n"
            f"```"
        )

    # Files info
    if files:
        file_list = "\n".join(f"  - `{f}`" for f in files[:20])
        if len(files) > 20:
            file_list += f"\n  - ... and {len(files) - 20} more files"
        assistant_lines.append(f"\n### Challenge Files\n{file_list}")

    # SHA256 flag hash
    if flag_sha256:
        assistant_lines.append(
            f"\n### Flag Verification\n"
            f"Flag SHA256 hash: `{flag_sha256}`\n\n"
            f"Verify your solution with:\n"
            f"```bash\n"
            f"echo -n 'your_flag_here' | sha256sum\n"
            f"```"
        )

    # Rehost info
    if rehost_md:
        # Truncate long rehost instructions
        rehost_preview = rehost_md[:1500]
        if len(rehost_md) > 1500:
            rehost_preview += "\n\n[... rehost instructions truncated ...]"
        assistant_lines.append(f"\n### Rehost Instructions\n{rehost_preview}")

    # Category-specific solution hints
    category_solution_hints = {
        "pwn": "\n### Exploitation Strategy\n"
        "1. Check binary protections: `checksec <binary>`\n"
        "2. Identify vulnerability: buffer overflow, format string, UAF, etc.\n"
        "3. Develop exploit: ROP chain, shellcode, heap feng shui\n"
        "4. Test locally with Docker, then submit remotely",
        "crypto": "\n### Solution Strategy\n"
        "1. Identify the cryptographic scheme\n"
        "2. Look for implementation flaws or mathematical weaknesses\n"
        "3. Write a solver script exploiting the vulnerability\n"
        "4. Recover the flag from decrypted/derived values",
        "rev": "\n### Reverse Engineering Strategy\n"
        "1. Identify the binary type and architecture: `file <binary>`\n"
        "2. Use disassembler/decompiler (Ghidra, IDA, radare2)\n"
        "3. Trace the flag-checking logic\n"
        "4. Patch or extract the flag directly",
        "web": "\n### Web Challenge Strategy\n"
        "1. Map the application: endpoints, forms, APIs\n"
        "2. Test for injection, auth bypass, path traversal\n"
        "3. Exploit the vulnerability to access the flag\n"
        "4. Submit the flag",
        "forensics": "\n### Forensics Strategy\n"
        "1. Identify file type: `file <challenge_file>`\n"
        "2. Extract hidden data: binwalk, strings, exiftool\n"
        "3. Analyze with appropriate tools (Wireshark, Volatility)\n"
        "4. Reconstruct the flag from evidence",
        "misc": "\n### Solution Strategy\n"
        "1. Identify the puzzle type (encoding, steganography, OSINT)\n"
        "2. Apply appropriate decoding/analysis tools\n"
        "3. Follow the chain of clues to find the flag",
    }
    hint = category_solution_hints.get(category, category_solution_hints["misc"])
    assistant_lines.append(hint)

    pair: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "\n".join(assistant_lines)},
        ],
        "source": "amazon-science/CTF-Dojo",
        "mitre_ids": [tactic_id],
        "mitre_tactic_id": tactic_id,
        "tactic": tactic_name,
        "kill_chain_phase": tactic_name,
        "ctf_event": event,
        "ctf_category": category,
        "challenge_id": challenge_id,
        "license": "CC-BY-NC-4.0",
        "license_uri": "https://creativecommons.org/licenses/by-nc/4.0/",
    }
    return pair


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Amazon CTF-Dojo challenges into AttackLM JSONL training pairs.",
    )
    parser.add_argument(
        "--repo-path",
        type=str,
        default=None,
        help="Path to local CTF-Dojo repo clone (default: data/CTF-Dojo)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit to N challenges (0 = all). Useful for testing.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for JSONL.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print stats without writing output.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter by category (pwn, crypto, rev, web, forensics, misc).",
    )
    args = parser.parse_args()

    # Resolve paths
    repo_path = Path(args.repo_path) if args.repo_path else DEFAULT_REPO_PATH
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_path = output_dir / "data.jsonl"

    print("AttackLM — Extract CTF-Dojo Training Pairs")
    print(f"  Repo path:    {repo_path}")
    print(f"  Output:       {output_path}")
    print()

    # Ensure repo is available
    repo_path = ensure_repo(repo_path)

    # Load archive
    archive = load_archive(repo_path)
    print(f"  Loaded {len(archive)} challenges from ctf_archive.json")

    # Apply category filter
    if args.category:
        category_filter = args.category.lower()
        archive = {
            k: v
            for k, v in archive.items()
            if v.get("category", "").lower() == category_filter
            or category_filter in v.get("category", "").lower()
        }
        print(f"  Filtered to {len(archive)} challenges in category '{args.category}'")

    # Apply limit
    challenge_items = list(archive.items())
    if args.limit > 0:
        challenge_items = challenge_items[: args.limit]
        print(f"  Limited to {args.limit} challenges")

    # Process challenges
    all_pairs: list[dict[str, Any]] = []
    skipped_no_desc: int = 0
    skipped_error: int = 0
    category_counts: dict[str, int] = {}
    event_counts: dict[str, int] = {}

    print(f"\n  Processing {len(challenge_items)} challenges...")

    for challenge_id, info in challenge_items:
        event = info.get("event", "unknown")
        challenge_name = info.get("challenge", "unknown")
        category = info.get("category", "unknown").lower()
        challenge_path = info.get("path", "")

        # Find the challenge directory
        # The path in ctf_archive.json is like "ctf-archive/0ctf2017/babyheap"
        challenge_dir = repo_path / challenge_path
        if not challenge_dir.exists():
            # Try with ctf-archive subdirectory
            challenge_dir = (
                repo_path
                / "ctf-archive"
                / challenge_path.replace("ctf-archive/", "", 1)
            )
        if not challenge_dir.exists():
            # Try directly under repo_path
            parts = challenge_path.split("/")
            if len(parts) >= 3:
                challenge_dir = repo_path / parts[1] / parts[2]
            if not challenge_dir.exists() and len(parts) >= 2:
                challenge_dir = repo_path / parts[-2] / parts[-1]

        # Read description
        description = ""
        if challenge_dir.exists():
            description = read_description(challenge_dir)

        # If no description file, construct from metadata
        if not description:
            description = (
                f"CTF challenge '{challenge_name}' from {event}. "
                f"Category: {category}. "
                f"Challenge ID: {challenge_id}."
            )
            # Don't skip — we can still generate a pair from metadata

        # Select system message based on category
        system_msg = SYSTEM_MSGS.get(category, DEFAULT_SYSTEM_MSG)

        # Pair type 1: Challenge description (always)
        pair1 = build_challenge_description_pair(
            challenge_id=challenge_id,
            event=event,
            challenge_name=challenge_name,
            category=category,
            description=description,
            system_msg=system_msg,
        )
        all_pairs.append(pair1)

        # Pair type 2: Solution / setup hints (if challenge.json or REHOST.md exists)
        challenge_json_data = None
        rehost_md = ""
        if challenge_dir.exists():
            challenge_json_data = read_challenge_json(challenge_dir)
            rehost_md = read_rehost_md(challenge_dir)

        if challenge_json_data or rehost_md:
            pair2 = build_solution_pair(
                challenge_id=challenge_id,
                event=event,
                challenge_name=challenge_name,
                category=category,
                challenge_json=challenge_json_data or {},
                rehost_md=rehost_md,
                system_msg=system_msg,
            )
            if pair2 is not None:
                all_pairs.append(pair2)

        # Track stats
        category_counts[category] = category_counts.get(category, 0) + 1
        event_counts[event] = event_counts.get(event, 0) + 1

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"  Challenges processed: {len(challenge_items)}")
    print(f"  Skipped (no description): {skipped_no_desc}")
    print(f"  Skipped (errors): {skipped_error}")
    print(f"  Total training pairs: {len(all_pairs)}")

    # Pair type breakdown
    desc_pairs = sum(
        1 for p in all_pairs if "Challenge Description" in p["messages"][2]["content"]
    )
    setup_pairs = len(all_pairs) - desc_pairs
    print(f"\n  By pair type:")
    print(f"    challenge_description: {desc_pairs}")
    print(f"    setup_solution_hints:  {setup_pairs}")

    # Category breakdown
    print(f"\n  By category:")
    for cat in sorted(category_counts.keys()):
        print(f"    {cat:15s}: {category_counts[cat]}")

    # Event breakdown (top 10)
    print(f"\n  Top events:")
    for event in sorted(
        event_counts.keys(), key=lambda e: event_counts[e], reverse=True
    )[:10]:
        print(f"    {event:30s}: {event_counts[event]}")

    # Unique MITRE IDs
    all_mitre: set[str] = set()
    for pair in all_pairs:
        all_mitre.update(pair.get("mitre_ids", []))
    print(f"\n  Unique MITRE ATT&CK tactic IDs: {len(all_mitre)}")
    print(f"    {', '.join(sorted(all_mitre))}")

    if args.dry_run:
        print(f"\n{'=' * 60}")
        print("  DRY RUN — No files written")

        # Show sample pair
        if all_pairs:
            print(f"\n  Sample pair (first):\n")
            sample = all_pairs[0]
            sample_json = json.dumps(sample, indent=2, ensure_ascii=False)
            print(sample_json[:3000])
            if len(sample_json) > 3000:
                print("  ... (truncated)")
        return

    # --- Write output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n  Output written: {len(all_pairs)} pairs → {output_path}")

    # --- Write SOURCE.md ---
    # For the default path, SOURCE.md goes to sources/ctf-dojo/SOURCE.md
    # For custom --output-dir, SOURCE.md goes inside the custom dir
    if args.output_dir:
        # Custom output: put SOURCE.md inside the custom output dir
        source_md_path = output_dir / "SOURCE.md"
        license_md_path = output_dir / "LICENSE.md"
    else:
        # Default: put SOURCE.md in the source root (.../sources/ctf-dojo/)
        source_dir = DEFAULT_OUTPUT_DIR.parent
        source_md_path = source_dir / "SOURCE.md"
        license_md_path = source_dir / "LICENSE.md"
    if not args.dry_run:
        source_md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(source_md_path, "w", encoding="utf-8") as f:
            f.write(
                f"# Source: Amazon CTF-Dojo\n"
                f"\n"
                f"- **Repository**: https://github.com/amazon-science/CTF-Dojo\n"
                f"- **License**: CC-BY-NC-4.0 (Creative Commons Attribution-NonCommercial 4.0)\n"
                f"- **License URI**: https://creativecommons.org/licenses/by-nc/4.0/\n"
                f"- **Rights Contact**: Amazon Science\n"
                f"- **Extraction Date**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d')}\n"
                f"- **Extractor Script**: scripts/extract_ctf_dojo.py\n"
                f"- **Records Extracted**: ~{len(all_pairs)}\n"
                f"- **Description**: CTF-Dojo provides 658 executable CTF-style challenges with "
                f"Docker-based reproducibility and verifiable feedback. Covers pwn, crypto, "
                f"rev, web, forensics, and misc categories from 20+ CTF events.\n"
                f"\n"
                f"**IMPORTANT**: CC-BY-NC-4.0 restricts commercial use. "
                f"This dataset may only be used for non-commercial research and "
                f"educational purposes.\n"
            )
        print(f"  Wrote SOURCE.md → {source_md_path}")

    # --- Write LICENSE.md ---
    if not args.dry_run and not license_md_path.exists():
        with open(license_md_path, "w", encoding="utf-8") as f:
            f.write(
                "# CC-BY-NC-4.0 License\n"
                "\n"
                "This data is derived from Amazon CTF-Dojo.\n"
                "\n"
                "Original license: Creative Commons Attribution-NonCommercial 4.0 International\n"
                "\n"
                "Full text: https://creativecommons.org/licenses/by-nc/4.0/\n"
                "\n"
                "You are free to:\n"
                "- Share — copy and redistribute the material in any medium or format\n"
                "- Adapt — remix, transform, and build upon the material\n"
                "\n"
                "Under the following terms:\n"
                "- Attribution — You must give appropriate credit\n"
                "- NonCommercial — You may not use the material for commercial purposes\n"
            )
        print(f"  Wrote LICENSE.md → {license_md_path}")


if __name__ == "__main__":
    main()
