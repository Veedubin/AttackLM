#!/usr/bin/env python3
# CREDITS — DATA SOURCE ATTRIBUTION
# ----------------------------------
# This script transforms data from: WhitzardAgent/CyberSecurity-1M
# HuggingFace: https://huggingface.co/datasets/WhitzardAgent/CyberSecurity-1M
# License:    Apache License 2.0 (academic use restriction on HF)
#
# The output JSONL is a *transformation* of upstream CyberSecurity-1M records
# into OpenAI-style chat triples. See /ATTRIBUTION.md for full per-source
# attribution and re-distribution guidance.
# ----------------------------------
"""Extract CyberSecurity-1M dataset from HuggingFace into AttackLM JSONL training pairs.

Downloads the dataset via HuggingFace ``datasets`` library (streaming mode
for memory efficiency), processes each category, and converts records to
OpenAI-style message triples with proper source attribution and MITRE ATT&CK
mapping where available.

Categories (16 total, 1.19M records):
    vulnerability (878K), cn_sec (58K), framework (58K), reference (44K),
    ctf (43K), tool (19K), vuln_research (19K), incident_response (18K),
    bug_bounty (17K), threat_intel (14K), offsec (10K), conference (5K),
    books (3K), news (2K), ai_security (2K), ics_ot (1K)

Output per category:
    ``data/datasets/buckets/sources/cybersecurity-1m/<category>/data.jsonl``

Usage:
    python scripts/extract_cybersecurity_1m.py
    python scripts/extract_cybersecurity_1m.py --limit 100 --categories vulnerability,ctf
    python scripts/extract_cybersecurity_1m.py --min-length 500 --output-dir /tmp/cs1m_out
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
DEFAULT_OUTPUT_DIR = (
    BASE_DIR / "data" / "datasets" / "buckets" / "sources" / "cybersecurity-1m"
)

# ---------------------------------------------------------------------------
# HF dataset config
# ---------------------------------------------------------------------------
HF_DATASET_ID = "WhitzardAgent/CyberSecurity-1M"
HF_SUBSET = "merged"

# ---------------------------------------------------------------------------
# System messages per category
# ---------------------------------------------------------------------------
SYSTEM_MESSAGES: dict[str, str] = {
    "vulnerability": (
        "You are a Vulnerability Analysis specialist. Analyze CVE records, "
        "assess severity, describe exploitation vectors, and provide "
        "remediation guidance mapped to MITRE ATT&CK."
    ),
    "ctf": (
        "You are a CTF and Security Training specialist. Walk through "
        "challenge solutions step-by-step, explaining techniques, tools, "
        "and security concepts used."
    ),
    "framework": (
        "You are a Security Framework specialist. Explain attack frameworks, "
        "detection rulesets, and defensive methodologies mapped to MITRE ATT&CK."
    ),
    "tool": (
        "You are a Security Tool specialist. Describe security tools, their "
        "usage, configuration, and integration into defensive workflows."
    ),
    "threat_intel": (
        "You are a Threat Intelligence analyst. Analyze APT reports, malware "
        "campaigns, IOCs, and TTPs mapped to MITRE ATT&CK."
    ),
    "incident_response": (
        "You are an Incident Response specialist. Provide structured IR "
        "procedures covering detection, containment, eradication, and recovery."
    ),
    "reference": (
        "You are a Security Reference specialist. Provide clear explanations "
        "of security concepts, cheat sheets, and curated reference materials."
    ),
    "vuln_research": (
        "You are a Vulnerability Research specialist. Analyze vulnerability "
        "details, exploitation techniques, and patch analysis."
    ),
    "bug_bounty": (
        "You are a Bug Bounty specialist. Describe vulnerability discovery "
        "methodologies, writeups, and responsible disclosure practices."
    ),
    "offsec": (
        "You are an Offensive Security specialist. Explain penetration testing "
        "methodologies and red team techniques for authorized security assessment."
    ),
    "conference": (
        "You are a Security Conference Knowledge specialist. Summarize and "
        "explain key insights from security conference talks and presentations."
    ),
    "books": (
        "You are a Security Literature specialist. Summarize and explain key "
        "concepts from cybersecurity books and educational materials."
    ),
    "news": (
        "You are a Security News analyst. Analyze cybersecurity news, breaches, "
        "and emerging threats with context and impact assessment."
    ),
    "ai_security": (
        "You are an AI Security specialist. Analyze AI/ML security concerns, "
        "adversarial attacks, prompt injection, and model safety."
    ),
    "ics_ot": (
        "You are an ICS/OT Security specialist. Analyze industrial control "
        "system vulnerabilities, SCADA security, and operational technology."
    ),
    "cn_sec": (
        "You are a Chinese Security Community specialist. Translate and analyze "
        "Chinese-language security research, vulnerabilities, and techniques."
    ),
}

DEFAULT_SYSTEM_MSG = (
    "You are a Cybersecurity specialist. Provide accurate, detailed "
    "analysis and guidance on security topics."
)

# ---------------------------------------------------------------------------
# Category-to-subfolder mapping (AttackLM bucket structure)
# ---------------------------------------------------------------------------
CATEGORY_TO_SUBFOLDER: dict[str, tuple[str, str]] = {
    # (bucket_type, tactic_or_topic)
    "vulnerability": ("defensive", "vulnerability_analysis"),
    "ctf": ("offensive", "ctf_training"),
    "framework": ("defensive", "detection_engineering"),
    "tool": ("defensive", "security_tools"),
    "threat_intel": ("defensive", "threat_intelligence"),
    "incident_response": ("defensive", "incident_response"),
    "reference": ("defensive", "security_reference"),
    "vuln_research": ("defensive", "vulnerability_analysis"),
    "bug_bounty": ("offensive", "bug_bounty"),
    "offsec": ("offensive", "penetration_testing"),
    "conference": ("defensive", "security_reference"),
    "books": ("defensive", "security_reference"),
    "news": ("defensive", "threat_intelligence"),
    "ai_security": ("defensive", "ai_security"),
    "ics_ot": ("defensive", "ics_security"),
    "cn_sec": ("defensive", "security_reference"),
}

# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------
ATTRIBUTION = {
    "source": "cybersecurity-1m",
    "source_uri": "https://huggingface.co/datasets/WhitzardAgent/CyberSecurity-1M",
    "license": "Apache-2.0",
    "license_uri": "https://huggingface.co/datasets/WhitzardAgent/CyberSecurity-1M",
    "rights_contact": "WhitzardAgent",
    "attribution_text": (
        "Apache License 2.0 — CyberSecurity-1M by WhitzardAgent. "
        "Academic use only. See https://huggingface.co/datasets/WhitzardAgent/CyberSecurity-1M."
    ),
}

# ---------------------------------------------------------------------------
# MITRE technique ID extraction
# ---------------------------------------------------------------------------
_MITRE_RE = re.compile(r"\b(T\d{4}(?:\.\d{3})?)\b", re.IGNORECASE)
_CVE_RE = re.compile(r"\b(CVE-\d{4}-\d{4,})\b", re.IGNORECASE)


def extract_mitre_ids(text: str) -> list[str]:
    """Extract MITRE ATT&CK technique IDs from text."""
    return sorted({m.group(1).upper() for m in _MITRE_RE.finditer(text)})


def extract_cve_ids(text: str) -> list[str]:
    """Extract CVE IDs from text."""
    return sorted({m.group(1).upper() for m in _CVE_RE.finditer(text)})


# ---------------------------------------------------------------------------
# Record filtering
# ---------------------------------------------------------------------------
def _content_length(record: dict[str, Any]) -> int:
    """Return total character length of content fields."""
    total = 0
    for field in ("description", "markdown", "exploit_code"):
        val = record.get(field)
        if isinstance(val, str):
            total += len(val)
    return total


# ---------------------------------------------------------------------------
# Pair builders — one per category type
# ---------------------------------------------------------------------------
def _build_vulnerability_pair(record: dict[str, Any]) -> dict[str, Any] | None:
    """Build a training pair from a vulnerability record."""
    title = record.get("title", "") or "Vulnerability Advisory"
    description = record.get("description", "") or ""
    markdown_content = record.get("markdown", "") or ""
    cve = record.get("cve", "") or ""
    url = record.get("url", "") or ""
    tags = record.get("tags", []) or []

    # Use markdown if available, otherwise description
    content = (
        markdown_content if len(markdown_content) > len(description) else description
    )
    if not content or len(content.strip()) < 50:
        return None

    # Extract MITRE IDs and CVE IDs
    all_text = f"{title} {content}"
    mitre_ids = extract_mitre_ids(all_text)
    cve_ids = [cve.upper()] if cve else extract_cve_ids(all_text)

    # Build user prompt
    cve_str = cve_ids[0] if cve_ids else "this vulnerability"
    tag_str = ", ".join(str(t) for t in tags[:5]) if tags else "general"
    technique_str = (
        ", ".join(mitre_ids) if mitre_ids else "appropriate MITRE ATT&CK technique"
    )

    user_msg = (
        f"Analyze {cve_str}. Describe the vulnerability, its impact, "
        f"exploitation vectors, and remediation steps. Map to {technique_str}."
    )

    # Build assistant response (truncated for training quality)
    max_content = 4000
    if len(content) > max_content:
        content = content[:max_content] + "\n\n[... truncated for training ...]"

    assistant_parts = [f"## {title}\n"]
    if cve_ids:
        assistant_parts.append(f"**CVE:** {', '.join(cve_ids)}\n")
    if mitre_ids:
        assistant_parts.append(f"**MITRE ATT&CK:** {', '.join(mitre_ids)}\n")
    if tags:
        assistant_parts.append(f"**Tags:** {', '.join(str(t) for t in tags[:10])}\n")
    if url:
        assistant_parts.append(f"**Source:** {url}\n")
    assistant_parts.append(f"\n{content}")

    assistant_msg = "\n".join(assistant_parts)

    pair: dict[str, Any] = {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_MESSAGES.get("vulnerability", DEFAULT_SYSTEM_MSG),
            },
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "mitre_ids": mitre_ids,
        **ATTRIBUTION,
    }

    # Add tactic info
    if mitre_ids:
        tactic_id = get_tactic_for_technique(mitre_ids[0])
        if tactic_id:
            pair["mitre_tactic_id"] = tactic_id
            tactic_name = get_tactic_name(tactic_id)
            if tactic_name:
                pair["tactic"] = tactic_name
                pair["kill_chain_phase"] = tactic_name

    return pair


def _build_ctf_pair(record: dict[str, Any]) -> dict[str, Any] | None:
    """Build a training pair from a CTF record."""
    title = record.get("title", "") or "CTF Challenge"
    description = record.get("description", "") or ""
    markdown_content = record.get("markdown", "") or ""
    tags = record.get("tags", []) or []
    url = record.get("url", "") or ""

    content = (
        markdown_content if len(markdown_content) > len(description) else description
    )
    if not content or len(content.strip()) < 50:
        return None

    all_text = f"{title} {content}"
    mitre_ids = extract_mitre_ids(all_text)
    technique_str = (
        ", ".join(mitre_ids) if mitre_ids else "security concepts and techniques"
    )

    user_msg = (
        f'Walk through the solution for the CTF challenge: "{title}". '
        f"Explain the techniques and tools used, mapping to {technique_str}."
    )

    max_content = 4000
    if len(content) > max_content:
        content = content[:max_content] + "\n\n[... truncated for training ...]"

    assistant_parts = [f"## CTF: {title}\n"]
    if mitre_ids:
        assistant_parts.append(f"**MITRE ATT&CK:** {', '.join(mitre_ids)}\n")
    if tags:
        assistant_parts.append(f"**Tags:** {', '.join(str(t) for t in tags[:10])}\n")
    if url:
        assistant_parts.append(f"**Source:** {url}\n")
    assistant_parts.append(f"\n{content}")

    pair: dict[str, Any] = {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_MESSAGES.get("ctf", DEFAULT_SYSTEM_MSG),
            },
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": "\n".join(assistant_parts)},
        ],
        "mitre_ids": mitre_ids,
        **ATTRIBUTION,
    }

    if mitre_ids:
        tactic_id = get_tactic_for_technique(mitre_ids[0])
        if tactic_id:
            pair["mitre_tactic_id"] = tactic_id
            tactic_name = get_tactic_name(tactic_id)
            if tactic_name:
                pair["tactic"] = tactic_name
                pair["kill_chain_phase"] = tactic_name

    return pair


def _build_generic_pair(record: dict[str, Any], category: str) -> dict[str, Any] | None:
    """Build a training pair from any category record."""
    title = record.get("title", "") or f"{category.replace('_', ' ').title()} Record"
    description = record.get("description", "") or ""
    markdown_content = record.get("markdown", "") or ""
    tags = record.get("tags", []) or []
    url = record.get("url", "") or ""

    content = (
        markdown_content if len(markdown_content) > len(description) else description
    )
    if not content or len(content.strip()) < 50:
        return None

    all_text = f"{title} {content}"
    mitre_ids = extract_mitre_ids(all_text)
    cve_ids = extract_cve_ids(all_text)
    technique_str = (
        ", ".join(mitre_ids) if mitre_ids else "relevant security techniques"
    )

    category_label = category.replace("_", " ").title()
    system_msg = SYSTEM_MESSAGES.get(category, DEFAULT_SYSTEM_MSG)

    user_msg = (
        f'Explain the following {category_label} topic: "{title}". '
        f"Provide detailed analysis mapping to {technique_str}."
    )

    max_content = 4000
    if len(content) > max_content:
        content = content[:max_content] + "\n\n[... truncated for training ...]"

    assistant_parts = [f"## {title}\n"]
    if mitre_ids:
        assistant_parts.append(f"**MITRE ATT&CK:** {', '.join(mitre_ids)}\n")
    if cve_ids:
        assistant_parts.append(f"**CVE:** {', '.join(cve_ids[:5])}\n")
    if tags:
        assistant_parts.append(f"**Tags:** {', '.join(str(t) for t in tags[:10])}\n")
    if url:
        assistant_parts.append(f"**Source:** {url}\n")
    assistant_parts.append(f"\n{content}")

    pair: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": "\n".join(assistant_parts)},
        ],
        "mitre_ids": mitre_ids,
        **ATTRIBUTION,
    }

    if mitre_ids:
        tactic_id = get_tactic_for_technique(mitre_ids[0])
        if tactic_id:
            pair["mitre_tactic_id"] = tactic_id
            tactic_name = get_tactic_name(tactic_id)
            if tactic_name:
                pair["tactic"] = tactic_name
                pair["kill_chain_phase"] = tactic_name

    return pair


# ---------------------------------------------------------------------------
# Pair builder dispatcher
# ---------------------------------------------------------------------------
def build_pair(record: dict[str, Any], category: str) -> dict[str, Any] | None:
    """Dispatch to the appropriate pair builder based on category."""
    if category == "vulnerability":
        return _build_vulnerability_pair(record)
    elif category == "ctf":
        return _build_ctf_pair(record)
    else:
        return _build_generic_pair(record, category)


# ---------------------------------------------------------------------------
# Dataset loading — local JSONL files
# ---------------------------------------------------------------------------
def load_category_from_local(
    category: str,
    local_dir: Path,
    limit: int = 0,
    min_length: int = 200,
) -> list[dict[str, Any]]:
    """Load and process a category from pre-downloaded JSONL files.

    Reads ``<local_dir>/merged/<category>.jsonl`` line-by-line for
    memory efficiency with the 1.19M record dataset.
    """
    jsonl_path = local_dir / "merged" / f"{category}.jsonl"
    if not jsonl_path.exists():
        # Also try without the merged/ subdirectory
        jsonl_path = local_dir / f"{category}.jsonl"
        if not jsonl_path.exists():
            print(
                f"  [{category}] Local file not found: {jsonl_path}",
                file=sys.stderr,
            )
            return []

    pairs: list[dict[str, Any]] = []
    skipped_short = 0
    skipped_empty = 0
    seen_titles: set[str] = set()

    print(f"  [{category}] Reading from {jsonl_path}...", file=sys.stderr)

    count = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            count += 1
            if limit > 0 and len(pairs) >= limit:
                break

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"  [{category}] JSON parse error at line {count}: {e}",
                    file=sys.stderr,
                )
                continue

            if not isinstance(record, dict):
                continue

            # Filter by content length
            content_len = _content_length(record)
            if content_len < min_length:
                skipped_short += 1
                continue

            # Dedup by title
            title = record.get("title", "")
            if title and title in seen_titles:
                continue
            if title:
                seen_titles.add(title)

            # Build training pair
            pair = build_pair(record, category)
            if pair is None:
                skipped_empty += 1
                continue

            pairs.append(pair)

            if count % 5000 == 0:
                print(
                    f"  [{category}] Processed {count} records, "
                    f"{len(pairs)} pairs extracted...",
                    file=sys.stderr,
                )

    print(
        f"  [{category}] Total scanned: {count}, "
        f"Extracted: {len(pairs)}, "
        f"Skipped (short): {skipped_short}, "
        f"Skipped (empty): {skipped_empty}",
        file=sys.stderr,
    )

    return pairs


# ---------------------------------------------------------------------------
# Dataset loading — HuggingFace streaming
# ---------------------------------------------------------------------------
def load_category_streaming(
    category: str,
    limit: int = 0,
    min_length: int = 200,
) -> list[dict[str, Any]]:
    """Load and process a single category from the HF dataset.

    Uses streaming to avoid loading the entire 1.19M record dataset into
    memory. Only keeps records that pass the min_length filter.

    Note: The CyberSecurity-1M dataset is gated. You must:
      1. Request access at https://huggingface.co/datasets/WhitzardAgent/CyberSecurity-1M
      2. Set HF_TOKEN environment variable: export HF_TOKEN=your_token
      3. Or use --local-dir to process pre-downloaded JSONL files
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "ERROR: 'datasets' package not installed. "
            "Install with: pip install datasets",
            file=sys.stderr,
        )
        sys.exit(1)

    pairs: list[dict[str, Any]] = []
    skipped_short = 0
    skipped_empty = 0
    seen_titles: set[str] = set()  # dedup by title

    print(f"  [{category}] Streaming from HuggingFace...", file=sys.stderr)

    try:
        ds = load_dataset(
            HF_DATASET_ID,
            data_files=f"merged/{category}.jsonl",
            split="train",
            streaming=True,
        )
    except Exception as e:
        print(f"  [{category}] ERROR loading dataset: {e}", file=sys.stderr)
        print(
            f"  [{category}] HINT: The CyberSecurity-1M dataset is gated. "
            f"Set HF_TOKEN or use --local-dir.",
            file=sys.stderr,
        )
        return []

    count = 0
    for record in ds:
        count += 1
        if limit > 0 and len(pairs) >= limit:
            break

        # Filter by content length
        content_len = _content_length(record)
        if content_len < min_length:
            skipped_short += 1
            continue

        # Dedup by title
        title = record.get("title", "")
        if title and title in seen_titles:
            continue
        if title:
            seen_titles.add(title)

        # Build training pair
        pair = build_pair(record, category)
        if pair is None:
            skipped_empty += 1
            continue

        pairs.append(pair)

        if count % 5000 == 0:
            print(
                f"  [{category}] Processed {count} records, "
                f"{len(pairs)} pairs extracted...",
                file=sys.stderr,
            )

    print(
        f"  [{category}] Total scanned: {count}, "
        f"Extracted: {len(pairs)}, "
        f"Skipped (short): {skipped_short}, "
        f"Skipped (empty): {skipped_empty}",
        file=sys.stderr,
    )

    return pairs


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------
def write_pairs(pairs: list[dict[str, Any]], output_path: Path) -> int:
    """Write training pairs to JSONL file. Returns count written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return len(pairs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract CyberSecurity-1M dataset from HuggingFace into AttackLM JSONL pairs"
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help=(
            "Comma-separated list of categories to process. "
            "Default: all 16 categories. "
            "Options: vulnerability,ctf,framework,tool,threat_intel,"
            "incident_response,reference,vuln_research,bug_bounty,"
            "offsec,conference,books,news,ai_security,ics_ot,cn_sec"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max records per category (0=all)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=200,
        help="Minimum content length in characters (default: 200)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/datasets/buckets/sources/cybersecurity-1m/)",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help=(
            "Local directory with pre-downloaded JSONL files "
            "(e.g., data/cybersecurity-1m/). Expects merged/<category>.jsonl "
            "or <category>.jsonl structure. Use this for gated datasets "
            "where HF streaming requires authentication."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process and print stats without writing files",
    )
    args = parser.parse_args()

    # All 16 categories
    ALL_CATEGORIES = [
        "vulnerability",
        "cn_sec",
        "framework",
        "reference",
        "ctf",
        "tool",
        "vuln_research",
        "incident_response",
        "bug_bounty",
        "threat_intel",
        "offsec",
        "conference",
        "books",
        "news",
        "ai_security",
        "ics_ot",
    ]

    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]
        # Validate
        invalid = [c for c in categories if c not in ALL_CATEGORIES]
        if invalid:
            print(
                f"ERROR: Invalid categories: {invalid}\n"
                f"Valid options: {', '.join(ALL_CATEGORIES)}",
                file=sys.stderr,
            )
            return 1
    else:
        # Process largest categories first (by record count)
        categories = [
            "vulnerability",
            "cn_sec",
            "framework",
            "reference",
            "ctf",
            "tool",
            "vuln_research",
            "incident_response",
            "bug_bounty",
            "threat_intel",
            "offsec",
            "conference",
            "books",
            "news",
            "ai_security",
            "ics_ot",
        ]

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    local_dir = Path(args.local_dir) if args.local_dir else None

    print("=" * 60, file=sys.stderr)
    print("AttackLM — CyberSecurity-1M Extractor", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(
        f"  Source:       {'local dir' if local_dir else HF_DATASET_ID}",
        file=sys.stderr,
    )
    if local_dir:
        print(f"  Local dir:    {local_dir}", file=sys.stderr)
    print(f"  Categories:   {', '.join(categories)}", file=sys.stderr)
    print(
        f"  Limit:        {'all' if args.limit == 0 else args.limit}", file=sys.stderr
    )
    print(f"  Min length:   {args.min_length} chars", file=sys.stderr)
    print(f"  Output dir:   {output_dir}", file=sys.stderr)
    print(f"  Dry run:      {args.dry_run}", file=sys.stderr)
    print(file=sys.stderr)

    total_pairs = 0
    stats: dict[str, int] = {}

    for category in categories:
        print(f"\n--- Processing: {category} ---", file=sys.stderr)

        if local_dir:
            pairs = load_category_from_local(
                category=category,
                local_dir=local_dir,
                limit=args.limit,
                min_length=args.min_length,
            )
        else:
            pairs = load_category_streaming(
                category=category,
                limit=args.limit,
                min_length=args.min_length,
            )

        stats[category] = len(pairs)
        total_pairs += len(pairs)

        if args.dry_run:
            # Print first 2 pairs as samples
            for i, pair in enumerate(pairs[:2]):
                print(f"\n  Sample {i + 1}:", file=sys.stderr)
                user_content = pair["messages"][1]["content"][:100]
                assistant_content = pair["messages"][2]["content"][:200]
                print(f"    User: {user_content}...", file=sys.stderr)
                print(f"    Assistant: {assistant_content}...", file=sys.stderr)
            continue

        # Write to per-category output
        bucket_type, tactic = CATEGORY_TO_SUBFOLDER.get(
            category, ("defensive", "security_reference")
        )
        category_output_dir = output_dir / category / bucket_type / tactic
        output_path = category_output_dir / "data.jsonl"

        written = write_pairs(pairs, output_path)
        print(
            f"  [{category}] Wrote {written} pairs to {output_path}",
            file=sys.stderr,
        )

    # Summary
    print(f"\n{'=' * 60}", file=sys.stderr)
    print("Extraction Summary", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"  Total pairs: {total_pairs}", file=sys.stderr)
    for cat, count in sorted(stats.items(), key=lambda x: -x[1]):
        pct = count / total_pairs * 100 if total_pairs > 0 else 0
        print(f"    {cat:25s}: {count:6d} ({pct:.1f}%)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
