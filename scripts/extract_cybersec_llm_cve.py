#!/usr/bin/env python3
# CREDITS — DATA SOURCE ATTRIBUTION
# ----------------------------------
# This script transforms data from: Bouquets/Cybersecurity-LLM-CVE
# Dataset:    https://huggingface.co/datasets/Bouquets/Cybersecurity-LLM-CVE
# License:    MIT (implied — see upstream repository for terms)
#
# The output JSONL is a *transformation* of upstream CVE Q&A pairs into
# OpenAI-style chat messages. See /ATTRIBUTION.md for full per-source
# attribution and re-distribution guidance.
# ----------------------------------
"""Extract CVE Q&A pairs from Cybersecurity-LLM-CVE HuggingFace dataset.

Downloads the Bouquets/Cybersecurity-LLM-CVE dataset (125K CVE instruction-
response pairs) and converts each record to AttackLM JSONL format with MITRE
ATT&CK technique extraction.

Output: ``data/datasets/buckets/sources/cybersec-llm-cve/vulnerability_analysis/data.jsonl``

Usage:
    python scripts/extract_cybersec_llm_cve.py
    python scripts/extract_cybersec_llm_cve.py --limit 100 --output-dir /tmp/test_cve
    python scripts/extract_cybersec_llm_cve.py --dry-run --limit 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset
except ImportError:
    print(
        "ERROR: 'datasets' package is required. Install with: pip install datasets",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "datasets"
    / "buckets"
    / "sources"
    / "cybersec-llm-cve"
    / "vulnerability_analysis"
)

# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------
DATASET_NAME = "Bouquets/Cybersecurity-LLM-CVE"
DATASET_SPLIT = "train"

# ---------------------------------------------------------------------------
# System message
# ---------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are a Cybersecurity Vulnerability Analysis specialist. Provide detailed "
    "information about CVEs including descriptions, affected products, severity, "
    "and remediation guidance based on MITRE ATT&CK frameworks."
)

# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------
ATTRIBUTION = {
    "source": "Cybersecurity-LLM-CVE",
    "source_uri": "https://huggingface.co/datasets/Bouquets/Cybersecurity-LLM-CVE",
    "license": "MIT",
    "license_uri": "https://opensource.org/licenses/MIT",
}

# ---------------------------------------------------------------------------
# MITRE ATT&CK extraction
# ---------------------------------------------------------------------------
_MITRE_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")


def extract_mitre_ids(text: str) -> list[str]:
    """Extract MITRE ATT&CK technique IDs from text.

    Matches patterns like T1566, T1566.001, T1059, T1059.005, etc.
    Returns sorted unique list of uppercase technique IDs.
    """
    matches = _MITRE_RE.findall(text)
    return sorted(set(m.upper() for m in matches))


# ---------------------------------------------------------------------------
# Build training pair from dataset record
# ---------------------------------------------------------------------------
def build_pair(record: dict[str, Any]) -> dict[str, Any] | None:
    """Build one AttackLM training pair from a HuggingFace dataset record.

    Expected record fields: instruction, inputs, outputs.
    Handles missing or empty fields gracefully.
    """
    instruction = record.get("instruction", "")
    inputs = record.get("inputs", "")
    outputs = record.get("outputs", "")

    # Skip records with empty instruction or response
    if not instruction or not outputs:
        return None

    # Build user content — include inputs if non-empty
    user_content = instruction.strip()
    if inputs and inputs.strip():
        user_content = f"{user_content}\n\nContext: {inputs.strip()}"

    # Extract MITRE ATT&CK IDs from the response text
    mitre_ids = extract_mitre_ids(outputs)

    pair = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": outputs.strip()},
        ],
        "source": ATTRIBUTION["source"],
        "source_uri": ATTRIBUTION["source_uri"],
        "license": ATTRIBUTION["license"],
        "license_uri": ATTRIBUTION["license_uri"],
        "mitre_ids": mitre_ids,
    }

    return pair


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract Cybersecurity-LLM-CVE dataset into AttackLM JSONL training pairs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(f"Output directory for data.jsonl (default: {DEFAULT_OUTPUT_DIR})"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of records to process (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sample pairs without writing files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_path = output_dir / "data.jsonl"

    print("AttackLM — Extract Cybersecurity-LLM-CVE Dataset", file=sys.stderr)
    print(f"  Dataset:     {DATASET_NAME}", file=sys.stderr)
    print(f"  Output:      {output_path}", file=sys.stderr)
    print(f"  Limit:       {args.limit or 'all'}", file=sys.stderr)
    print(file=sys.stderr)

    # Download dataset with streaming for memory efficiency
    print("Downloading dataset (streaming mode)...", file=sys.stderr)
    try:
        dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)
    except Exception as exc:
        print(f"ERROR: Failed to load dataset: {exc}", file=sys.stderr)
        print(
            "Make sure 'datasets' is installed and you have internet access.",
            file=sys.stderr,
        )
        return 1

    # Process records
    pairs: list[dict[str, Any]] = []
    skipped = 0
    total_response_chars = 0
    total_mitre_found = 0
    processed = 0

    for record in dataset:
        if args.limit and processed >= args.limit:
            break

        processed += 1
        pair = build_pair(record)

        if pair is None:
            skipped += 1
            continue

        total_response_chars += len(pair["messages"][1]["content"])
        total_mitre_found += len(pair["mitre_ids"])
        pairs.append(pair)

    # Compute summary stats
    total_pairs = len(pairs)
    avg_response_len = total_response_chars / total_pairs if total_pairs else 0
    mitre_coverage = (
        sum(1 for p in pairs if p["mitre_ids"]) / total_pairs * 100
        if total_pairs
        else 0.0
    )

    print(f"\nExtraction complete:", file=sys.stderr)
    print(f"  Records processed: {processed}", file=sys.stderr)
    print(f"  Valid pairs:       {total_pairs}", file=sys.stderr)
    print(f"  Skipped:           {skipped}", file=sys.stderr)
    print(f"  Avg response len:  {avg_response_len:.0f} chars", file=sys.stderr)
    print(
        f"  MITRE coverage:    {mitre_coverage:.1f}% of pairs have technique IDs",
        file=sys.stderr,
    )
    print(f"  Total MITRE IDs:   {total_mitre_found}", file=sys.stderr)

    # Dry run: print samples
    if args.dry_run:
        print(f"\n--- DRY RUN: Sample pairs ---", file=sys.stderr)
        for pair in pairs[:5]:
            print(json.dumps(pair, indent=2))
        if total_pairs > 5:
            print(f"\n... and {total_pairs - 5} more pairs", file=sys.stderr)
        return 0

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\nWrote {total_pairs} pairs to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
