#!/usr/bin/env python3
# CREDITS — DATA SOURCE ATTRIBUTION
# ----------------------------------
# This script transforms data from: CyberLLMInstruct
# Paper:      ACM Workshop on AI+Sec 2025 —
#             "CyberLLMInstruct: Pseudo-malicious instruction-response
#              pairs for security LLM fine-tuning"
# Dataset:    54,928 instruction-response pairs
# License:    Academic / Research Use (verify before redistribution)
#
# The output JSONL is a *transformation* of upstream instruction-response
# pairs into OpenAI-style chat triples. See /ATTRIBUTION.md for full
# per-source attribution and re-distribution guidance.
# ----------------------------------
"""Extract CyberLLMInstruct dataset into AttackLM JSONL training pairs.

Supports two modes:

1. **HuggingFace download** — If the dataset is available on HuggingFace,
   use ``--hf-dataset`` to specify the repo ID (e.g.
   ``username/cyberllm-instruct``).  Requires the ``datasets`` package.

2. **Local files** — Use ``--input-dir`` to point at a directory containing
   the dataset files (JSON, JSONL, CSV, or Parquet).  This is the default
   mode when no ``--hf-dataset`` is provided.

The dataset contains pseudo-malicious instruction-response pairs designed
for security LLM fine-tuning. Each pair is tagged with ``"pseudo_malicious":
True`` to indicate that the responses are synthetically generated for
training purposes and should not be used as factual security references.

Output:
    ``data/datasets/buckets/sources/cyberllm-instruct/offensive_security/data.jsonl``

Usage:
    # From local files (default)
    python scripts/extract_cyberllm_instruct.py --input-dir /path/to/cyberllm-instruct

    # From HuggingFace (requires datasets package)
    python scripts/extract_cyberllm_instruct.py --hf-dataset username/cyberllm-instruct

    # With limits and dry-run
    python scripts/extract_cyberllm_instruct.py --input-dir ./data/cyberllm-instruct --limit 100 --dry-run
"""

from __future__ import annotations

import argparse
import csv
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
    BASE_DIR
    / "data"
    / "datasets"
    / "buckets"
    / "sources"
    / "cyberllm-instruct"
    / "offensive_security"
)
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "data.jsonl"

# ---------------------------------------------------------------------------
# System message
# ---------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are a cybersecurity specialist trained on pseudo-malicious instruction-"
    "response pairs. Provide security-relevant analysis, technique identification, "
    "and defensive recommendations based on adversarial patterns."
)

# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------
ATTRIBUTION = {
    "source": "cyberllm-instruct",
    "source_uri": "https://github.com/CyberLLMInstruct",
    "license": "Research/Academic",
    "license_uri": "",
    "rights_contact": "CyberLLMInstruct Authors (ACM AI+Sec 2025)",
    "attribution_text": (
        "CyberLLMInstruct: Pseudo-malicious instruction-response pairs for "
        "security LLM fine-tuning. ACM Workshop on AI+Sec 2025. "
        "For research and academic use only."
    ),
}

# ---------------------------------------------------------------------------
# MITRE ATT&CK technique ID regex
# ---------------------------------------------------------------------------
_MITRE_RE = re.compile(r"\b(T\d{4}(?:\.\d{3})?)\b")


def extract_mitre_ids(text: str) -> list[str]:
    """Extract MITRE ATT&CK technique IDs from text."""
    return sorted(set(_MITRE_RE.findall(text)))


def tag_pair_with_tactic(pair: dict[str, Any]) -> None:
    """Add MITRE tactic metadata to a pair dict, if any technique IDs found."""
    mitre_ids = pair.get("mitre_ids", [])
    for tech_id in mitre_ids:
        tactic_id = get_tactic_for_technique(tech_id)
        if tactic_id:
            pair["mitre_tactic_id"] = tactic_id
            tactic_name = get_tactic_name(tactic_id)
            if tactic_name:
                pair["tactic"] = tactic_name
                pair["kill_chain_phase"] = tactic_name
            break


# ---------------------------------------------------------------------------
# Row normalisation
# ---------------------------------------------------------------------------
def _normalise_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Normalise a single dataset row into an instruction + response pair.

    Handles multiple column naming conventions found in instruction-tuning
    datasets: ``instruction``/``input``/``output``,
    ``prompt``/``response``, ``question``/``answer``, etc.

    Returns None if the row cannot be normalised.
    """
    # --- Determine instruction field ---
    instruction = ""
    for key in ("instruction", "prompt", "question", "input", "query", "ask"):
        if key in row and row[key]:
            instruction = str(row[key]).strip()
            break

    # Combine instruction + input if both present (Alpaca-style)
    extra_input = ""
    for key in ("input", "context", "background"):
        if key in row and row[key] and key != "instruction":
            extra_input = str(row[key]).strip()
            break

    if extra_input and instruction:
        instruction = f"{instruction}\n\n{extra_input}"
    elif extra_input and not instruction:
        instruction = extra_input

    if not instruction or len(instruction) < 5:
        return None

    # --- Determine response field ---
    response = ""
    for key in ("output", "response", "answer", "completion", "target", "reply"):
        if key in row and row[key]:
            response = str(row[key]).strip()
            break

    if not response or len(response) < 5:
        return None

    return {"instruction": instruction, "response": response}


# ---------------------------------------------------------------------------
# Build training pair
# ---------------------------------------------------------------------------
def build_pair(instruction: str, response: str, idx: int) -> dict[str, Any]:
    """Build one AttackLM training pair from an instruction-response pair."""
    # Extract MITRE IDs from both instruction and response
    all_text = f"{instruction}\n{response}"
    mitre_ids = extract_mitre_ids(all_text)

    # Determine a short topic from the instruction (first 80 chars)
    topic = instruction[:80].replace("\n", " ").strip()
    if len(instruction) > 80:
        topic = topic.rstrip() + "…"

    user_msg = f"How would a security analyst respond to the following instruction?\n\n{instruction}"

    # Build the assistant content, annotating pseudo-malicious nature
    assistant_msg = response

    pair: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "mitre_ids": mitre_ids,
        "pseudo_malicious": True,
        "pair_index": idx,
        **ATTRIBUTION,
    }

    # Add tactic metadata
    tag_pair_with_tactic(pair)

    return pair


# ---------------------------------------------------------------------------
# Parse local files
# ---------------------------------------------------------------------------
def _parse_jsonl(filepath: Path) -> list[dict[str, Any]]:
    """Parse a JSONL file (one JSON object per line)."""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except json.JSONDecodeError:
                print(
                    f"  [WARN] Skipping malformed line {line_num} in {filepath.name}",
                    file=sys.stderr,
                )
    return rows


def _parse_json(filepath: Path) -> list[dict[str, Any]]:
    """Parse a JSON file (may be a list of objects or a single object)."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        rows = [item for item in data if isinstance(item, dict)]
    elif isinstance(data, dict):
        # Check if it's a HuggingFace-style split dict
        # (e.g. {"train": [...], "test": [...]})
        rows = []
        for key in ("train", "test", "validation", "val"):
            if key in data and isinstance(data[key], list):
                rows.extend(item for item in data[key] if isinstance(item, dict))
        if not rows:
            # Single object
            rows.append(data)
    else:
        rows = []

    return rows


def _parse_csv(filepath: Path) -> list[dict[str, Any]]:
    """Parse a CSV file."""
    rows = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def parse_local_files(input_dir: Path) -> list[dict[str, Any]]:
    """Walk input_dir for JSON, JSONL, CSV, and Parquet files and parse them."""
    all_rows: list[dict[str, Any]] = []
    file_count = 0

    # JSONL files
    for jsonl_file in sorted(input_dir.rglob("*.jsonl")):
        print(f"  Parsing JSONL: {jsonl_file.name}", file=sys.stderr)
        rows = _parse_jsonl(jsonl_file)
        all_rows.extend(rows)
        file_count += 1

    # JSON files
    for json_file in sorted(input_dir.rglob("*.json")):
        print(f"  Parsing JSON: {json_file.name}", file=sys.stderr)
        rows = _parse_json(json_file)
        all_rows.extend(rows)
        file_count += 1

    # CSV files
    for csv_file in sorted(input_dir.rglob("*.csv")):
        print(f"  Parsing CSV: {csv_file.name}", file=sys.stderr)
        rows = _parse_csv(csv_file)
        all_rows.extend(rows)
        file_count += 1

    # Parquet files
    for parquet_file in sorted(input_dir.rglob("*.parquet")):
        try:
            import pyarrow.parquet as pq  # type: ignore[import-untyped]

            print(f"  Parsing Parquet: {parquet_file.name}", file=sys.stderr)
            table = pq.read_table(parquet_file)
            for i in range(table.num_rows):
                row = {col: table[col][i].as_py() for col in table.column_names}
                all_rows.append(row)
            file_count += 1
        except ImportError:
            print(
                f"  [WARN] pyarrow not installed, skipping {parquet_file.name}. "
                f"Install with: pip install pyarrow",
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"  [WARN] Error parsing {parquet_file.name}: {exc}", file=sys.stderr)

    print(
        f"  Parsed {file_count} files, {len(all_rows)} total rows",
        file=sys.stderr,
    )
    return all_rows


# ---------------------------------------------------------------------------
# Parse HuggingFace dataset
# ---------------------------------------------------------------------------
def parse_hf_dataset(repo_id: str, split: str = "train") -> list[dict[str, Any]]:
    """Download and parse a HuggingFace dataset.

    Requires the ``datasets`` package.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        print(
            "ERROR: The 'datasets' package is required for HuggingFace download.\n"
            "  Install with: pip install datasets",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  Downloading HuggingFace dataset: {repo_id}", file=sys.stderr)
    try:
        ds = load_dataset(repo_id, split=split)
    except Exception as exc:
        print(f"ERROR: Failed to download dataset '{repo_id}': {exc}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict[str, Any]] = []
    for item in ds:
        row = {k: v for k, v in item.items() if v is not None}
        rows.append(row)

    print(f"  Downloaded {len(rows)} rows from HuggingFace", file=sys.stderr)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract CyberLLMInstruct dataset into AttackLM JSONL training pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # From local files (default)
  python scripts/extract_cyberllm_instruct.py --input-dir ./data/cyberllm-instruct

  # From HuggingFace (requires datasets package)
  python scripts/extract_cyberllm_instruct.py --hf-dataset username/cyberllm-instruct

  # With limits and dry-run
  python scripts/extract_cyberllm_instruct.py --input-dir ./data/cyberllm-instruct --limit 100 --dry-run
""",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Local directory containing CyberLLMInstruct data files (JSON, JSONL, CSV, Parquet).",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help="HuggingFace dataset repo ID (e.g. 'username/cyberllm-instruct'). Requires datasets package.",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="train",
        help="HuggingFace dataset split to use (default: train).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output path for JSONL.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit to N instruction-response pairs (0 = all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print stats without writing output.",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT_PATH

    # --- Determine data source ---
    if args.hf_dataset:
        all_rows = parse_hf_dataset(args.hf_dataset, split=args.hf_split)
        source_desc = f"HuggingFace: {args.hf_dataset} (split={args.hf_split})"
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"ERROR: Input directory not found: {input_dir}", file=sys.stderr)
            return 1
        all_rows = parse_local_files(input_dir)
        source_desc = f"Local: {input_dir}"
    else:
        print(
            "ERROR: No data source specified. Use --input-dir for local files or "
            "--hf-dataset for HuggingFace download.",
            file=sys.stderr,
        )
        print(
            "\nManual download instructions:\n"
            "  1. Search for 'CyberLLMInstruct' on HuggingFace or the paper's GitHub repo\n"
            "  2. Download the dataset files to a local directory\n"
            "  3. Run: python scripts/extract_cyberllm_instruct.py --input-dir /path/to/data\n",
            file=sys.stderr,
        )
        return 1

    print(f"\nAttackLM — Extract CyberLLMInstruct Training Pairs", file=sys.stderr)
    print(f"  Source:  {source_desc}", file=sys.stderr)
    print(f"  Output:  {output_path}", file=sys.stderr)
    print(f"  Raw rows: {len(all_rows)}", file=sys.stderr)
    print()

    # --- Normalise rows to instruction-response pairs ---
    pairs: list[dict[str, Any]] = []
    skipped = 0

    for idx, row in enumerate(all_rows):
        normalised = _normalise_row(row)
        if normalised is None:
            skipped += 1
            continue

        pair = build_pair(
            instruction=normalised["instruction"],
            response=normalised["response"],
            idx=idx,
        )
        pairs.append(pair)

        if args.limit > 0 and len(pairs) >= args.limit:
            break

    print(f"  Normalised: {len(pairs)} pairs, Skipped: {skipped}", file=sys.stderr)

    # --- Stats ---
    mitre_count = sum(1 for p in pairs if p.get("mitre_ids"))
    tactic_count = sum(1 for p in pairs if p.get("mitre_tactic_id"))
    all_mitre: set[str] = set()
    for p in pairs:
        all_mitre.update(p.get("mitre_ids", []))

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  Total pairs:       {len(pairs)}", file=sys.stderr)
    print(f"  Skipped rows:      {skipped}", file=sys.stderr)
    print(f"  Pairs w/ MITRE:    {mitre_count}", file=sys.stderr)
    print(f"  Pairs w/ tactic:   {tactic_count}", file=sys.stderr)
    print(f"  Unique MITRE IDs:  {len(all_mitre)}", file=sys.stderr)
    if all_mitre:
        ids_str = ", ".join(sorted(all_mitre)[:20])
        if len(all_mitre) > 20:
            ids_str += "…"
        print(f"    {ids_str}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    if args.dry_run:
        print("\n  DRY RUN — No files written", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)
        # Show sample pairs
        if pairs:
            print(f"\n  Sample pair (first):\n", file=sys.stderr)
            sample = pairs[0]
            print(
                json.dumps(sample, indent=2, ensure_ascii=False)[:3000], file=sys.stderr
            )
            if len(pairs) > 1:
                print(f"\n  Sample pair (last):\n", file=sys.stderr)
                last = pairs[-1]
                print(
                    json.dumps(last, indent=2, ensure_ascii=False)[:3000],
                    file=sys.stderr,
                )
        return 0

    # --- Write output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n  Written: {len(pairs)} pairs → {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
