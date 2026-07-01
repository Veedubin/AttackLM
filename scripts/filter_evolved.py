#!/usr/bin/env python3
"""Quality validation script for evolved training pairs.

After evolve_pairs.py generates longer training pairs, this script validates
them before mixing into training data. It runs quality checks and filters
out bad pairs.

Quality checks:
    1. JSONL Structure Validation — every record must have a valid ``messages``
       array with proper ``role``/``content`` fields. Roles must be one of:
       system, user, assistant.
    2. Length Increase Check — the evolved pair's total word count must be at
       least 2x the original. Loads the original from the source file and
       compares.
    3. MITRE ID Preservation — if the original had ``mitre_ids``, the evolved
       record must have the same ``mitre_ids`` (no hallucinated new IDs).
    4. Provenance Preservation — all metadata fields from original must be
       present: ``source``, ``source_uri``, ``license``, ``license_uri``,
       ``rights_contact``.
    5. No Hallucinated Content — assistant responses must not contain obviously
       fake technique names or MITRE IDs that don't exist in the original.
    6. Deduplication — no near-duplicate evolved pairs. Uses Jaccard similarity
       on word sets (threshold 0.9).

CLI interface:
    python scripts/filter_evolved.py \\
        --input data/datasets/evolved/metasploit-framework_multi_turn_abc123.jsonl \\
        --original data/datasets/buckets/sources/metasploit-framework/

    python scripts/filter_evolved.py --input data/datasets/evolved/ --all

    python scripts/filter_evolved.py --input data/datasets/evolved/ --dry-run

Output:
    Writes filtered JSONL to same directory with ``_filtered`` suffix.
    Prints report: total pairs, passed, failed, failure reasons with counts.
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

VALID_ROLES: set[str] = {"system", "user", "assistant"}

REQUIRED_PROVENANCE_FIELDS: list[str] = [
    "source",
    "source_uri",
    "license",
    "license_uri",
    "rights_contact",
]

# Regex for MITRE technique IDs (T1xxx, T1xxx.yyy, AML.Txxxx)
_TECHNIQUE_RE = re.compile(r"\b(T\d{4}(?:\.\d{3})?)\b", re.IGNORECASE)
_ATLAS_RE = re.compile(r"\b(AML\.T\d{4}(?:\.\d{3})?)\b", re.IGNORECASE)

# Default minimum length increase factor (evolved must be >= this * original word count)
DEFAULT_MIN_LENGTH_FACTOR = 2.0

# Default Jaccard similarity threshold for near-duplicate detection
DEFAULT_DEDUP_JACCARD_THRESHOLD = 0.9

# Base directory: one level up from this script
BASE_DIR = Path(__file__).resolve().parent.parent
EVOLVED_DIR = BASE_DIR / "data" / "datasets" / "evolved"
SOURCES_DIR = BASE_DIR / "data" / "datasets" / "buckets" / "sources"


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------


def validate_jsonl_structure(record: dict) -> list[str]:
    """Check 1: Validate JSONL structure — messages array with proper roles.

    Returns list of failure reasons (empty = pass).
    """
    reasons: list[str] = []

    if "messages" not in record:
        reasons.append("missing_messages_field")
        return reasons

    messages = record["messages"]
    if not isinstance(messages, list):
        reasons.append("messages_not_array")
        return reasons

    if len(messages) == 0:
        reasons.append("messages_empty")
        return reasons

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            reasons.append(f"message_{i}_not_dict")
            continue
        if "role" not in msg:
            reasons.append(f"message_{i}_missing_role")
        elif msg["role"] not in VALID_ROLES:
            reasons.append(f"message_{i}_invalid_role_{msg['role']}")
        if "content" not in msg:
            reasons.append(f"message_{i}_missing_content")

    return reasons


def check_length_increase(
    evolved_record: dict,
    original_record: dict,
    min_factor: float = DEFAULT_MIN_LENGTH_FACTOR,
) -> list[str]:
    """Check 2: Evolved word count must be >= min_factor * original.

    Returns list of failure reasons (empty = pass).
    """
    reasons: list[str] = []

    evolved_words = _total_word_count(evolved_record)
    original_words = _total_word_count(original_record)

    if original_words == 0:
        # Original has no content — evolved must have something
        if evolved_words == 0:
            reasons.append("both_original_and_evolved_empty")
        return reasons

    ratio = evolved_words / original_words
    if ratio < min_factor:
        reasons.append(
            f"length_increase_insufficient: "
            f"{evolved_words}/{original_words}={ratio:.2f}x "
            f"(need >= {min_factor}x)"
        )

    return reasons


def check_mitre_id_preservation(
    evolved_record: dict, original_record: dict
) -> list[str]:
    """Check 3: Evolved must preserve original mitre_ids (no added IDs).

    Returns list of failure reasons (empty = pass).
    """
    reasons: list[str] = []

    original_ids: set[str] = set(
        tid.upper() for tid in (original_record.get("mitre_ids") or [])
    )
    evolved_ids: set[str] = set(
        tid.upper() for tid in (evolved_record.get("mitre_ids") or [])
    )

    if original_ids and evolved_ids != original_ids:
        added = evolved_ids - original_ids
        missing = original_ids - evolved_ids
        if added:
            reasons.append(f"mitre_ids_added: {sorted(added)}")
        if missing:
            reasons.append(f"mitre_ids_missing: {sorted(missing)}")

    # If original had mitre_ids but evolved has none, that's a failure
    if original_ids and not evolved_ids:
        reasons.append("mitre_ids_removed_all")

    return reasons


def check_provenance_preservation(
    evolved_record: dict, original_record: dict
) -> list[str]:
    """Check 4: All metadata fields from original must be present in evolved.

    Returns list of failure reasons (empty = pass).
    """
    reasons: list[str] = []

    for field in REQUIRED_PROVENANCE_FIELDS:
        orig_val = original_record.get(field)
        evo_val = evolved_record.get(field)
        if orig_val and not evo_val:
            reasons.append(f"provenance_missing_{field}")
        elif orig_val and evo_val and orig_val != evo_val:
            reasons.append(f"provenance_mismatch_{field}")

    return reasons


def check_no_hallucinated_content(
    evolved_record: dict, original_record: dict
) -> list[str]:
    """Check 5: Assistant responses must not contain fake MITRE IDs.

    Extracts technique IDs from evolved assistant content and ensures
    they are all present in the original record's mitre_ids or content.

    Returns list of failure reasons (empty = pass).
    """
    reasons: list[str] = []

    # Collect all known technique IDs from original (structured + content)
    known_ids: set[str] = set()
    for tid in original_record.get("mitre_ids") or []:
        known_ids.add(tid.upper())
    for msg in original_record.get("messages", []):
        content = msg.get("content", "")
        if not content:
            continue
        for m in _TECHNIQUE_RE.finditer(content):
            known_ids.add(m.group(1).upper())
        for m in _ATLAS_RE.finditer(content):
            known_ids.add(m.group(1).upper())

    # Extract technique IDs from evolved assistant content
    evolved_assistant_ids: set[str] = set()
    for msg in evolved_record.get("messages", []):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if not content:
                continue
            for m in _TECHNIQUE_RE.finditer(content):
                evolved_assistant_ids.add(m.group(1).upper())
            for m in _ATLAS_RE.finditer(content):
                evolved_assistant_ids.add(m.group(1).upper())

    # Any IDs in evolved that aren't in original are hallucinated
    hallucinated = evolved_assistant_ids - known_ids
    if hallucinated:
        reasons.append(f"hallucinated_mitre_ids: {sorted(hallucinated)}")

    return reasons


def check_deduplication(
    records: list[dict], threshold: float = DEFAULT_DEDUP_JACCARD_THRESHOLD
) -> dict[int, str]:
    """Check 6: Near-duplicate detection using Jaccard similarity on word sets.

    Returns dict mapping record index to failure reason.
    """
    duplicates: dict[int, str] = {}
    word_sets: list[frozenset[str]] = []

    for i, record in enumerate(records):
        words = frozenset(_total_word_list(record))
        word_sets.append(words)

    # Compare each pair; mark the second one as duplicate
    for i in range(len(records)):
        if i in duplicates:
            continue
        for j in range(i + 1, len(records)):
            if j in duplicates:
                continue
            if not word_sets[i] or not word_sets[j]:
                continue
            intersection = word_sets[i] & word_sets[j]
            union = word_sets[i] | word_sets[j]
            if not union:
                continue
            jaccard = len(intersection) / len(union)
            if jaccard >= threshold:
                duplicates[j] = f"near_duplicate_of_record_{i} (jaccard={jaccard:.3f})"

    return duplicates


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _total_word_count(record: dict) -> int:
    """Count total words across all messages in a record."""
    total = 0
    for msg in record.get("messages", []):
        content = msg.get("content", "")
        if content:
            total += len(content.split())
    return total


def _total_word_list(record: dict) -> list[str]:
    """Collect all words across all messages in a record."""
    words: list[str] = []
    for msg in record.get("messages", []):
        content = msg.get("content", "")
        if content:
            words.extend(content.lower().split())
    return words


def load_jsonl(path: Path) -> list[dict]:
    """Load all records from a JSONL file."""
    records: list[dict] = []
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"  WARNING: malformed JSON at {path.name}:{line_num}: {exc}",
                    file=sys.stderr,
                )
    return records


def load_original_records(original_path: Path) -> list[dict]:
    """Load original records from source directory.

    Scans for data*.jsonl files under the source path (recursive).
    """
    if original_path.is_file():
        return load_jsonl(original_path)

    # Directory: scan for all data*.jsonl files recursively
    records: list[dict] = []
    if not original_path.is_dir():
        return records

    for jsonl_file in sorted(original_path.rglob("data*.jsonl")):
        records.extend(load_jsonl(jsonl_file))

    return records


def build_original_index(records: list[dict]) -> dict[str, dict]:
    """Build a lookup index from original records.

    Key is a content hash of the first user message content.
    This allows matching evolved records to their originals.
    """
    index: dict[str, dict] = {}
    for rec in records:
        # Use the first user message content as a matching key
        key = _record_match_key(rec)
        if key:
            index[key] = rec
    return index


def _record_match_key(record: dict) -> str:
    """Create a deterministic match key from a record.

    Uses the first 200 chars of the first user message, lowercased
    and stripped. This is used to match evolved records to their
    originals.
    """
    for msg in record.get("messages", []):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # Normalize: lowercase, strip whitespace, take first 200 chars
            normalized = content.lower().strip()[:200]
            return normalized
    return ""


def parse_evolved_filename(filename: str) -> str | None:
    """Extract source name from evolved filename.

    Evolved filenames follow the pattern:
        <source>_<suffix>.jsonl
    e.g. metasploit-framework_multi_turn_abc123.jsonl

    Returns the source name (e.g. 'metasploit-framework') or None.
    """
    stem = Path(filename).stem
    # Remove _filtered suffix if present
    if stem.endswith("_filtered"):
        stem = stem[: -len("_filtered")]

    # Split on underscore and reconstruct the source name
    # Known source names can contain underscores (e.g. metasploit-framework)
    # but the evolved suffix pattern is: _<type>_<hash>
    # We look for known suffixes: _multi_turn_, _single_turn_, _evolved_
    for suffix in ("_multi_turn_", "_single_turn_", "_evolved_"):
        idx = stem.find(suffix)
        if idx > 0:
            return stem[:idx]

    # Fallback: try to find source from directory structure
    return None


def discover_evolved_files(input_path: Path) -> list[Path]:
    """Discover evolved JSONL files from the input path.

    If input is a file, return it. If a directory, return all *.jsonl
    files (excluding *_filtered.jsonl files to avoid re-processing).
    """
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        files = sorted(
            p for p in input_path.glob("*.jsonl") if not p.stem.endswith("_filtered")
        )
        return files

    return []


def find_original_path(source_name: str, original_arg: Path | None) -> Path | None:
    """Find the original data directory for a given source name.

    Search order:
        1. If --original was provided and is a directory, check for
           sources/<source_name>/ under it.
        2. Check default SOURCES_DIR/<source_name>/.
        3. If --original was provided and is a file, use it directly.
    """
    if original_arg is not None:
        if original_arg.is_file():
            return original_arg
        # Check for source subdirectory
        candidate = original_arg / source_name
        if candidate.is_dir():
            return candidate
        # Check if original_arg IS the source directory
        if original_arg.name == source_name and original_arg.is_dir():
            return original_arg
        # Try it as-is
        if original_arg.is_dir():
            return original_arg

    # Default: look in SOURCES_DIR
    default_path = SOURCES_DIR / source_name
    if default_path.is_dir():
        return default_path

    return None


# ---------------------------------------------------------------------------
# Filtering logic
# ---------------------------------------------------------------------------


def validate_single_record(
    evolved: dict,
    original: dict | None,
    min_factor: float = DEFAULT_MIN_LENGTH_FACTOR,
) -> list[str]:
    """Run all per-record quality checks on an evolved record.

    Returns list of failure reasons (empty = pass).
    """
    all_reasons: list[str] = []

    # Check 1: JSONL structure
    all_reasons.extend(validate_jsonl_structure(evolved))

    # If we have an original, run comparison checks
    if original is not None:
        # Check 2: Length increase
        all_reasons.extend(check_length_increase(evolved, original, min_factor))

        # Check 3: MITRE ID preservation
        all_reasons.extend(check_mitre_id_preservation(evolved, original))

        # Check 4: Provenance preservation
        all_reasons.extend(check_provenance_preservation(evolved, original))

        # Check 5: No hallucinated content
        all_reasons.extend(check_no_hallucinated_content(evolved, original))

    return all_reasons


def filter_evolved_file(
    evolved_path: Path,
    original_path: Path | None = None,
    dry_run: bool = False,
    min_factor: float = DEFAULT_MIN_LENGTH_FACTOR,
    dedup_threshold: float = DEFAULT_DEDUP_JACCARD_THRESHOLD,
) -> dict[str, Any]:
    """Filter a single evolved JSONL file.

    Returns a report dict with stats and the list of passed records.
    """
    # Load evolved records
    evolved_records = load_jsonl(evolved_path)
    if not evolved_records:
        return {
            "file": str(evolved_path),
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped_no_original": 0,
            "failure_reasons": Counter(),
            "duplicates_removed": 0,
        }

    # Load and index original records
    original_records: list[dict] = []
    original_index: dict[str, dict] = {}
    if original_path is not None:
        original_records = load_original_records(original_path)
        original_index = build_original_index(original_records)

    # Per-record validation
    passed: list[dict] = []
    failed: list[tuple[dict, list[str]]] = []
    skipped_no_original = 0
    reason_counter: Counter = Counter()

    for i, evolved in enumerate(evolved_records):
        # Find matching original
        match_key = _record_match_key(evolved)
        original = original_index.get(match_key) if match_key else None

        if original_path is not None and original is None:
            skipped_no_original += 1
            # Still validate structure, but skip comparison checks
            reasons = validate_jsonl_structure(evolved)
            if reasons:
                failed.append((evolved, reasons))
                for r in reasons:
                    reason_counter[r.split(":")[0]] += 1
            else:
                # No original found — can't verify, skip this record
                reason_counter["no_matching_original"] += 1
                failed.append((evolved, ["no_matching_original"]))
            continue

        reasons = validate_single_record(evolved, original, min_factor)

        if reasons:
            failed.append((evolved, reasons))
            # Normalize reason keys (strip detail after colon for counting)
            for r in reasons:
                key = r.split(":")[0].strip()
                reason_counter[key] += 1
        else:
            passed.append(evolved)

    # Check 6: Deduplication (only on passed records)
    dup_map = check_deduplication(passed, dedup_threshold)
    duplicates_removed = len(dup_map)
    if dup_map:
        # Remove duplicate records from passed list (in reverse order to
        # preserve indices)
        for idx in sorted(dup_map.keys(), reverse=True):
            passed.pop(idx)
        for idx, reason in dup_map.items():
            reason_counter["near_duplicate"] += 1

    report = {
        "file": str(evolved_path),
        "total": len(evolved_records),
        "passed": len(passed),
        "failed": len(failed),
        "skipped_no_original": skipped_no_original,
        "failure_reasons": dict(reason_counter.most_common()),
        "duplicates_removed": duplicates_removed,
    }

    # Write filtered output (unless dry run)
    if not dry_run and passed:
        # Output path: same directory, _filtered suffix
        output_path = evolved_path.parent / (evolved_path.stem + "_filtered.jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            for rec in passed:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        report["output_path"] = str(output_path)

    return report, passed


def print_report(report: dict) -> None:
    """Print a human-readable filter report."""
    print()
    print("=" * 72)
    print(f"  Filter Report: {Path(report['file']).name}")
    print("=" * 72)
    print(f"  Total records:              {report['total']:,}")
    print(f"  Passed:                      {report['passed']:,}")
    print(f"  Failed:                      {report['failed']:,}")
    print(f"  Duplicates removed:          {report['duplicates_removed']:,}")
    print(f"  Skipped (no original):       {report['skipped_no_original']:,}")

    if report["failure_reasons"]:
        print()
        print("  Failure Reasons:")
        for reason, count in sorted(
            report["failure_reasons"].items(), key=lambda x: -x[1]
        ):
            print(f"    {reason}: {count}")

    if "output_path" in report:
        print()
        print(f"  Output: {report['output_path']}")

    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Quality validation for evolved training pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Path to an evolved JSONL file or directory of evolved files. "
            "If a directory, all *.jsonl files (excluding *_filtered.jsonl) "
            "are processed."
        ),
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=None,
        help=(
            "Path to the original source directory or file. "
            "If a directory, it should point to the source under "
            "data/datasets/buckets/sources/<source>/. "
            "If omitted, the script tries to auto-detect from the "
            "evolved filename."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Process all evolved files in the input directory. "
            "Requires --input to be a directory."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report only — don't write filtered output files.",
    )
    parser.add_argument(
        "--min-length-factor",
        type=float,
        default=DEFAULT_MIN_LENGTH_FACTOR,
        help=(
            f"Minimum word count increase factor (default: {DEFAULT_MIN_LENGTH_FACTOR}). "
            "Evolved must be >= this * original word count."
        ),
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=DEFAULT_DEDUP_JACCARD_THRESHOLD,
        help=(
            f"Jaccard similarity threshold for near-duplicate detection "
            f"(default: {DEFAULT_DEDUP_JACCARD_THRESHOLD})."
        ),
    )

    args = parser.parse_args(argv)

    min_factor = args.min_length_factor
    dedup_threshold = args.dedup_threshold

    # Discover evolved files
    evolved_files = discover_evolved_files(args.input)
    if not evolved_files:
        print(
            f"ERROR: No evolved JSONL files found at {args.input}",
            file=sys.stderr,
        )
        return 1

    if args.input.is_file():
        # Single file mode
        evolved_path = args.input
        source_name = parse_evolved_filename(evolved_path.name)
        original_path = (
            find_original_path(source_name, args.original)
            if source_name
            else args.original
        )

        if original_path is None and source_name:
            print(
                f"WARNING: Could not find original source for '{source_name}'. "
                f"Comparison checks will be skipped.",
                file=sys.stderr,
            )
        elif original_path is None:
            print(
                "WARNING: Could not determine source name from filename. "
                "Comparison checks will be skipped. Use --original to specify.",
                file=sys.stderr,
            )

        report, passed = filter_evolved_file(
            evolved_path,
            original_path,
            dry_run=args.dry_run,
            min_factor=min_factor,
            dedup_threshold=dedup_threshold,
        )
        print_report(report)

        if args.dry_run:
            print("\n(dry run — no files written)")

    elif args.input.is_dir():
        # Directory mode: process all evolved files
        total_reports: list[dict] = []
        total_passed = 0
        total_failed = 0
        total_records = 0

        for evolved_path in evolved_files:
            source_name = parse_evolved_filename(evolved_path.name)
            original_path = (
                find_original_path(source_name, args.original)
                if source_name
                else args.original
            )

            if original_path is None and source_name:
                print(
                    f"  WARNING: No original found for '{source_name}', "
                    f"skipping comparison checks for {evolved_path.name}",
                    file=sys.stderr,
                )
            elif original_path is None:
                print(
                    f"  WARNING: Cannot determine source for {evolved_path.name}, "
                    f"skipping comparison checks. Use --original.",
                    file=sys.stderr,
                )

            report, passed = filter_evolved_file(
                evolved_path,
                original_path,
                dry_run=args.dry_run,
                min_factor=min_factor,
                dedup_threshold=dedup_threshold,
            )
            print_report(report)
            total_reports.append(report)
            total_passed += report["passed"]
            total_failed += report["failed"]
            total_records += report["total"]

        # Summary
        if len(evolved_files) > 1:
            print()
            print("=" * 72)
            print("  AGGREGATE SUMMARY")
            print("=" * 72)
            print(f"  Files processed:    {len(evolved_files)}")
            print(f"  Total records:       {total_records:,}")
            print(f"  Total passed:        {total_passed:,}")
            print(f"  Total failed:        {total_failed:,}")
            agg_reasons: Counter = Counter()
            for r in total_reports:
                for reason, count in r.get("failure_reasons", {}).items():
                    agg_reasons[reason] += count
            if agg_reasons:
                print()
                print("  Aggregate Failure Reasons:")
                for reason, count in agg_reasons.most_common():
                    print(f"    {reason}: {count}")
            print("=" * 72)

        if args.dry_run:
            print("\n(dry run — no files written)")

    else:
        print(f"ERROR: Input path does not exist: {args.input}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
