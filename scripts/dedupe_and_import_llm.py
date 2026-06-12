#!/usr/bin/env python3
"""
dedupe_and_import_llm.py — Dedupe + import LLM-generated synthetic data into
the per-source layout.

Takes the *_llm.jsonl files in data/datasets/synthetic/, deduplicates them
by user question text (same question = same pair), and writes the
deduplicated entries into
`data/datasets/buckets/sources/llm-generated/<bucket>/data_llm.jsonl`.

The per-source layout (v0.3.0+) is canonical — see
`data/datasets/buckets/sources/README.md`. The three-tier provenance
fields (source=llm-generated, license=GPL-3.0, source_uri, etc.) are
added by `scripts/stamp_and_reorg.py` after the dedupe+import step.

Usage:
    python scripts/dedupe_and_import_llm.py
    python scripts/dedupe_and_import_llm.py --dry-run   # Preview without writing
    python scripts/dedupe_and_import_llm.py --verbose   # Show per-category breakdown
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SYNTHETIC_DIR = PROJECT_DIR / "data" / "datasets" / "synthetic"
SOURCES_DIR = PROJECT_DIR / "data" / "datasets" / "buckets" / "sources"
LLM_SOURCE_DIR = SOURCES_DIR / "llm-generated"

# Map LLM category name (from *_llm.jsonl files) to bucket directory.
# ics_scada → ics/attacks (manifest uses "ics", not "ics_scada")
# supply_chain → supply_chain/attacks
# All others match by name.
CATEGORY_TO_BUCKET_DIR: dict[str, str] = {
    "web_app": "web_app/attacks",
    "cloud": "cloud/attacks",
    "social_engineering": "social_engineering/phishing",
    "supply_chain": "supply_chain/attacks",
    "ics_scada": "ics/attacks",
    "wireless": "wireless/attacks",
}

# Normalize text for dedup: lowercase, collapse whitespace, strip punctuation.
_NORMALIZE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize_question(q: str) -> str:
    """Normalize a question for fuzzy dedup matching.

    Strategy: lowercase, strip punctuation, collapse whitespace.
    Two questions that differ only in punctuation/case/spacing are
    considered duplicates.
    """
    q = q.lower().strip()
    q = _PUNCT_RE.sub(" ", q)
    q = _NORMALIZE_RE.sub(" ", q).strip()
    return q


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, skip blank lines, parse each line."""
    entries: list[dict] = []
    with open(path) as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(
                    f"  WARNING: {path.name}:{line_no} — JSON parse error: {e}",
                    file=sys.stderr,
                )
    return entries


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    """Write entries to a JSONL file, one entry per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")


def _is_placeholder(entry: dict) -> bool:
    """Detect a mock/test entry that shouldn't be imported.

    Heuristics:
    - Assistant content is very short (< 100 chars) and generic
    - User question is "Say only the word X." or similar mock pattern
    """
    try:
        msgs = entry["messages"]
        user_q = msgs[1]["content"]
        assistant_a = msgs[2]["content"]
    except (KeyError, IndexError, TypeError):
        return False

    if len(assistant_a) < 50 and "mock" in user_q.lower():
        return True
    if user_q.strip().lower().startswith("say only the word"):
        return True
    return False


def process_category(
    cat_key: str,
    bucket_dir_rel: str,
    *,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    """Dedupe + import one category. Returns stats dict."""
    src_path = SYNTHETIC_DIR / f"{cat_key}_llm.jsonl"
    bucket_path = LLM_SOURCE_DIR / bucket_dir_rel
    dst_path = bucket_path / "data_llm.jsonl"

    if not src_path.exists():
        return {
            "category": cat_key,
            "source_exists": False,
            "skipped_reason": "no source file",
        }

    raw_entries = _read_jsonl(src_path)
    n_raw = len(raw_entries)

    if n_raw == 0:
        return {
            "category": cat_key,
            "source_exists": True,
            "raw_count": 0,
            "imported": 0,
            "skipped_reason": "empty source file",
        }

    # Detect placeholders (mock data) and exclude them
    placeholders = [e for e in raw_entries if _is_placeholder(e)]
    real_entries = [e for e in raw_entries if not _is_placeholder(e)]
    n_placeholders = len(placeholders)

    # Dedupe by normalized user question (keep first occurrence)
    seen_questions: set[str] = set()
    unique_entries: list[dict] = []
    duplicate_count = 0
    for e in real_entries:
        try:
            q = e["messages"][1]["content"]
        except (KeyError, IndexError, TypeError):
            # Malformed entry — skip
            duplicate_count += 1
            continue
        nq = _normalize_question(q)
        if nq in seen_questions:
            duplicate_count += 1
            continue
        seen_questions.add(nq)
        unique_entries.append(e)

    # Dedupe by full content (in case different questions have identical A's)
    seen_content: set[str] = set()
    final_entries: list[dict] = []
    content_dups = 0
    for e in unique_entries:
        try:
            content_blob = json.dumps(
                {
                    "q": e["messages"][1]["content"],
                    "a": e["messages"][2]["content"],
                },
                sort_keys=True,
                ensure_ascii=False,
            )
        except (KeyError, IndexError, TypeError):
            content_dups += 1
            continue
        if content_blob in seen_content:
            content_dups += 1
            continue
        seen_content.add(content_blob)
        final_entries.append(e)

    stats = {
        "category": cat_key,
        "source_exists": True,
        "raw_count": n_raw,
        "placeholder_count": n_placeholders,
        "real_count": len(real_entries),
        "unique_questions": len(unique_entries),
        "question_dups_dropped": duplicate_count,
        "content_dups_dropped": content_dups,
        "imported": len(final_entries),
        "source_path": str(src_path),
        "destination_path": str(dst_path),
    }

    if not dry_run and final_entries:
        _write_jsonl(dst_path, final_entries)

    if verbose:
        print(f"  {cat_key:20s} ({bucket_dir_rel})")
        print(f"    source raw:    {n_raw}")
        print(f"    placeholders: {n_placeholders}")
        print(f"    real entries: {len(real_entries)}")
        print(
            f"    unique Q:     {len(unique_entries)}  "
            f"(dropped {duplicate_count} Q-dups)"
        )
        print(
            f"    final unique: {len(final_entries)}  "
            f"(dropped {content_dups} content-dups)"
        )
        print(f"    → {dst_path}")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dedupe and import LLM-generated synthetic data into buckets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview dedupe stats without writing files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-category breakdown.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("AttackLM — Dedupe + Import LLM Synthetic Data into Buckets")
    print("=" * 70)
    print(f"Source:      {SYNTHETIC_DIR}")
    print(f"Destination: {LLM_SOURCE_DIR}/<bucket>/data_llm.jsonl")
    print(f"Mode:        {'DRY RUN' if args.dry_run else 'WRITE'}")
    print()

    grand_raw = 0
    grand_placeholders = 0
    grand_imported = 0
    grand_dropped = 0

    for cat_key, bucket_dir in CATEGORY_TO_BUCKET_DIR.items():
        stats = process_category(
            cat_key,
            bucket_dir,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        if not stats.get("source_exists"):
            print(f"⏭️  {cat_key:20s} — no source file, skipping")
            continue
        if "skipped_reason" in stats:
            print(f"⏭️  {cat_key:20s} — {stats['skipped_reason']}")
            continue
        grand_raw += stats.get("raw_count", 0)
        grand_placeholders += stats.get("placeholder_count", 0)
        grand_imported += stats.get("imported", 0)
        grand_dropped += stats.get("question_dups_dropped", 0) + stats.get(
            "content_dups_dropped", 0
        )

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Raw entries read:    {grand_raw:>6}")
    print(f"  Placeholders dropped:{grand_placeholders:>6}")
    print(f"  Duplicates dropped:  {grand_dropped:>6}")
    print(f"  Imported into buckets: {grand_imported:>6}")
    print()

    if args.dry_run:
        print("DRY RUN — no files written. Re-run without --dry-run to commit.")
        return 0

    print("Done. Re-run scripts/stamp_and_reorg.py to add provenance fields,")
    print("then scripts/rebuild_manifest.py to update manifest.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
