#!/usr/bin/env python3
# CREDITS — DATA SOURCE ATTRIBUTION
# ----------------------------------
# This script transforms data from: trendmicro-ailab/Primus-Seed
# Repository: https://huggingface.co/datasets/trendmicro-ailab/Primus-Seed
# License:    ODC-By (Open Data Commons Attribution License)
# Paper:      https://arxiv.org/abs/2502.11191
#
# The output JSONL is a *transformation* of upstream Primus-Seed documents
# into OpenAI-style chat triples. See /ATTRIBUTION.md for full per-source
# attribution and re-distribution guidance.
# ----------------------------------
"""Deterministic extraction of PRIMUS-Seed cybersecurity documents into AttackLM JSONL training pairs.

Downloads the PRIMUS-Seed dataset from HuggingFace and generates instruction-response
pairs from raw text documents. For each document, it creates one or more pairs
depending on document length:

- Short documents (< chunk_size words): one pair with the full text as response
- Long documents: chunked into segments, one pair per chunk

MITRE ATT&CK technique IDs are extracted from document text via regex.

Output: ``data/datasets/buckets/sources/primus-seed/<category>/data.jsonl``

Usage:
    python scripts/extract_primus_seed.py
    python scripts/extract_primus_seed.py --limit 100 --chunk-size 500
    python scripts/extract_primus_seed.py --config mitre --dry-run
    python scripts/extract_primus_seed.py --limit 10 --output-dir /tmp/test_primus
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
    BASE_DIR / "data" / "datasets" / "buckets" / "sources" / "primus-seed"
)

# ---------------------------------------------------------------------------
# System message
# ---------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are a cybersecurity threat intelligence analyst. Analyze security documents, "
    "extract tactics, techniques, and procedures (TTPs), identify indicators of "
    "compromise (IOCs), and provide actionable security recommendations."
)

# ---------------------------------------------------------------------------
# Instruction templates by config/category
# ---------------------------------------------------------------------------
INSTRUCTION_TEMPLATES: dict[str, str] = {
    "mitre": (
        "Analyze the following MITRE ATT&CK content. Extract key techniques, "
        "tactics, and procedural details. Provide a structured summary of the "
        "adversary behaviors described."
    ),
    "cybersecurity_companies_websites": (
        "Analyze the following security documentation from a cybersecurity vendor. "
        "Extract key security concepts, threat indicators, detection guidance, "
        "and mitigation recommendations."
    ),
    "cybersecurity_wikis": (
        "Analyze the following cybersecurity encyclopedia article. Extract key "
        "threat concepts, attack techniques, defense strategies, and relevant "
        "MITRE ATT&CK references."
    ),
    "default": (
        "Analyze the following security document and extract key TTPs, "
        "indicators, and recommendations."
    ),
}

# ---------------------------------------------------------------------------
# Config → AttackLM category mapping
# ---------------------------------------------------------------------------
CONFIG_TO_CATEGORY: dict[str, str] = {
    "mitre": "mitre-attack",
    "cybersecurity_companies_websites": "vendor-documentation",
    "cybersecurity_wikis": "cybersecurity-wiki",
}

# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------
ATTRIBUTION = {
    "source": "primus-seed",
    "source_uri": "https://huggingface.co/datasets/trendmicro-ailab/Primus-Seed",
    "license": "ODC-By",
    "license_uri": "https://opendatacommons.org/licenses/by/1-0/",
    "rights_contact": "Trend Micro AI Lab",
    "attribution_text": (
        "ODC-By — Open Data Commons Attribution License. "
        "See https://opendatacommons.org/licenses/by/1-0/ for terms. "
        "Original data by Trend Micro AI Lab."
    ),
}

# ---------------------------------------------------------------------------
# MITRE ATT&CK technique ID extraction
# ---------------------------------------------------------------------------
_MITRE_RE = re.compile(r"\b(T\d{4}(?:\.\d{3})?)\b", re.IGNORECASE)


def extract_mitre_ids(text: str) -> list[str]:
    """Extract MITRE ATT&CK technique IDs from text.

    Returns sorted unique list of uppercase technique IDs (e.g. T1059.001).
    """
    matches = _MITRE_RE.findall(text)
    return sorted({m.upper() for m in matches})


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 1000) -> list[str]:
    """Split text into chunks of approximately *chunk_size* words.

    Tries to break at sentence boundaries (period + space) to keep
    context coherent within each chunk.

    Parameters
    ----------
    text:
        The document text to chunk.
    chunk_size:
        Target number of words per chunk.

    Returns
    -------
    list[str]
        List of text chunks.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))

        # Try to break at a sentence boundary within a tolerance window
        if end < len(words):
            # Look for sentence-ending punctuation within the last 15% of the chunk
            search_start = end - max(1, chunk_size // 7)
            best_break = end
            for i in range(end, max(search_start, start), -1):
                if i < len(words) and words[i - 1].rstrip().endswith((".", "!", "?")):
                    best_break = i
                    break
            end = best_break

        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end

    return chunks


# ---------------------------------------------------------------------------
# Build training pair
# ---------------------------------------------------------------------------
def build_pair(
    *,
    instruction: str,
    response: str,
    mitre_ids: list[str],
    category: str,
    chunk_index: int | None = None,
    total_chunks: int | None = None,
) -> dict[str, Any]:
    """Build one training pair from a document chunk.

    Parameters
    ----------
    instruction:
        The user prompt / instruction text.
    response:
        The assistant response (document content).
    mitre_ids:
        List of MITRE ATT&CK technique IDs found in the text.
    category:
        The source category (e.g. "mitre-attack").
    chunk_index:
        1-based chunk index if the document was split.
    total_chunks:
        Total number of chunks if the document was split.

    Returns
    -------
    dict
        OpenAI-style message triple with metadata.
    """
    # Build user content with chunk annotation if applicable
    user_content = instruction
    if chunk_index is not None and total_chunks is not None and total_chunks > 1:
        user_content += f" (Part {chunk_index}/{total_chunks})"

    # Build assistant response with MITRE technique summary if found
    assistant_content = response
    if mitre_ids:
        technique_lines = []
        for tid in mitre_ids:
            tactic_id = get_tactic_for_technique(tid)
            tactic_name = get_tactic_name(tactic_id) if tactic_id else "Unknown"
            technique_lines.append(f"- {tid} ({tactic_name})")
        mitre_summary = "\n\n**MITRE ATT&CK Techniques Detected:**\n" + "\n".join(
            technique_lines
        )
        assistant_content += mitre_summary

    pair: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "mitre_ids": mitre_ids,
        "category": category,
        **ATTRIBUTION,
    }

    # Add tactic info from first MITRE ID
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
# Load dataset from HuggingFace
# ---------------------------------------------------------------------------
def load_primus_seed(
    config: str | None = None,
    limit: int = 0,
    streaming: bool = True,
    local_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Load documents from PRIMUS-Seed dataset on HuggingFace.

    Parameters
    ----------
    config:
        Specific config to load. One of "mitre", "cybersecurity_companies_websites",
        "cybersecurity_wikis", or None for all configs.
    limit:
        Maximum number of documents to load per config (0 = all).
    streaming:
        If True, use streaming mode to avoid downloading the full dataset.
    local_dir:
        If set, load from local directory instead of HuggingFace.
        Expects a directory with .jsonl.gz or .jsonl files organized
        by config: <local_dir>/<config>/*.jsonl.gz

    Returns
    -------
    list[dict]
        List of raw document dicts from the dataset.
    """
    import gzip

    configs = (
        [config]
        if config
        else ["cybersecurity_companies_websites", "cybersecurity_wikis", "mitre"]
    )

    all_docs: list[dict[str, Any]] = []

    # --- Local directory mode ---
    if local_dir:
        local_path = Path(local_dir)
        if not local_path.exists():
            print(f"ERROR: Local directory not found: {local_path}", file=sys.stderr)
            return all_docs

        for cfg in configs:
            cfg_dir = local_path / cfg
            if not cfg_dir.exists():
                # Also check web_crawler_official_dump subdirectory
                cfg_dir = local_path / "web_crawler_official_dump" / cfg
            if not cfg_dir.exists():
                print(
                    f"  [WARN] Config directory not found: {cfg_dir}", file=sys.stderr
                )
                continue

            count = 0
            # Load .jsonl.gz files
            for jsonl_file in sorted(cfg_dir.glob("*.jsonl.gz")):
                print(f"  Loading: {jsonl_file.name} ...", file=sys.stderr)
                try:
                    with gzip.open(jsonl_file, "rt", encoding="utf-8") as f:
                        for line in f:
                            if limit > 0 and count >= limit:
                                break
                            try:
                                doc = json.loads(line.strip())
                            except json.JSONDecodeError:
                                continue
                            doc["_config"] = cfg
                            all_docs.append(doc)
                            count += 1
                except Exception as exc:
                    print(
                        f"  [WARN] Failed to load {jsonl_file}: {exc}", file=sys.stderr
                    )

            # Also try plain .jsonl files
            for jsonl_file in sorted(cfg_dir.glob("*.jsonl")):
                print(f"  Loading: {jsonl_file.name} ...", file=sys.stderr)
                try:
                    with open(jsonl_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if limit > 0 and count >= limit:
                                break
                            try:
                                doc = json.loads(line.strip())
                            except json.JSONDecodeError:
                                continue
                            doc["_config"] = cfg
                            all_docs.append(doc)
                            count += 1
                except Exception as exc:
                    print(
                        f"  [WARN] Failed to load {jsonl_file}: {exc}", file=sys.stderr
                    )

            print(f"    {cfg}: loaded {count} documents", file=sys.stderr)

        return all_docs

    # --- HuggingFace mode ---
    from datasets import load_dataset

    for cfg in configs:
        print(f"  Loading config: {cfg} ...", file=sys.stderr)
        try:
            ds = load_dataset(
                "trendmicro-ailab/Primus-Seed",
                cfg,
                split="train",
                streaming=streaming,
            )
        except Exception as exc:
            print(f"  [WARN] Failed to load config '{cfg}': {exc}", file=sys.stderr)
            print(
                "  Hint: This is a gated dataset. Run `huggingface-cli login` first "
                "and accept the license at https://huggingface.co/datasets/trendmicro-ailab/Primus-Seed",
                file=sys.stderr,
            )
            continue

        count = 0
        for doc in ds:
            doc["_config"] = cfg
            all_docs.append(doc)
            count += 1
            if limit > 0 and count >= limit:
                break

        print(f"    {cfg}: loaded {count} documents", file=sys.stderr)

    return all_docs


# ---------------------------------------------------------------------------
# Detect text field name in a document
# ---------------------------------------------------------------------------
def _detect_text_field(doc: dict[str, Any]) -> str | None:
    """Detect the field containing document text.

    PRIMUS-Seed may use 'text', 'content', 'document', 'body', or similar
    field names. We try common candidates in priority order.
    """
    candidates = ["text", "content", "document", "body", "passage", "raw_text", "input"]
    for field in candidates:
        val = doc.get(field)
        if isinstance(val, str) and len(val.strip()) > 20:
            return field
    # Fallback: find the longest string field
    best_field = None
    best_len = 0
    for key, val in doc.items():
        if isinstance(val, str) and len(val) > best_len:
            best_field = key
            best_len = len(val)
    return best_field


# ---------------------------------------------------------------------------
# Detect category field
# ---------------------------------------------------------------------------
def _detect_category_field(doc: dict[str, Any]) -> str | None:
    """Detect a category or type field in the document."""
    candidates = ["category", "type", "source", "domain", "label", "class"]
    for field in candidates:
        if field in doc and isinstance(doc[field], str) and doc[field].strip():
            return field
    return None


# ---------------------------------------------------------------------------
# Process documents into training pairs
# ---------------------------------------------------------------------------
def process_documents(
    docs: list[dict[str, Any]],
    chunk_size: int = 1000,
    max_pairs: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Process raw documents into training pairs.

    Parameters
    ----------
    docs:
        Raw documents from load_primus_seed.
    chunk_size:
        Target number of words per chunk for long documents.
    max_pairs:
        Maximum total pairs to generate (0 = unlimited).

    Returns
    -------
    tuple[list[dict], dict[str, int]]
        (all_pairs, stats) where stats maps category → pair count.
    """
    all_pairs: list[dict[str, Any]] = []
    stats: dict[str, int] = {}

    # Auto-detect field names from first document
    text_field = None
    category_field = None
    if docs:
        text_field = _detect_text_field(docs[0])
        category_field = _detect_category_field(docs[0])
        print(f"  Detected text field: {text_field}", file=sys.stderr)
        print(f"  Detected category field: {category_field}", file=sys.stderr)

    for i, doc in enumerate(docs):
        if max_pairs > 0 and len(all_pairs) >= max_pairs:
            break

        config = doc.get("_config", "unknown")
        category = CONFIG_TO_CATEGORY.get(config, config)

        # Override category if document has its own category field
        if category_field and category_field in doc:
            raw_cat = str(doc[category_field]).strip().lower()
            # Normalize category for filesystem
            category = raw_cat.replace(" ", "-").replace("/", "-")

        # Extract text
        text = ""
        if text_field and text_field in doc:
            text = str(doc[text_field]).strip()
        else:
            # Fallback: concatenate all string fields
            parts = []
            for key, val in doc.items():
                if key.startswith("_") or key == text_field:
                    continue
                if isinstance(val, str) and len(val.strip()) > 10:
                    parts.append(val.strip())
            text = "\n\n".join(parts)

        if not text or len(text) < 50:
            continue

        # Extract MITRE IDs from full text before chunking
        mitre_ids = extract_mitre_ids(text)

        # Get instruction template
        instruction = INSTRUCTION_TEMPLATES.get(
            config, INSTRUCTION_TEMPLATES["default"]
        )

        # Chunk if needed
        chunks = chunk_text(text, chunk_size)
        total_chunks = len(chunks)

        for j, chunk in enumerate(chunks):
            if max_pairs > 0 and len(all_pairs) >= max_pairs:
                break

            pair = build_pair(
                instruction=instruction,
                response=chunk,
                mitre_ids=mitre_ids
                if j == 0
                else [],  # Only attach MITRE IDs to first chunk
                category=category,
                chunk_index=j + 1 if total_chunks > 1 else None,
                total_chunks=total_chunks if total_chunks > 1 else None,
            )
            all_pairs.append(pair)
            stats[category] = stats.get(category, 0) + 1

        if (i + 1) % 10000 == 0:
            print(
                f"  Processed {i + 1}/{len(docs)} documents, "
                f"{len(all_pairs)} pairs generated",
                file=sys.stderr,
            )

    return all_pairs, stats


# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
def write_output(
    pairs: list[dict[str, Any]],
    output_dir: Path,
    stats: dict[str, int],
) -> dict[str, int]:
    """Write training pairs to per-category JSONL files.

    Creates the directory structure:
        <output_dir>/<category>/data.jsonl

    Returns dict mapping category → number of pairs written.
    """
    # Group pairs by category
    by_category: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        cat = pair.get("category", "unknown")
        by_category.setdefault(cat, []).append(pair)

    written: dict[str, int] = {}
    for category, cat_pairs in sorted(by_category.items()):
        cat_dir = output_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        output_path = cat_dir / "data.jsonl"

        with open(output_path, "w", encoding="utf-8") as f:
            for pair in cat_pairs:
                f.write(json.dumps(pair) + "\n")

        written[category] = len(cat_pairs)
        print(f"  {category}: {len(cat_pairs)} pairs → {output_path}", file=sys.stderr)

    # Write SOURCE.md
    source_md = output_dir / "SOURCE.md"
    total_pairs = sum(written.values())
    category_lines = "\n".join(
        f"- **{cat}**: {count} pairs" for cat, count in sorted(written.items())
    )
    source_content = f"""# Source: trendmicro-ailab/Primus-Seed

- **Repository**: https://huggingface.co/datasets/trendmicro-ailab/Primus-Seed
- **License**: ODC-By (Open Data Commons Attribution License)
- **License URI**: https://opendatacommons.org/licenses/by/1-0/
- **Rights Contact**: Trend Micro AI Lab
- **Extraction Date**: {__import__("datetime").date.today().isoformat()}
- **Extractor Script**: scripts/extract_primus_seed.py
- **Records Extracted**: ~{total_pairs}
- **Description**: PRIMUS-Seed is a high-quality cybersecurity text dataset composed of data crawled from MITRE, Wikipedia, and cybersecurity company websites, as well as CTI manually collected by threat experts. Contains 674K documents across 3 categories (currently released: cybersecurity_companies_websites, cybersecurity_wikis, mitre).

## Categories

{category_lines}
"""
    source_md.write_text(source_content, encoding="utf-8")
    print(f"  SOURCE.md → {source_md}", file=sys.stderr)

    # Write LICENSE.md
    license_md = output_dir / "LICENSE.md"
    license_content = """# ODC-By — Open Data Commons Attribution License

This dataset is derived from [PRIMUS-Seed](https://huggingface.co/datasets/trendmicro-ailab/Primus-Seed)
by Trend Micro AI Lab, licensed under the [Open Data Commons Attribution License (ODC-By) v1.0](https://opendatacommons.org/licenses/by/1-0/).

## Key Requirements

- **Attribution**: You must give appropriate credit to Trend Micro AI Lab, provide a link to the ODC-By license, and indicate if changes were made.
- **ShareAlike**: No additional restrictions — you may not apply legal terms or technological measures that restrict others from doing anything the license permits.

## Full License Text

See https://opendatacommons.org/licenses/by/1-0/ for the complete license text.

## Attribution

When using this data, please cite:

```bibtex
@misc{yu2025primus,
      title={PRIMUS: A Pioneering Collection of Open-Source Datasets for Cybersecurity LLM Training},
      author={Yao-Ching Yu and Tsun-Han Chiang and Cheng-Wei Tsai and Chien-Ming Huang and Wen-Kwang Tsao},
      year={2025},
      eprint={2502.11191},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```
"""
    license_md.write_text(license_content, encoding="utf-8")
    print(f"  LICENSE.md → {license_md}", file=sys.stderr)

    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract PRIMUS-Seed cybersecurity documents into AttackLM training pairs",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        choices=["mitre", "cybersecurity_companies_websites", "cybersecurity_wikis"],
        help="Only process one config (default: all 3 configs)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of documents to process per config (0=all)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Target number of words per chunk for long documents (default: 1000)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Maximum total pairs to generate (0=unlimited)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/datasets/buckets/sources/primus-seed/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without writing files",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Download full dataset instead of streaming (uses more disk but faster iteration)",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Load from local directory instead of HuggingFace. "
        "Expects <dir>/<config>/*.jsonl.gz or *.jsonl files. "
        "Useful for offline or pre-authenticated scenarios.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR

    source = (
        f"local: {args.local_dir}"
        if args.local_dir
        else "trendmicro-ailab/Primus-Seed (HuggingFace)"
    )
    print("=" * 60, file=sys.stderr)
    print("AttackLM — Extract PRIMUS-Seed Dataset", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Source:      {source}", file=sys.stderr)
    print(f"  Config:      {args.config or 'all'}", file=sys.stderr)
    print(f"  Limit:       {args.limit or 'unlimited'}", file=sys.stderr)
    print(f"  Chunk size:  {args.chunk_size} words", file=sys.stderr)
    print(f"  Output:      {output_dir}", file=sys.stderr)
    print(file=sys.stderr)

    # Load dataset
    print("Loading dataset...", file=sys.stderr)
    docs = load_primus_seed(
        config=args.config,
        limit=args.limit,
        streaming=not args.no_streaming,
        local_dir=args.local_dir,
    )

    if not docs:
        print("ERROR: No documents loaded.", file=sys.stderr)
        if not args.local_dir:
            print("  This is a gated dataset. Please:", file=sys.stderr)
            print("  1. Run: huggingface-cli login", file=sys.stderr)
            print(
                "  2. Accept the license at: https://huggingface.co/datasets/trendmicro-ailab/Primus-Seed",
                file=sys.stderr,
            )
            print("  3. Re-run this script", file=sys.stderr)
            print(
                "  Or use --local-dir to load from pre-downloaded data.",
                file=sys.stderr,
            )
        else:
            print(
                f"  Check that the local directory contains .jsonl.gz or .jsonl files.",
                file=sys.stderr,
            )
        return 1

    print(f"\nLoaded {len(docs)} documents. Processing...", file=sys.stderr)

    # Process documents
    pairs, stats = process_documents(
        docs,
        chunk_size=args.chunk_size,
        max_pairs=args.max_pairs,
    )

    print(f"\nProcessing complete:", file=sys.stderr)
    print(f"  Documents processed: {len(docs)}", file=sys.stderr)
    print(f"  Training pairs generated: {len(pairs)}", file=sys.stderr)
    print(f"  Categories:", file=sys.stderr)
    for cat, count in sorted(stats.items()):
        print(f"    {cat}: {count}", file=sys.stderr)

    # Count MITRE-tagged pairs
    mitre_tagged = sum(1 for p in pairs if p.get("mitre_ids"))
    print(f"  Pairs with MITRE ATT&CK tags: {mitre_tagged}", file=sys.stderr)

    if args.dry_run:
        print(f"\nDRY RUN — no files written.", file=sys.stderr)
        # Print a sample pair
        if pairs:
            print(f"\nSample pair:", file=sys.stderr)
            print(json.dumps(pairs[0], indent=2)[:2000], file=sys.stderr)
        return 0

    # Write output
    print(f"\nWriting output...", file=sys.stderr)
    written = write_output(pairs, output_dir, stats)

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Extraction Complete", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"  Total pairs written: {sum(written.values())}", file=sys.stderr)
    for cat, count in sorted(written.items()):
        print(f"    {cat}: {count}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
