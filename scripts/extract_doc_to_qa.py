#!/usr/bin/env python3
# CREDITS — DATA SOURCE ATTRIBUTION
# ----------------------------------
# This script generates Q&A training pairs from unstructured security
# documents using the Augmentoolkit pattern (chunk → question → verify → answer).
#
# Input documents are user-provided; attribution is configured per-source
# via CLI flags (--source-name, --source-uri, --license).
#
# Pipeline:
#   1. Load document (.txt, .md, .pdf via pymupdf if available)
#   2. Chunk into configurable word-count segments with overlap
#   3. For each chunk: generate question → verify answerability → generate answer
#   4. Write verified Q&A pairs to JSONL
#
# Output: ``data/datasets/buckets/sources/doc-qa/<bucket>/<tactic>/data.jsonl``
#
# Usage:
#     python scripts/extract_doc_to_qa.py --input docs/nist_sp800-63b.txt
#     python scripts/extract_doc_to_qa.py --input docs/ --limit 50 --dry-run
#     python scripts/extract_doc_to_qa.py --input advisory.pdf --model my-model --api-url http://localhost:1234/v1
# ----------------------------------
"""Convert unstructured security documents into Q&A training pairs.

Uses the Augmentoolkit pattern: chunk → generate question → verify
answerability → generate answer. Designed for NIST bulletins, CISA
advisories, CERT alerts, security vendor reports, and academic papers.

Requires a local or remote OpenAI-compatible LLM endpoint for generation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional PDF support
# ---------------------------------------------------------------------------
try:
    import pymupdf  # type: ignore[import-untyped]

    HAS_PYMUPDF = True
except ImportError:
    try:
        import fitz as pymupdf  # type: ignore[import-untyped]  # noqa: F401

        HAS_PYMUPDF = True
    except ImportError:
        HAS_PYMUPDF = False

# ---------------------------------------------------------------------------
# Optional OpenAI client
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

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
    / "doc-qa"
    / "defensive"
    / "knowledge_base"
)

# ---------------------------------------------------------------------------
# System message
# ---------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are a cybersecurity knowledge specialist. You provide accurate, "
    "technical answers about security concepts, threats, mitigations, and "
    "best practices based on authoritative source material."
)

# ---------------------------------------------------------------------------
# Prompt templates (Augmentoolkit pattern)
# ---------------------------------------------------------------------------
QUESTION_PROMPT = """Based on the following security document excerpt, create a specific, technical question that tests understanding of the content. The question should be answerable from the text alone.

Document: {chunk}

Question:"""

VERIFY_PROMPT = """Can the following question be fully answered using only the information in the document excerpt? Answer YES or NO.

Document: {chunk}
Question: {question}

Answerable (YES/NO):"""

ANSWER_PROMPT = """Answer the following question based on the document excerpt. Be specific, technical, and include relevant details from the text.

Document: {chunk}
Question: {question}

Answer:"""

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "qwen2.5-coder-14b-instruct-uncensored"
DEFAULT_API_URL = "http://localhost:1234/v1"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2.0


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------
def load_document(filepath: Path) -> str:
    """Load a document from file, supporting .txt, .md, and .pdf.

    Parameters
    ----------
    filepath:
        Path to the document file.

    Returns
    -------
    str
        The full text content of the document.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is unsupported.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = filepath.suffix.lower()

    if suffix in (".txt", ".md", ".rst"):
        return filepath.read_text(encoding="utf-8", errors="replace")

    if suffix == ".pdf":
        if not HAS_PYMUPDF:
            raise ValueError(
                "PDF support requires pymupdf. Install with: pip install pymupdf"
            )
        doc = pymupdf.open(str(filepath))
        pages: list[str] = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n\n".join(pages)

    raise ValueError(
        f"Unsupported file format: {suffix}. Supported: .txt, .md, .rst, .pdf"
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into word-count-based chunks with overlap.

    Parameters
    ----------
    text:
        The full document text.
    chunk_size:
        Number of words per chunk (default: 1000).
    overlap:
        Number of words to overlap between chunks (default: 100).

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
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += chunk_size - overlap

    return chunks


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------
def create_client(api_url: str, model: str) -> tuple[Any, str]:
    """Create an OpenAI-compatible client.

    Parameters
    ----------
    api_url:
        Base URL for the API (e.g. ``http://localhost:1234/v1``).
    model:
        Model name to use for generation.

    Returns
    -------
    tuple[Any, str]
        A 2-tuple of ``(client, model_name)``.

    Raises
    ------
    ImportError
        If the ``openai`` package is not installed.
    """
    if not HAS_OPENAI:
        raise ImportError(
            "The 'openai' package is required for LLM-based extraction. "
            "Install with: pip install openai"
        )

    client = OpenAI(base_url=api_url, api_key="not-needed")
    return client, model


def llm_generate(
    client: Any,
    model: str,
    prompt: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> str:
    """Generate text from the LLM with retry logic.

    Parameters
    ----------
    client:
        OpenAI client instance.
    model:
        Model name.
    prompt:
        The prompt to send.
    max_retries:
        Maximum number of retries on failure.

    Returns
    -------
    str
        The generated text, stripped of whitespace.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048,
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = RETRY_DELAY_SECONDS * (attempt + 1)
                print(
                    f"    [LLM retry {attempt + 1}/{max_retries}] {exc}",
                    file=sys.stderr,
                )
                time.sleep(wait)
            else:
                print(
                    f"    [LLM failed after {max_retries} retries] {exc}",
                    file=sys.stderr,
                )
                return ""


# ---------------------------------------------------------------------------
# Q&A generation pipeline
# ---------------------------------------------------------------------------
def generate_question(client: Any, model: str, chunk: str) -> str:
    """Generate a question from a document chunk.

    Parameters
    ----------
    client:
        OpenAI client instance.
    model:
        Model name.
    chunk:
        The document chunk text.

    Returns
    -------
    str
        A generated question, or empty string on failure.
    """
    prompt = QUESTION_PROMPT.format(chunk=chunk)
    return llm_generate(client, model, prompt)


def verify_question(client: Any, model: str, chunk: str, question: str) -> bool:
    """Verify that a question can be answered from the chunk.

    Parameters
    ----------
    client:
        OpenAI client instance.
    model:
        Model name.
    chunk:
        The document chunk text.
    question:
        The question to verify.

    Returns
    -------
    bool
        ``True`` if the LLM answers YES, ``False`` otherwise.
    """
    prompt = VERIFY_PROMPT.format(chunk=chunk, question=question)
    response = llm_generate(client, model, prompt)
    if not response:
        return False
    # Parse YES/NO from response — take first word or first line
    first_line = response.strip().split("\n")[0].strip().upper()
    first_word = first_line.split()[0] if first_line.split() else ""
    return first_word == "YES"


def generate_answer(client: Any, model: str, chunk: str, question: str) -> str:
    """Generate an answer to a question based on a document chunk.

    Parameters
    ----------
    client:
        OpenAI client instance.
    model:
        Model name.
    chunk:
        The document chunk text.
    question:
        The question to answer.

    Returns
    -------
    str
        The generated answer.
    """
    prompt = ANSWER_PROMPT.format(chunk=chunk, question=question)
    return llm_generate(client, model, prompt)


def process_chunk(
    client: Any,
    model: str,
    chunk: str,
    attribution: dict[str, str],
    source_file: str,
    chunk_index: int,
) -> dict[str, Any] | None:
    """Process a single chunk through the full Q&A pipeline.

    Pipeline: generate question → verify → generate answer.

    Parameters
    ----------
    client:
        OpenAI client instance.
    model:
        Model name.
    chunk:
        The document chunk text.
    attribution:
        Attribution dict with source, source_uri, license, etc.
    source_file:
        Name of the source file.
    chunk_index:
        Index of the chunk (0-based).

    Returns
    -------
    dict[str, Any] | None
        A training pair dict, or ``None`` if verification failed.
    """
    # Step 1: Generate question
    question = generate_question(client, model, chunk)
    if not question:
        print(
            f"    [chunk {chunk_index}] Failed to generate question, skipping",
            file=sys.stderr,
        )
        return None

    # Step 2: Verify question is answerable
    is_answerable = verify_question(client, model, chunk, question)
    if not is_answerable:
        print(
            f"    [chunk {chunk_index}] Question not answerable from chunk, skipping",
            file=sys.stderr,
        )
        return None

    # Step 3: Generate answer
    answer = generate_answer(client, model, chunk, question)
    if not answer:
        print(
            f"    [chunk {chunk_index}] Failed to generate answer, skipping",
            file=sys.stderr,
        )
        return None

    pair: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "mitre_ids": [],
        "source_file": source_file,
        "chunk_index": chunk_index,
        **attribution,
    }
    return pair


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def discover_files(input_path: Path) -> list[Path]:
    """Discover document files from a path (file or directory).

    Parameters
    ----------
    input_path:
        Path to a single file or directory.

    Returns
    -------
    list[Path]
        Sorted list of document file paths.
    """
    supported_extensions = {".txt", ".md", ".rst", ".pdf"}

    if input_path.is_file():
        if input_path.suffix.lower() in supported_extensions:
            return [input_path]
        print(f"  Skipping unsupported file: {input_path}", file=sys.stderr)
        return []

    if input_path.is_dir():
        files: list[Path] = []
        for ext in supported_extensions:
            files.extend(sorted(input_path.rglob(f"*{ext}")))
        return files

    print(f"  Path not found: {input_path}", file=sys.stderr)
    return []


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def process_document(
    client: Any,
    model: str,
    filepath: Path,
    chunk_size: int,
    chunk_overlap: int,
    attribution: dict[str, str],
    limit: int,
) -> list[dict[str, Any]]:
    """Process a single document through the Q&A pipeline.

    Parameters
    ----------
    client:
        OpenAI client instance.
    model:
        Model name.
    filepath:
        Path to the document file.
    chunk_size:
        Words per chunk.
    chunk_overlap:
        Overlap words between chunks.
    attribution:
        Attribution dict.
    limit:
        Maximum number of Q&A pairs (0 = unlimited).

    Returns
    -------
    list[dict[str, Any]]
        List of training pairs.
    """
    print(f"\n  Processing: {filepath.name}")

    # Load document
    try:
        text = load_document(filepath)
    except (FileNotFoundError, ValueError) as exc:
        print(f"    ERROR: {exc}", file=sys.stderr)
        return []

    if not text.strip():
        print("    Empty document, skipping", file=sys.stderr)
        return []

    # Chunk
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
    print(f"    {len(chunks)} chunks ({chunk_size} words, {chunk_overlap} overlap)")

    # Process each chunk
    pairs: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        if limit > 0 and len(pairs) >= limit:
            break

        pair = process_chunk(
            client=client,
            model=model,
            chunk=chunk,
            attribution=attribution,
            source_file=filepath.name,
            chunk_index=idx,
        )
        if pair is not None:
            pairs.append(pair)
            print(f"    [chunk {idx}] ✓ Q&A pair generated", file=sys.stderr)
        else:
            print(f"    [chunk {idx}] ✗ Skipped", file=sys.stderr)

    return pairs


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_output(pairs: list[dict[str, Any]], output_path: Path) -> int:
    """Write training pairs to JSONL file. Returns count written.

    Parameters
    ----------
    pairs:
        List of training pair dicts.
    output_path:
        Path to the output JSONL file.

    Returns
    -------
    int
        Number of pairs written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    return len(pairs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert unstructured security documents into Q&A training pairs "
        "using the Augmentoolkit pattern (chunk → question → verify → answer).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Process a single text file
  python scripts/extract_doc_to_qa.py --input docs/advisory.txt

  # Process all documents in a directory
  python scripts/extract_doc_to_qa.py --input docs/

  # Dry run with custom model and API
  python scripts/extract_doc_to_qa.py --input docs/ --dry-run --model my-model

  # Custom chunk size and overlap
  python scripts/extract_doc_to_qa.py --input report.pdf --chunk-size 500 --chunk-overlap 50

  # Custom attribution
  python scripts/extract_doc_to_qa.py --input bulletin.txt --source-name "CISA Advisory" --source-uri "https://..." --license "CC-BY-4.0"
""",
    )

    # Input/Output
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a single document file or directory of documents",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for JSONL files (default: data/datasets/buckets/sources/doc-qa/...)",
    )

    # Attribution
    parser.add_argument(
        "--source-name",
        type=str,
        default="",
        help="Source name for attribution (default: derived from filename)",
    )
    parser.add_argument(
        "--source-uri",
        type=str,
        default="",
        help="Source URI for attribution",
    )
    parser.add_argument(
        "--license",
        type=str,
        default="Unknown",
        help="License for attribution (default: Unknown)",
    )
    parser.add_argument(
        "--attribution-text",
        type=str,
        default="",
        help="Full attribution text for records",
    )

    # Chunking
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Words per chunk (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Overlap words between chunks (default: {DEFAULT_CHUNK_OVERLAP})",
    )

    # LLM configuration
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_API_URL,
        help=f"OpenAI-compatible API URL (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Max LLM API retries per call (default: {DEFAULT_MAX_RETRIES})",
    )

    # Pipeline control
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of Q&A pairs to generate (0 = unlimited)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process documents and print stats without writing files",
    )

    # Bucket/tactic placement
    parser.add_argument(
        "--bucket",
        type=str,
        default="defensive",
        help="Bucket category (default: defensive)",
    )
    parser.add_argument(
        "--tactic",
        type=str,
        default="knowledge_base",
        help="MITRE tactic subdirectory (default: knowledge_base)",
    )

    args = parser.parse_args()

    # Validate dependencies
    if not HAS_OPENAI:
        print(
            "ERROR: The 'openai' package is required. Install with: pip install openai",
            file=sys.stderr,
        )
        return 1

    # Resolve paths
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"ERROR: Input path does not exist: {input_path}", file=sys.stderr)
        return 1

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            BASE_DIR
            / "data"
            / "datasets"
            / "buckets"
            / "sources"
            / "doc-qa"
            / args.bucket
            / args.tactic
        )

    # Build attribution
    source_name = args.source_name or input_path.stem
    attribution: dict[str, str] = {
        "source": source_name,
        "source_uri": args.source_uri,
        "license": args.license,
        "attribution_text": args.attribution_text or f"Source: {source_name}",
    }

    # Discover files
    files = discover_files(input_path)
    if not files:
        print("No document files found.", file=sys.stderr)
        return 1

    print("AttackLM — Doc-to-QA Extractor (Augmentoolkit Pattern)")
    print(f"  Input:       {input_path}")
    print(f"  Files:       {len(files)}")
    print(f"  Model:       {args.model}")
    print(f"  API URL:     {args.api_url}")
    print(f"  Chunk size:  {args.chunk_size} words")
    print(f"  Overlap:     {args.chunk_overlap} words")
    print(f"  Source:      {source_name}")
    print(f"  License:     {args.license}")
    print(f"  Output dir:  {output_dir}")

    # Create LLM client
    client, model = create_client(args.api_url, args.model)

    # Process all documents
    all_pairs: list[dict[str, Any]] = []
    for filepath in files:
        if args.limit > 0 and len(all_pairs) >= args.limit:
            break

        pairs = process_document(
            client=client,
            model=model,
            filepath=filepath,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            attribution=attribution,
            limit=args.limit - len(all_pairs) if args.limit > 0 else 0,
        )
        all_pairs.extend(pairs)

    # Stats
    total = len(all_pairs)
    verified_count = sum(1 for p in all_pairs if p is not None)
    print(f"\n{'=' * 60}")
    print("  Extraction Complete")
    print(f"{'=' * 60}")
    print(f"  Total Q&A pairs: {total}")
    print(f"  Verified pairs:  {verified_count}")
    print(f"  Source files:    {len(files)}")

    if args.dry_run:
        print(f"\n  DRY RUN — No files written.")
        # Print sample pairs
        for pair in all_pairs[:3]:
            question = pair["messages"][1]["content"][:100]
            answer_preview = pair["messages"][2]["content"][:100]
            print(f"    Q: {question}...")
            print(f"    A: {answer_preview}...")
            print()
        if len(all_pairs) > 3:
            print(f"    ... and {len(all_pairs) - 3} more pairs")
        return 0

    # Write output
    output_path = output_dir / "data.jsonl"
    written = write_output(all_pairs, output_path)
    print(f"\n  Wrote {written} pairs to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
