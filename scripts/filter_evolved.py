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
    7. Judge-and-Revise (optional) — uses an LLM to evaluate each pair on four
       dimensions: factual accuracy, completeness, security relevance, and
       clarity. Pairs scoring below ``--judge-threshold`` are discarded.

Judge-and-Revise:
    When ``--judge-model`` is specified, pairs that pass all rule-based filters
    are additionally evaluated by an LLM judge. The judge scores each pair on:
    - Factual Accuracy (0-5): Are technical details correct?
    - Completeness (0-5): Does the response fully address the instruction?
    - Security Relevance (0-5): Is this useful for security training?
    - Clarity (0-5): Is the response well-structured and clear?
    The overall_score (0.0-1.0) must meet ``--judge-threshold`` (default 0.7).
    Results are cached to avoid re-judging on subsequent runs.

CLI interface:
    # Basic filtering (rule-based only)
    python scripts/filter_evolved.py \\
        --input data/datasets/evolved/metasploit-framework_multi_turn_abc123.jsonl \\
        --original data/datasets/buckets/sources/metasploit-framework/

    # With Judge-and-Revise (LLM quality filtering)
    python scripts/filter_evolved.py \\
        --input data/datasets/evolved/ --all \\
        --judge-model qwen2.5-coder-7b \\
        --judge-threshold 0.7 \\
        --judge-cache ~/.cache/attacklm/judge_cache.json

    # With remote API judge
    python scripts/filter_evolved.py \\
        --input data/datasets/evolved/ --all \\
        --judge-model gpt-4o-mini \\
        --judge-api-url https://api.openai.com/v1/chat/completions \\
        --judge-api-key sk-... \\
        --judge-max-pairs 500

    python scripts/filter_evolved.py --input data/datasets/evolved/ --dry-run

Output:
    Writes filtered JSONL to same directory with ``_filtered`` suffix.
    Prints report: total pairs, passed, failed, failure reasons, judge stats.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
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

# Default judge settings
DEFAULT_JUDGE_THRESHOLD = 0.7
DEFAULT_JUDGE_API_URL = "http://localhost:1234/v1/chat/completions"
DEFAULT_JUDGE_CACHE = Path.home() / ".cache" / "attacklm" / "judge_cache.json"

# Judge prompt template — evaluates training pair quality on 4 dimensions
JUDGE_PROMPT_TEMPLATE = """\
You are a quality judge for cybersecurity training data. Evaluate the following \
training pair on four criteria. Be strict and objective.

**Instruction (user message):**
{instruction}

**Response (assistant message):**
{response}

Score each criterion on a 0–5 integer scale:
- **Factual Accuracy (0-5):** Are technical details correct? Any wrong tool names, \
techniques, commands, or MITRE references?
- **Completeness (0-5):** Does the response fully address the instruction? Are \
important steps missing?
- **Security Relevance (0-5):** Is this useful for security training (offensive or \
defensive)? Or is it generic filler?
- **Clarity (0-5):** Is the response well-structured, concise, and easy to follow?

Respond with JSON ONLY — no markdown, no commentary:
{{"factual_accuracy": <0-5>, "completeness": <0-5>, "security_relevance": <0-5>, \
"clarity": <0-5>, "overall_score": <0.0-1.0>, "pass": <true|false>}}"""

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
    judge_model: str | None = None,
    judge_api_url: str = DEFAULT_JUDGE_API_URL,
    judge_api_key: str | None = None,
    judge_threshold: float = DEFAULT_JUDGE_THRESHOLD,
    judge_max_pairs: int | None = None,
    judge_cache: Path | None = None,
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

    # Check 7: Judge-and-Revise (optional LLM quality filtering)
    judge_stats: dict[str, Any] = {
        "judge_evaluated": 0,
        "judge_passed": 0,
        "judge_failed": 0,
        "judge_threshold": judge_threshold,
        "judge_score_distribution": {},
        "judge_errors": 0,
        "judge_cache_hits": 0,
    }
    if judge_model is not None:
        passed, judge_stats = judge_pairs(
            passed,
            model=judge_model,
            api_url=judge_api_url,
            api_key=judge_api_key,
            threshold=judge_threshold,
            max_pairs=judge_max_pairs,
            cache_path=judge_cache,
        )

    report = {
        "file": str(evolved_path),
        "total": len(evolved_records),
        "passed": len(passed),
        "failed": len(failed),
        "skipped_no_original": skipped_no_original,
        "failure_reasons": dict(reason_counter.most_common()),
        "duplicates_removed": duplicates_removed,
        "judge_evaluated": judge_stats["judge_evaluated"],
        "judge_passed": judge_stats["judge_passed"],
        "judge_failed": judge_stats["judge_failed"],
        "judge_threshold": judge_stats["judge_threshold"],
        "judge_score_distribution": judge_stats["judge_score_distribution"],
        "judge_errors": judge_stats["judge_errors"],
        "judge_cache_hits": judge_stats["judge_cache_hits"],
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


# ---------------------------------------------------------------------------
# Judge-and-Revise: LLM-based quality filtering
# ---------------------------------------------------------------------------


def _judge_cache_key(record: dict) -> str:
    """Compute a deterministic cache key from a record's content.

    Uses SHA-256 of the JSON-serialized messages for stable hashing.
    """
    messages = record.get("messages", [])
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_judge_cache(cache_path: Path) -> dict[str, dict]:
    """Load judge result cache from disk.

    Returns dict mapping cache keys to judge result dicts.
    Returns empty dict if cache doesn't exist or is malformed.
    """
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _save_judge_cache(cache: dict[str, dict], cache_path: Path) -> None:
    """Persist judge result cache to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False, indent=2)


def _extract_pair_texts(record: dict) -> tuple[str, str]:
    """Extract the instruction (first user msg) and response (first assistant msg).

    Returns (instruction, response) strings. Falls back to empty strings if
    the expected roles are not found.
    """
    instruction = ""
    response = ""
    for msg in record.get("messages", []):
        if msg.get("role") == "user" and not instruction:
            instruction = msg.get("content", "")
        if msg.get("role") == "assistant" and not response:
            response = msg.get("content", "")
    return instruction, response


def _call_judge_api(
    prompt: str,
    model: str,
    api_url: str,
    api_key: str | None = None,
    timeout: int = 60,
) -> str:
    """Call an OpenAI-compatible chat completions API with the judge prompt.

    Returns the raw response content string.
    Raises urllib.error.URLError on network failures.
    """
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 512,
            "stream": False,
        }
    ).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(api_url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp_data = json.loads(resp.read().decode("utf-8"))

    # Extract content from OpenAI-compatible response format
    choices = resp_data.get("choices", [])
    if not choices:
        return ""
    content = choices[0].get("message", {}).get("content", "")
    return content


def _parse_judge_response(raw: str) -> dict | None:
    """Parse the judge model's JSON response.

    Returns the parsed dict or None if parsing fails.
    Tries to extract JSON from markdown code blocks if direct parse fails.
    """
    # Try direct parse first
    try:
        result = json.loads(raw.strip())
        if isinstance(result, dict) and "overall_score" in result:
            return result
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(1).strip())
            if isinstance(result, dict) and "overall_score" in result:
                return result
        except json.JSONDecodeError:
            pass

    # Try finding any JSON object in the response
    brace_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if brace_match:
        try:
            result = json.loads(brace_match.group(0))
            if isinstance(result, dict) and "overall_score" in result:
                return result
        except json.JSONDecodeError:
            pass

    return None


def judge_single_pair(
    record: dict,
    model: str,
    api_url: str,
    api_key: str | None = None,
    cache: dict[str, dict] | None = None,
    timeout: int = 60,
) -> dict:
    """Judge a single training pair using the LLM.

    Returns a dict with keys:
        - overall_score (float): 0.0-1.0
        - pass (bool): whether overall_score >= threshold
        - factual_accuracy (int): 0-5
        - completeness (int): 0-5
        - security_relevance (int): 0-5
        - clarity (int): 0-5
        - cached (bool): whether the result was from cache
        - error (str|None): error message if judge call failed
    """
    cache_key = _judge_cache_key(record)

    # Check cache first
    if cache is not None and cache_key in cache:
        result = cache[cache_key].copy()
        result["cached"] = True
        return result

    instruction, response = _extract_pair_texts(record)
    if not instruction or not response:
        return {
            "overall_score": 0.0,
            "pass": False,
            "factual_accuracy": 0,
            "completeness": 0,
            "security_relevance": 0,
            "clarity": 0,
            "cached": False,
            "error": "empty_instruction_or_response",
        }

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        instruction=instruction[:4000],
        response=response[:8000],
    )

    try:
        raw = _call_judge_api(prompt, model, api_url, api_key, timeout)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return {
            "overall_score": 0.0,
            "pass": False,
            "factual_accuracy": 0,
            "completeness": 0,
            "security_relevance": 0,
            "clarity": 0,
            "cached": False,
            "error": f"api_error: {exc}",
        }

    parsed = _parse_judge_response(raw)
    if parsed is None:
        return {
            "overall_score": 0.0,
            "pass": False,
            "factual_accuracy": 0,
            "completeness": 0,
            "security_relevance": 0,
            "clarity": 0,
            "cached": False,
            "error": f"parse_error: could not parse judge response",
        }

    result = {
        "overall_score": float(parsed.get("overall_score", 0.0)),
        "factual_accuracy": int(parsed.get("factual_accuracy", 0)),
        "completeness": int(parsed.get("completeness", 0)),
        "security_relevance": int(parsed.get("security_relevance", 0)),
        "clarity": int(parsed.get("clarity", 0)),
        "pass": bool(parsed.get("pass", False)),
        "cached": False,
        "error": None,
    }

    # Store in cache
    if cache is not None:
        cache[cache_key] = {k: v for k, v in result.items() if k != "cached"}

    return result


def judge_pairs(
    records: list[dict],
    model: str,
    api_url: str = DEFAULT_JUDGE_API_URL,
    api_key: str | None = None,
    threshold: float = DEFAULT_JUDGE_THRESHOLD,
    max_pairs: int | None = None,
    cache_path: Path | None = None,
    timeout: int = 60,
) -> tuple[list[dict], dict[str, Any]]:
    """Judge-and-Revise: run LLM quality evaluation on pairs that passed rule-based filters.

    Args:
        records: List of records that passed rule-based filtering.
        model: Model name for the judge API.
        api_url: OpenAI-compatible API endpoint URL.
        api_key: Optional API key for authentication.
        threshold: Minimum overall_score to keep (0.0-1.0).
        max_pairs: Maximum number of pairs to judge (None = all).
        cache_path: Path to judge result cache file (None = no caching).
        timeout: API call timeout in seconds.

    Returns:
        Tuple of (passed_records, judge_stats) where judge_stats contains:
            - judge_evaluated: number of pairs judged
            - judge_passed: number that passed judge
            - judge_failed: number that failed judge
            - judge_threshold: the threshold used
            - judge_score_distribution: histogram of overall_score buckets
            - judge_errors: number of pairs with judge errors
            - judge_cache_hits: number of cache hits
    """
    if not records:
        return [], {
            "judge_evaluated": 0,
            "judge_passed": 0,
            "judge_failed": 0,
            "judge_threshold": threshold,
            "judge_score_distribution": {},
            "judge_errors": 0,
            "judge_cache_hits": 0,
        }

    # Limit number of pairs to judge
    to_judge = records[:max_pairs] if max_pairs is not None else records
    remaining = records[max_pairs:] if max_pairs is not None else []

    # Load cache if specified
    cache: dict[str, dict] | None = None
    if cache_path is not None:
        cache = _load_judge_cache(cache_path)
    else:
        cache = {}

    passed: list[dict] = []
    stats: dict[str, Any] = {
        "judge_evaluated": len(to_judge),
        "judge_passed": 0,
        "judge_failed": 0,
        "judge_threshold": threshold,
        "judge_score_distribution": Counter(),
        "judge_errors": 0,
        "judge_cache_hits": 0,
    }

    # Score distribution buckets: 0.0-0.1, 0.1-0.2, ..., 0.9-1.0
    def _score_bucket(score: float) -> str:
        bucket = int(score * 10) / 10
        return f"{bucket:.1f}-{bucket + 0.1:.1f}"

    # Progress reporting
    total = len(to_judge)
    print(f"\n  Judge-and-Revise: evaluating {total} pairs with model '{model}'...")
    print(f"  API endpoint: {api_url}")
    print(f"  Quality threshold: {threshold}")
    if cache_path:
        print(f"  Cache: {cache_path}")

    for i, record in enumerate(to_judge, 1):
        result = judge_single_pair(record, model, api_url, api_key, cache, timeout)

        if result["cached"]:
            stats["judge_cache_hits"] += 1

        if result.get("error"):
            stats["judge_errors"] += 1
            # On error, keep the record (don't filter on API failure)
            passed.append(record)
            print(
                f"    [{i}/{total}] ERROR: {result['error']} — keeping record",
                file=sys.stderr,
            )
            continue

        overall = result["overall_score"]
        bucket = _score_bucket(overall)
        stats["judge_score_distribution"][bucket] += 1

        # Use explicit pass/fail from judge, but also enforce threshold
        judge_pass = result["pass"] and overall >= threshold
        if judge_pass:
            stats["judge_passed"] += 1
            passed.append(record)
        else:
            stats["judge_failed"] += 1

        # Progress indicator (every 10 or at end)
        if i % 10 == 0 or i == total:
            print(
                f"    [{i}/{total}] score={overall:.2f} "
                f"{'PASS' if judge_pass else 'FAIL'} "
                f"(F={result['factual_accuracy']} "
                f"C={result['completeness']} "
                f"S={result['security_relevance']} "
                f"Cl={result['clarity']})"
            )

    # Add records beyond max_pairs limit (not judged, pass through)
    passed.extend(remaining)

    # Save cache if specified
    if cache_path is not None and cache is not None:
        _save_judge_cache(cache, cache_path)
        print(f"  Judge cache saved to {cache_path}")

    # Convert Counter to regular dict for JSON serialization
    stats["judge_score_distribution"] = dict(stats["judge_score_distribution"])

    print(
        f"\n  Judge results: {stats['judge_passed']} passed, "
        f"{stats['judge_failed']} failed, "
        f"{stats['judge_errors']} errors, "
        f"{stats['judge_cache_hits']} cache hits"
    )

    return passed, stats


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

    if report.get("judge_evaluated") is not None:
        print()
        print("  Judge-and-Revise:")
        print(f"    Evaluated by judge:       {report['judge_evaluated']:,}")
        print(f"    Passed judge:             {report['judge_passed']:,}")
        print(f"    Failed judge:             {report['judge_failed']:,}")
        print(f"    Judge threshold:          {report['judge_threshold']:.2f}")
        if report.get("judge_score_distribution"):
            print("    Score distribution:")
            for bucket, count in sorted(report["judge_score_distribution"].items()):
                print(f"      {bucket}: {count}")

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
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help=(
            "Model name for Judge-and-Revise LLM quality filtering. "
            "When set, pairs that pass rule-based filters are also evaluated "
            "by the LLM judge. Use a local model name (e.g. 'qwen2.5-coder-7b') "
            "for LM Studio/Ollama, or any OpenAI-compatible model name."
        ),
    )
    parser.add_argument(
        "--judge-threshold",
        type=float,
        default=DEFAULT_JUDGE_THRESHOLD,
        help=(
            f"Minimum overall quality score (0.0-1.0) to keep a pair when "
            f"Judge-and-Revise is enabled (default: {DEFAULT_JUDGE_THRESHOLD}). "
            f"Only used with --judge-model."
        ),
    )
    parser.add_argument(
        "--judge-max-pairs",
        type=int,
        default=None,
        help=(
            "Maximum number of pairs to send to the LLM judge. "
            "Pairs beyond this limit pass through without judging. "
            "Useful for cost control when using paid API endpoints. "
            "Default: judge all pairs that passed rule-based filters."
        ),
    )
    parser.add_argument(
        "--judge-api-url",
        type=str,
        default=DEFAULT_JUDGE_API_URL,
        help=(
            f"OpenAI-compatible API endpoint URL for the judge model "
            f"(default: {DEFAULT_JUDGE_API_URL}). "
            f"Works with LM Studio, Ollama (with OpenAI compat), "
            f"and any OpenAI-compatible server."
        ),
    )
    parser.add_argument(
        "--judge-api-key",
        type=str,
        default=None,
        help=(
            "API key for the judge endpoint. Not needed for local models "
            "(LM Studio, Ollama). Required for OpenAI and other cloud APIs."
        ),
    )
    parser.add_argument(
        "--judge-cache",
        type=Path,
        default=None,
        help=(
            f"Path to judge result cache file. Enables caching of LLM judge "
            f"results to avoid re-evaluating the same pairs. "
            f"Default: {DEFAULT_JUDGE_CACHE} when --judge-model is set, "
            f"or no caching when --judge-model is not set."
        ),
    )

    args = parser.parse_args(argv)

    min_factor = args.min_length_factor
    dedup_threshold = args.dedup_threshold

    # Resolve judge settings
    judge_model = args.judge_model
    judge_threshold = args.judge_threshold
    judge_max_pairs = args.judge_max_pairs
    judge_api_url = args.judge_api_url
    judge_api_key = args.judge_api_key
    judge_cache = args.judge_cache
    # If judge-model is set but no cache path, use default
    if judge_model is not None and judge_cache is None:
        judge_cache = DEFAULT_JUDGE_CACHE

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
            judge_model=judge_model,
            judge_api_url=judge_api_url,
            judge_api_key=judge_api_key,
            judge_threshold=judge_threshold,
            judge_max_pairs=judge_max_pairs,
            judge_cache=judge_cache,
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
        total_judge_evaluated = 0
        total_judge_passed = 0
        total_judge_failed = 0

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
                judge_model=judge_model,
                judge_api_url=judge_api_url,
                judge_api_key=judge_api_key,
                judge_threshold=judge_threshold,
                judge_max_pairs=judge_max_pairs,
                judge_cache=judge_cache,
            )
            print_report(report)
            total_reports.append(report)
            total_passed += report["passed"]
            total_failed += report["failed"]
            total_records += report["total"]
            total_judge_evaluated += report.get("judge_evaluated", 0)
            total_judge_passed += report.get("judge_passed", 0)
            total_judge_failed += report.get("judge_failed", 0)

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
            if total_judge_evaluated > 0:
                print(f"  Judge evaluated:     {total_judge_evaluated:,}")
                print(f"  Judge passed:        {total_judge_passed:,}")
                print(f"  Judge failed:        {total_judge_failed:,}")
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
