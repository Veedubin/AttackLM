#!/usr/bin/env python3
"""
generate_dpo_pairs.py — Generate DPO preference pairs using Constitutional AI self-critique.

For each instruction-response pair in SFT data, the script:
1. Takes the instruction and the existing (or generated) response.
2. Applies constitutional critique: the LLM reviews the response against
   a set of security-specific principles.
3. If violations are found, generates a revised response that addresses them.
4. Emits a DPO preference pair: (rejected=original, chosen=revised).

Only pairs where the critique found genuine violations are kept — pairs with
no critique findings carry no preference signal and are skipped.

Output format (JSONL):
    {
      "prompt": "<instruction>",
      "chosen": "<revised response>",
      "rejected": "<original response>",
      "critique": "<the critique text>",
      "constitution_violations": ["principle 3", "principle 7"],
      "source": "...",
      "mitre_ids": [...],
      ...
    }

Usage:
    python scripts/generate_dpo_pairs.py --source metasploit-framework --max-pairs 50
    python scripts/generate_dpo_pairs.py --input data/datasets/buckets/sources/atomic-red-team/base/discovery/data.jsonl
    python scripts/generate_dpo_pairs.py --source sigma-hq --temperature 0.7 --dry-run
    python scripts/generate_dpo_pairs.py --constitution my_constitution.txt --source metasploit-framework
    python scripts/generate_dpo_pairs.py --model qwen2.5:14b --max-pairs 100
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "datasets" / "buckets" / "sources"
OUTPUT_DIR = PROJECT_DIR / "data" / "datasets" / "dpo_pairs"

# ---------------------------------------------------------------------------
# Security Constitution — principles the model should follow
# ---------------------------------------------------------------------------
SECURITY_CONSTITUTION: list[str] = [
    "Always reference specific CVE IDs, CWE IDs, or MITRE ATT&CK technique IDs when applicable.",
    "Include concrete commands, code snippets, or tool examples rather than abstract descriptions.",
    "Explain both the attack technique AND its detection/mitigation.",
    "Prioritize accuracy over comprehensiveness — if unsure, state uncertainty.",
    "Structure responses with clear sections: Overview, Technique, Commands/Tools, Detection, Mitigation.",
    "Use technical terminology correctly — define acronyms on first use.",
    "Include references to original sources (CVEs, advisories, research papers) when available.",
    "For red team content, include OPSEC considerations and cleanup steps.",
    "For blue team content, include detection rules (Sigma, YARA, Snort) where applicable.",
    "Avoid generating actual exploit code for unpatched vulnerabilities — describe the technique instead.",
]

# ---------------------------------------------------------------------------
# Backend configuration — supports Ollama or OpenAI-compatible APIs (LMStudio, etc.)
# ---------------------------------------------------------------------------
BACKEND = os.environ.get("BACKEND", "ollama").lower().strip()

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")

LMSTUDIO_URL = os.environ.get(
    "LMSTUDIO_URL", "http://localhost:1234/v1/chat/completions"
)
LMSTUDIO_MODEL = os.environ.get(
    "LMSTUDIO_MODEL",
    "qwen2.5-coder-14b-instruct-uncensored",
)

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

API_TIMEOUT = int(os.environ.get("API_TIMEOUT", "300"))  # seconds

# Default temperature — higher than evolve_pairs because critique needs creativity
DEFAULT_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.8"))

# Inter-batch pause (seconds) — can be disabled with --no-sleep
BATCH_PAUSE = 2

# Retry configuration
MAX_RETRIES = 3

# Connection pool — reuse HTTP session for faster subsequent calls
_session: requests.Session | None = None


def _get_session() -> requests.Session:
    """Return a reusable requests.Session for connection pooling."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"Content-Type": "application/json"})
    return _session


# ---------------------------------------------------------------------------
# Live progress — Rich (optional) with plain-text fallback
# ---------------------------------------------------------------------------
try:
    from rich.progress import (
        Progress,
        TextColumn,
        BarColumn,
        MofNCompleteColumn,
        TimeElapsedColumn,
    )
    from rich.console import Console

    _rich_available = True
except ImportError:
    _rich_available = False


def _rich_progress(total_steps: int):
    """Return a Rich Progress context manager if available, else None."""
    if not _rich_available:
        return None
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[green]{task.fields[pairs]}/{task.fields[target]} pairs"),
        TextColumn("[yellow]{task.fields[tok_per_sec]:,.0f} tok/s"),
        TextColumn("[magenta]{task.fields[pair_per_sec]:,.1f} pair/s"),
        TextColumn("[cyan]{task.fields[latency_ms]:,.0f} ms"),
        TimeElapsedColumn(),
    )


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
def get_backend_info(model_override: str | None = None) -> dict:
    """Return the active backend URL, model name, and type.

    Args:
        model_override: If provided, use this model name instead of the default.
    """
    if BACKEND == "lmstudio":
        return {
            "url": LMSTUDIO_URL,
            "model": model_override or LMSTUDIO_MODEL,
            "type": "openai",
        }
    if BACKEND == "openai":
        return {
            "url": f"{OPENAI_BASE_URL}/chat/completions",
            "model": model_override or OPENAI_MODEL,
            "type": "openai",
        }
    # default: ollama
    return {
        "url": OLLAMA_URL,
        "model": model_override or OLLAMA_MODEL,
        "type": "ollama",
    }


# ---------------------------------------------------------------------------
# Thinking-block removal — strip model reasoning traces
# ---------------------------------------------------------------------------
THINKING_PATTERNS: list[re.Pattern] = [
    re.compile(r"<think>.*?</think>", re.DOTALL),
    re.compile(r"<thinking>.*?</thinking>", re.DOTALL),
    re.compile(r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>", re.DOTALL),
    re.compile(r"<\|thinking\|>.*?<\|/thinking\|>", re.DOTALL),
]


def strip_thinking(text: str) -> str:
    """Remove thinking/reasoning blocks from model output."""
    for pat in THINKING_PATTERNS:
        text = pat.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# LLM API — Unified backend (Ollama, LMStudio, OpenAI)
# ---------------------------------------------------------------------------
def call_llm(
    messages: list[dict],
    temperature: float | None = None,
    model_override: str | None = None,
    max_retries: int = MAX_RETRIES,
) -> dict:
    """Call the active LLM backend and return {content, usage, latency_ms}.

    Args:
        messages: List of {role, content} dicts for the conversation.
        temperature: Sampling temperature. Defaults to DEFAULT_TEMPERATURE.
        model_override: Override the model name for this call.
        max_retries: Number of retry attempts on transient errors.

    Returns:
        dict with keys: content (str), usage (dict), latency_ms (float)

    Raises SystemExit on permanent failure.
    """
    temp = temperature if temperature is not None else DEFAULT_TEMPERATURE
    backend = get_backend_info(model_override=model_override)
    url = backend["url"]
    model = backend["model"]
    btype = backend["type"]
    session = _get_session()

    headers: dict[str, str] = {"Content-Type": "application/json"}

    for attempt in range(max_retries + 1):
        payload: dict = {}

        if btype == "openai":
            if BACKEND == "openai" and OPENAI_API_KEY:
                headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": 8192,
            }
        else:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temp, "num_predict": 8192},
            }

        start = time.perf_counter()
        try:
            response = session.post(
                url, headers=headers, json=payload, timeout=API_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            latency_ms = (time.perf_counter() - start) * 1000

            if btype == "openai":
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                usage_dict = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
            else:
                content = data["message"]["content"]
                usage_dict = {
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": (
                        data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                    ),
                }

            result = strip_thinking(content)
            if result.strip():
                return {
                    "content": result,
                    "usage": usage_dict,
                    "latency_ms": latency_ms,
                }

            # Empty response — retry
            if attempt < max_retries:
                print(f"    Empty response, retrying ({attempt + 1}/{max_retries})...")
                time.sleep(1)
                continue
            print("    WARNING: All retries returned empty response", file=sys.stderr)
            return {"content": result, "usage": usage_dict, "latency_ms": latency_ms}

        except requests.ConnectionError:
            if attempt < max_retries:
                print(
                    f"    Connection error, retrying ({attempt + 1}/{max_retries})..."
                )
                time.sleep(2)
                continue
            print(
                f"\nERROR: Cannot connect to LLM backend at {url} "
                f"(backend={BACKEND}) after {max_retries + 1} attempts.\n"
                f"  LMStudio: Start the server in LMStudio UI "
                f"(Developer tab -> Server -> Start).\n"
                f"  Ollama:   Start with: ollama serve",
                file=sys.stderr,
            )
            sys.exit(1)

        except requests.Timeout:
            if attempt < max_retries:
                print(f"    Timeout, retrying ({attempt + 1}/{max_retries})...")
                continue
            print(
                f"\nERROR: LLM request timed out after {API_TIMEOUT}s "
                f"on all {max_retries + 1} attempts.",
                file=sys.stderr,
            )
            sys.exit(1)

        except requests.HTTPError as exc:
            status = exc.response.status_code if hasattr(exc, "response") else 0
            if 500 <= status < 600 and attempt < max_retries:
                print(f"    HTTP {status}, retrying ({attempt + 1}/{max_retries})...")
                time.sleep(2)
                continue
            body = exc.response.text[:500] if hasattr(exc, "response") else ""
            print(
                f"\nERROR: LLM returned HTTP {status}\n  {body}",
                file=sys.stderr,
            )
            sys.exit(1)

        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            if attempt < max_retries:
                print(f"    Parse error, retrying ({attempt + 1}/{max_retries})...")
                time.sleep(1)
                continue
            print(f"\nERROR: Unexpected LLM response format: {exc}", file=sys.stderr)
            sys.exit(1)

    return {
        "content": "",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "latency_ms": 0,
    }


# ---------------------------------------------------------------------------
# Input loading — read AttackLM JSONL files
# ---------------------------------------------------------------------------
def discover_source_files(source: str | None = None) -> list[Path]:
    """Find all JSONL files under data/datasets/buckets/sources/<source>/.

    If source is None, discover all sources.
    """
    if source:
        source_dir = DATA_DIR / source
        if not source_dir.is_dir():
            print(
                f"ERROR: Source directory not found: {source_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        files = sorted(source_dir.rglob("data*.jsonl"))
        if not files:
            print(
                f"ERROR: No JSONL files found under {source_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        return files

    # Discover all sources
    all_files: list[Path] = []
    for source_dir in sorted(DATA_DIR.iterdir()):
        if source_dir.is_dir() and not source_dir.name.startswith("_"):
            all_files.extend(source_dir.rglob("data*.jsonl"))
    return sorted(all_files)


def load_records(files: list[Path], max_records: int | None = None) -> list[dict]:
    """Load JSONL records from files, deduplicating by content hash.

    Returns list of valid record dicts with at minimum:
      - messages: list of {role, content} dicts
      - source: str
    """
    records: list[dict] = []
    seen_hashes: set[str] = set()

    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                for line_num, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        print(
                            f"  WARNING: Malformed JSON in {fpath.name}:{line_num}, skipping",
                            file=sys.stderr,
                        )
                        continue

                    # Validate minimum structure
                    messages = record.get("messages")
                    if not isinstance(messages, list) or len(messages) < 2:
                        continue

                    # Check messages have required fields
                    valid = True
                    for msg in messages:
                        if not isinstance(msg, dict):
                            valid = False
                            break
                        if "role" not in msg or "content" not in msg:
                            valid = False
                            break
                        if (
                            not isinstance(msg["content"], str)
                            or len(msg["content"].strip()) < 10
                        ):
                            valid = False
                            break
                    if not valid:
                        continue

                    # Deduplicate by content hash of assistant response
                    assistant_msgs = [
                        m for m in messages if m.get("role") == "assistant"
                    ]
                    if not assistant_msgs:
                        continue
                    content_hash = hashlib.sha256(
                        assistant_msgs[0]["content"].encode()
                    ).hexdigest()[:16]
                    if content_hash in seen_hashes:
                        continue
                    seen_hashes.add(content_hash)

                    records.append(record)
                    if max_records and len(records) >= max_records:
                        return records

        except OSError as exc:
            print(f"  WARNING: Cannot read {fpath}: {exc}", file=sys.stderr)

    return records


def load_input_file(input_path: Path, max_records: int | None = None) -> list[dict]:
    """Load JSONL records from a single file path."""
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    return load_records([input_path], max_records=max_records)


# ---------------------------------------------------------------------------
# Constitution loading
# ---------------------------------------------------------------------------
def load_constitution(path: Path | None = None) -> list[str]:
    """Load constitution principles from a text file, or return the default.

    File format: one principle per line. Empty lines and lines starting
    with # are skipped.
    """
    if path is None:
        return SECURITY_CONSTITUTION

    if not path.exists():
        print(f"ERROR: Constitution file not found: {path}", file=sys.stderr)
        sys.exit(1)

    principles: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            principles.append(line)

    if not principles:
        print(
            f"ERROR: No principles found in {path}",
            file=sys.stderr,
        )
        sys.exit(1)

    return principles


# ---------------------------------------------------------------------------
# Constitutional AI: Critique prompt
# ---------------------------------------------------------------------------
def build_critique_prompt(
    instruction: str,
    response: str,
    constitution: list[str],
) -> list[dict]:
    """Build the critique prompt messages.

    Asks the LLM to review the response against each constitution principle
    and identify violations.
    """
    principles_text = "\n".join(f"  {i + 1}. {p}" for i, p in enumerate(constitution))

    system_content = (
        "You are a Constitutional AI reviewer specializing in cybersecurity training data. "
        "Your job is to critique AI responses against a set of security-specific principles "
        "and identify concrete violations that make the response worse than it could be.\n\n"
        "Be strict but fair. Only flag violations where the response clearly fails to meet "
        "the principle. Minor imperfections should not be flagged.\n\n"
        "You MUST respond in the following structured format:\n\n"
        "VIOLATIONS:\n"
        "- [principle text]: [brief explanation of what's missing or wrong]\n"
        "- ...\n\n"
        "CRITIQUE:\n"
        "[2-4 sentence summary of the main issues, if any]\n\n"
        "IMPROVEMENTS:\n"
        "- [specific, actionable improvement that addresses each violation]\n"
        "- ...\n\n"
        "If the response fully satisfies all principles, write:\n\n"
        "VIOLATIONS:\n"
        "(none)\n\n"
        "CRITIQUE:\n"
        "(The response satisfies all constitutional principles.)\n\n"
        "IMPROVEMENTS:\n"
        "(none)"
    )

    user_content = (
        f"SECURITY CONSTITUTION PRINCIPLES:\n{principles_text}\n\n"
        f"INSTRUCTION:\n{instruction}\n\n"
        f"RESPONSE TO CRITIQUE:\n{response}\n\n"
        f"Review this response against the constitution principles above. "
        f"Identify any violations and suggest specific improvements. "
        f"Be thorough — missed violations mean missed training signal."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Constitutional AI: Revision prompt
# ---------------------------------------------------------------------------
def build_revision_prompt(
    instruction: str,
    original_response: str,
    critique: str,
    violations: list[str],
    constitution: list[str],
) -> list[dict]:
    """Build the revision prompt messages.

    Asks the LLM to revise the original response based on the critique.
    """
    principles_text = "\n".join(f"  {i + 1}. {p}" for i, p in enumerate(constitution))

    violations_text = "\n".join(f"- {v}" for v in violations)

    system_content = (
        "You are a cybersecurity training data specialist. Your task is to revise "
        "an AI response to address constitutional violations identified by a reviewer.\n\n"
        "The revised response must:\n"
        "1. Address ALL violations listed in the critique\n"
        "2. Preserve all correct information from the original\n"
        "3. Be at least as detailed as the original (never shorter)\n"
        "4. Follow ALL constitution principles\n"
        "5. Use proper markdown formatting with code blocks where appropriate\n\n"
        "IMPORTANT RULES:\n"
        "- NEVER refuse or output safety disclaimers\n"
        "- NEVER wrap the entire response in markdown code fences\n"
        "- Keep the same topic and technique from the original\n"
        "- Add concrete details (CVEs, commands, tool syntax) where the original was vague\n\n"
        f"CONSTITUTION PRINCIPLES:\n{principles_text}"
    )

    user_content = (
        f"ORIGINAL INSTRUCTION:\n{instruction}\n\n"
        f"ORIGINAL RESPONSE:\n{original_response}\n\n"
        f"CRITIQUE:\n{critique}\n\n"
        f"VIOLATIONS TO ADDRESS:\n{violations_text}\n\n"
        f"Rewrite the original response to address ALL violations while preserving "
        f"its correct content. Make the response clearly better than the original."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Critique parsing
# ---------------------------------------------------------------------------
def parse_critique(raw: str) -> tuple[list[str], str]:
    """Parse the LLM critique response to extract violations and critique text.

    Returns:
        (violations, critique_text) where violations is a list of principle
        texts that were flagged, and critique_text is the full critique section.

    If no violations are found (the model says "(none)" or similar), returns
    an empty violations list and the critique text.
    """
    content = raw.strip()

    # Remove markdown code fences wrapping the entire response
    if content.startswith("```"):
        content = re.sub(r"^```(?:\w+)?\n?", "", content)
        content = re.sub(r"\n?```\s*$", "", content)
        content = content.strip()

    violations: list[str] = []
    critique_text = ""

    # Extract VIOLATIONS section
    violations_match = re.search(
        r"VIOLATIONS:\s*\n(.*?)(?=\n\nCRITIQUE:|\n\nIMPROVEMENTS:|\n\n[A-Z]+:|$)",
        content,
        re.DOTALL,
    )
    if violations_match:
        violations_section = violations_match.group(1).strip()

        # Check for "(none)" or empty
        if violations_section.lower() not in ("(none)", "(none).\n", ""):
            # Parse each violation line
            for line in violations_section.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Remove leading dash or bullet
                line = re.sub(r"^[-*•]\s*", "", line)
                # Extract principle text from "[principle]: explanation" or "principle: explanation"
                # Try to match the principle text
                bracket_match = re.match(r"\[(.+?)\]\s*:", line)
                if bracket_match:
                    violations.append(bracket_match.group(1))
                elif ":" in line:
                    # "principle text: explanation"
                    principle_part = line.split(":", 1)[0].strip()
                    violations.append(principle_part)
                elif line:
                    violations.append(line)

    # Extract CRITIQUE section
    critique_match = re.search(
        r"CRITIQUE:\s*\n(.*?)(?=\n\nIMPROVEMENTS:|\n\n[A-Z]+:|$)",
        content,
        re.DOTALL,
    )
    if critique_match:
        critique_text = critique_match.group(1).strip()
        # Remove "(The response satisfies...)" placeholder if it's the only content
        if critique_text.startswith("(") and critique_text.endswith(")"):
            inner = critique_text[1:-1]
            if "satisfies" in inner.lower():
                critique_text = ""

    # If no structured sections found, use the raw text as critique
    if not violations and not critique_text:
        critique_text = content

    return violations, critique_text


# ---------------------------------------------------------------------------
# DPO pair generation
# ---------------------------------------------------------------------------
def generate_dpo_pair(
    record: dict,
    constitution: list[str],
    temperature: float,
    model_override: str | None = None,
    dry_run: bool = False,
) -> dict | None:
    """Generate a DPO preference pair from a single SFT record.

    Steps:
    1. Extract instruction and response from the SFT record.
    2. Generate a constitutional critique of the response.
    3. Parse the critique to identify violations.
    4. If violations found, generate a revised response.
    5. Return the DPO pair dict, or None if no violations found.

    Args:
        record: SFT data record with 'messages' array.
        constitution: List of constitutional principles.
        temperature: LLM sampling temperature.
        model_override: Override the model name for LLM calls.
        dry_run: If True, print prompts instead of calling LLM.

    Returns:
        DPO pair dict or None if no violations found.
    """
    messages = record.get("messages", [])

    # Extract user instruction and assistant response
    instruction = ""
    original_response = ""
    for msg in messages:
        if msg.get("role") == "user":
            instruction = msg["content"]
        elif msg.get("role") == "assistant":
            original_response = msg["content"]

    if not instruction or not original_response:
        return None

    # Step 1: Critique the original response
    critique_messages = build_critique_prompt(
        instruction, original_response, constitution
    )

    if dry_run:
        print(f"\n{'=' * 70}")
        print("DRY RUN — Critique prompt")
        print(f"{'=' * 70}")
        for msg in critique_messages:
            role = msg["role"].upper()
            text = msg["content"][:500]
            print(f"\n--- {role} ---\n{text}...")
        return None

    critique_result = call_llm(
        critique_messages, temperature=temperature, model_override=model_override
    )
    critique_raw = critique_result["content"]

    if not critique_raw.strip():
        print("  WARNING: Empty critique response, skipping", file=sys.stderr)
        return None

    # Step 2: Parse critique to extract violations
    violations, critique_text = parse_critique(critique_raw)

    # Step 3: If no violations found, skip this pair
    if not violations:
        return None

    # Step 4: Generate revised response addressing the violations
    revision_messages = build_revision_prompt(
        instruction,
        original_response,
        critique_text or critique_raw,
        violations,
        constitution,
    )

    revision_result = call_llm(
        revision_messages, temperature=temperature, model_override=model_override
    )
    revised_response = revision_result["content"]

    if not revised_response.strip():
        print("  WARNING: Empty revision response, skipping", file=sys.stderr)
        return None

    # Step 5: Build DPO pair
    dpo_pair = {
        "prompt": instruction,
        "chosen": revised_response,
        "rejected": original_response,
        "critique": critique_text or critique_raw,
        "constitution_violations": violations,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Preserve provenance fields
    for key in (
        "source",
        "source_uri",
        "license",
        "license_uri",
        "rights_contact",
        "upstream_copyright",
        "upstream_license_uri",
        "attribution_required",
        "bsd_3_clause_notice",
        "derived_from",
        "mitre_ids",
        "platforms",
        "upstream_module_path",
    ):
        if key in record:
            dpo_pair[key] = record[key]

    # Add LLM usage metrics
    dpo_pair["llm_usage"] = {
        "critique": critique_result["usage"],
        "revision": revision_result["usage"],
    }
    dpo_pair["llm_latency_ms"] = {
        "critique": critique_result["latency_ms"],
        "revision": revision_result["latency_ms"],
    }

    return dpo_pair


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------
def process_records(
    records: list[dict],
    constitution: list[str],
    max_pairs: int | None,
    temperature: float,
    model_override: str | None,
    dry_run: bool,
    no_sleep: bool,
    source_name: str,
) -> None:
    """Process records and generate DPO pairs."""
    content_hash = hashlib.sha256(
        json.dumps(records[:10], sort_keys=True).encode()
    ).hexdigest()[:8]

    if dry_run:
        print(f"\n{'=' * 70}")
        print("DRY RUN — Constitutional AI DPO Pair Generator")
        print(f"{'=' * 70}")
        print(f"  Records: {len(records)}")
        print(f"  Constitution: {len(constitution)} principles")
        print(f"  Temperature: {temperature}")
        print(f"  Max pairs: {max_pairs or 'unlimited'}")
        print()
        # Show dry-run for first 3 records
        for i, record in enumerate(records[:3]):
            generate_dpo_pair(
                record,
                constitution=constitution,
                temperature=temperature,
                model_override=model_override,
                dry_run=True,
            )
        return

    # Output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"dpo_{source_name}_{content_hash}.jsonl"
    raw_log_path = OUTPUT_DIR / f"dpo_{source_name}_{content_hash}_raw.log"

    # Open output and raw log files
    raw_log = open(raw_log_path, "w", encoding="utf-8")

    all_pairs: list[dict] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_latency_ms = 0.0
    skipped = 0
    backend = get_backend_info(model_override=model_override)

    effective_max = max_pairs if max_pairs else len(records)

    # Calculate batches
    batch_size = 1  # Process one record at a time (two LLM calls per record)
    total_batches = len(records)

    print(
        f"\n[ Constitutional AI DPO ]  {len(records)} records  |  "
        f"max_pairs={effective_max}  |  "
        f"backend={BACKEND}  model={backend['model']}  temp={temperature}"
    )

    # Plain fallback: minimal live status line
    def _plain_status(
        processed: int,
        pairs: int,
        skipped_count: int,
        latency_ms: float,
        tok_count: int,
    ):
        tok_per_sec = tok_count / max(latency_ms / 1000, 0.001)
        pair_per_sec = pairs / max(latency_ms / 1000, 0.001)
        bar_len = 20
        filled = int(bar_len * processed / total_batches)
        bar = "█" * filled + "░" * (bar_len - filled)
        line = (
            f"\r  {bar}  {processed}/{total_batches} processed  "
            f"{pairs} pairs  {skipped_count} skipped  "
            f"{tok_per_sec:,.0f} tok/s  {pair_per_sec:,.1f} pair/s  "
            f"{latency_ms:,.0f}ms"
        )
        print(line.ljust(120), end="", flush=True)

    progress_ctx = _rich_progress(total_batches)
    progress = None
    task_id = None

    try:
        if progress_ctx is not None:
            progress = progress_ctx.__enter__()
            task_id = progress.add_task(
                "DPO pairs",
                total=total_batches,
                pairs=0,
                target=effective_max,
                tok_per_sec=0,
                pair_per_sec=0,
                latency_ms=0,
            )

        for i, record in enumerate(records):
            # Check max pairs limit
            if max_pairs and len(all_pairs) >= max_pairs:
                break

            pair = generate_dpo_pair(
                record,
                constitution=constitution,
                temperature=temperature,
                model_override=model_override,
                dry_run=False,
            )

            if pair is not None:
                all_pairs.append(pair)

                # Track metrics
                for call_type in ("critique", "revision"):
                    usage = pair.get("llm_usage", {}).get(call_type, {})
                    total_prompt_tokens += usage.get("prompt_tokens", 0)
                    total_completion_tokens += usage.get("completion_tokens", 0)
                    lat = pair.get("llm_latency_ms", {}).get(call_type, 0)
                    total_latency_ms += lat

                # Write to output immediately (streaming)
                entry_to_write = {
                    k: v
                    for k, v in pair.items()
                    if k not in ("llm_usage", "llm_latency_ms")
                }
                with open(output_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry_to_write, ensure_ascii=False) + "\n")

                # Log raw response
                raw_log.write(f"=== DPO pair {len(all_pairs)} ===\n")
                raw_log.write(json.dumps(pair, indent=2, ensure_ascii=False)[:3000])
                raw_log.write("\n\n")
                raw_log.flush()
            else:
                skipped += 1

            # Update progress
            batch_tok = total_prompt_tokens + total_completion_tokens
            if progress is not None and task_id is not None:
                tok_per_sec = batch_tok / max(total_latency_ms / 1000, 0.001)
                pair_per_sec = len(all_pairs) / max(total_latency_ms / 1000, 0.001)
                progress.update(
                    task_id,
                    advance=1,
                    pairs=len(all_pairs),
                    tok_per_sec=tok_per_sec,
                    pair_per_sec=pair_per_sec,
                    latency_ms=total_latency_ms,
                )
            else:
                _plain_status(
                    i + 1,
                    len(all_pairs),
                    skipped,
                    total_latency_ms,
                    batch_tok,
                )

            # Inter-record pause
            if not no_sleep and i + 1 < total_batches:
                time.sleep(BATCH_PAUSE)

    finally:
        if progress is not None:
            progress_ctx.__exit__(None, None, None)
        else:
            print()  # newline after plain bar
        raw_log.close()

    # Summary
    elapsed_total = total_latency_ms / 1000
    avg_tok_per_sec = (total_prompt_tokens + total_completion_tokens) / max(
        elapsed_total, 0.001
    )
    avg_pair_per_sec = len(all_pairs) / max(elapsed_total, 0.001)

    print(
        f"\n  Results: {len(all_pairs)} DPO pairs generated  |  "
        f"{skipped} skipped (no violations)  |  "
        f"{avg_tok_per_sec:,.0f} tok/s avg  |  "
        f"{avg_pair_per_sec:,.1f} pair/s avg  |  "
        f"{elapsed_total:.1f}s total"
    )
    print(f"  Output: {output_path.name}")

    # Write metadata
    meta_path = OUTPUT_DIR / f"dpo_{source_name}_{content_hash}_meta.json"
    metadata = {
        "method": "constitutional_ai_dpo",
        "source": source_name,
        "created": datetime.now(timezone.utc).isoformat(),
        "input_count": len(records),
        "output_count": len(all_pairs),
        "skipped_count": skipped,
        "constitution_principles": len(constitution),
        "backend": BACKEND,
        "model": backend["model"],
        "temperature": temperature,
        "metrics": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "avg_tokens_per_sec": round(avg_tok_per_sec, 1),
            "avg_pairs_per_sec": round(avg_pair_per_sec, 2),
            "total_seconds": round(elapsed_total, 2),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    print(f"  Metadata: {meta_path.name}")

    if len(all_pairs) == 0:
        print(
            "\n  WARNING: No DPO pairs generated. This means the critique model "
            "found no constitutional violations in any responses.\n"
            "  Try:\n"
            "    - Increasing --temperature for more diverse critique\n"
            "    - Using a different --model with better critique ability\n"
            "    - Using a stricter --constitution file\n",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate DPO preference pairs using Constitutional AI self-critique.\n\n"
            "For each SFT instruction-response pair, the script:\n"
            "  1. Critiques the response against security constitution principles\n"
            "  2. If violations found, generates a revised response\n"
            "  3. Emits (rejected=original, chosen=revised) as a DPO pair\n\n"
            "Only pairs where the critique found violations are kept — "
            "no preference signal without a real quality gap."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/generate_dpo_pairs.py --source metasploit-framework --max-pairs 50\n"
            "  python scripts/generate_dpo_pairs.py --input data/datasets/buckets/sources/atomic-red-team/base/discovery/data.jsonl\n"
            "  python scripts/generate_dpo_pairs.py --source sigma-hq --temperature 0.7\n"
            "  python scripts/generate_dpo_pairs.py --constitution my_principles.txt --source metasploit-framework\n"
            "  python scripts/generate_dpo_pairs.py --model qwen2.5:14b --max-pairs 100 --dry-run\n"
        ),
    )

    # Input selection (mutually exclusive group)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Source directory under data/datasets/buckets/sources/ to process. "
            "If neither --source nor --input specified, processes all sources."
        ),
    )
    input_group.add_argument(
        "--input",
        type=str,
        default=None,
        help="Direct path to a single JSONL file to process.",
    )

    # Generation parameters
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Maximum number of DPO pairs to generate. Default: process all records.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=(
            f"LLM temperature for critique and revision (default: {DEFAULT_TEMPERATURE}). "
            "Higher = more diverse critique. Range: 0.1-1.5."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Override the LLM model name for critique and revision calls. "
            "Uses the default model from BACKEND env var if not specified."
        ),
    )

    # Constitution
    parser.add_argument(
        "--constitution",
        type=str,
        default=None,
        help=(
            "Path to a text file with custom constitution principles (one per line). "
            "If not specified, uses the built-in security constitution."
        ),
    )

    # Execution options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print prompts without calling LLM.",
    )
    parser.add_argument(
        "--no-sleep",
        action="store_true",
        default=False,
        help="Remove the inter-record pause for faster generation.",
    )

    args = parser.parse_args()

    # Validate temperature
    if not 0.1 <= args.temperature <= 1.5:
        parser.error(
            f"--temperature must be between 0.1 and 1.5, got {args.temperature}"
        )

    # Load constitution
    constitution_path = Path(args.constitution) if args.constitution else None
    constitution = load_constitution(constitution_path)

    # Load records
    if args.input:
        input_path = Path(args.input)
        records = load_input_file(input_path)
        source_name = input_path.stem
    elif args.source:
        source_files = discover_source_files(args.source)
        records = load_records(source_files)
        source_name = args.source
    else:
        source_files = discover_source_files()
        records = load_records(source_files)
        source_name = "all_sources"

    if not records:
        print("ERROR: No valid records found.", file=sys.stderr)
        sys.exit(1)

    # Apply max-pairs as a record limit (we process records until we have enough pairs)
    # But we also cap the number of records we attempt
    max_records = args.max_pairs * 3 if args.max_pairs else None  # ~33% yield expected
    if max_records and len(records) > max_records:
        records = records[:max_records]

    backend = get_backend_info(model_override=args.model)

    print("AttackLM Constitutional AI DPO Pair Generator")
    print("=" * 50)
    print(f"Backend:        {BACKEND}")
    print(f"URL:            {backend['url']}")
    print(f"Model:          {backend['model']}")
    print(f"Constitution:   {len(constitution)} principles")
    print(f"Source:         {source_name}")
    print(f"Records:        {len(records)}")
    print(f"Max pairs:      {args.max_pairs or 'unlimited'}")
    print(f"Temperature:    {args.temperature}")
    print(f"Output dir:     {OUTPUT_DIR}")
    if args.no_sleep:
        print("No-sleep:       enabled (no inter-record pauses)")
    print()

    # Show constitution principles
    print("Constitution principles:")
    for i, principle in enumerate(constitution, 1):
        print(f"  {i}. {principle}")
    print()

    process_records(
        records=records,
        constitution=constitution,
        max_pairs=args.max_pairs,
        temperature=args.temperature,
        model_override=args.model,
        dry_run=args.dry_run,
        no_sleep=args.no_sleep,
        source_name=source_name,
    )

    if not args.dry_run:
        print(f"\nDone. Check {OUTPUT_DIR}/ for output files.")


if __name__ == "__main__":
    main()
