#!/usr/bin/env python3
# CREDITS — DATA SOURCE ATTRIBUTION
# ----------------------------------
# This script transforms data from: RTA — Red Team Automation
# Repository: https://github.com/endgameinc/RTA
# License:    GNU Affero General Public License v3.0 (AGPLv3)
# Copyright:  (C) 2018 info@endgame.com
#
# ⚠️  AGPLv3 NOTICE:
#     RTA is the only AGPL-licensed source in AttackLM. AGPLv3 has
#     network-distribution implications for the trained model. If you
#     need an AGPL-clean deployment, retrain the model after removing
#     the `tools/rta` bucket (or simply skip this script).
#
#     This public repository provides the source code and intermediate
#     JSONL files, which satisfies the AGPLv3 §13 source-availability
#     requirement for network-distributed derivatives.
# ----------------------------------
"""Extract RTA (Red Team Automation) TTP scripts into AttackLM JSONL training pairs.

Parses each ``.py`` file in ``data/RTA/red_ttp/`` (skipping ``__init__.py``
and ``common.py``) and produces 2-3 OpenAI-style message triples per script:

  - **Pair type 1** — technique explanation (always generated)
  - **Pair type 2** — command extraction from ``common.execute()`` calls
  - **Pair type 3** — operational context for multi-ATT&CK-ID techniques

Output: ``data/datasets/rta_training_pairs.jsonl``

Usage:
    python scripts/extract_rta_to_jsonl.py
    python scripts/extract_rta_to_jsonl.py --dry-run --max-scripts 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

# Import shared MITRE tactic lookup
sys.path.insert(0, str(Path(__file__).parent))
from mitre_tactic_lookup import get_tactic_for_technique, get_tactic_name

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RTA_DIR = BASE_DIR / "data" / "RTA" / "red_ttp"
DATASETS_DIR = BASE_DIR / "data" / "datasets"
OUTPUT_PATH = DATASETS_DIR / "rta_training_pairs.jsonl"

# ---------------------------------------------------------------------------
# System message
# ---------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are an authorized red team adversary emulation specialist. "
    "Provide precise MITRE ATT&CK technique implementations for security "
    "validation testing."
)

# ---------------------------------------------------------------------------
# Regex patterns for header extraction
# ---------------------------------------------------------------------------
_RE_NAME = re.compile(r"^#\s*Name:\s*(.+)$", re.MULTILINE)
_RE_RTA = re.compile(r"^#\s*[rR][tT][aA]:\s*(.+)$", re.MULTILINE)
_RE_ATTCK = re.compile(r"^#\s*ATT&CK:[^\S\n]*(.*?)$", re.MULTILINE)
_RE_DESC = re.compile(r"^#\s*Description:\s*(.+)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Regex patterns for function extraction
# ---------------------------------------------------------------------------
_RE_MAIN_DEF = re.compile(
    r"^def\s+main\s*\((?:[^()]*|\([^()]*\))*\)\s*:",
    re.MULTILINE,
)


def _extract_indented_block(source: str, def_match: re.Match) -> str:
    """Extract the body of a function starting at *def_match*.

    The body is everything indented deeper than the ``def`` line, up to
    the next line at the same or lesser indentation (or end of file).
    """
    start_pos = def_match.end()
    # Determine the indentation of the def line itself
    def_line_start = source.rfind("\n", 0, def_match.start()) + 1
    def_line = source[def_line_start : def_match.start() + len("def")]
    leading_ws = def_line[: len(def_line) - len(def_line.lstrip())]
    func_indent_level = len(leading_ws)

    lines = source[start_pos:].splitlines(keepends=True)

    body_lines: list[str] = []
    for line in lines:
        if line.strip() == "" or line.strip().startswith("#"):
            body_lines.append(line)
            continue
        current_indent = len(line) - len(line.lstrip())
        if current_indent <= func_indent_level and line.strip():
            break
        body_lines.append(line)

    return "".join(body_lines).rstrip()


def _extract_all_functions(source: str) -> dict[str, str]:
    """Return a mapping of function name → source body for all ``def`` blocks."""
    funcs: dict[str, str] = {}
    # Match `def` lines including nested parentheses in arguments
    # e.g. `def main(target_file=common.get_path("bin", "myapp.exe")):`
    for m in re.finditer(
        r"^(\s*)def\s+(\w+)\s*\((?:[^()]*|\([^()]*\))*\)\s*:",
        source,
        re.MULTILINE,
    ):
        name = m.group(2)
        body = _extract_indented_block(source, m)
        funcs[name] = body
    return funcs


def _extract_common_execute_calls(main_body: str) -> list[str]:
    """Extract command arrays/strings from ``common.execute(...)`` calls.

    Returns a list of human-readable command descriptions.
    """
    commands: list[str] = []

    # Match common.execute([...]) — list-form calls
    for m in re.finditer(r"common\.execute\s*\(\s*\[([^\]]+)\]", main_body):
        raw = m.group(1)
        items = re.findall(r'(?:"([^"]*)"|\'([^\']*)\')', raw)
        parts = [a or b for a, b in items]
        if parts:
            commands.append(" ".join(parts))

    # Match common.execute("string" or common.execute(string_var) — string-form calls
    for m in re.finditer(
        r'common\.execute\s*\(\s*(?:"([^"]+)"|\'([^\']+)\')',
        main_body,
    ):
        cmd = m.group(1) or m.group(2)
        if cmd:
            commands.append(cmd)

    # Match common.execute(command) where command is a bare variable or
    # string with .format() — capture the whole expression
    for m in re.finditer(
        r"common\.execute\s*\(\s*([^[\]\"\')\n]+(?:\.format\([^)]*\))?)\s*(?:,\s*\w+=\s*[^)]+)?\s*\)",
        main_body,
    ):
        expr = m.group(1).strip()
        # Skip if already captured as list-form or string-form above
        if expr.startswith("[") or expr.startswith('"') or expr.startswith("'"):
            continue
        if expr and not expr.startswith("common."):
            commands.append(f"<variable: {expr}>")

    return commands


def _extract_common_log_calls(main_body: str) -> list[str]:
    """Extract log messages from ``common.log(...)`` calls."""
    logs: list[str] = []
    for m in re.finditer(
        r'common\.log\s*\(\s*(?:"([^"]*)"|\'([^\']*)\')',
        main_body,
    ):
        msg = m.group(1) or m.group(2)
        if msg:
            logs.append(msg)
    return logs


# ---------------------------------------------------------------------------
# Parse a single RTA script
# ---------------------------------------------------------------------------
def parse_rta_script(filepath: Path) -> dict | None:
    """Parse one RTA ``.py`` script and return extracted metadata.

    Returns ``None`` if the file cannot be parsed or lacks required headers.
    """
    SKIP_FILES = {"__init__.py", "common.py"}
    if filepath.name in SKIP_FILES:
        return None

    try:
        source = filepath.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    # --- Extract headers ---
    name_m = _RE_NAME.search(source)
    rta_m = _RE_RTA.search(source)
    attck_m = _RE_ATTCK.search(source)
    desc_m = _RE_DESC.search(source)

    # Name is always required; ATT&CK may be empty for some RTA scripts
    if not name_m:
        return None

    name = name_m.group(1).strip()
    rta_file = rta_m.group(1).strip() if rta_m else filepath.name
    raw_attck = attck_m.group(1).strip() if attck_m else ""
    description = desc_m.group(1).strip() if desc_m else ""

    # Parse comma-separated ATT&CK IDs, stripping parenthetical annotations
    # e.g. "T1064 (Scripting), T1086 (PowerShell)" → ["T1064", "T1086"]
    raw_tokens = [t.strip() for t in re.split(r"[,;]\s*", raw_attck) if t.strip()]
    mitre_ids: list[str] = []
    for token in raw_tokens:
        # Strip parenthetical like "(Scripting)", "(PowerShell)"
        clean = re.sub(r"\s*\([^)]*\)", "", token).strip()
        if clean:
            mitre_ids.append(clean)

    # If no ATT&CK IDs found, mark as TBD so the script is still included
    if not mitre_ids:
        mitre_ids = ["TBD"]

    # --- Extract main() body ---
    all_funcs = _extract_all_functions(source)
    main_body = all_funcs.get("main", "")
    if not main_body and _RE_MAIN_DEF.search(source):
        main_body = _extract_indented_block(source, _RE_MAIN_DEF.search(source))

    # --- Extract commands and log messages ---
    execute_commands = _extract_common_execute_calls(main_body)
    log_messages = _extract_common_log_calls(main_body)

    # --- Collect helper functions called from main ---
    helper_funcs: dict[str, str] = {}
    for func_name, func_body in all_funcs.items():
        if func_name == "main":
            continue
        if func_name in main_body:
            helper_funcs[func_name] = func_body

    return {
        "name": name,
        "rta_file": rta_file,
        "mitre_ids": mitre_ids,
        "description": description,
        "main_body": main_body,
        "execute_commands": execute_commands,
        "log_messages": log_messages,
        "helper_funcs": helper_funcs,
        "source_file": filepath.name,
    }


# ---------------------------------------------------------------------------
# Build training pairs
# ---------------------------------------------------------------------------
def _build_pair_technique_explanation(parsed: dict) -> dict:
    """Pair type 1: technique explanation with code."""
    ids_str = ", ".join(parsed["mitre_ids"])
    main_body = parsed["main_body"]
    helper_text = ""
    if parsed["helper_funcs"]:
        helper_text = "\n\n### Helper Functions\n"
        for fname, fbody in parsed["helper_funcs"].items():
            helper_text += f"\n```python\ndef {fname}(...):\n{textwrap.indent(fbody, '    ')}\n```\n"

    assistant = (
        f"## {parsed['name']}\n"
        f"MITRE ATT&CK: {ids_str}\n\n"
        f"{parsed['description']}\n\n"
        f"### Implementation (Python RTA)\n"
        f"```python\n{main_body}\n```"
        f"{helper_text}"
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {
                "role": "user",
                "content": (
                    f"Explain how {parsed['name']} (ATT&CK {ids_str}) works. "
                    f"What commands does it execute on Windows?"
                ),
            },
            {"role": "assistant", "content": assistant},
        ],
        "source": "endgameinc/RTA",
        "mitre_ids": parsed["mitre_ids"],
    }


def _build_pair_command_extraction(parsed: dict) -> dict | None:
    """Pair type 2: command extraction from common.execute calls."""
    commands = parsed["execute_commands"]
    if not commands:
        return None

    ids_str = ", ".join(parsed["mitre_ids"])
    cmd_list = "\n".join(f"  - `{c}`" for c in commands)
    log_ctx = ""
    if parsed["log_messages"]:
        log_ctx = "\n\n**Operational log entries:**\n" + "\n".join(
            f"  - {m}" for m in parsed["log_messages"]
        )

    assistant = f"The {parsed['name']} technique executes:\n\n{cmd_list}{log_ctx}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {
                "role": "user",
                "content": f"What commands implement {parsed['name']} ({ids_str}) on Windows?",
            },
            {"role": "assistant", "content": assistant},
        ],
        "source": "endgameinc/RTA",
        "mitre_ids": parsed["mitre_ids"],
    }


def _build_pair_operational_context(parsed: dict) -> dict | None:
    """Pair type 3: multi-technique operational context (only if 2+ ATT&CK IDs)."""
    if len(parsed["mitre_ids"]) < 2:
        return None

    ids_str = ", ".join(parsed["mitre_ids"])
    mitre_list = ", ".join(f"`{tid}`" for tid in parsed["mitre_ids"])

    # Build an explanation of technique interrelation from the script flow
    log_steps = parsed["log_messages"]
    cmd_steps = parsed["execute_commands"]

    flow_parts: list[str] = []
    if log_steps:
        flow_parts.append("**Execution flow (from log messages):**")
        flow_parts.extend(f"  {i + 1}. {s}" for i, s in enumerate(log_steps))

    if cmd_steps:
        flow_parts.append("\n**Commands per step:**")
        flow_parts.extend(f"  - `{c}`" for c in cmd_steps)

    if not flow_parts and parsed["main_body"]:
        flow_parts.append(
            f"**Implementation overview:**\n```python\n{parsed['main_body'][:800]}\n```"
        )

    flow_text = "\n".join(flow_parts)

    assistant = (
        f"The {parsed['name']} technique chains multiple ATT&CK tactics:\n\n"
        f"MITRE techniques involved: {mitre_list}\n\n"
        f"These techniques work together as part of a coordinated operation. "
        f"The script implements {parsed['description'].lower() if parsed['description'] else 'the technique'} "
        f"by combining {len(parsed['mitre_ids'])} distinct ATT&CK techniques.\n\n"
        f"{flow_text}"
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {
                "role": "user",
                "content": (
                    f"{parsed['name']} maps to multiple ATT&CK techniques: {ids_str}. "
                    f"How do they interrelate?"
                ),
            },
            {"role": "assistant", "content": assistant},
        ],
        "source": "endgameinc/RTA",
        "mitre_ids": parsed["mitre_ids"],
    }


def _tag_pair_with_tactic(pair: dict) -> None:
    """Add mitre_tactic_id, tactic, and kill_chain_phase fields to a pair dict."""
    mitre_tactic_id = None
    if pair.get("mitre_ids"):
        for tech_id in pair["mitre_ids"]:
            tactic_id = get_tactic_for_technique(tech_id)
            if tactic_id:
                mitre_tactic_id = tactic_id
                break
    if mitre_tactic_id:
        pair["mitre_tactic_id"] = mitre_tactic_id
        tactic_name = get_tactic_name(mitre_tactic_id)
        if tactic_name:
            pair["tactic"] = tactic_name
            pair["kill_chain_phase"] = tactic_name


def _print_tactic_coverage(all_pairs: list[dict]) -> None:
    """Print tactic coverage stats for --validate-mitre."""
    from mitre_tactic_lookup import TACTIC_INFO

    tactic_counts: dict[str, int] = {tid: 0 for tid in TACTIC_INFO}
    unmapped = 0
    for pair in all_pairs:
        tid = pair.get("mitre_tactic_id")
        if tid and tid in tactic_counts:
            tactic_counts[tid] += 1
        else:
            unmapped += 1

    print("\nTactic coverage:")
    for tid in sorted(tactic_counts.keys()):
        name = TACTIC_INFO[tid]
        count = tactic_counts[tid]
        print(f"  {tid} {name}: {count}")
    total = len(all_pairs)
    mapped = total - unmapped
    print(f"Total pairs: {total}, Mapped: {mapped}, Unmapped: {unmapped}")


def generate_pairs(parsed: dict) -> list[dict]:
    """Generate 2-3 training pairs from a parsed RTA script."""
    pairs: list[dict] = []

    # Pair type 1 — always
    pair1 = _build_pair_technique_explanation(parsed)
    _tag_pair_with_tactic(pair1)
    pairs.append(pair1)

    # Pair type 2 — if common.execute calls found
    pair2 = _build_pair_command_extraction(parsed)
    if pair2 is not None:
        _tag_pair_with_tactic(pair2)
        pairs.append(pair2)

    # Pair type 3 — if multiple ATT&CK IDs
    pair3 = _build_pair_operational_context(parsed)
    if pair3 is not None:
        _tag_pair_with_tactic(pair3)
        pairs.append(pair3)

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract RTA TTP scripts into AttackLM JSONL training pairs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse scripts and print stats without writing files.",
    )
    parser.add_argument(
        "--max-scripts",
        type=int,
        default=0,
        help="Limit to N scripts (0 = all). Useful for testing.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output path for JSONL (default: data/datasets/rta_training_pairs.jsonl).",
    )
    parser.add_argument(
        "--validate-mitre",
        action="store_true",
        help="Print MITRE tactic coverage stats and exit without writing files.",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else OUTPUT_PATH

    if not RTA_DIR.exists():
        print(f"ERROR: RTA directory not found: {RTA_DIR}")
        print("  Clone the RTA repo first: data/RTA/")
        return

    print("AttackLM — Extract RTA Training Pairs")
    print(f"  RTA dir:    {RTA_DIR}")
    print(f"  Output:     {output_path}")
    print()

    # --- Discover scripts ---
    scripts = sorted(
        p for p in RTA_DIR.glob("*.py") if p.name not in ("__init__.py", "common.py")
    )

    if args.max_scripts > 0:
        scripts = scripts[: args.max_scripts]

    print(f"  Found {len(scripts)} TTP scripts to parse")

    # --- Parse and generate ---
    all_pairs: list[dict] = []
    skipped: list[str] = []
    per_script_stats: list[tuple[str, int]] = []

    for script_path in scripts:
        parsed = parse_rta_script(script_path)
        if parsed is None:
            skipped.append(script_path.name)
            continue

        pairs = generate_pairs(parsed)
        all_pairs.extend(pairs)
        per_script_stats.append((script_path.name, len(pairs)))

        # Verbose per-script output
        ids_str = ", ".join(parsed["mitre_ids"])
        print(
            f"  [{script_path.name}] "
            f"{parsed['name']!r} (ATT&CK {ids_str}) "
            f"→ {len(pairs)} pairs "
            f"({len(parsed['execute_commands'])} commands extracted)"
        )

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"  Parsed:  {len(per_script_stats)} scripts")
    print(f"  Skipped: {len(skipped)} scripts")
    if skipped:
        for s in skipped:
            print(f"    - {s}")
    print(f"  Total training pairs: {len(all_pairs)}")

    # Count by pair type
    type_counts = {"explanation": 0, "command_extraction": 0, "operational_context": 0}
    for pair in all_pairs:
        user_msg = pair["messages"][1]["content"]
        if "interrelate" in user_msg:
            type_counts["operational_context"] += 1
        elif "What commands implement" in user_msg:
            type_counts["command_extraction"] += 1
        else:
            type_counts["explanation"] += 1

    print(f"\n  By pair type:")
    for ptype, count in type_counts.items():
        print(f"    {ptype:25s}: {count}")

    # Unique MITRE IDs
    all_mitre: set[str] = set()
    for pair in all_pairs:
        all_mitre.update(pair.get("mitre_ids", []))
    print(f"\n  Unique MITRE ATT&CK IDs: {len(all_mitre)}")
    print(f"    {', '.join(sorted(all_mitre))}")

    if args.validate_mitre:
        _print_tactic_coverage(all_pairs)
        print(f"\n{'=' * 60}")
        print("  VALIDATE MITRE — No files written")
        print(f"{'=' * 60}")
        return

    if args.dry_run:
        print(f"\n{'=' * 60}")
        print("  DRY RUN — No files written")
        print(f"{'=' * 60}")

        # Show sample pair for verification
        if all_pairs:
            print(f"\n  Sample pair (first):\n")
            sample = all_pairs[0]
            print(json.dumps(sample, indent=2)[:2000])
            if len(json.dumps(sample, indent=2)) > 2000:
                print("  ... (truncated)")
        return

    # --- Write output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\n  Output written: {len(all_pairs)} pairs → {output_path}")


if __name__ == "__main__":
    main()
