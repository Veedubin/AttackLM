#!/usr/bin/env python3
# CREDITS — DATA SOURCE ATTRIBUTION
# ----------------------------------
# This script transforms data from: Atomic Red Team (redcanaryco)
# Repository: https://github.com/redcanaryco/atomic-red-team
# License:    MIT License
# Copyright:  (c) Red Canary, LLC. All rights reserved.
#
# The output JSONL is a *transformation* of upstream atomic test YAML
# files into OpenAI-style chat triples. See /ATTRIBUTION.md for full
# per-source attribution and re-distribution guidance.
# ----------------------------------
"""Deterministic extraction of Atomic Red Team YAML tests into AttackLM JSONL training pairs.

Walks ``data/atomic-red-team/atomics/`` and parses every ``.yaml`` file. For each
``atomic_tests`` entry, generates 2-3 OpenAI-style message triples:

  - **Pair type 1** — test explanation with command and cleanup (always)
  - **Pair type 2** — dependency/prerequisite setup (if dependencies exist)

``#{variable}`` placeholders are resolved from ``input_arguments`` defaults.
``PathToAtomicsFolder`` is replaced with ``C:\\AtomicRedTeam\\atomics``.

Output: ``data/datasets/atomic_red_team_training_pairs.jsonl``

Usage:
    uv run python scripts/extract_atomic_red_team_to_jsonl.py
    uv run python scripts/extract_atomic_red_team_to_jsonl.py --dry-run --max-files 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

# Import shared MITRE tactic lookup
sys.path.insert(0, str(Path(__file__).parent))
from mitre_tactic_lookup import get_tactic_for_technique, get_tactic_name

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
ATOMICS_DIR = BASE_DIR / "data" / "atomic-red-team" / "atomics"
DATASETS_DIR = BASE_DIR / "data" / "datasets"
OUTPUT_PATH = DATASETS_DIR / "atomic_red_team_training_pairs.jsonl"

# ---------------------------------------------------------------------------
# System message
# ---------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are an authorized red team adversary emulation specialist. "
    "Provide precise MITRE ATT&CK technique implementations for security "
    "validation testing."
)

# ---------------------------------------------------------------------------
# Placeholder replacement
# ---------------------------------------------------------------------------
ATOMICS_FOLDER = r"C:\AtomicRedTeam\atomics"

# Regex matching #{variable_name} placeholders
_RE_PLACEHOLDER = re.compile(r"#\{(\w+)\}")


def _resolve_placeholders(text: str, input_arguments: dict[str, Any] | None) -> str:
    """Replace ``#{var}`` placeholders with default values from *input_arguments*.

    Also replaces ``PathToAtomicsFolder`` with the canonical local path.
    Unresolved placeholders (no matching ``input_arguments`` key) are left as-is.
    """
    if not text:
        return text

    # Build lookup: argument_name → default_value
    defaults: dict[str, str] = {}
    if input_arguments and isinstance(input_arguments, dict):
        for arg_name, arg_spec in input_arguments.items():
            if isinstance(arg_spec, dict):
                default_val = arg_spec.get("default", "")
                if default_val is not None:
                    defaults[arg_name] = str(default_val)
            else:
                # Some rare cases where input_arguments value is a scalar
                defaults[arg_name] = str(arg_spec)

    def _replacer(m: re.Match) -> str:
        var = m.group(1)
        if var in defaults:
            return defaults[var]
        return m.group(0)  # leave unresolved placeholders as-is

    resolved = _RE_PLACEHOLDER.sub(_replacer, text)

    # Replace PathToAtomicsFolder with canonical path
    resolved = resolved.replace("PathToAtomicsFolder", ATOMICS_FOLDER)

    return resolved


# ---------------------------------------------------------------------------
# Executor name → code-fence language
# ---------------------------------------------------------------------------
EXECUTOR_LANG: dict[str, str] = {
    "command_prompt": "batch",
    "powershell": "powershell",
    "bash": "bash",
    "sh": "bash",
    "python": "python",
}


def _lang_for_executor(executor_name: str) -> str:
    """Return code-fence language label for an executor name."""
    lower = executor_name.lower()
    for key, lang in EXECUTOR_LANG.items():
        if key in lower:
            return lang
    return "text"


# ---------------------------------------------------------------------------
# Parse a single YAML file
# ---------------------------------------------------------------------------
def parse_yaml_file(filepath: Path) -> dict[str, Any] | None:
    """Parse one Atomic Red Team YAML file.

    Returns a dict with keys: ``attack_technique``, ``display_name``,
    ``atomic_tests``, or ``None`` if the file is not parseable.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"  [WARN] YAML parse error in {filepath}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  [WARN] Error reading {filepath}: {e}", file=sys.stderr)
        return None

    if not isinstance(data, dict):
        return None

    attack_technique = data.get("attack_technique", "")
    display_name = data.get("display_name", "")
    atomic_tests = data.get("atomic_tests", [])

    if not attack_technique or not isinstance(atomic_tests, list):
        return None

    return {
        "attack_technique": str(attack_technique),
        "display_name": str(display_name),
        "atomic_tests": atomic_tests,
    }


# ---------------------------------------------------------------------------
# Build training pairs
# ---------------------------------------------------------------------------


def _tag_pair_with_tactic(pair: dict[str, Any]) -> None:
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


def _print_tactic_coverage(all_pairs: list[dict[str, Any]]) -> None:
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


def _build_pair_test_explanation(
    *,
    attack_technique: str,
    display_name: str,
    test_name: str,
    description: str,
    platforms: list[str],
    executor_name: str,
    command: str,
    cleanup_command: str | None,
    elevation_required: bool,
    input_arguments: dict[str, Any] | None,
) -> dict[str, Any]:
    """Pair type 1: test explanation with command and cleanup."""
    cmd_resolved = _resolve_placeholders(command, input_arguments)
    cleanup_resolved = (
        _resolve_placeholders(cleanup_command, input_arguments)
        if cleanup_command
        else None
    )

    platform_str = ", ".join(platforms)
    lang = _lang_for_executor(executor_name)

    elevation_note = ""
    if elevation_required:
        elevation_note = (
            "\n\n> **Requires elevated privileges (run as administrator/root).**"
        )

    cleanup_block = ""
    if cleanup_resolved and cleanup_resolved.strip():
        cleanup_block = f"\n\n### Cleanup\n```{lang}\n{cleanup_resolved.strip()}\n```"

    assistant = (
        f"## {test_name}\n"
        f"MITRE ATT&CK: {attack_technique} ({display_name})\n\n"
        f"{description.strip()}\n\n"
        f"### Command ({executor_name})\n"
        f"```{lang}\n{cmd_resolved.strip()}\n```"
        f"{cleanup_block}"
        f"{elevation_note}"
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {
                "role": "user",
                "content": f"How do I test {attack_technique} ({display_name}) on {platform_str}?",
            },
            {"role": "assistant", "content": assistant},
        ],
        "source": "redcanaryco/atomic-red-team",
        "mitre_ids": [attack_technique],
        "platforms": platforms,
    }


def _build_pair_dependency_setup(
    *,
    attack_technique: str,
    display_name: str,
    test_name: str,
    dependencies: list[dict[str, Any]],
    input_arguments: dict[str, Any] | None,
) -> dict[str, Any]:
    """Pair type 2: dependency/prerequisite setup instructions."""
    dep_lines: list[str] = []
    for dep in dependencies:
        dep_desc = _resolve_placeholders(
            dep.get("description", ""), input_arguments
        ).strip()
        prereq = _resolve_placeholders(
            dep.get("prereq_command", ""), input_arguments
        ).strip()
        get_prereq = _resolve_placeholders(
            dep.get("get_prereq_command", ""), input_arguments
        ).strip()

        block = f"- {dep_desc}"
        if prereq:
            block += f"\n  Check:\n  ```\n  {prereq}\n  ```"
        if get_prereq:
            block += f"\n  Install:\n  ```\n  {get_prereq}\n  ```"
        dep_lines.append(block)

    assistant = f"Before running {test_name}:\n\n" + "\n\n".join(dep_lines)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {
                "role": "user",
                "content": f"What prerequisites are needed before running {test_name} ({attack_technique})?",
            },
            {"role": "assistant", "content": assistant},
        ],
        "source": "redcanaryco/atomic-red-team",
        "mitre_ids": [attack_technique],
    }


def generate_pairs_from_test(
    *,
    attack_technique: str,
    display_name: str,
    test: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate 2-3 training pairs from a single atomic test entry."""
    pairs: list[dict[str, Any]] = []

    test_name = test.get("name", "Unknown Test")
    description = test.get("description", "")
    platforms = test.get("supported_platforms", [])
    if not platforms:
        platforms = ["unknown"]

    input_arguments = test.get("input_arguments")
    executor = test.get("executor")

    # Skip tests without an executor or with manual-only executor (no command)
    if not executor or not isinstance(executor, dict):
        return pairs

    executor_name = executor.get("name", "")
    command = executor.get("command", "")

    # Manual tests have 'steps' instead of 'command' — skip them
    if not command or not command.strip():
        return pairs

    cleanup_command = executor.get("cleanup_command", "")
    elevation_required = executor.get("elevation_required", False)

    # Pair type 1 — test explanation (always)
    pair1 = _build_pair_test_explanation(
        attack_technique=attack_technique,
        display_name=display_name,
        test_name=test_name,
        description=description,
        platforms=platforms,
        executor_name=executor_name,
        command=command,
        cleanup_command=cleanup_command or None,
        elevation_required=elevation_required,
        input_arguments=input_arguments,
    )
    _tag_pair_with_tactic(pair1)
    pairs.append(pair1)

    # Pair type 2 — dependency setup (only if dependencies exist)
    dependencies = test.get("dependencies")
    if dependencies and isinstance(dependencies, list) and len(dependencies) > 0:
        pair2 = _build_pair_dependency_setup(
            attack_technique=attack_technique,
            display_name=display_name,
            test_name=test_name,
            dependencies=dependencies,
            input_arguments=input_arguments,
        )
        _tag_pair_with_tactic(pair2)
        pairs.append(pair2)

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Atomic Red Team YAML tests into AttackLM JSONL training pairs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse files and print stats without writing output.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit to N YAML files (0 = all). Useful for testing.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output path for JSONL.",
    )
    parser.add_argument(
        "--validate-mitre",
        action="store_true",
        help="Print MITRE tactic coverage stats and exit without writing files.",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else OUTPUT_PATH

    if not ATOMICS_DIR.exists():
        print(f"ERROR: Atomics directory not found: {ATOMICS_DIR}")
        print("  Clone the Atomic Red Team repo first: data/atomic-red-team/")
        return

    print("AttackLM — Extract Atomic Red Team Training Pairs")
    print(f"  Atomics dir: {ATOMICS_DIR}")
    print(f"  Output:      {output_path}")
    print()

    # --- Discover YAML files (skip Indexes/) ---
    yaml_files = sorted(
        p for p in ATOMICS_DIR.rglob("*.yaml") if "Indexes" not in p.parts
    )

    if args.max_files > 0:
        yaml_files = yaml_files[: args.max_files]

    print(f"  Found {len(yaml_files)} YAML files to parse")

    # --- Parse and generate ---
    all_pairs: list[dict[str, Any]] = []
    skipped: list[str] = []
    per_file_stats: list[tuple[str, int, int]] = []  # (filename, tests, pairs)

    total_tests = 0
    skipped_manual = 0
    skipped_no_executor = 0

    for yaml_path in yaml_files:
        parsed = parse_yaml_file(yaml_path)
        if parsed is None:
            skipped.append(yaml_path.name)
            continue

        attack_technique = parsed["attack_technique"]
        display_name = parsed["display_name"]
        file_pairs = 0
        file_tests = 0

        for test in parsed["atomic_tests"]:
            if not isinstance(test, dict):
                continue

            file_tests += 1
            total_tests += 1

            # Check for manual-only executor before generating pairs
            executor = test.get("executor")
            if not executor or not isinstance(executor, dict):
                skipped_no_executor += 1
                continue

            command = executor.get("command", "")
            if not command or not command.strip():
                skipped_manual += 1
                continue

            pairs = generate_pairs_from_test(
                attack_technique=attack_technique,
                display_name=display_name,
                test=test,
            )
            all_pairs.extend(pairs)
            file_pairs += len(pairs)

        per_file_stats.append((yaml_path.name, file_tests, file_pairs))

        # Verbose per-file output
        print(
            f"  [{yaml_path.name}] "
            f"{attack_technique} ({display_name}) "
            f"→ {file_tests} tests, {file_pairs} pairs"
        )

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"  YAML files parsed:  {len(per_file_stats)}")
    print(f"  YAML files skipped: {len(skipped)}")
    if skipped:
        for s in skipped:
            print(f"    - {s}")
    print(f"  Total atomic tests: {total_tests}")
    print(f"  Skipped (no executor): {skipped_no_executor}")
    print(f"  Skipped (manual only): {skipped_manual}")
    print(f"  Total training pairs: {len(all_pairs)}")

    # Count by pair type
    type_counts = {"test_explanation": 0, "dependency_setup": 0}
    for pair in all_pairs:
        user_msg = pair["messages"][1]["content"]
        if "prerequisites" in user_msg:
            type_counts["dependency_setup"] += 1
        else:
            type_counts["test_explanation"] += 1

    print(f"\n  By pair type:")
    for ptype, count in type_counts.items():
        print(f"    {ptype:25s}: {count}")

    # Unique MITRE IDs
    all_mitre: set[str] = set()
    all_platforms: set[str] = set()
    for pair in all_pairs:
        all_mitre.update(pair.get("mitre_ids", []))
        all_platforms.update(pair.get("platforms", []))
    print(f"\n  Unique MITRE ATT&CK IDs: {len(all_mitre)}")
    print(
        f"    {', '.join(sorted(all_mitre)[:20])}{'...' if len(all_mitre) > 20 else ''}"
    )
    print(f"  Platforms: {', '.join(sorted(all_platforms))}")

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
            sample_json = json.dumps(sample, indent=2, ensure_ascii=False)
            print(sample_json[:3000])
            if len(sample_json) > 3000:
                print("  ... (truncated)")

            # Show dependency pair if available
            dep_pairs = [
                p for p in all_pairs if "prerequisites" in p["messages"][1]["content"]
            ]
            if dep_pairs:
                print(f"\n  Sample dependency pair:\n")
                dep_json = json.dumps(dep_pairs[0], indent=2, ensure_ascii=False)
                print(dep_json[:3000])
                if len(dep_json) > 3000:
                    print("  ... (truncated)")
        return

    # --- Write output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n  Output written: {len(all_pairs)} pairs → {output_path}")


if __name__ == "__main__":
    main()
