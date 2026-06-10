#!/usr/bin/env python3
# CREDITS — DATA SOURCE ATTRIBUTION
# ----------------------------------
# This script transforms data from: MITRE Caldera plugins
#   - Stockpile:  https://github.com/mitre/stockpile
#   - Arsenal:    https://github.com/mitre/caldera (plugins/arsenal/)
#   - Manx:       https://github.com/mitre/caldera (plugins/manx/)
#   - Access:     https://github.com/mitre/caldera (plugins/access/)
# License:    Apache License 2.0
# Copyright:  (c) MITRE Corporation. Approved for public release.
#
# The output JSONL is a *transformation* of upstream Caldera ability
# YAML into chat triples. See /ATTRIBUTION.md for full details.
# ----------------------------------
"""Deterministic extraction of Caldera plugin repos into AttackLM JSONL training pairs.

Walks each plugin's data/abilities/ directory tree, parses YAML ability files,
and generates two types of training pairs per ability-platform-executor combination:
  1. Technique explanation (structured markdown)
  2. Command extraction (raw commands + cleanup)

Usage:
    uv run python scripts/extract_caldera_plugins_to_jsonl.py [--dry-run] [--max-abilities N]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

# Import shared MITRE tactic lookup
sys.path.insert(0, str(Path(__file__).parent))
from mitre_tactic_lookup import (
    get_tactic_for_technique,
    get_tactic_name,
    resolve_tactic_name_to_id,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "datasets" / "caldera_plugins_training_pairs.jsonl"

SYSTEM_MSG_DEFAULT = (
    "You are an authorized red team adversary emulation specialist. "
    "Provide precise MITRE ATT&CK technique implementations for security validation testing."
)

SYSTEM_MSG_ATLAS = (
    "You are an authorized red team adversary emulation specialist focused on "
    "AI/ML system security. Provide precise MITRE ATLAS technique implementations "
    "for adversarial ML security validation testing."
)

# Plugin definitions: (repo_dir, source_label, system_msg, tags)
PLUGINS = [
    {
        "dir": DATA_DIR / "stockpile" / "data" / "abilities",
        "source": "mitre/stockpile",
        "system_msg": SYSTEM_MSG_DEFAULT,
        "tags": [],
    },
    {
        "dir": DATA_DIR / "arsenal" / "data" / "abilities",
        "source": "mitre-atlas/arsenal",
        "system_msg": SYSTEM_MSG_ATLAS,
        "tags": ["atlas", "ai_ml"],
    },
    {
        "dir": DATA_DIR / "manx" / "data" / "abilities",
        "source": "mitre/manx",
        "system_msg": SYSTEM_MSG_DEFAULT,
        "tags": ["reverse_shell"],
    },
    {
        "dir": DATA_DIR / "access" / "data" / "abilities",
        "source": "mitre/access",
        "system_msg": SYSTEM_MSG_DEFAULT,
        "tags": ["reconnaissance"],
    },
]

# Executor → code-fence language mapping
EXECUTOR_LANG = {
    "psh": "powershell",
    "pwsh": "powershell",
    "cmd": "batch",
    "sh": "bash",
    "python": "python",
    "shellcode_amd64": None,  # skip shellcode
    "shellcode_x86": None,
}

SKIP_EXECUTORS = {"shellcode_amd64", "shellcode_x86"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_technique(ability: dict[str, Any]) -> tuple[str, str]:
    """Extract (attack_id, technique_name) from either dict or flat format.

    Stockpile/manx/access use: technique: {attack_id: ..., name: ...}
    Arsenal uses:              technique_id: AML.T0037, technique_name: Data from local system
    Some Arsenal YAMLs set technique: null and use flat fields instead.
    """
    tech = ability.get("technique")
    if isinstance(tech, dict) and tech:
        attack_id = tech.get("attack_id", "Unknown")
        tech_name = tech.get("name", "Unknown Technique")
    else:
        # Flat format (Arsenal) or technique is null/empty
        attack_id = ability.get("technique_id", "Unknown")
        tech_name = ability.get("technique_name", "Unknown Technique")
    return str(attack_id), str(tech_name)


def resolve_executor_lang(executor: str) -> str | None:
    """Map executor name to code-fence language. Returns None for skipped executors."""
    if executor in SKIP_EXECUTORS:
        return None
    if executor in EXECUTOR_LANG:
        return EXECUTOR_LANG[executor]
    # For unknown executors, try a best-effort match
    lower = executor.lower()
    if "psh" in lower or "pwsh" in lower:
        return "powershell"
    if "cmd" in lower:
        return "batch"
    if "sh" in lower:
        return "bash"
    if "python" in lower:
        return "python"
    return None


def expand_platform_key(platform_key: str) -> list[str]:
    """Split composite platform keys like 'darwin,linux' into individual platforms."""
    if "," in platform_key:
        return [p.strip() for p in platform_key.split(",")]
    return [platform_key]


def build_technique_explanation_pair(
    *,
    name: str,
    attack_id: str,
    technique_name: str,
    tactic: str,
    description: str,
    platform: str,
    executor: str,
    command: str,
    cleanup: str | None,
    system_msg: str,
    source: str,
    tags: list[str],
) -> dict[str, Any]:
    """Build pair type 1: technique explanation (structured markdown)."""
    lang = resolve_executor_lang(executor) or "text"

    user_content = f"Explain how {name} (ATT&CK {attack_id}) works on {platform}."
    if "atlas" in tags or "ai_ml" in tags:
        user_content = f"Explain how {name} (ATLAS {attack_id}) works on {platform}."

    # Pre-resolve tactic name for display (will be normalized fully below)
    display_tactic = tactic
    tactic_id_check = resolve_tactic_name_to_id(tactic) if tactic else None
    if tactic_id_check:
        display_tactic = get_tactic_name(tactic_id_check) or tactic

    assistant_lines = [
        f"## {name}",
        f"MITRE ATT&CK: {attack_id} ({technique_name})",
        f"Tactic: {display_tactic}",
        "",
        description,
        "",
        f"### Commands ({platform}/{executor})",
        f"```{lang}",
        command,
        "```",
    ]
    if cleanup:
        assistant_lines += [
            "",
            "### Cleanup",
            f"```{lang}",
            cleanup,
            "```",
        ]

    mitre_ids = [attack_id]
    # Normalize tactic: try YAML tactic first, then fall back to technique lookup
    resolved_tactic_id = resolve_tactic_name_to_id(tactic) if tactic else None
    if not resolved_tactic_id and mitre_ids:
        for tech_id in mitre_ids:
            resolved_tactic_id = get_tactic_for_technique(tech_id)
            if resolved_tactic_id:
                break
    # For Arsenal/ATLAS, default to Prompt Injection tactic
    if not resolved_tactic_id and ("atlas" in tags or "ai_ml" in tags):
        resolved_tactic_id = "TA0040"
    # Final fallback: Discovery
    if not resolved_tactic_id:
        resolved_tactic_id = "TA0007"
    resolved_tactic_name = get_tactic_name(resolved_tactic_id) or tactic

    pair: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "\n".join(assistant_lines)},
        ],
        "source": source,
        "mitre_ids": mitre_ids,
        "mitre_tactic_id": resolved_tactic_id,
        "tactic": resolved_tactic_name,
        "kill_chain_phase": resolved_tactic_name,
        "platform": platform,
    }
    if tags:
        pair["tags"] = tags
    return pair


def build_command_extraction_pair(
    *,
    name: str,
    attack_id: str,
    platform: str,
    command: str,
    cleanup: str | None,
    system_msg: str,
    source: str,
    tactic: str,
    tags: list[str],
) -> dict[str, Any]:
    """Build pair type 2: command extraction (raw commands)."""
    id_label = f"ATT&CK {attack_id}"
    if "atlas" in tags or "ai_ml" in tags:
        id_label = f"ATLAS {attack_id}"

    user_content = f"What commands implement {name} ({id_label}) on {platform}?"

    assistant_content = command
    if cleanup:
        assistant_content += f"\n\nCleanup: {cleanup}"

    # Normalize tactic: try YAML tactic first, then fall back to technique lookup
    resolved_tactic_id = resolve_tactic_name_to_id(tactic) if tactic else None
    if not resolved_tactic_id and attack_id:
        resolved_tactic_id = get_tactic_for_technique(attack_id)
    if not resolved_tactic_id and ("atlas" in tags or "ai_ml" in tags):
        resolved_tactic_id = "TA0040"
    if not resolved_tactic_id:
        resolved_tactic_id = "TA0007"
    resolved_tactic_name = get_tactic_name(resolved_tactic_id) or tactic

    pair: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "source": source,
        "mitre_ids": [attack_id],
        "mitre_tactic_id": resolved_tactic_id,
        "tactic": resolved_tactic_name,
        "kill_chain_phase": resolved_tactic_name,
        "platform": platform,
    }
    if tags:
        pair["tags"] = tags
    return pair


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------


def process_ability(
    ability: dict[str, Any],
    source: str,
    system_msg: str,
    tags: list[str],
) -> list[dict[str, Any]]:
    """Process a single YAML ability entry, producing training pairs for each
    platform/executor combination."""
    pairs: list[dict[str, Any]] = []

    name = ability.get("name", "Unknown Ability")
    description = ability.get("description", "").strip()
    tactic = ability.get("tactic", "unknown")
    attack_id, technique_name = extract_technique(ability)
    platforms = ability.get("platforms", {})

    if not platforms:
        return pairs

    for platform_key, executors in platforms.items():
        expanded_platforms = expand_platform_key(platform_key)

        # executors can be a dict {executor_name: details} when single key,
        # or could be a dict with executor names as keys
        if not isinstance(executors, dict):
            continue

        for executor, details in executors.items():
            # Skip non-executor keys like 'parsers', 'requirements'
            if not isinstance(details, dict):
                continue
            # Skip shellcode and other non-command executors
            if executor in SKIP_EXECUTORS:
                continue

            command = details.get("command", "")
            if isinstance(command, str):
                command = command.strip()
            if not command:
                continue

            cleanup = details.get("cleanup", "")
            if isinstance(cleanup, str):
                cleanup = cleanup.strip() or None
            else:
                cleanup = None

            for platform in expanded_platforms:
                # Pair type 1: technique explanation
                pairs.append(
                    build_technique_explanation_pair(
                        name=name,
                        attack_id=attack_id,
                        tactic=tactic,
                        description=description,
                        platform=platform,
                        executor=executor,
                        command=command,
                        cleanup=cleanup,
                        system_msg=system_msg,
                        source=source,
                        tags=tags,
                        technique_name=technique_name,
                    )
                )
                # Pair type 2: command extraction
                pairs.append(
                    build_command_extraction_pair(
                        name=name,
                        attack_id=attack_id,
                        tactic=tactic,
                        platform=platform,
                        command=command,
                        cleanup=cleanup,
                        system_msg=system_msg,
                        source=source,
                        tags=tags,
                    )
                )

    return pairs


def process_yaml_file(
    filepath: Path, source: str, system_msg: str, tags: list[str]
) -> list[dict[str, Any]]:
    """Parse a single YAML file (which contains a list of abilities)."""
    pairs: list[dict[str, Any]] = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"  [WARN] YAML parse error in {filepath}: {e}", file=sys.stderr)
        return pairs
    except Exception as e:
        print(f"  [WARN] Error reading {filepath}: {e}", file=sys.stderr)
        return pairs

    if not isinstance(data, list):
        return pairs

    for ability in data:
        if not isinstance(ability, dict):
            continue
        pairs.extend(process_ability(ability, source, system_msg, tags))

    return pairs


def extract_plugin(
    abilities_dir: Path,
    source: str,
    system_msg: str,
    tags: list[str],
    max_abilities: int | None = None,
) -> list[dict[str, Any]]:
    """Walk a plugin's abilities directory and extract all training pairs."""
    all_pairs: list[dict[str, Any]] = []
    ability_count = 0

    if not abilities_dir.is_dir():
        print(f"  [SKIP] Directory not found: {abilities_dir}", file=sys.stderr)
        return all_pairs

    yaml_files = sorted(abilities_dir.rglob("*.yml"))
    print(f"  Scanning {len(yaml_files)} YAML files in {abilities_dir}")

    for filepath in yaml_files:
        pairs = process_yaml_file(filepath, source, system_msg, tags)
        if pairs:
            ability_count += 1
            all_pairs.extend(pairs)

        if max_abilities and ability_count >= max_abilities:
            break

    print(f"  Extracted {len(all_pairs)} pairs from {ability_count} abilities")
    return all_pairs


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Caldera plugin YAML abilities into AttackLM JSONL training pairs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary only, do not write output file",
    )
    parser.add_argument(
        "--max-abilities",
        type=int,
        default=None,
        help="Limit number of abilities processed per plugin (for testing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output file path",
    )
    parser.add_argument(
        "--validate-mitre",
        action="store_true",
        help="Print MITRE tactic coverage stats and exit without writing files.",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else OUTPUT_FILE
    total_pairs = 0
    per_source_counts: dict[str, int] = {}
    all_pairs: list[dict[str, Any]] = []

    print("AttackLM Caldera Plugin Extractor")
    print("=" * 50)

    for plugin in PLUGINS:
        print(f"\nProcessing {plugin['source']}...")
        pairs = extract_plugin(
            abilities_dir=plugin["dir"],
            source=plugin["source"],
            system_msg=plugin["system_msg"],
            tags=plugin["tags"],
            max_abilities=args.max_abilities,
        )
        per_source_counts[plugin["source"]] = len(pairs)
        total_pairs += len(pairs)
        all_pairs.extend(pairs)

    print(f"\n{'=' * 50}")
    print(f"Total training pairs: {total_pairs}")
    print("\nPer-source breakdown:")
    for source, count in per_source_counts.items():
        print(f"  {source}: {count} pairs")

    if args.validate_mitre:
        _print_tactic_coverage(all_pairs)
        print(f"\n{'=' * 50}")
        print("VALIDATE MITRE — No output file written.")
        return

    if args.dry_run:
        print("\n[DRY RUN] No output file written.")
        # Show a sample pair
        if all_pairs:
            print("\n--- Sample pair (first entry) ---")
            sample = all_pairs[0]
            print(json.dumps(sample, indent=2, ensure_ascii=False)[:2000])
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nWritten {total_pairs} pairs to {output_path}")


if __name__ == "__main__":
    main()
