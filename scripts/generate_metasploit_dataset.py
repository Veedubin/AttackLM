#!/usr/bin/env python3
"""
generate_metasploit_dataset.py — Convert the Metasploit manifest into
instruction-following training pairs in the same JSONL format used by the
rest of the AttackLM pipeline.

Strategy
--------
The Metasploit corpus is rich but heterogeneous. We synthesize multiple
question phrasings per module to maximize training diversity:

1. "Tool usage" — how to invoke the module in msfconsole
2. "CVE lookup" — given a CVE, recommend the matching module
3. "Technique mapping" — given an MITRE technique, list MSF modules
4. "Documentation-grounded" — scenario + msfconsole output as assistant
5. "Detection" — given the module, what telemetry does it produce

All outputs follow the OpenAI messages format and the AttackLM Few-Shot
prompt contract (technique name + MITRE ID, exact commands, artifacts, cleanup).

Usage:
    python generate_metasploit_dataset.py
    python generate_metasploit_dataset.py --per-module 2 --max-modules 500
    python generate_metasploit_dataset.py --no-cve --no-detect
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
MANIFESTS_DIR = BASE_DIR / "data" / "manifests"
DATASETS_DIR = BASE_DIR / "data" / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Tactic display names
# ---------------------------------------------------------------------------
TACTIC_DISPLAY = {
    "execution": "EXECUTION",
    "persistence": "PERSISTENCE",
    "privilege_escalation": "PRIVILEGE ESCALATION",
    "defense_evasion": "DEFENSE EVASION",
    "credential_access": "CREDENTIAL ACCESS",
    "discovery": "DISCOVERY",
    "lateral_movement": "LATERAL MOVEMENT",
    "command_and_control": "COMMAND AND CONTROL",
    "collection": "COLLECTION",
    "exfiltration": "EXFILTRATION",
}

# ---------------------------------------------------------------------------
# Question templates (the user turn)
# ---------------------------------------------------------------------------
TOOL_QUESTIONS = [
    "How do I run the {name} module in Metasploit? Show the full msfconsole invocation with options.",
    "Demonstrate use of `{module_path}` to accomplish {short_name}. Include setting RHOSTS and running.",
    "I need to use the Metasploit module `{module_path}` — give me the exact msfconsole commands and required options.",
    "Walk me through running `{module_path}` in msfconsole. What options must be set?",
    "Provide the complete msfconsole session for invoking `{module_path}`.",
    "I have a Meterpreter session. Run the post module `{module_path}` — what commands do I type?",
    "Configure and execute the Metasploit module `{module_path}` against a target on port {{RPORT}}. Show the full session.",
]

CVE_QUESTIONS = [
    "I need to exploit {cves}. Which Metasploit module should I use and how do I run it?",
    "Is there a Metasploit module for {cves}? Give me the full command sequence.",
    "I'm pentesting a target vulnerable to {cves}. Walk me through the msfconsole exploit chain.",
    "Show me the Metasploit module that targets {cves} and the exact commands to exploit it.",
]

TECHNIQUE_QUESTIONS = [
    "Which Metasploit modules implement MITRE ATT&CK {tech_list}? Give the full module paths and one-line summary for each.",
    "List all Metasploit modules that map to MITRE technique {tech_list}.",
    "I'm working the {tactic} phase of an engagement. What Metasploit modules map to {tech_list}?",
    "As part of {tactic}, I want to use Metasploit modules covering {tech_list}. List the top 3 most useful with example invocations.",
]

DETECTION_QUESTIONS = [
    "I'm a detection engineer. The Metasploit module `{module_path}` was just run in my environment. What telemetry would I expect?",
    "What Sysmon/EDR events does `{module_path}` produce? Be specific about Event IDs and process trees.",
    "Detection rules for `{module_path}` — what should I monitor for?",
    "If `{module_path}` is executed in our network, what Windows Event IDs and process behaviors should trigger an alert?",
]

DOC_QUESTIONS = [
    "Walk me through the {name} Metasploit module end-to-end. Include setup, options, and a sample msfconsole session.",
    "I'm a red teamer. Demonstrate the {name} module with a realistic scenario, including the msfconsole output I should expect.",
    "Show me how {name} works in practice — set up the target, configure the module, run it, and what the output looks like.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _short_name(name: str) -> str:
    """Compress a long module name into a phrase for question templates."""
    name = re.sub(
        r"\b(exploit|module|post|gather|auxiliary|scan|bruteforce|remote|local)\b",
        "",
        name,
        flags=re.IGNORECASE,
    )
    return name.strip().lower() or "the target"


def _stable_cmd_text(record: dict) -> str | None:
    """Pick the first msfconsole command from a Metasploit record."""
    for cmd in record.get("commands", []):
        text = (cmd.get("command") or "").strip()
        if text:
            return text
    return None


def _format_options(advanced: bool, opts: list[dict]) -> str:
    """Render an options list for the assistant response."""
    if not opts:
        return ""
    lines = []
    for o in opts:
        opt_type = o.get("type") or "?"
        name = o.get("name") or "?"
        req = "required" if o.get("required") else "optional"
        default = o.get("default")
        desc = o.get("description") or ""
        if default is None or default == "nil":
            default_str = "(no default)"
        else:
            default_str = repr(default)
        line = f"- `{name}` ({opt_type}, {req}, default={default_str})"
        if desc:
            line += f" — {desc}"
        lines.append(line)
    header = "**Advanced Options:**\n" if advanced else "**Options:**\n"
    return header + "\n".join(lines) + "\n"


def _format_technique_header(record: dict) -> str:
    """MITRE technique line(s) for the response header."""
    techs = record.get("mitre_techniques") or []
    if not techs:
        # Fall back to the module's primary MITRE tag (from manifest)
        return ""
    parts = [f"**MITRE ATT&CK:** {t['id']} — {t['name']}" for t in techs]
    return "\n".join(parts) + "\n"


def _format_references(record: dict) -> str:
    """CVE / EDB / MSB / BID references block."""
    bits: list[str] = []
    for key, label in (
        ("cves", "CVE"),
        ("edb_ids", "EDB"),
        ("msb_ids", "MSB"),
        ("osvdb_ids", "OSVDB"),
        ("bid_ids", "BID"),
    ):
        for v in record.get(key, []):
            bits.append(f"- {label}: {v}")
    if not bits and record.get("urls"):
        for u in record["urls"][:2]:
            bits.append(f"- URL: {u}")
    return "\n".join(bits)


def _format_notes(record: dict) -> str:
    notes = record.get("notes") or {}
    parts: list[str] = []
    for key, label in (
        ("stability", "Stability"),
        ("side_effects", "Side Effects"),
        ("reliability", "Reliability"),
        ("aka", "AKA"),
    ):
        vals = notes.get(key) or []
        if vals:
            parts.append(f"**{label}:** {', '.join(vals)}")
    return "\n".join(parts)


def _scenario_block(record: dict) -> str:
    """Wrap the documentation scenario into a code-fenced block."""
    out = (record.get("scenario_output") or "").strip()
    if not out:
        return ""
    # Truncate long outputs to ~60 lines to keep training pairs manageable
    lines = out.splitlines()
    if len(lines) > 60:
        lines = lines[:60] + ["... (truncated)"]
    return "**Sample msfconsole Output:**\n```\n" + "\n".join(lines) + "\n```"


# ---------------------------------------------------------------------------
# Pair builders
# ---------------------------------------------------------------------------
def build_tool_pair(record: dict, system_prompt: str) -> dict | None:
    cmd = _stable_cmd_text(record)
    if not cmd:
        return None
    name = record.get("name") or record.get("module_path")
    module_path = record.get("module_path") or ""
    short = _short_name(name)
    q = random.choice(TOOL_QUESTIONS).format(
        name=name,
        short_name=short,
        module_path=module_path,
    )
    response = f"**Module: `{module_path}`** — {name}\n\n"
    response += _format_technique_header(record)
    response += "\n**Invocation (msfconsole):**\n"
    response += f"```\n{cmd}\n```\n"
    response += "\n" + _format_options(False, record.get("options") or [])
    if record.get("advanced_options"):
        response += "\n" + _format_options(True, record.get("advanced_options") or [])
    refs = _format_references(record)
    if refs:
        response += f"\n**References:**\n{refs}\n"
    notes = _format_notes(record)
    if notes:
        response += f"\n{notes}\n"
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
            {"role": "assistant", "content": response.strip()},
        ]
    }


def build_cve_pair(record: dict, system_prompt: str) -> dict | None:
    cves = record.get("cves") or []
    if not cves:
        return None
    cmd = _stable_cmd_text(record)
    if not cmd:
        return None
    cve_str = ", ".join(cves[:3])
    q = random.choice(CVE_QUESTIONS).format(cves=cve_str)
    name = record.get("name") or record.get("module_path")
    module_path = record.get("module_path") or ""
    response = f"**Module: `{module_path}`** — {name}\n\n"
    response += f"**Vulnerabilities:** {', '.join(cves)}\n\n"
    response += _format_technique_header(record)
    response += "\n**Invocation:**\n```\n" + cmd + "\n```\n"
    response += "\n" + _format_options(False, record.get("options") or [])
    refs = _format_references(record)
    if refs:
        response += f"\n**References:**\n{refs}\n"
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
            {"role": "assistant", "content": response.strip()},
        ]
    }


def build_technique_pair(
    record: dict, system_prompt: str, tactic_disp: str
) -> dict | None:
    techs = record.get("mitre_techniques") or []
    if not techs:
        return None
    tech_list = ", ".join(t["id"] for t in techs)
    q = random.choice(TECHNIQUE_QUESTIONS).format(
        tech_list=tech_list,
        tactic=tactic_disp,
    )
    name = record.get("name") or record.get("module_path")
    module_path = record.get("module_path") or ""
    response = f"**MITRE {tech_list}** — {tactic_disp}\n\n"
    response += f"**Module: `{module_path}`** — {name}\n\n"
    response += f"**Description:** {record.get('description', '').strip() or '(see module source)'}\n\n"
    response += f"**Rank:** {record.get('rank') or 'n/a'}, "
    response += f"**Platform:** {', '.join(record.get('platform') or ['n/a'])}\n"
    refs = _format_references(record)
    if refs:
        response += f"\n**References:**\n{refs}\n"
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
            {"role": "assistant", "content": response.strip()},
        ]
    }


def build_detection_pair(record: dict, system_prompt: str) -> dict | None:
    module_path = record.get("module_path") or ""
    if not module_path:
        return None
    name = record.get("name") or module_path
    q = random.choice(DETECTION_QUESTIONS).format(module_path=module_path)
    response = f"**Module: `{module_path}`** — {name}\n\n"
    platform = record.get("platform") or []
    if platform:
        response += f"**Platforms:** {', '.join(platform)}\n"
    notes = _format_notes(record)
    if notes:
        response += f"\n{notes}\n"
    response += "\n**Telemetry to monitor:**\n"
    if "windows" in platform:
        response += (
            "- Sysmon Event ID 1 (Process Create) for msfconsole/msf payload child processes\n"
            "- Sysmon Event ID 3 (Network Connect) to attacker LHOST\n"
            "- Sysmon Event ID 11 (File Create) for stager temp files\n"
            "- Windows Event ID 4688 (Process Created) with command line auditing\n"
        )
    elif "linux" in platform or "unix" in platform:
        response += (
            "- Auditd execve syscall for payload execution\n"
            "- Auditd SYSCALL connect() to attacker LHOST\n"
            "- File integrity monitoring on WritableDir path\n"
            "- Process tracking for shell/Meterpreter forks\n"
        )
    else:
        response += (
            "- Network IDS alerts on payload callbacks to LHOST\n"
            "- Process telemetry on the target system\n"
        )
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
            {"role": "assistant", "content": response.strip()},
        ]
    }


def build_doc_pair(record: dict, system_prompt: str) -> dict | None:
    if not record.get("scenario_output"):
        return None
    name = record.get("name") or record.get("module_path")
    module_path = record.get("module_path") or ""
    q = random.choice(DOC_QUESTIONS).format(name=name)
    response = f"**Module: `{module_path}`** — {name}\n\n"
    response += _format_technique_header(record)
    desc = (record.get("description") or "").strip()
    if desc:
        response += f"**Description:**\n{desc}\n\n"
    response += _scenario_block(record) + "\n"
    if record.get("authors"):
        response += f"\n**Authors:** {', '.join(record['authors'][:3])}\n"
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
            {"role": "assistant", "content": response.strip()},
        ]
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Metasploit manifest into training JSONL."
    )
    parser.add_argument(
        "--per-module",
        type=int,
        default=3,
        help="How many pair types to emit per module (random subset).",
    )
    parser.add_argument(
        "--max-modules",
        type=int,
        default=0,
        help="Cap the number of modules processed (0 = no cap).",
    )
    parser.add_argument("--no-cve", action="store_true", help="Skip CVE-keyed pairs.")
    parser.add_argument(
        "--no-detect", action="store_true", help="Skip detection pairs."
    )
    parser.add_argument(
        "--no-doc", action="store_true", help="Skip documentation-grounded pairs."
    )
    parser.add_argument("--no-tool", action="store_true", help="Skip tool usage pairs.")
    parser.add_argument(
        "--no-technique", action="store_true", help="Skip technique-mapping pairs."
    )
    args = parser.parse_args()

    enabled_builders: dict[str, callable] = {}
    if not args.no_tool:
        enabled_builders["tool"] = build_tool_pair
    if not args.no_cve:
        enabled_builders["cve"] = build_cve_pair
    if not args.no_technique:
        enabled_builders["technique"] = build_technique_pair
    if not args.no_detect:
        enabled_builders["detection"] = build_detection_pair
    if not args.no_doc:
        enabled_builders["doc"] = build_doc_pair

    if not enabled_builders:
        print("ERROR: at least one pair builder must be enabled")
        return

    # Read the JSONL corpus (full records)
    jsonl_path = MANIFESTS_DIR / "metasploit_modules.jsonl"
    if not jsonl_path.exists():
        print(f"ERROR: Run parse_metasploit_to_jsonl.py first ({jsonl_path} not found)")
        return

    print(f"Reading modules from {jsonl_path}")
    records: list[dict] = []
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"  Loaded {len(records)} module records")

    if args.max_modules:
        random.shuffle(records)
        records = records[: args.max_modules]
        print(f"  Capped to {len(records)} modules (--max-modules)")

    # Read the by-tactic manifest to know which technique each module belongs
    # to. We use it to build a system prompt matching the tactic display name.
    by_tactic_path = MANIFESTS_DIR / "metasploit_by_tactic.json"
    module_to_tactic_name: dict[str, str] = {}
    if by_tactic_path.exists():
        with open(by_tactic_path) as fh:
            by_tactic = json.load(fh)
        for tactic_id, manifest in by_tactic.items():
            tactic_name = manifest.get("tactic_name", tactic_id.lower())
            for tech in manifest.get("techniques", []):
                for mod in tech.get("modules", []):
                    mp = mod.get("module_path")
                    if mp and mp not in module_to_tactic_name:
                        module_to_tactic_name[mp] = tactic_name

    # Bucket pairs by tactic
    pairs_by_tactic: dict[str, list[dict]] = defaultdict(list)
    skipped_no_data = 0

    for record in records:
        module_path = record.get("module_path") or ""
        # Determine tactic via the manifest grouping; fall back to module_type
        tactic_name = module_to_tactic_name.get(module_path)
        if not tactic_name:
            # No MITRE tag — bucket by module type so the data isn't lost
            # (exploit, post, auxiliary, payload, encoder, evasion, nop)
            mtype = (record.get("module_type") or "module").rstrip("s")
            tactic_name = f"metasploit_{mtype}"
        tactic_disp = TACTIC_DISPLAY.get(tactic_name, "METASPLOIT FRAMEWORK")
        system_prompt = (
            f"You are an authorized Red Team {tactic_disp} specialist with deep "
            f"knowledge of the Metasploit Framework. You provide precise, "
            f"actionable module invocations, option configurations, CVEs, and "
            f"MITRE ATT&CK mappings for adversary emulation and security "
            f"validation."
        )

        # Try each builder, collect any that succeed
        builders = list(enabled_builders.items())
        random.shuffle(builders)
        for name, builder in builders[: args.per_module]:
            try:
                if name == "technique":
                    pair = builder(record, system_prompt, tactic_disp)
                else:
                    pair = builder(record, system_prompt)
            except Exception as exc:  # noqa: BLE001
                continue
            if pair is None:
                skipped_no_data += 1
                continue
            pairs_by_tactic[tactic_name].append(pair)

    # Write per-tactic JSONL files
    print()
    total = 0
    for tactic_name, pairs in pairs_by_tactic.items():
        if not pairs:
            continue
        outfile = DATASETS_DIR / f"metasploit_{tactic_name}_dataset.jsonl"
        with open(outfile, "w") as fh:
            for p in pairs:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"  {tactic_name:25s}: {len(pairs):5d} pairs → {outfile.name}")
        total += len(pairs)

    # Write a combined file
    combined_path = DATASETS_DIR / "metasploit_combined_dataset.jsonl"
    with open(combined_path, "w") as fh:
        for pairs in pairs_by_tactic.values():
            for p in pairs:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  {'COMBINED':25s}: {total:5d} pairs → {combined_path.name}")
    print(
        f"\n  TOTAL: {total} training pairs (skipped {skipped_no_data} empty builders)"
    )


if __name__ == "__main__":
    main()
