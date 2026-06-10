#!/usr/bin/env python3
# CREDITS — DATA SOURCE ATTRIBUTION
# ----------------------------------
# This script transforms data from: Metasploit Framework
# Repository: https://github.com/rapid7/metasploit-framework
# License:    BSD 3-Clause License
# Copyright:  (C) 2006-2026, Rapid7, Inc. All rights reserved.
#
# The output JSONL is a *transformation* of upstream Metasploit module
# files (Ruby) and documentation into chat triples describing what each
# module does, with options, references, and msfconsole invocations.
# See /ATTRIBUTION.md for full details.
# ----------------------------------
"""
parse_metasploit_to_jsonl.py — Mine Metasploit Framework modules and produce
training data for AttackLM.

This script walks the metasploit-framework checkout and extracts structured
information from every module file. It emits two artifacts:

1. data/manifests/metasploit_modules.jsonl  (one record per module, full info)
2. data/manifests/metasploit_by_tactic.json (per-tactic manifest, merged into
   the existing pipeline by extract_by_tactic.py)

Each record captures the Metasploit module metadata, references (CVE, EDB,
URL), MITRE ATT&CK technique tags, options, and the rendered msfconsole
invocation sequence (commands). The accompanying documentation/*.md file
(when present) is attached as description_doc, including realistic
``msfconsole`` output captured in the Scenarios section.

Usage:
    python parse_metasploit_to_jsonl.py              # full run
    python parse_metasploit_to_jsonl.py --dry-run    # summary only
    python parse_metasploit_to_jsonl.py --limit 50   # sample
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
MSF_DIR = DATA_DIR / "metasploit-framework"
MODULES_DIR = MSF_DIR / "modules"
DOCS_DIR = MSF_DIR / "documentation" / "modules"
OUTPUT_DIR = DATA_DIR / "manifests"

# ---------------------------------------------------------------------------
# Regex patterns (compiled once)
# ---------------------------------------------------------------------------

# 'ATT&CK', Mitre::Attack::Technique::T1059_COMMAND_AND_SCRIPTING_INTERPRETER
# Captures T1059 (sub optional) and the friendly suffix
ATTACK_CONST_RE = re.compile(
    r"Mitre::Attack::Technique::(T\d{4})(?:_(\d{3}))?_([A-Z_]+)"
)

# 'CVE', '2017-0143' or 'CVE', '1978-1234'
CVE_RE = re.compile(r"^(\d{4}-\d{4,7})$")

# A reference tuple: [ 'TAG', 'value' ]  (may appear with or without quotes)
# Match common forms:
#   [ 'CVE', '2017-0143' ],
#   ['CVE', '2017-0143'],
#   [ 'URL', 'https://...' ],
#   [ 'OSVDB', '12345' ],
#   [ 'EDB', '42030' ],
#   [ 'MSB', 'MS17-010' ],
#   [ 'BID', '403' ],
REF_LINE_RE = re.compile(r"\[\s*'([A-Z&]+)'\s*,\s*('([^']*)'|\"([^\"]*)\")\s*\]")

# 'Name' => 'something' or "something" (allow %q{...} multi-line)
NAME_RE = re.compile(
    r"'Name'\s*=>\s*(?:%q\{([^}]*)\}|'([^']*)'|\"([^\"]*)\")", re.DOTALL
)
# Same for Description, Author, Platform, Arch, Privileged, Rank, SessionTypes,
# DisclosureDate, License, DefaultTarget, DefaultOptions, PayloadCompat
DESC_RE = re.compile(
    r"'Description'\s*=>\s*(%q\{([^}]*)\}|'([^']*)'|\"([^\"]*)\")", re.DOTALL
)
AUTHOR_RE = re.compile(r"'Author'\s*=>\s*\[([^\]]*)\]", re.DOTALL)
# Match 'Field' => [ ... ], %w[ ... ], or a single constant
# Examples:
#   'Platform' => [ 'win' ],
#   'Platform' => %w[unix aix],
#   'Arch' => ARCH_AARCH64,
#   'Arch' => [ ARCH_AARCH64, ARCH_X64 ]
STRING_LIST_RE = re.compile(
    r"'(Platform|Arch|SessionTypes|Targets|AKA)'\s*=>\s*"
    r"(?:\[([^\]]*)\]|%w\[([^\]]*)\]|([A-Z_][A-Z0-9_]*))"
)
# Architecture constant map (from lib/rex/arch.rb values)
ARCH_CONST_MAP = {
    "ARCH_AARCH64": "aarch64",
    "ARCH_ARMLE": "armle",
    "ARCH_ARMBE": "armbe",
    "ARCH_X86": "x86",
    "ARCH_X64": "x64",
    "ARCH_PPC": "ppc",
    "ARCH_PPC64": "ppc64",
    "ARCH_MIPS": "mips",
    "ARCH_MIPS64": "mips64",
    "ARCH_SPARC": "sparc",
    "ARCH_CMD": "cmd",
    "ARCH_PYTHON": "python",
    "ARCH_PHP": "php",
    "ARCH_TTY": "tty",
}
# Match 'Field' => value, where value is a constant/primitive literal.
# Examples:
#   'DisclosureDate' => '2013-09-24',
#   'Privileged' => true,
#   'DefaultTarget' => 0,
#   'Convention' => 'sockedi handleedi http https'
SCALAR_RE = re.compile(
    r"'(Rank|Privileged|DisclosureDate|DefaultTarget|Convention|Compat)'\s*=>\s*"
    r"([^,\n]+?)(?=,|\n|$)"
)
# Notes arrays contain Ruby constants (no quotes). Examples:
#   'Stability' => [CRASH_SAFE],
#   'SideEffects' => [ARTIFACTS_ON_DISK, SCREEN_EFFECTS]
NOTES_RE = re.compile(r"'(Stability|SideEffects|Reliability|AKA)'\s*=>\s*\[([^\]]*)\]")
# Top-level Ruby constant assignment:  Rank = ExcellentRanking
RANK_CONST_RE = re.compile(r"^\s*Rank\s*=\s*([A-Za-z_]+Ranking)\s*$", re.MULTILINE)
RANK_CONST_MAP = {
    "ManualRanking": "manual",
    "LowRanking": "low",
    "AverageRanking": "average",
    "NormalRanking": "normal",
    "GoodRanking": "good",
    "GreatRanking": "great",
    "ExcellentRanking": "excellent",
}
OPTION_RE = re.compile(
    r"Opt(String|Int|Bool|Raw|Rex|Enum|Path|Address|AddressRange|Port|Numeric|Channel|File)\.new\(\s*'([A-Za-z0-9_]+)'\s*,\s*\[([^\]]*)\]\s*\)",
    re.DOTALL,
)
# class MetasploitModule < Msf::Exploit::Remote
# class MetasploitModule < Msf::Post
# class MetasploitModule < Msf::Auxiliary
# class MetasploitModule < Msf::Evasion
# module MetasploitModule (payloads)
CLASS_RE = re.compile(r"^(?:class|module)\s+MetasploitModule\s*<\s*(Msf::[A-Za-z:]+)")
# References block - find the entire array literal. We need a balanced
# bracket match because values can be on continuation lines.
# Strategy: locate "'References' => [" then walk forward counting brackets.
REFERENCES_BLOCK_RE = re.compile(r"'References'\s*=>\s*\[", re.DOTALL)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _module_path_from_file(path: Path) -> str:
    """Convert modules/exploits/windows/smb/foo.rb -> exploit/windows/smb/foo"""
    rel = path.relative_to(MODULES_DIR)
    parts = list(rel.parts)
    if parts and parts[-1].endswith(".rb"):
        parts[-1] = parts[-1][:-3]
    return "/".join(parts)


def _doc_path_for(module_file: Path) -> Path | None:
    """Mirror the module path under documentation/modules/ to find its .md doc."""
    rel = module_file.relative_to(MODULES_DIR)
    parts = list(rel.parts)
    if parts and parts[-1].endswith(".rb"):
        parts[-1] = parts[-1][:-3] + ".md"
    candidate = DOCS_DIR / Path(*parts)
    return candidate if candidate.is_file() else None


def _extract_scenario_output(doc_text: str) -> str:
    """Pull the ``msfconsole`` example output block from a documentation .md."""
    # Look for ``` blocks following a "Scenarios" heading
    m = re.search(
        r"##\s*Scenarios.*?```(?:msf|console)?\n(.*?)```",
        doc_text,
        re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def _extract_options(opt_section: str) -> list[dict]:
    """Parse Opt*.new(...) invocations for module options.

    Format inside the call: OptString.new('NAME', [ required, 'description', default ]).
    """
    options: list[dict] = []
    for match in OPTION_RE.finditer(opt_section):
        opt_type = match.group(1)
        opt_name = match.group(2)
        meta = match.group(3)
        description = ""
        default: Any = None
        required: bool | None = None
        # Required flag (first token)
        m_req = re.match(r"\s*(true|false)\s*,", meta)
        if m_req:
            required = m_req.group(1) == "true"
        # Description is the first quoted string after the required flag
        m_desc = re.search(r",\s*'((?:[^'\\]|\\.)*)'", meta)
        if m_desc:
            description = m_desc.group(1).replace("\\'", "'")
        # Default is whatever follows the description's closing quote
        default = _extract_default_value(meta, description)
        options.append(
            {
                "name": opt_name,
                "type": opt_type,
                "required": required,
                "default": default,
                "description": description,
            }
        )
    return options


def _extract_default_value(meta: str, description: str) -> Any:
    """Return the literal that follows the description string in [.., default]."""
    if description:
        idx = meta.find(f"'{description}'")
        if idx >= 0:
            idx += len(f"'{description}'")
    else:
        idx = 0
    tail = meta[idx:].lstrip(",").strip()
    if not tail or tail.startswith("]") or tail.startswith(")"):
        return None
    tail = tail.rstrip("]").rstrip(")").rstrip(",").strip()
    if tail == "nil":
        return None
    if tail == "true":
        return True
    if tail == "false":
        return False
    if (tail.startswith("'") and tail.endswith("'")) or (
        tail.startswith('"') and tail.endswith('"')
    ):
        return tail[1:-1]
    if tail.lstrip("-").isdigit():
        try:
            return int(tail)
        except ValueError:
            pass
    return tail


def _parse_options_block(source: str) -> tuple[list[dict], list[dict]]:
    """Extract register_options([...]) and register_advanced_options([...]).

    Metasploit modules call these with the array contents on multiple lines
    (e.g. one Opt*.new per line), so we bracket-count to find the matching
    closing ']' rather than relying on a regex.
    """
    required_opts: list[dict] = []
    advanced_opts: list[dict] = []
    for marker, sink in (
        ("register_options", required_opts),
        ("register_advanced_options", advanced_opts),
    ):
        search_from = 0
        while True:
            m = re.search(rf"{marker}\s*\(", source[search_from:])
            if not m:
                break
            paren_open = search_from + m.end()
            # Expect an opening '[' immediately after the '('
            if paren_open >= len(source) or source[paren_open] != "[":
                search_from = paren_open
                continue
            depth = 1
            j = paren_open + 1
            while j < len(source) and depth > 0:
                if source[j] == "[":
                    depth += 1
                elif source[j] == "]":
                    depth -= 1
                j += 1
            block = source[paren_open + 1 : j - 1]
            for opt in _extract_options(block):
                sink.append(opt)
            search_from = j
    return required_opts, advanced_opts
    return required_opts, advanced_opts


def _coerce_scalar(raw: str) -> Any:
    """Convert a Ruby-ish scalar literal to a Python value."""
    raw = raw.strip().rstrip(",").strip()
    if raw in ("true",):
        return True
    if raw in ("false",):
        return False
    if raw in ("nil",):
        return None
    if (raw.startswith("'") and raw.endswith("'")) or (
        raw.startswith('"') and raw.endswith('"')
    ):
        return raw[1:-1]
    return raw


def _parse_string_list_field(match: re.Match) -> list[str]:
    """Extract items from `'Field' => [ ... ]`, `%w[ ... ]`, or single const.

    Group 2 = bracketed contents, Group 3 = %w contents, Group 4 = single const.
    """
    raw = match.group(2) or match.group(3) or ""
    if raw:
        items = re.findall(r"'([^']*)'", raw)
        if items:
            return items
        items = re.findall(r'"([^"]*)"', raw)
        if items:
            return items
        # %w[a b c] - bare words separated by spaces
        items = re.findall(r"\b([A-Za-z0-9_./-]+)\b", raw)
        if items:
            return items
    # Single constant form: 'Arch' => ARCH_X64
    single = match.group(4)
    if single:
        # Map common architecture constants to friendly names
        if single in ARCH_CONST_MAP:
            return [ARCH_CONST_MAP[single]]
        return [single]
    return []


def _parse_authors(match: re.Match) -> list[str]:
    """Extract individual author names from 'Author' => [ 'a', 'b' ].

    Strips trailing inline comments (e.g. 'Sean <x@y>', # note) and skips
    empty entries that can result from trailing commas in the source.
    """
    block = match.group(1)
    # Remove any inline # comments so they don't leak into author names
    block = re.sub(r"#[^\n]*", "", block)
    names = re.findall(r"'((?:[^'\\]|\\.)*)'", block)
    cleaned: list[str] = []
    for n in names:
        n = n.replace("\\'", "'").strip()
        if n and n != ",":
            cleaned.append(n)
    return cleaned


def _parse_references(source: str) -> dict[str, list[str]]:
    """Walk the 'References' => [ ... ] block and bucket by tag.

    Uses bracket counting to find the matching closing ']' because values
    can be on continuation lines and contain commas.
    """
    bucket: dict[str, list[str]] = defaultdict(list)
    m = REFERENCES_BLOCK_RE.search(source)
    if not m:
        return dict(bucket)
    start = m.end()  # Position right after "["
    depth = 1
    i = start
    while i < len(source) and depth > 0:
        ch = source[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        i += 1
    if depth != 0:
        return dict(bucket)
    block = source[start : i - 1]
    for line_match in REF_LINE_RE.finditer(block):
        tag = line_match.group(1)
        value = line_match.group(3) or line_match.group(4) or ""
        if not value:
            continue
        # Skip the ATT&CK Mitre::Attack::Technique::T... pattern - handled
        # separately by _parse_attack_techniques
        if value.startswith("Mitre::Attack::Technique::"):
            continue
        bucket[tag].append(value)
    return dict(bucket)


def _parse_attack_techniques(source: str) -> list[dict]:
    """Return [{'id': 'T1059', 'sub': '.001' or None, 'name': 'COMMAND_AND_SCRIPTING_INTERPRETER'}]"""
    techniques: list[dict] = []
    seen: set[str] = set()
    for match in ATTACK_CONST_RE.finditer(source):
        base = match.group(1)
        sub = match.group(2)
        suffix = match.group(3)
        tech_id = f"{base}.{sub}" if sub else base
        if tech_id in seen:
            continue
        seen.add(tech_id)
        # Convert CONSTANT_CASE to Title Case for the technique name
        name = suffix.replace("_", " ").title()
        techniques.append({"id": tech_id, "name": name})
    return techniques


def _module_type_from_path(path: Path) -> str:
    rel = path.relative_to(MODULES_DIR)
    return rel.parts[0] if rel.parts else "unknown"


def _synthesize_commands(module_path: str, options: list[dict]) -> list[dict]:
    """Produce an msfconsole command sequence for the module.

    Output mirrors the Atomic Red Team manifest format so the existing
    extract_by_tactic merge will accept it.
    """
    use_cmd = f"use {module_path}"
    set_cmds = []
    for opt in options:
        if opt.get("default") is None or opt["default"] == "nil":
            continue
        # Don't synthesize options that require the user's RHOST/RPORT
        if opt["name"] in {"RHOSTS", "RHOST", "RPORT", "TARGETURI"}:
            continue
        default = opt["default"]
        # Quote strings
        if (
            isinstance(default, str)
            and not default.isdigit()
            and not default.upper() in {"TRUE", "FALSE"}
        ):
            default_str = f'"{default}"'
        else:
            default_str = str(default)
        set_cmds.append(f"set {opt['name']} {default_str}")
    commands = [
        {
            "executor": "msfconsole",
            "command": " && ".join([use_cmd, *set_cmds, "run"]),
            "cleanup": None,
        }
    ]
    return commands


def parse_module_file(path: Path) -> dict[str, Any] | None:
    """Parse a single Metasploit module .rb file into a structured record."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    # Skip non-module / test files
    if "MetasploitModule" not in source and "MetasploitPayload" not in source:
        return None

    module_path = _module_path_from_file(path)
    module_type = _module_type_from_path(path)

    # Class line (determines base mixin)
    class_match = CLASS_RE.search(source)
    base_class = class_match.group(1) if class_match else "Msf::Module"

    # Name
    name_match = NAME_RE.search(source)
    name = (
        name_match.group(1) or name_match.group(2) or name_match.group(3)
        if name_match
        else path.stem
    )
    # Name from %q{...} may have leading whitespace/newlines - keep the
    # single-line summary (first non-empty line)
    if "\n" in name:
        name = next(
            (line.strip() for line in name.splitlines() if line.strip()), name.strip()
        )

    # Description
    desc_match = DESC_RE.search(source)
    description = ""
    if desc_match:
        raw_desc = (
            desc_match.group(2) or desc_match.group(3) or desc_match.group(4) or ""
        )
        # Collapse whitespace for JSON readability but keep newlines
        description = "\n".join(line.rstrip() for line in raw_desc.splitlines()).strip()

    # Authors
    author_match = AUTHOR_RE.search(source)
    authors = _parse_authors(author_match) if author_match else []

    # Lists
    platform: list[str] = []
    arch: list[str] = []
    session_types: list[str] = []
    aka: list[str] = []
    for m in STRING_LIST_RE.finditer(source):
        field = m.group(1)
        items = _parse_string_list_field(m)
        if field == "Platform":
            platform = items
        elif field == "Arch":
            arch = items
        elif field == "SessionTypes":
            session_types = items
        elif field == "AKA":
            aka = items

    # Scalars
    rank = None
    privileged: bool | None = None
    disclosure_date = None
    default_target = None
    stability: list[str] = []
    side_effects: list[str] = []
    reliability: list[str] = []
    notes_aka: list[str] = []
    compat = None
    for m in SCALAR_RE.finditer(source):
        field = m.group(1)
        value = _coerce_scalar(m.group(2))
        if field == "Rank":
            rank = value
        elif field == "Privileged":
            privileged = bool(value) if isinstance(value, bool) else None
        elif field == "DisclosureDate":
            disclosure_date = value
        elif field == "DefaultTarget":
            try:
                default_target = int(value)
            except (TypeError, ValueError):
                default_target = value
        elif field == "Compat":
            compat = value

    # Top-level  Rank = ExcellentRanking  constant assignment
    rank_const = RANK_CONST_RE.search(source)
    if rank_const and rank is None:
        rank = RANK_CONST_MAP.get(rank_const.group(1), rank_const.group(1))

    # Notes arrays: 'Stability' => [CRASH_SAFE, ...]
    for m in NOTES_RE.finditer(source):
        field = m.group(1)
        # Extract constants: CRASH_SAFE, ARTIFACTS_ON_DISK, etc.
        items = [c.strip() for c in re.findall(r"[A-Z][A-Z0-9_]+", m.group(2))]
        if field == "Stability":
            stability = items
        elif field == "SideEffects":
            side_effects = items
        elif field == "Reliability":
            reliability = items
        elif field == "AKA":
            notes_aka = items

    # References (CVE/EDB/URL/etc) + MITRE techniques
    references = _parse_references(source)
    attack_techniques = _parse_attack_techniques(source)

    # Options
    required_opts, advanced_opts = _parse_options_block(source)

    # Documentation (if exists)
    doc_path = _doc_path_for(path)
    description_doc = ""
    scenario_output = ""
    if doc_path is not None:
        try:
            doc_text = doc_path.read_text(encoding="utf-8", errors="replace")
            description_doc = doc_text
            scenario_output = _extract_scenario_output(doc_text)
        except OSError:
            pass

    # Synthesize msfconsole command sequence
    commands = _synthesize_commands(module_path, required_opts + advanced_opts)

    return {
        "module_path": module_path,
        "module_type": module_type,
        "base_class": base_class,
        "name": name,
        "description": description,
        "authors": authors,
        "platform": platform,
        "arch": arch,
        "session_types": session_types,
        "privileged": privileged,
        "rank": rank,
        "disclosure_date": disclosure_date,
        "default_target": default_target,
        "cves": references.get("CVE", []),
        "urls": references.get("URL", []),
        "edb_ids": references.get("EDB", []),
        "msb_ids": references.get("MSB", []),
        "osvdb_ids": references.get("OSVDB", []),
        "bid_ids": references.get("BID", []),
        "mitre_techniques": attack_techniques,
        "notes": {
            "stability": stability,
            "side_effects": side_effects,
            "reliability": reliability,
            "aka": list(dict.fromkeys(aka + notes_aka)),
        },
        "compat": compat,
        "options": required_opts,
        "advanced_options": advanced_opts,
        "commands": commands,
        "has_documentation": doc_path is not None,
        "scenario_output": scenario_output,
        "description_doc": description_doc,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_by_tactic(records: list[dict]) -> dict[str, dict]:
    """Group records by MITRE tactic, mapping technique IDs to their tactics.

    The technique-to-tactic mapping is reused from extract_by_tactic.py
    to keep consistency.
    """
    sys.path.insert(0, str(SCRIPT_DIR))
    from extract_by_tactic import TECHNIQUE_TO_TACTIC, TACTIC_INFO, _base_technique_id  # type: ignore

    manifests: dict[str, dict] = {}
    for tactic_id, info in TACTIC_INFO.items():
        manifests[tactic_id] = {
            "tactic_id": tactic_id,
            "tactic_name": info["name"],
            "source": "metasploit-framework",
            "techniques": defaultdict(
                lambda: {
                    "technique_id": None,
                    "name": None,
                    "modules": [],
                }
            ),
        }

    # Records with MITRE techniques -> routed by technique
    routed = 0
    unrouted = 0
    for record in records:
        if not record["mitre_techniques"]:
            unrouted += 1
            continue
        for tech in record["mitre_techniques"]:
            tactic_id = TECHNIQUE_TO_TACTIC.get(_base_technique_id(tech["id"]))
            if tactic_id is None:
                unrouted += 1
                continue
            bucket = manifests[tactic_id]["techniques"][tech["id"]]
            bucket["technique_id"] = tech["id"]
            bucket["name"] = bucket["name"] or tech["name"]
            # Strip the large description_doc to keep the by-tactic manifest
            # small (the full corpus lives in metasploit_modules.jsonl)
            slim = {k: v for k, v in record.items() if k != "description_doc"}
            bucket["modules"].append(slim)
            routed += 1

    # Materialize defaultdicts, sort modules
    for tactic_id, manifest in manifests.items():
        techs = list(manifest["techniques"].values())
        for t in techs:
            t["modules"].sort(key=lambda m: m["module_path"])
        techs.sort(key=lambda t: t["technique_id"] or "")
        manifest["techniques"] = techs
        # Drop empty tactics
        if not techs:
            pass

    print(f"  [Metasploit] Routed {routed} module-tech pairs")
    print(f"  [Metasploit] Unrouted (no technique/tactic match): {unrouted}")
    return manifests


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine Metasploit Framework modules into AttackLM training data."
    )
    parser.add_argument("--dry-run", action="store_true", help="Summarize only")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after parsing N module files (0 = no limit).",
    )
    parser.add_argument(
        "--module-types",
        nargs="*",
        default=None,
        help="Restrict to these module types (exploit, post, auxiliary, payload, evasion, encoder).",
    )
    args = parser.parse_args()

    if not MODULES_DIR.is_dir():
        print(f"ERROR: Metasploit modules directory not found: {MODULES_DIR}")
        print("  Run scripts/clone_repos.sh first.")
        sys.exit(1)

    print(f"AttackLM — Metasploit miner")
    print(f"  MSF_DIR:   {MSF_DIR}")
    print(f"  MODULES:   {MODULES_DIR}")
    print(f"  DOCS:      {DOCS_DIR}")
    print(f"  OUTPUT:    {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect module files
    module_files: list[Path] = []
    type_filter = set(args.module_types) if args.module_types else None
    for sub in sorted(MODULES_DIR.iterdir()):
        if not sub.is_dir():
            continue
        if type_filter and sub.name not in type_filter:
            continue
        module_files.extend(sub.rglob("*.rb"))
    module_files.sort()
    print(f"  Found {len(module_files)} Ruby files under modules/")

    records: list[dict] = []
    skipped = 0
    parse_errors = 0
    for idx, path in enumerate(module_files, 1):
        if args.limit and len(records) >= args.limit:
            break
        try:
            rec = parse_module_file(path)
        except Exception as exc:  # noqa: BLE001
            parse_errors += 1
            if parse_errors <= 5:
                print(f"  ! parse error on {path}: {exc}")
            continue
        if rec is None:
            skipped += 1
            continue
        records.append(rec)

    print(f"  Parsed {len(records)} modules ({skipped} skipped, {parse_errors} errors)")

    # Stats
    by_type: dict[str, int] = defaultdict(int)
    by_tech: dict[str, int] = defaultdict(int)
    with_mitre = 0
    for r in records:
        by_type[r["module_type"]] += 1
        if r["mitre_techniques"]:
            with_mitre += 1
            for t in r["mitre_techniques"]:
                by_tech[t["id"]] += 1

    print(f"  Modules by type: {dict(by_type)}")
    print(f"  Modules with MITRE references: {with_mitre}")
    print(
        f"  Top MITRE techniques: "
        + ", ".join(
            f"{tid}({cnt})"
            for tid, cnt in sorted(by_tech.items(), key=lambda x: -x[1])[:15]
        )
    )

    if args.dry_run:
        print("\n  DRY RUN — no files written.")
        return

    # Write the full corpus
    full_path = OUTPUT_DIR / "metasploit_modules.jsonl"
    with open(full_path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {full_path} ({len(records)} records)")

    # Write the by-tactic manifest
    by_tactic = aggregate_by_tactic(records)
    tactic_path = OUTPUT_DIR / "metasploit_by_tactic.json"
    with open(tactic_path, "w", encoding="utf-8") as fh:
        json.dump(by_tactic, fh, indent=2, ensure_ascii=False)
    print(f"  Wrote {tactic_path}")

    # Per-tactic summary
    print()
    print(f"  {'Tactic':25s} {'Techs':>6s} {'Modules':>8s}")
    print(f"  {'-' * 25:25s} {'-' * 6:>6s} {'-' * 8:>8s}")
    for tid, manifest in sorted(by_tactic.items()):
        tcount = len(manifest["techniques"])
        mcount = sum(len(t["modules"]) for t in manifest["techniques"])
        if tcount == 0:
            continue
        print(f"  {manifest['tactic_name']:25s} {tcount:6d} {mcount:8d}")


if __name__ == "__main__":
    main()
