#!/usr/bin/env python3
# CREDITS — DATA SOURCE ATTRIBUTION
# ----------------------------------
# This script transforms data from: Guardicore Infection Monkey
# Repository: https://github.com/guardicore/monkey
# License:    GNU General Public License v3.0
# Copyright:  (C) 2007 Free Software Foundation, Inc.
#
# The output JSONL is a *transformation* of upstream Infection Monkey
# plugin manifests into chat triples. The GPLv3 may apply to the
# JSONL files themselves but not to model weights learned from them.
# See /ATTRIBUTION.md for full re-distribution guidance.
# ----------------------------------
"""Deterministic extraction of Guardicore Infection Monkey repo into AttackLM JSONL training pairs.

Walks the Infection Monkey repo at data/infection_monkey/ and produces two types
of training pairs per plugin/exploit component:

  1. Plugin documentation — structured markdown with MITRE ATT&CK mapping
  2. Code extraction — actual source code snippets with purpose summaries

Usage:
    uv run python scripts/extract_infection_monkey_to_jsonl.py [--dry-run]
"""

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Import shared MITRE tactic lookup
sys.path.insert(0, str(Path(__file__).parent))
from mitre_tactic_lookup import get_tactic_for_technique, get_tactic_name

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MONKEY_DIR = DATA_DIR / "infection_monkey" / "monkey"
OUTPUT_FILE = DATA_DIR / "datasets" / "infection_monkey_training_pairs.jsonl"

SYSTEM_MSG = (
    "You are an authorized red team adversary emulation specialist. "
    "Provide precise MITRE ATT&CK technique implementations for security validation testing."
)

SOURCE_LABEL = "guardicore/infection_monkey"

# MITRE ATT&CK mapping: plugin_type → MITRE technique IDs
MITRE_MAP: dict[str, list[str]] = {
    "exploit": ["T1210"],
    "credentials_collector": ["T1003"],
    "fingerprint": ["T1046", "T1592", "T1595"],
    "payload": ["T1486", "T1496"],
    "scanner": ["T1046"],
    "brute_force": ["T1110"],
    "propagator": ["T1210"],
}

# Human-readable MITRE technique names
MITRE_NAMES: dict[str, str] = {
    "T1210": "Exploitation of Remote Services",
    "T1003": "OS Credential Dumping",
    "T1046": "Network Service Scanning",
    "T1486": "Data Encrypted for Impact",
    "T1496": "Resource Hijacking",
    "T1592": "Gather Victim Host Information",
    "T1595": "Active Scanning",
    "T1110": "Brute Force",
}

# Directories to scan for source code (relative to MONKEY_DIR)
SOURCE_DIRS = [
    "infection_monkey/exploit",
    "infection_monkey/network_scanning",
    "infection_monkey/master",
    "infection_monkey/plugin",
]

# Mapping: source file keyword → plugin name for cross-referencing
FILE_TO_PLUGIN_MAP: dict[str, str] = {
    "smb_fingerprinter": "smb",
    "ssh_fingerprinter": "ssh",
    "http_fingerprinter": "http",
    "mssql_fingerprinter": "mssql",
    "tcp_scanner": "tcp_scanner",
    "ping_scanner": "ping_scanner",
    "scan_target_generator": "scan_target_generator",
    "propagator": "propagator",
    "exploiter": "exploiter",
    "automated_master": "automated_master",
    "ip_scanner": "ip_scanner",
    "polymorphic_agent_binary_repository_decorator": "polymorphic_binary",
    "http_agent_binary_server": "http_binary_server",
    "http_agent_binary_request_handler": "http_binary_handler",
    "http_agent_binary_server_factory": "http_binary_factory",
    "http_agent_binary_server_registrar": "http_binary_registrar",
    "caching_agent_binary_repository": "caching_binary_repo",
    "island_api_agent_otp_provider": "otp_provider",
    "ip_scan_results": "scan_results",
    "credentials_collector_plugin_factory": "credentials_collector",
    "exploiter_plugin_factory": "exploiter",
    "payload_plugin_factory": "payload",
    "i_plugin_factory": "plugin_factory_interface",
    "multiprocessing_plugin_wrapper": "plugin_wrapper",
    "queued_agent_event_publisher": "event_publisher",
}

# Plugin type classification by file keywords
FILE_PLUGIN_TYPE_MAP: dict[str, str] = {
    "smb_fingerprinter": "fingerprint",
    "ssh_fingerprinter": "fingerprint",
    "http_fingerprinter": "fingerprint",
    "mssql_fingerprinter": "fingerprint",
    "tcp_scanner": "scanner",
    "ping_scanner": "scanner",
    "scan_target_generator": "scanner",
    "propagator": "propagator",
    "exploiter": "exploit",
    "automated_master": "exploit",
    "ip_scanner": "scanner",
    "polymorphic_agent_binary_repository_decorator": "exploit",
    "http_agent_binary_server": "exploit",
    "http_agent_binary_request_handler": "exploit",
    "http_agent_binary_server_factory": "exploit",
    "http_agent_binary_server_registrar": "exploit",
    "caching_agent_binary_repository": "exploit",
    "island_api_agent_otp_provider": "exploit",
    "ip_scan_results": "scanner",
    "credentials_collector_plugin_factory": "credentials_collector",
    "exploiter_plugin_factory": "exploit",
    "payload_plugin_factory": "payload",
    "i_plugin_factory": "exploit",
    "multiprocessing_plugin_wrapper": "exploit",
    "queued_agent_event_publisher": "exploit",
}

# Human-readable descriptions for source files (for training pair generation)
FILE_DESCRIPTIONS: dict[str, str] = {
    "smb_fingerprinter": (
        "SMB fingerprinter that sends SMBv1 negotiation and session setup packets to determine "
        "the OS version and whether SMB is running on port 445."
    ),
    "ssh_fingerprinter": (
        "SSH fingerprinter that parses SSH banners to detect OpenSSH and infer Linux distribution "
        "from version strings."
    ),
    "http_fingerprinter": (
        "HTTP fingerprinter that sends HEAD requests to potential HTTP(S) ports and identifies "
        "server software from response headers."
    ),
    "mssql_fingerprinter": (
        "MSSQL fingerprinter that queries the SQL Browser service on UDP port 1434 to discover "
        "MSSQL instances and their TCP listening ports."
    ),
    "tcp_scanner": (
        "TCP port scanner using non-blocking sockets and select() for concurrent connection "
        "attempts with banner grabbing on open ports."
    ),
    "ping_scanner": (
        "ICMP ping scanner that uses subprocess to execute ping commands and determines host OS "
        "from TTL values in responses."
    ),
    "scan_target_generator": (
        "Network scan target generator that compiles IP ranges from subnets, local interfaces, "
        "and inaccessible subnets for segmentation testing."
    ),
    "propagator": (
        "Core propagation orchestrator that coordinates network scanning and exploitation threads "
        "to spread to vulnerable hosts."
    ),
    "exploiter": (
        "Multi-threaded exploiter that dequeues discovered hosts, runs configured exploit plugins "
        "against them, and reports results."
    ),
    "automated_master": (
        "Automated breach and attack simulation master that orchestrates credential collection, "
        "network propagation, and payload execution phases."
    ),
    "ip_scanner": (
        "Multi-threaded IP scanner that runs ping, TCP port scan, and fingerprinting against "
        "target hosts using a shared queue."
    ),
    "polymorphic_agent_binary_repository_decorator": (
        "Decorator that appends random bytes to agent binaries, emulating polymorphic malware "
        "where each copy has a different hash."
    ),
    "http_agent_binary_server": (
        "HTTP server that serves agent binaries for download during exploitation, with "
        "reservation-based access control and random download URLs."
    ),
    "credentials_collector_plugin_factory": (
        "Factory that creates credentials collector plugins with event publishing capabilities "
        "for credential harvesting operations."
    ),
    "exploiter_plugin_factory": (
        "Factory that creates exploiter plugins with access to binary repositories, event "
        "publishers, and propagation credential stores."
    ),
    "payload_plugin_factory": (
        "Factory that creates payload plugins with access to the Island C2 server address for "
        "payload execution operations."
    ),
    "http_agent_binary_request_handler": (
        "HTTP request handler for agent binary downloads, validating reservation IDs and "
        "serving appropriate OS-specific binaries to authenticated exploit requests."
    ),
    "http_agent_binary_server_factory": (
        "Factory that creates HTTP agent binary server instances with configured port "
        "selection for exploitation payload delivery."
    ),
    "http_agent_binary_server_registrar": (
        "Registrar that manages agent binary download reservations, coordinating between "
        "exploit plugins and the HTTP binary server for payload delivery."
    ),
    "caching_agent_binary_repository": (
        "Caching repository that downloads agent binaries from the Island C2 server with "
        "multiprocess-safe caching to avoid redundant downloads during exploitation."
    ),
    "island_api_agent_otp_provider": (
        "One-time password provider that obtains OTP tokens from the Island C2 server for "
        "agent authentication during exploitation."
    ),
    "ip_scan_results": (
        "Data structure for aggregating ping scan, TCP port scan, and fingerprint results "
        "for each target host during network discovery."
    ),
    "i_plugin_factory": (
        "Abstract plugin factory interface that defines the contract for creating exploit, "
        "credential, and payload plugin instances."
    ),
    "multiprocessing_plugin_wrapper": (
        "Plugin wrapper that runs exploiter and payload plugins in separate processes for "
        "isolation and fault tolerance during attack execution."
    ),
    "queued_agent_event_publisher": (
        "Thread-safe event publisher that queues agent events for asynchronous publishing, "
        "used by exploit and scanning modules to report activity."
    ),
}

# Maximum source code snippet length (characters) for pair type 2
MAX_SNIPPET_LENGTH = 4000


# ---------------------------------------------------------------------------
# Phase 1: Parse plugin manifests from hard_coded_manifests/
# ---------------------------------------------------------------------------


@dataclass
class PluginManifest:
    """Extracted plugin manifest data."""

    name: str
    plugin_type: str
    title: str
    description: str
    safe: bool
    supported_os: list[str] = field(default_factory=list)
    target_os: list[str] = field(default_factory=list)
    version: str = "0.0.0"


def _parse_manifest_dict(node: ast.Call) -> Optional[dict[str, Any]]:
    """Extract keyword arguments from an AgentPluginManifest() AST Call node."""
    result: dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg is None:
            continue
        key = kw.arg
        # Handle different AST literal types
        value = _ast_literal_value(kw.value, key)
        if value is not None:
            result[key] = value
    return result if result else None


def _ast_literal_value(node: ast.expr, key: str) -> Any:
    """Extract a literal value from an AST expression node."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return [_ast_literal_value(elt, key) for elt in node.elts]
    if isinstance(node, ast.Name):
        # Handle OperatingSystem.LINUX, OperatingSystem.WINDOWS etc.
        return node.id
    if isinstance(node, ast.Attribute):
        # e.g. AgentPluginType.FINGERPRINTER → "FINGERPRINTER"
        return node.attr
    if isinstance(node, ast.Call):
        # e.g. PluginName("smb") → "smb"
        if node.args and isinstance(node.args[0], ast.Constant):
            return node.args[0].value
        return None
    return None


def _normalize_os(os_list: list[Any]) -> list[str]:
    """Normalize OS values to human-readable strings."""
    result = []
    for item in os_list:
        val = str(item).upper()
        if "LINUX" in val:
            result.append("Linux")
        elif "WINDOWS" in val:
            result.append("Windows")
        else:
            result.append(str(item))
    return result


def _normalize_plugin_type(raw_type: Any) -> str:
    """Normalize plugin type to a lowercase key matching MITRE_MAP."""
    val = str(raw_type).upper()
    type_map = {
        "FINGERPRINTER": "fingerprint",
        "EXPLOITER": "exploit",
        "CREDENTIALS_COLLECTOR": "credentials_collector",
        "CREDENTIAL_COLLECTOR": "credentials_collector",
        "PAYLOAD": "payload",
    }
    return type_map.get(val, val.lower())


def parse_hard_coded_manifests(manifests_dir: Path) -> list[PluginManifest]:
    """Parse all Python files in hard_coded_manifests/ directory.

    Uses AST parsing to extract AgentPluginManifest definitions without
    importing the monkey codebase.
    """
    manifests: list[PluginManifest] = []

    if not manifests_dir.is_dir():
        print(f"  [SKIP] Directory not found: {manifests_dir}", file=sys.stderr)
        return manifests

    py_files = sorted(manifests_dir.glob("*.py"))
    print(f"  Scanning {len(py_files)} Python files in {manifests_dir}")

    for filepath in py_files:
        if filepath.name.startswith("__"):
            continue

        try:
            source = filepath.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except SyntaxError as e:
            print(f"  [WARN] AST parse error in {filepath}: {e}", file=sys.stderr)
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            # Look for dict assignments like HARD_CODED_..._MANIFESTS = { ... }
            if not isinstance(node.value, ast.Dict):
                continue

            for key_node, value_node in zip(node.value.keys, node.value.values):
                if not isinstance(value_node, ast.Call):
                    continue
                # Check if it's an AgentPluginManifest() call
                func = value_node.func
                func_name = ""
                if isinstance(func, ast.Name):
                    func_name = func.id
                elif isinstance(func, ast.Attribute):
                    func_name = func.attr

                if func_name != "AgentPluginManifest":
                    continue

                # Extract manifest key name
                key_name = ""
                if isinstance(key_node, ast.Constant):
                    key_name = str(key_node.value)
                elif isinstance(key_node, ast.Str):
                    key_name = key_node.s

                parsed = _parse_manifest_dict(value_node)
                if parsed is None:
                    continue

                plugin_type = _normalize_plugin_type(parsed.get("plugin_type", ""))
                supported_os = _normalize_os(
                    parsed.get("supported_operating_systems", [])
                )
                target_os = _normalize_os(parsed.get("target_operating_systems", []))
                safe_val = parsed.get("safe", False)

                manifest = PluginManifest(
                    name=parsed.get("name", key_name) or key_name,
                    plugin_type=plugin_type,
                    title=parsed.get("title", key_name),
                    description=parsed.get("description", ""),
                    safe=bool(safe_val),
                    supported_os=supported_os,
                    target_os=target_os,
                    version=str(parsed.get("version", "0.0.0")),
                )
                manifests.append(manifest)

    return manifests


# ---------------------------------------------------------------------------
# Phase 2: Extract source code from exploit/scanning/master directories
# ---------------------------------------------------------------------------


@dataclass
class SourceFile:
    """Extracted source code file with classification."""

    path: Path
    plugin_name: str
    plugin_type: str
    description: str
    key_functions: list[str]
    source_code: str
    relative_path: str


def _extract_key_functions(source: str) -> list[str]:
    """Extract top-level function and class method names from Python source."""
    functions: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return functions

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append(f"{node.name}.{child.name}")

    return functions


def _classify_file(stem: str) -> tuple[str, str]:
    """Return (plugin_name, plugin_type) for a file stem."""
    plugin_name = FILE_TO_PLUGIN_MAP.get(stem, stem)
    plugin_type = FILE_PLUGIN_TYPE_MAP.get(stem, "unknown")
    return plugin_name, plugin_type


def extract_source_files(monkey_dir: Path) -> list[SourceFile]:
    """Walk configured source directories and extract Python files with
    exploit/scanning/master logic."""
    source_files: list[SourceFile] = []

    for rel_dir in SOURCE_DIRS:
        scan_dir = monkey_dir / rel_dir
        if not scan_dir.is_dir():
            print(f"  [SKIP] Directory not found: {scan_dir}", file=sys.stderr)
            continue

        py_files = sorted(scan_dir.glob("*.py"))
        print(f"  Scanning {len(py_files)} Python files in {scan_dir}")

        for filepath in py_files:
            if filepath.name.startswith("__"):
                continue

            stem = filepath.stem
            plugin_name, plugin_type = _classify_file(stem)
            description = FILE_DESCRIPTIONS.get(stem, "")

            try:
                source_code = filepath.read_text(encoding="utf-8")
            except Exception as e:
                print(f"  [WARN] Error reading {filepath}: {e}", file=sys.stderr)
                continue

            key_functions = _extract_key_functions(source_code)
            relative_path = filepath.relative_to(monkey_dir)

            source_files.append(
                SourceFile(
                    path=filepath,
                    plugin_name=plugin_name,
                    plugin_type=plugin_type,
                    description=description,
                    key_functions=key_functions,
                    source_code=source_code,
                    relative_path=str(relative_path),
                )
            )

    return source_files


# ---------------------------------------------------------------------------
# Phase 3: MITRE ATT&CK cross-referencing
# ---------------------------------------------------------------------------


def get_mitre_ids(plugin_type: str) -> list[str]:
    """Map plugin type to MITRE ATT&CK technique IDs."""
    return MITRE_MAP.get(plugin_type, [])


def get_mitre_label(mitre_ids: list[str]) -> str:
    """Build human-readable MITRE label like 'T1210 (Exploitation of Remote Services)'."""
    parts = []
    for mid in mitre_ids:
        name = MITRE_NAMES.get(mid, "Unknown")
        parts.append(f"{mid} ({name})")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------


def build_plugin_documentation_pair(manifest: PluginManifest) -> dict[str, Any]:
    """Build pair type 1: Plugin documentation with MITRE ATT&CK mapping."""
    mitre_ids = get_mitre_ids(manifest.plugin_type)
    mitre_label = get_mitre_label(mitre_ids) if mitre_ids else "Unmapped"
    platforms = ", ".join(manifest.supported_os) if manifest.supported_os else "All"

    user_content = (
        f"What is the {manifest.title} plugin in Infection Monkey? "
        f"How does it map to MITRE ATT&CK?"
    )

    # Build description of what the plugin enables
    technique_desc = _technique_description(manifest.plugin_type)

    assistant_lines = [
        f"## {manifest.title}",
        f"Type: {manifest.plugin_type}",
        f"MITRE ATT&CK: {mitre_label}",
        f"Platforms: {platforms}",
        f"Safe: {manifest.safe}",
        "",
        manifest.description,
        "",
        f"This plugin enables {technique_desc} simulation in breach and attack "
        f"simulation scenarios.",
    ]

    pair: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "\n".join(assistant_lines)},
        ],
        "source": SOURCE_LABEL,
        "mitre_ids": mitre_ids,
        "plugin_type": manifest.plugin_type,
    }
    _tag_pair_with_tactic(pair)
    return pair


def _technique_description(plugin_type: str) -> str:
    """Generate a human-readable description of the technique enabled by a plugin type."""
    descriptions = {
        "fingerprint": "network service fingerprinting and OS identification",
        "scanner": "network service scanning and host discovery",
        "exploit": "remote service exploitation and lateral movement",
        "credentials_collector": "credential harvesting and OS credential dumping",
        "payload": "payload execution (ransomware/resource hijacking)",
        "propagator": "automated propagation and lateral movement",
        "brute_force": "credential brute forcing",
    }
    return descriptions.get(plugin_type, f"{plugin_type} operations")


def build_code_extraction_pair(source_file: SourceFile) -> dict[str, Any]:
    """Build pair type 2: Code extraction with source snippet and summary."""
    mitre_ids = get_mitre_ids(source_file.plugin_type)
    mitre_label = get_mitre_label(mitre_ids) if mitre_ids else "Unmapped"

    # Truncate source code if too long
    snippet = source_file.source_code
    if len(snippet) > MAX_SNIPPET_LENGTH:
        # Try to cut at a function boundary
        cutoff = snippet.rfind("\ndef ", 0, MAX_SNIPPET_LENGTH)
        if cutoff > MAX_SNIPPET_LENGTH // 2:
            snippet = snippet[:cutoff].rstrip() + "\n    # ... (truncated)"
        else:
            snippet = snippet[:MAX_SNIPPET_LENGTH].rstrip() + "\n    # ... (truncated)"

    user_content = (
        f"Show the implementation of {source_file.plugin_name} "
        f"({mitre_label}) in Infection Monkey."
    )

    # Build summary
    func_list = ", ".join(source_file.key_functions[:5])
    if len(source_file.key_functions) > 5:
        func_list += f", +{len(source_file.key_functions) - 5} more"

    summary = source_file.description
    platforms = (
        "Linux, Windows"
        if source_file.plugin_type in ("fingerprint", "scanner")
        else "cross-platform"
    )

    assistant_content = (
        f"```python\n{snippet}\n```\n\n"
        f"This code {summary} Key functions: {func_list}. "
        f"Target platforms: {platforms}."
    )

    pair: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "source": SOURCE_LABEL,
        "mitre_ids": mitre_ids,
        "plugin_type": source_file.plugin_type,
    }
    _tag_pair_with_tactic(pair)
    return pair


def build_infrastructure_pair(source_file: SourceFile) -> Optional[dict[str, Any]]:
    """Build a training pair for infrastructure/orchestration code that doesn't
    map directly to a single plugin but represents important attack infrastructure."""
    # Only build for key infrastructure files
    infra_files = {
        "propagator",
        "exploiter",
        "automated_master",
        "ip_scanner",
        "http_binary_server",
        "polymorphic_binary",
        "http_binary_handler",
        "caching_binary_repo",
    }

    if source_file.plugin_name not in infra_files:
        return None

    mitre_ids = get_mitre_ids(source_file.plugin_type)
    mitre_label = get_mitre_label(mitre_ids) if mitre_ids else "Unmapped"

    user_content = (
        f"How does Infection Monkey implement {source_file.plugin_name.replace('_', ' ')} "
        f"for {mitre_label}?"
    )

    snippet = source_file.source_code
    if len(snippet) > MAX_SNIPPET_LENGTH:
        cutoff = snippet.rfind("\nclass ", 0, MAX_SNIPPET_LENGTH)
        if cutoff < MAX_SNIPPET_LENGTH // 2:
            cutoff = snippet.rfind("\ndef ", 0, MAX_SNIPPET_LENGTH)
        if cutoff > MAX_SNIPPET_LENGTH // 2:
            snippet = snippet[:cutoff].rstrip() + "\n    # ... (truncated)"
        else:
            snippet = snippet[:MAX_SNIPPET_LENGTH].rstrip() + "\n    # ... (truncated)"

    assistant_content = (
        f"```python\n{snippet}\n```\n\n"
        f"This code implements {source_file.description} "
        f"It operates as part of the Infection Monkey attack chain for MITRE ATT&CK {mitre_label}."
    )

    pair: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "source": SOURCE_LABEL,
        "mitre_ids": mitre_ids,
        "plugin_type": source_file.plugin_type,
    }
    _tag_pair_with_tactic(pair)
    return pair


# ---------------------------------------------------------------------------
# Tactic tagging helper
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


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------


def extract_all(
    monkey_dir: Path,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Run the full extraction pipeline and return training pairs."""
    all_pairs: list[dict[str, Any]] = []

    # Phase 1: Parse plugin manifests
    print("\nPhase 1: Parsing hard-coded plugin manifests...")
    manifests_dir = monkey_dir / "common" / "hard_coded_manifests"
    manifests = parse_hard_coded_manifests(manifests_dir)
    print(f"  Found {len(manifests)} plugin manifests")

    for manifest in manifests:
        # Pair type 1: Plugin documentation
        all_pairs.append(build_plugin_documentation_pair(manifest))

    # Phase 2: Extract source code
    print("\nPhase 2: Extracting source code from exploit/scanning directories...")
    source_files = extract_source_files(monkey_dir)
    print(f"  Found {len(source_files)} source files")

    # Build lookup: plugin_name → SourceFile for cross-referencing
    source_by_plugin: dict[str, SourceFile] = {}
    for sf in source_files:
        source_by_plugin[sf.plugin_name] = sf

    # Generate pair type 2 for each source file
    for sf in source_files:
        # Skip pure factory/plugin infrastructure files without attack logic
        if (
            sf.plugin_name in ("credentials_collector", "payload")
            and "factory" in sf.path.stem
        ):
            # These are just factory wiring, no attack logic
            continue

        # Code extraction pair
        all_pairs.append(build_code_extraction_pair(sf))

        # Infrastructure pair for key orchestration code
        infra_pair = build_infrastructure_pair(sf)
        if infra_pair is not None:
            all_pairs.append(infra_pair)

    # Phase 3: Cross-reference — add pairs for manifests that have matching source
    print("\nPhase 3: Cross-referencing manifests with source code...")
    matched = 0
    for manifest in manifests:
        sf = source_by_plugin.get(manifest.name)
        if sf is None:
            continue
        matched += 1
        # Already covered by pair type 2, just log the match

    print(f"  Matched {matched}/{len(manifests)} manifests to source code")

    return all_pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Infection Monkey repo into AttackLM JSONL training pairs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary only, do not write output file",
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

    print("AttackLM Infection Monkey Extractor")
    print("=" * 50)

    if not MONKEY_DIR.is_dir():
        print(f"[ERROR] Monkey directory not found: {MONKEY_DIR}", file=sys.stderr)
        sys.exit(1)

    all_pairs = extract_all(MONKEY_DIR, dry_run=args.dry_run)

    # Statistics
    by_type: dict[str, int] = {}
    for pair in all_pairs:
        pt = pair.get("plugin_type", "unknown")
        by_type[pt] = by_type.get(pt, 0) + 1

    total = len(all_pairs)

    print(f"\n{'=' * 50}")
    print(f"Total training pairs: {total}")
    print("\nBreakdown by plugin type:")
    for pt, count in sorted(by_type.items()):
        print(f"  {pt}: {count} pairs")

    if args.validate_mitre:
        _print_tactic_coverage(all_pairs)
        print(f"\n{'=' * 50}")
        print("VALIDATE MITRE — No output file written.")
        return

    if args.dry_run:
        print("\n[DRY RUN] No output file written.")
        if all_pairs:
            print("\n--- Sample pair (first entry) ---")
            sample = all_pairs[0]
            print(json.dumps(sample, indent=2, ensure_ascii=False)[:2000])
            print("\n--- Sample pair (manifest documentation) ---")
            for p in all_pairs:
                if "What is the" in p["messages"][1]["content"]:
                    print(json.dumps(p, indent=2, ensure_ascii=False)[:2000])
                    break
            print("\n--- Sample pair (code extraction) ---")
            for p in all_pairs:
                if "Show the implementation" in p["messages"][1]["content"]:
                    print(json.dumps(p, indent=2, ensure_ascii=False)[:2000])
                    break
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nWritten {total} pairs to {output_path}")


if __name__ == "__main__":
    main()
