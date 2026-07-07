"""``opencode.json`` merge algorithm (§4 of the init-cli design doc).

The merge is a *deep merge with user-preservation semantics*: the user's
existing ``opencode.json`` is the base, the shipped config is the overlay,
and the algorithm only ever ADDS missing entries — it never removes,
overwrites existing dicts, or touches the ``provider`` block.

Public API
----------
* :func:`merge_opencode_json` — pure merge, returns new dict.
* :func:`merge_opencode_json_with_diff` — merge + human-readable change list.
* :func:`format_diff_for_display` — pretty-print a change list.
* :func:`parse_opencode_json` — ``json.loads`` wrapper raising
  :class:`OpenCodeJsonInvalid`.
* :func:`serialize_opencode_json` — canonical ``json.dumps`` format for
  stable round-trips.
* :data:`PLUGIN_REFERENCE`, :data:`INSTRUCTIONS_REFERENCE` — the canonical
  entries added to the ``plugin`` / ``instructions`` arrays.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any

from ._errors import OpenCodeJsonInvalid

__all__ = [
    "INSTRUCTIONS_REFERENCE",
    "PLUGIN_REFERENCE",
    "format_diff_for_display",
    "merge_opencode_json",
    "merge_opencode_json_with_diff",
    "parse_opencode_json",
    "serialize_opencode_json",
]

#: The plugin entry added to the ``plugin`` array.
PLUGIN_REFERENCE = "@veedubin/neuralgentics"

#: The instructions entry added to the ``instructions`` array.
INSTRUCTIONS_REFERENCE = "AGENTS.md"

#: Top-level scalar keys added from the shipped config when missing.
_TOP_LEVEL_SCALARS: tuple[str, ...] = (
    "$schema",
    "autoupdate",
    "tool_output",
    "compaction",
    "small_model",
)

#: Dict-valued sections merged key-by-key (add new keys only).
_DICT_SECTIONS: tuple[str, ...] = ("mcp", "lsp", "formatter")


def _union_arrays(
    user: list[object] | None, shipped: list[object] | None
) -> list[object]:
    """Case-sensitive, order-preserving union of two arrays.

    User entries come first (in their original order), followed by any
    shipped entries not already present. Duplicates are dropped.
    """
    result: list[object] = []
    seen: set[object] = set()
    for source in (user, shipped):
        if not source:
            continue
        for item in source:
            # Use a hashable key for membership tracking. Lists/dicts are
            # unhashable; fall back to id() for those (rare in opencode.json
            # arrays, which are normally strings).
            try:
                key = item
                hash(key)
            except TypeError:
                key = id(item)
            if key not in seen:
                seen.add(key)
                result.append(item)
    return result


def merge_opencode_json(
    user_json: dict[str, Any], shipped_json: dict[str, Any]
) -> dict[str, Any]:
    """Merge a user ``opencode.json`` with the shipped config.

    Implements §4.1 of the design doc exactly. Returns a NEW dict; the
    inputs are never mutated.

    Rules:
      1. ``plugin`` array → union (dedup, case-sensitive).
      2. ``instructions`` array → union (dedup).
      3. ``provider`` → preserved entirely from the user.
      4. ``mcp`` / ``lsp`` / ``formatter`` → add shipped keys missing in user.
      5. Top-level scalars (``$schema``, ``autoupdate``, ``tool_output``,
         ``compaction``, ``small_model``) → add from shipped if missing.
    """
    result: dict[str, Any] = copy.deepcopy(user_json)

    # 1 & 2 — array sections.
    result["plugin"] = _union_arrays(
        user_json.get("plugin"), shipped_json.get("plugin")
    )
    result["instructions"] = _union_arrays(
        user_json.get("instructions"), shipped_json.get("instructions")
    )

    # 3 — provider: PRESERVE user's entirely when present. When the user has
    # no provider block (e.g. empty user_json / "no existing opencode.json"
    # edge case in §4.5), add the shipped one verbatim.
    if "provider" not in result and "provider" in shipped_json:
        result["provider"] = copy.deepcopy(shipped_json["provider"])

    # 4 — dict sections: add shipped keys missing in user.
    for section in _DICT_SECTIONS:
        shipped_section = shipped_json.get(section)
        if not isinstance(shipped_section, Mapping):
            continue
        result_section = result.setdefault(section, {})
        for key, value in shipped_section.items():
            if key not in result_section:
                result_section[key] = copy.deepcopy(value)

    # 5 — top-level scalars.
    for key in _TOP_LEVEL_SCALARS:
        if key not in result and key in shipped_json:
            result[key] = copy.deepcopy(shipped_json[key])

    return result


def merge_opencode_json_with_diff(
    user_json: dict[str, Any], shipped_json: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Like :func:`merge_opencode_json` but also returns change descriptions.

    The returned list is empty when nothing changed (idempotent re-run).
    Change strings are stable and human-readable, e.g.::

        Added '@veedubin/neuralgentics' to plugin array
        Added MCP server 'searxng'
        Set small_model
    """
    result = merge_opencode_json(user_json, shipped_json)
    changes: list[str] = []

    # Plugin additions.
    user_plugin = user_json.get("plugin") or []
    shipped_plugin = shipped_json.get("plugin") or []
    for item in shipped_plugin:
        if item not in user_plugin:
            changes.append(f"Added {item!r} to plugin array")

    # Instructions additions.
    user_instr = user_json.get("instructions") or []
    shipped_instr = shipped_json.get("instructions") or []
    for item in shipped_instr:
        if item not in user_instr:
            changes.append(f"Added {item!r} to instructions array")

    # Dict-section additions.
    section_labels = {
        "mcp": "MCP server",
        "lsp": "LSP server",
        "formatter": "formatter",
    }
    for section, label in section_labels.items():
        shipped_section = shipped_json.get(section)
        if not isinstance(shipped_section, Mapping):
            continue
        user_section = user_json.get(section)
        if not isinstance(user_section, Mapping):
            user_section = {}
        for key in shipped_section:
            if key not in user_section:
                changes.append(f"Added {label} {key!r}")

    # Top-level scalar additions.
    for key in _TOP_LEVEL_SCALARS:
        if key not in user_json and key in shipped_json:
            changes.append(f"Set {key}")

    return result, changes


def format_diff_for_display(changes: list[str]) -> str:
    """Pretty-print a change list (one ``+ `` line per change)."""
    return "\n".join(f"  + {change}" for change in changes)


def parse_opencode_json(text: str) -> dict[str, Any]:
    """Parse an ``opencode.json`` document.

    Wraps :func:`json.loads` and re-raises parse errors as
    :class:`OpenCodeJsonInvalid` (exit code 3) with the original error
    included in the message.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as err:
        raise OpenCodeJsonInvalid(
            f"opencode.json is not valid JSON: {err.msg} (line {err.lineno}, column {err.colno})"
        ) from err
    if not isinstance(parsed, dict):
        raise OpenCodeJsonInvalid(
            f"opencode.json must be a JSON object, got {type(parsed).__name__}"
        )
    return parsed


def serialize_opencode_json(obj: dict[str, Any]) -> str:
    """Canonical serialization for stable on-disk diffs.

    ``json.dumps`` with ``indent=2`` and ``sort_keys=True`` plus a trailing
    newline so subsequent ``init`` runs produce byte-identical output.
    """
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"
