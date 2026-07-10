"""Tests for attacklm.neuralgentics_init._merge.

The merge algorithm is the heart of ``attacklm --init``: it determines
what gets added to the user's ``opencode.json`` when they bootstrap
the AttackLM project. The user-preservation semantics are critical
(nevers overwrites an existing ``provider`` block, never removes
keys) and the diff list is what shows up on the user's terminal.

Coverage:
  - merge_opencode_json: every merge rule (plugin, instructions,
    provider, mcp/lsp/formatter, top-level scalars)
  - merge_opencode_json_with_diff: change-list output
  - format_diff_for_display: human-readable rendering
  - parse_opencode_json: valid + invalid inputs
  - serialize_opencode_json: stable round-trip
"""

from __future__ import annotations

import copy
import json

import pytest

from attacklm.neuralgentics_init._errors import OpenCodeJsonInvalid
from attacklm.neuralgentics_init._merge import (
    INSTRUCTIONS_REFERENCE,
    PLUGIN_REFERENCE,
    format_diff_for_display,
    merge_opencode_json,
    merge_opencode_json_with_diff,
    parse_opencode_json,
    serialize_opencode_json,
)


# ---------------------------------------------------------------------------
# merge_opencode_json
# ---------------------------------------------------------------------------


class TestMergePluginArray:
    def test_empty_arrays_uses_shipped(self):
        result = merge_opencode_json({"plugin": []}, {"plugin": [PLUGIN_REFERENCE]})
        assert result["plugin"] == [PLUGIN_REFERENCE]

    def test_user_only(self):
        result = merge_opencode_json(
            {"plugin": ["user-only"]}, {"plugin": [PLUGIN_REFERENCE]}
        )
        assert result["plugin"] == ["user-only", PLUGIN_REFERENCE]

    def test_shipped_only(self):
        result = merge_opencode_json(
            {"plugin": []}, {"plugin": [PLUGIN_REFERENCE, "other-shipped"]}
        )
        assert result["plugin"] == [PLUGIN_REFERENCE, "other-shipped"]

    def test_dedup_case_sensitive(self):
        # Same string with different case is treated as DIFFERENT entries.
        result = merge_opencode_json(
            {"plugin": ["UserOnly"]},
            {"plugin": ["useronly", "UserOnly"]},
        )
        # User's "UserOnly" comes first; the shipped entries are deduped
        # against it (the matching one is dropped).
        assert "UserOnly" in result["plugin"]
        # "useronly" (lowercase) is new — should be appended.
        assert "useronly" in result["plugin"]

    def test_dedup_preserves_user_order(self):
        result = merge_opencode_json(
            {"plugin": ["b", "a"]},
            {"plugin": ["a", "c"]},
        )
        # User order is "b", "a"; shipped "a" is dup-skipped, "c" is new.
        assert result["plugin"] == ["b", "a", "c"]

    def test_no_shipped_plugin(self):
        # If shipped has no plugin, user's plugin survives unchanged.
        result = merge_opencode_json({"plugin": ["a"]}, {})
        assert result["plugin"] == ["a"]

    def test_no_user_plugin(self):
        # If user has no plugin key at all, shipped is used.
        result = merge_opencode_json({}, {"plugin": [PLUGIN_REFERENCE]})
        assert result["plugin"] == [PLUGIN_REFERENCE]


class TestMergeInstructionsArray:
    def test_adds_instructions_reference(self):
        result = merge_opencode_json(
            {"instructions": []},
            {"instructions": [INSTRUCTIONS_REFERENCE]},
        )
        assert result["instructions"] == [INSTRUCTIONS_REFERENCE]

    def test_dedupes_already_present(self):
        # If the user already has AGENTS.md, we don't add a duplicate.
        result = merge_opencode_json(
            {"instructions": [INSTRUCTIONS_REFERENCE]},
            {"instructions": [INSTRUCTIONS_REFERENCE, "EXTRA.md"]},
        )
        assert result["instructions"] == [INSTRUCTIONS_REFERENCE, "EXTRA.md"]


class TestMergeProvider:
    def test_preserves_user_provider(self):
        user_provider = {"openai": {"apiKey": "secret-key", "models": ["gpt-4"]}}
        shipped_provider = {"anthropic": {"apiKey": "secret"}}
        result = merge_opencode_json(
            {"provider": user_provider}, {"provider": shipped_provider}
        )
        # User's provider is preserved entirely.
        assert result["provider"] == user_provider

    def test_adds_shipped_provider_when_user_has_none(self):
        shipped_provider = {"anthropic": {"apiKey": "secret"}}
        result = merge_opencode_json({}, {"provider": shipped_provider})
        assert result["provider"] == shipped_provider

    def test_user_provider_not_mutated(self):
        user = {"provider": {"openai": {"apiKey": "x"}}}
        shipped = {"provider": {"anthropic": {"apiKey": "y"}}}
        user_copy = copy.deepcopy(user)
        merge_opencode_json(user, shipped)
        assert user == user_copy


class TestMergeDictSections:
    def test_adds_missing_keys(self):
        result = merge_opencode_json(
            {"mcp": {"existing": {"command": "x"}}},
            {"mcp": {"new": {"command": "y"}, "existing": {"command": "z"}}},
        )
        # User's "existing" survives; new "new" is added.
        assert result["mcp"]["existing"] == {"command": "x"}
        assert result["mcp"]["new"] == {"command": "y"}

    def test_preserves_user_values_under_same_key(self):
        result = merge_opencode_json(
            {"mcp": {"server-a": {"command": "user-cmd"}}},
            {"mcp": {"server-a": {"command": "shipped-cmd"}}},
        )
        # User wins on key collision.
        assert result["mcp"]["server-a"] == {"command": "user-cmd"}

    def test_lsp_section_works(self):
        result = merge_opencode_json(
            {},
            {"lsp": {"pyright": {"command": "pyright-langserver"}}},
        )
        assert result["lsp"]["pyright"] == {"command": "pyright-langserver"}

    def test_formatter_section_works(self):
        result = merge_opencode_json(
            {},
            {"formatter": {"black": {"command": "black"}}},
        )
        assert result["formatter"]["black"] == {"command": "black"}

    def test_empty_shipped_section_no_op(self):
        result = merge_opencode_json({"mcp": {"x": {}}}, {"mcp": {}})
        # User's section is preserved.
        assert result["mcp"] == {"x": {}}

    def test_non_dict_shipped_section_ignored(self):
        result = merge_opencode_json({}, {"mcp": "not-a-dict"})
        # Should not raise; no mcp key in result.
        assert "mcp" not in result


class TestMergeTopLevelScalars:
    def test_adds_missing_scalars(self):
        result = merge_opencode_json(
            {},
            {
                "$schema": "https://opencode.ai/schema.json",
                "autoupdate": True,
                "tool_output": "terminal",
                "compaction": False,
                "small_model": "anthropic/claude-3-haiku",
            },
        )
        assert result["$schema"] == "https://opencode.ai/schema.json"
        assert result["autoupdate"] is True
        assert result["tool_output"] == "terminal"
        assert result["compaction"] is False
        assert result["small_model"] == "anthropic/claude-3-haiku"

    def test_preserves_user_scalars(self):
        result = merge_opencode_json(
            {"$schema": "user-schema"},
            {"$schema": "shipped-schema", "autoupdate": True},
        )
        # User's $schema wins.
        assert result["$schema"] == "user-schema"
        # But autoupdate is added from shipped.
        assert result["autoupdate"] is True


class TestMergeImmutability:
    def test_inputs_not_mutated(self):
        user = {"plugin": ["a"], "mcp": {"x": {}}, "provider": {"p": 1}}
        shipped = {"plugin": ["b"], "mcp": {"y": {}}, "provider": {"p": 2}}
        user_copy = copy.deepcopy(user)
        shipped_copy = copy.deepcopy(shipped)
        merge_opencode_json(user, shipped)
        assert user == user_copy
        assert shipped == shipped_copy


class TestMergeIdempotent:
    def test_repeat_merge_is_noop(self):
        # Merging the same shipped config twice should produce the same
        # result as merging once.
        user = {"plugin": ["existing"], "mcp": {"x": {}}}
        shipped = {
            "plugin": [PLUGIN_REFERENCE],
            "mcp": {"x": {}, "y": {}},
            "instructions": [INSTRUCTIONS_REFERENCE],
        }
        once = merge_opencode_json(user, shipped)
        # Now merge the "once" result with the same shipped — should
        # be a no-op (no new entries to add).
        twice = merge_opencode_json(once, shipped)
        assert once == twice


# ---------------------------------------------------------------------------
# merge_opencode_json_with_diff
# ---------------------------------------------------------------------------


class TestMergeWithDiff:
    def test_empty_diff_on_idempotent_run(self):
        # After one merge, the second merge should produce no changes.
        user = {"plugin": [PLUGIN_REFERENCE], "instructions": [INSTRUCTIONS_REFERENCE]}
        shipped = {
            "plugin": [PLUGIN_REFERENCE],
            "instructions": [INSTRUCTIONS_REFERENCE],
        }
        _, first_diff = merge_opencode_json_with_diff(user, shipped)
        # No changes because everything is already present.
        assert first_diff == []

    def test_diff_lists_new_plugin(self):
        _, diff = merge_opencode_json_with_diff(
            {"plugin": []},
            {"plugin": [PLUGIN_REFERENCE]},
        )
        assert any("plugin" in line for line in diff)
        assert any(PLUGIN_REFERENCE in line for line in diff)

    def test_diff_lists_new_instructions(self):
        _, diff = merge_opencode_json_with_diff(
            {"instructions": []},
            {"instructions": [INSTRUCTIONS_REFERENCE]},
        )
        assert any("AGENTS.md" in line or "instructions" in line for line in diff)

    def test_diff_lists_mcp_server(self):
        _, diff = merge_opencode_json_with_diff(
            {},
            {"mcp": {"my-server": {"command": "x"}}},
        )
        assert any("MCP server" in line and "my-server" in line for line in diff)

    def test_diff_lists_lsp_server(self):
        _, diff = merge_opencode_json_with_diff(
            {},
            {"lsp": {"pyright": {"command": "p"}}},
        )
        assert any("LSP server" in line and "pyright" in line for line in diff)

    def test_diff_lists_formatter(self):
        _, diff = merge_opencode_json_with_diff(
            {},
            {"formatter": {"black": {"command": "black"}}},
        )
        assert any("formatter" in line.lower() and "black" in line for line in diff)

    def test_diff_lists_top_level_scalars(self):
        _, diff = merge_opencode_json_with_diff(
            {},
            {"small_model": "anthropic/claude-3-haiku"},
        )
        assert any("small_model" in line for line in diff)

    def test_diff_does_not_list_user_existing_entries(self):
        # If the user already has the plugin, the diff should not
        # report "Added ..." because there's nothing to add.
        _, diff = merge_opencode_json_with_diff(
            {"plugin": [PLUGIN_REFERENCE]},
            {"plugin": [PLUGIN_REFERENCE, "extra"]},
        )
        # "extra" IS new and should appear.
        assert any("extra" in line for line in diff)
        # But PLUGIN_REFERENCE should NOT appear (already present).
        added_plugin_lines = [
            line for line in diff if "Added" in line and PLUGIN_REFERENCE in line
        ]
        assert added_plugin_lines == []


# ---------------------------------------------------------------------------
# format_diff_for_display
# ---------------------------------------------------------------------------


class TestFormatDiffForDisplay:
    def test_empty_diff(self):
        assert format_diff_for_display([]) == ""

    def test_single_change(self):
        result = format_diff_for_display(["Added foo"])
        assert result == "  + Added foo"

    def test_multiple_changes(self):
        result = format_diff_for_display(["First", "Second", "Third"])
        assert result == "  + First\n  + Second\n  + Third"


# ---------------------------------------------------------------------------
# parse_opencode_json
# ---------------------------------------------------------------------------


class TestParseOpencodeJson:
    def test_valid_dict(self):
        result = parse_opencode_json('{"plugin": []}')
        assert result == {"plugin": []}

    def test_valid_empty_dict(self):
        assert parse_opencode_json("{}") == {}

    def test_invalid_json_raises(self):
        with pytest.raises(OpenCodeJsonInvalid) as exc_info:
            parse_opencode_json("{ not valid json")
        assert "opencode.json" in str(exc_info.value)
        assert "not valid JSON" in str(exc_info.value)

    def test_top_level_array_raises(self):
        with pytest.raises(OpenCodeJsonInvalid) as exc_info:
            parse_opencode_json("[1, 2, 3]")
        assert "JSON object" in str(exc_info.value)

    def test_top_level_string_raises(self):
        with pytest.raises(OpenCodeJsonInvalid):
            parse_opencode_json('"just a string"')


# ---------------------------------------------------------------------------
# serialize_opencode_json
# ---------------------------------------------------------------------------


class TestSerializeOpencodeJson:
    def test_basic(self):
        result = serialize_opencode_json({"a": 1, "b": 2})
        assert json.loads(result) == {"a": 1, "b": 2}

    def test_trailing_newline(self):
        result = serialize_opencode_json({})
        assert result.endswith("\n")

    def test_sorted_keys(self):
        # Important for stable diffs.
        result = serialize_opencode_json({"z": 1, "a": 2, "m": 3})
        # The substring "a" should appear before "m" before "z".
        assert result.index('"a"') < result.index('"m"') < result.index('"z"')

    def test_indent_two(self):
        result = serialize_opencode_json({"a": {"b": 1}})
        # Inner dict should be indented 4 spaces (2 for outer + 2 for inner).
        assert '  "b":' in result

    def test_roundtrip(self):
        original = {
            "plugin": ["a", "b"],
            "mcp": {"server": {"command": "x", "args": []}},
            "small_model": "anthropic/claude-3-haiku",
        }
        serialized = serialize_opencode_json(original)
        assert parse_opencode_json(serialized) == original
