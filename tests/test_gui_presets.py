"""Tests for attacklm.gui.presets (Preset dataclass, BUILTIN_PRESETS, ensure_builtin_presets).

Coverage for the Preset data model, slugify behavior, and the list_all / load /
delete / save round-trip. Uses ``tmp_path`` to redirect ``PRESETS_DIR`` via
``monkeypatch`` so no real ``~/.config/attacklm/presets`` is touched.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from attacklm.gui.presets import (
    BUILTIN_PRESETS,
    PRESETS_DIR,
    Preset,
    _slugify,
    ensure_builtin_presets,
)


# --- _slugify ---


class TestSlugify:
    def test_lowercases(self):
        assert _slugify("MyPreset") == "mypreset"

    def test_replaces_non_alphanumeric(self):
        assert _slugify("Q-GaLore Spectrum") == "q_galore_spectrum"

    def test_strips_leading_trailing_underscores(self):
        assert _slugify("  spaces  ") == "spaces"
        assert _slugify("___weird___") == "weird"

    def test_collapses_runs(self):
        assert _slugify("a!!b??c") == "a_b_c"
        assert _slugify("a   b") == "a_b"

    def test_special_chars_no_filesystem_errors(self):
        # The motivating use-case: "FP8 (H100/Blackwell)" must not
        # produce a path with "/" in it.
        result = _slugify("FP8 (H100/Blackwell)")
        assert "/" not in result
        assert "\\" not in result
        assert result == "fp8_h100_blackwell"

    def test_empty_string(self):
        assert _slugify("") == ""

    def test_already_slugified(self):
        assert _slugify("already_slugged") == "already_slugged"

    def test_unicode_collapsed(self):
        # Non-ASCII alphanumeric gets collapsed to a single underscore
        # (then stripped from the ends).
        assert _slugify("café") == "caf"


# --- Preset dataclass ---


class TestPresetDataclass:
    def test_default_params_is_empty_dict(self):
        p = Preset(name="x")
        assert p.params == {}
        assert p.description == ""

    def test_filename_uses_slugified_name(self):
        p = Preset(name="Q-GaLore Spectrum")
        assert p.filename == "q_galore_spectrum.json"

    def test_path_under_presets_dir(self):
        p = Preset(name="any")
        assert p.path.parent == PRESETS_DIR
        assert p.path.name == "any.json"

    def test_two_presets_with_same_slug_share_filename(self):
        # "FP8" and "fp8" must produce the same filename (case-insensitive
        # collision is by design — _slugify lowercases).
        assert Preset(name="FP8").filename == Preset(name="fp8").filename


# --- Preset.save / load round-trip ---


class TestPresetSaveLoadRoundTrip:
    def test_save_then_load_returns_equivalent_preset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setattr("attacklm.gui.presets.PRESETS_DIR", tmp_path)
        original = Preset(
            name="My Preset",
            description="a test",
            params={"epochs": 5, "use_galore": True},
        )
        original.save()
        loaded = Preset.load("My Preset")
        assert loaded is not None
        assert loaded.name == "My Preset"
        assert loaded.description == "a test"
        assert loaded.params == {"epochs": 5, "use_galore": True}

    def test_load_missing_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setattr("attacklm.gui.presets.PRESETS_DIR", tmp_path)
        assert Preset.load("nonexistent") is None

    def test_save_creates_presets_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        # PRESETS_DIR points at a not-yet-existing subdir.
        nested = tmp_path / "deep" / "nested" / "presets"
        monkeypatch.setattr("attacklm.gui.presets.PRESETS_DIR", nested)
        Preset(name="auto-create").save()
        assert nested.is_dir()
        assert (nested / "auto_create.json").is_file()

    def test_save_writes_valid_json(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setattr("attacklm.gui.presets.PRESETS_DIR", tmp_path)
        Preset(name="json-test", description="d", params={"k": "v"}).save()
        path = tmp_path / "json_test.json"  # slugified
        data = json.loads(path.read_text())
        assert data == {"name": "json-test", "description": "d", "params": {"k": "v"}}

    def test_load_handles_missing_optional_fields(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setattr("attacklm.gui.presets.PRESETS_DIR", tmp_path)
        # Manually write a file without the optional "description" or "params".
        (tmp_path / "minimal.json").write_text(json.dumps({"name": "minimal"}))
        loaded = Preset.load("minimal")
        assert loaded is not None
        assert loaded.name == "minimal"
        assert loaded.description == ""
        assert loaded.params == {}


# --- Preset.list_all ---


class TestPresetListAll:
    def test_list_all_empty(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setattr("attacklm.gui.presets.PRESETS_DIR", tmp_path)
        assert Preset.list_all() == []

    def test_list_all_returns_sorted_names(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setattr("attacklm.gui.presets.PRESETS_DIR", tmp_path)
        Preset(name="zeta").save()
        Preset(name="alpha").save()
        Preset(name="mu").save()
        assert Preset.list_all() == ["alpha", "mu", "zeta"]

    def test_list_all_creates_dir_if_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        nested = tmp_path / "new" / "presets"
        monkeypatch.setattr("attacklm.gui.presets.PRESETS_DIR", nested)
        Preset.list_all()  # should not raise
        assert nested.is_dir()

    def test_list_all_skips_corrupt_files(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setattr("attacklm.gui.presets.PRESETS_DIR", tmp_path)
        Preset(name="good").save()
        (tmp_path / "bad.json").write_text("{ not valid json")
        # Bad file is skipped; good file is listed.
        assert Preset.list_all() == ["good"]


# --- Preset.delete ---


class TestPresetDelete:
    def test_delete_existing_returns_true(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setattr("attacklm.gui.presets.PRESETS_DIR", tmp_path)
        Preset(name="to-delete").save()
        assert Preset.delete("to-delete") is True
        assert not (tmp_path / "to_delete.json").exists()

    def test_delete_missing_returns_false(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setattr("attacklm.gui.presets.PRESETS_DIR", tmp_path)
        assert Preset.delete("never-existed") is False


# --- BUILTIN_PRESETS ---


class TestBuiltinPresets:
    def test_builtin_presets_is_non_empty_list(self):
        assert isinstance(BUILTIN_PRESETS, list)
        assert len(BUILTIN_PRESETS) >= 5  # we ship at least 5 built-ins

    def test_all_builtins_have_unique_names(self):
        names = [p.name for p in BUILTIN_PRESETS]
        assert len(names) == len(set(names)), f"Duplicate preset names: {names}"

    def test_all_builtins_have_dict_params(self):
        for p in BUILTIN_PRESETS:
            assert isinstance(p.params, dict), f"{p.name} params is not a dict"
            assert p.name, f"Empty name in {p}"

    def test_all_builtins_have_nonempty_description(self):
        for p in BUILTIN_PRESETS:
            assert p.description, f"{p.name} has empty description"

    def test_known_builtins_present(self):
        names = {p.name for p in BUILTIN_PRESETS}
        # Spot-check: we promised Q-GaLore, QLoRA, and DeepSpeed presets
        # in the README. If any of these disappear, that's a regression.
        for expected in ("3B Q-GaLore Spectrum", "3B LoRA Default", "DeepSpeed 40B+"):
            assert expected in names, f"Built-in preset {expected!r} missing"


# --- ensure_builtin_presets ---


class TestEnsureBuiltinPresets:
    def test_creates_only_missing_presets(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setattr("attacklm.gui.presets.PRESETS_DIR", tmp_path)
        # Pre-seed one of the builtins so ensure_builtin_presets should
        # NOT overwrite it.
        existing = BUILTIN_PRESETS[0]
        Preset(
            name=existing.name,
            description="USER-EDITED-DO-NOT-OVERWRITE",
            params={"x": 1},
        ).save()

        ensure_builtin_presets()

        # The pre-seeded version should still have the user-edited
        # description, not the built-in one.
        loaded = Preset.load(existing.name)
        assert loaded.description == "USER-EDITED-DO-NOT-OVERWRITE"
        assert loaded.params == {"x": 1}

    def test_creates_all_when_presets_dir_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setattr("attacklm.gui.presets.PRESETS_DIR", tmp_path)
        ensure_builtin_presets()
        listed = Preset.list_all()
        assert len(listed) == len(BUILTIN_PRESETS)
        for builtin in BUILTIN_PRESETS:
            assert builtin.name in listed
