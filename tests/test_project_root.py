"""Tests for attacklm._project_root.

The module exposes module-level constants (BASE_DIR, BUCKETS_DIR, ...)
that are computed once at import time. We can't easily change
``__main__.__file__`` after import (because the constants are
already cached), so we test what's actually testable:

  1. The constants are valid Path objects that exist as parents.
  2. require_manifest() works when the manifest is where it should be
     (i.e., the real repo path).
  3. require_manifest() exits with a helpful error when the manifest
     is missing.

The detailed _resolve_base_dir() logic is exercised by the real
``attacklm --init`` flow in production; the unit tests here are
regression guards against "did someone break the module imports?"
or "did someone break the directory layout constants?".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    """BASE_DIR and friends are module-level Path constants."""

    def test_base_dir_is_a_path(self):
        from attacklm._project_root import BASE_DIR

        assert isinstance(BASE_DIR, Path)

    def test_base_dir_exists_as_directory(self):
        from attacklm._project_root import BASE_DIR

        # The repo root must exist (we're running tests from it).
        assert BASE_DIR.is_dir()

    def test_base_dir_has_data_dir(self):
        from attacklm._project_root import BASE_DIR

        # The data/ directory is gitignored in the public repo (it
        # lives in the separate attacklm-dataset repo since v0.11.0).
        # So the directory may NOT exist on a fresh clone. We just
        # assert that the *expected path* is well-formed.
        data_path = BASE_DIR / "data"
        assert isinstance(data_path, Path)
        # If it exists, it must be a directory.
        if data_path.exists():
            assert data_path.is_dir()

    def test_base_dir_has_scripts_dir(self):
        from attacklm._project_root import BASE_DIR

        assert (BASE_DIR / "scripts").is_dir()

    def test_datasets_dir_under_base_dir(self):
        from attacklm._project_root import BASE_DIR, DATASETS_DIR

        assert DATASETS_DIR == BASE_DIR / "data" / "datasets"

    def test_buckets_dir_under_datasets_dir(self):
        from attacklm._project_root import BUCKETS_DIR, DATASETS_DIR

        assert BUCKETS_DIR == DATASETS_DIR / "buckets"

    def test_sources_dir_under_buckets_dir(self):
        from attacklm._project_root import BUCKETS_DIR, SOURCES_DIR

        assert SOURCES_DIR == BUCKETS_DIR / "sources"

    def test_models_dir_under_base_dir(self):
        from attacklm._project_root import BASE_DIR, MODELS_DIR

        assert MODELS_DIR == BASE_DIR / "models"

    def test_logs_dir_under_base_dir(self):
        from attacklm._project_root import BASE_DIR, LOGS_DIR

        assert LOGS_DIR == BASE_DIR / "logs"

    def test_presets_dir_under_base_dir(self):
        from attacklm._project_root import BASE_DIR, PRESETS_DIR

        assert PRESETS_DIR == BASE_DIR / "presets"

    def test_evolved_dir_under_datasets_dir(self):
        from attacklm._project_root import DATASETS_DIR, EVOLVED_DIR

        assert EVOLVED_DIR == DATASETS_DIR / "evolved"


# ---------------------------------------------------------------------------
# _resolve_base_dir()
# ---------------------------------------------------------------------------


class TestResolveBaseDir:
    """Test the _resolve_base_dir() function in isolation by manipulating
    sys.modules to re-evaluate it under controlled __main__.__file__."""

    def test_returns_path_for_script_in_scripts_dir(self, tmp_path):
        """When __main__.__file__ is in <root>/scripts/, returns the root."""
        # Build fake layout: <tmp>/scripts/fake.py + <tmp>/data/datasets/buckets/manifest.json
        (tmp_path / "data" / "datasets" / "buckets").mkdir(parents=True)
        (tmp_path / "data" / "datasets" / "buckets" / "manifest.json").write_text("{}")
        (tmp_path / "scripts").mkdir()
        script = tmp_path / "scripts" / "fake.py"
        script.write_text("")

        import __main__ as _main

        original_file = getattr(_main, "__file__", None)
        try:
            _main.__file__ = str(script)
            from attacklm._project_root import _resolve_base_dir

            assert _resolve_base_dir() == tmp_path
        finally:
            if original_file is None:
                if hasattr(_main, "__file__"):
                    delattr(_main, "__file__")
            else:
                _main.__file__ = original_file

    def test_falls_back_to_cwd_when_no_manifest(self, tmp_path, monkeypatch):
        """When source-tree detection fails (no manifest), use CWD."""
        (tmp_path / "scripts").mkdir()
        script = tmp_path / "scripts" / "fake.py"
        script.write_text("")

        import __main__ as _main

        original_file = getattr(_main, "__file__", None)
        try:
            _main.__file__ = str(script)
            monkeypatch.chdir(tmp_path)
            from attacklm._project_root import _resolve_base_dir

            assert _resolve_base_dir() == tmp_path
        finally:
            if original_file is None:
                if hasattr(_main, "__file__"):
                    delattr(_main, "__file__")
            else:
                _main.__file__ = original_file

    def test_falls_back_to_cwd_when_no_file(self, monkeypatch, tmp_path):
        """When __main__ has no __file__ at all, use CWD."""
        import __main__ as _main

        original_file = getattr(_main, "__file__", None)
        try:
            if hasattr(_main, "__file__"):
                delattr(_main, "__file__")
            monkeypatch.chdir(tmp_path)
            from attacklm._project_root import _resolve_base_dir

            assert _resolve_base_dir() == tmp_path
        finally:
            if original_file is not None:
                _main.__file__ = original_file


# ---------------------------------------------------------------------------
# require_manifest()
# ---------------------------------------------------------------------------


class TestRequireManifest:
    def test_returns_path_when_manifest_exists(self):
        """The AttackLM repo has a real manifest.json — require_manifest should work."""
        from attacklm._project_root import BUCKETS_DIR, require_manifest

        manifest = BUCKETS_DIR / "manifest.json"
        if not manifest.exists():
            pytest.skip("manifest.json not present in this checkout")
        # If we get here, the function should return that path.
        assert require_manifest() == manifest

    def test_exits_with_helpful_message_when_no_manifest(
        self, monkeypatch, tmp_path, capsys
    ):
        """When the manifest is missing, exit 1 with a helpful message."""
        # Force BUCKETS_DIR to point at a non-existent location.
        import attacklm._project_root as pr

        monkeypatch.setattr(pr, "BUCKETS_DIR", tmp_path / "does_not_exist")
        with pytest.raises(SystemExit) as exc_info:
            pr.require_manifest()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Manifest not found" in captured.err
        assert "git clone" in captured.err
        assert "attacklm train --all" in captured.err
