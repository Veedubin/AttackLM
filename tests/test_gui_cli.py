"""Tests for attacklm.gui.cli (the TUI CLI entry point).

The CLI bridge is a thin wrapper that constructs AttackLMApp and calls
.run(). We can't test the full TUI lifecycle in a unit test (that needs
a Textual pilot), but we can:
  - Verify the module imports cleanly.
  - Verify the ``main()`` function exists and is callable.
  - Verify the docstring and module structure.

For full TUI behavior, see tests/test_gui.py.
"""

from __future__ import annotations

import pytest


class TestGuiCliModule:
    def test_module_imports(self):
        import attacklm.gui.cli  # noqa: F401

    def test_module_has_main(self):
        import attacklm.gui.cli

        assert callable(attacklm.gui.cli.main)

    def test_module_has_dunder_name_guard(self):
        import attacklm.gui.cli

        # The module should have a `if __name__ == "__main__"` block
        # that calls sys.exit(main()). Source-code check is good enough
        # here.
        import inspect

        source = inspect.getsource(attacklm.gui.cli)
        assert '__name__ == "__main__"' in source
        assert "sys.exit(main())" in source

    def test_main_returns_int(self):
        """main() is documented to return 0 on success. We can't
        actually run a TUI in a unit test, but the signature is
        `def main() -> int`."""
        import inspect

        import attacklm.gui.cli

        sig = inspect.signature(attacklm.gui.cli.main)
        assert sig.return_annotation is int

    def test_main_docstring_mentions_tui(self):
        import attacklm.gui.cli

        # The docstring should reference what the function does.
        assert (
            "TUI" in attacklm.gui.cli.main.__doc__
            or "gui" in attacklm.gui.cli.main.__doc__.lower()
        )


class TestGuiPackageExports:
    def test_widgets_package_exports_attach_tooltip(self):
        from attacklm.gui.widgets import attach_tooltip

        assert callable(attach_tooltip)

    def test_presets_module_exports_preset_class(self):
        from attacklm.gui.presets import Preset, BUILTIN_PRESETS

        assert isinstance(BUILTIN_PRESETS, list)
        assert all(isinstance(p, Preset) for p in BUILTIN_PRESETS)
