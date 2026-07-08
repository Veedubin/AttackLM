"""Tests for the new Audit screen + tooltip coverage."""

from __future__ import annotations

import pytest
from textual.widgets import Static

from attacklm.gui.app import AttackLMApp
from attacklm.gui.screens.audit import AuditFormScreen
from attacklm.gui.widgets import TOOLTIPS, attach_tooltip


def _patch_presets(monkeypatch, tmp_path):
    """Monkeypatch preset dir and ensure_builtin_presets for test isolation."""
    monkeypatch.setattr("attacklm.gui.presets.PRESETS_DIR", tmp_path)
    monkeypatch.setattr("attacklm.gui.app.ensure_builtin_presets", lambda: None)


class TestAuditScreen:
    """The Audit screen must mount, expose the right widget IDs, and
    have tooltips set on every input."""

    @pytest.mark.asyncio
    async def test_audit_screen_mounts(self, tmp_path, monkeypatch) -> None:
        """The Audit screen mounts without error."""
        _patch_presets(monkeypatch, tmp_path)
        app = AttackLMApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            # Push the audit screen directly (don't click — pilot.click() needs
            # a real terminal to compute screen offsets; we're headless).
            app.push_screen(AuditFormScreen())
            await pilot.pause()
            assert app.screen.__class__.__name__ == "AuditFormScreen"

    @pytest.mark.asyncio
    async def test_audit_extraction_has_model_input(
        self, tmp_path, monkeypatch
    ) -> None:
        """The Extraction tab has a model input."""
        _patch_presets(monkeypatch, tmp_path)
        app = AttackLMApp()
        async with app.run_test() as pilot:
            app.push_screen(AuditFormScreen())
            await pilot.pause()
            widget = app.screen.query_one("#audit_model")
            assert widget is not None

    @pytest.mark.asyncio
    async def test_audit_mia_method_select_exists(self, tmp_path, monkeypatch) -> None:
        """The MIA tab has a Select widget for the MIA method."""
        _patch_presets(monkeypatch, tmp_path)
        app = AttackLMApp()
        async with app.run_test() as pilot:
            app.push_screen(AuditFormScreen())
            await pilot.pause()
            widget = app.screen.query_one("#audit_mia_method")
            assert widget is not None
            # The Select has 4 options
            assert len(widget._options) >= 4  # type: ignore[attr-defined]


class TestTooltips:
    """Tooltips must be set on every expected widget in the Audit screen
    and on the main menu buttons."""

    @pytest.mark.asyncio
    async def test_audit_screen_inputs_have_tooltips(
        self, tmp_path, monkeypatch
    ) -> None:
        """Every input field in the audit screen has a non-empty tooltip."""
        _patch_presets(monkeypatch, tmp_path)
        app = AttackLMApp()
        async with app.run_test() as pilot:
            app.push_screen(AuditFormScreen())
            await pilot.pause()
            for widget_id in (
                "#audit_model",
                "#audit_dataset_root",
                "#audit_top_k",
                "#audit_max_new_tokens",
            ):
                widget = app.screen.query_one(widget_id)
                assert widget.tooltip, f"{widget_id} has no tooltip"
                assert len(widget.tooltip) > 10, f"{widget_id} tooltip too short"

    @pytest.mark.asyncio
    async def test_main_menu_buttons_have_tooltips(self, tmp_path, monkeypatch) -> None:
        """Every main menu button has a tooltip."""
        _patch_presets(monkeypatch, tmp_path)
        app = AttackLMApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            # Stay on the main menu (default screen after on_mount)
            for btn_id in (
                "#btn-train",
                "#btn-init",
                "#btn-balance",
                "#btn-build",
                "#btn-infer",
                "#btn-eval",
                "#btn-steer",
                "#btn-bench",
                "#btn-audit",
            ):
                widget = app.screen.query_one(btn_id)
                assert widget.tooltip, f"{btn_id} has no tooltip"

    def test_tooltips_dict_has_all_required_keys(self) -> None:
        """The TOOLTIPS dict has every key the audit screen + main menu need."""
        required_keys = {
            "btn-train",
            "btn-init",
            "btn-balance",
            "btn-build",
            "btn-infer",
            "btn-eval",
            "btn-steer",
            "btn-bench",
            "btn-audit",  # main menu
            "audit_model",
            "audit_dataset_root",
            "audit_source_filter",
            "audit_attack",
            "audit_mia_method",
            "audit_mia_threshold",
            "audit_mia_percentile",
            "audit_top_k",
            "audit_max_new_tokens",
            "audit_temperature",
            "audit_max_records",  # audit screen
        }
        missing = required_keys - set(TOOLTIPS.keys())
        assert not missing, f"Missing tooltip keys: {missing}"

    def test_attach_tooltip_no_op_for_missing_key(self) -> None:
        """attach_tooltip() doesn't raise for unknown keys."""
        widget = Static("test")
        attach_tooltip(widget, "nonexistent_key_xyz")
        assert widget.tooltip is None  # No-op
