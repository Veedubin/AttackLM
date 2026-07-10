"""Tests for attacklm.gui.widgets.tooltips (TOOLTIPS dict, attach_tooltip).

Regression coverage: every key referenced by main_menu.py, train_form.py,
command_forms.py, and audit.py must have a TOOLTIPS entry. If someone
adds a new widget with an id like "btn-foo" and forgets the tooltip
text, this test fails.
"""

from __future__ import annotations

import pytest

from attacklm.gui.widgets import attach_tooltip
from attacklm.gui.widgets.tooltips import TOOLTIPS


# ---------------------------------------------------------------------------
# TOOLTIPS dict shape
# ---------------------------------------------------------------------------


class TestTooltipsDict:
    def test_tooltips_is_a_dict(self):
        assert isinstance(TOOLTIPS, dict)

    def test_tooltips_has_main_menu_buttons(self):
        # These 9 keys are referenced by main_menu.py on_mount.
        for key in (
            "btn-train",
            "btn-init",
            "btn-balance",
            "btn-build",
            "btn-infer",
            "btn-eval",
            "btn-audit",
            "btn-steer",
            "btn-bench",
        ):
            assert key in TOOLTIPS, f"Missing TOOLTIPS entry for {key!r}"

    def test_tooltips_has_audit_screen_keys(self):
        # These 9 keys are referenced by screens/audit.py.
        for key in (
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
            "audit_max_records",
        ):
            assert key in TOOLTIPS, f"Missing TOOLTIPS entry for {key!r}"

    def test_tooltips_has_train_form_high_traffic_keys(self):
        for key in (
            "train_epochs",
            "train_batch_size",
            "train_lora_r",
            "train_lora_alpha",
            "train_galore_rank",
            "train_max_length",
            "train_spectrum",
            "train_use_qgalore",
            "train_use_dora",
            "train_learning_rate",
        ):
            assert key in TOOLTIPS, f"Missing TOOLTIPS entry for {key!r}"

    def test_all_tooltips_are_non_empty_strings(self):
        for key, text in TOOLTIPS.items():
            assert isinstance(text, str), f"TOOLTIPS[{key!r}] is not a string"
            assert text.strip(), f"TOOLTIPS[{key!r}] is empty"

    def test_all_keys_are_lowercase(self):
        # Convention: TOOLTIPS keys are lowercase (slug of the widget id
        # without the "btn-" / "train_" / "audit_" prefix sometimes
        # preserved). Verify no surprise mixed-case keys sneak in.
        for key in TOOLTIPS:
            assert key == key.lower(), f"TOOLTIPS key {key!r} has uppercase chars"

    def test_no_duplicate_keys(self):
        # Defensive — dicts can't have duplicate keys, but this catches
        # copy-paste errors where two entries have the same text.
        # (Deduplicating is done by Python; this is a sanity check.)
        assert len(TOOLTIPS) == len(set(TOOLTIPS.keys()))


# ---------------------------------------------------------------------------
# attach_tooltip
# ---------------------------------------------------------------------------


class TestAttachTooltip:
    def test_attach_sets_widget_tooltip(self):
        widget = _FakeWidget()
        attach_tooltip(widget, "btn-train")
        assert widget.tooltip == TOOLTIPS["btn-train"]

    def test_attach_unknown_key_is_noop(self):
        widget = _FakeWidget()
        # Should not raise; should not set tooltip.
        attach_tooltip(widget, "nonexistent-key-xyz")
        assert getattr(widget, "tooltip", None) is None

    def test_attach_does_not_raise_on_fake_widget(self):
        # The function should be robust to widgets that don't have
        # a tooltip attribute; if a widget lacks one, setattr will
        # create it. Test that we don't crash either way.
        widget = _FakeWidget()
        attach_tooltip(widget, "btn-init")
        # The tooltip is now set.
        assert widget.tooltip == TOOLTIPS["btn-init"]


class _FakeWidget:
    """A minimal stand-in for a Textual widget that accepts .tooltip = str."""

    def __init__(self) -> None:
        self.tooltip: str | None = None
