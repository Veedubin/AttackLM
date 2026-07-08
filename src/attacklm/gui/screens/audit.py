"""Audit screen — run an inversion-attack audit (extraction or MIA).

Built on top of the existing _BaseCommandScreen pattern. The audit
delegates to the `attacklm audit` subcommand, which in turn calls
attacklm-dataset/scripts/inversion_audit.py with the new --attack
and --mia-method flags (v0.4.0+).
"""

from __future__ import annotations

import asyncio

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import (
    Button,
    Label,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from attacklm.gui.screens.command_forms import _BaseCommandScreen
from attacklm.gui.widgets import attach_tooltip


class AuditFormScreen(_BaseCommandScreen):
    """Run an inversion-attack audit (extraction or MIA).

    Two tabs:
    1. Extraction — Carlini 2021 prefix-completion extraction
    2. MIA — Membership inference attack (reference, zlib, per_token)

    The MIA tab exposes --mia-method. The Extraction tab exposes the
    generation parameters (top_k, max_new_tokens, temperature).
    """

    CSS = """
    #audit-tabs {
        height: auto;
    }

    #audit-tabs ContentSwitcher {
        height: auto;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        # Track the user's chosen attack class so we can include the
        # right flags when building the command.
        self.attack_class: str = "extraction"

    def compose(self) -> ComposeResult:
        with Container(id="cmd-container"):
            yield Label("Inversion Audit", id="cmd-title")
            yield Label(
                "Run a privacy audit against a trained AttackLM model. "
                "Extraction tests for verbatim regurgitation; MIA tests for "
                "training-set membership.",
                id="cmd-desc",
            )
            with TabbedContent(id="audit-tabs", initial="tab-extraction"):
                with TabPane("Extraction", id="tab-extraction"):
                    yield self._row(
                        "Model path", "audit_model", placeholder="/path/to/model"
                    )
                    yield self._row(
                        "Dataset root",
                        "audit_dataset_root",
                        placeholder="data/datasets/buckets/sources",
                    )
                    yield self._row(
                        "Source filter",
                        "audit_source_filter",
                        placeholder="metasploit,sigma-hq (optional)",
                    )
                    yield self._row(
                        "Top-K (completions)",
                        "audit_top_k",
                        placeholder="20",
                        value="20",
                    )
                    yield self._row(
                        "Max new tokens",
                        "audit_max_new_tokens",
                        placeholder="256",
                        value="256",
                    )
                    yield self._row(
                        "Temperature",
                        "audit_temperature",
                        placeholder="1.0",
                        value="1.0",
                    )
                    yield self._row(
                        "Max records (per source)",
                        "audit_max_records",
                        placeholder="50",
                        value="50",
                    )
                    with Horizontal(id="cmd-button-row"):
                        yield Button(
                            "Run Extraction Audit",
                            id="btn-run-extraction",
                            variant="primary",
                        )
                        yield Button("Back", id="btn-back")

                with TabPane("MIA", id="tab-mia"):
                    yield self._row(
                        "Model path", "audit_mia_model", placeholder="/path/to/model"
                    )
                    yield self._row(
                        "Dataset root",
                        "audit_mia_dataset_root",
                        placeholder="data/datasets/buckets/sources",
                    )
                    yield self._row(
                        "Source filter",
                        "audit_mia_source_filter",
                        placeholder="metasploit,sigma-hq (optional)",
                    )
                    yield Static("MIA method:", classes="form-label")
                    yield Select(
                        options=[
                            ("reference (NLL only)", "reference"),
                            ("zlib (NLL - zlib_length)", "zlib"),
                            ("per_token (MUSE 2023 default)", "per_token"),
                            ("lira (v0.5.0+, requires shadow models)", "lira"),
                        ],
                        value="per_token",
                        id="audit_mia_method",
                        allow_blank=False,
                    )
                    yield Static("Threshold mode:", classes="form-label")
                    yield Select(
                        options=[
                            ("percentile (recommended)", "percentile"),
                            ("median (calibration artifact)", "median"),
                            ("holdout_file (external calibration)", "holdout_file"),
                        ],
                        value="percentile",
                        id="audit_mia_threshold",
                        allow_blank=False,
                    )
                    yield self._row(
                        "Percentile", "audit_mia_percentile", placeholder="5", value="5"
                    )
                    yield self._row(
                        "Max records (per source)",
                        "audit_mia_max_records",
                        placeholder="50",
                        value="50",
                    )
                    with Horizontal(id="cmd-button-row"):
                        yield Button(
                            "Run MIA Audit", id="btn-run-mia", variant="primary"
                        )
                        yield Button("Back", id="btn-back")

            yield RichLog(id="cmd-output", highlight=True, wrap=True)

    def on_mount(self) -> None:
        """Attach tooltips to every input/widget in this screen."""
        # Extraction tab
        for key, widget_id in [
            ("audit_model", "#audit_model"),
            ("audit_dataset_root", "#audit_dataset_root"),
            ("audit_source_filter", "#audit_source_filter"),
            ("audit_top_k", "#audit_top_k"),
            ("audit_max_new_tokens", "#audit_max_new_tokens"),
            ("audit_temperature", "#audit_temperature"),
            ("audit_max_records", "#audit_max_records"),
        ]:
            try:
                attach_tooltip(self.query_one(widget_id), key)
            except Exception:
                pass  # Widget may not be in this tab if tabs are lazy

        # MIA tab
        for key, widget_id in [
            ("audit_mia_method", "#audit_mia_method"),
            ("audit_mia_threshold", "#audit_mia_threshold"),
            ("audit_mia_percentile", "#audit_mia_percentile"),
        ]:
            try:
                attach_tooltip(self.query_one(widget_id), key)
            except Exception:
                pass

        # Note: the MIA form uses `audit_mia_*` IDs to avoid collisions
        # with the extraction tab's `audit_*` IDs. The tooltip keys in
        # tooltips.py use the same `audit_*` prefix for documentation;
        # lookups in MIA's on_mount use the explicit mapping above.
        # The full tooltip text is in tooltips.TOOLTIPS for review.

        # Also attach tooltips to the MIA Inputs (they have different IDs
        # but share the same semantics as extraction tab fields)
        for widget_id in (
            "#audit_mia_model",
            "#audit_mia_dataset_root",
            "#audit_mia_source_filter",
            "#audit_mia_max_records",
        ):
            try:
                # Reuse the extraction tooltip text — the field is the same
                base_key = widget_id.replace("#audit_mia_", "audit_")
                attach_tooltip(self.query_one(widget_id), base_key)
            except Exception:
                pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses — dispatch to the right run path."""
        btn_id = event.button.id
        if btn_id == "btn-back":
            self.app.pop_screen()
            return
        if btn_id == "btn-run-extraction":
            self.attack_class = "extraction"
            asyncio.create_task(self._run_extraction())
            return
        if btn_id == "btn-run-mia":
            self.attack_class = "mia"
            asyncio.create_task(self._run_mia())
            return

    async def _run_extraction(self) -> None:
        """Build the attacklm audit command for extraction and run it."""
        values = self._get_values()
        model = values.get("audit_model", "").strip()
        dataset_root = values.get("audit_dataset_root", "").strip()
        source_filter = values.get("audit_source_filter", "").strip()
        top_k = values.get("audit_top_k", "20").strip() or "20"
        max_new_tokens = values.get("audit_max_new_tokens", "256").strip() or "256"
        temperature = values.get("audit_temperature", "1.0").strip() or "1.0"
        max_records = values.get("audit_max_records", "50").strip() or "50"

        if not model or not dataset_root:
            log = self.query_one("#cmd-output", RichLog)
            log.clear()
            log.write("[bold red]Error:[/] Model path and Dataset root are required.")
            return

        cmd = [
            "attacklm",
            "audit",
            "--attack",
            "extraction",
            "--model",
            model,
            "--dataset-root",
            dataset_root,
            "--top-k",
            top_k,
            "--max-new-tokens",
            max_new_tokens,
            "--temperature",
            temperature,
            "--max-records",
            max_records,
        ]
        if source_filter:
            cmd += ["--source-filter", *source_filter.split(",")]

        await self._run_and_display(cmd)

    async def _run_mia(self) -> None:
        """Build the attacklm audit command for MIA and run it."""
        values = self._get_values()
        model = values.get("audit_mia_model", "").strip()
        dataset_root = values.get("audit_mia_dataset_root", "").strip()
        source_filter = values.get("audit_mia_source_filter", "").strip()
        mia_method = self.query_one("#audit_mia_method").value
        # Select.value is a string or a `Select.BLANK` sentinel;
        # we set allow_blank=False so it's always a string here.
        threshold_mode = self.query_one("#audit_mia_threshold").value
        if isinstance(mia_method, object) and not isinstance(mia_method, str):
            mia_method = "per_token"
        if isinstance(threshold_mode, object) and not isinstance(threshold_mode, str):
            threshold_mode = "percentile"
        percentile = values.get("audit_mia_percentile", "5").strip() or "5"
        max_records = values.get("audit_mia_max_records", "50").strip() or "50"

        if not model or not dataset_root:
            log = self.query_one("#cmd-output", RichLog)
            log.clear()
            log.write("[bold red]Error:[/] Model path and Dataset root are required.")
            return

        cmd = [
            "attacklm",
            "audit",
            "--attack",
            "mia",
            "--mia-method",
            mia_method,
            "--mia-threshold-mode",
            threshold_mode,
            "--mia-percentile",
            percentile,
            "--model",
            model,
            "--dataset-root",
            dataset_root,
            "--max-records",
            max_records,
        ]
        if source_filter:
            cmd += ["--source-filter", *source_filter.split(",")]

        await self._run_and_display(cmd)
