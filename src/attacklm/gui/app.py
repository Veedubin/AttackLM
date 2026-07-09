from __future__ import annotations

# PROVENANCE METADATA — src/attacklm/gui/app.py
# ================================================================================
# Attack class:        N/A (this file is a Textual TUI wrapper around the
#                      CLI; it does not implement any attack)
# Original authors:    Veedubin (in-repo author)
# Paper title:         N/A (internal)
# Year / venue:        2026 / in-repo
# Paper URL:           N/A
# Canonical repo:      https://github.com/Veedubin/AttackLM (this repo)
#
# Implementation:
#   Type:              ORIGINAL_WORK (TUI wrapper around the CLI)
#   Lines of port:     N/A
#   Upstream license:  N/A
#
# TUI framework:       Textual (https://github.com/Textualize/textual),
#                      MIT License, Will McGuinness / Textualize
#
# Related attack papers (implemented in the attacklm-dataset sibling repo,
# invoked via the `attacklm audit` subcommand from this TUI):
#   - Carlini et al. 2021 — https://arxiv.org/abs/2012.07805
#   - Carlini et al. 2022 — https://arxiv.org/abs/2112.03570
#
# Data sources: N/A
#
# Rights claim contact: veedubin.legal@example.com
# See:                  https://github.com/Veedubin/attacklm-dataset/blob/main/RIGHTS.md
# ================================================================================
"""Main Textual application for AttackLM GUI."""


from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Header, Footer

from attacklm.gui.presets import ensure_builtin_presets
from attacklm.gui.screens.main_menu import MainMenuScreen


class AttackLMApp(App):
    """Terminal GUI wrapper for AttackLM CLI tools."""

    TITLE = "AttackLM GUI"
    SUB_TITLE = "v0.10.0"  # Updated at release time by re-release agent
    CSS_PATH = None  # We'll use inline CSS for now

    DEFAULT_CSS = """
    Tooltip {
        background: $surface;
        border: solid $accent;
        padding: 0 1;
        max-width: 60;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("m", "main_menu", "Main Menu", show=True),
        Binding("?", "help", "Help", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        # Textual 2.0+ supports tooltip_delay as an App attribute.
        # 0.5s hover before showing — comfortable for terminal users.
        self.tooltip_delay = 0.5
        self.attacklm_dir = self._find_attacklm_dir()

    def _find_attacklm_dir(self) -> Path:
        """Find the AttackLM project directory."""
        # Try common locations
        candidates = [
            Path.cwd(),
            Path.home() / "Projects" / "reverse_engineering" / "AttackLM",
            Path.home() / "AttackLM",
        ]
        for c in candidates:
            if (c / "pyproject.toml").exists() and (
                c / "scripts" / "train_template.py"
            ).exists():
                return c
        return Path.cwd()

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

    def on_mount(self) -> None:
        ensure_builtin_presets()
        self.push_screen(MainMenuScreen())

    def action_main_menu(self) -> None:
        """Return to main menu."""
        while len(self.screen_stack) > 1:
            self.pop_screen()
