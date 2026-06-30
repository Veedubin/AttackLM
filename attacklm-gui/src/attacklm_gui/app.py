"""Main Textual application for AttackLM GUI."""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Header, Footer

from attacklm_gui.presets import ensure_builtin_presets
from attacklm_gui.screens.main_menu import MainMenuScreen


class AttackLMApp(App):
    """Terminal GUI wrapper for AttackLM CLI tools."""

    TITLE = "AttackLM GUI"
    SUB_TITLE = "v0.1.0"
    CSS_PATH = None  # We'll use inline CSS for now

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("m", "main_menu", "Main Menu", show=True),
        Binding("?", "help", "Help", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
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
