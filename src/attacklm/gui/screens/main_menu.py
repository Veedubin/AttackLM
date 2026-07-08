"""Main menu screen for AttackLM GUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Button, Label

from attacklm.gui.widgets import attach_tooltip


class MainMenuScreen(Screen):
    """Main menu with command launchers (v0.11.0 — added Audit screen)."""

    CSS = """
    MainMenuScreen {
        align: center middle;
    }

    #menu-container {
        width: 50;
        height: auto;
        border: solid $accent;
        padding: 1 2;
    }

    #title {
        text-align: center;
        text-style: bold;
        padding: 1 0;
    }

    #subtitle {
        text-align: center;
        color: $text-muted;
        padding-bottom: 1;
    }

    Button {
        width: 100%;
        margin: 1 0;
    }

    #preset-label {
        text-align: center;
        color: $text-muted;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Container(id="menu-container"):
            yield Label("AttackLM GUI", id="title")
            yield Label("Terminal Training Manager", id="subtitle")
            yield Button("🏋️  Train Model", id="btn-train", variant="primary")
            yield Button("🚀  Init Dataset", id="btn-init")
            yield Button("⚖️  Balance Dataset", id="btn-balance")
            yield Button("📦  Build & Install", id="btn-build")
            yield Button("🧠  Run Inference", id="btn-infer")
            yield Button("📊  Evaluate", id="btn-eval")
            yield Button("🔍  Audit", id="btn-audit")
            yield Button("🧭  Steer Model", id="btn-steer")
            yield Button("📏  Benchmark", id="btn-bench")
            yield Label(
                "Presets: 3B Q-GaLore | 3B LoRA | 7B Q-GaLore | COAP | FP8 | BitNet",
                id="preset-label",
            )

    def on_mount(self) -> None:
        """Attach tooltips to all menu buttons."""
        for btn_id, tooltip_key in [
            ("#btn-train", "btn-train"),
            ("#btn-init", "btn-init"),
            ("#btn-balance", "btn-balance"),
            ("#btn-build", "btn-build"),
            ("#btn-infer", "btn-infer"),
            ("#btn-eval", "btn-eval"),
            ("#btn-audit", "btn-audit"),
            ("#btn-steer", "btn-steer"),
            ("#btn-bench", "btn-bench"),
        ]:
            try:
                attach_tooltip(self.query_one(btn_id), tooltip_key)
            except Exception:
                pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        btn_id = event.button.id

        if btn_id == "btn-train":
            from attacklm.gui.screens.train_form import TrainFormScreen

            self.app.push_screen(TrainFormScreen())
        elif btn_id == "btn-init":
            from attacklm.gui.screens.command_forms import InitFormScreen

            self.app.push_screen(InitFormScreen())
        elif btn_id == "btn-balance":
            from attacklm.gui.screens.command_forms import BalanceFormScreen

            self.app.push_screen(BalanceFormScreen())
        elif btn_id == "btn-build":
            from attacklm.gui.screens.command_forms import BuildFormScreen

            self.app.push_screen(BuildFormScreen())
        elif btn_id == "btn-infer":
            from attacklm.gui.screens.command_forms import InferFormScreen

            self.app.push_screen(InferFormScreen())
        elif btn_id == "btn-eval":
            from attacklm.gui.screens.command_forms import EvalFormScreen

            self.app.push_screen(EvalFormScreen())
        elif btn_id == "btn-audit":
            from attacklm.gui.screens.audit import AuditFormScreen

            self.app.push_screen(AuditFormScreen())
        elif btn_id == "btn-steer":
            from attacklm.gui.screens.command_forms import SteerFormScreen

            self.app.push_screen(SteerFormScreen())
        elif btn_id == "btn-bench":
            from attacklm.gui.screens.command_forms import BenchFormScreen

            self.app.push_screen(BenchFormScreen())
