"""Main menu screen for AttackLM GUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Label, Static


class MainMenuScreen(Screen):
    """Main menu with command launchers."""

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
            yield Button("📊  Extract Data", id="btn-extract")
            yield Button("⚖️  Balance Dataset", id="btn-balance")
            yield Button("🧠  Run Inference", id="btn-infer")
            yield Button("🔗  Merge Adapter", id="btn-merge")
            yield Button("📦  Build & Install", id="btn-build")
            yield Button("🔧  Pipeline", id="btn-pipeline")
            yield Button("🚀  Init Dataset", id="btn-init")
            yield Label(
                "Presets: 3B Q-GaLore | 3B LoRA | 7B Q-GaLore", id="preset-label"
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        btn_id = event.button.id

        if btn_id == "btn-train":
            from attacklm_gui.screens.train_form import TrainFormScreen

            self.app.push_screen(TrainFormScreen())
        elif btn_id == "btn-extract":
            from attacklm_gui.screens.command_forms import ExtractFormScreen

            self.app.push_screen(ExtractFormScreen())
        elif btn_id == "btn-balance":
            from attacklm_gui.screens.command_forms import BalanceFormScreen

            self.app.push_screen(BalanceFormScreen())
        elif btn_id == "btn-infer":
            from attacklm_gui.screens.command_forms import InferFormScreen

            self.app.push_screen(InferFormScreen())
        elif btn_id == "btn-merge":
            from attacklm_gui.screens.command_forms import MergeFormScreen

            self.app.push_screen(MergeFormScreen())
        elif btn_id == "btn-build":
            from attacklm_gui.screens.command_forms import BuildFormScreen

            self.app.push_screen(BuildFormScreen())
        elif btn_id == "btn-pipeline":
            from attacklm_gui.screens.command_forms import PipelineFormScreen

            self.app.push_screen(PipelineFormScreen())
        elif btn_id == "btn-init":
            from attacklm_gui.screens.command_forms import InitFormScreen

            self.app.push_screen(InitFormScreen())
