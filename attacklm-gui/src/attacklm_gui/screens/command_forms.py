"""Simpler command form screens for non-training AttackLM commands."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Input, Label, RichLog, Static


class _BaseCommandScreen(Screen):
    """Base class for simple command form screens."""

    CSS = """
    _BaseCommandScreen {
        align: center middle;
    }

    #cmd-container {
        width: 60;
        height: 80%;
        border: solid $accent;
        background: $surface;
    }

    #cmd-title {
        text-align: center;
        text-style: bold;
        padding: 1 0;
        background: $accent;
        color: $text;
    }

    .form-row {
        height: 3;
        margin: 0 0;
        padding: 0 1;
    }

    .form-label {
        width: 20;
        padding: 0 1;
        text-align: right;
    }

    .form-input {
        width: 35;
    }

    #cmd-output {
        height: 1fr;
        border-top: solid $surface-darken-1;
    }

    #cmd-button-row {
        dock: bottom;
        height: 3;
        align: center middle;
    }

    #cmd-button-row Button {
        margin: 0 1;
    }
    """

    def _row(
        self, label: str, input_id: str, placeholder: str = "", value: str = ""
    ) -> Horizontal:
        return Horizontal(
            Label(label, classes="form-label"),
            Input(
                placeholder=placeholder, value=value, id=input_id, classes="form-input"
            ),
            classes="form-row",
        )

    def _get_values(self) -> dict:
        values = {}
        for widget in self.query("Input"):
            if widget.id:
                values[widget.id] = widget.value
        return values

    async def _run_and_display(self, cmd: list[str]) -> None:
        """Run a command and display output."""
        import asyncio

        log = self.query_one("#cmd-output", RichLog)
        log.clear()

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        if process.stdout:
            async for line in process.stdout:
                decoded = line.decode("utf-8", errors="replace").rstrip("\n")
                log.write(decoded)

        await process.wait()
        log.write(f"\n[bold]Exit code: {process.returncode}[/]")


class ExtractFormScreen(_BaseCommandScreen):
    """Extract data from sources."""

    def compose(self) -> ComposeResult:
        with Container(id="cmd-container"):
            yield Label("Extract Data", id="cmd-title")
            yield self._row("Source (optional)", "source", placeholder="all")
            yield self._row("Output Dir", "output", placeholder="data/datasets/buckets")
            with Horizontal(id="cmd-button-row"):
                yield Button("Run", id="btn-run", variant="primary")
                yield Button("Back", id="btn-back")
            yield RichLog(id="cmd-output", highlight=True, wrap=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
        elif event.button.id == "btn-run":
            values = self._get_values()
            cmd = ["attacklm-extract"]
            if values.get("source"):
                cmd.extend(["--source", values["source"]])
            if values.get("output"):
                cmd.extend(["--output", values["output"]])
            import asyncio

            asyncio.create_task(self._run_and_display(cmd))


class BalanceFormScreen(_BaseCommandScreen):
    """Balance a dataset."""

    def compose(self) -> ComposeResult:
        with Container(id="cmd-container"):
            yield Label("Balance Dataset", id="cmd-title")
            yield self._row(
                "Input Dataset", "input", placeholder="data/datasets/buckets"
            )
            yield self._row(
                "Output Path",
                "output",
                placeholder="data/datasets/balanced/balanced.jsonl",
            )
            yield self._row("Cap Size", "cap", placeholder="1000")
            with Horizontal(id="cmd-button-row"):
                yield Button("Run", id="btn-run", variant="primary")
                yield Button("Back", id="btn-back")
            yield RichLog(id="cmd-output", highlight=True, wrap=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
        elif event.button.id == "btn-run":
            values = self._get_values()
            cmd = ["attacklm-balance"]
            if values.get("input"):
                cmd.extend(["--input", values["input"]])
            if values.get("output"):
                cmd.extend(["--output", values["output"]])
            if values.get("cap"):
                cmd.extend(["--cap", values["cap"]])
            import asyncio

            asyncio.create_task(self._run_and_display(cmd))


class InferFormScreen(_BaseCommandScreen):
    """Run inference with a trained model."""

    def compose(self) -> ComposeResult:
        with Container(id="cmd-container"):
            yield Label("Run Inference", id="cmd-title")
            yield self._row(
                "Model Path",
                "model",
                placeholder="models/attacklm-3b-qgalore-spectrum_2026-06-29_20-14",
            )
            yield self._row(
                "Prompt",
                "prompt",
                placeholder="Explain how to detect a phishing attack",
            )
            yield self._row("Max Tokens", "max_tokens", placeholder="512", value="512")
            yield self._row(
                "Temperature", "temperature", placeholder="0.7", value="0.7"
            )
            with Horizontal(id="cmd-button-row"):
                yield Button("Run", id="btn-run", variant="primary")
                yield Button("Back", id="btn-back")
            yield RichLog(id="cmd-output", highlight=True, wrap=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
        elif event.button.id == "btn-run":
            values = self._get_values()
            cmd = ["attacklm-infer"]
            if values.get("model"):
                cmd.extend(["--model", values["model"]])
            if values.get("prompt"):
                cmd.extend(["--prompt", values["prompt"]])
            if values.get("max_tokens"):
                cmd.extend(["--max-tokens", values["max_tokens"]])
            if values.get("temperature"):
                cmd.extend(["--temperature", values["temperature"]])
            import asyncio

            asyncio.create_task(self._run_and_display(cmd))


class MergeFormScreen(_BaseCommandScreen):
    """Merge LoRA adapter into base model."""

    def compose(self) -> ComposeResult:
        with Container(id="cmd-container"):
            yield Label("Merge Adapter", id="cmd-title")
            yield self._row(
                "Base Model", "base", placeholder="Qwen/Qwen2.5-Coder-3B-Instruct"
            )
            yield self._row("Adapter Path", "adapter", placeholder="models/my-adapter")
            yield self._row(
                "Output Path", "output", placeholder="models/merged/my-model"
            )
            with Horizontal(id="cmd-button-row"):
                yield Button("Run", id="btn-run", variant="primary")
                yield Button("Back", id="btn-back")
            yield RichLog(id="cmd-output", highlight=True, wrap=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
        elif event.button.id == "btn-run":
            values = self._get_values()
            cmd = ["attacklm-merge"]
            if values.get("base"):
                cmd.extend(["--base", values["base"]])
            if values.get("adapter"):
                cmd.extend(["--adapter", values["adapter"]])
            if values.get("output"):
                cmd.extend(["--output", values["output"]])
            import asyncio

            asyncio.create_task(self._run_and_display(cmd))


class BuildFormScreen(_BaseCommandScreen):
    """Build: merge → GGUF → install to LM Studio."""

    def compose(self) -> ComposeResult:
        with Container(id="cmd-container"):
            yield Label("Build & Install", id="cmd-title")
            yield self._row(
                "Adapter Path",
                "adapter",
                placeholder="models/my-adapter (or leave empty for --merged)",
            )
            yield self._row(
                "Merged Model",
                "merged",
                placeholder="models/attacklm-3b-qgalore-spectrum_2026-06-29_20-14",
            )
            yield self._row("Model Name", "name", placeholder="attacklm-3b-qgalore")
            yield self._row(
                "Quantization", "quant", placeholder="Q4_K_M", value="Q4_K_M"
            )
            with Horizontal(id="cmd-button-row"):
                yield Button("Run", id="btn-run", variant="primary")
                yield Button("Back", id="btn-back")
            yield RichLog(id="cmd-output", highlight=True, wrap=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
        elif event.button.id == "btn-run":
            values = self._get_values()
            cmd = ["attacklm-build", "--install-lmstudio"]
            if values.get("adapter"):
                cmd.extend(["--adapter", values["adapter"]])
            if values.get("merged"):
                cmd.extend(["--merged", values["merged"]])
            if values.get("name"):
                cmd.extend(["--name", values["name"]])
            if values.get("quant"):
                cmd.extend(["--quant", values["quant"]])
            import asyncio

            asyncio.create_task(self._run_and_display(cmd))


class PipelineFormScreen(_BaseCommandScreen):
    """Run the training pipeline."""

    def compose(self) -> ComposeResult:
        with Container(id="cmd-container"):
            yield Label("Pipeline", id="cmd-title")
            yield self._row("Config YAML", "config", placeholder="pipeline.yaml")
            with Horizontal(id="cmd-button-row"):
                yield Button("Run", id="btn-run", variant="primary")
                yield Button("Back", id="btn-back")
            yield RichLog(id="cmd-output", highlight=True, wrap=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
        elif event.button.id == "btn-run":
            values = self._get_values()
            cmd = ["attacklm-pipeline"]
            if values.get("config"):
                cmd.extend(["--config", values["config"]])
            import asyncio

            asyncio.create_task(self._run_and_display(cmd))


class InitFormScreen(_BaseCommandScreen):
    """Initialize the AttackLM dataset."""

    def compose(self) -> ComposeResult:
        with Container(id="cmd-container"):
            yield Label("Initialize Dataset", id="cmd-title")
            yield Label(
                "This will clone all upstream data sources and run extractors.",
                id="cmd-desc",
            )
            yield Label(
                "This may take 10-30 minutes depending on network speed.", id="cmd-warn"
            )
            with Horizontal(id="cmd-button-row"):
                yield Button("Run attacklm-init", id="btn-run", variant="primary")
                yield Button("Back", id="btn-back")
            yield RichLog(id="cmd-output", highlight=True, wrap=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
        elif event.button.id == "btn-run":
            import asyncio

            asyncio.create_task(self._run_and_display(["attacklm-init"]))
