"""Simpler command form screens for non-training AttackLM commands.

Uses the unified 'attacklm <subcommand>' CLI format (v0.10.0+).
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Input, Label, RichLog, Select, Static


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


class InitFormScreen(_BaseCommandScreen):
    """Initialize the AttackLM dataset.

    Uses 'attacklm init' with optional flags for individual steps.
    """

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
            yield self._row("Extra flags", "flags", placeholder="--yes --dry-run")
            with Horizontal(id="cmd-button-row"):
                yield Button("Run attacklm init", id="btn-run", variant="primary")
                yield Button("Back", id="btn-back")
            yield RichLog(id="cmd-output", highlight=True, wrap=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
        elif event.button.id == "btn-run":
            values = self._get_values()
            cmd = ["attacklm", "init"]
            if values.get("flags"):
                cmd.extend(values["flags"].split())
            import asyncio

            asyncio.create_task(self._run_and_display(cmd))


class BalanceFormScreen(_BaseCommandScreen):
    """Balance a dataset using 'attacklm balance'."""

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
            cmd = ["attacklm", "balance"]
            if values.get("input"):
                cmd.extend(["--input", values["input"]])
            if values.get("output"):
                cmd.extend(["--output", values["output"]])
            if values.get("cap"):
                cmd.extend(["--cap", values["cap"]])
            import asyncio

            asyncio.create_task(self._run_and_display(cmd))


class InferFormScreen(_BaseCommandScreen):
    """Run inference using 'attacklm infer'."""

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
            cmd = ["attacklm", "infer"]
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


class BuildFormScreen(_BaseCommandScreen):
    """Build: merge → GGUF → install using 'attacklm build'."""

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
                yield Button("Full Build", id="btn-run", variant="primary")
                yield Button("Merge Only", id="btn-merge")
                yield Button("GGUF Only", id="btn-gguf")
                yield Button("Back", id="btn-back")
            yield RichLog(id="cmd-output", highlight=True, wrap=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
        else:
            values = self._get_values()
            if event.button.id == "btn-merge":
                cmd = ["attacklm", "build", "--merge-only"]
            elif event.button.id == "btn-gguf":
                cmd = ["attacklm", "build", "--gguf-only"]
            else:
                cmd = ["attacklm", "build", "--install-lmstudio"]
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


class EvalFormScreen(_BaseCommandScreen):
    """Evaluation suite using 'attacklm eval'."""

    def compose(self) -> ComposeResult:
        with Container(id="cmd-container"):
            yield Label("Evaluation Suite", id="cmd-title")
            yield self._row(
                "Base Model", "base_model", placeholder="Qwen/Qwen2.5-Coder-3B-Instruct"
            )
            yield self._row("Adapter Path", "adapter", placeholder="models/attacklm-3b")
            yield self._row("Output", "output", placeholder="evals/retention.json")
            yield self._row("Extra flags", "flags", placeholder="--max-samples 50")
            with Horizontal(id="cmd-button-row"):
                yield Button("Retention Eval", id="btn-retention", variant="primary")
                yield Button("Collect Ref", id="btn-collect-ref")
                yield Button("Score", id="btn-score")
                yield Button("Compare", id="btn-compare")
                yield Button("Golden", id="btn-golden")
                yield Button("Back", id="btn-back")
            yield RichLog(id="cmd-output", highlight=True, wrap=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return

        values = self._get_values()
        # Build the base command
        if event.button.id == "btn-retention":
            cmd = ["attacklm", "eval"]
        elif event.button.id == "btn-collect-ref":
            cmd = ["attacklm", "eval", "--collect-ref"]
        elif event.button.id == "btn-score":
            cmd = ["attacklm", "eval", "--score"]
        elif event.button.id == "btn-compare":
            cmd = ["attacklm", "eval", "--compare"]
        elif event.button.id == "btn-golden":
            cmd = ["attacklm", "eval", "--golden"]
        else:
            return

        if values.get("base_model"):
            cmd.extend(["--base-model", values["base_model"]])
        if values.get("adapter"):
            cmd.extend(["--adapter", values["adapter"]])
        if values.get("output"):
            cmd.extend(["--output", values["output"]])
        if values.get("flags"):
            cmd.extend(values["flags"].split())
        import asyncio

        asyncio.create_task(self._run_and_display(cmd))


class PipelineFormScreen(_BaseCommandScreen):
    """Run the training pipeline using 'attacklm pipeline'."""

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
            cmd = ["attacklm", "pipeline"]
            if values.get("config"):
                cmd.extend(["--config", values["config"]])
            import asyncio

            asyncio.create_task(self._run_and_display(cmd))


class SteerFormScreen(_BaseCommandScreen):
    """Steering: extract/apply/sweep/diagnose activation vectors.

    Uses 'attacklm steer' with mode selector.
    """

    def compose(self) -> ComposeResult:
        with Container(id="cmd-container"):
            yield Label("Steer Model", id="cmd-title")
            yield Horizontal(
                Label("Mode:", classes="form-label"),
                Select(
                    [
                        ("extract", "extract"),
                        ("apply", "apply"),
                        ("sweep", "sweep"),
                        ("diagnose", "diagnose"),
                    ],
                    id="steer_mode",
                    value="extract",
                ),
                classes="form-row",
            )
            yield self._row(
                "Model Path",
                "model",
                placeholder="models/attacklm-3b-qgalore-spectrum",
            )
            yield self._row(
                "Output Dir",
                "output",
                placeholder="steering/vectors",
            )
            yield self._row("Extra flags", "flags", placeholder="--layers 12 16 20")
            with Horizontal(id="cmd-button-row"):
                yield Button("Run", id="btn-run", variant="primary")
                yield Button("Back", id="btn-back")
            yield RichLog(id="cmd-output", highlight=True, wrap=True)

    def _get_values(self) -> dict:
        values = super()._get_values()
        select = self.query_one("#steer_mode", Select)
        if select.value is not None:
            values["steer_mode"] = select.value
        return values

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
        elif event.button.id == "btn-run":
            values = self._get_values()
            mode = values.get("steer_mode", "extract")
            cmd = ["attacklm", "steer", mode]
            if values.get("model"):
                cmd.extend(["--model", values["model"]])
            if values.get("output"):
                cmd.extend(["--output", values["output"]])
            if values.get("flags"):
                cmd.extend(values["flags"].split())
            import asyncio

            asyncio.create_task(self._run_and_display(cmd))


class BenchFormScreen(_BaseCommandScreen):
    """Benchmark: evaluate model performance.

    Uses 'attacklm bench' with argv passthrough.
    """

    def compose(self) -> ComposeResult:
        with Container(id="cmd-container"):
            yield Label("Benchmark", id="cmd-title")
            yield self._row(
                "Model Path",
                "model",
                placeholder="models/attacklm-3b-qgalore-spectrum",
            )
            yield self._row(
                "Benchmark",
                "benchmark",
                placeholder="mmlu (or leave empty for default)",
            )
            yield self._row(
                "Extra flags", "flags", placeholder="--shots 5 --batch-size 8"
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
            cmd = ["attacklm", "bench"]
            if values.get("model"):
                cmd.extend(["--model", values["model"]])
            if values.get("benchmark"):
                cmd.extend(["--benchmark", values["benchmark"]])
            if values.get("flags"):
                cmd.extend(values["flags"].split())
            import asyncio

            asyncio.create_task(self._run_and_display(cmd))
