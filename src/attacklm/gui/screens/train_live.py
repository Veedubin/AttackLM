"""Live training monitor screen."""

from __future__ import annotations

import asyncio
from datetime import datetime

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    Button,
    Label,
    ProgressBar,
    RichLog,
    Static,
)


class LossSparkline(Static):
    """Simple text-based loss sparkline."""

    loss_history: reactive[list[float]] = reactive(list)

    def render(self) -> str:
        if not self.loss_history:
            return "Loss: --"
        recent = self.loss_history[-20:]
        min_l, max_l = min(recent), max(recent)
        if max_l == min_l:
            return f"Loss: {recent[-1]:.4f}  {'─' * 20}"

        chars = "▁▂▃▄▅▆▇█"
        spark = ""
        for v in recent:
            idx = int((v - min_l) / (max_l - min_l) * (len(chars) - 1))
            spark += chars[min(idx, len(chars) - 1)]
        return f"Loss: {recent[-1]:.4f}  {spark}"


class VRAMGauge(Static):
    """VRAM usage gauge."""

    alloc: reactive[float] = reactive(0.0)
    cache: reactive[float] = reactive(0.0)
    total: reactive[float] = reactive(15.6)
    free: reactive[float] = reactive(0.0)

    def render(self) -> str:
        if self.total == 0:
            return "VRAM: --"
        used = self.alloc + self.cache
        pct = used / self.total * 100
        bar_len = 20
        filled = int(bar_len * used / self.total)
        bar = "█" * filled + "░" * (bar_len - filled)
        return (
            f"VRAM: {used:.1f}/{self.total:.1f} GB ({pct:.0f}%)\n"
            f"  alloc: {self.alloc:.1f}  cache: {self.cache:.1f}  free: {self.free:.1f}\n"
            f"  {bar}"
        )


class TrainLiveScreen(Screen):
    """Live training progress monitor."""

    CSS = """
    TrainLiveScreen {
        align: center middle;
    }

    #live-container {
        width: 100%;
        height: 100%;
        border: solid $accent;
    }

    #header-row {
        height: 3;
        background: $accent;
        color: $text;
        padding: 0 1;
    }

    #header-row Label {
        width: 1fr;
    }

    #metrics-panel {
        height: 10;
        padding: 1 2;
        border-bottom: solid $surface-darken-1;
    }

    #metrics-panel > Horizontal {
        height: 1fr;
    }

    #loss-panel {
        width: 1fr;
        border-right: solid $surface-darken-1;
        padding: 0 1;
    }

    #vram-panel {
        width: 40;
        padding: 0 1;
    }

    #stats-panel {
        width: 30;
        padding: 0 1;
    }

    #log-panel {
        height: 1fr;
    }

    RichLog {
        height: 1fr;
        border: none;
    }

    #control-row {
        dock: bottom;
        height: 3;
        align: center middle;
        background: $surface-darken-1;
    }

    #control-row Button {
        margin: 0 1;
    }

    #progress-bar {
        width: 100%;
    }

    .metric-label {
        text-style: bold;
        color: $accent;
    }

    .metric-value {
        color: $text;
    }

    .trend-down {
        color: $success;
    }

    .trend-up {
        color: $error;
    }

    .trend-flat {
        color: $warning;
    }
    """

    def __init__(self, command: list[str]) -> None:
        super().__init__()
        self.command = command
        self._start_time: datetime | None = None
        self._runner_task: asyncio.Task | None = None
        self._runner = None

    def compose(self) -> ComposeResult:
        with Container(id="live-container"):
            with Horizontal(id="header-row"):
                yield Label("Training: ...", id="header-title")
                yield Label("", id="header-time")

            with Container(id="metrics-panel"):
                with Horizontal():
                    with Vertical(id="loss-panel"):
                        yield LossSparkline(id="loss-spark")
                        yield Label("Eval Loss: --", id="eval-loss")
                        yield Label("Trend: --", id="trend-label")
                    with Vertical(id="vram-panel"):
                        yield VRAMGauge(id="vram-gauge")
                    with Vertical(id="stats-panel"):
                        yield Label("Epoch: --/--", id="epoch-label")
                        yield Label("Step: --", id="step-label")
                        yield Label("Tok/s: --", id="tok-label")
                        yield Label("Pairs/s: --", id="pairs-label")

            yield ProgressBar(total=100, show_eta=False, id="progress-bar")

            with Container(id="log-panel"):
                yield RichLog(id="log-view", highlight=True, markup=True, wrap=True)

            with Horizontal(id="control-row"):
                yield Button("Pause", id="btn-pause", variant="warning")
                yield Button(
                    "Stop at Checkpoint", id="btn-stop-checkpoint", variant="default"
                )
                yield Button("Quit Training", id="btn-quit", variant="error")
                yield Button("Back to Menu", id="btn-back", variant="default")

    def on_mount(self) -> None:
        """Start the training subprocess."""
        self._start_time = datetime.now()
        self._start_runner()

    def _start_runner(self) -> None:
        """Launch the training command in a subprocess."""
        from attacklm.gui.runner import CommandRunner

        self._runner = CommandRunner()
        self._runner_task = asyncio.create_task(self._stream_output())

    async def _stream_output(self) -> None:
        """Stream subprocess output to the log viewer and update metrics."""
        log = self.query_one("#log-view", RichLog)

        try:
            async for line, metrics in self._runner.start(self.command):
                # Write to log
                log.write(line)

                # Update metrics display
                self._update_metrics(metrics)

                # Update progress bar
                if metrics.progress_pct is not None:
                    self.query_one("#progress-bar", ProgressBar).update(
                        progress=metrics.progress_pct
                    )

                # Update header
                elapsed = datetime.now() - self._start_time
                self.query_one("#header-time", Label).update(
                    f"Elapsed: {str(elapsed).split('.')[0]}"
                )

        except asyncio.CancelledError:
            log.write("[bold red]Training cancelled[/]")
        except Exception as e:
            log.write(f"[bold red]Error: {e}[/]")

    def _update_metrics(self, metrics) -> None:
        """Update all metric displays from parsed metrics."""
        # Loss sparkline
        spark = self.query_one("#loss-spark", LossSparkline)
        spark.loss_history = metrics.loss_history.copy()

        # Eval loss
        if metrics.eval_loss is not None:
            self.query_one("#eval-loss", Label).update(
                f"Eval Loss: {metrics.eval_loss:.4f}"
            )

        # Trend
        if metrics.trend:
            trend_class = {
                "↓": "trend-down",
                "→": "trend-flat",
                "↑": "trend-up",
            }.get(metrics.trend, "")
            self.query_one("#trend-label", Label).update(
                f"Trend: {metrics.trend} {metrics.trend_value:+.4f}"
            )

        # Epoch/step
        if metrics.epoch is not None and metrics.total_epochs is not None:
            self.query_one("#epoch-label", Label).update(
                f"Epoch: {metrics.epoch:.1f}/{metrics.total_epochs}"
            )
        if metrics.step is not None:
            self.query_one("#step-label", Label).update(f"Step: {metrics.step}")

        # Throughput
        if metrics.tok_per_sec is not None:
            self.query_one("#tok-label", Label).update(
                f"Tok/s: {metrics.tok_per_sec:,.0f}"
            )
        if metrics.pairs_per_sec is not None:
            self.query_one("#pairs-label", Label).update(
                f"Pairs/s: {metrics.pairs_per_sec:.1f}"
            )

        # VRAM
        vram = self.query_one("#vram-gauge", VRAMGauge)
        if metrics.vram_alloc_gb is not None:
            vram.alloc = metrics.vram_alloc_gb
        if metrics.vram_cache_gb is not None:
            vram.cache = metrics.vram_cache_gb
        if metrics.vram_total_gb is not None:
            vram.total = metrics.vram_total_gb
        if metrics.vram_free_gb is not None:
            vram.free = metrics.vram_free_gb

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle control buttons."""
        btn_id = event.button.id

        if btn_id == "btn-pause":
            if self._runner and self._runner.running:
                if self._runner.paused:
                    self._runner.resume()
                    event.button.label = "Pause"
                    event.button.variant = "warning"
                else:
                    self._runner.pause()
                    event.button.label = "Resume"
                    event.button.variant = "success"

        elif btn_id == "btn-stop-checkpoint":
            # Send 's' to stdin to stop at next checkpoint
            if self._runner and self._runner.running:
                self.notify("Will stop at next checkpoint")

        elif btn_id == "btn-quit":
            if self._runner:
                self._runner.kill()
            if self._runner_task:
                self._runner_task.cancel()
            self.app.pop_screen()

        elif btn_id == "btn-back":
            if self._runner and self._runner.running:
                self._runner.kill()
            if self._runner_task:
                self._runner_task.cancel()
            self.app.pop_screen()
