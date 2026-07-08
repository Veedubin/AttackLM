"""Subprocess manager for running AttackLM CLI commands with real-time output."""

from __future__ import annotations

import asyncio
import os
import re
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator


@dataclass
class TrainingMetrics:
    """Parsed metrics from training output."""

    loss: float | None = None
    eval_loss: float | None = None
    epoch: float | None = None
    step: int | None = None
    total_epochs: int | None = None
    tok_per_sec: float | None = None
    pairs_per_sec: float | None = None
    vram_alloc_gb: float | None = None
    vram_cache_gb: float | None = None
    vram_total_gb: float | None = None
    vram_free_gb: float | None = None
    trend: str | None = None  # "↓", "→", "↑"
    trend_value: float | None = None
    progress_pct: float | None = None
    raw_line: str = ""

    # History
    loss_history: list[float] = field(default_factory=list)
    eval_loss_history: list[float] = field(default_factory=list)


# Regex patterns for parsing training output
_LOSS_RE = re.compile(r"loss\s+([\d.]+)")
_EVAL_LOSS_RE = re.compile(r"Eval Loss:\s*([\d.]+)")
_EPOCH_RE = re.compile(r"Epoch\s+([\d.]+)/(\d+)")
_STEP_RE = re.compile(r"Step\s+(\d+)")
_TOK_RE = re.compile(r"([\d,]+)\s*tok/s")
_PAIRS_RE = re.compile(r"([\d.]+)\s*pairs/s")
_VRAM_RE = re.compile(
    r"VRAM\s+(?:alloc\s+([\d.]+)\s*(?:cache\s+([\d.]+))?\s*/?\s*([\d.]+)\s*GB|"
    r"([\d.]+)GB\s*free\s*/\s*([\d.]+)GB\s*total)"
)
_VRAM_ALLOC_RE = re.compile(r"alloc\s+([\d.]+)")
_VRAM_CACHE_RE = re.compile(r"cache\s+([\d.]+)")
_VRAM_FREE_RE = re.compile(r"([\d.]+)GB\s*free\s*/\s*([\d.]+)GB\s*total")
_VRAM_COMPACT_RE = re.compile(
    r"VRAM\s+([\d.]+)/([\d.]+)\s*GB\s*\(\s*([\d.]+)\s*alloc\s*\+\s*([\d.]+)\s*cache"
)
_VRAM_ALLOC_CACHE_TOTAL_RE = re.compile(
    r"alloc\s+([\d.]+)\s*cache\s+([\d.]+)\s*/\s*([\d.]+)\s*GB"
)
_TREND_RE = re.compile(r"trend\s+([↓→↑])\s+([+-]?[\d.]+)")
_PROGRESS_RE = re.compile(r"(\d+)%")


def parse_training_line(line: str, metrics: TrainingMetrics) -> TrainingMetrics:
    """Parse a single line of training output and update metrics."""
    metrics.raw_line = line

    if m := _LOSS_RE.search(line):
        loss = float(m.group(1))
        if loss > 0:  # ignore 0.0000 (eval step marker)
            metrics.loss = loss
            metrics.loss_history.append(loss)

    if m := _EVAL_LOSS_RE.search(line):
        metrics.eval_loss = float(m.group(1))
        metrics.eval_loss_history.append(metrics.eval_loss)

    if m := _EPOCH_RE.search(line):
        metrics.epoch = float(m.group(1))
        metrics.total_epochs = int(m.group(2))

    if m := _STEP_RE.search(line):
        metrics.step = int(m.group(1))

    if m := _TOK_RE.search(line):
        metrics.tok_per_sec = float(m.group(1).replace(",", ""))

    if m := _PAIRS_RE.search(line):
        metrics.pairs_per_sec = float(m.group(1))

    # VRAM: try compact format first (newer training output), then individual patterns
    if m := _VRAM_COMPACT_RE.search(line):
        metrics.vram_alloc_gb = float(m.group(3))
        metrics.vram_cache_gb = float(m.group(4))
        metrics.vram_total_gb = float(m.group(2))
    if m := _VRAM_ALLOC_CACHE_TOTAL_RE.search(line):
        metrics.vram_alloc_gb = float(m.group(1))
        metrics.vram_cache_gb = float(m.group(2))
        metrics.vram_total_gb = float(m.group(3))
    if m := _VRAM_ALLOC_RE.search(line):
        metrics.vram_alloc_gb = float(m.group(1))
    if m := _VRAM_CACHE_RE.search(line):
        metrics.vram_cache_gb = float(m.group(1))
    if m := _VRAM_FREE_RE.search(line):
        metrics.vram_free_gb = float(m.group(1))
        metrics.vram_total_gb = float(m.group(2))

    if m := _TREND_RE.search(line):
        metrics.trend = m.group(1)
        metrics.trend_value = float(m.group(2))

    if m := _PROGRESS_RE.search(line):
        metrics.progress_pct = float(m.group(1))

    return metrics


async def run_command(
    cmd: list[str],
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> AsyncIterator[tuple[str, TrainingMetrics]]:
    """Run a command and yield (line, metrics) tuples as output arrives.

    Args:
        cmd: Command and arguments to execute.
        cwd: Working directory for the subprocess.
        env: Environment variables (merged with current env).

    Yields:
        Tuple of (raw_line, current_metrics) for each line of output.
    """
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(cwd) if cwd else None,
        env=full_env,
    )

    metrics = TrainingMetrics()

    async def _read_stream() -> None:
        """Read lines from stdout. This is a helper for the async generator."""
        pass

    if process.stdout:
        async for line in process.stdout:
            decoded = line.decode("utf-8", errors="replace").rstrip("\n")
            metrics = parse_training_line(decoded, metrics)
            yield decoded, metrics

    await process.wait()
    metrics.raw_line = f"Process exited with code {process.returncode}"
    yield metrics.raw_line, metrics


class CommandRunner:
    """Manages a running subprocess with pause/resume/kill support."""

    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None
        self._paused = False
        self.metrics = TrainingMetrics()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    @property
    def paused(self) -> bool:
        return self._paused

    async def start(
        self, cmd: list[str], cwd: str | Path | None = None
    ) -> AsyncIterator[tuple[str, TrainingMetrics]]:
        """Start a command and yield output lines."""
        full_env = os.environ.copy()

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(cwd) if cwd else None,
            env=full_env,
            preexec_fn=os.setsid,  # create process group for signal sending
        )

        self.metrics = TrainingMetrics()

        if self._process.stdout:
            async for line in self._process.stdout:
                decoded = line.decode("utf-8", errors="replace").rstrip("\n")
                self.metrics = parse_training_line(decoded, self.metrics)
                yield decoded, self.metrics

        await self._process.wait()
        self.metrics.raw_line = f"Process exited with code {self._process.returncode}"
        yield self.metrics.raw_line, self.metrics

    def pause(self) -> None:
        """Pause the running process (SIGSTOP)."""
        if self._process and self._process.returncode is None:
            os.killpg(os.getpgid(self._process.pid), signal.SIGSTOP)
            self._paused = True

    def resume(self) -> None:
        """Resume a paused process (SIGCONT)."""
        if self._process and self._process.returncode is None:
            os.killpg(os.getpgid(self._process.pid), signal.SIGCONT)
            self._paused = False

    def kill(self) -> None:
        """Kill the running process (SIGTERM, then SIGKILL)."""
        if self._process and self._process.returncode is None:
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            self._paused = False
