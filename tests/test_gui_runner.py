"""Tests for attacklm.gui.runner (parse_training_line + CommandRunner).

parse_training_line is a pure function with regex-based parsing; we test
it with a corpus of training-log lines that have appeared in real
QLoRA/GaLore/DeepSpeed runs. The goal is regression coverage: if anyone
changes a regex and breaks a known-good output format, the test fails.

CommandRunner is exercised via the public start/pause/resume/kill API
with a real subprocess (echo, sleep) — these are integration-level
smoke tests, not unit tests of the SIGSTOP/SIGCONT machinery.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from unittest.mock import MagicMock

import pytest

from attacklm.gui.runner import (
    CommandRunner,
    TrainingMetrics,
    parse_training_line,
)


# ---------------------------------------------------------------------------
# parse_training_line — regex coverage
# ---------------------------------------------------------------------------


class TestParseLoss:
    def test_basic_loss(self):
        m = parse_training_line("Epoch 1/20  10% | loss 2.6871", TrainingMetrics())
        assert m.loss == 2.6871

    def test_loss_history_appended(self):
        m = TrainingMetrics()
        parse_training_line("loss 2.0", m)
        parse_training_line("loss 1.5", m)
        parse_training_line("loss 1.0", m)
        assert m.loss_history == [2.0, 1.5, 1.0]

    def test_zero_loss_ignored(self):
        # "0.0000" is used as an eval-step marker; shouldn't overwrite
        # a real loss value.
        m = TrainingMetrics()
        parse_training_line("loss 2.5", m)
        parse_training_line("loss 0.0000", m)
        assert m.loss == 2.5  # preserved

    def test_loss_with_many_decimals(self):
        m = parse_training_line("loss 0.00123456", TrainingMetrics())
        assert m.loss == pytest.approx(0.00123456)


class TestParseEvalLoss:
    def test_basic_eval_loss(self):
        m = parse_training_line("Eval Loss: 2.4903", TrainingMetrics())
        assert m.eval_loss == 2.4903
        assert m.eval_loss_history == [2.4903]

    def test_eval_loss_does_not_set_train_loss(self):
        m = parse_training_line("Eval Loss: 2.4903", TrainingMetrics())
        assert m.loss is None


class TestParseEpoch:
    def test_basic_epoch(self):
        m = parse_training_line("Epoch 5/20", TrainingMetrics())
        assert m.epoch == 5.0
        assert m.total_epochs == 20

    def test_decimal_epoch(self):
        m = parse_training_line("Epoch 1.5/3", TrainingMetrics())
        assert m.epoch == 1.5
        assert m.total_epochs == 3


class TestParseStep:
    def test_basic_step(self):
        m = parse_training_line("Step 1234", TrainingMetrics())
        assert m.step == 1234


class TestParseTokPerSec:
    def test_basic(self):
        m = parse_training_line("1882 tok/s", TrainingMetrics())
        assert m.tok_per_sec == 1882.0

    def test_with_comma_separator(self):
        m = parse_training_line("12,345 tok/s", TrainingMetrics())
        assert m.tok_per_sec == 12345.0


class TestParsePairsPerSec:
    def test_basic(self):
        m = parse_training_line("0.5 pairs/s", TrainingMetrics())
        assert m.pairs_per_sec == 0.5


class TestParseVramCompact:
    def test_compact_format(self):
        # Newer training output: "VRAM 1.0/15.6 GB (5.9 alloc + 4.8 cache)"
        m = parse_training_line(
            "VRAM 1.0/15.6 GB (5.9 alloc + 4.8 cache)", TrainingMetrics()
        )
        # Compact regex sets alloc + cache + total.
        assert m.vram_alloc_gb == 5.9
        assert m.vram_cache_gb == 4.8
        assert m.vram_total_gb == 15.6

    def test_alloc_cache_total_format(self):
        m = parse_training_line("alloc 5.9 cache 4.8 /15.6 GB", TrainingMetrics())
        assert m.vram_alloc_gb == 5.9
        assert m.vram_cache_gb == 4.8
        assert m.vram_total_gb == 15.6


class TestParseVramFree:
    def test_free_total_format(self):
        m = parse_training_line(
            "Post-eval VRAM: 3.23GB free / 15.57GB total", TrainingMetrics()
        )
        assert m.vram_free_gb == 3.23
        assert m.vram_total_gb == 15.57


class TestParseTrend:
    def test_downward_trend(self):
        m = parse_training_line("trend ↓ -0.1516", TrainingMetrics())
        assert m.trend == "↓"
        assert m.trend_value == -0.1516

    def test_upward_trend(self):
        m = parse_training_line("trend ↑ +0.05", TrainingMetrics())
        assert m.trend == "↑"
        assert m.trend_value == 0.05

    def test_flat_trend(self):
        m = parse_training_line("trend → 0.0", TrainingMetrics())
        assert m.trend == "→"
        assert m.trend_value == 0.0


class TestParseProgress:
    def test_basic_percent(self):
        m = parse_training_line("50% done", TrainingMetrics())
        assert m.progress_pct == 50.0


class TestParseRealisticTrainingLine:
    """One full QLoRA training line, exercising every regex at once."""

    REAL_LINE = (
        "Epoch 3/20   15% | loss 1.8523 | 2,123 tok/s | 0.3 pairs/s | "
        "VRAM 1.0/15.6 GB (5.9 alloc + 4.8 cache) | trend ↓ -0.0516"
    )

    def test_all_fields(self):
        m = parse_training_line(self.REAL_LINE, TrainingMetrics())
        assert m.epoch == 3.0
        assert m.total_epochs == 20
        assert m.progress_pct == 15.0
        assert m.loss == 1.8523
        assert m.tok_per_sec == 2123.0
        assert m.pairs_per_sec == 0.3
        assert m.vram_alloc_gb == 5.9
        assert m.vram_cache_gb == 4.8
        assert m.vram_total_gb == 15.6
        assert m.trend == "↓"
        assert m.trend_value == -0.0516
        assert m.raw_line == self.REAL_LINE

    def test_empty_line_does_not_raise(self):
        m = parse_training_line("", TrainingMetrics())
        # All fields should be None / empty
        assert m.loss is None
        assert m.epoch is None
        assert m.raw_line == ""

    def test_garbage_line_does_not_raise(self):
        m = parse_training_line("???? not a training line ????", TrainingMetrics())
        # Should not raise, fields should remain None
        assert m.loss is None
        assert m.epoch is None
        assert m.tok_per_sec is None


# ---------------------------------------------------------------------------
# CommandRunner — public API smoke tests
# ---------------------------------------------------------------------------


class TestCommandRunnerProperties:
    def test_initial_state(self):
        r = CommandRunner()
        assert r.running is False
        assert r.paused is False
        assert isinstance(r.metrics, TrainingMetrics)
        assert r.metrics.loss is None

    def test_paused_default_false(self):
        r = CommandRunner()
        assert r.paused is False

    def test_pause_without_process_does_not_crash(self):
        r = CommandRunner()
        # Should be a no-op, not raise.
        r.pause()
        assert r.paused is False

    def test_resume_without_process_does_not_crash(self):
        r = CommandRunner()
        r.resume()
        assert r.paused is False

    def test_kill_without_process_does_not_crash(self):
        r = CommandRunner()
        # Should be a no-op, not raise.
        r.kill()


@pytest.mark.skipif(
    os.name == "nt", reason="CommandRunner uses os.killpg / signals (POSIX only)"
)
class TestCommandRunnerRealProcess:
    """End-to-end tests using a real subprocess. These exercise the
    start/pause/resume/kill machinery against a real OS process."""

    @pytest.mark.asyncio
    async def test_start_runs_command_and_yields_lines(self):
        r = CommandRunner()
        lines = []
        async for line, _metrics in r.start(
            [sys.executable, "-c", "print('hello'); print('world')"]
        ):
            lines.append(line)
        assert "hello" in lines
        assert "world" in lines
        assert r.running is False  # exited

    @pytest.mark.asyncio
    async def test_start_emits_exit_marker(self):
        r = CommandRunner()
        exit_marker_seen = False
        async for line, _metrics in r.start([sys.executable, "-c", "print('done')"]):
            if "Process exited" in line:
                exit_marker_seen = True
        assert exit_marker_seen

    @pytest.mark.asyncio
    async def test_pause_resume_sends_signals(self):
        # We can't easily test SIGSTOP in a Python test (it would freeze
        # the test runner). Instead, verify the methods are callable
        # without crashing when a process is running, and that
        # `paused` reflects state.
        r = CommandRunner()

        async def _drive():
            async for _line, _metrics in r.start(
                [sys.executable, "-c", "import time; time.sleep(0.5); print('ok')"]
            ):
                pass

        task = asyncio.create_task(_drive())
        await asyncio.sleep(0.05)  # let the subprocess start
        # The process may or may not be running by now depending on scheduler.
        # Don't assert on r.running; just verify pause/resume don't crash.
        try:
            r.pause()
            # Pause and immediately resume — process is short-lived.
            r.resume()
        except ProcessLookupError:
            # Process already exited; that's fine.
            pass
        await task


# ---------------------------------------------------------------------------
# TrainingMetrics dataclass
# ---------------------------------------------------------------------------


class TestTrainingMetrics:
    def test_default_construction(self):
        m = TrainingMetrics()
        assert m.loss is None
        assert m.eval_loss is None
        assert m.epoch is None
        assert m.step is None
        assert m.tok_per_sec is None
        assert m.pairs_per_sec is None
        assert m.vram_alloc_gb is None
        assert m.vram_cache_gb is None
        assert m.vram_total_gb is None
        assert m.vram_free_gb is None
        assert m.trend is None
        assert m.trend_value is None
        assert m.progress_pct is None
        assert m.raw_line == ""
        assert m.loss_history == []
        assert m.eval_loss_history == []

    def test_history_lists_are_independent_per_instance(self):
        # Regression: a shared default_factory would cause one metrics
        # object to see another's history.
        m1 = TrainingMetrics()
        m2 = TrainingMetrics()
        m1.loss_history.append(1.0)
        assert m2.loss_history == []
