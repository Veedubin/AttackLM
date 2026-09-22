"""Tests for the Queue screen (v0.19.0)."""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, RichLog, Select

from attacklm.gui.app import AttackLMApp
from attacklm.gui.screens.queue import QueueScreen
from attacklm.gui.widgets.tooltips import TOOLTIPS
from attacklm.queue.db import QueueDB


def _patch_presets(monkeypatch, tmp_path):
    monkeypatch.setattr("attacklm.gui.app.ensure_builtin_presets", lambda: None)


WIDGET_IDS = ["queue-table", "queue_base_model", "queue_adapter", "queue_attack",
              "btn-queue-refresh", "btn-queue-enqueue", "btn-queue-start",
              "btn-queue-stop", "btn-queue-retry", "btn-back", "cmd-output"]


class TestQueueScreen:
    @pytest.mark.asyncio
    async def test_mounts_and_has_widgets(self, tmp_path, monkeypatch):
        _patch_presets(monkeypatch, tmp_path)
        app = AttackLMApp()
        async with app.run_test() as pilot:
            app.push_screen(QueueScreen(db_path=tmp_path / "q.db"))
            await pilot.pause()
            assert app.screen.__class__.__name__ == "QueueScreen"
            for wid in WIDGET_IDS:
                assert app.screen.query_one(f"#{wid}") is not None, wid

    @pytest.mark.asyncio
    async def test_table_lists_tasks(self, tmp_path, monkeypatch):
        _patch_presets(monkeypatch, tmp_path)
        db = QueueDB(tmp_path / "q.db")
        db.add_task(type="audit_prompt_injection", label="PI smoke", args={"adapter": "/a"})
        app = AttackLMApp()
        async with app.run_test() as pilot:
            app.push_screen(QueueScreen(db_path=db.db_path))
            await pilot.pause()
            table = app.screen.query_one("#queue-table")
            assert table.row_count == 1
            assert "PI smoke" in str(table.get_row_at(0))

    @pytest.mark.asyncio
    async def test_inputs_have_tooltips(self, tmp_path, monkeypatch):
        _patch_presets(monkeypatch, tmp_path)
        app = AttackLMApp()
        async with app.run_test() as pilot:
            app.push_screen(QueueScreen(db_path=tmp_path / "q.db"))
            await pilot.pause()
            for wid in ("queue_base_model", "queue_adapter", "queue_attack",
                        "btn-queue-enqueue", "btn-queue-start", "btn-queue-stop", "btn-queue-retry"):
                assert app.screen.query_one(f"#{wid}").tooltip, wid

    def test_main_menu_tooltip_key(self):
        assert "btn-queue" in TOOLTIPS

    def test_enqueue_command_shape(self, tmp_path):
        screen = QueueScreen(db_path=tmp_path / "q.db")
        cmd = screen._enqueue_cmd(base_model="b", adapter="/a", attack="core")
        assert cmd[:4] == ["attacklm", "queue", "--db-path", str(tmp_path / "q.db")]
        assert "gauntlet" in cmd and "core" in cmd
        cmd = screen._enqueue_cmd(base_model="b", adapter="/a", attack="1")
        assert "add-audit" in cmd and cmd[cmd.index("--attack") + 1] == "1"
        assert cmd[cmd.index("--adapter") + 1] == "/a" and cmd[cmd.index("--base-model") + 1] == "b"

    def test_enqueue_command_chains_to_latest_when_fields_empty(self, tmp_path):
        """IMPORTANT #1 (final review): with neither field filled, a
        gauntlet/audit has no adapter, base model, or deps and the runner
        marks it failed immediately — chain to the latest train task."""
        screen = QueueScreen(db_path=tmp_path / "q.db")
        cmd = screen._enqueue_cmd(base_model="", adapter="", attack="core")
        assert cmd[cmd.index("--after") + 1] == "latest"
        cmd = screen._enqueue_cmd(base_model="", adapter="", attack="1")
        assert "add-audit" in cmd
        assert cmd[cmd.index("--depends-on") + 1] == "latest"
        assert "--base-model" not in cmd and "--adapter" not in cmd


class TestQueueButtonPresses:
    """Exercise on_button_pressed via real button presses (Button.press()
    posts the Pressed event without needing screen coordinates), with
    QueueScreen._run_then_refresh monkeypatched to capture the command
    list(s) instead of spawning subprocesses."""

    def _capture(self, monkeypatch):
        captured: list[list[list[str]]] = []

        async def fake_run_then_refresh(self, cmds):
            captured.append(cmds)

        monkeypatch.setattr(QueueScreen, "_run_then_refresh", fake_run_then_refresh)
        return captured

    @pytest.mark.asyncio
    async def test_enqueue_numeric_attack_with_model_fields(self, tmp_path, monkeypatch):
        _patch_presets(monkeypatch, tmp_path)
        captured = self._capture(monkeypatch)
        app = AttackLMApp()
        async with app.run_test() as pilot:
            app.push_screen(QueueScreen(db_path=tmp_path / "q.db"))
            await pilot.pause()
            app.screen.query_one("#queue_adapter", Input).value = "/a"
            app.screen.query_one("#queue_attack", Select).value = "1"
            app.screen.query_one("#btn-queue-enqueue", Button).press()
            await pilot.pause()

        assert len(captured) == 1
        cmds = captured[0]
        assert len(cmds) == 1
        cmd = cmds[0]
        assert "add-audit" in cmd
        assert cmd[cmd.index("--attack") + 1] == "1"
        assert cmd[cmd.index("--adapter") + 1] == "/a"

    @pytest.mark.asyncio
    async def test_enqueue_numeric_attack_without_fields(self, tmp_path, monkeypatch):
        """IMPORTANT #1 (final review): a single audit with no model/adapter
        must chain to the latest train task, not enqueue an unrunnable task."""
        _patch_presets(monkeypatch, tmp_path)
        captured = self._capture(monkeypatch)
        app = AttackLMApp()
        async with app.run_test() as pilot:
            app.push_screen(QueueScreen(db_path=tmp_path / "q.db"))
            await pilot.pause()
            app.screen.query_one("#queue_attack", Select).value = "1"
            app.screen.query_one("#btn-queue-enqueue", Button).press()
            await pilot.pause()

        assert len(captured) == 1
        cmds = captured[0]
        assert len(cmds) == 1
        cmd = cmds[0]
        assert "add-audit" in cmd
        assert cmd[cmd.index("--attack") + 1] == "1"
        assert cmd[cmd.index("--depends-on") + 1] == "latest"
        assert "--base-model" not in cmd and "--adapter" not in cmd

    @pytest.mark.asyncio
    async def test_enqueue_gauntlet_quick_no_model_fields(self, tmp_path, monkeypatch):
        _patch_presets(monkeypatch, tmp_path)
        captured = self._capture(monkeypatch)
        app = AttackLMApp()
        async with app.run_test() as pilot:
            app.push_screen(QueueScreen(db_path=tmp_path / "q.db"))
            await pilot.pause()
            app.screen.query_one("#queue_attack", Select).value = "quick"
            app.screen.query_one("#btn-queue-enqueue", Button).press()
            await pilot.pause()

        assert len(captured) == 1
        cmds = captured[0]
        assert len(cmds) == 1
        cmd = cmds[0]
        # IMPORTANT #1 (final review): with no model/adapter given, chain to
        # the latest train task instead of enqueueing an unrunnable gauntlet.
        assert "gauntlet" in cmd and "quick" in cmd
        assert cmd[-2:] == ["--after", "latest"]

    @pytest.mark.asyncio
    async def test_enqueue_gauntlet_quick_with_adapter_expands(self, tmp_path, monkeypatch):
        _patch_presets(monkeypatch, tmp_path)
        captured = self._capture(monkeypatch)
        app = AttackLMApp()
        async with app.run_test() as pilot:
            app.push_screen(QueueScreen(db_path=tmp_path / "q.db"))
            await pilot.pause()
            app.screen.query_one("#queue_adapter", Input).value = "/a"
            app.screen.query_one("#queue_attack", Select).value = "quick"
            app.screen.query_one("#btn-queue-enqueue", Button).press()
            await pilot.pause()

        assert len(captured) == 1
        cmds = captured[0]
        assert len(cmds) == 2
        attacks = []
        for cmd in cmds:
            assert "add-audit" in cmd
            assert cmd[cmd.index("--adapter") + 1] == "/a"
            attacks.append(cmd[cmd.index("--attack") + 1])
        assert attacks == ["1", "2"]

    @pytest.mark.asyncio
    async def test_start_runner_button(self, tmp_path, monkeypatch):
        _patch_presets(monkeypatch, tmp_path)
        captured = self._capture(monkeypatch)
        app = AttackLMApp()
        async with app.run_test() as pilot:
            app.push_screen(QueueScreen(db_path=tmp_path / "q.db"))
            await pilot.pause()
            app.screen.query_one("#btn-queue-start", Button).press()
            await pilot.pause()

        db_path = tmp_path / "q.db"
        assert captured == [[["attacklm", "queue", "--db-path", str(db_path), "start", "--detach"]]]

    @pytest.mark.asyncio
    async def test_stop_runner_button(self, tmp_path, monkeypatch):
        _patch_presets(monkeypatch, tmp_path)
        captured = self._capture(monkeypatch)
        app = AttackLMApp()
        async with app.run_test() as pilot:
            app.push_screen(QueueScreen(db_path=tmp_path / "q.db"))
            await pilot.pause()
            app.screen.query_one("#btn-queue-stop", Button).press()
            await pilot.pause()

        db_path = tmp_path / "q.db"
        assert captured == [[["attacklm", "queue", "--db-path", str(db_path), "stop"]]]

    @pytest.mark.asyncio
    async def test_retry_with_empty_table_shows_message(self, tmp_path, monkeypatch):
        _patch_presets(monkeypatch, tmp_path)
        captured = self._capture(monkeypatch)
        app = AttackLMApp()
        async with app.run_test() as pilot:
            app.push_screen(QueueScreen(db_path=tmp_path / "q.db"))
            await pilot.pause()
            app.screen.query_one("#btn-queue-retry", Button).press()
            await pilot.pause()
            log = app.screen.query_one("#cmd-output", RichLog)
            text = "\n".join(strip.text for strip in log.lines)

        assert captured == []
        assert "Select a task first" in text
