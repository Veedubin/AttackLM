"""Queue screen — view the task queue, enqueue audits/gauntlets, start/stop the runner.

Reads the queue DB directly (read-only) and shells out to `attacklm queue …`
for every mutation, so the TUI and the CLI can never disagree about state.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Checkbox, DataTable, Label, RichLog, Select

from attacklm.gui.screens.command_forms import _BaseCommandScreen
from attacklm.gui.widgets import attach_tooltip
from attacklm.queue.db import DEFAULT_DB_NAME, DEFAULT_QUEUE_DIR, QueueDB

_ATTACK_CHOICES = [
    ("1 — Prompt injection", "1"),
    ("2 — System-prompt extraction", "2"),
    ("3 — Canary extraction", "3"),
    ("7 — Calibration", "7"),
    ("Gauntlet: core (1,2,3,7)", "core"),
    ("Gauntlet: quick (1,2)", "quick"),
]
# Members of each gauntlet preset, by short attack id — kept as a local
# dict (rather than importing attacklm.queue.gauntlet.GAUNTLET_PRESETS) so
# this screen stays decoupled from the queue internals other agents are
# editing in parallel. Must stay consistent with _ATTACK_CHOICES above.
_GAUNTLET_MEMBERS: dict[str, list[str]] = {
    "core": ["1", "2", "3", "7"],
    "quick": ["1", "2"],
}
_GAUNTLETS = set(_GAUNTLET_MEMBERS)
_COLUMNS = ("#", "Type", "Status", "Label", "Depends", "Artifact")


class QueueScreen(_BaseCommandScreen):
    """Task queue: list, enqueue, start/stop the runner, retry."""

    CSS = _BaseCommandScreen.CSS + """
    #cmd-container { width: 100; }
    #queue-table { height: 12; }
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        super().__init__()
        self._db_path = Path(db_path) if db_path else DEFAULT_QUEUE_DIR / DEFAULT_DB_NAME

    def compose(self) -> ComposeResult:
        with Container(id="cmd-container"):
            yield Label("Queue", id="cmd-title")
            yield DataTable(id="queue-table", cursor_type="row")
            yield self._row("Base model", "queue_base_model",
                            placeholder="huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated")
            yield self._row("Adapter", "queue_adapter", placeholder="models/attacklm-3b_16g")
            yield Horizontal(
                Label("Attack:", classes="form-label"),
                # allow_blank=False: Select.BLANK is truthy, so a cleared
                # Select would silently bypass the `attack or "core"`
                # fallback below and produce a garbage command.
                Select(_ATTACK_CHOICES, id="queue_attack", value="core", allow_blank=False),
                classes="form-row",
            )
            yield Horizontal(
                Checkbox(
                    "Baseline (compare against the untuned base model)",
                    value=True,
                    id="queue_baseline",
                ),
                classes="form-row",
            )
            with Horizontal(id="cmd-button-row"):
                yield Button("Enqueue", id="btn-queue-enqueue", variant="primary")
                yield Button("Start runner", id="btn-queue-start", variant="success")
                yield Button("Stop runner", id="btn-queue-stop")
                yield Button("Retry selected", id="btn-queue-retry")
                yield Button("Compare latest", id="btn-queue-compare")
                yield Button("Refresh", id="btn-queue-refresh")
                yield Button("Back", id="btn-back")
            yield RichLog(id="cmd-output", highlight=True, wrap=True)

    def on_mount(self) -> None:
        super().on_mount()
        table = self.query_one("#queue-table", DataTable)
        table.add_columns(*_COLUMNS)
        for wid in ("queue_base_model", "queue_adapter", "queue_attack", "queue_baseline",
                    "btn-queue-enqueue", "btn-queue-start", "btn-queue-stop", "btn-queue-retry",
                    "btn-queue-compare"):
            try:
                attach_tooltip(self.query_one(f"#{wid}"), wid)
            except Exception:
                pass
        self._refresh()

    # ----- data -----------------------------------------------------------

    def _refresh(self) -> None:
        table = self.query_one("#queue-table", DataTable)
        table.clear()
        db = QueueDB(self._db_path)
        for t in db.list_tasks(limit=200):
            table.add_row(str(t.id), t.type, t.status, t.label,
                          ",".join(str(d) for d in t.depends_on_list) or "—",
                          t.artifact_path or "", key=str(t.id))
        runner = db.get_all_runner_state()
        pid = runner.get("pid") or ""
        self.query_one("#cmd-title", Label).update(
            f"Queue — runner {'PID ' + pid if pid else 'not running'} — {self._db_path}"
        )

    def _selected_task_id(self) -> int | None:
        table = self.query_one("#queue-table", DataTable)
        if table.row_count == 0:
            return None
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        return int(row_key.value) if row_key and row_key.value else None

    # ----- commands -------------------------------------------------------

    def _base_cmd(self) -> list[str]:
        return ["attacklm", "queue", "--db-path", str(self._db_path)]

    def _enqueue_cmd(self, base_model: str, adapter: str, attack: str, baseline: bool = True) -> list[str]:
        cmd = self._base_cmd()
        # With neither field filled, a gauntlet/audit has no adapter, no
        # base model, and no deps — the runner would mark it failed on the
        # spot. Chain it to the latest train task instead; if there isn't
        # one yet, the CLI reports that into the log rather than silently
        # creating an unrunnable task.
        chain_to_latest = not base_model and not adapter
        if attack in _GAUNTLETS:
            cmd += ["gauntlet", attack]
            if chain_to_latest:
                cmd += ["--after", "latest"]
            if not baseline:
                cmd += ["--no-baseline"]
        else:
            cmd += ["add-audit", "--attack", attack]
            if base_model:
                cmd += ["--base-model", base_model]
            if adapter:
                cmd += ["--adapter", adapter]
            if chain_to_latest:
                cmd += ["--depends-on", "latest"]
        return cmd

    def _enqueue_cmds(
        self, base_model: str, adapter: str, attack: str, baseline: bool = True
    ) -> list[list[str]]:
        """Build the command(s) to run for the Enqueue button.

        Gauntlets normally just enqueue `gauntlet <preset>`, which depends on
        the latest train task and inherits its model/adapter. But if the user
        explicitly filled in a base model or adapter, that selection must be
        preserved — so expand the preset into one `add-audit` per member
        instead of silently running `--attack all` (which would drop the
        preset choice, e.g. 'quick' becoming every shipped attack).

        `baseline` (the queue_baseline checkbox) only affects the plain
        `gauntlet <preset>` command: `--no-baseline` disables the auto-queued
        base-model baseline run added by `attacklm queue gauntlet` (see
        commit 99cb095). The expanded add-audit path below never auto-queues
        a baseline, so `baseline` doesn't apply there.
        """
        if attack in _GAUNTLETS and (base_model or adapter):
            cmds = []
            for member in _GAUNTLET_MEMBERS[attack]:
                cmd = self._base_cmd() + ["add-audit", "--attack", member]
                if base_model:
                    cmd += ["--base-model", base_model]
                if adapter:
                    cmd += ["--adapter", adapter]
                cmds.append(cmd)
            return cmds
        return [self._enqueue_cmd(base_model, adapter, attack, baseline)]

    async def _run_then_refresh(self, cmds: list[list[str]]) -> None:
        for cmd in cmds:
            await self._run_and_display(cmd)
        self._refresh()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-back":
            self.app.pop_screen()
        elif bid == "btn-queue-refresh":
            self._refresh()
        elif bid == "btn-queue-enqueue":
            values = self._get_values()
            attack = str(self.query_one("#queue_attack", Select).value or "core")
            baseline = self.query_one("#queue_baseline", Checkbox).value
            cmds = self._enqueue_cmds(
                values.get("queue_base_model", ""), values.get("queue_adapter", ""), attack, baseline
            )
            asyncio.create_task(self._run_then_refresh(cmds))
        elif bid == "btn-queue-start":
            asyncio.create_task(self._run_then_refresh([self._base_cmd() + ["start", "--detach"]]))
        elif bid == "btn-queue-stop":
            asyncio.create_task(self._run_then_refresh([self._base_cmd() + ["stop"]]))
        elif bid == "btn-queue-retry":
            tid = self._selected_task_id()
            if tid is None:
                self.query_one("#cmd-output", RichLog).write("[yellow]Select a task first.[/]")
                return
            asyncio.create_task(self._run_then_refresh([self._base_cmd() + ["retry", str(tid)]]))
        elif bid == "btn-queue-compare":
            asyncio.create_task(self._run_then_refresh([self._base_cmd() + ["compare"]]))
