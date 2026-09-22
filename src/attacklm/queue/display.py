"""Display module — rich-based or plain-text rendering for queue list/status."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from attacklm.queue.db import Task, QueueDB
from attacklm.queue.registry import REGISTRY

# Try importing rich; fall back to plain text.
try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


# Status display configuration.
STATUS_ICONS = {
    "pending": "○",
    "ready": "◎",
    "running": "●",
    "completed": "✓",
    "failed": "✗",
    "interrupted": "⚡",
    "cancelled": "✗",
    "blocked": "⊘",
    "pending_unimplemented": "◐",
}

STATUS_COLORS = {
    "pending": "dim",
    "ready": "cyan",
    "running": "blue",
    "completed": "green",
    "failed": "red",
    "interrupted": "yellow",
    "cancelled": "dim",
    "blocked": "magenta",
    "pending_unimplemented": "dim",
}


def _format_duration(task: Task) -> str:
    """Format a task's duration as a human-readable string."""
    if not task.started_at:
        return "—"

    started = task.started_at
    if isinstance(started, str):
        try:
            started = datetime.fromisoformat(started)
        except (ValueError, TypeError):
            return "—"

    finished = task.finished_at
    if finished:
        if isinstance(finished, str):
            try:
                finished = datetime.fromisoformat(finished)
            except (ValueError, TypeError):
                finished = None
        if finished:
            delta = finished - started
            return _format_timedelta(delta)

    # Still running — show elapsed.
    now = datetime.now(timezone.utc)
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    delta = now - started
    return _format_timedelta(delta) + " (running)"


def _format_timedelta(delta) -> str:
    """Format a timedelta as '1h12m' or '8m12s'."""
    total_seconds = int(delta.total_seconds())
    if total_seconds < 0:
        return "—"
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    if hours > 0:
        return f"{hours}h{minutes}m"
    if minutes > 0:
        return f"{minutes}m{seconds}s"
    return f"{seconds}s"


def _format_depends(depends_on: Any) -> str:
    """Format a depends_on field for display."""
    if isinstance(depends_on, str):
        import json

        try:
            deps = json.loads(depends_on)
        except (json.JSONDecodeError, TypeError):
            return str(depends_on)
    elif isinstance(depends_on, list):
        deps = depends_on
    else:
        return "—"

    if not deps:
        return "—"
    return "[" + ", ".join(str(d) for d in deps) + "]"


def _format_created(created_at: Any) -> str:
    """Format created_at timestamp for display."""
    if not created_at:
        return "—"
    if isinstance(created_at, str):
        try:
            dt = datetime.fromisoformat(created_at)
            return dt.strftime("%H:%M:%S")
        except (ValueError, TypeError):
            return str(created_at)[:8]
    return str(created_at)[:8]


def list_tasks(tasks: list[Task], runner_info: dict[str, Any] | None = None) -> str:
    """Render a task list. Returns formatted string."""
    if _HAS_RICH:
        return _list_rich(tasks, runner_info)
    return _list_plain(tasks, runner_info)


def _list_rich(tasks: list[Task], runner_info: dict[str, Any] | None = None) -> str:
    """Render a task list using rich tables."""
    from io import StringIO

    sio = StringIO()
    console = Console(file=sio, width=120, no_color=False, force_terminal=False)
    table = Table(title="AttackLM Queue", show_lines=True)

    table.add_column("ID", style="bold", width=4)
    table.add_column("Type", width=32)
    table.add_column("Status", width=18)
    table.add_column("Depends", width=12)
    table.add_column("Created", width=10)
    table.add_column("Dur.", width=12)
    table.add_column("Artifact", width=30)

    for task in tasks:
        spec = REGISTRY.get(task.type)
        type_label = spec.label if spec else task.type
        # Add ⚠ for unimplemented tasks.
        if spec and not spec.implemented:
            type_label += " ⚠"

        status = task.status
        icon = STATUS_ICONS.get(status, "?")
        color = STATUS_COLORS.get(status, "white")
        status_text = f"{icon} {status}"

        artifact = task.artifact_path or "—"
        if artifact != "—" and len(artifact) > 28:
            artifact = "..." + artifact[-25:]

        table.add_row(
            str(task.id),
            type_label,
            f"[{color}]{status_text}[/{color}]",
            _format_depends(task.depends_on),
            _format_created(task.created_at),
            _format_duration(task),
            artifact,
        )

    console.print(table)

    # Runner info footer.
    if runner_info:
        _print_runner_info_rich(console, runner_info)
    else:
        console.print("[dim]Runner: not running[/dim]")

    return sio.getvalue()


def _print_runner_info_rich(console: Console, info: dict[str, Any]) -> None:
    """Print runner state info."""
    pid = info.get("pid")
    heartbeat = info.get("heartbeat")
    current_task = info.get("current_task")

    parts = []
    if pid:
        parts.append(f"PID {pid}")
    if heartbeat:
        parts.append(f"heartbeat {heartbeat}")
    if current_task:
        parts.append(f"current task #{current_task}")

    if parts:
        console.print(f"[bold]Runner:[/bold] {', '.join(parts)}")
    else:
        console.print("[dim]Runner: not running[/dim]")


def _list_plain(tasks: list[Task], runner_info: dict[str, Any] | None = None) -> str:
    """Render a task list in plain text (no rich)."""
    lines: list[str] = []

    # Header.
    header = f"{'ID':>4}  {'Type':<32}  {'Status':<18}  {'Depends':<12}  {'Created':<10}  {'Dur.':<12}  {'Artifact':<30}"
    lines.append(header)
    lines.append("-" * len(header))

    for task in tasks:
        spec = REGISTRY.get(task.type)
        type_label = spec.label if spec else task.type
        if spec and not spec.implemented:
            type_label += " ⚠"

        icon = STATUS_ICONS.get(task.status, "?")
        status_str = f"{icon} {task.status}"

        artifact = task.artifact_path or "—"
        if artifact != "—" and len(artifact) > 28:
            artifact = "..." + artifact[-25:]

        lines.append(
            f"{task.id:>4}  {type_label:<32}  {status_str:<18}  "
            f"{_format_depends(task.depends_on):<12}  "
            f"{_format_created(task.created_at):<10}  "
            f"{_format_duration(task):<12}  "
            f"{artifact:<30}"
        )

    # Runner info footer.
    if runner_info:
        pid = runner_info.get("pid", "")
        hb = runner_info.get("heartbeat", "")
        ct = runner_info.get("current_task", "")
        parts = []
        if pid:
            parts.append(f"PID {pid}")
        if hb:
            parts.append(f"heartbeat {hb}")
        if ct:
            parts.append(f"current task #{ct}")
        if parts:
            lines.append(f"Runner: {', '.join(parts)}")
        else:
            lines.append("Runner: not running")
    else:
        lines.append("Runner: not running")

    return "\n".join(lines)


def status_summary(db: QueueDB) -> str:
    """Render overall queue status summary."""
    all_tasks = db.list_tasks(limit=10000)

    # Count by status.
    counts: dict[str, int] = {}
    for task in all_tasks:
        counts[task.status] = counts.get(task.status, 0) + 1

    # Get runner state.
    pid = db.get_runner_state("pid")
    heartbeat = db.get_runner_state("heartbeat")
    current_task = db.get_runner_state("current_task")

    # Find next task (read-only — never claim from a status command).
    next_task = db.find_next_ready_task()

    lines = [
        "AttackLM Queue — status",
        f"  Runner:       {_runner_status(pid, heartbeat, current_task)}",
    ]

    if current_task:
        task = db.get_task(int(current_task))
        if task:
            lines.append(
                f"  Current:      #{task.id} {task.label} — {_format_duration(task)}"
            )

    lines.extend(
        [
            f"  Pending:      {counts.get('pending', 0)}   Ready: {counts.get('ready', 0)}   Running: {counts.get('running', 0)}   Completed: {counts.get('completed', 0)}",
            f"  Failed: {counts.get('failed', 0)}     Interrupted: {counts.get('interrupted', 0)}   Blocked: {counts.get('blocked', 0)}   Unimplemented: {counts.get('pending_unimplemented', 0)}",
        ]
    )

    if next_task:
        lines.append(
            f"  Next up:      #{next_task.id} {next_task.label} [waits on {_format_depends(next_task.depends_on)}]"
        )

    lines.append(f"  DB:           {db.db_path}")

    return "\n".join(lines)


def _runner_status(
    pid: str | None, heartbeat: str | None, current_task: str | None
) -> str:
    """Format runner status string."""
    if not pid:
        return "not running"
    parts = [f"running (PID {pid})"]
    if heartbeat:
        parts.append(f"heartbeat {heartbeat}")
    return ", ".join(parts)


def task_detail(task: Task, db: QueueDB) -> str:
    """Render detailed status for a single task."""
    spec = REGISTRY.get(task.type)
    label = spec.label if spec else task.type
    if spec and not spec.implemented:
        label += " ⚠"

    lines = [
        f"Task #{task.id} — {label}",
        f"  Status:       {task.status}",
    ]

    if task.pid:
        lines.append(f"  PID:          {task.pid}")
    if task.started_at:
        lines.append(f"  Started:      {task.started_at}")
        lines.append(f"  Duration:     {_format_duration(task)}")
    if task.finished_at:
        lines.append(f"  Finished:     {task.finished_at}")
    if task.error:
        lines.append(f"  Error:        {task.error}")

    # Dependency info.
    deps = task.depends_on_list
    if deps:
        dep_strs = []
        for dep_id in deps:
            dep = db.get_task(dep_id)
            if dep:
                dep_spec = REGISTRY.get(dep.type)
                dep_label = dep_spec.label if dep_spec else dep.type
                dep_strs.append(f"#{dep_id} ({dep.status})")
            else:
                dep_strs.append(f"#{dep_id} (unknown)")
        lines.append(f"  Depends on:   {', '.join(dep_strs)}")

        # Show adapter inheritance.
        if spec and spec.consumes_artifact == "adapter":
            adapter_path = task.args_dict.get("adapter")
            if adapter_path:
                if adapter_path == "<inherited>":
                    lines.append(f"  Adapter:      (inherited from dep)")
                else:
                    lines.append(f"  Adapter:      {adapter_path}")
            else:
                lines.append(f"  Adapter:      (will be resolved from deps)")

    # Output/Artifact info.
    if task.artifact_path:
        lines.append(f"  Artifact:     {task.artifact_path}")
    if task.log_path:
        lines.append(f"  Log:          {task.log_path}  (tail -f to follow)")

    # Args.
    args = task.args_dict
    if args:
        lines.append("  Args:")
        for key, value in sorted(args.items()):
            lines.append(f"    --{key.replace('_', '-')}  {value}")

    # Events.
    events = db.get_events(task.id)
    if events:
        lines.append("  Events:")
        for event in events:
            at = event.get("at", "?")
            ev = event.get("event", "?")
            lines.append(f"    {at}  {ev}")

    return "\n".join(lines)
