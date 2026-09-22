"""QueueDB — SQLite-backed persistent task queue for AttackLM.

Uses WAL mode, 30s busy timeout, and forward-only migrations.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from attacklm.queue.migrations import apply_migrations

# Default queue database location.
DEFAULT_QUEUE_DIR = Path("evals/queue")
DEFAULT_DB_NAME = "queue.db"


class Task:
    """Row proxy for a task in the queue."""

    __slots__ = (
        "id",
        "type",
        "label",
        "status",
        "args",
        "depends_on",
        "artifact_path",
        "artifact_kind",
        "result",
        "error",
        "created_at",
        "started_at",
        "finished_at",
        "pid",
        "log_path",
        "timeout_seconds",
        "gauntlet",
    )

    def __init__(self, row: dict[str, Any]) -> None:
        for slot in self.__slots__:
            setattr(self, slot, row.get(slot))

    @property
    def depends_on_list(self) -> list[int]:
        """Parse the JSON depends_on field."""
        raw = self.depends_on or "[]"
        if isinstance(raw, list):
            return raw
        return json.loads(raw)

    @property
    def args_dict(self) -> dict[str, Any]:
        """Parse the JSON args field."""
        raw = self.args or "{}"
        if isinstance(raw, dict):
            return raw
        return json.loads(raw)

    def __repr__(self) -> str:
        return f"Task(id={self.id}, type={self.type!r}, status={self.status!r})"


class QueueDB:
    """Persistent task queue backed by SQLite."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            db_path = DEFAULT_QUEUE_DIR / DEFAULT_DB_NAME
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Apply migrations (creates the DB if it doesn't exist).
        apply_migrations(self.db_path)

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with WAL mode and busy timeout."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """Context manager that yields a connection and commits on exit."""
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Task CRUD
    # ------------------------------------------------------------------

    def add_task(
        self,
        type: str,
        label: str,
        args: dict[str, Any],
        depends_on: list[int] | None = None,
        timeout_seconds: int | None = None,
        gauntlet: str | None = None,
    ) -> int:
        """Insert a new task. Returns the task id."""
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO tasks (type, label, status, args, depends_on,
                                      timeout_seconds, gauntlet)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    type,
                    label,
                    "pending",
                    json.dumps(args),
                    json.dumps(depends_on or []),
                    timeout_seconds,
                    gauntlet,
                ),
            )
            task_id = cur.lastrowid
            conn.execute(
                "INSERT INTO task_events (task_id, event, detail) VALUES (?, ?, ?)",
                (task_id, "created", json.dumps({"type": type, "label": label})),
            )
            return task_id

    def get_task(self, task_id: int) -> Task | None:
        """Get a single task by id."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                return None
            return Task(dict(row))

    def list_tasks(
        self,
        status: str | None = None,
        type: str | None = None,
        limit: int = 100,
    ) -> list[Task]:
        """List tasks, optionally filtered by status and/or type."""
        with self._conn() as conn:
            clauses: list[str] = []
            params: list[Any] = []
            if status:
                # Support comma-separated statuses
                statuses = [s.strip() for s in status.split(",")]
                if len(statuses) == 1:
                    clauses.append("status = ?")
                    params.append(statuses[0])
                else:
                    placeholders = ",".join("?" * len(statuses))
                    clauses.append(f"status IN ({placeholders})")
                    params.extend(statuses)
            if type:
                clauses.append("type = ?")
                params.append(type)
            where = " AND ".join(clauses) if clauses else "1=1"
            rows = conn.execute(
                f"SELECT * FROM tasks WHERE {where} ORDER BY id LIMIT ?",
                (*params, limit),
            ).fetchall()
            return [Task(dict(r)) for r in rows]

    def update_task(self, task_id: int, **fields: Any) -> None:
        """Update arbitrary fields on a task."""
        if not fields:
            return
        set_clauses = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [task_id]
        with self._conn() as conn:
            conn.execute(f"UPDATE tasks SET {set_clauses} WHERE id = ?", values)

    def mark_task(
        self,
        task_id: int,
        status: str,
        error: str | None = None,
        artifact_path: str | None = None,
        artifact_kind: str | None = None,
        result: str | None = None,
        pid: int | None = None,
        log_path: str | None = None,
    ) -> None:
        """Transition a task to a new status with optional metadata."""
        updates: dict[str, Any] = {"status": status}
        if error is not None:
            updates["error"] = error
        if artifact_path is not None:
            updates["artifact_path"] = artifact_path
        if artifact_kind is not None:
            updates["artifact_kind"] = artifact_kind
        if result is not None:
            updates["result"] = result
        if pid is not None:
            updates["pid"] = pid
        if log_path is not None:
            updates["log_path"] = log_path
        if status == "running":
            updates["started_at"] = datetime.now(timezone.utc).isoformat()
        if status in ("completed", "failed", "interrupted", "cancelled"):
            updates["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.update_task(task_id, **updates)
        # Record the event.
        detail = json.dumps({"status": status, "error": error})
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO task_events (task_id, event, detail) VALUES (?, ?, ?)",
                (task_id, status, detail),
            )

    def add_event(
        self, task_id: int, event: str, detail: dict[str, Any] | None = None
    ) -> None:
        """Append an event to the task_events audit log."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO task_events (task_id, event, detail) VALUES (?, ?, ?)",
                (task_id, event, json.dumps(detail) if detail else None),
            )

    def get_events(self, task_id: int) -> list[dict[str, Any]]:
        """Get all events for a task, ordered by time."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM task_events WHERE task_id = ? ORDER BY at",
                (task_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Runner state
    # ------------------------------------------------------------------

    def set_runner_state(self, key: str, value: str) -> None:
        """Upsert a runner_state key/value pair."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runner_state (key, value) VALUES (?, ?)",
                (key, value),
            )

    def get_runner_state(self, key: str) -> str | None:
        """Read a runner_state value."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM runner_state WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else None

    def get_all_runner_state(self) -> dict[str, str]:
        """Get all runner_state key/value pairs."""
        with self._conn() as conn:
            rows = conn.execute("SELECT key, value FROM runner_state").fetchall()
            return {r["key"]: r["value"] for r in rows}

    # ------------------------------------------------------------------
    # Queue operations
    # ------------------------------------------------------------------

    def pick_next_ready_task(self) -> Task | None:
        """Atomically pick the next task whose deps are all completed.

        Sets status to 'running' and claims it. Returns None if no
        eligible task exists.
        """
        with self._conn() as conn:
            # Find tasks that are pending/ready and whose deps are all completed.
            rows = conn.execute(
                "SELECT * FROM tasks WHERE status IN ('pending', 'ready') ORDER BY id"
            ).fetchall()
            for row in rows:
                task = Task(dict(row))
                deps = task.depends_on_list
                if not deps:
                    # No deps — eligible immediately.
                    conn.execute(
                        "UPDATE tasks SET status = 'running' WHERE id = ? AND status IN ('pending', 'ready')",
                        (task.id,),
                    )
                    # Verify we claimed it (atomic check).
                    check = conn.execute(
                        "SELECT status FROM tasks WHERE id = ?", (task.id,)
                    ).fetchone()
                    if check and check["status"] == "running":
                        conn.commit()
                        return self.get_task(task.id)
                    continue
                # Check all deps are completed.
                deps_completed = all(
                    conn.execute(
                        "SELECT 1 FROM tasks WHERE id = ? AND status = 'completed'",
                        (dep_id,),
                    ).fetchone()
                    for dep_id in deps
                )
                if deps_completed:
                    conn.execute(
                        "UPDATE tasks SET status = 'running' WHERE id = ? AND status IN ('pending', 'ready')",
                        (task.id,),
                    )
                    check = conn.execute(
                        "SELECT status FROM tasks WHERE id = ?", (task.id,)
                    ).fetchone()
                    if check and check["status"] == "running":
                        conn.commit()
                        return self.get_task(task.id)
            return None

    def recompute_blocked(self) -> list[int]:
        """Scan all pending/ready tasks and mark those with failed/cancelled/blocked deps as blocked.

        Returns the list of newly-blocked task ids.
        """
        blocked_ids: list[int] = []
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE status IN ('pending', 'ready', 'blocked')"
            ).fetchall()
            for row in rows:
                task = Task(dict(row))
                deps = task.depends_on_list
                if not deps:
                    continue
                terminal_statuses = {"failed", "cancelled", "blocked"}
                any_dep_terminal = any(
                    conn.execute(
                        "SELECT status FROM tasks WHERE id = ?",
                        (dep_id,),
                    ).fetchone()["status"]
                    in terminal_statuses
                    for dep_id in deps
                    if conn.execute(
                        "SELECT 1 FROM tasks WHERE id = ?", (dep_id,)
                    ).fetchone()
                )
                if any_dep_terminal and task.status != "blocked":
                    conn.execute(
                        "UPDATE tasks SET status = 'blocked' WHERE id = ?",
                        (task.id,),
                    )
                    blocked_ids.append(task.id)
        return blocked_ids

    def remove_task(self, task_id: int, force: bool = False) -> bool:
        """Remove a task. Only allowed if status is pending/ready/blocked/interrupted,
        unless force=True. Returns True if removed."""
        task = self.get_task(task_id)
        if task is None:
            return False
        removable = {"pending", "ready", "blocked", "interrupted"}
        if not force and task.status not in removable:
            return False
        with self._conn() as conn:
            conn.execute("DELETE FROM task_events WHERE task_id = ?", (task_id,))
            conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        return True

    def clean_tasks(
        self,
        status: str = "completed",
        older_than_days: int | None = None,
        yes: bool = False,
    ) -> int:
        """Delete tasks matching status/age criteria. Returns count deleted."""
        with self._conn() as conn:
            clauses = []
            params: list[Any] = []
            if status != "all":
                clauses.append("status = ?")
                params.append(status)
            if older_than_days is not None:
                clauses.append("finished_at < datetime('now', ?)")
                params.append(f"-{older_than_days} days")
            where = " AND ".join(clauses) if clauses else "1=1"
            count = conn.execute(
                f"SELECT COUNT(*) FROM tasks WHERE {where}", params
            ).fetchone()[0]
            task_ids = [
                r[0]
                for r in conn.execute(
                    f"SELECT id FROM tasks WHERE {where}", params
                ).fetchall()
            ]
            for tid in task_ids:
                conn.execute("DELETE FROM task_events WHERE task_id = ?", (tid,))
            conn.execute(f"DELETE FROM tasks WHERE {where}", params)
            return count

    def reset(self) -> None:
        """Drop and recreate all tables. DANGEROUS — only for `queue reset --yes`."""
        # Remove the database file entirely and re-create via migrations.
        self.db_path.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            p = Path(str(self.db_path) + suffix)
            p.unlink(missing_ok=True)
        apply_migrations(self.db_path)
