"""Forward-only schema migrations for the AttackLM queue database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

# Each migration is (version, list_of_sql_statements).
# Migrations are applied in order, forward-only (never roll back).
# v1 creates all tables from scratch.
MIGRATIONS: list[tuple[int, list[str]]] = [
    (
        1,
        [
            # -- schema_version: tracks which migrations have been applied ------
            """CREATE TABLE IF NOT EXISTS schema_version (
                version     INTEGER NOT NULL,
                applied_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            )""",
            # -- tasks: the core queue table ------------------------------------
            """CREATE TABLE IF NOT EXISTS tasks (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                type            TEXT    NOT NULL,
                label           TEXT    NOT NULL,
                status          TEXT    NOT NULL DEFAULT 'pending',
                args            TEXT    NOT NULL,
                depends_on      TEXT    NOT NULL DEFAULT '[]',
                artifact_path   TEXT,
                artifact_kind   TEXT,
                result          TEXT,
                error           TEXT,
                created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
                started_at      TEXT,
                finished_at     TEXT,
                pid             INTEGER,
                log_path        TEXT,
                timeout_seconds INTEGER,
                gauntlet        TEXT
            )""",
            "CREATE INDEX IF NOT EXISTS idx_tasks_status    ON tasks(status)",
            "CREATE INDEX IF NOT EXISTS idx_tasks_depends   ON tasks(depends_on)",
            "CREATE INDEX IF NOT EXISTS idx_tasks_created   ON tasks(created_at)",
            # -- task_events: append-only audit log ------------------------------
            """CREATE TABLE IF NOT EXISTS task_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id     INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                event       TEXT    NOT NULL,
                detail      TEXT,
                at          TEXT    NOT NULL DEFAULT (datetime('now'))
            )""",
            "CREATE INDEX IF NOT EXISTS idx_events_task ON task_events(task_id, at)",
            # -- runner_state: single-row IPC channel ---------------------------
            """CREATE TABLE IF NOT EXISTS runner_state (
                key         TEXT PRIMARY KEY,
                value       TEXT NOT NULL
            )""",
        ],
    ),
]


def current_schema_version(conn: sqlite3.Connection) -> int:
    """Return the highest version in schema_version, or 0 if the table doesn't exist yet."""
    try:
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        return int(row[0]) if row[0] is not None else 0
    except sqlite3.OperationalError:
        return 0


def apply_migrations(db_path: Path) -> int:
    """Apply all pending migrations to the database at *db_path*.

    Returns the final schema version.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        # Enable WAL mode and set busy timeout BEFORE any writes.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")

        current = current_schema_version(conn)
        for version, stmts in MIGRATIONS:
            if version <= current:
                continue
            with conn:
                for stmt in stmts:
                    conn.execute(stmt)
                conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (version,),
                )
            current = version
        return current
    finally:
        conn.close()
