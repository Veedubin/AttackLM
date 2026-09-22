"""Runner — single-worker task execution loop for AttackLM queue.

Uses subprocess.Popen with start_new_session=True for crash isolation.
SIGINT-safe: catches SIGINT and exits after the current task finishes.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from attacklm.queue.argv import _resolve_argv, read_adapter_base, resolve_steps
from attacklm.queue.db import DEFAULT_QUEUE_DIR, QueueDB, Task
from attacklm.queue.registry import REGISTRY, TaskSpec

logger = logging.getLogger("attacklm.queue.runner")

# Per-task log directory.
LOG_DIR = DEFAULT_QUEUE_DIR / "logs"
ARTIFACT_DIR = DEFAULT_QUEUE_DIR / "artifacts"


def _ensure_dirs() -> None:
    """Create log and artifact directories if they don't exist."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _child_env() -> dict[str, str]:
    """Build the environment for child processes."""
    env = os.environ.copy()
    env["ATTACKLM_QUEUE_TASK_ID"] = ""  # Set per-task below
    return env


def _collect_artifact(
    spec: TaskSpec, task: Task, log_path: Path
) -> dict[str, Any] | None:
    """Parse the task's log file to extract artifact info.

    For train tasks, grep for 'Final adapter: <path>'.
    For other tasks, check if the output file exists.
    """
    if spec.produces_artifact == "adapter":
        # Grep the log for the adapter path.
        adapter_path = _grep_adapter_path(log_path)
        if adapter_path is None:
            return None
        # Try to extract base_model from args.
        base_model = task.args_dict.get("base_model") or read_adapter_base(adapter_path) or ""
        return {
            "kind": "adapter",
            "adapter_path": adapter_path,
            "base_model": base_model,
        }

    if spec.produces_artifact == "report":
        # Check if the output file was created.
        output = task.args_dict.get("output", "")
        if output and Path(output).exists():
            return {
                "kind": "report",
                "report_path": output,
            }
        return None

    if spec.produces_artifact == "data":
        # gen_calibration_holdouts — check if the output files exist.
        output_dir = task.args_dict.get("output_dir", "data/bench")
        bench_dir = Path(output_dir)
        expected = [
            bench_dir / "calibration_in.jsonl",
            bench_dir / "calibration_near.jsonl",
            bench_dir / "calibration_ood.jsonl",
        ]
        if all(p.exists() for p in expected):
            return {
                "kind": "data",
                "output_dir": str(bench_dir),
                "files": [str(p) for p in expected],
            }
        return None

    return None


_ADAPTER_LINE = re.compile(r"^\s*(?:Final adapter|Adapter):\s+(\S.*?)\s*$", re.MULTILINE)


def _grep_adapter_path(log_path: Path) -> str | None:
    """Return the last 'Adapter: <path>' / 'Final adapter: <path>' line in a log.

    train_all.py prints '  Adapter: …' for single-model runs and
    '  Final adapter: …' for curriculum runs — both indented.
    """
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return None
    matches = _ADAPTER_LINE.findall(text)
    return matches[-1] if matches else None


def _run_subprocess(
    argv: list[str],
    log_path: Path,
    timeout_s: int | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Run a subprocess with stdout/stderr captured to log_path.

    Uses start_new_session=True for clean process group management.
    Returns the process exit code.
    """
    _ensure_dirs()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "ab", buffering=0) as logf:
        # Write a header.
        header = (
            f"\n# ===== task started at {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n"
        )
        logf.write(header.encode())

        proc = subprocess.Popen(
            argv,
            stdout=logf,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=env or os.environ.copy(),
            cwd=cwd,
        )
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            # Kill the process group.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            raise

    return rc


def _run_steps(
    steps: list[list[str]],
    log_path: Path,
    timeout_s: int | None,
    cwd: str | None,
    env: dict[str, str] | None,
) -> int:
    """Run pipeline steps sequentially; stop and return the first non-zero rc."""
    deadline = time.monotonic() + timeout_s if timeout_s else None
    for i, argv in enumerate(steps, 1):
        remaining = None if deadline is None else max(1, int(deadline - time.monotonic()))
        with open(log_path, "ab", buffering=0) as logf:
            logf.write(f"\n# ----- step {i}/{len(steps)}: {Path(argv[1]).name} -----\n".encode())
        rc = _run_subprocess(argv, log_path, timeout_s=remaining, cwd=cwd, env=env)
        if rc != 0:
            return rc
    return 0


class _Tail(threading.Thread):
    """Stream a growing log file to stdout until stopped (for `start --follow`)."""

    def __init__(self, path: Path) -> None:
        super().__init__(daemon=True)
        self._path = path
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=2)

    def run(self) -> None:
        pos = 0
        while True:
            try:
                with open(self._path, "rb") as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
                if chunk:
                    sys.stdout.write(chunk.decode(errors="replace"))
                    sys.stdout.flush()
            except OSError:
                pass
            if self._stop.is_set():
                break
            time.sleep(0.5)


def _execute_task(db: QueueDB, task: Task, follow: bool = False) -> None:
    """Execute a single task via subprocess."""
    spec = REGISTRY.get(task.type)
    if spec is None:
        db.mark_task(task.id, status="failed", error=f"Unknown task type: {task.type}")
        return

    # Check if task type is implemented.
    if not spec.implemented:
        db.mark_task(
            task.id,
            status="pending_unimplemented",
            error=f"{task.type} not implemented — see docs/MODEL_ATTACKS_SURVEY.md",
        )
        logger.info(f"Task #{task.id} skipped: {task.type} not implemented")
        return

    # Resolve argv/steps (including adapter inheritance from deps). Pipeline
    # tasks (e.g. audit_canary_pipeline) resolve to multiple steps; everything
    # else stays a single-step subprocess run through the module-level
    # `_resolve_argv` so existing tests that monkeypatch it keep working.
    if spec.runner_mode == "pipeline":
        steps = resolve_steps(task, spec, db)
    else:
        argv = _resolve_argv(task, spec, db)
        steps = None if argv is None else [argv]
    if steps is None:
        db.mark_task(
            task.id,
            status="failed",
            error="could not resolve --adapter/--base-model from dependencies",
        )
        return
    task = db.get_task(task.id)  # args were persisted by _resolve_argv/resolve_steps

    # Set up log path and artifact dir.
    _ensure_dirs()
    log_path = LOG_DIR / f"{task.id}.log"
    artifact_dir = ARTIFACT_DIR / str(task.id)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Mark task as running.
    db.mark_task(
        task.id,
        status="running",
        pid=os.getpid(),
        log_path=str(log_path),
    )
    db.add_event(
        task.id, "started", {"pid": os.getpid(), "argv": steps[-1], "steps": len(steps)}
    )

    # Set ATTACKLM_QUEUE_TASK_ID in the child env.
    env = os.environ.copy()
    env["ATTACKLM_QUEUE_TASK_ID"] = str(task.id)

    # Determine timeout.
    timeout = task.timeout_seconds or spec.default_timeout_s

    # Update the runner's current_task.
    db.set_runner_state("current_task", str(task.id))

    # Run the subprocess(es), optionally tailing the log to stdout.
    tail = _Tail(log_path) if follow else None
    if tail:
        tail.start()
    try:
        rc = _run_steps(
            steps,
            log_path,
            timeout_s=timeout,
            cwd=str(Path.cwd()),
            env=env,
        )
    except subprocess.TimeoutExpired:
        db.mark_task(task.id, status="failed", error="timeout")
        db.add_event(task.id, "failed", {"reason": "timeout"})
        return
    except KeyboardInterrupt:
        # Subprocess was killed externally — mark as interrupted.
        db.mark_task(task.id, status="interrupted", error="killed")
        db.add_event(task.id, "interrupted", {})
        raise
    except Exception as exc:
        db.mark_task(task.id, status="failed", error=str(exc))
        db.add_event(task.id, "failed", {"exception": str(exc)})
        return
    finally:
        if tail:
            tail.stop()

    if rc != 0:
        db.mark_task(task.id, status="failed", error=f"rc={rc}")
        db.add_event(task.id, "failed", {"rc": rc})
        return

    # Collect artifact.
    artifact = _collect_artifact(spec, task, log_path)
    if spec.produces_artifact and artifact is None:
        # Artifact expected but not found.
        db.mark_task(
            task.id,
            status="failed",
            error=f"expected {spec.produces_artifact} artifact not found",
        )
        db.add_event(task.id, "failed", {"reason": "artifact_missing"})
        return

    # Mark as completed.
    artifact_path = (
        artifact.get("adapter_path")
        or artifact.get("report_path")
        or artifact.get("output_dir")
        if artifact
        else None
    )
    artifact_kind = spec.produces_artifact
    result_json = json.dumps(artifact) if artifact else None

    db.mark_task(
        task.id,
        status="completed",
        artifact_path=artifact_path,
        artifact_kind=artifact_kind,
        result=result_json,
    )
    db.add_event(task.id, "completed", {"artifact": artifact_path})

    logger.info(f"Task #{task.id} completed: {spec.label}")


def _recover_interrupted(db: QueueDB) -> list[int]:
    """Find tasks marked as 'running' with stale PIDs and mark them 'interrupted'.

    Called at the start of `queue start` to recover from crashes.
    """
    running_tasks = db.list_tasks(status="running")
    recovered = []
    for task in running_tasks:
        pid = task.pid
        if pid is None:
            # No pid recorded — assume interrupted.
            db.mark_task(
                task.id, status="interrupted", error="runner crashed mid-task (no pid)"
            )
            db.add_event(task.id, "interrupted", {"reason": "no_pid"})
            recovered.append(task.id)
            continue
        try:
            os.kill(pid, 0)  # Check if process is alive.
        except ProcessLookupError:
            # PID is dead — task was interrupted.
            db.mark_task(task.id, status="interrupted", error="runner crashed mid-task")
            db.add_event(task.id, "interrupted", {"reason": "dead_pid", "pid": pid})
            recovered.append(task.id)
        except PermissionError:
            # PID exists but we can't signal it — assume it's running (different user).
            pass
    return recovered


def _register_runner(db: QueueDB, pid: int | None = None) -> None:
    """Register the runner as alive."""
    if pid is None:
        pid = os.getpid()
    db.set_runner_state("pid", str(pid))
    db.set_runner_state("heartbeat", time.strftime("%Y-%m-%dT%H:%M:%S"))


def _deregister_runner(db: QueueDB) -> None:
    """Clear the runner's PID and current_task."""
    db.set_runner_state("pid", "")
    db.set_runner_state("current_task", "")
    db.set_runner_state("stop_signal", "")


def _heartbeat(db: QueueDB) -> None:
    """Update the runner's heartbeat timestamp."""
    db.set_runner_state("heartbeat", time.strftime("%Y-%m-%dT%H:%M:%S"))


class _StopRequested(Exception):
    """Raised when a stop signal is detected."""

    pass


def run_loop(
    db_path: Path | str | None = None,
    poll_interval: float = 5.0,
    follow: bool = False,
    exit_when_idle: bool = False,
) -> None:
    """Main runner loop. Polls for eligible tasks and executes them.

    Args:
        db_path: Path to the queue database.
        poll_interval: Seconds between polls.
        follow: If True, stream the current task's log to stdout.
        exit_when_idle: If True, return as soon as no task is eligible
            (used by tests and by `start --exit-when-idle`).
    """
    db = QueueDB(db_path)
    stop_requested = False

    # Recover any interrupted tasks from a previous run.
    recovered = _recover_interrupted(db)
    if recovered:
        logger.info(f"Recovered interrupted tasks: {recovered}")

    # Re-check blocked status.
    blocked = db.recompute_blocked()
    if blocked:
        logger.info(f"Marked blocked tasks: {blocked}")

    # Check for stop signal from a previous run.
    stop_signal = db.get_runner_state("stop_signal")
    if stop_signal == "1":
        db.set_runner_state("stop_signal", "")

    # Register this runner.
    _register_runner(db)

    def _handle_sigint(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        logger.info("SIGINT received — will stop after current task")

    _prev_sigint_handler = signal.signal(signal.SIGINT, _handle_sigint)

    try:
        while not stop_requested:
            # Check for stop signal from `queue stop`.
            if db.get_runner_state("stop_signal") == "1":
                logger.info("Stop signal received — exiting after current task")
                break

            _heartbeat(db)

            # Pick next eligible task.
            task = db.pick_next_ready_task()
            if task is None:
                # Recheck blocked status.
                db.recompute_blocked()
                if exit_when_idle:
                    break
                time.sleep(poll_interval)
                continue

            logger.info(f"Executing task #{task.id}: {task.type} ({task.label})")
            _execute_task(db, task, follow=follow)

            # Recheck blocked after task completion.
            db.recompute_blocked()

    finally:
        signal.signal(signal.SIGINT, _prev_sigint_handler)
        _deregister_runner(db)
        logger.info("Runner stopped")
