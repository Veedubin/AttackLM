"""Queue subcommand CLI — dispatch for `attacklm queue ...`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from attacklm.queue.db import QueueDB, Task, DEFAULT_QUEUE_DIR, DEFAULT_DB_NAME
from attacklm.queue.registry import REGISTRY, resolve_attack, ATTACK_ALIASES
from attacklm.queue.gauntlet import expand_gauntlet, GAUNTLET_PRESETS
from attacklm.queue.display import list_tasks, status_summary, task_detail
from attacklm.queue.runner import run_loop


def _get_db(args: argparse.Namespace) -> QueueDB:
    """Get a QueueDB instance from the parsed args."""
    db_path = getattr(args, "db_path", None)
    if db_path:
        return QueueDB(db_path)
    return QueueDB()


def _cmd_add_train(args: argparse.Namespace) -> int:
    """Handle `attacklm queue add train`."""
    db = _get_db(args)

    # Build args dict from CLI flags.
    task_args: dict[str, Any] = {}
    if args.base_model:
        task_args["base_model"] = args.base_model
    if args.single_model:
        task_args["single_model"] = True
    if args.single_model_name:
        task_args["single_model_name"] = args.single_model_name
    if args.include_orchestrator:
        task_args["include_orchestrator"] = True
    if args.model_attacks:
        task_args["model_attacks"] = True
    if args.include_tools:
        task_args["include_tools"] = True
    if args.epochs:
        task_args["epochs"] = args.epochs
    if args.batch_size:
        task_args["batch_size"] = args.batch_size
    if args.extra_argv:
        task_args["extra_argv"] = list(args.extra_argv)

    label = args.label or "Train (single-model)"
    timeout = args.timeout or None

    task_id = db.add_task(
        type="train",
        label=label,
        args=task_args,
        timeout_seconds=timeout,
    )
    print(f"Task #{task_id} created: {label!r} [pending]")
    return 0


def _cmd_add_audit(args: argparse.Namespace) -> int:
    """Handle `attacklm queue add audit`."""
    db = _get_db(args)

    # Resolve attack id(s).
    try:
        attack_keys = resolve_attack(
            args.attack, include_unshipped=args.include_unshipped
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Resolve depends_on.
    depends_on: list[int] = []
    if args.depends_on:
        for dep in args.depends_on.split(","):
            dep = dep.strip()
            if dep == "latest":
                # Find the most recently added train task.
                train_tasks = db.list_tasks(type="train", limit=1)
                # Get the last train task (highest id).
                all_train = db.list_tasks(type="train", limit=10000)
                if not all_train:
                    print(
                        "Error: no train tasks found; cannot use 'latest'",
                        file=sys.stderr,
                    )
                    return 1
                depends_on.append(all_train[-1].id)
            else:
                try:
                    depends_on.append(int(dep))
                except ValueError:
                    print(f"Error: invalid depends_on id: {dep!r}", file=sys.stderr)
                    return 1

    # Build and add each audit task.
    task_ids = []
    for key in attack_keys:
        spec = REGISTRY[key]
        task_args: dict[str, Any] = {}
        if args.adapter:
            task_args["adapter"] = args.adapter
        if args.base_model:
            task_args["base_model"] = args.base_model

        label = spec.label
        timeout = args.timeout or spec.default_timeout_s

        tid = db.add_task(
            type=key,
            label=label,
            args=task_args,
            depends_on=depends_on if depends_on else None,
            timeout_seconds=timeout,
        )
        task_ids.append(tid)
        print(f"Task #{tid} created: {label!r} [pending, depends_on={depends_on}]")

    return 0


def _cmd_add_holdouts(args: argparse.Namespace) -> int:
    """Handle `attacklm queue add holdouts`."""
    db = _get_db(args)

    if args.kind != "calibration":
        print(f"Error: unknown holdout kind: {args.kind!r}", file=sys.stderr)
        return 1

    spec = REGISTRY["gen_calibration_holdouts"]
    task_args: dict[str, Any] = {}
    if args.output_dir:
        task_args["output_dir"] = args.output_dir

    task_id = db.add_task(
        type="gen_calibration_holdouts",
        label=spec.label,
        args=task_args,
        timeout_seconds=spec.default_timeout_s,
    )
    print(f"Task #{task_id} created: {spec.label!r} [pending]")
    return 0


def _cmd_chain(args: argparse.Namespace) -> int:
    """Handle `attacklm queue chain train ... --then audit/gauntlet ...`."""
    db = _get_db(args)

    # Step 1: Add the train task.
    train_args: dict[str, Any] = {}
    if args.base_model:
        train_args["base_model"] = args.base_model
    if args.single_model:
        train_args["single_model"] = True
    if args.single_model_name:
        train_args["single_model_name"] = args.single_model_name
    if args.include_orchestrator:
        train_args["include_orchestrator"] = True
    if args.model_attacks:
        train_args["model_attacks"] = True
    if args.include_tools:
        train_args["include_tools"] = True
    if args.epochs:
        train_args["epochs"] = args.epochs
    if args.batch_size:
        train_args["batch_size"] = args.batch_size
    if args.train_extra:
        train_args["extra_argv"] = list(args.train_extra)

    label = args.label or "Train (single-model)"
    timeout = args.train_timeout or None

    train_id = db.add_task(
        type="train",
        label=label,
        args=train_args,
        timeout_seconds=timeout,
    )
    print(f"Task #{train_id} created: {label!r} [pending]")

    # Step 2: Add the gauntlet/audit tasks depending on the train task.
    then = args.then
    if then == "gauntlet":
        preset = args.gauntlet_preset or "core"
        try:
            tasks = expand_gauntlet(preset, after_task_id=train_id)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    elif then == "audit":
        try:
            attack_keys = resolve_attack(
                args.attack or "all",
                include_unshipped=args.include_unshipped,
            )
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        tasks = []
        for key in attack_keys:
            spec = REGISTRY[key]
            tasks.append(
                {
                    "type": key,
                    "label": spec.label,
                    "args": {},
                    "depends_on": [train_id],
                    "timeout_seconds": spec.default_timeout_s,
                    "gauntlet": None,
                }
            )
    else:
        print(
            f"Error: --then must be 'audit' or 'gauntlet', got {then!r}",
            file=sys.stderr,
        )
        return 1

    # Insert gauntlet tasks.
    holdout_id = None
    # First pass: insert any gen_calibration_holdouts tasks to get their IDs.
    for task_def in tasks:
        if task_def["type"] == "gen_calibration_holdouts":
            spec = REGISTRY["gen_calibration_holdouts"]
            holdout_id = db.add_task(
                type="gen_calibration_holdouts",
                label=spec.label,
                args=task_def.get("args", {}),
                timeout_seconds=spec.default_timeout_s,
            )
            print(
                f"Task #{holdout_id} created: {spec.label!r} [pending] (auto-inserted)"
            )
            break

    # Second pass: insert all tasks, using holdout_id for calibration deps.
    for task_def in tasks:
        # Respect explicit depends_on (even if empty []); only default to
        # [train_id] when the key is absent (e.g. audit tasks from --then audit).
        depends_on = task_def.get("depends_on", [train_id])
        # If this is the holdout task, we already inserted it — skip.
        if task_def["type"] == "gen_calibration_holdouts" and holdout_id is not None:
            # Already inserted — skip adding it again.
            # But update its depends_on if needed.
            if depends_on:
                db.update_task(holdout_id, depends_on=json.dumps(depends_on))
            continue
        # Auto-insert calibration holdout dep if needed.
        if task_def.pop("_needs_holdout_dep", False):
            if holdout_id is None:
                holdout_spec = REGISTRY["gen_calibration_holdouts"]
                holdout_id = db.add_task(
                    type="gen_calibration_holdouts",
                    label=holdout_spec.label,
                    args={},
                    timeout_seconds=holdout_spec.default_timeout_s,
                )
                print(
                    f"Task #{holdout_id} created: {holdout_spec.label!r} [pending] (auto-inserted)"
                )
            depends_on.append(holdout_id)

        tid = db.add_task(
            type=task_def["type"],
            label=task_def["label"],
            args=task_def.get("args", {}),
            depends_on=depends_on,
            timeout_seconds=task_def.get("timeout_seconds"),
            gauntlet=task_def.get("gauntlet"),
        )
        print(
            f"Task #{tid} created: {task_def['label']!r} [pending, depends_on={depends_on}]"
        )

    print(f"\nChain ready. Run `attacklm queue start` to begin.")
    return 0


def _cmd_gauntlet(args: argparse.Namespace) -> int:
    """Handle `attacklm queue gauntlet <preset>`."""
    db = _get_db(args)
    preset = args.preset

    # Resolve --after (depends_on).
    depends_on: list[int] = []
    if args.after:
        for dep in args.after.split(","):
            dep = dep.strip()
            if dep == "latest":
                all_train = db.list_tasks(type="train", limit=10000)
                if not all_train:
                    print(
                        "Error: no train tasks found; cannot use 'latest'",
                        file=sys.stderr,
                    )
                    return 1
                depends_on.append(all_train[-1].id)
            else:
                try:
                    depends_on.append(int(dep))
                except ValueError:
                    print(f"Error: invalid --after id: {dep!r}", file=sys.stderr)
                    return 1

    try:
        tasks = expand_gauntlet(
            preset,
            after_task_id=depends_on[0] if depends_on else None,
            recipe_path=args.recipe,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Insert tasks.
    # First pass: insert any gen_calibration_holdouts tasks to get their IDs.
    holdout_id = None
    for task_def in tasks:
        if task_def["type"] == "gen_calibration_holdouts":
            spec = REGISTRY["gen_calibration_holdouts"]
            holdout_id = db.add_task(
                type="gen_calibration_holdouts",
                label=spec.label,
                args=task_def.get("args", {}),
                depends_on=task_def.get("depends_on", []),
                timeout_seconds=spec.default_timeout_s,
                gauntlet=task_def.get("gauntlet"),
            )
            print(
                f"Task #{holdout_id} created: {spec.label!r} [pending, depends_on={task_def.get('depends_on', [])}]"
            )
            break

    # Second pass: insert all other tasks, using holdout_id for calibration deps.
    for task_def in tasks:
        if task_def["type"] == "gen_calibration_holdouts" and holdout_id is not None:
            continue  # already inserted in first pass
        # Respect explicit depends_on (even if empty []); only default to
        # the user's --after when the key is absent.
        deps = task_def.get("depends_on", list(depends_on))
        # Auto-insert calibration holdout dep if needed (holdout not in list).
        if task_def.pop("_needs_holdout_dep", False):
            if holdout_id is None:
                holdout_spec = REGISTRY["gen_calibration_holdouts"]
                holdout_id = db.add_task(
                    type="gen_calibration_holdouts",
                    label=holdout_spec.label,
                    args={},
                    timeout_seconds=holdout_spec.default_timeout_s,
                )
                print(
                    f"Task #{holdout_id} created: {holdout_spec.label!r} [pending] (auto-inserted)"
                )
            deps.append(holdout_id)

        tid = db.add_task(
            type=task_def["type"],
            label=task_def["label"],
            args=task_def.get("args", {}),
            depends_on=deps,
            timeout_seconds=task_def.get("timeout_seconds"),
            gauntlet=task_def.get("gauntlet"),
        )
        print(
            f"Task #{tid} created: {task_def['label']!r} [pending, depends_on={deps}]"
        )

    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    """Handle `attacklm queue list`."""
    db = _get_db(args)
    tasks = db.list_tasks(status=args.status, type=args.type, limit=args.limit or 100)

    # Get runner state.
    runner_info = db.get_all_runner_state()

    output = list_tasks(tasks, runner_info)
    print(output)
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    """Handle `attacklm queue status`."""
    db = _get_db(args)

    if args.task_id:
        task = db.get_task(args.task_id)
        if task is None:
            print(f"Task #{args.task_id} not found.", file=sys.stderr)
            return 1
        print(task_detail(task, db))
    else:
        print(status_summary(db))

    return 0


def _cmd_start(args: argparse.Namespace) -> int:
    """Handle `attacklm queue start`."""
    db = _get_db(args)

    # Check for an already-running runner.
    pid_str = db.get_runner_state("pid")
    if pid_str:
        try:
            existing_pid = int(pid_str)
            # Check if it's still alive.
            try:
                os.kill(existing_pid, 0)
                # Process is alive.
                if not args.force:
                    print(
                        f"Error: runner already running (PID {existing_pid}). "
                        f"Use 'attacklm queue stop' first, or '--force' to override.",
                        file=sys.stderr,
                    )
                    return 1
                print(
                    f"Warning: overriding existing runner (PID {existing_pid})",
                    file=sys.stderr,
                )
            except ProcessLookupError:
                # Process is dead — safe to start.
                pass
        except (ValueError, TypeError):
            pass

    run_loop(
        db_path=args.db_path if hasattr(args, "db_path") and args.db_path else None,
        poll_interval=args.poll_interval or 5.0,
        follow=args.follow,
        detach=args.detach,
    )
    return 0


def _cmd_stop(args: argparse.Namespace) -> int:
    """Handle `attacklm queue stop`."""
    db = _get_db(args)

    # Set the stop signal.
    db.set_runner_state("stop_signal", "1")

    # Report current task.
    current_task = db.get_runner_state("current_task")
    pid = db.get_runner_state("pid")

    if pid:
        print(f"Stop signal sent to runner (PID {pid}).")
        if current_task:
            print(f"Current task #{current_task} will finish before stopping.")
    else:
        print("Stop signal set. No runner appears to be running.")

    return 0


def _cmd_remove(args: argparse.Namespace) -> int:
    """Handle `attacklm queue remove <task_id>`."""
    db = _get_db(args)
    task_id = args.task_id

    # Check for dependents.
    all_tasks = db.list_tasks(limit=10000)
    dependents = [t for t in all_tasks if task_id in t.depends_on_list]

    if dependents and not args.force:
        dep_ids = [t.id for t in dependents]
        print(
            f"Error: task #{task_id} has dependents {dep_ids}. "
            f"Remove them first or use --force.",
            file=sys.stderr,
        )
        return 1

    if args.force and dependents:
        # Cascade: mark dependents as blocked.
        dep_ids = [t.id for t in dependents]
        for t in dependents:
            db.mark_task(t.id, status="blocked", error=f"dependency #{task_id} removed")
        print(f"Cascaded: tasks {dep_ids} marked as blocked (dependency removed).")

    removed = db.remove_task(task_id, force=args.force)
    if removed:
        print(f"Task #{task_id} removed.")
        return 0
    else:
        print(
            f"Could not remove task #{task_id} (status not removable). Use --force.",
            file=sys.stderr,
        )
        return 1


def _cmd_retry(args: argparse.Namespace) -> int:
    """Handle `attacklm queue retry <task_id>`."""
    db = _get_db(args)
    task = db.get_task(args.task_id)
    if task is None:
        print(f"Task #{args.task_id} not found.", file=sys.stderr)
        return 1

    retryable = {"failed", "interrupted", "blocked", "pending_unimplemented"}
    if task.status not in retryable:
        print(
            f"Cannot retry task #{args.task_id} with status {task.status!r}.",
            file=sys.stderr,
        )
        return 1

    # Reset to pending, clear error.
    db.update_task(
        args.task_id,
        status="pending",
        error=None,
        started_at=None,
        finished_at=None,
        pid=None,
    )
    db.add_event(args.task_id, "retry", {"from_status": task.status})
    print(f"Task #{args.task_id} reset to pending (was {task.status}).")
    return 0


def _cmd_clean(args: argparse.Namespace) -> int:
    """Handle `attacklm queue clean`."""
    db = _get_db(args)

    # Check if runner is active.
    pid = db.get_runner_state("pid")
    if pid:
        try:
            existing_pid = int(pid)
            try:
                os.kill(existing_pid, 0)
                print(
                    "Error: runner is running. Stop it first with 'attacklm queue stop'.",
                    file=sys.stderr,
                )
                return 1
            except ProcessLookupError:
                pass
        except (ValueError, TypeError):
            pass

    count = db.clean_tasks(
        status=args.status,
        older_than_days=args.older_than,
        yes=args.yes,
    )
    print(f"Removed {count} task(s).")
    return 0


def _cmd_reset(args: argparse.Namespace) -> int:
    """Handle `attacklm queue reset` — DANGEROUS."""
    if not args.yes:
        print(
            "Error: --yes is required for reset. This will delete ALL queue data.",
            file=sys.stderr,
        )
        return 1

    db = _get_db(args)
    db.reset()
    print("Queue database reset.")
    return 0


# ------------------------------------------------------------------
# Build the subcommand parser
# ------------------------------------------------------------------


def _cmd_queue_default(args: argparse.Namespace) -> int:
    """Default handler for `attacklm queue` with no subcommand — show status."""
    db = _get_db(args)
    print(status_summary(db))
    return 0


def build_queue_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the `queue` subcommand to the main CLI parser."""
    queue_p = subparsers.add_parser(
        "queue",
        help="Task queue for training and audit orchestration",
        description=(
            "AttackLM task queue — persist, chain, and monitor training "
            "jobs and audit gauntlets.\n\n"
            "Usage:\n"
            "  attacklm queue add-train --single-model\n"
            "  attacklm queue chain --single-model --then gauntlet core\n"
            "  attacklm queue start --follow\n"
            "  attacklm queue list\n"
        ),
    )
    queue_p.set_defaults(func=_cmd_queue_default)

    # Shared --db-path flag.
    queue_p.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to queue database (default: evals/queue/queue.db)",
    )

    queue_sub = queue_p.add_subparsers(dest="queue_command", help="Queue subcommands")

    # ---- add train ----
    add_train_p = queue_sub.add_parser("add-train", help="Add a training task")
    add_train_p.add_argument(
        "--label", type=str, default=None, help="Human-friendly label"
    )
    add_train_p.add_argument(
        "--base-model", type=str, default=None, help="HuggingFace model id"
    )
    add_train_p.add_argument(
        "--single-model",
        action="store_true",
        default=False,
        help="Single-model training",
    )
    add_train_p.add_argument(
        "--single-model-name", type=str, default=None, help="Name for single model"
    )
    add_train_p.add_argument(
        "--include-orchestrator",
        action="store_true",
        default=False,
        help="Include orchestrator data",
    )
    add_train_p.add_argument(
        "--model-attacks",
        action="store_true",
        default=False,
        help="Include model attacks data",
    )
    add_train_p.add_argument(
        "--include-tools", action="store_true", default=False, help="Include tools data"
    )
    add_train_p.add_argument(
        "--epochs", type=int, default=None, help="Number of training epochs"
    )
    add_train_p.add_argument("--batch-size", type=int, default=None, help="Batch size")
    add_train_p.add_argument(
        "--timeout", type=int, default=None, help="Timeout in seconds (default: none)"
    )
    add_train_p.add_argument(
        "extra_argv",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to train_all.py",
    )
    add_train_p.set_defaults(func=_cmd_add_train)

    # ---- add audit ----
    add_audit_p = queue_sub.add_parser("add-audit", help="Add an audit task")
    add_audit_p.add_argument(
        "--attack",
        type=str,
        required=True,
        help="Attack id: 1-7, name, or 'all'",
    )
    add_audit_p.add_argument(
        "--include-unshipped",
        action="store_true",
        default=False,
        help="Include unimplemented attacks (4/5/6) as pending_unimplemented",
    )
    add_audit_p.add_argument(
        "--depends-on", type=str, default=None, help="Task id(s) or 'latest'"
    )
    add_audit_p.add_argument(
        "--adapter", type=str, default=None, help="Explicit adapter path"
    )
    add_audit_p.add_argument(
        "--base-model", type=str, default=None, help="Explicit base model id"
    )
    add_audit_p.add_argument(
        "--label", type=str, default=None, help="Human-friendly label"
    )
    add_audit_p.add_argument(
        "--timeout", type=int, default=None, help="Timeout in seconds"
    )
    add_audit_p.set_defaults(func=_cmd_add_audit)

    # ---- add holdouts ----
    add_holdouts_p = queue_sub.add_parser(
        "add-holdouts", help="Add a holdout generation task"
    )
    add_holdouts_p.add_argument(
        "--kind",
        type=str,
        default="calibration",
        choices=["calibration"],
        help="Kind of holdout to generate (default: calibration)",
    )
    add_holdouts_p.add_argument(
        "--output-dir", type=str, default=None, help="Output directory"
    )
    add_holdouts_p.set_defaults(func=_cmd_add_holdouts)

    # ---- chain ----
    chain_p = queue_sub.add_parser(
        "chain",
        help="Chain a train task followed by a gauntlet or audit",
        description=(
            "The headline flow: train then audit.\n\n"
            "  attacklm queue chain train --single-model --then gauntlet core\n"
            "  attacklm queue chain train --single-model --then audit --attack all"
        ),
    )
    chain_p.add_argument(
        "--label", type=str, default=None, help="Label for the train task"
    )
    chain_p.add_argument(
        "--base-model", type=str, default=None, help="HuggingFace model id for training"
    )
    chain_p.add_argument(
        "--single-model",
        action="store_true",
        default=False,
        help="Single-model training",
    )
    chain_p.add_argument(
        "--single-model-name", type=str, default=None, help="Name for single model"
    )
    chain_p.add_argument(
        "--include-orchestrator",
        action="store_true",
        default=False,
        help="Include orchestrator data",
    )
    chain_p.add_argument(
        "--model-attacks",
        action="store_true",
        default=False,
        help="Include model attacks data",
    )
    chain_p.add_argument(
        "--include-tools", action="store_true", default=False, help="Include tools data"
    )
    chain_p.add_argument(
        "--epochs", type=int, default=None, help="Number of training epochs"
    )
    chain_p.add_argument("--batch-size", type=int, default=None, help="Batch size")
    chain_p.add_argument(
        "--train-timeout", type=int, default=None, help="Train timeout in seconds"
    )
    chain_p.add_argument(
        "--then",
        type=str,
        required=True,
        choices=["audit", "gauntlet"],
        help="What to chain after training: 'audit' or 'gauntlet'",
    )
    chain_p.add_argument(
        "--attack",
        type=str,
        default=None,
        help="Attack id for --then audit (default: all shipped)",
    )
    chain_p.add_argument(
        "--gauntlet-preset",
        type=str,
        default="core",
        help="Gauntlet preset for --then gauntlet (default: core)",
    )
    chain_p.add_argument(
        "--include-unshipped",
        action="store_true",
        default=False,
        help="Include unimplemented attacks",
    )
    chain_p.add_argument(
        "train_extra", nargs=argparse.REMAINDER, help="Extra args for train_all.py"
    )
    chain_p.set_defaults(func=_cmd_chain)

    # ---- gauntlet ----
    gauntlet_p = queue_sub.add_parser("gauntlet", help="Add a gauntlet of audit tasks")
    gauntlet_p.add_argument(
        "preset",
        type=str,
        help="Gauntlet preset: core, full, quick, memorization",
    )
    gauntlet_p.add_argument(
        "--after",
        type=str,
        default=None,
        help="Task id(s) or 'latest' to depend on",
    )
    gauntlet_p.add_argument(
        "--recipe",
        type=str,
        default=None,
        help="Path to a YAML recipe file",
    )
    gauntlet_p.set_defaults(func=_cmd_gauntlet)

    # ---- list ----
    list_p = queue_sub.add_parser("list", help="List tasks in the queue")
    list_p.add_argument(
        "--status", type=str, default=None, help="Filter by status (comma-separated)"
    )
    list_p.add_argument("--type", type=str, default=None, help="Filter by task type")
    list_p.add_argument("--limit", type=int, default=None, help="Max tasks to show")
    list_p.set_defaults(func=_cmd_list)

    # ---- status ----
    status_p = queue_sub.add_parser("status", help="Show queue or task status")
    status_p.add_argument(
        "task_id", type=int, nargs="?", default=None, help="Task id for detail view"
    )
    status_p.set_defaults(func=_cmd_status)

    # ---- start ----
    start_p = queue_sub.add_parser("start", help="Start the queue runner")
    start_p.add_argument(
        "--follow", action="store_true", default=False, help="Stream current task's log"
    )
    start_p.add_argument(
        "--detach", action="store_true", default=False, help="Run in background"
    )
    start_p.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between polls (default: 5)",
    )
    start_p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Override existing runner check",
    )
    start_p.set_defaults(func=_cmd_start)

    # ---- stop ----
    stop_p = queue_sub.add_parser(
        "stop", help="Stop the queue runner after current task"
    )
    stop_p.set_defaults(func=_cmd_stop)

    # ---- remove ----
    remove_p = queue_sub.add_parser("remove", help="Remove a task from the queue")
    remove_p.add_argument("task_id", type=int, help="Task id to remove")
    remove_p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Remove even if not removable status",
    )
    remove_p.set_defaults(func=_cmd_remove)

    # ---- retry ----
    retry_p = queue_sub.add_parser("retry", help="Retry a failed/interrupted task")
    retry_p.add_argument("task_id", type=int, help="Task id to retry")
    retry_p.set_defaults(func=_cmd_retry)

    # ---- clean ----
    clean_p = queue_sub.add_parser("clean", help="Clean completed/failed tasks")
    clean_p.add_argument(
        "--status",
        type=str,
        default="completed",
        help="Status to clean (default: completed). Use 'all' for everything.",
    )
    clean_p.add_argument(
        "--older-than",
        type=int,
        default=None,
        help="Only clean tasks older than N days",
    )
    clean_p.add_argument(
        "--yes", action="store_true", default=False, help="Confirm deletion"
    )
    clean_p.set_defaults(func=_cmd_clean)

    # ---- reset ----
    reset_p = queue_sub.add_parser(
        "reset", help="DANGER: Drop and recreate the queue database"
    )
    reset_p.add_argument(
        "--yes", action="store_true", default=False, help="Confirm reset"
    )
    reset_p.set_defaults(func=_cmd_reset)


# Need to import os for PID checks in _cmd_start and _cmd_clean.
import os
