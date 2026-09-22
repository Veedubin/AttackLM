"""Queue subcommand CLI — dispatch for `attacklm queue ...`."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from attacklm.queue.db import QueueDB, DEFAULT_QUEUE_DIR
from attacklm.queue.registry import REGISTRY, resolve_attack
from attacklm.queue.gauntlet import expand_gauntlet
from attacklm.queue.baseline import baseline_exists, baseline_task_defs, ensure_baseline
from attacklm.queue.compare import SHIPPED_ATTACKS, compare_runs, latest_subject, render_table, select_run, to_json
from attacklm.queue.history import append_jsonl, history_rows, render_history
from attacklm.queue.display import list_tasks, status_summary, task_detail
from attacklm.queue.runner import run_loop


def _get_db(args: argparse.Namespace) -> QueueDB:
    """Get a QueueDB instance from the parsed args."""
    db_path = getattr(args, "db_path", None)
    if db_path:
        return QueueDB(db_path)
    return QueueDB()


def _parse_dep_ids(db: QueueDB, raw: str | None, flag: str) -> list[int] | None:
    """Parse 'latest' / '1,2' into task ids. Prints the error and returns None on failure."""
    if not raw:
        return []
    ids: list[int] = []
    for dep in raw.split(","):
        dep = dep.strip()
        if dep == "latest":
            all_train = db.list_tasks(type="train", limit=10000)
            if not all_train:
                print("Error: no train tasks found; cannot use 'latest'", file=sys.stderr)
                return None
            ids.append(all_train[-1].id)
        else:
            try:
                ids.append(int(dep))
            except ValueError:
                print(f"Error: invalid {flag} id: {dep!r}", file=sys.stderr)
                return None
    return ids


def _insert_task_defs(db: QueueDB, tasks: list[dict[str, Any]], default_deps: list[int]) -> list[int]:
    """Insert expanded task defs in two passes (holdout generator first).

    Shared by `chain` and `gauntlet` — the duplicate-holdout bug of v0.18.0
    lived in two copies of this loop.
    """
    created: list[int] = []
    holdout_id: int | None = None
    holdout_spec = REGISTRY["gen_calibration_holdouts"]

    for task_def in tasks:
        if task_def["type"] == "gen_calibration_holdouts":
            holdout_id = db.add_task(
                type="gen_calibration_holdouts",
                label=holdout_spec.label,
                args=task_def.get("args", {}),
                depends_on=task_def.get("depends_on", []),
                timeout_seconds=holdout_spec.default_timeout_s,
                gauntlet=task_def.get("gauntlet"),
            )
            created.append(holdout_id)
            print(f"Task #{holdout_id} created: {holdout_spec.label!r} [pending] (auto-inserted)")
            break

    for task_def in tasks:
        if task_def["type"] == "gen_calibration_holdouts" and holdout_id is not None:
            continue
        # Respect explicit depends_on (even []); default only when the key is absent.
        deps = list(task_def.get("depends_on", default_deps))
        if task_def.pop("_needs_holdout_dep", False):
            if holdout_id is None:
                holdout_id = db.add_task(
                    type="gen_calibration_holdouts",
                    label=holdout_spec.label,
                    args={},
                    timeout_seconds=holdout_spec.default_timeout_s,
                )
                created.append(holdout_id)
                print(f"Task #{holdout_id} created: {holdout_spec.label!r} [pending] (auto-inserted)")
            deps.append(holdout_id)
        tid = db.add_task(
            type=task_def["type"],
            label=task_def["label"],
            args=task_def.get("args", {}),
            depends_on=deps,
            timeout_seconds=task_def.get("timeout_seconds"),
            gauntlet=task_def.get("gauntlet"),
        )
        created.append(tid)
        print(f"Task #{tid} created: {task_def['label']!r} [pending, depends_on={deps}]")
    return created


def _queue_baseline(db: QueueDB, base_model: str | None, preset: str, args: argparse.Namespace) -> None:
    """Auto-queue a baseline gauntlet unless --no-baseline (or the base is unknown)."""
    if getattr(args, "no_baseline", False):
        return
    if not base_model:
        print("Note: no baseline queued (base model unknown — pass --base-model, or run `attacklm queue baseline <base>`).")
        return
    ids = ensure_baseline(db, base_model, preset, _insert_task_defs)
    if ids:
        print(f"Baseline for {base_model} ({preset}) queued: tasks "
              f"{', '.join('#' + str(i) for i in ids)} "
              f"(~doubles this run's GPU time; --no-baseline to skip)")
    else:
        print(f"Baseline for {base_model} ({preset}) already queued/completed.")


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
    depends_on = _parse_dep_ids(db, args.depends_on, "--depends-on")
    if depends_on is None:
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

        if args.label:
            label = f"{args.label} — {spec.label}" if len(attack_keys) > 1 else args.label
        else:
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
        # I3: do NOT fall back to baseline.DEFAULT_BASE_MODEL here -- that
        # model no longer exists on HF, so falling back to it queued GPU
        # work guaranteed to fail (and ensure_baseline would re-queue it
        # forever, since "failed" isn't a _LIVE status). With no explicit
        # --base-model, take the "base model unknown" branch instead.
        _queue_baseline(db, args.base_model, preset, args)
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

    _insert_task_defs(db, tasks, [train_id])
    print("\nChain ready. Run `attacklm queue start` to begin.")
    return 0


def _cmd_gauntlet(args: argparse.Namespace) -> int:
    """Handle `attacklm queue gauntlet <preset>`."""
    db = _get_db(args)
    preset = args.preset

    # Resolve --after (depends_on).
    depends_on = _parse_dep_ids(db, args.after, "--after")
    if depends_on is None:
        return 1

    try:
        tasks = expand_gauntlet(
            preset,
            after_task_ids=depends_on,
            recipe_path=args.recipe,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    base = getattr(args, "base_model", None)
    if not base:
        for dep in depends_on:
            dep_task = db.get_task(dep)
            # I3: no DEFAULT_BASE_MODEL fallback -- an empty base_model on
            # the train dep means "base model unknown", not "assume the
            # (removed-from-HF) default".
            if dep_task is not None and dep_task.type == "train":
                base = dep_task.args_dict.get("base_model")
                break
    _queue_baseline(db, base, preset, args)

    _insert_task_defs(db, tasks, list(depends_on))
    return 0


def _cmd_baseline(args: argparse.Namespace) -> int:
    """Handle `attacklm queue baseline <base-model>`."""
    db = _get_db(args)
    preset = args.preset or "core"
    existing = baseline_exists(db, args.base_model, preset)
    if existing and not args.force:
        print(f"Baseline for {args.base_model} ({preset}) already exists: tasks "
              f"{', '.join('#' + str(i) for i in existing)}. Use --force to queue a fresh one.")
        return 0

    ids = _insert_task_defs(db, baseline_task_defs(args.base_model, preset), [])
    print(f"Baseline for {args.base_model} ({preset}) queued: tasks {', '.join('#' + str(i) for i in ids)}")
    return 0


def _preset_attacks_for_compare(preset: str) -> list[str]:
    from attacklm.queue.gauntlet import GAUNTLET_PRESETS

    if preset not in GAUNTLET_PRESETS:
        # Minor (final review): an unknown preset used to silently fall
        # back to every shipped attack -- error like expand_gauntlet does
        # instead of comparing against attacks the caller never asked for.
        raise ValueError(
            f"Unknown gauntlet preset: {preset!r}. Available: {', '.join(GAUNTLET_PRESETS)}"
        )
    keys = [d["type"] for d in GAUNTLET_PRESETS[preset] if d["type"] in SHIPPED_ATTACKS]
    return keys or SHIPPED_ATTACKS


def _cmd_compare(args: argparse.Namespace) -> int:
    """Handle `attacklm queue compare [A] [B]`."""
    db = _get_db(args)
    preset = args.preset or "core"
    try:
        attacks = _preset_attacks_for_compare(preset)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.a and args.b:
        run_a = select_run(db, subject=args.a, attacks=attacks)
        run_b = select_run(db, subject=args.b, attacks=attacks)
        if run_a is None or run_b is None:
            missing = args.a if run_a is None else args.b
            print(f"No completed gauntlet for {missing}", file=sys.stderr)
            return 1
    else:
        if args.a:
            subject = args.a
            run_b = select_run(db, subject=subject, attacks=attacks)
            if run_b is None:
                print(f"No completed gauntlet for {subject}", file=sys.stderr)
                return 1
            base = run_b.base_model
        else:
            # I2: pass the SAME preset-filtered attacks into latest_subject
            # that select_run below will use -- an unfiltered scan could
            # name a subject whose only completed work is outside this
            # preset, and select_run(attacks=attacks) would then find
            # nothing for it (previously an unguarded AttributeError on
            # run_b.base_model; --preset memorization was a real repro).
            latest = latest_subject(db, attacks=attacks)
            if latest is None:
                print("No completed gauntlet with an adapter yet.", file=sys.stderr)
                return 1
            subject, base = latest
            run_b = select_run(db, subject=subject, attacks=attacks)
            if run_b is None:
                print(f"No completed gauntlet for {subject}", file=sys.stderr)
                return 1
        if base is None:
            # I4: don't try (and fail) to look up a baseline for an
            # unknown base -- tell the caller to disambiguate explicitly.
            print(
                f"Could not determine the base model for {subject}; pass both sides "
                f"explicitly: attacklm queue compare <A> <B>",
                file=sys.stderr,
            )
            return 1
        run_a = select_run(db, base_model=base, attacks=attacks)
        if run_a is None:
            # I4: a baseline that's queued but not finished isn't the same
            # as "no baseline" -- "run `queue baseline`" is useless advice
            # when one is already in the queue.
            queued = baseline_exists(db, base, preset)
            if queued:
                print(
                    f"Baseline for {base} is queued but not finished yet: tasks "
                    f"{', '.join('#' + str(i) for i in queued)} — run `attacklm queue start`",
                    file=sys.stderr,
                )
            else:
                print(f"No baseline for {base}. Run: attacklm queue baseline {base}", file=sys.stderr)
            return 1

    rows, unpaired = compare_runs(run_a, run_b, attacks, resamples=args.resamples or 2000, seed=args.seed or 42)
    if args.json:
        print(json.dumps(to_json(run_a, run_b, rows, unpaired), indent=2))
    else:
        print(render_table(run_a, run_b, rows, unpaired))
    return 0


def _cmd_history(args: argparse.Namespace) -> int:
    """Handle `attacklm queue history`."""
    db = _get_db(args)
    # I5: --subject is the preferred flag (matches `compare`'s vocabulary);
    # --adapter is kept as an alias for compatibility.
    subject = getattr(args, "subject", None) or getattr(args, "adapter", None)
    rows = history_rows(db, subject=subject, limit=args.limit or 100)
    if args.jsonl:
        n = append_jsonl(rows, Path(args.jsonl))
        print(f"Appended {n} new row(s) to {args.jsonl}")
    print(render_history(rows))
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


def _detach_argv(args: argparse.Namespace) -> list[str]:
    """Build the argv for the detached background runner subprocess.

    Factored out from `_cmd_start` so it's unit-testable without actually
    spawning a subprocess.
    """
    cmd = [sys.executable, "-m", "attacklm", "queue"]
    if getattr(args, "db_path", None):
        cmd += ["--db-path", args.db_path]
    cmd += ["start", "--poll-interval", str(args.poll_interval or 5.0)]
    if getattr(args, "exit_when_idle", False):
        cmd.append("--exit-when-idle")
    if getattr(args, "force", False):
        # The child does its own "already running" check on startup — it
        # needs --force too, or it refuses to start over the very runner
        # this command was told to override.
        cmd.append("--force")
    return cmd


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

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.detach:
        if getattr(args, "follow", False):
            print(
                "Warning: --follow is ignored in detached mode (nothing to stream to).",
                file=sys.stderr,
            )
        DEFAULT_QUEUE_DIR.mkdir(parents=True, exist_ok=True)
        runner_log = DEFAULT_QUEUE_DIR / "runner.log"
        cmd = _detach_argv(args)
        with open(runner_log, "ab") as logf:
            proc = subprocess.Popen(
                cmd, stdout=logf, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
        print(f"Runner started in background (PID {proc.pid}). Log: {runner_log}")
        return 0

    run_loop(
        db_path=args.db_path if hasattr(args, "db_path") and args.db_path else None,
        poll_interval=args.poll_interval or 5.0,
        follow=args.follow,
        exit_when_idle=getattr(args, "exit_when_idle", False),
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

    task = db.get_task(task_id)
    if task is None:
        print(f"Task #{task_id} not found.", file=sys.stderr)
        return 1
    if task.status == "running":
        print(
            f"Error: task #{task_id} is running. Stop the runner first ('attacklm queue stop').",
            file=sys.stderr,
        )
        return 1

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
        # Cascade: only relabel dependents that are still pending/ready/
        # blocked — a completed/failed/running/etc. dependent already has
        # its own outcome and must not be overwritten.
        cascadable = {"pending", "ready", "blocked"}
        to_block = [t for t in dependents if t.status in cascadable]
        if to_block:
            dep_ids = [t.id for t in to_block]
            for t in to_block:
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

    # Reset to pending, clear error and any stale artifact from the failed run.
    db.update_task(
        args.task_id,
        status="pending",
        error=None,
        started_at=None,
        finished_at=None,
        pid=None,
        result=None,
        artifact_path=None,
        artifact_kind=None,
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
    if not args.yes:
        print(f"Would remove {count} task(s). Pass --yes to delete.")
        return 0
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
            "  attacklm queue baseline <base-model>\n"
            "  attacklm queue start --follow\n"
            "  attacklm queue list\n"
            "  attacklm queue compare [<subject>]\n"
            "  attacklm queue history\n\n"
            "`chain`/`gauntlet` auto-queue a base-model baseline (--no-baseline to skip); "
            "`compare`/`history` are separate reporting subcommands."
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
        "--no-baseline",
        action="store_true",
        default=False,
        help="Skip the automatic base-model baseline gauntlet",
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
    gauntlet_p.add_argument(
        "--no-baseline",
        action="store_true",
        default=False,
        help="Skip the automatic base-model baseline gauntlet",
    )
    gauntlet_p.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model for the automatic baseline (default: from the --after train task, else the abliterated Qwen 3B)",
    )
    gauntlet_p.set_defaults(func=_cmd_gauntlet)

    # ---- baseline ----
    baseline_p = queue_sub.add_parser(
        "baseline", help="Queue a base-model baseline gauntlet (no adapter)"
    )
    baseline_p.add_argument("base_model", type=str, help="Base model to baseline")
    baseline_p.add_argument(
        "--preset", type=str, default="core", help="Gauntlet preset (default: core)"
    )
    baseline_p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Queue a fresh baseline even if one already exists",
    )
    baseline_p.set_defaults(func=_cmd_baseline)

    # ---- compare ----
    compare_p = queue_sub.add_parser(
        "compare", help="Compare a subject's latest gauntlet against a baseline (or another subject)"
    )
    compare_p.add_argument(
        "a",
        type=str,
        nargs="?",
        default=None,
        help="Adapter path (or base-model path for a merged model) for side A "
             "(default: the base model's baseline). With only A given, A is "
             "compared as side B against its own baseline.",
    )
    compare_p.add_argument(
        "b",
        type=str,
        nargs="?",
        default=None,
        help="Adapter path (or base-model path for a merged model) for side B "
             "(default: the newest completed subject)",
    )
    compare_p.add_argument(
        "--preset", type=str, default="core", help="Which attacks to compare (default: core)"
    )
    compare_p.add_argument(
        "--json", action="store_true", default=False, help="Print JSON instead of a table"
    )
    compare_p.add_argument(
        "--resamples", type=int, default=2000, help="Bootstrap resamples (default: 2000)"
    )
    compare_p.add_argument(
        "--seed", type=int, default=42, help="Bootstrap RNG seed (default: 42)"
    )
    compare_p.set_defaults(func=_cmd_compare)

    # ---- history ----
    history_p = queue_sub.add_parser("history", help="List completed audits, newest first")
    history_p.add_argument(
        "--subject", type=str, default=None,
        help="Filter to one subject: an adapter path, or a merged model's base_model",
    )
    history_p.add_argument(
        "--adapter", type=str, default=None, help="Alias for --subject (kept for compatibility)"
    )
    history_p.add_argument(
        "--limit", type=int, default=100, help="Max rows to show (default: 100)"
    )
    history_p.add_argument(
        "--jsonl", type=str, default=None, help="Append new rows to this JSONL file"
    )
    history_p.set_defaults(func=_cmd_history)

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
        "--detach",
        action="store_true",
        default=False,
        help="Run the runner in the background (log: evals/queue/runner.log)",
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
    start_p.add_argument(
        "--exit-when-idle",
        action="store_true",
        default=False,
        help="Exit when no task is eligible instead of polling forever",
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
