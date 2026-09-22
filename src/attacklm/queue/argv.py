"""Argument serialization — convert task args dicts to CLI argv lists."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from attacklm.queue.db import Task, QueueDB
from attacklm.queue.registry import REGISTRY, TaskSpec, _SCRIPTS_DIR


# Mapping from Python arg keys to CLI flag names.
# Keys not in this mapping are passed through as-is with --key value.
_ARG_FLAG_MAP: dict[str, str] = {
    "single_model": "--single-model",
    "single_model_name": "--single-model-name",
    "include_orchestrator": "--include-orchestrator",
    "model_attacks": "--model-attacks",
    "include_tools": "--include-tools",
    "batch_size": "--batch-size",
    "max_new_tokens": "--max-new-tokens",
    "base_model": "--base-model",
    "in_distribution": "--in-distribution",
    "near_ood": "--near-ood",
    "ood": "--ood",
    "num_canaries": "--num-canaries",
    "canary_format": "--canary-format",
    "inject_split": "--inject-split",
    "output_dir": "--output-dir",
}

# Boolean flags that should be passed as --flag (no value) when True.
_BOOLEAN_FLAGS: set[str] = {
    "single_model",
    "include_orchestrator",
    "model_attacks",
    "include_tools",
}


def is_adapter_dir(path: str | Path) -> bool:
    """True if *path* is a PEFT adapter directory (has adapter_config.json)."""
    return (Path(path) / "adapter_config.json").is_file()


def read_adapter_base(adapter_dir: str | Path) -> str | None:
    """Return base_model_name_or_path from an adapter's adapter_config.json."""
    cfg = Path(adapter_dir) / "adapter_config.json"
    try:
        data = json.loads(cfg.read_text())
    except (OSError, ValueError):
        return None
    base = data.get("base_model_name_or_path")
    return base or None


def _args_to_argv(args: dict[str, Any], schema: dict) -> list[str]:
    """Convert a task args dict to a CLI argv list.

    Only well-known keys from the schema are serialized. Everything
    else in extra_argv is appended verbatim.
    """
    argv: list[str] = []
    extra_argv = args.pop("extra_argv", []) or []

    for key, value in args.items():
        if value is None:
            continue
        flag = _ARG_FLAG_MAP.get(key, f"--{key.replace('_', '-')}")
        if key in _BOOLEAN_FLAGS:
            if value:
                argv.append(flag)
        else:
            argv.extend([flag, str(value)])

    argv.extend(extra_argv)
    return argv


def _find_adapter_in_deps(depends_on: list[int], db: QueueDB) -> str | None:
    """Find the most recently completed adapter path from dependencies."""
    adapter_path = None
    for dep_id in depends_on:
        dep = db.get_task(dep_id)
        if dep is None:
            continue
        if dep.status != "completed":
            continue
        result_str = dep.result
        if not result_str:
            # Also check artifact_path directly.
            if dep.artifact_path:
                adapter_path = dep.artifact_path
            continue
        try:
            result = (
                json.loads(result_str) if isinstance(result_str, str) else result_str
            )
        except (json.JSONDecodeError, TypeError):
            if dep.artifact_path:
                adapter_path = dep.artifact_path
            continue
        if isinstance(result, dict) and result.get("kind") == "adapter":
            adapter_path = result.get("adapter_path")
    return adapter_path


def _find_base_model_in_deps(depends_on: list[int], db: QueueDB) -> str | None:
    """Find the base_model from a completed training dependency."""
    for dep_id in depends_on:
        dep = db.get_task(dep_id)
        if dep is None or dep.status != "completed":
            continue
        result_str = dep.result
        if not result_str:
            continue
        try:
            result = (
                json.loads(result_str) if isinstance(result_str, str) else result_str
            )
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(result, dict) and "base_model" in result:
            return result["base_model"]
    return None


def _resolve_argv(task: Task, spec: TaskSpec, db: QueueDB) -> list[str] | None:
    """Build the full argv for executing a task, resolving deps at runtime.

    Returns None if required dependencies can't be resolved.
    """
    import sys

    args = task.args_dict.copy()

    # Start with the Python interpreter and script path.
    argv = [sys.executable, str(spec.script_path)]

    # Resolve adapter and base_model from dependencies if needed.
    if spec.consumes_artifact == "adapter":
        depends_on = task.depends_on_list
        # S2: a --base-model that is really an adapter dir (adapter_config.json)
        # is treated as the adapter; the true base comes from its config.
        if args.get("base_model") and not args.get("adapter") and is_adapter_dir(args["base_model"]):
            args["adapter"] = args["base_model"]
            args["base_model"] = None
        if not args.get("adapter"):
            adapter_path = _find_adapter_in_deps(depends_on, db)
            if adapter_path is None:
                return None
            args["adapter"] = adapter_path
        if not args.get("base_model"):
            base_model = _find_base_model_in_deps(depends_on, db)
            if not base_model:
                # R6: train task may have stored '' — fall back to the adapter's config.
                base_model = read_adapter_base(args["adapter"])
            if not base_model:
                return None
            args["base_model"] = base_model

    # Set default output path if not specified.
    if "output" not in args or args.get("output") is None:
        if spec.produces_artifact == "report":
            from attacklm.queue.db import DEFAULT_QUEUE_DIR

            args["output"] = str(
                DEFAULT_QUEUE_DIR
                / "artifacts"
                / str(task.id)
                / f"{spec.type}_report.json"
            )
        elif spec.produces_artifact == "data":
            args["output_dir"] = args.get(
                "output_dir",
                str(Path("data") / "bench"),
            )

    # Set default questions/holdout paths if not specified.
    if task.type == "audit_prompt_injection" and "questions" not in args:
        args["questions"] = "data/bench/prompt_injection_holdout.jsonl"
    elif task.type == "audit_system_prompt" and "questions" not in args:
        args["questions"] = "data/bench/system_prompt_holdout.jsonl"

    # Set default calibration holdout paths if not specified.
    if task.type == "audit_calibration":
        if "in_distribution" not in args:
            args["in_distribution"] = "data/bench/calibration_in.jsonl"
        if "near_ood" not in args:
            args["near_ood"] = "data/bench/calibration_near.jsonl"
        if "ood" not in args:
            args["ood"] = "data/bench/calibration_ood.jsonl"

    # R4: persist the resolved args (output, adapter, base_model, defaults) so
    # _collect_artifact and `queue status` see exactly what the script got.
    db.update_task(task.id, args=json.dumps(args))
    argv.extend(_args_to_argv(dict(args), spec.arg_schema))

    return argv
