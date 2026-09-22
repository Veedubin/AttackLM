"""Gauntlet presets — expand 'core'/'full'/'quick' into task lists."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from attacklm.queue.registry import REGISTRY, resolve_attack

# Built-in gauntlet presets.
GAUNTLET_PRESETS: dict[str, list[dict[str, Any]]] = {
    "core": [
        {"type": "audit_prompt_injection"},
        {"type": "audit_system_prompt"},
        {"type": "audit_canary_pipeline"},
        {"type": "audit_calibration"},
    ],
    "full": [
        {"type": "audit_prompt_injection"},
        {"type": "audit_system_prompt"},
        {"type": "audit_canary_pipeline"},
        {"type": "audit_gcg"},
        {"type": "audit_backdoor"},
        {"type": "audit_repeated_sampling"},
        {"type": "audit_calibration"},
    ],
    "quick": [
        {"type": "audit_prompt_injection"},
        {"type": "audit_system_prompt"},
    ],
    "memorization": [
        {"type": "audit_canary_pipeline"},
        {"type": "audit_repeated_sampling"},
    ],
}


def _calibration_holdouts_missing() -> bool:
    """Check if any of the 3 calibration holdout files are missing."""
    bench_dir = Path("data/bench")
    required = [
        bench_dir / "calibration_in.jsonl",
        bench_dir / "calibration_near.jsonl",
        bench_dir / "calibration_ood.jsonl",
    ]
    return any(not p.exists() for p in required)


def expand_gauntlet(
    preset_name: str,
    after_task_id: int | None = None,
    recipe_path: str | None = None,
    extra_args: dict[str, Any] | None = None,
    after_task_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Expand a gauntlet preset into a list of task specs with dependencies.

    Args:
        preset_name: One of 'core', 'full', 'quick', 'memorization', or a custom name.
        after_task_id: A single task id to depend on (kept for compatibility).
        recipe_path: Optional YAML recipe file path for custom gauntlets.
        extra_args: Optional dict of extra args to merge into each task.
        after_task_ids: every audit task depends on all of these. Takes
            precedence over after_task_id when given.

    Returns:
        List of dicts suitable for db.add_task(), each with:
          - type, label, args, depends_on, timeout_seconds, gauntlet
    """
    if recipe_path:
        tasks = _load_recipe(recipe_path)
    elif preset_name in GAUNTLET_PRESETS:
        tasks = GAUNTLET_PRESETS[preset_name]
    else:
        raise ValueError(
            f"Unknown gauntlet preset: {preset_name!r}. "
            f"Available: {', '.join(GAUNTLET_PRESETS.keys())}"
        )

    if after_task_ids is None:
        after_task_ids = [after_task_id] if after_task_id is not None else []

    needs_calibration_holdout = False
    result: list[dict[str, Any]] = []

    for task_def in tasks:
        task_type = task_def["type"]
        spec = REGISTRY[task_type]

        # Build args dict from the task definition + extra_args.
        args: dict[str, Any] = dict(task_def.get("args", {}))
        if extra_args:
            args.update(extra_args)

        # If this is a calibration audit and holdouts are missing,
        # we'll auto-insert a gen_calibration_holdouts task.
        if task_type == "audit_calibration" and _calibration_holdouts_missing():
            needs_calibration_holdout = True

        # Set up depends_on.
        depends_on: list[int] = list(after_task_ids)

        result.append(
            {
                "type": task_type,
                "label": spec.label,
                "args": args,
                "depends_on": depends_on,
                "timeout_seconds": spec.default_timeout_s,
                "gauntlet": preset_name,
            }
        )

    # Auto-insert gen_calibration_holdouts if needed.
    if needs_calibration_holdout:
        holdout_spec = REGISTRY["gen_calibration_holdouts"]
        holdout_task = {
            "type": "gen_calibration_holdouts",
            "label": holdout_spec.label,
            "args": {},
            "depends_on": [],  # No deps — can run immediately
            "timeout_seconds": holdout_spec.default_timeout_s,
            "gauntlet": preset_name,
        }
        result.insert(0, holdout_task)

        # Add holdout task as a dependency of the calibration audit task.
        # The holdout task's id will be assigned when we insert it.
        # We'll fix up the depends_on after insertion.
        for t in result:
            if t["type"] == "audit_calibration":
                # Mark that this needs the holdout task as a dep.
                # The CLI will resolve this after inserting the holdout task.
                t["_needs_holdout_dep"] = True

    return result


def _load_recipe(recipe_path: str) -> list[dict[str, Any]]:
    """Load a gauntlet recipe from a YAML file."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        # Fallback: try to parse simple YAML manually.
        return _load_recipe_simple(recipe_path)

    with open(recipe_path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "tasks" not in data:
        raise ValueError(f"Recipe {recipe_path!r} must have a 'tasks' key")

    tasks = []
    for entry in data["tasks"]:
        if not isinstance(entry, dict) or "type" not in entry:
            raise ValueError(f"Each task must have a 'type' key: {entry!r}")
        task_type = entry["type"]
        if task_type not in REGISTRY:
            raise ValueError(f"Unknown task type in recipe: {task_type!r}")
        tasks.append({"type": task_type, "args": entry.get("args", {})})

    return tasks


def _load_recipe_simple(recipe_path: str) -> list[dict[str, Any]]:
    """Simple YAML parser for recipes when pyyaml is not installed."""
    import re

    with open(recipe_path) as f:
        content = f.read()

    # Very basic parsing — just enough for the minimal recipe format.
    tasks = []
    current_type = None
    current_args: dict[str, Any] = {}

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Match: - type: audit_prompt_injection
        m = re.match(r"-\s*type:\s*(\S+)", line)
        if m:
            if current_type:
                tasks.append({"type": current_type, "args": current_args})
            current_type = m.group(1)
            current_args = {}
            continue
        # Match: key: value
        m = re.match(r"(\w+):\s*(.+)", line)
        if m and current_type:
            key, value = m.group(1), m.group(2)
            if key == "type":
                current_type = value
            else:
                # Try to parse value as int, float, or bool.
                if value.lower() in ("true", "false"):
                    current_args[key] = value.lower() == "true"
                else:
                    try:
                        current_args[key] = int(value)
                    except ValueError:
                        try:
                            current_args[key] = float(value)
                        except ValueError:
                            current_args[key] = value

    if current_type:
        tasks.append({"type": current_type, "args": current_args})

    return tasks
