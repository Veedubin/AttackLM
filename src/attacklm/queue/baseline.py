"""Base-model baselines — a gauntlet with no adapter, tagged gauntlet="baseline".

`compare` reads these as side A. `chain`/`gauntlet` queue one automatically
unless --no-baseline; `queue baseline <base>` queues one explicitly.
"""

from __future__ import annotations

from typing import Any, Callable

from attacklm.queue.db import QueueDB
from attacklm.queue.gauntlet import expand_gauntlet
from attacklm.queue.registry import REGISTRY

# What train_all.py falls back to when --base-model is not given (scripts/train_all.py ~L1278).
DEFAULT_BASE_MODEL = "huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated"

_LIVE = ("pending", "ready", "running", "completed")


def baseline_task_defs(base_model: str, preset: str) -> list[dict[str, Any]]:
    """Task defs for a baseline gauntlet: every audit in *preset* on *base_model*, no adapter."""
    defs = expand_gauntlet(preset, extra_args={"base_model": base_model})
    for d in defs:
        d["gauntlet"] = "baseline"
        d["depends_on"] = []
        if d["type"] != "gen_calibration_holdouts":
            d["label"] = f"Baseline: {preset} on {base_model}"
    return defs


def _preset_attacks(preset: str) -> list[str]:
    return [
        d["type"]
        for d in expand_gauntlet(preset)
        if d["type"] != "gen_calibration_holdouts" and REGISTRY[d["type"]].implemented
    ]


def baseline_exists(db: QueueDB, base_model: str, preset: str) -> list[int]:
    """Ids of live baseline tasks covering every shipped attack in *preset*, else []."""
    ids: list[int] = []
    for attack in _preset_attacks(preset):
        hit = None
        for t in db.list_tasks(type=attack, limit=10000):
            if (
                t.gauntlet == "baseline"
                and t.status in _LIVE
                and not t.args_dict.get("adapter")
                and t.args_dict.get("base_model") == base_model
            ):
                hit = t.id  # newest wins (id ascending)
        if hit is None:
            return []
        ids.append(hit)
    return ids


def ensure_baseline(
    db: QueueDB,
    base_model: str,
    preset: str,
    insert: Callable[[QueueDB, list[dict[str, Any]], list[int]], list[int]],
) -> list[int]:
    """Queue a baseline for (base_model, preset) unless one is live. Returns inserted ids."""
    if baseline_exists(db, base_model, preset):
        return []
    return insert(db, baseline_task_defs(base_model, preset), [])
