"""`attacklm queue history` — one row per completed audit, optional JSONL export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from attacklm.queue.compare import SHIPPED_ATTACKS, load_report
from attacklm.queue.db import QueueDB

HEADLINE_METRIC = {
    "audit_prompt_injection": "overall_asr",
    "audit_system_prompt": "overall_leakage",
    "audit_canary_pipeline": "exact_token_rate",
    "audit_calibration": "ece",
}


def _headline_value(attack: str, report: dict[str, Any] | None) -> float | None:
    """The headline metric's value for one report.

    I1: `audit_calibration` has no flat `summary` dict -- the real writer
    (scripts/eval_calibration.py) nests brier/ece under
    `results.{in_distribution,near_ood,ood}`, each of which may be `None`.
    The headline for history is `results.in_distribution.ece`.
    """
    if report is None:
        return None
    if attack == "audit_calibration":
        in_dist = (report.get("results") or {}).get("in_distribution") or {}
        return in_dist.get("ece")
    metric = HEADLINE_METRIC.get(attack)
    if metric is None:
        return None
    return (report.get("summary") or {}).get(metric)


def history_rows(db: QueueDB, subject: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attack in SHIPPED_ATTACKS:
        metric = HEADLINE_METRIC.get(attack)
        if metric is None:
            # Minor (final review): a future SHIPPED_ATTACKS entry with no
            # HEADLINE_METRIC entry must be skipped, not KeyError.
            continue
        for t in db.list_tasks(status="completed", type=attack, limit=10000):
            args = t.args_dict
            # I5: a merged model has no adapter, so its subject is its
            # base_model -- it must not be mislabelled "(baseline)" just
            # because `adapter` is absent. Only a tagged baseline is one.
            s = args.get("adapter") or args.get("base_model")
            if subject and s != subject:
                continue
            report = load_report(t)
            value = _headline_value(attack, report)
            rows.append({
                "task_id": t.id,
                "finished_at": t.finished_at,
                "attack": attack,
                "subject": "(baseline)" if t.gauntlet == "baseline" else s,
                "base_model": args.get("base_model"),
                "metric": metric,
                "value": value,
                "report": t.artifact_path,
            })
    rows.sort(key=lambda r: r["task_id"], reverse=True)
    return rows[:limit]


def append_jsonl(rows: list[dict[str, Any]], path: Path) -> int:
    seen: set[tuple[int, str | None]] = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                seen.add((d["task_id"], d.get("report")))
            except (ValueError, KeyError):
                # Minor (final review): a malformed/partial line in an
                # existing history file must not crash the append -- skip
                # it (it also won't dedup against, which is acceptable).
                continue
    new = [r for r in rows if (r["task_id"], r.get("report")) not in seen]
    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            for r in reversed(new):  # oldest first in the file
                f.write(json.dumps(r) + "\n")
    return len(new)


def render_history(rows: list[dict[str, Any]]) -> str:
    cols = ("#", "finished", "attack", "subject", "base model", "metric", "value", "report")
    cells = [(str(r["task_id"]), (r["finished_at"] or "")[:19], r["attack"], r["subject"],
              r["base_model"] or "", r["metric"], "—" if r["value"] is None else f"{r['value']:.3f}",
              r["report"] or "") for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)] if cells else [len(c) for c in cols]
    lines = ["  ".join(c.ljust(w) for c, w in zip(cols, widths))]
    lines += ["  ".join(v.ljust(w) for v, w in zip(row, widths)) for row in cells]
    return "\n".join(lines)
