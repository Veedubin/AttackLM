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


def history_rows(db: QueueDB, adapter: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attack in SHIPPED_ATTACKS:
        for t in db.list_tasks(status="completed", type=attack, limit=10000):
            a = t.args_dict.get("adapter") or None
            if adapter and a != adapter:
                continue
            report = load_report(t)
            metric = HEADLINE_METRIC[attack]
            value = report.get("summary", {}).get(metric) if report else None
            rows.append({
                "task_id": t.id,
                "finished_at": t.finished_at,
                "attack": attack,
                "adapter": a or "(baseline)",
                "base_model": t.args_dict.get("base_model"),
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
            if line.strip():
                d = json.loads(line)
                seen.add((d["task_id"], d.get("report")))
    new = [r for r in rows if (r["task_id"], r.get("report")) not in seen]
    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            for r in reversed(new):  # oldest first in the file
                f.write(json.dumps(r) + "\n")
    return len(new)


def render_history(rows: list[dict[str, Any]]) -> str:
    cols = ("#", "finished", "attack", "adapter", "base model", "metric", "value", "report")
    cells = [(str(r["task_id"]), (r["finished_at"] or "")[:19], r["attack"], r["adapter"],
              r["base_model"] or "", r["metric"], "—" if r["value"] is None else f"{r['value']:.3f}",
              r["report"] or "") for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)] if cells else [len(c) for c in cols]
    lines = ["  ".join(c.ljust(w) for c, w in zip(cols, widths))]
    lines += ["  ".join(v.ljust(w) for v, w in zip(row, widths)) for row in cells]
    return "\n".join(lines)
