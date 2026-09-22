"""`attacklm queue compare` — pair two audit runs and report Δ with CIs.

A "run" is the newest completed task per attack type for one side: a
*subject* (an adapter path, or a merged model identified by its base_model
when it has no adapter), or a base model's tagged *baseline*. Items are
paired by id across the two sides so the bootstrap works on per-prompt
differences.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from attacklm.queue.db import QueueDB, Task
from attacklm.queue.registry import REGISTRY
from attacklm.queue.stats import paired_bootstrap, verdict

try:
    from rich.console import Console
    from rich.table import Table

    _HAS_RICH = True
except ImportError:  # pragma: no cover - exercised only without rich
    _HAS_RICH = False

SHIPPED_ATTACKS = [
    "audit_prompt_injection",
    "audit_system_prompt",
    "audit_canary_pipeline",
    "audit_calibration",
]

# attack -> [(metric, id_key, category_key)]
ITEM_METRICS: dict[str, list[tuple[str, str, str]]] = {
    "audit_prompt_injection": [("asr", "question_id", "tier")],
    "audit_system_prompt": [("asr", "question_id", "tier")],
    "audit_canary_pipeline": [
        ("exact_token", "canary_id", "prefix"),
        ("near_verbatim", "canary_id", "prefix"),
    ],
}
# attack -> summary keys shown as Δ only (no per-item scores exist)
SUMMARY_METRICS: dict[str, list[str]] = {"audit_calibration": ["brier", "ece"]}


@dataclass
class Run:
    adapter: str | None
    base_model: str | None
    subject: str | None = None
    tasks: dict[str, Task] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.subject if self.subject else f"baseline {self.base_model}"


@dataclass
class Row:
    attack: str
    metric: str
    category: str
    n: int
    a: float | None
    b: float | None
    delta: float | None
    lo: float | None
    hi: float | None
    verdict: str
    note: str = ""


def _newest_completed(db: QueueDB, attack: str, pred) -> Task | None:
    tasks = db.list_tasks(status="completed", type=attack, limit=10000)
    for task in reversed(tasks):  # list_tasks is id-ascending
        if pred(task):
            return task
    return None


def select_run(
    db: QueueDB,
    subject: str | None = None,
    base_model: str | None = None,
    attacks: list[str] | None = None,
) -> Run | None:
    """Newest completed task per attack for a subject, or for a base model's
    tagged baseline.

    A task matches `subject` when its adapter equals it, or — for a merged
    model with no adapter — its base_model equals it and the task isn't
    tagged as a baseline (baselines are reachable only via `base_model=`).
    """
    attacks = attacks or SHIPPED_ATTACKS
    if subject:

        def pred(t: Task) -> bool:
            args = t.args_dict
            if args.get("adapter") == subject:
                return True
            return (
                not args.get("adapter")
                and args.get("base_model") == subject
                and t.gauntlet != "baseline"
            )
    elif base_model:

        def pred(t: Task) -> bool:
            return (
                t.gauntlet == "baseline"
                and not t.args_dict.get("adapter")
                and t.args_dict.get("base_model") == base_model
            )
    else:
        raise ValueError("select_run needs subject or base_model")
    tasks = {}
    for attack in attacks:
        task = _newest_completed(db, attack, pred)
        if task is not None:
            tasks[attack] = task
    if not tasks:
        return None
    resolved_base = base_model or next(
        (t.args_dict.get("base_model") for t in tasks.values() if t.args_dict.get("base_model")), None
    )
    adapter = (
        subject
        if subject and any(t.args_dict.get("adapter") == subject for t in tasks.values())
        else None
    )
    return Run(adapter=adapter, base_model=resolved_base, subject=subject, tasks=tasks)


def latest_subject(db: QueueDB) -> tuple[str, str | None] | None:
    """(subject, base_model) of the newest completed non-baseline audit.

    `subject` is the task's adapter if it has one, else its base_model (a
    merged model with no adapter).
    """

    def pred(t: Task) -> bool:
        if t.gauntlet == "baseline":
            return False
        args = t.args_dict
        return bool(args.get("adapter")) or bool(args.get("base_model"))

    newest: Task | None = None
    for attack in SHIPPED_ATTACKS:
        t = _newest_completed(db, attack, pred)
        if t is not None and (newest is None or t.id > newest.id):
            newest = t
    if newest is None:
        return None
    args = newest.args_dict
    subject = args.get("adapter") or args.get("base_model")
    return subject, args.get("base_model")


def load_report(task: Task) -> dict[str, Any] | None:
    path = task.artifact_path or task.args_dict.get("output")
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def pair_items(
    ra: dict[str, Any], rb: dict[str, Any], metric: str, id_key: str, cat_key: str
) -> tuple[dict[str, tuple[list[float], list[float]]], int]:
    """Join results[] by id. Returns {category: (a_vals, b_vals)} incl. 'overall', and the unpaired count."""
    ia = {r[id_key]: r for r in ra.get("results", []) if id_key in r}
    ib = {r[id_key]: r for r in rb.get("results", []) if id_key in r}
    common = [k for k in ia if k in ib]
    unpaired = (len(ia) - len(common)) + (len(ib) - len(common))
    cats: dict[str, tuple[list[float], list[float]]] = {"overall": ([], [])}
    for k in common:
        va, vb = float(ia[k].get(metric) or 0.0), float(ib[k].get(metric) or 0.0)
        cats["overall"][0].append(va)
        cats["overall"][1].append(vb)
        cat = str(ia[k].get(cat_key, "?"))
        cats.setdefault(cat, ([], []))
        cats[cat][0].append(va)
        cats[cat][1].append(vb)
    return cats, unpaired


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def compare_runs(
    run_a: Run, run_b: Run, attacks: list[str], resamples: int = 2000, seed: int = 42
) -> tuple[list[Row], dict[str, int]]:
    rows: list[Row] = []
    unpaired: dict[str, int] = {}
    for attack in attacks:
        ta, tb = run_a.tasks.get(attack), run_b.tasks.get(attack)
        if ta is None or tb is None:
            rows.append(Row(attack, "", "overall", 0, None, None, None, None, None, "n/a",
                            "missing on " + ("A" if ta is None else "B")))
            continue
        ra, rb = load_report(ta), load_report(tb)
        if ra is None or rb is None:
            rows.append(Row(attack, "", "overall", 0, None, None, None, None, None, "n/a",
                            "report missing on " + ("A" if ra is None else "B")))
            continue
        for metric, id_key, cat_key in ITEM_METRICS.get(attack, []):
            cats, unp = pair_items(ra, rb, metric, id_key, cat_key)
            unpaired[attack] = unp
            ordered = ["overall"] + sorted(c for c in cats if c != "overall")
            for cat in ordered:
                a_vals, b_vals = cats[cat]
                iv = paired_bootstrap(a_vals, b_vals, resamples=resamples, seed=seed)
                rows.append(Row(attack, metric, cat, iv.n, _mean(a_vals), _mean(b_vals),
                                iv.delta, iv.lo, iv.hi, verdict(iv, higher_is_worse=True)))
        for key in SUMMARY_METRICS.get(attack, []):
            sa, sb = ra.get("summary", {}).get(key), rb.get("summary", {}).get(key)
            if sa is None or sb is None:
                rows.append(Row(attack, key, "overall", 0, sa, sb, None, None, None, "n/a", "no summary value"))
            else:
                rows.append(Row(attack, key, "overall", 0, float(sa), float(sb), float(sb) - float(sa), None, None, "—"))
            unpaired.setdefault(attack, 0)
    return rows, unpaired


def _side(run: Run) -> dict[str, Any]:
    return {
        "subject": run.subject,
        "adapter": run.adapter,
        "base_model": run.base_model,
        "tasks": {k: t.id for k, t in run.tasks.items()},
    }


def to_json(run_a: Run, run_b: Run, rows: list[Row], unpaired: dict[str, int]) -> dict[str, Any]:
    return {"a": _side(run_a), "b": _side(run_b),
            "rows": [row.__dict__ for row in rows], "unpaired": unpaired}


def _fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def _label(attack: str) -> str:
    spec = REGISTRY.get(attack)
    return spec.label if spec else attack


def render_table(run_a: Run, run_b: Run, rows: list[Row], unpaired: dict[str, int]) -> str:
    header = f"A = {run_a.name}\nB = {run_b.name}\nΔ = B − A (higher is worse for asr/leakage/extraction)"
    footer = "unpaired items: " + ", ".join(f"{_label(k)}: {v}" for k, v in unpaired.items()) if unpaired else ""
    cells = [(_label(r.attack), r.metric, r.category, str(r.n) if r.n else "—", _fmt(r.a), _fmt(r.b),
              _fmt(r.delta), "—" if r.lo is None else f"[{r.lo:+.3f}, {r.hi:+.3f}]",
              r.verdict + (f" ({r.note})" if r.note else "")) for r in rows]
    columns = ("Attack", "Metric", "Category", "n", "A", "B", "Δ", "95% CI", "Verdict")
    if _HAS_RICH:
        from io import StringIO

        buf = StringIO()
        console = Console(file=buf, force_terminal=False, width=140)
        table = Table(title="attacklm queue compare")
        for c in columns:
            table.add_column(c)
        for row, r in zip(cells, rows):
            style = {"WORSE": "red", "BETTER": "green"}.get(r.verdict, "")
            table.add_row(*row, style=style)
        console.print(header)
        console.print(table)
        if footer:
            console.print(footer)
        return buf.getvalue()
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(columns)] if cells else [len(c) for c in columns]
    lines = [header, "  ".join(c.ljust(w) for c, w in zip(columns, widths))]
    lines += ["  ".join(v.ljust(w) for v, w in zip(row, widths)) for row in cells]
    if footer:
        lines.append(footer)
    return "\n".join(lines)
