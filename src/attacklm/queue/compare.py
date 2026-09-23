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
from attacklm.queue.stats import Interval, paired_bootstrap, verdict

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
    "bench_cybermetric",
    "bench_ctibench_mcq",
    "bench_ctibench_ate",
    "bench_secbench_en",
    "bench_seceval",
    "bench_applied_attack",
]

# attack -> [(metric, id_key, category_key)]
ITEM_METRICS: dict[str, list[tuple[str, str, str]]] = {
    "audit_prompt_injection": [("asr", "question_id", "tier")],
    "audit_system_prompt": [("asr", "question_id", "tier")],
    "audit_canary_pipeline": [
        ("exact_token", "canary_id", "prefix"),
        ("near_verbatim", "canary_id", "prefix"),
    ],
    # Capability benchmarks. An item the harness could not parse omits "score"
    # entirely, so the existing missing-metric rule counts it unpaired rather
    # than scoring it a genuine 0.0. Phase 2 adds a parallel "score_clean"
    # entry, which contaminated items omit the same way.
    # score_clean is the contamination-corrected metric. A contaminated item
    # OMITS it, so the missing-metric rule counts that item unpaired in the
    # clean comparison while it still pairs in the raw one -- two verdicts from
    # one run, no new logic here.
    "bench_cybermetric": [
        ("score", "question_id", "category"),
        ("score_clean", "question_id", "category"),
    ],
    "bench_ctibench_mcq": [
        ("score", "question_id", "category"),
        ("score_clean", "question_id", "category"),
    ],
    "bench_ctibench_ate": [
        ("score", "question_id", "category"),
        ("score_clean", "question_id", "category"),
    ],
    "bench_secbench_en": [
        ("score", "question_id", "category"),
        ("score_clean", "question_id", "category"),
    ],
    "bench_seceval": [
        ("score", "question_id", "category"),
        ("score_clean", "question_id", "category"),
    ],
    # Layer 1 authored applied set (Phase 3). One run yields BOTH capability
    # (score / score_clean, higher is better) AND posture (refused, higher is
    # worse for a model built to answer). The two directions inside one attack
    # are what _higher_is_worse's per-metric override exists for.
    "bench_applied_attack": [
        ("score", "question_id", "category"),
        ("score_clean", "question_id", "category"),
        ("refused", "question_id", "category"),
        # Judge verdicts among answered items (present only when a judge ran).
        # taught is the target posture; overshared is the arming failure.
        ("taught", "question_id", "category"),
        ("overshared", "question_id", "category"),
    ],
}

# Metric DIRECTION per attack, used for the BETTER/WORSE label.
#
# Every audit measures a failure -- attack success rate, system-prompt
# leakage, canary extraction, calibration error -- so a rise is a regression
# and `higher_is_worse` is True. A capability benchmark measures success, so a
# rise is an improvement. Without this distinction a model that genuinely got
# BETTER at answering security questions would be reported as WORSE, inverting
# the signal that training decisions are steered by.
#
# Unknown attacks default to True: for this project a new metric is far more
# likely to measure failure, and over-reporting a regression is the safer
# error.
HIGHER_IS_WORSE: dict[str, bool] = {
    "audit_prompt_injection": True,
    "audit_system_prompt": True,
    "audit_canary_pipeline": True,
    "audit_calibration": True,
    "bench_cybermetric": False,
    "bench_ctibench_mcq": False,
    "bench_ctibench_ate": False,
    "bench_secbench_en": False,
    "bench_seceval": False,
    # Applied set: capability rises are improvements...
    "bench_applied_attack": False,
    # ...but a rise in the refusal rate is a regression for a model whose
    # entire thesis is that it answers where a stock model refuses. This
    # attack:metric key overrides the per-attack direction above for that one
    # metric; every other metric on the pack falls through to False.
    "bench_applied_attack:refused": True,
    # A rise in oversharing (emitting operational content) is the arming
    # failure -- unambiguously WORSE -- so it must override the pack default
    # too, or "teaches don't arm" would read a MORE-arming model as better.
    # taught rising is BETTER, which the pack default (False) already gives.
    "bench_applied_attack:overshared": True,
}


def _higher_is_worse(attack: str, metric: str) -> bool:
    """Verdict direction for one (attack, metric).

    A single pack can carry metrics of opposite direction -- a capability
    score (higher better) and a posture refusal rate (higher worse). The
    per-metric key `attack:metric` wins; then the per-attack key; then True,
    since for this project an unregistered metric most likely measures failure
    and over-reporting a regression is the safer error.
    """
    if f"{attack}:{metric}" in HIGHER_IS_WORSE:
        return HIGHER_IS_WORSE[f"{attack}:{metric}"]
    return HIGHER_IS_WORSE.get(attack, True)

# attack -> metric keys shown as Δ only (no per-item scores exist).
# audit_calibration's real report shape (scripts/eval_calibration.py) is
# {"results": {"in_distribution": {...}, "near_ood": {...}|None, "ood": {...}|None}},
# not a flat "summary" dict -- each present set has its own "brier"/"ece".
SUMMARY_METRICS: dict[str, list[str]] = {"audit_calibration": ["brier", "ece"]}
CALIBRATION_SETS = ("in_distribution", "near_ood", "ood")


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
        if subject:
            # `subject` may not have a non-baseline run of its own -- e.g.
            # the untuned base model itself, which only ever runs tagged
            # gauntlet="baseline". Fall back to treating `subject=X` as
            # "X's run, or X's baseline", so a baseline is addressable as
            # an explicit compare side too.
            return select_run(db, base_model=subject, attacks=attacks)
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


def latest_subject(db: QueueDB, attacks: list[str] | None = None) -> tuple[str, str | None] | None:
    """(subject, base_model) of the newest completed non-baseline audit.

    `subject` is the task's adapter if it has one, else its base_model (a
    merged model with no adapter). Scans `attacks` (default `SHIPPED_ATTACKS`)
    -- callers that then feed the result into `select_run(attacks=...)`
    should pass the same preset-filtered list here, so the two never
    disagree about which attack types exist (I2: a mismatch made
    `select_run` return None for a subject `latest_subject` just reported).
    """
    attacks = attacks or SHIPPED_ATTACKS

    def pred(t: Task) -> bool:
        if t.gauntlet == "baseline":
            return False
        args = t.args_dict
        return bool(args.get("adapter")) or bool(args.get("base_model"))

    newest: Task | None = None
    for attack in attacks:
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


def _is_errored(record: dict[str, Any]) -> bool:
    """True for a generation-error sentinel, not a real grade.

    Attack scripts (e.g. `audit_prompt_injection.py`) reuse `asr = 0.5` for
    both a genuine "ambiguous" grade and a caught generation exception,
    distinguished only by `score == "error"` (and an `error` message key).
    Pairing an error sentinel as if it were a real 0.5 would silently
    corrupt the comparison.
    """
    return record.get("score") == "error" or "error" in record


def pair_items(
    ra: dict[str, Any], rb: dict[str, Any], metric: str, id_key: str, cat_key: str
) -> tuple[dict[str, tuple[list[float], list[float]]], int]:
    """Join results[] by id. Returns {category: (a_vals, b_vals)} incl. 'overall', and the unpaired count.

    An item counts as paired only when *both* sides have the metric key
    present and non-None, and neither side is a generation-error sentinel
    (see `_is_errored`). An item present on both sides but missing the
    metric, or flagged as errored, on either counts as unpaired rather than
    being silently scored 0.0/0.5 -- a malformed report must not look like
    a clean grade.
    """
    ia = {r[id_key]: r for r in ra.get("results", []) if id_key in r}
    ib = {r[id_key]: r for r in rb.get("results", []) if id_key in r}
    common = [k for k in ia if k in ib]
    unpaired = (len(ia) - len(common)) + (len(ib) - len(common))
    cats: dict[str, tuple[list[float], list[float]]] = {"overall": ([], [])}
    for k in common:
        if _is_errored(ia[k]) or _is_errored(ib[k]):
            unpaired += 1
            continue
        va_raw, vb_raw = ia[k].get(metric), ib[k].get(metric)
        if va_raw is None or vb_raw is None:
            unpaired += 1
            continue
        va, vb = float(va_raw), float(vb_raw)
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
            # A canary report has two metrics (exact_token, near_verbatim);
            # each call to pair_items recomputes its own unpaired count over
            # the SAME id join, so a later metric must not shrink what an
            # earlier one already found -- take the max, don't overwrite.
            unpaired[attack] = max(unpaired.get(attack, 0), unp)
            ordered = ["overall"] + sorted(c for c in cats if c != "overall")
            for cat in ordered:
                a_vals, b_vals = cats[cat]
                iv = paired_bootstrap(a_vals, b_vals, resamples=resamples, seed=seed)
                if iv.n == 0:
                    # C1: an empty category (no paired items at all -- e.g.
                    # disjoint ids) must not read as a confident "SAME" with
                    # Δ=0.000 and CI [0,0]; it's not evidence of "no
                    # difference", it's evidence of nothing to compare.
                    rows.append(Row(attack, metric, cat, 0, None, None, None, None, None,
                                    "n/a", "no paired items"))
                    continue
                rows.append(Row(attack, metric, cat, iv.n, _mean(a_vals), _mean(b_vals),
                                iv.delta, iv.lo, iv.hi,
                                verdict(iv, _higher_is_worse(attack, metric))))
        if attack == "audit_calibration":
            # I1: no flat "summary" dict exists for calibration -- the real
            # writer (scripts/eval_calibration.py) nests brier/ece under
            # results.{in_distribution,near_ood,ood}, each of which may be
            # None (near_ood/ood are optional CLI flags). One row per set
            # present on BOTH sides, per metric.
            ra_sets, rb_sets = ra.get("results") or {}, rb.get("results") or {}
            for set_name in CALIBRATION_SETS:
                set_a, set_b = ra_sets.get(set_name), rb_sets.get(set_name)
                if set_a is None or set_b is None:
                    continue
                for key in SUMMARY_METRICS.get(attack, []):
                    sa, sb = set_a.get(key), set_b.get(key)
                    if sa is None or sb is None:
                        rows.append(Row(attack, key, set_name, 0, sa, sb, None, None, None,
                                        "n/a", "no value"))
                    else:
                        rows.append(Row(attack, key, set_name, 0, float(sa), float(sb),
                                        float(sb) - float(sa), None, None, "—"))
            unpaired.setdefault(attack, 0)
    return rows, unpaired


def _side(run: Run) -> dict[str, Any]:
    return {
        "subject": run.subject,
        "adapter": run.adapter,
        "base_model": run.base_model,
        "tasks": {k: t.id for k, t in run.tasks.items()},
    }


def _subject_interval(b_vals: list[float], resamples: int, seed: int):
    """CI of the SUBJECT's own score, for the ladder's dead-horse floor rule.

    paired_bootstrap over (zeros, b_vals) gives the CI of mean(b - 0), i.e. of
    the subject's score itself. Reusing the tested paired machinery beats
    writing a second, separately-buggy one-sample bootstrap.
    """
    return paired_bootstrap([0.0] * len(b_vals), b_vals, resamples=resamples, seed=seed)


def ladder_advice(
    run_b: Run, rows: list[Row], resamples: int = 2000, seed: int = 42
) -> dict[str, dict[str, Any]]:
    """What the escalation ladder advises next, per capability benchmark.

    Advisory only: it never queues GPU work on its own. Gated on the CLEAN
    metric where one exists, per the design -- escalating on the raw CI would
    stop the ladder while the number we actually believe is still noise.
    """
    try:
        from attacklm.bench.ladder import decide_next_rung
        from attacklm.bench.packs import get_pack
    except ImportError:
        return {}

    by_attack: dict[str, dict[str, Any]] = {}
    for attack, task in run_b.tasks.items():
        if attack not in ITEM_METRICS or not attack.startswith("bench_"):
            continue

        report = load_report(task)
        if report is None:
            continue
        pack_name = task.args_dict.get("pack") or report.get("metadata", {}).get("pack")
        if not pack_name:
            continue
        try:
            pack = get_pack(pack_name)
        except (KeyError, ValueError, RuntimeError):
            continue

        # Prefer the corrected metric; fall back to raw when no check ran.
        candidates = [r for r in rows if r.attack == attack and r.category == "overall"]
        row = next((r for r in candidates if r.metric == "score_clean"), None)
        gated_on = "clean"
        if row is None or row.n == 0:
            row = next((r for r in candidates if r.metric == "score"), None)
            gated_on = "raw"
        if row is None or row.delta is None:
            continue

        interval = Interval(row.delta, row.lo or 0.0, row.hi or 0.0, row.n)
        current = (report.get("metadata", {}).get("resolved_config") or {}).get("rung")

        subject_hi = None
        if row.b is not None and row.n:
            # Reconstruct the subject's own CI from its report items.
            metric = "score_clean" if gated_on == "clean" else "score"
            vals = [
                float(item[metric])
                for item in report.get("results", [])
                if item.get(metric) is not None
            ]
            if vals:
                subject_hi = _subject_interval(vals, resamples, seed).hi

        try:
            decision = decide_next_rung(
                pack.ladder,
                current,
                interval,
                subject_ci_hi=subject_hi,
                chance_level=pack.chance_level,
            )
        except ValueError:
            continue

        by_attack[attack] = {
            "pack": pack.name,
            "gated_on": gated_on,
            "current_rung": current,
            "escalate": decision.escalate,
            "next_rung": decision.next_rung,
            "reason": decision.reason,
        }
    return by_attack


def to_json(run_a: Run, run_b: Run, rows: list[Row], unpaired: dict[str, int]) -> dict[str, Any]:
    payload = {"a": _side(run_a), "b": _side(run_b),
               "rows": [row.__dict__ for row in rows], "unpaired": unpaired}
    advice = ladder_advice(run_b, rows)
    if advice:
        payload["ladder"] = advice
    return payload


def _fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def _label(attack: str) -> str:
    spec = REGISTRY.get(attack)
    return spec.label if spec else attack


def _run_header(run: Run) -> str:
    """`name (#id, #id · YYYY-MM-DD)` -- I7: the header must name which
    tasks were actually compared, not just the subject, so two compares of
    the same subject at different times aren't indistinguishable."""
    ids = sorted(t.id for t in run.tasks.values())
    if not ids:
        return run.name
    id_str = ", ".join(f"#{i}" for i in ids)
    dates = sorted(t.finished_at for t in run.tasks.values() if t.finished_at)
    date_str = f" · {dates[-1][:10]}" if dates else ""
    return f"{run.name} ({id_str}{date_str})"


def render_table(run_a: Run, run_b: Run, rows: list[Row], unpaired: dict[str, int]) -> str:
    header = f"A = {_run_header(run_a)}\nB = {_run_header(run_b)}\nΔ = B − A (higher is worse for asr/leakage/extraction)"
    footer = "unpaired items: " + ", ".join(f"{_label(k)}: {v}" for k, v in unpaired.items()) if unpaired else ""

    # Ladder advice, if any capability benchmark is in this comparison.
    advice_lines = []
    for attack, adv in ladder_advice(run_b, rows).items():
        if adv["escalate"]:
            advice_lines.append(
                f"{_label(attack)}: rerun at rung {adv['next_rung']} "
                f"(CI still wide on the {adv['gated_on']} score)"
            )
        else:
            advice_lines.append(f"{_label(attack)}: stop — {adv['reason']}")
    if advice_lines:
        footer = (footer + "\n" if footer else "") + "ladder: " + "; ".join(advice_lines)
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
