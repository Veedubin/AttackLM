#!/usr/bin/env python3
"""The escalation ladder must actually be consulted, not merely exist.

Advisory only: it never queues GPU work on its own, because auto-spending a
GPU hour off the back of a comparison would be a surprising thing for a
reporting command to do.
"""

from __future__ import annotations

import json
from pathlib import Path

from attacklm.queue.compare import Row, ladder_advice
from attacklm.queue.db import Task


def _task(tmp_path, pack, rung, scores, metric="score_clean"):
    report = {
        "metadata": {"pack": pack, "resolved_config": {"rung": rung}},
        "summary": {},
        "results": [
            {"question_id": f"q{i}", "category": "c", metric: v, "score": v}
            for i, v in enumerate(scores)
        ],
    }
    path = Path(tmp_path) / f"{pack}.json"
    path.write_text(json.dumps(report))
    # Task wraps a DB row dict, not kwargs.
    return Task({
        "id": 1,
        "type": f"bench_{pack.replace('-', '_')}",
        "label": "l",
        "status": "completed",
        "args": json.dumps({"pack": pack, "output": str(path)}),
        "depends_on": "[]",
        "artifact_path": str(path),
    })


class _Run:
    def __init__(self, tasks):
        self.tasks = tasks


def _row(attack, metric, n, delta, lo, hi, b=0.6):
    return Row(attack, metric, "overall", n, 0.45, b, delta, lo, hi, "BETTER")


def test_decisive_ci_advises_stop(tmp_path):
    task = _task(tmp_path, "ctibench-mcq", 100, [1.0, 0.0] * 10)
    run = _Run({"bench_ctibench_mcq": task})
    rows = [_row("bench_ctibench_mcq", "score_clean", 20, 0.15, 0.08, 0.22)]
    adv = ladder_advice(run, rows)["bench_ctibench_mcq"]
    assert adv["escalate"] is False
    assert adv["reason"] == "stop_decisive"
    assert adv["gated_on"] == "clean"


def test_wide_ci_advises_escalation_to_the_next_rung(tmp_path):
    task = _task(tmp_path, "ctibench-mcq", 100, [1.0, 0.0] * 10)
    run = _Run({"bench_ctibench_mcq": task})
    rows = [_row("bench_ctibench_mcq", "score_clean", 20, 0.05, -0.30, 0.40)]
    adv = ladder_advice(run, rows)["bench_ctibench_mcq"]
    assert adv["escalate"] is True
    assert adv["next_rung"] == 200          # ctibench-mcq ladder: 100,200,400,800,null
    assert adv["current_rung"] == 100


def test_falls_back_to_raw_when_no_clean_metric(tmp_path):
    task = _task(tmp_path, "ctibench-mcq", 100, [1.0, 0.0] * 10, metric="score")
    run = _Run({"bench_ctibench_mcq": task})
    rows = [_row("bench_ctibench_mcq", "score", 20, 0.05, -0.30, 0.40)]
    adv = ladder_advice(run, rows)["bench_ctibench_mcq"]
    assert adv["gated_on"] == "raw"


def test_clean_is_preferred_over_raw_when_both_present(tmp_path):
    task = _task(tmp_path, "ctibench-mcq", 100, [1.0, 0.0] * 10)
    run = _Run({"bench_ctibench_mcq": task})
    rows = [
        _row("bench_ctibench_mcq", "score", 20, 0.15, 0.08, 0.22),      # decisive
        _row("bench_ctibench_mcq", "score_clean", 18, 0.05, -0.3, 0.4),  # still wide
    ]
    adv = ladder_advice(run, rows)["bench_ctibench_mcq"]
    assert adv["gated_on"] == "clean"
    assert adv["escalate"] is True, "must not stop on the raw CI alone"


def test_audits_are_not_laddered(tmp_path):
    task = _task(tmp_path, "ctibench-mcq", 100, [1.0, 0.0] * 10)
    run = _Run({"audit_prompt_injection": task})
    assert ladder_advice(run, [_row("audit_prompt_injection", "asr", 20, 0.1, 0.05, 0.2)]) == {}


def test_unknown_pack_is_skipped_not_fatal(tmp_path):
    task = _task(tmp_path, "not-a-pack", 100, [1.0, 0.0] * 10)
    run = _Run({"bench_ctibench_mcq": task})
    rows = [_row("bench_ctibench_mcq", "score_clean", 20, 0.05, -0.3, 0.4)]
    assert ladder_advice(run, rows) == {}
