#!/usr/bin/env python3
"""Tests for scripts/bench_run.py — the capability benchmark CLI.

Hermetic: the harness subprocess is patched out, so no GPU, no network and no
inspect_ai install is required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import bench_run  # noqa: E402

from attacklm.bench.adapters.inspect_adapter import ParsedSample  # noqa: E402


def _items_file(tmp_path, n=4):
    p = tmp_path / "items.jsonl"
    p.write_text(
        "\n".join(
            json.dumps(
                {
                    "question_id": f"q{i}",
                    "category": "cat_a" if i < 2 else "cat_b",
                    "tier": "mcq",
                    "messages": [{"role": "user", "content": f"q{i}"}],
                    "ground_truth": {"type": "mcq_choice", "answer": "C"},
                    "metadata": {},
                }
            )
            for i in range(n)
        )
    )
    return p


def _local_pack(tmp_path, name="demo"):
    d = tmp_path / "packs"
    d.mkdir(exist_ok=True)
    (d / f"{name}.yaml").write_text(
        f"""
name: {name}
harness_task: inspect_evals/{name}
mode: local_scored
source:
  kind: local
license: MIT
redistributable: true
metric: accuracy
chance_level: 0.25
ladder: [100, null]
categories_from: field:category
"""
    )
    return d


def _harness_pack(tmp_path, name="hpack"):
    d = tmp_path / "packs"
    d.mkdir(exist_ok=True)
    (d / f"{name}.yaml").write_text(
        f"""
name: {name}
harness_task: inspect_evals/{name}
mode: harness_scored
source:
  kind: local
license: MIT
redistributable: true
metric: accuracy
chance_level: 0.25
ladder: [100, null]
categories_from: field:category
"""
    )
    return d


# --------------------------------------------------------------------------
# argument defaults
# --------------------------------------------------------------------------


def test_parse_args_defaults_match_spec():
    args = bench_run.parse_args(["--pack", "demo", "--base-model", "m"])
    assert args.max_tokens == 512
    assert args.temperature == 0.0
    assert args.seed == 42
    assert args.sample_seed == 42
    assert args.invalid_policy == "count_wrong"
    assert args.backend == "vllm"
    assert args.stop == []
    assert args.rung is None


# --------------------------------------------------------------------------
# local_scored: we own the items and the scoring
# --------------------------------------------------------------------------


def test_local_scored_writes_report_with_resolved_config(tmp_path):
    out = tmp_path / "report.json"
    items = _items_file(tmp_path)
    parsed = [
        ParsedSample("q0", "C", None, None, None, False, {}),
        ParsedSample("q1", "A", None, None, None, False, {}),
        ParsedSample("q2", "C", None, None, None, False, {}),
        ParsedSample("q3", "nope", None, None, None, False, {}),
    ]
    with patch.object(bench_run, "_run_harness", return_value=parsed):
        rc = bench_run.main([
            "--pack", "demo", "--packs-dir", str(_local_pack(tmp_path)),
            "--questions", str(items), "--base-model", "m", "--output", str(out),
        ])
    assert rc == 0

    rep = json.loads(out.read_text())
    assert rep["summary"]["score_raw"]["n"] == 4
    assert rep["summary"]["score_raw"]["score"] == 0.5  # q0, q2 correct
    assert rep["metadata"]["resolved_config"]["max_tokens"] == 512
    assert rep["metadata"]["resolved_config"]["invalid_policy"] == "count_wrong"
    assert rep["metadata"]["harness"]["name"] == "inspect_ai"
    assert rep["metadata"]["mode"] == "local_scored"


def test_local_scored_pairs_by_id_not_position(tmp_path):
    """Completions arriving out of order must score against their own item."""
    out = tmp_path / "report.json"
    items = _items_file(tmp_path)
    parsed = [
        ParsedSample("q3", "nope", None, None, None, False, {}),
        ParsedSample("q2", "C", None, None, None, False, {}),
        ParsedSample("q1", "A", None, None, None, False, {}),
        ParsedSample("q0", "C", None, None, None, False, {}),
    ]
    with patch.object(bench_run, "_run_harness", return_value=parsed):
        bench_run.main([
            "--pack", "demo", "--packs-dir", str(_local_pack(tmp_path)),
            "--questions", str(items), "--base-model", "m", "--output", str(out),
        ])

    by_id = {r["question_id"]: r for r in json.loads(out.read_text())["results"]}
    assert by_id["q0"]["score"] == 1.0
    assert by_id["q1"]["score"] == 0.0
    assert by_id["q2"]["score"] == 1.0
    assert by_id["q3"]["valid"] is False


def test_local_scored_missing_completion_is_invalid_not_crash(tmp_path):
    out = tmp_path / "report.json"
    items = _items_file(tmp_path)
    parsed = [ParsedSample("q0", "C", None, None, None, False, {})]
    with patch.object(bench_run, "_run_harness", return_value=parsed):
        rc = bench_run.main([
            "--pack", "demo", "--packs-dir", str(_local_pack(tmp_path)),
            "--questions", str(items), "--base-model", "m", "--output", str(out),
        ])
    assert rc == 0
    by_id = {r["question_id"]: r for r in json.loads(out.read_text())["results"]}
    assert by_id["q0"]["score"] == 1.0
    assert by_id["q1"]["valid"] is False


def test_rung_limits_items(tmp_path):
    out = tmp_path / "report.json"
    items = _items_file(tmp_path, n=20)
    with patch.object(bench_run, "_run_harness", return_value=[]):
        bench_run.main([
            "--pack", "demo", "--packs-dir", str(_local_pack(tmp_path)),
            "--questions", str(items), "--base-model", "m", "--output", str(out),
            "--rung", "5",
        ])
    assert len(json.loads(out.read_text())["results"]) == 5


# --------------------------------------------------------------------------
# harness_scored: the harness owns the items and the scoring
# --------------------------------------------------------------------------


def test_harness_scored_uses_harness_scores(tmp_path):
    out = tmp_path / "report.json"
    parsed = [
        ParsedSample("h0", "ANSWER: B", 1.0, "B", "persistence", True, {}),
        ParsedSample("h1", "ANSWER: C", 0.0, "C", "detection", True, {}),
        ParsedSample("h2", "I cannot.", None, "", "execution", False, {}),
    ]
    with patch.object(bench_run, "_run_harness", return_value=parsed):
        rc = bench_run.main([
            "--pack", "hpack", "--packs-dir", str(_harness_pack(tmp_path)),
            "--base-model", "m", "--output", str(out),
        ])
    assert rc == 0

    rep = json.loads(out.read_text())
    by_id = {r["question_id"]: r for r in rep["results"]}
    assert by_id["h0"]["score"] == 1.0
    assert by_id["h1"]["score"] == 0.0
    # Unparseable: counted wrong under the default policy, but flagged invalid.
    assert by_id["h2"]["score"] == 0.0
    assert by_id["h2"]["valid"] is False
    assert rep["summary"]["score_raw"]["invalid"] == 1
    assert rep["metadata"]["mode"] == "harness_scored"


def test_harness_scored_carries_category_from_harness(tmp_path):
    out = tmp_path / "report.json"
    parsed = [ParsedSample("h0", "x", 1.0, "B", "persistence", True, {})]
    with patch.object(bench_run, "_run_harness", return_value=parsed):
        bench_run.main([
            "--pack", "hpack", "--packs-dir", str(_harness_pack(tmp_path)),
            "--base-model", "m", "--output", str(out),
        ])
    assert json.loads(out.read_text())["results"][0]["category"] == "persistence"


def test_harness_scored_exclude_policy_omits_score(tmp_path):
    out = tmp_path / "report.json"
    parsed = [
        ParsedSample("h0", "x", 1.0, "B", "c", True, {}),
        ParsedSample("h1", "no", None, "", "c", False, {}),
    ]
    with patch.object(bench_run, "_run_harness", return_value=parsed):
        bench_run.main([
            "--pack", "hpack", "--packs-dir", str(_harness_pack(tmp_path)),
            "--base-model", "m", "--output", str(out),
            "--invalid-policy", "exclude",
        ])
    results = {r["question_id"]: r for r in json.loads(out.read_text())["results"]}
    assert "score" in results["h0"]
    assert "score" not in results["h1"]


def test_harness_scored_ignores_questions_file(tmp_path):
    """The harness owns the dataset; a stray --questions must not be used."""
    out = tmp_path / "report.json"
    parsed = [ParsedSample("h0", "x", 1.0, "B", "c", True, {})]
    with patch.object(bench_run, "_run_harness", return_value=parsed):
        bench_run.main([
            "--pack", "hpack", "--packs-dir", str(_harness_pack(tmp_path)),
            "--questions", str(_items_file(tmp_path)),
            "--base-model", "m", "--output", str(out),
        ])
    ids = [r["question_id"] for r in json.loads(out.read_text())["results"]]
    assert ids == ["h0"]


def test_unknown_pack_errors_cleanly(tmp_path):
    with pytest.raises(KeyError, match="nope"):
        bench_run.main([
            "--pack", "nope", "--packs-dir", str(_local_pack(tmp_path)),
            "--base-model", "m", "--output", str(tmp_path / "r.json"),
        ])


# --------------------------------------------------------------------------
# dual scoring end to end through the CLI
# --------------------------------------------------------------------------


def test_report_carries_null_clean_when_decontam_skipped(tmp_path):
    out = tmp_path / "report.json"
    parsed = [ParsedSample("h0", "x", 1.0, "B", "c", True, {})]
    with patch.object(bench_run, "_run_harness", return_value=parsed):
        bench_run.main([
            "--pack", "hpack", "--packs-dir", str(_harness_pack(tmp_path)),
            "--base-model", "m", "--output", str(out), "--no-decontam",
        ])
    summary = json.loads(out.read_text())["summary"]
    assert summary["score_clean"] is None
    assert summary["score_clean_reason"]


def test_decontam_runs_by_default_and_reports_unchecked_without_records(tmp_path):
    """No training records on disk -> checked False, never a silent zero."""
    out = tmp_path / "report.json"
    parsed = [ParsedSample("h0", "x", 1.0, "B", "c", True, {})]
    with patch.object(bench_run, "_run_harness", return_value=parsed):
        bench_run.main([
            "--pack", "hpack", "--packs-dir", str(_harness_pack(tmp_path)),
            "--base-model", "m", "--output", str(out),
        ])
    rep = json.loads(out.read_text())
    assert rep["metadata"]["decontamination"]["checked"] is False
    assert rep["summary"]["score_clean"] is None


def test_clean_score_emitted_when_training_records_supplied(tmp_path):
    """A real overlap must lower the clean score below the raw one."""
    records = tmp_path / "train.jsonl"
    shared = "China Chopper is a web shell used for persistence on IIS servers " * 3
    records.write_text(json.dumps({
        "messages": [{"role": "assistant", "content": shared}], "source": "sigma-hq",
    }))

    items = tmp_path / "items.jsonl"
    items.write_text("\n".join([
        json.dumps({"question_id": "q0", "category": "c", "tier": "t",
                    "messages": [{"role": "user", "content": shared}],
                    "ground_truth": {"type": "mcq_choice", "answer": "C"}, "metadata": {}}),
        json.dumps({"question_id": "q1", "category": "c", "tier": "t",
                    "messages": [{"role": "user", "content": "something wholly unrelated"}],
                    "ground_truth": {"type": "mcq_choice", "answer": "C"}, "metadata": {}}),
    ]))

    out = tmp_path / "report.json"
    parsed = [ParsedSample("q0", "C", None, None, None, False, {}),   # correct
              ParsedSample("q1", "A", None, None, None, False, {})]   # wrong
    with patch.object(bench_run, "_run_harness", return_value=parsed):
        bench_run.main([
            "--pack", "demo", "--packs-dir", str(_local_pack(tmp_path)),
            "--questions", str(items), "--base-model", "m", "--output", str(out),
            "--training-records", str(records),
        ])

    rep = json.loads(out.read_text())
    if not rep["metadata"]["decontamination"]["checked"]:
        pytest.skip("decontam/datasketch unavailable in this environment")
    assert rep["summary"]["score_raw"]["score"] == 0.5          # q0 right, q1 wrong
    by_id = {r["question_id"]: r for r in rep["results"]}
    assert by_id["q0"]["contaminated"] is True
    assert "score_clean" not in by_id["q0"]
    assert rep["summary"]["score_clean"]["score"] == 0.0        # only q1 survives


# --------------------------------------------------------------------------
# posture: the same completions scored for refused/answered (Phase 3)
# --------------------------------------------------------------------------


def _posture_pack(tmp_path, name="applied"):
    d = tmp_path / "packs"
    d.mkdir(exist_ok=True)
    (d / f"{name}.yaml").write_text(
        f"""
name: {name}
harness_task: inspect_evals/{name}
mode: local_scored
source:
  kind: local
license: MIT
redistributable: true
metric: micro_f1
chance_level: 0.0
ladder: [100, null]
categories_from: field:category
posture: true
"""
    )
    return d


def test_posture_pack_reports_refusal_rate(tmp_path):
    out = tmp_path / "report.json"
    items = _items_file(tmp_path, n=2)
    parsed = [
        ParsedSample("q0", "I'm sorry, but I can't assist with that.", None, None, None, False, {}),
        ParsedSample("q1", "The activity maps to T1071.", None, None, None, False, {}),
    ]
    with patch.object(bench_run, "_run_harness", return_value=parsed):
        rc = bench_run.main([
            "--pack", "applied", "--packs-dir", str(_posture_pack(tmp_path)),
            "--questions", str(items), "--base-model", "m", "--output", str(out),
            "--no-decontam",
        ])
    assert rc == 0
    rep = json.loads(out.read_text())
    assert rep["summary"]["posture"]["refusal_rate"] == 0.5
    byid = {r["question_id"]: r for r in rep["results"]}
    assert byid["q0"]["refused"] == 1.0 and byid["q0"]["posture"] == "refused"
    assert byid["q1"]["refused"] == 0.0 and byid["q1"]["posture"] == "answered"
    assert "posture" in rep["metadata"]["resolved_config"]


def test_non_posture_pack_has_no_posture_block(tmp_path):
    out = tmp_path / "report.json"
    items = _items_file(tmp_path, n=2)
    parsed = [
        ParsedSample("q0", "I'm sorry, but I can't assist with that.", None, None, None, False, {}),
        ParsedSample("q1", "C", None, None, None, False, {}),
    ]
    with patch.object(bench_run, "_run_harness", return_value=parsed):
        bench_run.main([
            "--pack", "demo", "--packs-dir", str(_local_pack(tmp_path)),
            "--questions", str(items), "--base-model", "m", "--output", str(out),
            "--no-decontam",
        ])
    rep = json.loads(out.read_text())
    assert "posture" not in rep["summary"]


# --------------------------------------------------------------------------
# posture judge: second harness pass grades answered completions (Phase 3)
# --------------------------------------------------------------------------


def test_judge_pass_refines_answered_to_taught_overshared(tmp_path):
    """With --judge-model, bench_run runs a SECOND harness pass whose
    completions are the judge's verdicts, and refines the posture labels."""
    out = tmp_path / "report.json"
    items = _items_file(tmp_path, n=3)  # q0, q1, q2
    # First pass: q0 refuses, q1 & q2 answer.
    main_samples = [
        ParsedSample("q0", "I'm sorry, but I can't assist with that.", None, None, None, False, {}),
        ParsedSample("q1", "This is SQL injection; use parameterised queries.", None, None, None, False, {}),
        ParsedSample("q2", "Here is a working exploit you can run as-is.", None, None, None, False, {}),
    ]
    # Judge pass: only the two ANSWERED ids are judged, in item order.
    judge_samples = [
        ParsedSample("q1", "VERDICT: taught", None, None, None, False, {}),
        ParsedSample("q2", "VERDICT: overshared", None, None, None, False, {}),
    ]
    with patch.object(bench_run, "_run_harness", side_effect=[main_samples, judge_samples]):
        rc = bench_run.main([
            "--pack", "applied", "--packs-dir", str(_posture_pack(tmp_path)),
            "--questions", str(items), "--base-model", "m", "--output", str(out),
            "--no-decontam", "--judge-model", "judge/m",
        ])
    assert rc == 0
    rep = json.loads(out.read_text())
    ps = rep["summary"]["posture"]
    assert ps["refusal_rate"] == round(1 / 3, 4)
    assert ps["taught_rate"] == round(1 / 3, 4)
    assert ps["overshared_rate"] == round(1 / 3, 4)
    byid = {r["question_id"]: r for r in rep["results"]}
    assert byid["q1"]["posture"] == "taught" and byid["q1"]["taught"] == 1.0
    assert byid["q2"]["posture"] == "overshared" and byid["q2"]["overshared"] == 1.0
    assert byid["q0"]["posture"] == "refused"
    assert rep["metadata"]["resolved_config"]["posture"]["judge_model"] == "judge/m"


def test_no_judge_model_keeps_posture_deterministic(tmp_path):
    out = tmp_path / "report.json"
    items = _items_file(tmp_path, n=2)
    parsed = [
        ParsedSample("q0", "This is SQL injection; use parameterised queries.", None, None, None, False, {}),
        ParsedSample("q1", "Here is a working exploit you can run as-is.", None, None, None, False, {}),
    ]
    # If a judge pass ran, _run_harness would be called twice; assert it runs once.
    with patch.object(bench_run, "_run_harness", return_value=parsed) as rh:
        bench_run.main([
            "--pack", "applied", "--packs-dir", str(_posture_pack(tmp_path)),
            "--questions", str(items), "--base-model", "m", "--output", str(out),
            "--no-decontam",
        ])
    assert rh.call_count == 1
    rep = json.loads(out.read_text())
    assert "taught_rate" not in rep["summary"]["posture"]
