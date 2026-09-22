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
