#!/usr/bin/env python3
"""Tests for src/attacklm/bench/report.py — the normalised report."""

from __future__ import annotations

from attacklm.bench.packs import Pack, PackSource
from attacklm.bench.report import build_report
from attacklm.bench.scorers import ItemScore

PACK = Pack(
    name="demo",
    harness_task="t",
    mode="local_scored",
    source=PackSource(kind="local"),
    license="MIT",
    redistributable=True,
    metric="accuracy",
    chance_level=0.25,
    reference_scores={},
    ladder=[100, None],
    categories_from="field:category",
)


def _scores():
    # Scores MUST vary by id: flat per-side values cannot distinguish id-based
    # from positional pairing (v0.20.0 fix-wave lesson).
    return [
        ItemScore("q0", "cat_a", 1.0, "C", True),
        ItemScore("q1", "cat_a", 0.0, "A", True),
        ItemScore("q2", "cat_b", 1.0, "B", True),
        ItemScore("q3", "cat_b", 0.0, None, False),
    ]


def test_summary_counts_and_score():
    rep = build_report(PACK, _scores(), {"model": "m"})
    assert rep["summary"]["score_raw"]["n"] == 4
    assert rep["summary"]["score_raw"]["invalid"] == 1
    assert rep["summary"]["score_raw"]["score"] == 0.5  # 2 of 4


def test_by_category_breakdown():
    rep = build_report(PACK, _scores(), {"model": "m"})
    by_cat = rep["summary"]["by_category"]
    assert by_cat["cat_a"] == {"score": 0.5, "n": 2}
    assert by_cat["cat_b"] == {"score": 0.5, "n": 2}


def test_results_carry_compare_contract_keys():
    rep = build_report(PACK, _scores(), {"model": "m"})
    first = rep["results"][0]
    assert first["question_id"] == "q0"
    assert first["category"] == "cat_a"
    assert first["score"] == 1.0
    assert first["valid"] is True


def test_excluded_item_omits_score_key_entirely():
    """score=None must OMIT the key so compare.pair_items counts it unpaired."""
    scores = [
        ItemScore("q0", "cat_a", 1.0, "C", True),
        ItemScore("q1", "cat_a", None, None, False),
    ]
    rep = build_report(PACK, scores, {"model": "m"})
    kept, dropped = rep["results"]
    assert "score" in kept
    assert "score" not in dropped
    assert dropped["valid"] is False
    # An excluded item is not in the denominator either.
    assert rep["summary"]["score_raw"]["n"] == 1
    assert rep["summary"]["score_raw"]["invalid"] == 1


def test_metadata_is_passed_through():
    rep = build_report(PACK, _scores(), {"model": "m", "backend": "vllm"})
    assert rep["metadata"]["model"] == "m"
    assert rep["metadata"]["backend"] == "vllm"
    assert rep["metadata"]["pack"] == "demo"
    assert rep["metadata"]["mode"] == "local_scored"
    assert rep["metadata"]["chance_level"] == 0.25


def test_empty_scores_do_not_divide_by_zero():
    rep = build_report(PACK, [], {"model": "m"})
    assert rep["summary"]["score_raw"] == {"score": None, "n": 0, "invalid": 0}
    assert rep["summary"]["by_category"] == {}
    assert rep["results"] == []


def test_score_clean_is_absent_until_phase_2():
    """Phase 1 emits raw only; a null clean score must never read as equal."""
    rep = build_report(PACK, _scores(), {"model": "m"})
    assert "score_clean" not in rep["summary"]


def test_report_is_json_serialisable():
    import json

    json.dumps(build_report(PACK, _scores(), {"model": "m"}))


def test_report_matches_live_compare_contract():
    """Assert against compare.py's real module, not a hand-copied fixture.

    The v0.20.0 calibration bug was a test that passed for years against a
    schema the real writer never produced.
    """
    from attacklm.queue.compare import ITEM_METRICS

    rep = build_report(PACK, _scores(), {"model": "m"})
    metric, id_key, cat_key = ("score", "question_id", "category")

    assert all(id_key in r and cat_key in r for r in rep["results"])
    assert any(metric in r for r in rep["results"])
    assert isinstance(ITEM_METRICS, dict)
    # Every existing entry is a list of 3-tuples; ours must match that shape.
    for entries in ITEM_METRICS.values():
        assert all(len(e) == 3 for e in entries)
