#!/usr/bin/env python3
"""Tests for src/attacklm/bench/report.py — the normalised report."""

from __future__ import annotations

import pytest

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


def test_score_clean_is_explicitly_null_without_a_check():
    """Superseded the Phase 1 "absent" contract with a stronger one.

    Absence was ambiguous -- a reader could take a missing key for "no
    correction needed". An explicit null plus a reason cannot be misread.
    """
    rep = build_report(PACK, _scores(), {"model": "m"})
    assert rep["summary"]["score_clean"] is None
    assert rep["summary"]["score_clean_reason"]


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


# --------------------------------------------------------------------------
# dual scoring -- the "SAE correction"
# --------------------------------------------------------------------------

from attacklm.bench.contamination import ContaminationResult, ItemOverlap  # noqa: E402


def _contam(**over):
    overlaps = {
        qid: ItemOverlap(qid, j, c, src) for qid, (j, c, src) in over.items()
    }
    n_bad = sum(1 for o in overlaps.values() if o.contaminated)
    rate = n_bad / len(overlaps) if overlaps else 0.0
    return ContaminationResult(True, 0.80, overlaps, n_bad, rate)


def test_no_contamination_check_yields_explicit_null_not_absence():
    """A missing correction must never read as 'corrected, no change'."""
    rep = build_report(PACK, _scores(), {"model": "m"})
    assert rep["summary"]["score_clean"] is None
    assert "score_clean_reason" in rep["summary"]


def test_clean_equals_raw_when_nothing_is_contaminated():
    contam = _contam(
        q0=(0.1, False, None), q1=(0.0, False, None),
        q2=(0.2, False, None), q3=(0.0, False, None),
    )
    rep = build_report(PACK, _scores(), {"model": "m"}, contam)
    assert rep["summary"]["score_clean"]["score"] == rep["summary"]["score_raw"]["score"]
    assert rep["summary"]["contamination_rate"] == 0.0


def test_clean_drops_when_a_correct_item_is_contaminated():
    # q0 scores 1.0 and is contaminated; excluding it must LOWER the clean score.
    contam = _contam(
        q0=(0.95, True, "sigma-hq"), q1=(0.0, False, None),
        q2=(0.1, False, None), q3=(0.0, False, None),
    )
    rep = build_report(PACK, _scores(), {"model": "m"}, contam)
    assert rep["summary"]["score_raw"]["score"] == 0.5     # q0, q2 correct of 4
    assert rep["summary"]["score_clean"]["score"] == pytest.approx(0.3333, abs=1e-4)
    assert rep["summary"]["score_clean"]["n"] == 3


def test_contaminated_item_omits_score_clean_but_keeps_score():
    contam = _contam(q0=(0.95, True, "sigma-hq"), q1=(0.0, False, None),
                     q2=(0.0, False, None), q3=(0.0, False, None))
    rep = build_report(PACK, _scores(), {"model": "m"}, contam)
    by_id = {r["question_id"]: r for r in rep["results"]}
    assert by_id["q0"]["score"] == 1.0
    assert "score_clean" not in by_id["q0"]      # unpaired in the clean comparison
    assert by_id["q1"]["score_clean"] == 0.0     # present and scored
    assert by_id["q0"]["contaminated"] is True
    assert by_id["q0"]["matched_source"] == "sigma-hq"


def test_sensitivity_curve_and_stratification_are_emitted():
    contam = _contam(q0=(0.95, True, "sigma-hq"), q1=(0.6, False, "metasploit"),
                     q2=(0.0, False, None), q3=(0.0, False, None))
    rep = build_report(PACK, _scores(), {"model": "m"}, contam)
    curve = rep["summary"]["sensitivity"]
    assert [row["threshold"] for row in curve] == [1.0, 0.9, 0.8, 0.7, 0.5]
    ns = [row["n"] for row in curve]
    assert ns == sorted(ns, reverse=True), "n must be monotonically non-increasing"
    assert "sigma-hq" in rep["summary"]["by_matched_source"]


def test_decontamination_block_lands_in_metadata():
    contam = _contam(q0=(0.95, True, "sigma-hq"), q1=(0.0, False, None),
                     q2=(0.0, False, None), q3=(0.0, False, None))
    rep = build_report(PACK, _scores(), {"model": "m"}, contam)
    dec = rep["metadata"]["decontamination"]
    assert dec["checked"] is True
    assert dec["threshold"] == 0.80
    assert dec["contaminated_items"] == 1


def test_unchecked_contamination_result_still_reports_null_clean():
    unchecked = ContaminationResult(False, 0.80, {}, 0, 0.0, reason="training set not available")
    rep = build_report(PACK, _scores(), {"model": "m"}, unchecked)
    assert rep["summary"]["score_clean"] is None
    assert rep["summary"]["score_clean_reason"] == "training set not available"
    assert rep["metadata"]["decontamination"]["checked"] is False


def test_report_emits_posture_block_and_per_item_refused():
    from attacklm.bench.posture import PostureScore
    scores = [
        ItemScore("q1", "c", 1.0, "T1071", True),
        ItemScore("q2", "c", None, None, False),
    ]
    posture = [
        PostureScore("q1", "c", "answered", {"refused": 0.0}),
        PostureScore("q2", "c", "refused", {"refused": 1.0}),
    ]
    rep = build_report(PACK, scores, {"model": "m"}, posture=posture)
    assert rep["summary"]["posture"]["refusal_rate"] == 0.5
    assert rep["summary"]["posture"]["n"] == 2
    byid = {r["question_id"]: r for r in rep["results"]}
    assert byid["q2"]["refused"] == 1.0 and byid["q1"]["refused"] == 0.0


def test_report_omits_posture_block_when_not_scored():
    rep = build_report(PACK, _scores(), {"model": "m"})
    assert "posture" not in rep["summary"]
    assert all("refused" not in r for r in rep["results"])
