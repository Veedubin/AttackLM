#!/usr/bin/env python3
"""Tests for src/attacklm/bench/scorers.py — the local_scored capability scorer."""

from __future__ import annotations

import pytest

from attacklm.bench.items import BenchItem
from attacklm.bench.scorers import (
    INVALID_POLICIES,
    InvalidResponseError,
    ItemScore,
    ScoreConfig,
    score_item,
)


def _mcq(answer="C", acceptable=None):
    gt = {"type": "mcq_choice", "answer": answer}
    if acceptable is not None:
        gt["acceptable"] = acceptable
    return BenchItem(
        question_id="q1",
        category="net",
        tier="mcq",
        messages=[{"role": "user", "content": "?"}],
        ground_truth=gt,
        metadata={},
    )


def test_correct_bare_letter():
    s = score_item(_mcq("C"), "C", ScoreConfig())
    assert isinstance(s, ItemScore)
    assert s.score == 1.0 and s.valid and s.extracted == "C"


def test_wrong_letter():
    s = score_item(_mcq("C"), "A", ScoreConfig())
    assert s.score == 0.0 and s.valid and s.extracted == "A"


def test_answer_in_prose():
    s = score_item(_mcq("C"), "The answer is C because it is the only one that fits.", ScoreConfig())
    assert s.score == 1.0 and s.extracted == "C"


def test_xml_wrapped_answer():
    """CyberMetric's own prompt asks for <xml>D</xml>."""
    s = score_item(_mcq("D"), "<xml>D</xml>", ScoreConfig())
    assert s.score == 1.0 and s.extracted == "D"


def test_lowercase_is_accepted():
    s = score_item(_mcq("B"), "b", ScoreConfig())
    assert s.score == 1.0 and s.extracted == "B"


def test_parenthesised_letter():
    s = score_item(_mcq("A"), "(A)", ScoreConfig())
    assert s.score == 1.0 and s.extracted == "A"


def test_acceptable_list_is_honoured():
    s = score_item(_mcq("C", acceptable=["C", "D"]), "D", ScoreConfig())
    assert s.score == 1.0


def test_no_letter_is_invalid_and_counts_wrong_by_default():
    s = score_item(_mcq("C"), "I cannot help with that.", ScoreConfig())
    assert s.valid is False
    assert s.score == 0.0  # count_wrong is the default
    assert s.extracted is None


def test_ambiguous_multiple_letters_is_invalid():
    """Guessing the first of several offered letters would invent a score."""
    s = score_item(_mcq("C"), "It could be A, or B, or C.", ScoreConfig())
    assert s.valid is False
    assert s.extracted is None


def test_explicit_marker_beats_stray_letters():
    s = score_item(_mcq("C"), "Not A. Not B. Answer: C", ScoreConfig())
    assert s.score == 1.0 and s.extracted == "C"


def test_exclude_policy_yields_none_score():
    s = score_item(_mcq("C"), "no answer here", ScoreConfig(invalid_policy="exclude"))
    assert s.valid is False and s.score is None


def test_fail_run_policy_raises():
    with pytest.raises(InvalidResponseError):
        score_item(_mcq("C"), "no answer here", ScoreConfig(invalid_policy="fail_run"))


def test_custom_answer_regex():
    cfg = ScoreConfig(answer_regex=r"ANSWER=([A-D])")
    s = score_item(_mcq("C"), "reasoning ... ANSWER=C", cfg)
    assert s.score == 1.0 and s.extracted == "C"


def test_custom_answer_regex_that_misses_is_invalid():
    cfg = ScoreConfig(answer_regex=r"ANSWER=([A-D])")
    s = score_item(_mcq("C"), "the answer is C", cfg)
    assert s.valid is False


def test_empty_completion_is_invalid():
    s = score_item(_mcq("C"), "", ScoreConfig())
    assert s.valid is False


def test_unknown_ground_truth_type_raises():
    item = BenchItem(
        "q1", "net", "mcq", [{"role": "user", "content": "?"}], {"type": "not_a_real_type"}, {}
    )
    with pytest.raises(ValueError, match="not_a_real_type"):
        score_item(item, "C", ScoreConfig())


def test_unknown_invalid_policy_raises():
    with pytest.raises(ValueError, match="nonsense"):
        score_item(_mcq("C"), "no letter", ScoreConfig(invalid_policy="nonsense"))


def test_count_wrong_is_the_default_policy():
    """`exclude` produced the 100%-accuracy artefact in the QCRI audit."""
    assert ScoreConfig().invalid_policy == "count_wrong"
    assert set(INVALID_POLICIES) == {"count_wrong", "exclude", "fail_run"}
