#!/usr/bin/env python3
"""Tests for src/attacklm/bench/scorers.py — the local_scored capability scorer."""

from __future__ import annotations

import pytest

from attacklm.bench.items import BenchItem
from attacklm.bench.scorers import (
    INVALID_POLICIES,
    SUBTECHNIQUE_POLICIES,
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


# --- attack_technique_set (CTI-Bench ATE) -----------------------------------
#
# Absolute scores here are low by nature: the published CTI-ATE reference has
# GPT-4 at 0.639 micro-F1 and LLAMA3-8B at 0.156.


def _ate(answer=None, acceptable=None):
    gt = {"type": "attack_technique_set"}
    if answer is not None:
        gt["answer"] = answer
    if acceptable is not None:
        gt["acceptable"] = acceptable
    return BenchItem(
        question_id="a1",
        category="ate",
        tier="ate",
        messages=[{"role": "user", "content": "threat report"}],
        ground_truth=gt,
        metadata={},
    )


def test_attack_perfect_match_scores_one():
    s = score_item(_ate(["T1059", "T1566"]), "We saw T1059 and T1566.", ScoreConfig())
    assert s.score == 1.0 and s.valid


def test_attack_complete_miss_scores_zero():
    s = score_item(_ate(["T1059", "T1566"]), "We saw T1003 and T1190.", ScoreConfig())
    assert s.score == 0.0 and s.valid


def test_attack_partial_overlap_gives_hand_computed_f1():
    """P={T1059,T1566}, G={T1566,T1003}: tp=1, p=0.5, r=0.5, f1=0.5."""
    s = score_item(_ate(["T1566", "T1003"]), "T1059 and T1566 were used.", ScoreConfig())
    assert s.score == pytest.approx(0.5)


def test_attack_ids_in_prose_markdown_and_parentheses():
    completion = "The actor used **T1059** (specifically T1059.001) and then T1566."
    s = score_item(_ate(["T1059", "T1566"]), completion, ScoreConfig())
    assert s.score == 1.0


def test_attack_ids_comma_separated():
    s = score_item(_ate(["T1059", "T1566", "T1003"]), "T1059, T1566.001, T1003", ScoreConfig())
    assert s.score == 1.0


def test_attack_id_lowercase_is_normalised():
    s = score_item(_ate(["T1059"]), "the technique is t1059", ScoreConfig())
    assert s.score == 1.0 and s.extracted == "T1059"


def test_attack_duplicate_ids_do_not_inflate_the_score():
    """Set semantics: repeating an id cannot buy recall it did not earn."""
    gold = ["T1059", "T1566"]
    once = score_item(_ate(gold), "T1059 was used.", ScoreConfig())
    many = score_item(_ate(gold), "T1059. Again T1059, and once more T1059.", ScoreConfig())
    # p=1.0, r=0.5 -> f1 = 2/3, identically for both completions.
    assert once.score == pytest.approx(2 / 3)
    assert many.score == pytest.approx(2 / 3)


def test_attack_three_digit_near_miss_does_not_extract():
    s = score_item(_ate(["T1059"]), "see T105 in the appendix", ScoreConfig())
    assert s.valid is False and s.extracted is None


def test_attack_five_digit_near_miss_does_not_extract():
    """T10593 must not be truncated into a spurious T1059."""
    s = score_item(_ate(["T1059"]), "ticket T10593 tracks this", ScoreConfig())
    assert s.valid is False and s.extracted is None


def test_attack_near_misses_do_not_pollute_a_real_id():
    s = score_item(_ate(["T1059"]), "T105, T10593 and T1059 appear.", ScoreConfig())
    assert s.score == 1.0 and s.extracted == "T1059"


def test_strip_collapses_parent_and_subtechnique_without_double_counting():
    gold = ["T1059"]
    both = score_item(_ate(gold), "T1059 and its child T1059.003", ScoreConfig())
    bare = score_item(_ate(gold), "T1059", ScoreConfig())
    assert both.extracted == "T1059"
    assert both.score == 1.0 and both.score == bare.score


def test_strip_normalises_the_answer_key_too():
    s = score_item(_ate(["T1566.001"]), "phishing, i.e. T1566", ScoreConfig())
    assert s.score == 1.0


def test_keep_policy_distinguishes_subtechnique_from_parent():
    cfg = ScoreConfig(attack_id_subtechniques="keep")
    s = score_item(_ate(["T1059"]), "T1059.003 was used", cfg)
    assert s.score == 0.0 and s.extracted == "T1059.003"


def test_either_policy_matches_parent_and_child_in_both_directions():
    cfg = ScoreConfig(attack_id_subtechniques="either")
    child_pred = score_item(_ate(["T1059"]), "T1059.003", cfg)
    parent_pred = score_item(_ate(["T1059.003"]), "T1059", cfg)
    assert child_pred.score == 1.0 and parent_pred.score == 1.0


def test_either_policy_does_not_match_siblings():
    cfg = ScoreConfig(attack_id_subtechniques="either")
    s = score_item(_ate(["T1059.003"]), "T1059.001", cfg)
    assert s.score == 0.0


def test_either_policy_many_to_one_counts_each_side_once():
    """Two children hitting one gold parent: p=2/2, r=1/1 -> f1=1.0.

    Counting matched pairs instead would give tp=2 against |G|=1.
    """
    cfg = ScoreConfig(attack_id_subtechniques="either")
    s = score_item(_ate(["T1059"]), "T1059.001 and T1059.003", cfg)
    assert s.score == pytest.approx(1.0)


def test_three_policies_give_different_hand_computed_scores():
    """Same input, three answers.

    completion -> {T1059.003, T1059, T1190}; key -> {T1059, T1566.001}.
    strip:  P={T1059,T1190}, G={T1059,T1566} -> tp=1, p=1/2, r=1/2 -> 0.5
    keep:   tp={T1059} -> p=1/3, r=1/2 -> 0.4
    either: p=2/3 (T1059.003 and T1059 hit), r=1/2 -> 4/7
    """
    gold = ["T1059", "T1566.001"]
    completion = "T1059.003, T1059 and T1190 were observed."
    strip = score_item(_ate(gold), completion, ScoreConfig(attack_id_subtechniques="strip"))
    keep = score_item(_ate(gold), completion, ScoreConfig(attack_id_subtechniques="keep"))
    either = score_item(_ate(gold), completion, ScoreConfig(attack_id_subtechniques="either"))
    assert strip.score == pytest.approx(0.5)
    assert keep.score == pytest.approx(0.4)
    assert either.score == pytest.approx(4 / 7)


def test_attack_no_ids_is_invalid_and_counts_wrong_by_default():
    s = score_item(_ate(["T1059"]), "I cannot help with that.", ScoreConfig())
    assert s.valid is False and s.score == 0.0 and s.extracted is None


def test_attack_no_ids_honours_exclude_policy():
    s = score_item(_ate(["T1059"]), "no ids here", ScoreConfig(invalid_policy="exclude"))
    assert s.valid is False and s.score is None


def test_attack_no_ids_honours_fail_run_policy():
    with pytest.raises(InvalidResponseError):
        score_item(_ate(["T1059"]), "no ids here", ScoreConfig(invalid_policy="fail_run"))


def test_attack_empty_answer_key_raises():
    """No correct answer is a broken item, not a zero score."""
    with pytest.raises(ValueError, match="empty"):
        score_item(_ate([]), "T1059", ScoreConfig())


def test_attack_missing_answer_key_raises():
    with pytest.raises(ValueError, match="empty"):
        score_item(_ate(), "T1059", ScoreConfig())


def test_attack_acceptable_is_used_when_answer_is_absent():
    s = score_item(_ate(acceptable=["T1059"]), "T1059", ScoreConfig())
    assert s.score == 1.0


def test_attack_malformed_answer_key_entry_raises():
    with pytest.raises(ValueError, match="T105"):
        score_item(_ate(["T105"]), "T1059", ScoreConfig())


def test_unknown_subtechnique_policy_raises():
    cfg = ScoreConfig(attack_id_subtechniques="nonsense")
    with pytest.raises(ValueError, match="nonsense"):
        score_item(_ate(["T1059"]), "T1059", cfg)


def test_subtechnique_policy_default_and_constant():
    """CTI-ATE tells the model to exclude sub-technique ids, hence `strip`."""
    assert ScoreConfig().attack_id_subtechniques == "strip"
    assert SUBTECHNIQUE_POLICIES == ("strip", "keep", "either")


def test_attack_extracted_field_reports_normalised_ids():
    s = score_item(_ate(["T1059"]), "t1566.001, T1059.003 and T1059", ScoreConfig())
    assert s.extracted == "T1059,T1566"


def test_key_precedence_matches_mcq_across_both_scorers():
    """`acceptable` wins over `answer` in BOTH scorers.

    The two scorers previously disagreed, which is a trap for whoever writes
    the next answer key: the same ground_truth dict would grade differently
    depending on its type.
    """
    mcq = BenchItem(
        "q1", "net", "mcq", [{"role": "user", "content": "?"}],
        {"type": "mcq_choice", "answer": "A", "acceptable": ["C"]}, {},
    )
    assert score_item(mcq, "C", ScoreConfig()).score == 1.0
    assert score_item(mcq, "A", ScoreConfig()).score == 0.0

    tech = BenchItem(
        "q2", "cti", "ate", [{"role": "user", "content": "?"}],
        {"type": "attack_technique_set", "answer": ["T1001"], "acceptable": ["T1059"]}, {},
    )
    assert score_item(tech, "T1059", ScoreConfig()).score == 1.0
    assert score_item(tech, "T1001", ScoreConfig()).score == 0.0


# --------------------------------------------------------------------------
# multiple-SELECT (SecEval is ~43% of these; SecBench has 9 in its EN subset)
# --------------------------------------------------------------------------


def _multi(answer=("A", "C")):
    return BenchItem(
        "q1", "net", "mcq", [{"role": "user", "content": "?"}],
        {"type": "mcq_multi", "answer": list(answer)}, {},
    )


def test_multi_exact_set_match_scores_one():
    assert score_item(_multi(), "Answer: AC", ScoreConfig()).score == 1.0


def test_multi_order_does_not_matter():
    assert score_item(_multi(), "Answer: CA", ScoreConfig()).score == 1.0


def test_multi_adjacent_letters_are_all_extracted():
    """"ABC" must not truncate to "A" -- that scored a correct answer 0.0."""
    s = score_item(_multi(("A", "B", "C")), "Answer: ABC", ScoreConfig())
    assert s.score == 1.0 and s.extracted == "ABC"


def test_multi_connector_word_and_is_not_read_as_option_d():
    """The "d" in "and" is itself a valid option letter."""
    s = score_item(_multi(("A", "C")), "Answer: A, B and C", ScoreConfig())
    assert s.extracted == "ABC"          # B included, D NOT invented
    assert "D" not in s.extracted


def test_multi_partial_overlap_is_zero_under_exact():
    assert score_item(_multi(("A", "C")), "Answer: A", ScoreConfig()).score == 0.0


def test_multi_partial_mode_gives_jaccard():
    cfg = ScoreConfig(multi_select="partial")
    # predicted {A}, gold {A,C} -> 1/2
    assert score_item(_multi(("A", "C")), "Answer: A", cfg).score == 0.5


def test_multi_superset_is_wrong_under_exact():
    assert score_item(_multi(("A", "C")), "Answer: ABC", ScoreConfig()).score == 0.0


def test_multi_refusal_is_invalid():
    s = score_item(_multi(), "I cannot help with that.", ScoreConfig())
    assert s.valid is False


def test_multi_answer_may_be_given_as_a_string_key():
    item = BenchItem("q1", "n", "m", [{"role": "user", "content": "?"}],
                     {"type": "mcq_multi", "answer": "AC"}, {})
    assert score_item(item, "Answer: AC", ScoreConfig()).score == 1.0


def test_multi_empty_key_raises():
    item = BenchItem("q1", "n", "m", [{"role": "user", "content": "?"}],
                     {"type": "mcq_multi", "answer": []}, {})
    with pytest.raises(ValueError, match="empty answer key"):
        score_item(item, "Answer: A", ScoreConfig())


def test_multi_bad_mode_raises():
    with pytest.raises(ValueError, match="nonsense"):
        score_item(_multi(), "Answer: A", ScoreConfig(multi_select="nonsense"))


# --- attack_technique_set: alternate acceptable answer-SETS -----------------
# For items whose vuln->ATT&CK mapping is legitimately non-unique, `acceptable`
# may be a LIST OF LISTS: several acceptable answer-sets, any one of which,
# fully named, scores 1.0. A flat list of id strings keeps its old meaning
# (a single required answer-set).


def test_attack_alternate_answer_sets_score_best_match():
    item = _ate(acceptable=[["T1190"], ["T1059"]])
    assert score_item(item, "This is T1190.", ScoreConfig()).score == 1.0
    assert score_item(item, "This is T1059.", ScoreConfig()).score == 1.0


def test_attack_alternate_sets_reject_a_wrong_technique():
    item = _ate(acceptable=[["T1190"], ["T1059"]])
    assert score_item(item, "This is T1055.", ScoreConfig()).score == 0.0


def test_attack_alternate_set_naming_one_is_not_penalised_for_the_other():
    # Naming T1190 fully must score 1.0 -- NOT recall 0.5 as a union gold
    # {T1190,T1059} would give. That difference is the whole point.
    item = _ate(acceptable=[["T1190"], ["T1059"]])
    assert score_item(item, "Only T1190 here.", ScoreConfig()).score == 1.0


def test_attack_multi_id_alternate_set_still_requires_its_members():
    # An acceptable set with two ids requires both of THAT set to score 1.0,
    # while a different single-id set remains a full-credit alternative.
    item = _ate(acceptable=[["T1059", "T1027"], ["T1204"]])
    assert score_item(item, "T1059 and T1027.", ScoreConfig()).score == 1.0
    assert score_item(item, "T1204.", ScoreConfig()).score == 1.0
    # naming only one of the two-id set scores its F1 (2*.5*.5/... vs the
    # T1204 alternate = 0), max = 0.6667
    assert score_item(item, "T1059 only.", ScoreConfig()).score == pytest.approx(2 / 3)


def test_attack_flat_acceptable_is_still_a_single_required_set():
    # BACKWARD COMPAT: a flat list means one gold set where all are required.
    item = _ate(acceptable=["T1190", "T1059"])
    assert score_item(item, "T1190 and T1059.", ScoreConfig()).score == 1.0
    assert score_item(item, "T1190 only.", ScoreConfig()).score == pytest.approx(2 / 3)


def test_attack_mixed_acceptable_shape_raises():
    item = _ate(acceptable=["T1190", ["T1059"]])
    with pytest.raises(ValueError, match="mix"):
        score_item(item, "T1190", ScoreConfig())


# --- code-review fixes: prose-bleed (#3) and options beyond D (#4) ---

def _mcq_with_options(answer, n_options):
    """A single-choice item whose question actually lists n_options options."""
    opts = "\n".join(f"{chr(65+i)}. option {i}" for i in range(n_options))
    return BenchItem(
        "q1", "net", "mcq",
        [{"role": "user", "content": f"Question?\n\n{opts}"}],
        {"type": "mcq_choice", "answer": answer}, {},
    )


def test_multi_rationale_after_answer_does_not_bleed():
    """'Answer: A' + a rationale starting with a B/C/D word must not add letters."""
    s = score_item(
        _multi(("A",)),
        "Answer: A\n\nBecause the attacker used phishing.",
        ScoreConfig(),
    )
    assert s.extracted == "A" and s.score == 1.0


def test_multi_rationale_same_line_does_not_bleed():
    s = score_item(
        _multi(("A", "C")),
        "Answer: A, C. Because Deep packet inspection was bypassed.",
        ScoreConfig(),
    )
    assert s.extracted == "AC" and s.score == 1.0


def test_multi_supports_options_beyond_d():
    """Options run past D; a correct A/E answer must extract E, not drop it."""
    s = score_item(_multi(("A", "E")), "The answer is A and E.", ScoreConfig())
    assert s.extracted == "AE" and s.score == 1.0


def test_single_choice_supports_option_e():
    """A 5-option single-choice item must be able to extract E."""
    s = score_item(_mcq_with_options("E", 5), "The answer is E.", ScoreConfig())
    assert s.score == 1.0 and s.extracted == "E"


def test_single_choice_stray_e_ignored_on_four_option_item():
    """On a 4-option item, a stray 'E' in prose must not create ambiguity."""
    s = score_item(_mcq_with_options("A", 4), "The answer is A (see E-mail logs).", ScoreConfig())
    assert s.score == 1.0
