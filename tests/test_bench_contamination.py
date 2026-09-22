#!/usr/bin/env python3
"""Tests for src/attacklm/bench/contamination.py — the SAE correction.

Hermetic by construction: nothing here needs ``datasketch`` or the sibling
``attacklm-dataset`` repo on disk. The MinHash seam (``_minhash_overlaps``) is
monkeypatched with an exact-Jaccard reference implementation, so these tests
exercise *our* thresholding, curve and stratification logic rather than
someone else's LSH recall. The one test that touches the real ``decontam.py``
skips when it cannot be imported.
"""

from __future__ import annotations

import pytest

from attacklm.bench import contamination as contam
from attacklm.bench.contamination import (
    DEFAULT_SENSITIVITY,
    DEFAULT_THRESHOLD,
    NO_MATCH,
    ContaminationResult,
    ItemOverlap,
    by_matched_source,
    check_contamination,
    sensitivity_curve,
)
from attacklm.bench.items import BenchItem

# Long enough that 20-char n-grams are meaningful for the real decontam path.
TEXT_A = (
    "Which ATT&CK technique does 'schtasks /create /sc onlogon' correspond to "
    "when observed on a Windows host with no change ticket?"
)
TEXT_B = (
    "A Linux auditd record shows execve of /usr/bin/base64 piped to curl. "
    "Name the exfiltration technique and the detection you would write."
)


def _item(qid: str, text: str, category: str = "cat") -> BenchItem:
    return BenchItem(
        question_id=qid,
        category=category,
        tier="t1",
        messages=[
            {"role": "system", "content": "You are a SOC analyst."},
            {"role": "user", "content": text},
        ],
        ground_truth={"type": "mcq", "answer": "A"},
    )


def _record(text: str, source: str) -> dict:
    """A training record as one really looks.

    Deliberately carries no system prompt: a training pair does not share the
    benchmark's instruction boilerplate, and _item_text now excludes system
    messages on the item side for the same reason (see the dilution tests at
    the bottom of this file).
    """
    return {
        "messages": [{"role": "user", "content": text}],
        "source": source,
    }


# --------------------------------------------------------------------------
# Hermetic stand-in for the real MinHash pass.
# Same contract as contamination._minhash_overlaps: returns
# {question_id: (max_jaccard, source)} for matches at or above ``floor``,
# or None when the machinery is unavailable.
# --------------------------------------------------------------------------


def _ngrams(text: str, n: int) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _record_text(rec: dict) -> str:
    return "\n".join(
        m.get("content", "") for m in rec.get("messages", []) if m.get("content")
    )


def _exact_overlaps(items, records_by_source, ngram_size, floor):
    best: dict[str, tuple[float, str]] = {}
    for item in items:
        item_ngrams = _ngrams(contam._item_text(item), ngram_size)
        top, top_source = 0.0, None
        for source, records in records_by_source.items():
            for rec in records:
                sim = _jaccard(item_ngrams, _ngrams(_record_text(rec), ngram_size))
                if sim > top:
                    top, top_source = sim, source
        if top_source is not None and top >= floor:
            best[item.question_id] = (top, top_source)
    return best


@pytest.fixture
def exact(monkeypatch):
    """Swap the LSH-backed seam for an exact, dependency-free equivalent."""
    monkeypatch.setattr(contam, "_minhash_overlaps", _exact_overlaps)


def _result(pairs: dict, threshold: float = DEFAULT_THRESHOLD) -> ContaminationResult:
    overlaps = {
        qid: ItemOverlap(qid, sim, sim >= threshold, source)
        for qid, (sim, source) in pairs.items()
    }
    hits = sum(1 for o in overlaps.values() if o.contaminated)
    total = len(overlaps)
    return ContaminationResult(
        checked=True,
        threshold=threshold,
        overlaps=overlaps,
        contaminated_items=hits,
        contamination_rate=hits / total if total else 0.0,
    )


# --------------------------------------------------------------------------
# Graceful absence
# --------------------------------------------------------------------------


def test_no_training_records_is_unchecked_not_clean():
    result = check_contamination([_item("q0", TEXT_A)], None)
    assert result.checked is False
    assert result.reason == "training set not available"
    assert result.overlaps == {}
    assert result.contaminated_items == 0
    assert result.contamination_rate == 0.0
    assert result.threshold == DEFAULT_THRESHOLD


def test_missing_decontam_is_unchecked(monkeypatch):
    monkeypatch.setattr(contam, "_minhash_overlaps", lambda *a, **k: None)
    result = check_contamination([_item("q0", TEXT_A)], [_record(TEXT_B, "alpha")])
    assert result.checked is False
    assert result.reason == "decontam unavailable"
    assert result.overlaps == {}


def test_bogus_scripts_dir_never_raises(monkeypatch, tmp_path):
    monkeypatch.setenv(contam.SCRIPTS_ENV, str(tmp_path / "nowhere"))
    assert contam._load_decontam() is None
    result = check_contamination([_item("q0", TEXT_A)], [_record(TEXT_B, "alpha")])
    assert result.checked is False
    assert result.reason == "decontam unavailable"


def test_a_raising_seam_degrades_to_unchecked(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("datasketch exploded mid-pass")

    monkeypatch.setattr(contam, "_minhash_overlaps", boom)
    result = check_contamination([_item("q0", TEXT_A)], [_record(TEXT_B, "alpha")])
    assert result.checked is False
    assert result.reason == "decontam unavailable"


# --------------------------------------------------------------------------
# Overlap detection
# --------------------------------------------------------------------------


def test_item_text_concatenates_message_contents():
    item = _item("q0", "second line")
    # The system message is excluded on purpose: shared boilerplate inflates
    # the Jaccard denominator and buries real overlaps (see the dilution tests).
    assert contam._item_text(item) == "second line"


def test_no_overlap_gives_zero_jaccard(exact):
    items = [_item("q0", TEXT_A), _item("q1", TEXT_B)]
    result = check_contamination(items, [_record("wholly unrelated prose", "alpha")])
    assert result.checked is True
    assert set(result.overlaps) == {"q0", "q1"}
    for overlap in result.overlaps.values():
        assert overlap.max_jaccard == 0.0
        assert overlap.contaminated is False
        assert overlap.matched_source is None
    assert result.contaminated_items == 0
    assert result.contamination_rate == 0.0


def test_exact_duplicate_is_contaminated(exact):
    items = [_item("dup", TEXT_A), _item("clean", TEXT_B)]
    records = [_record(TEXT_A, "sigma-hq"), _record("unrelated prose", "atomic")]
    result = check_contamination(items, records)
    assert result.overlaps["dup"].max_jaccard == pytest.approx(1.0)
    assert result.overlaps["dup"].contaminated is True
    assert result.overlaps["dup"].matched_source == "sigma-hq"
    assert result.overlaps["clean"].contaminated is False
    assert result.contaminated_items == 1
    assert result.contamination_rate == pytest.approx(0.5)


def test_threshold_boundary_is_inclusive(monkeypatch):
    monkeypatch.setattr(
        contam,
        "_minhash_overlaps",
        lambda *a, **k: {"at": (0.80, "alpha"), "below": (0.7999, "alpha")},
    )
    items = [_item("at", TEXT_A), _item("below", TEXT_B)]
    result = check_contamination(items, [_record(TEXT_A, "alpha")], threshold=0.80)
    assert result.overlaps["at"].contaminated is True
    assert result.overlaps["below"].contaminated is False
    assert result.contaminated_items == 1


def test_empty_items_does_not_divide_by_zero(exact):
    result = check_contamination([], [_record(TEXT_A, "alpha")])
    assert result.checked is True
    assert result.overlaps == {}
    assert result.contamination_rate == 0.0
    assert sensitivity_curve(result, {}) == [
        {"threshold": t, "score": None, "n": 0, "checked": True}
        for t in DEFAULT_SENSITIVITY
    ]
    assert by_matched_source(result, {}) == {}


# --------------------------------------------------------------------------
# Sensitivity curve
# --------------------------------------------------------------------------


SPREAD = {
    "q0": (0.00, None),
    "q1": (0.55, "alpha"),
    "q2": (0.72, "alpha"),
    "q3": (0.85, "beta"),
    "q4": (1.00, "beta"),
}
SPREAD_SCORES = {"q0": 0.0, "q1": 0.25, "q2": 0.5, "q3": 1.0, "q4": 1.0}


def test_sensitivity_curve_is_monotonically_non_increasing_in_n():
    curve = sensitivity_curve(_result(SPREAD), SPREAD_SCORES)
    assert [row["threshold"] for row in curve] == list(DEFAULT_SENSITIVITY)
    counts = [row["n"] for row in curve]
    assert counts == sorted(counts, reverse=True)
    assert all(a >= b for a, b in zip(counts, counts[1:]))
    assert counts == [4, 4, 3, 2, 1]


def test_threshold_one_keeps_everything_but_exact_duplicates():
    curve = sensitivity_curve(_result(SPREAD), SPREAD_SCORES)
    top = curve[0]
    assert top["threshold"] == 1.0
    assert top["n"] == 4  # only q4, an exact duplicate, drops out
    assert top["score"] == pytest.approx(0.4375)


def test_curve_excludes_at_or_above_each_threshold():
    curve = {row["threshold"]: row for row in sensitivity_curve(_result(SPREAD), SPREAD_SCORES)}
    # t=0.7 keeps q0 and q1 only.
    assert curve[0.7]["n"] == 2
    assert curve[0.7]["score"] == pytest.approx(0.125)
    # t=0.5 keeps q0 only — the aggressive end of the sweep.
    assert curve[0.5]["n"] == 1
    assert curve[0.5]["score"] == pytest.approx(0.0)


def test_items_absent_from_overlaps_are_never_excluded():
    result = _result({"q0": (0.95, "alpha")})
    curve = sensitivity_curve(result, {"q0": 1.0, "unseen": 0.0}, thresholds=(0.5,))
    assert curve[0]["n"] == 1
    assert curve[0]["score"] == pytest.approx(0.0)


def test_unchecked_curve_is_distinguishable_from_a_clean_one(exact):
    items = [_item("q0", TEXT_A), _item("q1", TEXT_B)]
    scores = {"q0": 1.0, "q1": 0.0}

    unchecked = sensitivity_curve(check_contamination(items, None), scores)
    clean = sensitivity_curve(
        check_contamination(items, [_record("wholly unrelated prose", "alpha")]), scores
    )

    assert all(row["checked"] is False for row in unchecked)
    assert all(row["score"] is None and row["n"] == 0 for row in unchecked)
    assert all(row["checked"] is True for row in clean)
    assert all(row["n"] == 2 and row["score"] == pytest.approx(0.5) for row in clean)
    assert unchecked != clean


# --------------------------------------------------------------------------
# Stratification by matched training source
# --------------------------------------------------------------------------


def test_by_matched_source_groups_and_labels_non_matches(exact):
    items = [_item("q0", TEXT_A), _item("q1", TEXT_B), _item("q2", "unrelated prose")]
    records = [_record(TEXT_A, "sigma-hq"), _record(TEXT_B, "sigma-hq")]
    result = check_contamination(items, records)
    strata = by_matched_source(result, {"q0": 1.0, "q1": 0.0, "q2": 0.25})
    assert strata["sigma-hq"] == {"score": pytest.approx(0.5), "n": 2}
    assert strata[NO_MATCH] == {"score": pytest.approx(0.25), "n": 1}
    assert NO_MATCH == "__no_match__"


def test_by_matched_source_separates_two_sources():
    strata = by_matched_source(_result(SPREAD), SPREAD_SCORES)
    assert strata["alpha"] == {"score": pytest.approx(0.375), "n": 2}
    assert strata["beta"] == {"score": pytest.approx(1.0), "n": 2}
    assert strata[NO_MATCH] == {"score": pytest.approx(0.0), "n": 1}


def test_unchecked_stratification_is_empty_not_clean():
    unchecked = check_contamination([_item("q0", TEXT_A)], None)
    assert by_matched_source(unchecked, {"q0": 1.0}) == {}
    # A genuinely clean check reports the __no_match__ bucket instead, so the
    # two can never be confused for one another.
    clean = _result({"q0": (0.0, None)})
    assert by_matched_source(clean, {"q0": 1.0}) == {NO_MATCH: {"score": 1.0, "n": 1}}


# --------------------------------------------------------------------------
# The real sibling-repo path (skipped when it is not installed)
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    contam._load_decontam() is None,
    reason="attacklm-dataset/scripts/decontam.py (or datasketch) not available",
)
def test_real_decontam_finds_an_exact_duplicate():
    items = [_item("dup", TEXT_A), _item("clean", TEXT_B)]
    result = check_contamination(items, [_record(TEXT_A, "sigma-hq")])
    assert result.checked is True
    assert result.reason is None
    assert result.overlaps["dup"].max_jaccard == pytest.approx(1.0)
    assert result.overlaps["dup"].contaminated is True
    assert result.overlaps["dup"].matched_source == "sigma-hq"
    assert result.overlaps["clean"].max_jaccard < DEFAULT_THRESHOLD
    assert result.contaminated_items == 1


# --------------------------------------------------------------------------
# boilerplate dilution -- the bug that made the correction silently useless
# --------------------------------------------------------------------------


def test_shared_system_prompt_is_excluded_from_comparison():
    """A long pack-wide instruction must not dilute a real overlap.

    Jaccard is a ratio over the union, so boilerplate shared by every item
    inflates the denominator for every pair. Measured on real CTI-Bench ATE:
    the item text was 8,157 chars of which the substantive description was 611
    (7.5%), and a VERBATIM copy of that description in the training set scored
    jaccard 0.0 -- entirely unflagged.
    """
    from attacklm.bench.contamination import _item_text
    from attacklm.bench.items import BenchItem

    boilerplate = "You are a CTI analyst. Follow these instructions. " * 40
    substance = "3PARA RAT communicates over HTTP and encrypts its traffic with a custom cipher."

    item = BenchItem(
        "q0", "cti", "ate",
        [{"role": "system", "content": boilerplate},
         {"role": "user", "content": substance}],
        {"type": "attack_technique_set", "answer": ["T1071"]}, {},
    )
    text = _item_text(item)
    assert substance in text
    assert boilerplate not in text, "system boilerplate must not be compared"


def test_verbatim_overlap_under_heavy_boilerplate_is_still_caught():
    from attacklm.bench.items import BenchItem

    boilerplate = "You are a CTI analyst. Follow these instructions. " * 40
    substance = "3PARA RAT communicates over HTTP and encrypts its traffic with a custom cipher."

    items = [
        BenchItem("q0", "cti", "ate",
                  [{"role": "system", "content": boilerplate},
                   {"role": "user", "content": substance}],
                  {"type": "attack_technique_set", "answer": ["T1071"]}, {}),
        BenchItem("q1", "cti", "ate",
                  [{"role": "system", "content": boilerplate},
                   {"role": "user", "content": "Something entirely unrelated to that."}],
                  {"type": "attack_technique_set", "answer": ["T1059"]}, {}),
    ]
    records = [{"messages": [{"role": "assistant", "content": substance}],
                "source": "sigma-hq"}]

    result = check_contamination(items, records)
    if not result.checked:
        pytest.skip("decontam/datasketch unavailable")
    assert result.overlaps["q0"].contaminated is True
    assert result.overlaps["q0"].max_jaccard > 0.9
    assert result.overlaps["q1"].contaminated is False


def test_system_only_item_falls_back_rather_than_comparing_nothing():
    from attacklm.bench.contamination import _item_text
    from attacklm.bench.items import BenchItem

    item = BenchItem("q0", "c", "t", [{"role": "system", "content": "only this"}], {}, {})
    assert _item_text(item) == "only this"
