"""Capability scoring for ``local_scored`` packs.

Packs whose harness already implements the benchmark (``harness_scored``, e.g.
``inspect_evals/cybermetric_500`` with its own ``choice()`` scorer) do not come
through here — normalising their scores is the point of that mode.

``mcq_choice`` (Phase 1) and ``attack_technique_set`` (Phase 2, backing
CTI-Bench's ATE task) are implemented; ``score_item`` dispatches on
``ground_truth["type"]``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from attacklm.bench.items import BenchItem

INVALID_POLICIES = ("count_wrong", "exclude", "fail_run")
SUBTECHNIQUE_POLICIES = ("strip", "keep", "either")

# Ordered most-explicit first: an <xml>D</xml> or "Answer: D" beats a stray
# capital letter elsewhere in the prose.
_MCQ_PATTERNS = (
    r"<xml>\s*([A-Da-d])\s*</xml>",
    r"(?:answer|option|choice)\s*(?:is|:|=)\s*\(?([A-Da-d])\)?",
    r"^\s*\(?([A-Da-d])\)?\s*[.):]?\s*$",
)
# Fallback sweep, used to detect ambiguity rather than to guess.
_ANY_LETTER = r"(?<![A-Za-z])([A-D])(?![A-Za-z])"

# T1059 (technique) or T1059.003 (sub-technique), as they appear in prose,
# markdown bold, parentheses or comma-separated lists. The lookbehind and the
# trailing (?!\d) keep the match off longer digit runs: neither T105 nor
# T10593 is a technique id, and neither must be allowed to yield "T1059".
_ATTACK_ID = re.compile(r"(?<![A-Za-z0-9])[Tt]\d{4}(?:\.\d{3})?(?!\d)")
_ATTACK_ID_EXACT = re.compile(r"\AT\d{4}(?:\.\d{3})?\Z")


class InvalidResponseError(RuntimeError):
    """An unparseable response met under ``invalid_policy="fail_run"``."""


@dataclass(frozen=True)
class ScoreConfig:
    """Extraction and scoring knobs.

    ``invalid_policy`` defaults to ``count_wrong`` deliberately: it is the
    conservative reading (an unparseable answer is not a correct answer) and it
    cannot manufacture the 100%-accuracy artefact that ``exclude`` produced for
    CTI-Bench in the QCRI pipeline audit.

    ``attack_id_subtechniques`` is a methodological choice, not a detail, so it
    is an explicit knob. CTI-ATE's own task instructions tell the model to
    report techniques *excluding* sub-technique ids, yet real threat-report
    narratives cite sub-techniques freely, so the default ``strip`` normalises
    both sides to the parent technique. ``keep`` compares ids as written;
    ``either`` treats a parent/child pair as a match.
    """

    answer_regex: str | None = None
    invalid_policy: str = "count_wrong"
    attack_id_subtechniques: str = "strip"


@dataclass(frozen=True)
class ItemScore:
    question_id: str
    category: str
    score: float | None
    extracted: str | None
    valid: bool


def _extract_mcq(completion: str, answer_regex: str | None) -> str | None:
    if answer_regex:
        m = re.search(answer_regex, completion)
        return m.group(1).upper() if m else None

    for pattern in _MCQ_PATTERNS:
        m = re.search(pattern, completion, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()

    # Nothing explicit. A single lone letter is an answer; several distinct
    # ones are ambiguous, and guessing the first would invent a score.
    found = {m.group(1) for m in re.finditer(_ANY_LETTER, completion.upper())}
    if len(found) == 1:
        return found.pop()
    return None


def _invalid(item: BenchItem, cfg: ScoreConfig) -> ItemScore:
    if cfg.invalid_policy == "fail_run":
        raise InvalidResponseError(f"{item.question_id}: no answer could be extracted")
    # `exclude` omits the score entirely, so report.py drops the key and
    # compare.pair_items counts the item unpaired instead of scoring it 0.0.
    score = None if cfg.invalid_policy == "exclude" else 0.0
    return ItemScore(item.question_id, item.category, score, None, False)


def _score_mcq_choice(item: BenchItem, completion: str, cfg: ScoreConfig) -> ItemScore:
    extracted = _extract_mcq(completion, cfg.answer_regex)
    if extracted is None:
        return _invalid(item, cfg)

    acceptable = item.ground_truth.get("acceptable") or [item.ground_truth["answer"]]
    acceptable = [str(a).upper() for a in acceptable]
    return ItemScore(
        item.question_id,
        item.category,
        1.0 if extracted in acceptable else 0.0,
        extracted,
        True,
    )


def _parent_technique(technique_id: str) -> str:
    """``T1059.003`` -> ``T1059``; a bare technique id is its own parent."""
    return technique_id.split(".", 1)[0]


def _extract_attack_ids(completion: str) -> set[str]:
    """Every technique id in the free text, upper-cased and de-duplicated."""
    return {m.group(0).upper() for m in _ATTACK_ID.finditer(completion)}


def _gold_attack_ids(item: BenchItem) -> set[str]:
    # Key precedence matches _score_mcq_choice above -- `acceptable` wins when
    # present, else `answer`. One convention across the file: a reader who
    # learns it from one scorer must not be surprised by the other.
    raw = item.ground_truth.get("acceptable")
    if raw is None:
        raw = item.ground_truth.get("answer")
    if isinstance(raw, str):
        raw = [raw]

    gold: set[str] = set()
    for entry in raw or []:
        technique_id = str(entry).strip().upper()
        if not _ATTACK_ID_EXACT.match(technique_id):
            raise ValueError(
                f"{item.question_id}: malformed technique id {entry!r} in the answer key"
            )
        gold.add(technique_id)

    # An item with no correct answer is a broken item, not a zero score: every
    # prediction would be a false positive and the f1 would be a fact about the
    # dataset rather than about the model.
    if not gold:
        raise ValueError(
            f"{item.question_id}: attack_technique_set answer key is empty"
        )
    return gold


def _either_matches(predicted: str, gold: str) -> bool:
    """The ``either`` relation: equal ids, or one is the other's parent.

    Siblings do not match — ``T1059.001`` and ``T1059.003`` are different
    answers, and only their shared parent makes them look related.
    """
    return (
        predicted == gold
        or _parent_technique(predicted) == gold
        or predicted == _parent_technique(gold)
    )


def _micro_f1(predicted: set[str], gold: set[str], policy: str) -> float:
    if policy == "either":
        # The relation is many-to-one in both directions: {T1059.001,
        # T1059.003} both hit a gold T1059, and a predicted T1059 hits every
        # gold sub-technique of T1059. Counting matched *pairs* would let one
        # id inflate the numerator past the size of its own side, so each side
        # is counted once per id instead — precision asks how many predictions
        # hit something, recall how many gold ids were hit. (The alternative,
        # a one-to-one maximum bipartite matching, is stricter: it would score
        # the first example 0.5 precision rather than 1.0.) Under `strip` and
        # `keep` the relation is plain equality, so the two counts collapse
        # back into the single |P ∩ G| of the spec.
        hits_p = sum(1 for p in predicted if any(_either_matches(p, g) for g in gold))
        hits_g = sum(1 for g in gold if any(_either_matches(p, g) for p in predicted))
    else:
        hits_p = hits_g = len(predicted & gold)

    precision = hits_p / len(predicted) if predicted else 0.0
    recall = hits_g / len(gold) if gold else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _score_attack_technique_set(
    item: BenchItem, completion: str, cfg: ScoreConfig
) -> ItemScore:
    if cfg.attack_id_subtechniques not in SUBTECHNIQUE_POLICIES:
        raise ValueError(
            f"unknown attack_id_subtechniques {cfg.attack_id_subtechniques!r}; "
            f"expected one of {SUBTECHNIQUE_POLICIES}"
        )

    # The answer key is validated before the completion is judged: a broken
    # item is a broken item whatever the model said.
    gold = _gold_attack_ids(item)

    predicted = _extract_attack_ids(completion)
    if not predicted:
        # Unparseable, not wrong: invalid_policy decides what that costs.
        return _invalid(item, cfg)

    if cfg.attack_id_subtechniques == "strip":
        predicted = {_parent_technique(t) for t in predicted}
        gold = {_parent_technique(t) for t in gold}

    return ItemScore(
        item.question_id,
        item.category,
        _micro_f1(predicted, gold, cfg.attack_id_subtechniques),
        ",".join(sorted(predicted)),
        True,
    )


_SCORERS = {
    "mcq_choice": _score_mcq_choice,
    "attack_technique_set": _score_attack_technique_set,
}


def score_item(item: BenchItem, completion: str, cfg: ScoreConfig) -> ItemScore:
    if cfg.invalid_policy not in INVALID_POLICIES:
        raise ValueError(
            f"unknown invalid_policy {cfg.invalid_policy!r}; expected one of {INVALID_POLICIES}"
        )

    gt_type = item.ground_truth.get("type")
    scorer = _SCORERS.get(gt_type)
    if scorer is None:
        raise ValueError(
            f"unsupported ground_truth.type {gt_type!r} "
            f"(supported: {tuple(_SCORERS)})"
        )
    return scorer(item, completion, cfg)
