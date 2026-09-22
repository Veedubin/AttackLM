"""Fine-tuning contamination check for capability benchmarks — the SAE correction.

WHY THIS EXISTS
---------------
We compare a fine-tuned model against the base model it was trained from.
Contamination inherited from *pretraining* cancels out in that comparison —
both sides carry the same base weights — so this module never measures it and
no report may claim anything about it. Overlap between a benchmark item and
the *fine-tuning* set does **not** cancel: it is train-on-test, and it inflates
the delta in the fine-tune's favour. That, and only that, is what is measured
here.

Every report therefore carries BOTH an uncorrected score and a corrected one,
and neither may be quoted without the other. Two further outputs exist because
a single corrected number is too easy to flatter:

* ``sensitivity_curve`` sweeps the Jaccard threshold (1.0 down to 0.5) instead
  of exposing a tunable contamination fraction. A slider can be turned until
  the number looks good and makes runs incomparable; a fixed sweep cannot.
* ``by_matched_source`` stratifies by the training source an item resembles.
  A model acing the items that look like one source while flunking everything
  else is memorising, not generalising — the single mean hides that.

The heavy lifting (character 20-gram MinHash LSH plus exact Jaccard
verification) belongs to ``attacklm-dataset/scripts/decontam.py``, a sibling
repo that is **not** a dependency of this one. It may simply be absent, as may
its own ``datasketch`` dependency. Absence is reported as
``checked=False`` with a reason and is never silently rendered as "corrected,
no change" — see ``sensitivity_curve`` and ``by_matched_source``, which return
None-scored rows and an empty mapping respectively so a caller cannot mistake
an absent correction for a clean bill of health.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from attacklm.bench.items import BenchItem

logger = logging.getLogger(__name__)

# 0.80 over character 20-grams is decontam.py's own default (MAI-Thinking-1
# §2.3.1); using the same number keeps our gate and the dataset repo's
# quarantine pass talking about the same thing.
DEFAULT_THRESHOLD = 0.80

# Swept high-to-low. 1.0 excludes only verbatim duplicates (so its score is
# effectively the uncorrected one) and 0.5 is deliberately aggressive.
DEFAULT_SENSITIVITY = (1.0, 0.9, 0.8, 0.7, 0.5)

#: Bucket key in :func:`by_matched_source` for items that matched nothing.
NO_MATCH = "__no_match__"

#: Env var overriding where ``decontam.py`` is looked for.
SCRIPTS_ENV = "ATTACKLM_DATASET_SCRIPTS"

REASON_NO_TRAINING_SET = "training set not available"
REASON_NO_DECONTAM = "decontam unavailable"

_UNKNOWN_SOURCE = "unknown"
_DECONTAM_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class ItemOverlap:
    """The closest fine-tuning record found for one benchmark item."""

    question_id: str
    max_jaccard: float
    contaminated: bool
    matched_source: str | None


@dataclass(frozen=True)
class ContaminationResult:
    """Outcome of one contamination pass over a pack's items.

    ``checked=False`` means no claim is being made. It is not a clean result,
    and the reporting helpers below keep the two apart by construction.
    """

    checked: bool
    threshold: float
    overlaps: dict[str, ItemOverlap]
    contaminated_items: int
    contamination_rate: float
    reason: str | None = None


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


# A system message is a pack-wide instruction, identical across every item, so
# it carries no information about whether THIS item overlaps the training set.
# Including it is not merely useless, it is actively harmful: Jaccard is a
# ratio over the union, so shared boilerplate inflates the denominator for
# every pair and drives real overlaps below the gate.
#
# Measured on CTI-Bench ATE: the item text is 8,157 chars of which the
# substantive description is 611 (7.5%). A VERBATIM copy of that description in
# the training set scored jaccard 0.0 and went unflagged -- the correction
# would have silently missed genuine train-on-test contamination on both
# flagship packs. Comparing user/assistant content only fixes it.
_COMPARED_ROLES = ("user", "assistant")


def _item_text(item: BenchItem) -> str:
    """The substantive text of an item, for overlap comparison.

    System messages are excluded (see above). Otherwise matches
    ``decontam._record_text``'s treatment of a training record -- non-empty
    contents joined by newline -- so both sides of a Jaccard are built the same
    way.
    """
    parts = [
        str(msg.get("content", ""))
        for msg in item.messages
        if msg.get("content") and msg.get("role") in _COMPARED_ROLES
    ]
    if not parts:
        # An item with only a system message: fall back rather than compare
        # nothing at all, which would silently report "no overlap".
        parts = [str(m.get("content", "")) for m in item.messages if m.get("content")]
    return "\n".join(parts)


def _group_by_source(records: list[dict]) -> dict[str, list[dict]]:
    """Group flat training records by their ``source`` field.

    decontam's index keys records as ``<source>::<index>``, which is how
    ``matched_source`` gets its value.
    """
    grouped: dict[str, list[dict]] = {}
    for record in records:
        source = str(record.get("source") or _UNKNOWN_SOURCE)
        grouped.setdefault(source, []).append(record)
    return grouped


def _overlap_floor(threshold: float) -> float:
    """Lowest similarity worth recording.

    The sweep in :func:`sensitivity_curve` asks about thresholds *below* the
    gate's, so recording matches only at ``threshold`` would make every row of
    the curve identical — a silent, flattering bug. Index down to the lowest
    threshold anyone will ask about.
    """
    floor = min(threshold, *DEFAULT_SENSITIVITY)
    return max(0.01, min(1.0, floor))


# ---------------------------------------------------------------------------
# Defensive import of the sibling repo's decontam.py
# ---------------------------------------------------------------------------


def _decontam_scripts_dir() -> Path:
    """Directory expected to hold ``decontam.py``.

    Defaults to ``<repos>/attacklm-dataset/scripts`` next to this checkout;
    override with ``ATTACKLM_DATASET_SCRIPTS``.
    """
    override = os.environ.get(SCRIPTS_ENV)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[4] / "attacklm-dataset" / "scripts"


def _load_decontam() -> Any | None:
    """Import ``decontam.py`` from the sibling repo, or return None.

    Returns None — never raises — when the repo is absent, when ``datasketch``
    is not installed, or when the module fails to execute. The import is done
    by file location so that the ``ATTACKLM_DATASET_SCRIPTS`` override is
    honoured rather than shadowed by an unrelated ``decontam`` on ``sys.path``.
    """
    scripts_dir = _decontam_scripts_dir()
    module_path = scripts_dir / "decontam.py"
    if not module_path.is_file():
        return None

    key = str(module_path)
    if key in _DECONTAM_CACHE:
        return _DECONTAM_CACHE[key]

    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        spec = importlib.util.spec_from_file_location("attacklm_decontam", module_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ImportError:
        # datasketch is an optional dependency of a repo we do not own.
        logger.debug("decontam.py present but not importable", exc_info=True)
        return None
    except Exception:
        logger.warning("decontam.py failed to load", exc_info=True)
        return None

    _DECONTAM_CACHE[key] = module
    return module


def _minhash_overlaps(
    items: list[BenchItem],
    records_by_source: dict[str, list[dict]],
    ngram_size: int,
    floor: float,
) -> dict[str, tuple[float, str]] | None:
    """One MinHash pass: ``{question_id: (max_jaccard, source)}``.

    Only items reaching ``floor`` appear. Returns None when decontam is
    unavailable. This is the single seam tests replace, which is what keeps
    them free of ``datasketch`` and of the sibling repo.
    """
    decontam = _load_decontam()
    if decontam is None:
        return None

    # decontam._record_text falls back to a top-level "text" field, which is
    # exactly the eval-fixture shape it documents.
    eval_records = [
        {"id": item.question_id, "text": _item_text(item)} for item in items
    ]

    # num_proc=1: this runs inside the queue runner, which must not fork a
    # process pool underneath itself.
    lsh, index_map = decontam.build_training_index(
        records_by_source,
        ngram_size=ngram_size,
        threshold=floor,
        num_proc=1,
    )
    pairs = decontam.find_overlaps(
        lsh,
        index_map,
        eval_records,
        ngram_size=ngram_size,
        threshold=floor,
    )

    best: dict[str, tuple[float, str]] = {}
    for pair in pairs:
        qid = str(pair["eval_id"])
        similarity = float(pair["similarity"])
        current = best.get(qid)
        if current is None or similarity > current[0]:
            best[qid] = (similarity, str(pair["training_source"]))
    return best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _unchecked(threshold: float, reason: str) -> ContaminationResult:
    return ContaminationResult(
        checked=False,
        threshold=threshold,
        overlaps={},
        contaminated_items=0,
        contamination_rate=0.0,
        reason=reason,
    )


def check_contamination(
    items: list[BenchItem],
    training_records: list[dict] | None,
    threshold: float = DEFAULT_THRESHOLD,
    ngram_size: int = 20,
) -> ContaminationResult:
    """Measure each item's closest overlap with the fine-tuning set.

    ``training_records is None`` means the training set could not be read; the
    result says so rather than pretending the pack is clean.
    """
    if training_records is None:
        return _unchecked(threshold, REASON_NO_TRAINING_SET)

    if not items:
        # Nothing to check; an honest empty result with no division by zero.
        return ContaminationResult(
            checked=True,
            threshold=threshold,
            overlaps={},
            contaminated_items=0,
            contamination_rate=0.0,
        )

    try:
        raw = _minhash_overlaps(
            items,
            _group_by_source(training_records),
            ngram_size,
            _overlap_floor(threshold),
        )
    except Exception:
        # A benchmark run must survive anything the sibling repo does.
        logger.warning("contamination check failed; reporting unchecked", exc_info=True)
        raw = None

    if raw is None:
        return _unchecked(threshold, REASON_NO_DECONTAM)

    overlaps: dict[str, ItemOverlap] = {}
    for item in items:
        similarity, source = raw.get(item.question_id, (0.0, None))
        similarity = float(similarity)
        overlaps[item.question_id] = ItemOverlap(
            question_id=item.question_id,
            max_jaccard=similarity,
            contaminated=similarity >= threshold,
            matched_source=source,
        )

    hits = sum(1 for overlap in overlaps.values() if overlap.contaminated)
    total = len(overlaps)
    return ContaminationResult(
        checked=True,
        threshold=threshold,
        overlaps=overlaps,
        contaminated_items=hits,
        contamination_rate=hits / total if total else 0.0,
    )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _max_jaccard(result: ContaminationResult, question_id: str) -> float:
    overlap = result.overlaps.get(question_id)
    return overlap.max_jaccard if overlap is not None else 0.0


def sensitivity_curve(
    result: ContaminationResult,
    scores: dict[str, float],
    thresholds: tuple[float, ...] = DEFAULT_SENSITIVITY,
) -> list[dict]:
    """Score the pack again at each threshold, excluding ``max_jaccard >= t``.

    Rows follow the order given. ``n`` is monotonically non-increasing as the
    threshold falls, because the excluded set only grows. An unchecked result
    yields ``score=None``/``n=0``/``checked=False`` rows: there is no corrected
    score to report, and the absence must stay visible.
    """
    rows: list[dict] = []
    for threshold in thresholds:
        if not result.checked:
            rows.append(
                {"threshold": threshold, "score": None, "n": 0, "checked": False}
            )
            continue
        kept = [
            value
            for question_id, value in scores.items()
            if _max_jaccard(result, question_id) < threshold
        ]
        rows.append(
            {
                "threshold": threshold,
                "score": _mean(kept),
                "n": len(kept),
                "checked": True,
            }
        )
    return rows


def by_matched_source(
    result: ContaminationResult,
    scores: dict[str, float],
) -> dict[str, dict]:
    """Stratify scores by the training source each item most resembles.

    Items that matched nothing land under ``__no_match__``. An unchecked
    result yields ``{}`` — distinguishable from a clean check, which always
    reports at least the ``__no_match__`` bucket.
    """
    if not result.checked:
        return {}

    grouped: dict[str, list[float]] = {}
    for question_id, value in scores.items():
        overlap = result.overlaps.get(question_id)
        source = overlap.matched_source if overlap is not None else None
        grouped.setdefault(source or NO_MATCH, []).append(value)

    return {
        source: {"score": _mean(values), "n": len(values)}
        for source, values in sorted(grouped.items())
    }
