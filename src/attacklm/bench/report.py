"""Build the normalised benchmark report consumed by ``queue compare``.

The ``results[]`` shape is fixed by ``compare.ITEM_METRICS``: each record
carries an id key (``question_id``), a category key (``category``) and the
metrics (``score``, and ``score_clean`` when a contamination check ran).

DUAL SCORING — the "SAE correction"
-----------------------------------
Every benchmark reports two headline numbers, and neither may be quoted
without the other:

``score_raw``
    Every item. The uncorrected "look what we can do" figure.
``score_clean``
    Items overlapping the fine-tuning corpus excluded at the standard
    threshold. The defensible figure.

A raw dyno number means nothing without the standard correction beside it, and
the same holds here. Both come from a SINGLE run: the contamination check is
post-processing over one MinHash pass, so the correction costs no extra GPU
time.

There is deliberately no knob for what FRACTION of contaminated items to
admit. That would make scores incomparable between runs and tunable until the
number flatters — the opposite of a standard correction. The contestable
parameter is where the line sits, and that is published as a whole sensitivity
curve rather than a single pick, so it cannot be cherry-picked.

A contaminated item OMITS ``score_clean`` entirely rather than writing a zero,
so ``compare.pair_items``' existing missing-metric rule counts it unpaired.
Exclusion removes the item from BOTH sides, since pairing is by
``question_id``, leaving the bootstrap valid.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from attacklm.bench.contamination import (
    ContaminationResult,
    by_matched_source,
    sensitivity_curve,
)
from attacklm.bench.packs import Pack
from attacklm.bench.scorers import ItemScore


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _block(scores: list[ItemScore], all_scores: list[ItemScore]) -> dict[str, Any]:
    values = [float(s.score) for s in scores if s.score is not None]
    return {
        "score": _mean(values),
        "n": len(values),
        "invalid": sum(1 for s in all_scores if not s.valid),
    }


def build_report(
    pack: Pack,
    scores: list[ItemScore],
    metadata: dict[str, Any],
    contamination: ContaminationResult | None = None,
    posture: list[Any] | None = None,
) -> dict[str, Any]:
    scored = [s for s in scores if s.score is not None]

    # Posture is orthogonal to capability: the SAME completion is scored for
    # both. It is keyed by question_id so per-item posture metrics land on the
    # same record the capability score does, and pairing downstream stays by id.
    posture_by_id = {p.question_id: p for p in (posture or [])}

    grouped: dict[str, list[float]] = {}
    for s in scored:
        grouped.setdefault(s.category, []).append(float(s.score))  # type: ignore[arg-type]
    by_category = {cat: {"score": _mean(vals), "n": len(vals)} for cat, vals in grouped.items()}

    checked = bool(contamination and contamination.checked)
    overlaps = contamination.overlaps if contamination else {}

    def _is_contaminated(question_id: str) -> bool:
        overlap = overlaps.get(question_id)
        return bool(overlap and overlap.contaminated)

    results: list[dict[str, Any]] = []
    for s in scores:
        record: dict[str, Any] = {
            "question_id": s.question_id,
            "category": s.category,
            "extracted": s.extracted,
            "valid": s.valid,
        }
        if s.score is not None:
            record["score"] = float(s.score)
            # score_clean is omitted -- not zeroed -- for a contaminated item,
            # and omitted entirely when no check ran, so an absent correction
            # can never be mistaken for a correction that found nothing.
            if checked and not _is_contaminated(s.question_id):
                record["score_clean"] = float(s.score)

        if checked:
            overlap = overlaps.get(s.question_id)
            record["contaminated"] = _is_contaminated(s.question_id)
            record["matched_source"] = overlap.matched_source if overlap else None
            record["max_jaccard"] = overlap.max_jaccard if overlap else 0.0

        # Posture metrics ride the same record. Absent when no posture check
        # ran (compare's missing-metric rule then leaves the item unpaired for
        # posture while it still pairs on capability) -- never zeroed.
        ps = posture_by_id.get(s.question_id)
        if ps is not None:
            for key, val in ps.metrics.items():
                record[key] = val
            record["posture"] = ps.label

        results.append(record)

    summary: dict[str, Any] = {
        "score_raw": _block(scored, scores),
        "by_category": by_category,
    }

    if checked and contamination is not None:
        clean = [s for s in scored if not _is_contaminated(s.question_id)]
        score_map = {s.question_id: float(s.score) for s in scored}  # type: ignore[arg-type]
        summary["score_clean"] = _block(clean, scores)
        summary["contamination_rate"] = contamination.contamination_rate
        summary["sensitivity"] = sensitivity_curve(contamination, score_map)
        summary["by_matched_source"] = by_matched_source(contamination, score_map)
    else:
        # Explicitly null, never absent-and-therefore-equal-to-raw.
        summary["score_clean"] = None
        summary["score_clean_reason"] = (
            contamination.reason if contamination else "no contamination check run"
        )

    if posture:
        labels = [p.label for p in posture]
        n = len(labels)
        summary["posture"] = {
            "n": n,
            "refusal_rate": round(labels.count("refused") / n, 4) if n else None,
            "answered_rate": round(labels.count("answered") / n, 4) if n else None,
            "evaded_rate": round(labels.count("evaded") / n, 4) if n else None,
        }

    meta = {
        "pack": pack.name,
        "mode": pack.mode,
        "metric": pack.metric,
        "chance_level": pack.chance_level,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **metadata,
    }
    if contamination is not None:
        meta["decontamination"] = {
            "checked": contamination.checked,
            "threshold": contamination.threshold,
            "contaminated_items": contamination.contaminated_items,
            "contamination_rate": contamination.contamination_rate,
            "reason": contamination.reason,
        }

    return {"metadata": meta, "summary": summary, "results": results}
