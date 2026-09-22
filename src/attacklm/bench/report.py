"""Build the normalised benchmark report consumed by ``queue compare``.

The ``results[]`` shape is fixed by ``compare.ITEM_METRICS``: each record
carries an id key (``question_id``), a category key (``category``) and the
metric (``score``). Phase 2 adds ``score_clean``, the contamination sensitivity
curve and ``by_matched_source`` as *additional* keys, so nothing here changes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from attacklm.bench.packs import Pack
from attacklm.bench.scorers import ItemScore


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def build_report(
    pack: Pack, scores: list[ItemScore], metadata: dict[str, Any]
) -> dict[str, Any]:
    scored = [s for s in scores if s.score is not None]
    values = [float(s.score) for s in scored]  # type: ignore[arg-type]

    grouped: dict[str, list[float]] = {}
    for s in scored:
        grouped.setdefault(s.category, []).append(float(s.score))  # type: ignore[arg-type]
    by_category = {cat: {"score": _mean(vals), "n": len(vals)} for cat, vals in grouped.items()}

    results: list[dict[str, Any]] = []
    for s in scores:
        record: dict[str, Any] = {
            "question_id": s.question_id,
            "category": s.category,
            "extracted": s.extracted,
            "valid": s.valid,
        }
        # A None score OMITS the key. compare.pair_items treats a missing
        # metric as unpaired rather than scoring it as a real 0.0 -- the same
        # rule that keeps error sentinels out of a comparison (v0.20.0 I1/I6).
        if s.score is not None:
            record["score"] = float(s.score)
        results.append(record)

    meta = {
        "pack": pack.name,
        "mode": pack.mode,
        "metric": pack.metric,
        "chance_level": pack.chance_level,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **metadata,
    }

    return {
        "metadata": meta,
        "summary": {
            "score_raw": {
                "score": _mean(values),
                "n": len(scored),
                "invalid": sum(1 for s in scores if not s.valid),
            },
            "by_category": by_category,
        },
        "results": results,
    }
