"""Benchmark items and deterministic, nested rung sampling."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchItem:
    """One graded question.

    Mirrors the shape the existing holdouts already use
    (``data/bench/prompt_injection_holdout.jsonl``) so a reader of this repo
    learns nothing new. ``ground_truth["type"]`` selects the scorer.
    """

    question_id: str
    category: str
    tier: str
    messages: list[dict[str, str]]
    ground_truth: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


def load_items(path: Path) -> list[BenchItem]:
    """Read a JSONL item file.

    Duplicate ``question_id``s are rejected: every comparison downstream pairs
    by id, so a duplicate would silently pair the wrong rows.
    """
    items: list[BenchItem] = []
    seen: set[str] = set()

    for lineno, line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)

        qid = rec["question_id"]
        if qid in seen:
            raise ValueError(f"{path}:{lineno}: duplicate question_id {qid!r}")
        seen.add(qid)

        items.append(
            BenchItem(
                question_id=qid,
                category=rec.get("category", "uncategorised"),
                tier=rec.get("tier", ""),
                messages=rec["messages"],
                ground_truth=rec["ground_truth"],
                metadata=rec.get("metadata", {}),
            )
        )
    return items


def sample_items(
    items: list[BenchItem], rung: int | None, seed: int = 42
) -> list[BenchItem]:
    """Take the first ``rung`` items of one seeded shuffle.

    Shuffling once and slicing — rather than sampling independently per rung —
    is what makes rungs *nested*: rung 200 is rung 100 plus 100 more. The
    escalation ladder relies on this so climbing a rung reuses the previous
    rung's work, and so both sides of an A/B comparison always see an identical
    item set.
    """
    ordered = list(items)
    random.Random(seed).shuffle(ordered)
    if rung is None:
        return ordered
    return ordered[:rung]
