"""Escalation ladder decision logic for benchmark sample sizing.

After each rung we have a paired-bootstrap confidence interval of the delta
between a subject model and a baseline. This module decides whether to escalate
to a larger sample (more GPU spend) based on whether more data could resolve
the verdict.

Rungs typically form a sequence like [100, 200, 400, 800, None] where None
means "the full set". The decision rules apply in strict order: once a stop
condition is met, escalation does not happen.
"""

from __future__ import annotations

from dataclasses import dataclass

from attacklm.queue.stats import Interval


@dataclass(frozen=True)
class LadderDecision:
    """Outcome of an escalation decision at one rung."""

    escalate: bool
    next_rung: int | None
    reason: str
    rung_index: int


def decide_next_rung(
    ladder: list[int | None],
    current_rung: int | None,
    interval: Interval,
    subject_ci_hi: float | None = None,
    chance_level: float = 0.25,
    ci_tight_width: float = 0.05,
    floor_margin: float = 0.05,
) -> LadderDecision:
    """Decide whether to escalate to the next rung of the ladder.

    Rules are applied in this strict order; the first match determines outcome.

    1. stop_no_data: interval.n == 0
    2. stop_decisive: interval.lo > 0 or interval.hi < 0 (CI excludes zero)
    3. stop_tight: (interval.hi - interval.lo) <= ci_tight_width AND not degenerate
    4. stop_floor: subject_ci_hi is not None and subject_ci_hi < (chance_level + floor_margin)
    5. stop_exhausted: current_rung is the LAST entry in ladder
    6. escalate: otherwise, move to next_rung

    Args:
        ladder: sequence of sample sizes (ints) or None for full set; must be non-empty
        current_rung: the sample size just completed; must be in ladder
        interval: paired-bootstrap CI of delta (lo, hi, delta, n)
        subject_ci_hi: upper bound of model CI (optional, for floor rule)
        chance_level: baseline performance (typically 0.25 for 4-way MC)
        ci_tight_width: max width to declare a tight CI
        floor_margin: margin above chance_level to declare model dead

    Returns:
        LadderDecision with escalate, next_rung, reason, and rung_index

    Raises:
        ValueError: if ladder is empty or current_rung not in ladder
    """
    if not ladder:
        raise ValueError("ladder must not be empty")

    try:
        rung_index = ladder.index(current_rung)
    except ValueError:
        raise ValueError(f"current_rung {current_rung!r} not in ladder {ladder}")

    # Rule: no data
    if interval.n == 0:
        return LadderDecision(
            escalate=False,
            next_rung=None,
            reason="stop_no_data",
            rung_index=rung_index,
        )

    # Rule: decisive (CI excludes zero)
    if interval.lo > 0.0 or interval.hi < 0.0:
        return LadderDecision(
            escalate=False,
            next_rung=None,
            reason="stop_decisive",
            rung_index=rung_index,
        )

    # Rule: tight CI (confident same), but not degenerate (lo == hi is artifact)
    if interval.lo != interval.hi and (interval.hi - interval.lo) <= ci_tight_width:
        return LadderDecision(
            escalate=False,
            next_rung=None,
            reason="stop_tight",
            rung_index=rung_index,
        )

    # Rule: floor (dead horse -- model indistinguishable from guessing)
    if subject_ci_hi is not None and subject_ci_hi < (chance_level + floor_margin):
        return LadderDecision(
            escalate=False,
            next_rung=None,
            reason="stop_floor",
            rung_index=rung_index,
        )

    # Rule: exhausted (already at the last rung, including None)
    if rung_index == len(ladder) - 1:
        return LadderDecision(
            escalate=False,
            next_rung=None,
            reason="stop_exhausted",
            rung_index=rung_index,
        )

    # Default: escalate to next rung
    next_rung = ladder[rung_index + 1]
    return LadderDecision(
        escalate=True,
        next_rung=next_rung,
        reason="escalate",
        rung_index=rung_index,
    )
