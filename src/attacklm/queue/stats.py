"""Statistics for `attacklm queue compare` — paired bootstrap and verdicts.

Pure Python on purpose: the queue must work without numpy/torch installed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Interval:
    """Mean paired difference (b - a) with a percentile confidence interval."""

    delta: float
    lo: float
    hi: float
    n: int


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    resamples: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> Interval:
    """95 % percentile bootstrap CI of mean(b - a) over paired items.

    Items are resampled with replacement as pairs, so per-item noise that
    cancels between the two sides does not widen the interval.
    """
    if len(a) != len(b):
        raise ValueError(f"paired series differ in length: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        return Interval(0.0, 0.0, 0.0, 0)
    diffs = [float(y) - float(x) for x, y in zip(a, b)]
    delta = _mean(diffs)
    if n == 1:
        return Interval(delta, delta, delta, 1)
    rng = random.Random(seed)
    means = sorted(
        _mean([diffs[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples)
    )
    alpha = (1.0 - ci) / 2.0
    lo = means[int(alpha * (resamples - 1))]
    hi = means[int((1.0 - alpha) * (resamples - 1))]
    return Interval(delta, lo, hi, n)


def verdict(iv: Interval, higher_is_worse: bool = True) -> str:
    """WORSE / BETTER when the CI excludes zero on that side, else SAME."""
    if iv.n < 2:
        return "SAME"
    if iv.lo > 0.0:
        return "WORSE" if higher_is_worse else "BETTER"
    if iv.hi < 0.0:
        return "BETTER" if higher_is_worse else "WORSE"
    return "SAME"
