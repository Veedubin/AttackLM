"""Statistics for `attacklm queue compare` — paired bootstrap and verdicts.

Pure Python on purpose: the queue must work without numpy/torch installed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

# Below this many paired items, a directional verdict isn't trustworthy even
# when the bootstrap CI happens to exclude zero -- there just isn't enough
# data to distinguish signal from noise.
MIN_PAIRED_N = 5


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
    """WORSE / BETTER when the CI excludes zero on that side, else SAME.

    `n == 0` -> "n/a" (nothing to compare). A degenerate interval
    (`lo == hi`, e.g. every paired diff is identical) is always "SAME"
    regardless of n -- a CI with zero width isn't confident evidence, it's
    an artifact of too little/too-uniform data (this is what let
    `paired_bootstrap([0,0,0],[1,1,1])` report a confident "WORSE" from 3
    items). `0 < n < MIN_PAIRED_N` -> "n<5": too few paired items to trust
    a directional verdict even when the CI happens to exclude zero.
    """
    if iv.n == 0:
        return "n/a"
    if iv.lo == iv.hi:
        return "SAME"
    if iv.n < MIN_PAIRED_N:
        return "n<5"
    if iv.lo > 0.0:
        return "WORSE" if higher_is_worse else "BETTER"
    if iv.hi < 0.0:
        return "BETTER" if higher_is_worse else "WORSE"
    return "SAME"
