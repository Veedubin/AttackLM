"""Tests for attacklm.queue.stats — paired bootstrap and verdicts."""

from __future__ import annotations

import pytest

from attacklm.queue.stats import Interval, paired_bootstrap, verdict


class TestPairedBootstrap:
    def test_all_equal_is_zero_delta(self):
        iv = paired_bootstrap([0.0, 1.0, 0.5, 1.0], [0.0, 1.0, 0.5, 1.0])
        assert iv.delta == 0.0 and iv.lo == 0.0 and iv.hi == 0.0 and iv.n == 4

    def test_constant_shift_has_tight_positive_ci(self):
        a = [0.0] * 20
        b = [1.0] * 20
        iv = paired_bootstrap(a, b)
        assert iv.delta == pytest.approx(1.0)
        assert iv.lo == pytest.approx(1.0) and iv.hi == pytest.approx(1.0)

    def test_mixed_shift_ci_excludes_zero(self):
        a = [0.0] * 30
        b = [1.0] * 25 + [0.0] * 5   # mean delta 0.83
        iv = paired_bootstrap(a, b, seed=1)
        assert iv.delta == pytest.approx(25 / 30)
        assert iv.lo > 0.5 and iv.hi <= 1.0

    def test_noise_ci_includes_zero(self):
        a = [0.0, 1.0] * 10
        b = [1.0, 0.0] * 10   # same mean, per-item +1/-1
        iv = paired_bootstrap(a, b, seed=3)
        assert iv.delta == pytest.approx(0.0)
        assert iv.lo < 0.0 < iv.hi

    def test_deterministic_for_seed(self):
        # NOTE: deviates from the brief's verbatim 5-item dataset. That dataset's
        # diffs ([1, 0, 0, 0, -1]) only admit 6 distinct resample means, so the
        # 2.5th/97.5th percentile of 2000 resamples lands on the same extreme
        # value (-0.6, 0.6) for ~199/200 seeds tested (including 7 vs 8) with
        # the brief's own reference implementation -- a property of the data,
        # not a bug in paired_bootstrap. This 10-item dataset has enough
        # distinct achievable means that seed=7 and seed=8 reliably diverge
        # (verified directly), which is what the test intends to check.
        a = [0.0, 0.5, 1.0, 0.0, 1.0, 0.25, 0.75, 0.6, 0.1, 0.9]
        b = [1.0, 0.5, 1.0, 0.0, 0.0, 0.8, 0.2, 0.4, 0.95, 0.05]
        assert paired_bootstrap(a, b, seed=7) == paired_bootstrap(a, b, seed=7)
        assert paired_bootstrap(a, b, seed=7) != paired_bootstrap(a, b, seed=8)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            paired_bootstrap([0.0], [0.0, 1.0])

    def test_empty(self):
        iv = paired_bootstrap([], [])
        assert iv.n == 0 and iv.delta == 0.0

    def test_single_item(self):
        iv = paired_bootstrap([0.0], [1.0])
        assert iv.n == 1 and iv.delta == 1.0 and iv.lo == 1.0 and iv.hi == 1.0


class TestVerdict:
    def test_worse_when_ci_above_zero_and_higher_is_worse(self):
        assert verdict(Interval(0.3, 0.1, 0.5, 20)) == "WORSE"

    def test_better_when_ci_below_zero_and_higher_is_worse(self):
        assert verdict(Interval(-0.3, -0.5, -0.1, 20)) == "BETTER"

    def test_same_when_ci_spans_zero(self):
        assert verdict(Interval(0.1, -0.2, 0.4, 20)) == "SAME"

    def test_flipped_direction(self):
        assert verdict(Interval(0.3, 0.1, 0.5, 20), higher_is_worse=False) == "BETTER"

    def test_same_when_n_below_two(self):
        assert verdict(Interval(1.0, 1.0, 1.0, 1)) == "SAME"
        assert verdict(Interval(0.0, 0.0, 0.0, 0)) == "SAME"

    def test_boundary_zero_is_same(self):
        assert verdict(Interval(0.2, 0.0, 0.4, 20)) == "SAME"
