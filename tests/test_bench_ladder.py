"""Tests for escalation ladder decision logic."""

from __future__ import annotations

import pytest

from attacklm.bench.ladder import decide_next_rung
from attacklm.queue.stats import Interval


class TestDecideNextRungRules:
    """Test each rule in strict order of application."""

    def test_stop_no_data(self):
        """interval.n == 0 should report stop_no_data before any other rule."""
        ladder = [100, 200, 400]
        interval = Interval(delta=0.0, lo=0.0, hi=0.0, n=0)
        decision = decide_next_rung(ladder, 100, interval)

        assert decision.escalate is False
        assert decision.reason == "stop_no_data"
        assert decision.rung_index == 0
        assert decision.next_rung is None

    def test_stop_decisive(self):
        """CI excludes zero (lo > 0 or hi < 0) -> stop_decisive."""
        ladder = [100, 200, 400]
        
        # Case: lo > 0
        interval_positive = Interval(delta=0.05, lo=0.01, hi=0.10, n=10)
        decision = decide_next_rung(ladder, 100, interval_positive)
        assert decision.escalate is False
        assert decision.reason == "stop_decisive"
        assert decision.rung_index == 0
        
        # Case: hi < 0
        interval_negative = Interval(delta=-0.05, lo=-0.10, hi=-0.01, n=10)
        decision = decide_next_rung(ladder, 100, interval_negative)
        assert decision.escalate is False
        assert decision.reason == "stop_decisive"
        assert decision.rung_index == 0

    def test_stop_tight_width(self):
        """CI width <= ci_tight_width -> stop_tight."""
        ladder = [100, 200, 400]
        interval = Interval(delta=0.0, lo=-0.02, hi=0.03, n=20)
        decision = decide_next_rung(ladder, 100, interval, ci_tight_width=0.05)

        assert decision.escalate is False
        assert decision.reason == "stop_tight"
        assert decision.rung_index == 0

    def test_stop_tight_not_degenerate_zero_width(self):
        """lo == hi exactly (degenerate) should NOT trigger stop_tight.
        
        This tests edge case: degenerate interval (lo==hi) is artifact of 
        too-little data, not confidence. Must explicitly NOT be tight.
        """
        ladder = [100, 200, 400]
        interval = Interval(delta=0.0, lo=0.0, hi=0.0, n=1)
        decision = decide_next_rung(ladder, 100, interval, ci_tight_width=0.05)

        assert decision.escalate is True
        assert decision.reason == "escalate"
        assert decision.next_rung == 200

    def test_stop_floor_dead_horse(self):
        """subject_ci_hi < (chance_level + floor_margin) -> stop_floor."""
        ladder = [100, 200, 400]
        # Use an interval that includes zero, so it doesn't trigger stop_decisive
        interval = Interval(delta=-0.05, lo=-0.1, hi=0.05, n=20)
        
        decision = decide_next_rung(
            ladder, 
            100, 
            interval,
            subject_ci_hi=0.29,  # 0.29 < 0.25 + 0.05
            chance_level=0.25,
            floor_margin=0.05,
        )

        assert decision.escalate is False
        assert decision.reason == "stop_floor"
        assert decision.rung_index == 0

    def test_stop_floor_not_triggered_when_subject_ci_hi_is_none(self):
        """stop_floor only applies when subject_ci_hi is not None."""
        ladder = [100, 200, 400]
        # Use an interval that includes zero (doesn't trigger decisive)
        interval = Interval(delta=-0.05, lo=-0.1, hi=0.05, n=20)
        
        decision = decide_next_rung(
            ladder,
            100,
            interval,
            subject_ci_hi=None,  # None means rule doesn't apply
            chance_level=0.25,
            floor_margin=0.05,
        )

        # Should escalate since no other stop condition applies
        assert decision.escalate is True
        assert decision.reason == "escalate"

    def test_stop_exhausted_at_last_rung_integer(self):
        """current_rung is the LAST entry (int) in ladder -> stop_exhausted."""
        ladder = [100, 200, 400]
        # Use wide interval that doesn't trigger other stops
        interval = Interval(delta=0.0, lo=-0.05, hi=0.05, n=20)
        
        decision = decide_next_rung(ladder, 400, interval)

        assert decision.escalate is False
        assert decision.reason == "stop_exhausted"
        assert decision.rung_index == 2
        assert decision.next_rung is None

    def test_stop_exhausted_at_last_rung_none(self):
        """current_rung is None (full set) -> stop_exhausted."""
        ladder = [100, 200, 400, None]
        # Use wide interval that doesn't trigger other stops
        interval = Interval(delta=0.0, lo=-0.05, hi=0.05, n=1000)
        
        decision = decide_next_rung(ladder, None, interval)

        assert decision.escalate is False
        assert decision.reason == "stop_exhausted"
        assert decision.rung_index == 3
        assert decision.next_rung is None

    def test_escalate_default(self):
        """No stop condition met -> escalate to next rung."""
        ladder = [100, 200, 400]
        interval = Interval(delta=0.0, lo=-0.05, hi=0.05, n=10)
        
        decision = decide_next_rung(ladder, 100, interval, ci_tight_width=0.05)

        assert decision.escalate is True
        assert decision.reason == "escalate"
        assert decision.next_rung == 200
        assert decision.rung_index == 0

    def test_escalate_from_middle_rung(self):
        """Escalate from middle rung to next."""
        ladder = [100, 200, 400, 800]
        interval = Interval(delta=0.0, lo=-0.05, hi=0.05, n=15)
        
        decision = decide_next_rung(ladder, 200, interval)

        assert decision.escalate is True
        assert decision.reason == "escalate"
        assert decision.next_rung == 400
        assert decision.rung_index == 1

    def test_rule_order_decisive_before_tight(self):
        """Rule order: decisive checked before tight.
        
        If both apply, decisive wins.
        """
        ladder = [100, 200, 400]
        # CI width is 0.01, which is tight, but also excludes zero (decisive)
        interval = Interval(delta=0.05, lo=0.045, hi=0.055, n=100)
        
        decision = decide_next_rung(ladder, 100, interval, ci_tight_width=0.05)

        assert decision.reason == "stop_decisive"

    def test_rule_order_tight_before_floor(self):
        """Rule order: tight checked before floor."""
        ladder = [100, 200, 400]
        # Width = 0.05, which is tight
        interval = Interval(delta=0.0, lo=-0.02, hi=0.03, n=20)
        
        decision = decide_next_rung(
            ladder,
            100,
            interval,
            subject_ci_hi=0.29,
            chance_level=0.25,
            floor_margin=0.05,
            ci_tight_width=0.05,
        )

        assert decision.reason == "stop_tight"

    def test_min_paired_n_escalates_not_stops(self):
        """0 < n < MIN_PAIRED_N must NOT stop via tight or floor rules.
        
        Escalate if other conditions don't forbid it.
        """
        ladder = [100, 200, 400]
        # n=3, which is < MIN_PAIRED_N (5)
        interval = Interval(delta=0.0, lo=0.0, hi=0.0, n=3)
        
        decision = decide_next_rung(ladder, 100, interval, ci_tight_width=0.05)

        assert decision.escalate is True
        assert decision.reason == "escalate"
        assert decision.next_rung == 200

    def test_no_data_before_degenerate_tight(self):
        """n==0 is checked before any width rules."""
        ladder = [100, 200, 400]
        interval = Interval(delta=0.0, lo=0.0, hi=0.0, n=0)
        
        decision = decide_next_rung(ladder, 100, interval, ci_tight_width=0.05)

        assert decision.reason == "stop_no_data"
        assert decision.rung_index == 0

    def test_error_current_rung_not_in_ladder(self):
        """current_rung not in ladder -> ValueError."""
        ladder = [100, 200, 400]
        interval = Interval(delta=0.0, lo=-0.02, hi=0.02, n=20)
        
        with pytest.raises(ValueError) as exc_info:
            decide_next_rung(ladder, 999, interval)
        
        assert "999" in str(exc_info.value)

    def test_error_empty_ladder(self):
        """empty ladder -> ValueError."""
        ladder = []
        interval = Interval(delta=0.0, lo=-0.02, hi=0.02, n=20)
        
        with pytest.raises(ValueError):
            decide_next_rung(ladder, 100, interval)

    def test_rung_index_correctly_computed(self):
        """rung_index is the index of current_rung in ladder."""
        ladder = [100, 200, 400, 800]
        interval = Interval(delta=0.0, lo=-0.05, hi=0.05, n=10)
        
        # current_rung = 200 is at index 1
        decision = decide_next_rung(ladder, 200, interval)
        assert decision.rung_index == 1
        
        # current_rung = 800 is at index 3
        decision = decide_next_rung(ladder, 800, interval)
        assert decision.rung_index == 3

    def test_rung_index_with_none_rung(self):
        """rung_index works when current_rung is None."""
        ladder = [100, 200, 400, None]
        interval = Interval(delta=0.0, lo=-0.05, hi=0.05, n=100)
        
        decision = decide_next_rung(ladder, None, interval)
        assert decision.rung_index == 3

    def test_defaults_for_optional_params(self):
        """Test default values of optional parameters."""
        ladder = [100, 200]
        # Wide interval to avoid triggering tight with defaults
        interval = Interval(delta=0.0, lo=-0.05, hi=0.05, n=20)
        
        # With defaults: ci_tight_width=0.05, floor_margin=0.05, chance_level=0.25
        decision = decide_next_rung(ladder, 100, interval)
        
        # Should escalate with default params
        assert decision.escalate is True
        assert decision.reason == "escalate"

    def test_stop_floor_boundary_exactly_at_threshold(self):
        """subject_ci_hi == (chance_level + floor_margin) should NOT stop."""
        ladder = [100, 200, 400]
        # Use interval that includes zero to avoid decisive
        interval = Interval(delta=-0.05, lo=-0.1, hi=0.05, n=20)
        
        # subject_ci_hi = 0.30, threshold = 0.25 + 0.05 = 0.30, so == not <
        decision = decide_next_rung(
            ladder,
            100,
            interval,
            subject_ci_hi=0.30,
            chance_level=0.25,
            floor_margin=0.05,
        )

        assert decision.escalate is True
        assert decision.reason == "escalate"

    def test_tight_boundary_exactly_at_threshold(self):
        """Width == ci_tight_width should stop as tight."""
        ladder = [100, 200, 400]
        # Width = 0.05
        interval = Interval(delta=0.0, lo=-0.025, hi=0.025, n=20)
        
        decision = decide_next_rung(ladder, 100, interval, ci_tight_width=0.05)

        assert decision.reason == "stop_tight"

    def test_next_rung_at_boundary_before_none(self):
        """When last rung is None, penultimate escalation points to None."""
        ladder = [100, 200, 400, None]
        interval = Interval(delta=0.0, lo=-0.05, hi=0.05, n=10)
        
        decision = decide_next_rung(ladder, 400, interval)

        assert decision.escalate is True
        assert decision.next_rung is None
        assert decision.rung_index == 2

    def test_escalate_with_low_confidence_not_at_bound(self):
        """Low confidence (wide CI) escalates if no stop rule applies."""
        ladder = [100, 200, 400, 800]
        interval = Interval(delta=0.0, lo=-0.2, hi=0.2, n=8)
        
        decision = decide_next_rung(ladder, 100, interval, ci_tight_width=0.05)

        assert decision.escalate is True
        assert decision.reason == "escalate"
        assert decision.next_rung == 200
