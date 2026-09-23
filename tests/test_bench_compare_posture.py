#!/usr/bin/env python3
"""Phase 3: a single Layer 1 run carries capability AND posture metrics whose
verdict directions are OPPOSITE. A rise in the capability score is BETTER; a
rise in the refusal rate is WORSE. The per-attack HIGHER_IS_WORSE flag alone
cannot express both, so _higher_is_worse resolves per (attack, metric)."""

from __future__ import annotations

from attacklm.queue.compare import ITEM_METRICS, _higher_is_worse
from attacklm.queue.stats import Interval, verdict


def test_applied_pack_registers_capability_and_posture_metrics():
    metrics = [m for m, _, _ in ITEM_METRICS["bench_applied_attack"]]
    assert "score" in metrics and "score_clean" in metrics and "refused" in metrics


def test_capability_metric_is_higher_is_better():
    assert _higher_is_worse("bench_applied_attack", "score") is False
    assert _higher_is_worse("bench_applied_attack", "score_clean") is False


def test_refusal_metric_is_higher_is_worse_on_the_same_pack():
    assert _higher_is_worse("bench_applied_attack", "refused") is True


def test_unregistered_metric_falls_through_to_attack_then_default():
    # falls through to the per-attack entry (False)
    assert _higher_is_worse("bench_applied_attack", "made_up") is False
    # unknown attack defaults to True (a new metric most likely measures failure)
    assert _higher_is_worse("bench_totally_unknown", "whatever") is True


def test_same_positive_delta_reads_opposite_on_the_two_metrics():
    # B scored higher than A on both metrics (lo > 0). For capability that is
    # BETTER; for refusal it is WORSE -- the exact inversion the override fixes.
    iv = Interval(delta=0.20, lo=0.05, hi=0.35, n=30)
    assert verdict(iv, _higher_is_worse("bench_applied_attack", "score")) == "BETTER"
    assert verdict(iv, _higher_is_worse("bench_applied_attack", "refused")) == "WORSE"


# --- code-review fixes #1/#2: applied pack shipped + judge metrics compared ---

def test_applied_pack_is_in_shipped_attacks():
    """Without this, queue compare filters the teach-don't-arm pack out of
    every comparison (cli._preset_attacks_for_compare keeps only shipped)."""
    from attacklm.queue import compare
    assert "bench_applied_attack" in compare.SHIPPED_ATTACKS


def test_judge_metrics_are_compared():
    from attacklm.queue import compare
    metrics = {m for m, _id, _cat in compare.ITEM_METRICS["bench_applied_attack"]}
    assert {"taught", "overshared"} <= metrics


def test_overshared_reads_as_worse_taught_as_better():
    """A model that overshares MORE must be reported WORSE, not BETTER -- the
    exact inversion the per-metric override exists to prevent."""
    from attacklm.queue.compare import _higher_is_worse as hw
    assert hw("bench_applied_attack", "overshared") is True
    assert hw("bench_applied_attack", "taught") is False
    assert hw("bench_applied_attack", "refused") is True
    assert hw("bench_applied_attack", "score") is False
