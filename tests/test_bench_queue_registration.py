#!/usr/bin/env python3
"""The capability benchmark must be a first-class queue task type."""

from __future__ import annotations

from attacklm.queue.compare import HIGHER_IS_WORSE, ITEM_METRICS, SHIPPED_ATTACKS
from attacklm.queue.gauntlet import GAUNTLET_PRESETS, expand_gauntlet
from attacklm.queue.registry import REGISTRY


def test_bench_task_registered():
    spec = REGISTRY["bench_cybermetric"]
    assert spec.script == "bench_run.py"
    assert spec.runner_mode == "subprocess"
    assert spec.produces_artifact == "report"
    assert spec.consumes_artifact == "adapter"
    assert spec.implemented is True


def test_bench_task_script_exists():
    assert REGISTRY["bench_cybermetric"].script_path.is_file()


def test_bench_arg_schema_covers_cli_contract():
    schema = REGISTRY["bench_cybermetric"].arg_schema
    for key in ("pack", "questions", "output", "base_model", "adapter", "rung"):
        assert key in schema, f"{key} missing from arg_schema"


def test_capability_preset_expands():
    assert "capability-quick" in GAUNTLET_PRESETS
    tasks = expand_gauntlet("capability-quick")
    assert [t["type"] for t in tasks] == ["bench_cybermetric"]
    assert tasks[0]["gauntlet"] == "capability-quick"
    assert tasks[0]["args"]["pack"] == "cybermetric-500"


def test_registered_in_shipped_attacks():
    assert "bench_cybermetric" in SHIPPED_ATTACKS


def test_item_metrics_contract():
    entries = ITEM_METRICS["bench_cybermetric"]
    assert ("score", "question_id", "category") in entries


def test_capability_metric_direction_is_inverted():
    """A capability score rising is an IMPROVEMENT, unlike every audit metric.

    Without this, a model that genuinely got better at answering security
    questions would be reported WORSE, inverting the signal training is
    steered by.
    """
    assert HIGHER_IS_WORSE["bench_cybermetric"] is False


def test_audit_metrics_remain_higher_is_worse():
    for attack in (
        "audit_prompt_injection",
        "audit_system_prompt",
        "audit_canary_pipeline",
        "audit_calibration",
    ):
        assert HIGHER_IS_WORSE[attack] is True


def test_unknown_attack_defaults_to_higher_is_worse():
    """Over-reporting a regression is the safer error for an unlabelled metric."""
    assert HIGHER_IS_WORSE.get("some_future_attack", True) is True


def test_every_shipped_attack_declares_a_direction():
    for attack in SHIPPED_ATTACKS:
        assert attack in HIGHER_IS_WORSE, f"{attack} has no declared metric direction"


def test_existing_attacks_untouched():
    """Regression: the audit attacks must keep working exactly as before."""
    for attack in (
        "audit_prompt_injection",
        "audit_system_prompt",
        "audit_canary_pipeline",
        "audit_calibration",
    ):
        assert attack in SHIPPED_ATTACKS
    assert ITEM_METRICS["audit_prompt_injection"] == [("asr", "question_id", "tier")]
    assert ITEM_METRICS["audit_system_prompt"] == [("asr", "question_id", "tier")]


def test_existing_gauntlet_presets_untouched():
    for preset in ("core", "full", "quick", "memorization"):
        assert preset in GAUNTLET_PRESETS
    assert [t["type"] for t in GAUNTLET_PRESETS["quick"]] == [
        "audit_prompt_injection",
        "audit_system_prompt",
    ]
