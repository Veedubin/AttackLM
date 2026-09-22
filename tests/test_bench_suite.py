#!/usr/bin/env python3
"""Suite switching — capability is the default, audits stay one flag away.

The audits measure whether a model can be ABUSED. Necessary, but not what the
model is for. So the default gauntlet became the capability suite, and the
audit gauntlet that used to be the default is reachable unchanged via
`--suite audits`.
"""

from __future__ import annotations

import pytest

from attacklm.queue.gauntlet import (
    DEFAULT_SUITE,
    GAUNTLET_PRESETS,
    SUITES,
    expand_gauntlet,
    resolve_suite,
)


def test_capability_is_the_default_suite():
    assert DEFAULT_SUITE == "capability"


def test_suites_cover_the_three_questions():
    assert set(SUITES) == {"capability", "audits", "all"}


def test_audits_suite_is_the_old_default_gauntlet_unchanged():
    """The existing tests must remain available, and identical."""
    assert resolve_suite("audits") == ["core"]
    # `core` may also auto-insert gen_calibration_holdouts when the holdout
    # files are absent -- pre-existing behaviour, a data-prep step rather than
    # an audit, so it is filtered out here rather than asserted against.
    audits = [t["type"] for t in expand_gauntlet("core") if t["type"].startswith("audit_")]
    assert audits == [
        "audit_prompt_injection",
        "audit_system_prompt",
        "audit_canary_pipeline",
        "audit_calibration",
    ]


def test_capability_suite_runs_only_bench_tasks():
    types = [t["type"] for t in expand_gauntlet("capability")]
    assert types, "capability preset is empty"
    assert all(t.startswith("bench_") for t in types), types


def test_capability_suite_includes_the_flagship():
    types = [t["type"] for t in expand_gauntlet("capability")]
    assert "bench_ctibench_mcq" in types
    assert "bench_ctibench_ate" in types


def test_all_suite_runs_both_families():
    presets = resolve_suite("all")
    types = [t["type"] for p in presets for t in expand_gauntlet(p)]
    assert any(t.startswith("bench_") for t in types)
    assert any(t.startswith("audit_") for t in types)


def test_unknown_suite_raises_and_names_the_options():
    with pytest.raises(ValueError, match="nonsense"):
        resolve_suite("nonsense")


def test_every_suite_maps_to_real_presets():
    for name in SUITES:
        for preset in resolve_suite(name):
            assert preset in GAUNTLET_PRESETS, f"suite {name} names unknown preset {preset}"


def test_packs_named_by_capability_preset_all_exist():
    from attacklm.bench.packs import PACKS_DIR, list_packs

    shipped = {p.name for p in list_packs(PACKS_DIR)}
    for task in expand_gauntlet("capability"):
        pack = task["args"].get("pack")
        assert pack in shipped, f"{task['type']} names missing pack {pack!r}"
