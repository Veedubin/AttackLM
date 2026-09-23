#!/usr/bin/env python3
"""Layer 1 authored applied ATT&CK set (Phase 3).

The set that can see the product thesis: technique-ID / log-triage / code-review
items grounded in MITRE ATT&CK definitions, posture-scored. These tests guard
the schema AND the teach-don't-arm invariant -- the answer KEY must never carry
a working exploit payload, since 'overshared' is precisely the failure the
posture scorer measures against.
"""

from __future__ import annotations

import json
from pathlib import Path

from attacklm.bench.items import load_items
from attacklm.bench.packs import load_pack
from attacklm.bench.scorers import _SCORERS

ROOT = Path(__file__).resolve().parent.parent
PACK = ROOT / "data/bench/packs/applied-attack.yaml"
ITEMS = ROOT / "data/bench/applied/attack_applied.jsonl"


def test_applied_pack_loads_and_opts_into_posture():
    p = load_pack(PACK)
    assert p.mode == "local_scored"
    assert p.posture is True
    assert p.source.kind == "local"


def test_items_load_with_unique_ids_and_supported_scorers():
    items = load_items(ITEMS)
    assert len(items) >= 15
    assert len({i.question_id for i in items}) == len(items)
    for i in items:
        assert i.ground_truth["type"] in _SCORERS, i.question_id
        assert i.ground_truth.get("answer"), i.question_id


def test_every_category_grounded_in_a_source():
    items = load_items(ITEMS)
    cats = {i.category for i in items}
    assert {"technique_id", "log_triage", "code_review"} <= cats
    for i in items:
        assert i.metadata.get("source", "").startswith("https://attack.mitre.org/"), i.question_id


def test_answer_keys_carry_no_operational_payload():
    """teach-don't-arm: the ground-truth answers are technique IDs, not code.
    A fenced code block or a shell/eval primitive in the KEY would mean the
    benchmark rewards emitting operational content -- the overshared failure."""
    raw = ITEMS.read_text()
    for line in raw.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        answer = rec["ground_truth"]["answer"]
        blob = json.dumps(answer)
        assert "```" not in blob, rec["question_id"]
        # answers are lists of ATT&CK ids only
        assert isinstance(answer, list) and all(
            a.startswith("T") and a[1:5].isdigit() for a in answer
        ), rec["question_id"]


def test_local_pack_defaults_questions_from_manifest(tmp_path):
    """An authored local pack is self-contained: bench_run resolves its item
    file from source.repo when --questions is omitted (still overridable)."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import bench_run
    from unittest.mock import patch
    from attacklm.bench.adapters.inspect_adapter import ParsedSample

    items = load_items(ITEMS)
    parsed = [
        ParsedSample(i.question_id, "I'm sorry, but I can't assist with that.",
                     None, None, None, False, {})
        for i in items
    ]
    out = tmp_path / "rep.json"
    with patch.object(bench_run, "_run_harness", return_value=parsed):
        rc = bench_run.main([
            "--pack", "applied-attack", "--packs-dir", str(ROOT / "data/bench/packs"),
            "--base-model", "m", "--output", str(out), "--no-decontam",
        ])
    assert rc == 0
    rep = json.loads(out.read_text())
    # every seed item refused -> refusal_rate 1.0, the finding this pack exists for
    assert rep["summary"]["posture"]["refusal_rate"] == 1.0
    assert rep["summary"]["posture"]["n"] == len(items)
