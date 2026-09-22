#!/usr/bin/env python3
"""Tests for src/attacklm/bench/packs.py — benchmark pack manifests."""

from __future__ import annotations

import pytest

from attacklm.bench.packs import PACKS_DIR, Pack, list_packs, load_pack

MINIMAL = """
name: demo
harness_task: inspect_evals/demo
mode: harness_scored
source:
  kind: huggingface
  repo: acme/demo
  config: main
  revision: abc123
  sha256: deadbeef
license: MIT
redistributable: true
metric: accuracy
chance_level: 0.25
reference_scores:
  gpt-4: 0.71
ladder: [100, 200, null]
categories_from: field:category
"""


def test_load_pack_parses_all_fields(tmp_path):
    p = tmp_path / "demo.yaml"
    p.write_text(MINIMAL)
    pack = load_pack(p)
    assert isinstance(pack, Pack)
    assert pack.name == "demo"
    assert pack.source.repo == "acme/demo"
    assert pack.source.revision == "abc123"
    assert pack.chance_level == 0.25
    assert pack.ladder == [100, 200, None]
    assert pack.reference_scores["gpt-4"] == 0.71


def test_missing_required_field_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(MINIMAL.replace("chance_level: 0.25\n", ""))
    with pytest.raises(ValueError, match="chance_level"):
        load_pack(p)


def test_unpinned_revision_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(MINIMAL.replace("revision: abc123", "revision: main"))
    with pytest.raises(ValueError, match="revision"):
        load_pack(p)


def test_local_source_may_omit_revision(tmp_path):
    p = tmp_path / "local.yaml"
    p.write_text(
        MINIMAL.replace("kind: huggingface", "kind: local")
        .replace("  revision: abc123\n", "")
        .replace("  repo: acme/demo\n", "")
    )
    pack = load_pack(p)
    assert pack.source.kind == "local"
    assert pack.source.revision is None


def test_bad_metric_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(MINIMAL.replace("metric: accuracy", "metric: vibes"))
    with pytest.raises(ValueError, match="metric"):
        load_pack(p)


def test_empty_ladder_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(MINIMAL.replace("ladder: [100, 200, null]", "ladder: []"))
    with pytest.raises(ValueError, match="ladder"):
        load_pack(p)


def test_mode_is_parsed(tmp_path):
    p = tmp_path / "demo.yaml"
    p.write_text(MINIMAL)
    assert load_pack(p).mode == "harness_scored"


def test_bad_mode_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(MINIMAL.replace("mode: harness_scored", "mode: guesswork"))
    with pytest.raises(ValueError, match="mode"):
        load_pack(p)


def test_github_raw_source_must_be_pinned(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        MINIMAL.replace("kind: huggingface", "kind: github_raw").replace(
            "revision: abc123", "revision: main"
        )
    )
    with pytest.raises(ValueError, match="revision"):
        load_pack(p)


def test_shipped_packs_all_load():
    packs = list_packs(PACKS_DIR)
    assert packs, "no shipped pack manifests found"
    assert {p.name for p in packs} >= {"cybermetric-500"}


def test_cybermetric_pack_is_harness_scored_and_pinned():
    """inspect_evals/cybermetric fetches and scores its own data (choice()),
    so we normalise ITS scores rather than re-deriving them."""
    pack = next(p for p in list_packs(PACKS_DIR) if p.name == "cybermetric-500")
    assert pack.mode == "harness_scored"
    assert pack.harness_task == "inspect_evals/cybermetric_500"
    assert pack.source.revision == "205262cdf5022ba890e792efd176fb19d42913fa"
    assert pack.redistributable is False  # CyberMetric ships no license


def test_non_redistributable_packs_have_no_committed_data():
    """Global constraint: fetch-only packs must never have data in the repo."""
    for pack in list_packs(PACKS_DIR):
        if pack.redistributable:
            continue
        cache = PACKS_DIR.parent / "cache" / pack.name
        assert not cache.exists(), f"{pack.name} is fetch-only but has committed data at {cache}"
