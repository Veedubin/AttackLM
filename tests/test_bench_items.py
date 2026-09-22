#!/usr/bin/env python3
"""Tests for src/attacklm/bench/items.py — items and nested rung sampling."""

from __future__ import annotations

import json

import pytest

from attacklm.bench.items import BenchItem, load_items, sample_items


def _item(i: int) -> dict:
    return {
        "question_id": f"q{i:04d}",
        "category": "cat_a" if i % 2 == 0 else "cat_b",
        "tier": "mcq",
        "messages": [{"role": "user", "content": f"question {i}"}],
        "ground_truth": {"type": "mcq_choice", "answer": "C"},
        "metadata": {"source_index": i},
    }


def _write(tmp_path, n):
    p = tmp_path / "items.jsonl"
    p.write_text("\n".join(json.dumps(_item(i)) for i in range(n)))
    return p


def test_load_items_parses(tmp_path):
    items = load_items(_write(tmp_path, 5))
    assert len(items) == 5
    assert isinstance(items[0], BenchItem)
    assert items[0].question_id == "q0000"
    assert items[0].ground_truth["answer"] == "C"


def test_load_items_skips_blank_lines(tmp_path):
    p = tmp_path / "items.jsonl"
    p.write_text(json.dumps(_item(0)) + "\n\n" + json.dumps(_item(1)) + "\n")
    assert len(load_items(p)) == 2


def test_load_items_defaults_optional_fields(tmp_path):
    p = tmp_path / "items.jsonl"
    p.write_text(json.dumps({
        "question_id": "q0",
        "messages": [{"role": "user", "content": "?"}],
        "ground_truth": {"type": "mcq_choice", "answer": "A"},
    }))
    item = load_items(p)[0]
    assert item.category == "uncategorised"
    assert item.tier == ""
    assert item.metadata == {}


def test_load_items_rejects_duplicate_ids(tmp_path):
    """Pairing is by id; duplicates would silently corrupt a comparison."""
    p = tmp_path / "items.jsonl"
    p.write_text(json.dumps(_item(0)) + "\n" + json.dumps(_item(0)))
    with pytest.raises(ValueError, match="duplicate"):
        load_items(p)


def test_sample_is_deterministic(tmp_path):
    items = load_items(_write(tmp_path, 50))
    a = [i.question_id for i in sample_items(items, 10, seed=42)]
    b = [i.question_id for i in sample_items(items, 10, seed=42)]
    assert a == b


def test_different_seed_gives_different_sample(tmp_path):
    items = load_items(_write(tmp_path, 50))
    a = {i.question_id for i in sample_items(items, 10, seed=42)}
    b = {i.question_id for i in sample_items(items, 10, seed=7)}
    assert a != b


def test_rungs_are_nested(tmp_path):
    """Rung N+1 must contain every item of rung N — the ladder depends on it."""
    items = load_items(_write(tmp_path, 50))
    small = {i.question_id for i in sample_items(items, 10, seed=42)}
    big = {i.question_id for i in sample_items(items, 25, seed=42)}
    assert small < big


def test_every_rung_pair_is_nested(tmp_path):
    """Stronger: the whole ladder must nest, not just one pair."""
    items = load_items(_write(tmp_path, 100))
    seen: set[str] = set()
    for rung in (10, 20, 40, 80, None):
        current = {i.question_id for i in sample_items(items, rung, seed=42)}
        assert seen <= current, f"rung {rung} dropped items from the previous rung"
        seen = current


def test_rung_none_returns_everything(tmp_path):
    items = load_items(_write(tmp_path, 50))
    assert len(sample_items(items, None, seed=42)) == 50


def test_rung_larger_than_corpus_returns_everything(tmp_path):
    items = load_items(_write(tmp_path, 10))
    assert len(sample_items(items, 999, seed=42)) == 10


def test_sample_does_not_mutate_input(tmp_path):
    items = load_items(_write(tmp_path, 20))
    before = [i.question_id for i in items]
    sample_items(items, 5, seed=42)
    assert [i.question_id for i in items] == before
