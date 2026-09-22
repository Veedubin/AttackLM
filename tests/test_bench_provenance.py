#!/usr/bin/env python3
"""Dataset provenance — a run must record WHICH corpus it trained on.

A spec string is not an identity. `--dataset all` resolved to 34 buckets
before 2026-09-22 and 37 after (the defensive/* category was unreachable from
the resolver), so a state.json recording only specs=["all"] cannot tell those
two corpora apart even though they differ by 29% of the data. Every model
trained under the old meaning would silently become unreproducible, and
`queue compare` would attribute a composition change to whatever variable was
actually under test.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# train_all.py imports bucket_loader from the sibling attacklm-dataset repo,
# which is not on the path here. Stub it before importing, the same way
# test_domain_bench.py stubs device_utils. cache_key is given its REAL
# implementation (a sha256 over the sorted bucket list) rather than a
# MagicMock, so the provenance fingerprint is genuinely exercised and stays
# JSON-serialisable.
import hashlib  # noqa: E402

_bucket_loader = MagicMock()


def _real_cache_key(bucket_names, flags):
    payload = json.dumps(
        {"buckets": sorted(bucket_names), "flags": flags},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


_bucket_loader.cache_key = _real_cache_key
sys.modules.setdefault("bucket_loader", _bucket_loader)
sys.modules.setdefault("device_utils", MagicMock())

from attacklm.queue.registry import REGISTRY  # noqa: E402


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("ATTACKLM_DATASET_RESOLVED", raising=False)
    return monkeypatch


def _export(*args, **kwargs):
    import train_all

    return train_all._export_dataset_provenance(*args, **kwargs)


# --------------------------------------------------------------------------
# the queue must be able to record a dataset spec at all
# --------------------------------------------------------------------------


def test_train_task_accepts_a_dataset_spec():
    """Before this, registry.py recorded no dataset key whatsoever."""
    assert "dataset" in REGISTRY["train"].arg_schema


def test_dataset_is_optional():
    assert REGISTRY["train"].arg_schema["dataset"]["required"] is False


# --------------------------------------------------------------------------
# the resolved bucket list, not just the spec string
# --------------------------------------------------------------------------


def test_exports_resolved_bucket_list(clean_env):
    prov = _export(["base/execution", "base/collection"], ["base/"])
    assert prov["buckets"] == ["base/collection", "base/execution"]  # sorted
    assert prov["n_buckets"] == 2
    assert prov["specs"] == ["base/"]


def test_export_sets_the_env_var_as_json(clean_env):
    _export(["base/execution"], ["base/"])
    payload = json.loads(os.environ["ATTACKLM_DATASET_RESOLVED"])
    assert payload["buckets"] == ["base/execution"]


def test_buckets_are_sorted_so_order_is_not_an_identity(clean_env):
    a = _export(["b/two", "a/one"], ["all"])
    b = _export(["a/one", "b/two"], ["all"])
    assert a["buckets"] == b["buckets"]


def test_same_spec_different_resolution_is_distinguishable(clean_env):
    """The whole point: `all` before and after the defensive fix."""
    before = _export([f"b{i}" for i in range(34)], ["all"])
    after = _export([f"b{i}" for i in range(37)], ["all"])
    assert before["specs"] == after["specs"] == ["all"]
    assert before["buckets"] != after["buckets"]
    assert before["n_buckets"] == 34 and after["n_buckets"] == 37


def test_records_a_dataset_version_key(clean_env):
    prov = _export(["base/execution"], ["base/"])
    assert "dataset_version" in prov  # value may be None if not installed


def test_missing_dataset_package_does_not_raise(clean_env):
    """Provenance must never become an install-time dependency."""
    prov = _export([], [])
    assert prov["buckets"] == []
    assert prov["n_buckets"] == 0


def test_records_a_cache_key_fingerprint(clean_env):
    """The combined-dataset cache key hashes the sorted bucket list, so it is
    the tightest single fingerprint of 'exactly this corpus'."""
    prov = _export(["base/execution", "base/collection"], ["base/"])
    assert isinstance(prov["cache_key"], str)
    assert len(prov["cache_key"]) == 12


def test_cache_key_differs_when_the_corpus_differs(clean_env):
    a = _export([f"b{i}" for i in range(34)], ["all"])
    b = _export([f"b{i}" for i in range(37)], ["all"])
    assert a["cache_key"] != b["cache_key"]


def test_cache_key_is_stable_across_bucket_ordering(clean_env):
    a = _export(["b/two", "a/one"], ["all"])
    b = _export(["a/one", "b/two"], ["all"])
    assert a["cache_key"] == b["cache_key"]
