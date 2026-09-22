"""Tests for baseline auto-insert / explicit command."""

from __future__ import annotations

import argparse

import pytest

from attacklm.queue import baseline as bl
from attacklm.queue import cli as qcli
from attacklm.queue import gauntlet as g
from attacklm.queue.db import QueueDB


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "_calibration_holdouts_missing", lambda: False)
    return QueueDB(tmp_path / "q.db")


def _chain_ns(db, **over):
    ns = dict(db_path=str(db.db_path), base_model=None, single_model=True, single_model_name=None,
              include_orchestrator=False, model_attacks=False, include_tools=False, epochs=None,
              batch_size=None, train_extra=None, label=None, train_timeout=None, then="gauntlet",
              gauntlet_preset="quick", attack=None, include_unshipped=False, no_baseline=False)
    ns.update(over)
    return argparse.Namespace(**ns)


class TestBaselineTaskDefs:
    def test_defs_are_tagged_and_carry_base_model(self):
        defs = bl.baseline_task_defs("hf/base", "quick")
        assert [d["type"] for d in defs] == ["audit_prompt_injection", "audit_system_prompt"]
        for d in defs:
            assert d["gauntlet"] == "baseline"
            assert d["args"]["base_model"] == "hf/base" and "adapter" not in d["args"]
            assert d["depends_on"] == []
            assert d["label"] == "Baseline: quick on hf/base"


class TestEnsureBaseline:
    def test_inserts_once(self, db):
        ids = bl.ensure_baseline(db, "hf/base", "quick", qcli._insert_task_defs)
        assert len(ids) == 2
        assert bl.baseline_exists(db, "hf/base", "quick") == ids
        assert bl.ensure_baseline(db, "hf/base", "quick", qcli._insert_task_defs) == []

    def test_partial_baseline_counts_as_missing(self, db):
        ids = bl.ensure_baseline(db, "hf/base", "quick", qcli._insert_task_defs)
        db.remove_task(ids[1])
        assert bl.baseline_exists(db, "hf/base", "quick") == []

    def test_failed_baseline_is_not_reused(self, db):
        ids = bl.ensure_baseline(db, "hf/base", "quick", qcli._insert_task_defs)
        db.mark_task(ids[0], status="failed", error="x")
        assert bl.baseline_exists(db, "hf/base", "quick") == []


class TestChainAndGauntletAutoBaseline:
    def test_chain_queues_baseline_with_default_base(self, db):
        assert qcli._cmd_chain(_chain_ns(db)) == 0
        tasks = db.list_tasks()
        bases = [t for t in tasks if t.gauntlet == "baseline"]
        assert len(bases) == 2 and all(t.args_dict["base_model"] == bl.DEFAULT_BASE_MODEL for t in bases)
        assert all(t.depends_on_list == [] for t in bases)
        real = [t for t in tasks if t.gauntlet == "quick"]
        assert len(real) == 2 and all(t.depends_on_list == [1] for t in real)

    def test_chain_uses_explicit_base(self, db):
        assert qcli._cmd_chain(_chain_ns(db, base_model="my/base")) == 0
        assert {t.args_dict["base_model"] for t in db.list_tasks() if t.gauntlet == "baseline"} == {"my/base"}

    def test_no_baseline_flag(self, db):
        assert qcli._cmd_chain(_chain_ns(db, no_baseline=True)) == 0
        assert not [t for t in db.list_tasks() if t.gauntlet == "baseline"]

    def test_second_chain_does_not_duplicate(self, db):
        qcli._cmd_chain(_chain_ns(db))
        qcli._cmd_chain(_chain_ns(db))
        assert len([t for t in db.list_tasks() if t.gauntlet == "baseline"]) == 2

    def test_gauntlet_after_train_queues_baseline_from_train_args(self, db):
        train = db.add_task(type="train", label="t", args={"base_model": "my/base"})
        ns = argparse.Namespace(db_path=str(db.db_path), preset="quick", after=str(train), recipe=None,
                                no_baseline=False, base_model=None)
        assert qcli._cmd_gauntlet(ns) == 0
        assert {t.args_dict["base_model"] for t in db.list_tasks() if t.gauntlet == "baseline"} == {"my/base"}

    def test_gauntlet_without_train_and_without_base_skips_baseline_with_notice(self, db, capsys):
        ns = argparse.Namespace(db_path=str(db.db_path), preset="quick", after=None, recipe=None,
                                no_baseline=False, base_model=None)
        assert qcli._cmd_gauntlet(ns) == 0
        assert not [t for t in db.list_tasks() if t.gauntlet == "baseline"]
        assert "no baseline queued" in capsys.readouterr().out.lower()


class TestBaselineCommand:
    def test_creates_then_reports_existing(self, db, capsys):
        ns = argparse.Namespace(db_path=str(db.db_path), base_model="hf/base", preset="quick", force=False)
        assert qcli._cmd_baseline(ns) == 0
        assert len([t for t in db.list_tasks() if t.gauntlet == "baseline"]) == 2
        capsys.readouterr()
        assert qcli._cmd_baseline(ns) == 0
        assert "already" in capsys.readouterr().out
        assert len([t for t in db.list_tasks() if t.gauntlet == "baseline"]) == 2

    def test_force_adds_fresh(self, db):
        ns = argparse.Namespace(db_path=str(db.db_path), base_model="hf/base", preset="quick", force=True)
        qcli._cmd_baseline(ns)
        qcli._cmd_baseline(ns)
        assert len([t for t in db.list_tasks() if t.gauntlet == "baseline"]) == 4
