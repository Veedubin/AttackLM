"""Tests for attacklm.queue — hermetic, uses temp databases, monkeypatched subprocess."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure the src directory is on the path.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from attacklm.queue.db import QueueDB  # noqa: E402 — must come after the sys.path.insert above
from attacklm.queue.registry import REGISTRY, resolve_attack  # noqa: E402
from attacklm.queue.argv import _resolve_argv, _args_to_argv  # noqa: E402
from attacklm.queue.gauntlet import expand_gauntlet  # noqa: E402
from attacklm.queue.runner import (  # noqa: E402
    _execute_task,
    _recover_interrupted,
    _grep_adapter_path,
)
from attacklm.queue.migrations import apply_migrations  # noqa: E402


@pytest.fixture
def db(tmp_path):
    """Create a temporary QueueDB for each test."""
    db_path = tmp_path / "queue.db"
    return QueueDB(db_path)


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------


class TestMigrations:
    def test_fresh_db_creates_schema_v1(self, tmp_path):
        db_path = tmp_path / "test.db"
        version = apply_migrations(db_path)
        assert version == 1

    def test_schema_version_table_exists(self, db):
        with db._conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()
            assert row[0] >= 1

    def test_tasks_table_exists(self, db):
        with db._conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()
            assert row[0] == 0  # Empty but exists


# ---------------------------------------------------------------------------
# DB CRUD
# ---------------------------------------------------------------------------


class TestQueueDB:
    def test_add_task(self, db):
        task_id = db.add_task(
            type="train",
            label="Train test",
            args={"single_model": True, "epochs": 3},
        )
        assert task_id == 1
        task = db.get_task(task_id)
        assert task is not None
        assert task.type == "train"
        assert task.label == "Train test"
        assert task.status == "pending"

    def test_add_task_with_deps(self, db):
        train_id = db.add_task(type="train", label="Train", args={})
        audit_id = db.add_task(
            type="audit_prompt_injection",
            label="Audit PI",
            args={},
            depends_on=[train_id],
        )
        task = db.get_task(audit_id)
        assert task.depends_on_list == [train_id]

    def test_list_tasks(self, db):
        db.add_task(type="train", label="T1", args={})
        db.add_task(type="train", label="T2", args={})
        tasks = db.list_tasks()
        assert len(tasks) == 2

    def test_list_tasks_filter_status(self, db):
        id1 = db.add_task(type="train", label="T1", args={})
        db.mark_task(id1, status="completed")
        db.add_task(type="train", label="T2", args={})
        pending = db.list_tasks(status="pending")
        assert len(pending) == 1

    def test_mark_task_running(self, db):
        id1 = db.add_task(type="train", label="T1", args={})
        db.mark_task(id1, status="running", pid=12345)
        task = db.get_task(id1)
        assert task.status == "running"
        assert task.pid == 12345

    def test_mark_task_completed(self, db):
        id1 = db.add_task(type="train", label="T1", args={})
        db.mark_task(
            id1,
            status="completed",
            artifact_path="/fake/path",
            result='{"kind":"adapter"}',
        )
        task = db.get_task(id1)
        assert task.status == "completed"
        assert task.artifact_path == "/fake/path"

    def test_mark_task_failed(self, db):
        id1 = db.add_task(type="train", label="T1", args={})
        db.mark_task(id1, status="failed", error="rc=1")
        task = db.get_task(id1)
        assert task.status == "failed"
        assert task.error == "rc=1"

    def test_remove_task(self, db):
        id1 = db.add_task(type="train", label="T1", args={})
        assert db.remove_task(id1) is True
        assert db.get_task(id1) is None

    def test_remove_task_force(self, db):
        id1 = db.add_task(type="train", label="T1", args={})
        db.mark_task(id1, status="completed")
        assert db.remove_task(id1) is False  # Can't remove completed without force
        assert db.remove_task(id1, force=True) is True

    def test_clean_tasks(self, db):
        id1 = db.add_task(type="train", label="T1", args={})
        db.mark_task(id1, status="completed")
        count = db.clean_tasks(status="completed", yes=True)
        assert count == 1
        assert db.get_task(id1) is None

    def test_reset_db(self, db):
        db.add_task(type="train", label="T1", args={})
        db.reset()
        # DB should be empty after reset.
        tasks = db.list_tasks()
        assert len(tasks) == 0

    def test_add_and_get_events(self, db):
        id1 = db.add_task(type="train", label="T1", args={})
        db.add_event(id1, "created", {"type": "train"})
        events = db.get_events(id1)
        assert len(events) >= 1

    def test_runner_state(self, db):
        db.set_runner_state("pid", "12345")
        assert db.get_runner_state("pid") == "12345"
        db.set_runner_state("heartbeat", "2026-01-01T00:00:00")
        state = db.get_all_runner_state()
        assert "pid" in state


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_contents_are_pinned(self):
        """Guard against a task type appearing or vanishing unnoticed.

        Asserts the expected set rather than a bare count, so an addition
        fails with the offending name in the diff instead of "10 != 9".
        """
        assert set(REGISTRY) == {
            "train",
            "gen_calibration_holdouts",
            "audit_prompt_injection",
            "audit_system_prompt",
            "audit_canary_pipeline",
            "audit_calibration",
            "audit_gcg",
            "audit_backdoor",
            "audit_repeated_sampling",
            "bench_cybermetric",
            "bench_ctibench_mcq",
            "bench_ctibench_ate",
            "bench_secbench_en",
            "bench_seceval",
        }

    def test_train_spec(self):
        spec = REGISTRY["train"]
        assert spec.type == "train"
        assert spec.produces_artifact == "adapter"
        assert spec.implemented is True
        assert spec.default_timeout_s is None

    def test_unimplemented_specs(self):
        for key in ["audit_gcg", "audit_backdoor", "audit_repeated_sampling"]:
            spec = REGISTRY[key]
            assert spec.implemented is False

    def test_resolve_attack_all(self):
        keys = resolve_attack("all")
        assert len(keys) == 4  # core 4 shipped

    def test_resolve_attack_all_with_unshipped(self):
        keys = resolve_attack("all", include_unshipped=True)
        assert len(keys) == 7

    def test_resolve_attack_by_number(self):
        assert resolve_attack("1") == ["audit_prompt_injection"]
        assert resolve_attack("7") == ["audit_calibration"]

    def test_resolve_attack_by_name(self):
        assert resolve_attack("gcg", include_unshipped=True) == ["audit_gcg"]

    def test_resolve_attack_unimplemented_requires_flag(self):
        with pytest.raises(ValueError, match="not yet implemented"):
            resolve_attack("4")

    def test_resolve_attack_unimplemented_with_flag(self):
        keys = resolve_attack("4", include_unshipped=True)
        assert keys == ["audit_gcg"]


# ---------------------------------------------------------------------------
# Argv
# ---------------------------------------------------------------------------


class TestArgv:
    def test_args_to_argv_basic(self):
        args = {
            "base_model": "qwen-3b",
            "epochs": 3,
            "single_model": True,
        }
        argv = _args_to_argv(args, REGISTRY["train"].arg_schema)
        assert "--base-model" in argv
        assert "qwen-3b" in argv
        assert "--epochs" in argv
        assert "--single-model" in argv
        assert "3" in argv

    def test_args_to_argv_boolean_flags(self):
        args = {"single_model": True, "include_orchestrator": False}
        argv = _args_to_argv(args, REGISTRY["train"].arg_schema)
        assert "--single-model" in argv
        assert "--include-orchestrator" not in argv

    def test_args_to_argv_extra_argv(self):
        args = {"extra_argv": ["--ga-lore", "--spectrum"]}
        argv = _args_to_argv(args, REGISTRY["train"].arg_schema)
        assert "--ga-lore" in argv
        assert "--spectrum" in argv

    def test_resolve_argv_with_adapter_from_dep(self, db):
        # Create a completed train task.
        train_id = db.add_task(
            type="train", label="Train", args={"base_model": "qwen-3b"}
        )
        db.mark_task(
            train_id,
            status="completed",
            artifact_path="/fake/adapter",
            result=json.dumps(
                {
                    "kind": "adapter",
                    "adapter_path": "/fake/adapter",
                    "base_model": "qwen-3b",
                }
            ),
        )

        # Create an audit task that depends on the train task.
        audit_id = db.add_task(
            type="audit_prompt_injection",
            label="Audit PI",
            args={},
            depends_on=[train_id],
        )
        task = db.get_task(audit_id)
        spec = REGISTRY["audit_prompt_injection"]

        argv = _resolve_argv(task, spec, db)
        assert argv is not None
        assert "--adapter" in argv
        adapter_idx = argv.index("--adapter")
        assert argv[adapter_idx + 1] == "/fake/adapter"

    def test_resolve_argv_no_adapter_fails(self, db):
        # Audit task with no deps and no explicit adapter → should fail.
        audit_id = db.add_task(
            type="audit_prompt_injection",
            label="Audit PI",
            args={},
        )
        task = db.get_task(audit_id)
        spec = REGISTRY["audit_prompt_injection"]
        argv = _resolve_argv(task, spec, db)
        assert argv is None  # Can't resolve adapter


# ---------------------------------------------------------------------------
# Gauntlet
# ---------------------------------------------------------------------------


class TestGauntlet:
    def test_core_preset_has_4_tasks_when_holdouts_exist(self):
        """When calibration holdouts already exist, core preset has exactly 4 audit tasks."""
        with patch(
            "attacklm.queue.gauntlet._calibration_holdouts_missing", return_value=False
        ):
            tasks = expand_gauntlet("core", after_task_id=1)
            assert len(tasks) == 4

    def test_core_preset_has_5_tasks_when_holdouts_missing(self):
        """When calibration holdouts are missing, core preset auto-inserts a gen task."""
        with patch(
            "attacklm.queue.gauntlet._calibration_holdouts_missing", return_value=True
        ):
            tasks = expand_gauntlet("core", after_task_id=1)
            assert len(tasks) == 5  # 1 holdout gen + 4 audits

    def test_full_preset_has_7_tasks_when_holdouts_exist(self):
        """When calibration holdouts already exist, full preset has 7 audit tasks."""
        with patch(
            "attacklm.queue.gauntlet._calibration_holdouts_missing", return_value=False
        ):
            tasks = expand_gauntlet("full", after_task_id=1)
            assert len(tasks) == 7

    def test_full_preset_has_8_tasks_when_holdouts_missing(self):
        """When calibration holdouts are missing, full preset auto-inserts a gen task."""
        with patch(
            "attacklm.queue.gauntlet._calibration_holdouts_missing", return_value=True
        ):
            tasks = expand_gauntlet("full", after_task_id=1)
            assert len(tasks) == 8  # 1 holdout gen + 7 audits

    def test_quick_preset_has_2_tasks(self):
        tasks = expand_gauntlet("quick", after_task_id=1)
        assert len(tasks) == 2

    def test_memorization_preset(self):
        tasks = expand_gauntlet("memorization", after_task_id=1)
        assert len(tasks) == 2

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown gauntlet preset"):
            expand_gauntlet("nonexistent")

    def test_all_tasks_have_depends_on(self):
        """When using after_task_id, all audit tasks should depend on it."""
        with patch(
            "attacklm.queue.gauntlet._calibration_holdouts_missing", return_value=False
        ):
            tasks = expand_gauntlet("core", after_task_id=5)
            for t in tasks:
                # Holdout gen tasks have no deps (can run anytime).
                # All other tasks depend on the train task.
                if t["type"] == "gen_calibration_holdouts":
                    continue
                assert 5 in t["depends_on"]

    def test_calibration_holdouts_auto_insert(self):
        """When calibration holdouts are missing, the gauntlet should auto-insert a gen task."""
        # This test assumes the holdout files don't exist (which they don't in a temp env).
        with patch(
            "attacklm.queue.gauntlet._calibration_holdouts_missing", return_value=True
        ):
            tasks = expand_gauntlet("core", after_task_id=1)
            # Should have 5 tasks: 1 holdout gen + 4 audits.
            holdout_tasks = [
                t for t in tasks if t["type"] == "gen_calibration_holdouts"
            ]
            assert len(holdout_tasks) == 1


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class TestRunner:
    def test_recover_interrupted_no_running(self, db):
        # No running tasks — should return empty.
        recovered = _recover_interrupted(db)
        assert recovered == []

    def test_recover_interrupted_with_stale_pid(self, db):
        # Add a running task with a fake PID that doesn't exist.
        task_id = db.add_task(type="train", label="T1", args={})
        db.update_task(task_id, status="running", pid=99999999)
        recovered = _recover_interrupted(db)
        assert task_id in recovered
        task = db.get_task(task_id)
        assert task.status == "interrupted"

    def test_execute_task_unimplemented(self, db):
        """Unimplemented tasks should be marked pending_unimplemented."""
        task_id = db.add_task(
            type="audit_gcg",
            label="Audit: GCG (4)",
            args={},
        )
        task = db.get_task(task_id)
        _execute_task(db, task)
        task = db.get_task(task_id)
        assert task.status == "pending_unimplemented"

    def test_execute_task_subprocess_mock(self, db, monkeypatch, tmp_path):
        """Test task execution with a mocked subprocess."""
        task_id = db.add_task(
            type="train",
            label="Train test",
            args={"single_model": True},
        )
        task = db.get_task(task_id)

        # Create a temp log dir and artifact dir.
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        artifact_dir = tmp_path / "artifacts" / str(task_id)
        artifact_dir.mkdir(parents=True)

        # Monkeypatch LOG_DIR and ARTIFACT_DIR to use temp paths.
        from attacklm.queue import runner

        monkeypatch.setattr(runner, "LOG_DIR", log_dir)
        monkeypatch.setattr(runner, "ARTIFACT_DIR", artifact_dir)

        # Mock _run_subprocess to return 0 (success) and write fake log.
        log_path = log_dir / f"{task_id}.log"
        log_path.write_text("Final adapter: /fake/adapter\n")

        def mock_run_subprocess(argv, log_path_arg, timeout_s=None, cwd=None, env=None):
            return 0

        monkeypatch.setattr(runner, "_run_subprocess", mock_run_subprocess)

        # Also mock _resolve_argv to return a simple command.
        def mock_resolve_argv(task, spec, db_arg):
            return [sys.executable, "-c", "print('ok')"]

        monkeypatch.setattr(runner, "_resolve_argv", mock_resolve_argv)

        _execute_task(db, task)

        task = db.get_task(task_id)
        assert task.status == "completed"


# ---------------------------------------------------------------------------
# Artifact collection
# ---------------------------------------------------------------------------


class TestArtifactCollection:
    def test_grep_adapter_path(self, tmp_path):
        log_file = tmp_path / "1.log"
        log_file.write_text(
            "Some output\nFinal adapter: /path/to/adapter\nMore output\n"
        )
        result = _grep_adapter_path(log_file)
        assert result == "/path/to/adapter"

    def test_grep_adapter_path_not_found(self, tmp_path):
        log_file = tmp_path / "2.log"
        log_file.write_text("Some output\nNo adapter here\n")
        result = _grep_adapter_path(log_file)
        assert result is None

    def test_grep_adapter_path_missing_file(self, tmp_path):
        log_file = tmp_path / "nonexistent.log"
        result = _grep_adapter_path(log_file)
        assert result is None


# ---------------------------------------------------------------------------
# Integration: Chain and Gauntlet via DB
# ---------------------------------------------------------------------------


class TestChainIntegration:
    def test_add_train(self, db):
        task_id = db.add_task(
            type="train",
            label="Train test",
            args={"single_model": True, "epochs": 3},
        )
        assert task_id == 1
        task = db.get_task(task_id)
        assert task.type == "train"
        assert task.status == "pending"

    def test_chain_train_then_audit(self, db):
        """Simulate: add train, then add audits depending on it."""
        train_id = db.add_task(
            type="train",
            label="Train",
            args={"single_model": True},
        )

        # Add audits depending on train.
        for attack_key in [
            "audit_prompt_injection",
            "audit_system_prompt",
            "audit_canary_pipeline",
            "audit_calibration",
        ]:
            spec = REGISTRY[attack_key]
            db.add_task(
                type=attack_key,
                label=spec.label,
                args={},
                depends_on=[train_id],
                timeout_seconds=spec.default_timeout_s,
            )

        tasks = db.list_tasks()
        assert len(tasks) == 5  # 1 train + 4 audits

        # Train has no deps.
        train = db.get_task(train_id)
        assert train.depends_on_list == []

        # Audits depend on train.
        audit = db.get_task(train_id + 1)
        assert train_id in audit.depends_on_list

    def test_blocked_cascade(self, db):
        """If a dep fails, dependents become blocked."""
        train_id = db.add_task(type="train", label="Train", args={})
        audit_id = db.add_task(
            type="audit_prompt_injection",
            label="Audit",
            args={},
            depends_on=[train_id],
        )

        # Train fails.
        db.mark_task(train_id, status="failed", error="rc=1")

        # Recheck blocked.
        blocked = db.recompute_blocked()
        assert audit_id in blocked

        audit = db.get_task(audit_id)
        assert audit.status == "blocked"

    def test_unimplemented_skip(self, db):
        """Unimplemented attacks are marked pending_unimplemented."""
        task_id = db.add_task(
            type="audit_gcg",
            label="Audit: GCG (4)",
            args={},
        )
        task = db.get_task(task_id)
        _execute_task(db, task)

        task = db.get_task(task_id)
        assert task.status == "pending_unimplemented"

    def test_pick_next_ready_task_no_deps(self, db):
        """A task with no deps should be immediately eligible."""
        db.add_task(type="train", label="Train", args={"single_model": True})
        task = db.pick_next_ready_task()
        assert task is not None
        assert task.type == "train"
        assert task.status == "running"

    def test_pick_next_ready_task_with_unmet_deps(self, db):
        """A task with unmet deps should not be picked."""
        train_id = db.add_task(type="train", label="Train", args={})
        db.add_task(
            type="audit_prompt_injection",
            label="Audit",
            args={},
            depends_on=[train_id],
        )
        # Train is pending (not completed), so audit should not be picked.
        # But train itself should be picked (no deps).
        task = db.pick_next_ready_task()
        assert task is not None
        assert task.type == "train"

    def test_pick_next_ready_task_with_met_deps(self, db):
        """A task whose deps are completed should be eligible."""
        train_id = db.add_task(type="train", label="Train", args={})
        db.mark_task(
            train_id,
            status="completed",
            artifact_path="/fake/adapter",
            result=json.dumps({"kind": "adapter", "adapter_path": "/fake/adapter"}),
        )

        audit_id = db.add_task(
            type="audit_prompt_injection",
            label="Audit",
            args={},
            depends_on=[train_id],
        )

        task = db.pick_next_ready_task()
        assert task is not None
        assert task.id == audit_id


# ---------------------------------------------------------------------------
# Calibration holdouts script (basic test)
# ---------------------------------------------------------------------------


class TestCalibrationHoldouts:
    def test_gen_calibration_holdouts_imports(self):
        """Verify the script can be imported."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "gen_calibration_holdouts",
            str(
                Path(__file__).resolve().parent.parent
                / "scripts"
                / "gen_calibration_holdouts.py"
            ),
        )
        assert spec is not None


# ---------------------------------------------------------------------------
# Regression: duplicate holdout task + empty-deps respect
# ---------------------------------------------------------------------------


class TestGauntletNoDuplicateHoldout:
    """Regression tests for the duplicate-holdout bug and empty-deps fix.

    Bug: `gauntlet full` (and `chain ... --then gauntlet core`) created
    TWO gen_calibration_holdouts tasks because expand_gauntlet() inserts
    one and the CLI's auto-insert logic created another. Also, the
    `or [train_id]` pattern in the CLI clobbered empty `depends_on=[]`
    on the holdout task, wrongly binding it to the train task.
    """

    def test_gauntlet_full_has_one_holdout_when_missing(self, monkeypatch):
        """expand_gauntlet('full') with missing holdouts → exactly 1 holdout task."""
        monkeypatch.setattr(
            "attacklm.queue.gauntlet._calibration_holdouts_missing",
            lambda: True,
        )
        tasks = expand_gauntlet("full", after_task_id=42)
        holdouts = [t for t in tasks if t["type"] == "gen_calibration_holdouts"]
        assert len(holdouts) == 1, f"expected 1 holdout, got {len(holdouts)}"
        # The holdout task should have NO deps (it can run immediately)
        assert holdouts[0]["depends_on"] == [], (
            f"holdout should have empty deps, got {holdouts[0]['depends_on']}"
        )
        # The calibration audit should have the _needs_holdout_dep flag
        calib = [t for t in tasks if t["type"] == "audit_calibration"]
        assert len(calib) == 1
        assert calib[0].get("_needs_holdout_dep") is True

    def test_gauntlet_core_has_one_holdout_when_missing(self, monkeypatch):
        """expand_gauntlet('core') with missing holdouts → exactly 1 holdout task."""
        monkeypatch.setattr(
            "attacklm.queue.gauntlet._calibration_holdouts_missing",
            lambda: True,
        )
        tasks = expand_gauntlet("core", after_task_id=1)
        holdouts = [t for t in tasks if t["type"] == "gen_calibration_holdouts"]
        assert len(holdouts) == 1

    def test_gauntlet_no_holdout_when_files_exist(self, monkeypatch):
        """expand_gauntlet('core') with existing holdouts → 0 holdout tasks."""
        monkeypatch.setattr(
            "attacklm.queue.gauntlet._calibration_holdouts_missing",
            lambda: False,
        )
        tasks = expand_gauntlet("core", after_task_id=1)
        holdouts = [t for t in tasks if t["type"] == "gen_calibration_holdouts"]
        assert len(holdouts) == 0
        # Calibration audit should NOT have the _needs_holdout_dep flag
        calib = [t for t in tasks if t["type"] == "audit_calibration"]
        assert len(calib) == 1
        assert "_needs_holdout_dep" not in calib[0]

    def test_full_gauntlet_task_count(self, monkeypatch):
        """full gauntlet with missing holdouts = 1 holdout + 7 audits = 8 tasks."""
        monkeypatch.setattr(
            "attacklm.queue.gauntlet._calibration_holdouts_missing",
            lambda: True,
        )
        tasks = expand_gauntlet("full", after_task_id=1)
        assert len(tasks) == 8, f"expected 8 tasks, got {len(tasks)}"

    def test_core_gauntlet_task_count_with_missing_holdouts(self, monkeypatch):
        """core gauntlet with missing holdouts = 1 holdout + 4 audits = 5 tasks."""
        monkeypatch.setattr(
            "attacklm.queue.gauntlet._calibration_holdouts_missing",
            lambda: True,
        )
        tasks = expand_gauntlet("core", after_task_id=1)
        assert len(tasks) == 5, f"expected 5 tasks, got {len(tasks)}"


# ---------------------------------------------------------------------------
# Task 1 regressions — runner/argv correctness (2026-09-22 review)
# ---------------------------------------------------------------------------

from attacklm.queue.argv import read_adapter_base, is_adapter_dir  # noqa: E402


def _fake_adapter(tmp_path, base="huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated"):
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}))
    return d


class TestAdapterConfigResolution:
    def test_read_adapter_base(self, tmp_path):
        d = _fake_adapter(tmp_path)
        assert read_adapter_base(d) == "huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated"

    def test_read_adapter_base_missing(self, tmp_path):
        assert read_adapter_base(tmp_path / "nope") is None

    def test_is_adapter_dir(self, tmp_path):
        assert is_adapter_dir(_fake_adapter(tmp_path))
        assert not is_adapter_dir(tmp_path)

    def test_base_model_pointing_at_adapter_is_swapped(self, db, tmp_path):
        """S2: the 2026-07-19 smoke test passed an adapter dir as --base-model."""
        d = _fake_adapter(tmp_path)
        tid = db.add_task(type="audit_prompt_injection", label="a", args={"base_model": str(d)})
        argv = _resolve_argv(db.get_task(tid), REGISTRY["audit_prompt_injection"], db)
        assert argv[argv.index("--adapter") + 1] == str(d)
        assert argv[argv.index("--base-model") + 1] == "huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated"

    def test_base_model_inherited_from_adapter_config_when_train_had_none(self, db, tmp_path):
        """R6: train task stored base_model='' → read it from the adapter."""
        d = _fake_adapter(tmp_path, base="qwen-from-config")
        train_id = db.add_task(type="train", label="t", args={})
        db.mark_task(train_id, status="completed", artifact_path=str(d),
                     result=json.dumps({"kind": "adapter", "adapter_path": str(d), "base_model": ""}))
        aid = db.add_task(type="audit_prompt_injection", label="a", args={}, depends_on=[train_id])
        argv = _resolve_argv(db.get_task(aid), REGISTRY["audit_prompt_injection"], db)
        assert argv[argv.index("--base-model") + 1] == "qwen-from-config"

    def test_adapter_is_optional_for_merged_model(self, db):
        """Task 10 GPU smoke test: a merged model has no adapter at all —
        every audit script declares --adapter optional, only --base-model
        is required. No deps, no --adapter given: must not fail."""
        tid = db.add_task(type="audit_prompt_injection", label="a", args={"base_model": "b"})
        argv = _resolve_argv(db.get_task(tid), REGISTRY["audit_prompt_injection"], db)
        assert argv is not None
        assert argv[argv.index("--base-model") + 1] == "b"
        assert "--adapter" not in argv


class TestResolvedArgsPersist:
    def test_default_output_is_persisted(self, db, tmp_path):
        """R4: the default --output must be visible to _collect_artifact."""
        d = _fake_adapter(tmp_path)
        tid = db.add_task(type="audit_prompt_injection", label="a",
                          args={"base_model": "b", "adapter": str(d)})
        argv = _resolve_argv(db.get_task(tid), REGISTRY["audit_prompt_injection"], db)
        out = argv[argv.index("--output") + 1]
        assert db.get_task(tid).args_dict["output"] == out
        assert out.endswith(f"artifacts/{tid}/audit_prompt_injection_report.json")


class TestGrepAdapterPathFormats:
    def test_indented_final_adapter(self, tmp_path):
        p = tmp_path / "l.log"
        p.write_text("  STAGE 2 OK\n  Final adapter: /m/curriculum\n")
        assert _grep_adapter_path(p) == "/m/curriculum"

    def test_single_model_adapter_line(self, tmp_path):
        """R3: single-model runs print '  Adapter: <path>' not 'Final adapter:'."""
        p = tmp_path / "l.log"
        p.write_text("  OK — single model complete in 3.0 min\n  Adapter: /m/single\n")
        assert _grep_adapter_path(p) == "/m/single"

    def test_last_match_wins(self, tmp_path):
        p = tmp_path / "l.log"
        p.write_text("  Adapter: /m/one\n...\n  Adapter: /m/two\n")
        assert _grep_adapter_path(p) == "/m/two"


class TestTimeoutMarksFailed:
    def test_timeout_marks_task_failed_not_crash(self, db, tmp_path, monkeypatch):
        """R2: TimeoutExpired path called mark_task(finished_at=...) → TypeError."""
        from attacklm.queue import runner
        monkeypatch.setattr(runner, "LOG_DIR", tmp_path / "logs")
        monkeypatch.setattr(runner, "ARTIFACT_DIR", tmp_path / "artifacts")
        tid = db.add_task(type="gen_calibration_holdouts", label="h", args={}, timeout_seconds=1)

        def boom(argv, log_path, timeout_s=None, cwd=None, env=None):
            raise subprocess.TimeoutExpired(cmd=argv, timeout=timeout_s or 1)

        monkeypatch.setattr(runner, "_run_subprocess", boom)
        _execute_task(db, db.get_task(tid))  # must not raise
        t = db.get_task(tid)
        assert t.status == "failed"
        assert t.error == "timeout"
        assert t.finished_at is not None


# ---------------------------------------------------------------------------
# Task 2 — canary pipeline that matches the scripts (R5)
# ---------------------------------------------------------------------------


class TestCanaryPipeline:
    def test_registry_schema_matches_script(self):
        schema = REGISTRY["audit_canary_pipeline"].arg_schema
        assert set(schema) == {"canaries", "num_canaries", "base_model", "adapter", "output", "max_new_tokens"}

    def test_steps_generate_then_probe(self, db, tmp_path, monkeypatch):
        from attacklm.queue.argv import resolve_steps
        from attacklm.queue import argv as argv_mod
        monkeypatch.setattr(argv_mod, "DEFAULT_QUEUE_DIR", tmp_path)
        d = _fake_adapter(tmp_path)
        tid = db.add_task(type="audit_canary_pipeline", label="c",
                          args={"base_model": "b", "adapter": str(d), "num_canaries": 7})
        steps = resolve_steps(db.get_task(tid), REGISTRY["audit_canary_pipeline"], db)
        assert len(steps) == 2
        gen, probe = steps
        assert gen[1].endswith("canary_generator.py")
        assert gen[gen.index("--count") + 1] == "7"
        canaries = gen[gen.index("--output") + 1]
        assert canaries.endswith(f"artifacts/{tid}/canaries.jsonl")
        assert probe[1].endswith("audit_canary_extraction.py")
        assert probe[probe.index("--canaries") + 1] == canaries
        for bad in ("--num-canaries", "--canary-format", "--inject-split"):
            assert bad not in probe

    def test_steps_skip_generation_when_canaries_given(self, db, tmp_path):
        from attacklm.queue.argv import resolve_steps
        d = _fake_adapter(tmp_path)
        tid = db.add_task(type="audit_canary_pipeline", label="c",
                          args={"base_model": "b", "adapter": str(d), "canaries": "data/canaries.jsonl"})
        steps = resolve_steps(db.get_task(tid), REGISTRY["audit_canary_pipeline"], db)
        assert len(steps) == 1
        assert steps[0][steps[0].index("--canaries") + 1] == "data/canaries.jsonl"

    def test_retry_regenerates_canaries(self, db, tmp_path, monkeypatch):
        """IMPORTANT #2 (final review): resolve_steps persists the
        auto-generated canaries path; a naive retry would see that path as
        already-supplied and skip straight to the probe against a file that
        may no longer exist. canaries_generated must force regeneration on
        every resolve, including after retry."""
        from attacklm.queue.argv import resolve_steps
        from attacklm.queue import argv as argv_mod
        from attacklm.queue.cli import _cmd_retry
        import argparse
        monkeypatch.setattr(argv_mod, "DEFAULT_QUEUE_DIR", tmp_path)
        d = _fake_adapter(tmp_path)
        tid = db.add_task(type="audit_canary_pipeline", label="c",
                          args={"base_model": "b", "adapter": str(d)})

        steps = resolve_steps(db.get_task(tid), REGISTRY["audit_canary_pipeline"], db)
        assert len(steps) == 2
        assert db.get_task(tid).args_dict.get("canaries_generated") is True

        db.mark_task(tid, status="failed", error="rc=1")
        assert _cmd_retry(argparse.Namespace(db_path=str(db.db_path), task_id=tid)) == 0

        steps2 = resolve_steps(db.get_task(tid), REGISTRY["audit_canary_pipeline"], db)
        assert len(steps2) == 2
        assert steps2[0][1].endswith("canary_generator.py")
        # canaries_generated must never leak into the probe's CLI argv.
        assert "--canaries-generated" not in steps2[1]

    def test_subprocess_mode_is_single_step(self, db, tmp_path):
        from attacklm.queue.argv import resolve_steps
        d = _fake_adapter(tmp_path)
        tid = db.add_task(type="audit_prompt_injection", label="a", args={"base_model": "b", "adapter": str(d)})
        steps = resolve_steps(db.get_task(tid), REGISTRY["audit_prompt_injection"], db)
        assert len(steps) == 1 and steps[0][1].endswith("audit_prompt_injection.py")

    def test_run_steps_stops_on_first_failure(self, tmp_path):
        from attacklm.queue.runner import _run_steps
        log = tmp_path / "s.log"
        steps = [[sys.executable, "-c", "print('one')"],
                 [sys.executable, "-c", "import sys; sys.exit(3)"],
                 [sys.executable, "-c", "print('never')"]]
        assert _run_steps(steps, log, timeout_s=30, cwd=str(tmp_path), env=None) == 3
        text = log.read_text()
        assert "one" in text and "never" not in text


# ---------------------------------------------------------------------------
# Task 3 — DB correctness (R1, R7, R9, R8-db)
# ---------------------------------------------------------------------------


class TestDbFixes:
    def test_status_does_not_claim(self, db):
        """R1: status_summary must not flip a pending task to running."""
        from attacklm.queue.display import status_summary
        tid = db.add_task(type="gen_calibration_holdouts", label="h", args={})
        status_summary(db)
        assert db.get_task(tid).status == "pending"

    def test_find_next_ready_is_read_only(self, db):
        tid = db.add_task(type="gen_calibration_holdouts", label="h", args={})
        assert db.find_next_ready_task().id == tid
        assert db.get_task(tid).status == "pending"

    def test_pick_uses_rowcount(self, db, monkeypatch):
        """R7: if another runner claimed the row first, we must not return it."""
        tid = db.add_task(type="gen_calibration_holdouts", label="h", args={})
        real_connect = db._connect

        def racy_connect():
            conn = real_connect()
            orig = conn.execute

            def execute(sql, *p):
                if sql.startswith("UPDATE tasks SET status = 'running'"):
                    # Simulate the other runner winning the race.
                    orig("UPDATE tasks SET status = 'running' WHERE id = ?", (tid,))
                return orig(sql, *p)

            conn.execute = execute
            return conn

        monkeypatch.setattr(db, "_connect", racy_connect)
        assert db.pick_next_ready_task() is None

    def test_clean_requires_yes(self, db):
        tid = db.add_task(type="gen_calibration_holdouts", label="h", args={})
        db.mark_task(tid, status="completed")
        assert db.clean_tasks(status="completed", yes=False) == 1
        assert db.get_task(tid) is not None
        assert db.clean_tasks(status="completed", yes=True) == 1
        assert db.get_task(tid) is None

    def test_clean_keeps_deps_of_pending_tasks(self, db):
        """R9: deleting a completed train task must not strand its pending audits."""
        train = db.add_task(type="train", label="t", args={})
        db.mark_task(train, status="completed")
        db.add_task(type="audit_prompt_injection", label="a", args={}, depends_on=[train])
        assert db.clean_tasks(status="completed", yes=True) == 0
        assert db.get_task(train) is not None

    def test_remove_refuses_running_even_with_force(self, db):
        """R8: deleting a running row makes the runner's next mark_task raise."""
        tid = db.add_task(type="gen_calibration_holdouts", label="h", args={})
        db.mark_task(tid, status="running", pid=os.getpid())
        assert db.remove_task(tid, force=True) is False
        assert db.get_task(tid) is not None

    def test_clean_protects_failed_and_interrupted_dependents(self, db):
        """IMPORTANT #3 (final review): a completed train task must not be
        cleaned while a failed/interrupted audit still depends on it —
        those statuses are retryable via `queue retry`, and retrying would
        find the dependency gone."""
        train = db.add_task(type="train", label="t", args={})
        db.mark_task(train, status="completed")
        audit = db.add_task(type="audit_prompt_injection", label="a", args={}, depends_on=[train])
        db.mark_task(audit, status="failed", error="rc=1")
        assert db.clean_tasks(status="completed", yes=True) == 0
        assert db.get_task(train) is not None

    def test_recompute_blocked_missing_dep(self, db):
        """IMPORTANT #3 (final review): a dep row that no longer exists
        (e.g. removed via `queue remove --force`) must not be silently
        ignored — the dependent gets marked blocked with a clear error
        instead of sitting pending forever."""
        tid = db.add_task(type="audit_prompt_injection", label="a", args={}, depends_on=[999])
        blocked = db.recompute_blocked()
        assert tid in blocked
        t = db.get_task(tid)
        assert t.status == "blocked"
        assert t.error == "dependency #999 no longer exists"


# ---------------------------------------------------------------------------
# Task 4 — queue CLI fixes + dedupe (R8-cli, R11, R12, R13, S1, S4)
# ---------------------------------------------------------------------------


class TestCliFixes:
    def _ns(self, **kw):
        import argparse
        return argparse.Namespace(**kw)

    def test_gauntlet_after_keeps_all_ids(self, db, tmp_path, monkeypatch):
        """R11: --after 1,2 must make every audit depend on both."""
        from attacklm.queue import cli as qcli
        from attacklm.queue import gauntlet as g
        monkeypatch.setattr(g, "_calibration_holdouts_missing", lambda: False)
        t1 = db.add_task(type="train", label="t1", args={})
        t2 = db.add_task(type="train", label="t2", args={})
        args = self._ns(db_path=str(db.db_path), preset="quick", after=f"{t1},{t2}", recipe=None,
                        no_baseline=True)
        assert qcli._cmd_gauntlet(args) == 0
        audits = db.list_tasks(type="audit_prompt_injection")
        assert audits and sorted(audits[0].depends_on_list) == [t1, t2]

    def test_add_audit_label_is_used(self, db, monkeypatch):
        """R12"""
        from attacklm.queue import cli as qcli
        args = self._ns(db_path=str(db.db_path), attack="1", include_unshipped=False,
                        depends_on=None, adapter="/a", base_model="b", label="my run", timeout=None)
        assert qcli._cmd_add_audit(args) == 0
        assert db.list_tasks()[0].label == "my run"

    def test_retry_clears_stale_result(self, db):
        """R13"""
        from attacklm.queue import cli as qcli
        tid = db.add_task(type="gen_calibration_holdouts", label="h", args={})
        db.mark_task(tid, status="failed", error="x", artifact_path="/old", artifact_kind="data", result="{}")
        assert qcli._cmd_retry(self._ns(db_path=str(db.db_path), task_id=tid)) == 0
        t = db.get_task(tid)
        assert t.status == "pending"
        assert t.result is None and t.artifact_path is None and t.artifact_kind is None and t.error is None

    def test_remove_running_is_refused(self, db, capsys):
        """R8"""
        from attacklm.queue import cli as qcli
        tid = db.add_task(type="gen_calibration_holdouts", label="h", args={})
        db.mark_task(tid, status="running", pid=os.getpid())
        assert qcli._cmd_remove(self._ns(db_path=str(db.db_path), task_id=tid, force=True)) == 1
        assert "running" in capsys.readouterr().err
        assert db.get_task(tid) is not None

    def test_remove_force_cascade_spares_finished_dependents(self, db):
        """Minor (final review): --force cascade must only relabel
        pending/ready/blocked dependents — a completed/running dependent
        already has its own outcome and must not be overwritten."""
        from attacklm.queue import cli as qcli
        train = db.add_task(type="train", label="t", args={})
        done_audit = db.add_task(type="audit_prompt_injection", label="done", args={}, depends_on=[train])
        pending_audit = db.add_task(type="audit_system_prompt", label="pending", args={}, depends_on=[train])
        db.mark_task(done_audit, status="completed", artifact_path="/r.json")

        assert qcli._cmd_remove(self._ns(db_path=str(db.db_path), task_id=train, force=True)) == 0

        assert db.get_task(done_audit).status == "completed"
        assert db.get_task(pending_audit).status == "blocked"

    def test_os_imported_at_top(self):
        """S1"""
        import inspect
        from attacklm.queue import cli as qcli
        src = inspect.getsource(qcli)
        assert src.index("import os") < src.index("def _get_db")

    def test_chain_and_gauntlet_share_insert(self, db, monkeypatch):
        """S4: both commands go through _insert_task_defs."""
        from attacklm.queue import cli as qcli
        calls = []
        real = qcli._insert_task_defs

        def spy(db_, tasks, default_deps):
            calls.append(len(tasks))
            return real(db_, tasks, default_deps)

        monkeypatch.setattr(qcli, "_insert_task_defs", spy)
        from attacklm.queue import gauntlet as g
        monkeypatch.setattr(g, "_calibration_holdouts_missing", lambda: False)
        chain_args = self._ns(db_path=str(db.db_path), base_model=None, single_model=True,
                              single_model_name=None, include_orchestrator=False, model_attacks=False,
                              include_tools=False, epochs=None, batch_size=None, train_extra=None,
                              label=None, train_timeout=None, then="gauntlet", gauntlet_preset="quick",
                              attack=None, include_unshipped=False, no_baseline=True)
        assert qcli._cmd_chain(chain_args) == 0
        assert qcli._cmd_gauntlet(self._ns(db_path=str(db.db_path), preset="quick", after=None, recipe=None,
                                           no_baseline=True)) == 0
        assert calls == [2, 2]

    def test_clean_dry_run_message(self, db, capsys):
        """Controller addition: --yes-less clean must not claim it removed anything."""
        from attacklm.queue import cli as qcli
        tid = db.add_task(type="gen_calibration_holdouts", label="h", args={})
        db.mark_task(tid, status="completed")
        args = self._ns(db_path=str(db.db_path), status="completed", older_than=None, yes=False)
        assert qcli._cmd_clean(args) == 0
        out = capsys.readouterr().out
        assert "Would remove 1 task(s)" in out
        assert "Removed" not in out
        assert db.get_task(tid) is not None

    def test_clean_yes_removes_and_reports(self, db, capsys):
        """Controller addition: --yes still deletes and reports 'Removed'."""
        from attacklm.queue import cli as qcli
        tid = db.add_task(type="gen_calibration_holdouts", label="h", args={})
        db.mark_task(tid, status="completed")
        args = self._ns(db_path=str(db.db_path), status="completed", older_than=None, yes=True)
        assert qcli._cmd_clean(args) == 0
        out = capsys.readouterr().out
        assert "Removed 1 task(s)" in out
        assert db.get_task(tid) is None

    def test_detach_argv_propagates_force(self, db):
        """IMPORTANT #4 (final review): the detached child does its own
        'already running' check on startup — without --force it would
        refuse to start over the very runner --detach --force was meant
        to override."""
        from attacklm.queue import cli as qcli
        args = self._ns(db_path=str(db.db_path), poll_interval=0.05,
                        exit_when_idle=False, force=True)
        cmd = qcli._detach_argv(args)
        assert "--force" in cmd

    def test_detach_argv_omits_force_when_not_set(self, db):
        from attacklm.queue import cli as qcli
        args = self._ns(db_path=str(db.db_path), poll_interval=0.05,
                        exit_when_idle=False, force=False)
        cmd = qcli._detach_argv(args)
        assert "--force" not in cmd
