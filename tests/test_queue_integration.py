"""End-to-end queue test with REAL subprocesses and stub scripts.

The v0.18.0 unit tests mocked the subprocess layer and passed while the
real runner failed on every task. This file runs the actual run_loop
against tiny stand-in scripts so the log/artifact/argv contracts are
exercised for real. No GPU, no network, ~2 s.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from attacklm.queue import argv as argv_mod
from attacklm.queue import registry as registry_mod
from attacklm.queue import runner as runner_mod
from attacklm.queue.db import QueueDB
from attacklm.queue.runner import run_loop

TRAIN_STUB = """\
import argparse, json, sys, pathlib
p = argparse.ArgumentParser()
p.add_argument("--single-model", action="store_true")
p.add_argument("--base-model", default=None)
a, _ = p.parse_known_args()
out = pathlib.Path("models") / "stub-adapter"
out.mkdir(parents=True, exist_ok=True)
(out / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "stub/base-3b"}))
print("  OK — single model complete in 0.0 min")
print(f"  Adapter: {out.resolve()}")
"""

AUDIT_STUB = """\
import argparse, json, pathlib
p = argparse.ArgumentParser()
p.add_argument("--base-model", required=True)
p.add_argument("--adapter")
p.add_argument("--questions")
p.add_argument("--canaries")
p.add_argument("--in-distribution"); p.add_argument("--near-ood"); p.add_argument("--ood")
p.add_argument("--output", required=True)
p.add_argument("--max-new-tokens", type=int, default=64)
a = p.parse_args()
o = pathlib.Path(a.output); o.parent.mkdir(parents=True, exist_ok=True)
o.write_text(json.dumps({"base_model": a.base_model, "adapter": a.adapter, "canaries": a.canaries}))
"""

CANARY_GEN_STUB = """\
import argparse, pathlib
p = argparse.ArgumentParser()
p.add_argument("--output", required=True); p.add_argument("--count", type=int, required=True); p.add_argument("--seed", type=int)
a = p.parse_args()
o = pathlib.Path(a.output); o.parent.mkdir(parents=True, exist_ok=True)
o.write_text("\\n".join('{"token": "c%d"}' % i for i in range(a.count)) + "\\n")
"""

HOLDOUT_STUB = """\
import argparse, pathlib
p = argparse.ArgumentParser(); p.add_argument("--output-dir", default="data/bench"); a = p.parse_args()
d = pathlib.Path(a.output_dir); d.mkdir(parents=True, exist_ok=True)
for n in ("calibration_in", "calibration_near", "calibration_ood"):
    (d / f"{n}.jsonl").write_text('{"q": "x"}\\n')
"""


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A cwd with stub scripts/ and every queue path redirected under tmp_path."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "train_all.py").write_text(TRAIN_STUB)
    for name in ("audit_prompt_injection.py", "audit_system_prompt.py",
                 "eval_calibration.py", "audit_canary_extraction.py"):
        (scripts / name).write_text(AUDIT_STUB)
    (scripts / "canary_generator.py").write_text(CANARY_GEN_STUB)
    (scripts / "gen_calibration_holdouts.py").write_text(HOLDOUT_STUB)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(registry_mod, "_SCRIPTS_DIR", scripts)
    monkeypatch.setattr(argv_mod, "_SCRIPTS_DIR", scripts)
    queue_dir = tmp_path / "evals" / "queue"
    monkeypatch.setattr(argv_mod, "DEFAULT_QUEUE_DIR", queue_dir)
    monkeypatch.setattr(runner_mod, "DEFAULT_QUEUE_DIR", queue_dir)
    monkeypatch.setattr(runner_mod, "LOG_DIR", queue_dir / "logs")
    monkeypatch.setattr(runner_mod, "ARTIFACT_DIR", queue_dir / "artifacts")
    return QueueDB(queue_dir / "queue.db")


def _drain(db):
    run_loop(db_path=db.db_path, poll_interval=0.05, exit_when_idle=True)


def test_train_then_core_gauntlet_end_to_end(sandbox):
    """The documented headline flow: chain --single-model --then gauntlet core."""
    from attacklm.queue.cli import _cmd_chain
    import argparse
    db = sandbox
    ns = argparse.Namespace(db_path=str(db.db_path), base_model=None, single_model=True,
                            single_model_name=None, include_orchestrator=False, model_attacks=False,
                            include_tools=False, epochs=None, batch_size=None, train_extra=None,
                            label=None, train_timeout=None, then="gauntlet", gauntlet_preset="core",
                            attack=None, include_unshipped=False)
    assert _cmd_chain(ns) == 0
    _drain(db)

    tasks = db.list_tasks()
    by_type = {t.type: t for t in tasks}
    assert {t.status for t in tasks} == {"completed"}, [(t.type, t.status, t.error) for t in tasks]

    train = by_type["train"]
    assert train.artifact_path.endswith("stub-adapter")
    assert json.loads(train.result)["base_model"] == "stub/base-3b"   # read from adapter_config.json

    for key in ("audit_prompt_injection", "audit_system_prompt", "audit_calibration", "audit_canary_pipeline"):
        t = by_type[key]
        report = json.loads(Path(t.artifact_path).read_text())
        assert report["base_model"] == "stub/base-3b"
        assert report["adapter"].endswith("stub-adapter")
        assert t.artifact_path == t.args_dict["output"]

    canaries = by_type["audit_canary_pipeline"].args_dict["canaries"]
    assert Path(canaries).exists() and len(Path(canaries).read_text().splitlines()) == 50
    assert json.loads(Path(by_type["audit_canary_pipeline"].artifact_path).read_text())["canaries"] == canaries


def test_adapter_dir_given_as_base_model(sandbox):
    """The exact mistake from the 2026-07-19 smoke test."""
    from attacklm.queue.cli import _cmd_add_audit
    import argparse
    db = sandbox
    adapter = Path("models/pretrained-adapter")
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "real/base"}))
    ns = argparse.Namespace(db_path=str(db.db_path), attack="1", include_unshipped=False, depends_on=None,
                            adapter=None, base_model=str(adapter), label=None, timeout=None)
    assert _cmd_add_audit(ns) == 0
    _drain(db)
    t = db.list_tasks()[0]
    assert t.status == "completed", t.error
    report = json.loads(Path(t.artifact_path).read_text())
    assert report == {"base_model": "real/base", "adapter": str(adapter), "canaries": None}


def test_audit_on_merged_model_without_adapter(sandbox):
    """Task 10 GPU smoke test regression: a merged model has no adapter at
    all. Every audit script declares --adapter optional — only --base-model
    is required — so this must succeed with no --adapter on the argv."""
    from attacklm.queue.cli import _cmd_add_audit
    import argparse
    db = sandbox
    ns = argparse.Namespace(db_path=str(db.db_path), attack="1", include_unshipped=False, depends_on=None,
                            adapter=None, base_model="stub/merged", label=None, timeout=None)
    assert _cmd_add_audit(ns) == 0
    _drain(db)
    t = db.list_tasks()[0]
    assert t.status == "completed", t.error
    report = json.loads(Path(t.artifact_path).read_text())
    assert report["base_model"] == "stub/merged"
    assert report["adapter"] is None


def test_failed_script_blocks_dependents(sandbox):
    db = sandbox
    (Path("scripts") / "train_all.py").write_text("import sys; print('boom'); sys.exit(2)\n")
    train = db.add_task(type="train", label="t", args={"single_model": True})
    audit = db.add_task(type="audit_prompt_injection", label="a", args={}, depends_on=[train])
    _drain(db)
    assert db.get_task(train).status == "failed"
    assert db.get_task(train).error == "rc=2"
    assert db.get_task(audit).status == "blocked"


def test_timeout_is_recorded(sandbox):
    db = sandbox
    (Path("scripts") / "gen_calibration_holdouts.py").write_text("import time; time.sleep(30)\n")
    tid = db.add_task(type="gen_calibration_holdouts", label="slow", args={}, timeout_seconds=1)
    _drain(db)
    t = db.get_task(tid)
    assert t.status == "failed" and t.error == "timeout"


def test_status_while_pending_does_not_consume(sandbox):
    from attacklm.queue.display import status_summary
    db = sandbox
    tid = db.add_task(type="gen_calibration_holdouts", label="h", args={})
    status_summary(db)
    _drain(db)
    assert db.get_task(tid).status == "completed"


def test_detach_spawns_background_runner(sandbox, monkeypatch):
    """--detach must return immediately and leave a runner that drains the queue."""
    from attacklm.queue.cli import _cmd_start
    import argparse
    import time
    db = sandbox
    tid = db.add_task(type="gen_calibration_holdouts", label="h", args={})
    ns = argparse.Namespace(db_path=str(db.db_path), force=False, follow=False, detach=True,
                            poll_interval=0.05, exit_when_idle=True)
    env_scripts = str(Path("scripts").resolve())
    monkeypatch.setenv("ATTACKLM_SCRIPTS_DIR", env_scripts)
    assert _cmd_start(ns) == 0
    for _ in range(100):
        if db.get_task(tid).status == "completed":
            break
        time.sleep(0.1)
    assert db.get_task(tid).status == "completed"
