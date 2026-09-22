"""Tests for attacklm.queue.compare on synthetic reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from attacklm.queue.compare import (
    Run, compare_runs, latest_subject, load_report, pair_items, render_table,
    select_run, to_json,
)
from attacklm.queue.db import QueueDB


def _pi_report(path: Path, base: str, adapter: str | None, asr_by_id: dict[str, float], tiers=None):
    tiers = tiers or {}
    results = [{"question_id": q, "tier": tiers.get(q, "direct"), "asr": v} for q, v in asr_by_id.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "metadata": {"base_model": base, "adapter": adapter},
        "summary": {"overall_asr": sum(asr_by_id.values()) / len(asr_by_id)},
        "results": results,
    }))
    return path


def _completed_audit(db, typ, base, adapter, report, gauntlet=None):
    args = {"base_model": base, "output": str(report)}
    if adapter:
        args["adapter"] = adapter
    tid = db.add_task(type=typ, label=typ, args=args, gauntlet=gauntlet)
    db.mark_task(tid, status="completed", artifact_path=str(report), artifact_kind="report")
    return tid


@pytest.fixture
def db(tmp_path):
    return QueueDB(tmp_path / "q.db")


class TestPairItems:
    def test_pairs_by_id_and_counts_unpaired(self):
        ra = {"results": [{"question_id": "q1", "tier": "direct", "asr": 0.0},
                          {"question_id": "q2", "tier": "indirect", "asr": 0.5},
                          {"question_id": "q9", "tier": "direct", "asr": 1.0}]}
        rb = {"results": [{"question_id": "q1", "tier": "direct", "asr": 1.0},
                          {"question_id": "q2", "tier": "indirect", "asr": 0.5},
                          {"question_id": "q8", "tier": "direct", "asr": 0.0}]}
        cats, unpaired = pair_items(ra, rb, "asr", "question_id", "tier")
        assert unpaired == 2
        assert cats["overall"] == ([0.0, 0.5], [1.0, 0.5])
        assert cats["direct"] == ([0.0], [1.0])
        assert cats["indirect"] == ([0.5], [0.5])

    def test_boolean_metrics_become_floats(self):
        ra = {"results": [{"canary_id": "c1", "prefix": "p", "exact_token": False}]}
        rb = {"results": [{"canary_id": "c1", "prefix": "p", "exact_token": True}]}
        cats, _ = pair_items(ra, rb, "exact_token", "canary_id", "prefix")
        assert cats["overall"] == ([0.0], [1.0])

    def test_item_missing_metric_key_is_unpaired(self):
        # Review ruling: a malformed report item that's missing the metric key
        # (on either side) must NOT silently score 0.0 -- it must drop out of
        # every category's arrays and be counted as unpaired instead.
        ra = {"results": [{"question_id": "q1", "tier": "direct", "asr": 0.0}]}
        rb = {"results": [{"question_id": "q1", "tier": "direct"}]}  # asr missing on B
        cats, unpaired = pair_items(ra, rb, "asr", "question_id", "tier")
        assert cats["overall"] == ([], [])
        assert "direct" not in cats
        assert unpaired == 1


class TestSelectRun:
    def test_selects_newest_completed_for_adapter(self, db, tmp_path):
        r1 = _pi_report(tmp_path / "1.json", "b", "/a", {"q1": 0.0})
        r2 = _pi_report(tmp_path / "2.json", "b", "/a", {"q1": 1.0})
        _completed_audit(db, "audit_prompt_injection", "b", "/a", r1)
        t2 = _completed_audit(db, "audit_prompt_injection", "b", "/a", r2)
        run = select_run(db, subject="/a")
        assert isinstance(run, Run)
        assert run.tasks["audit_prompt_injection"].id == t2
        assert run.base_model == "b" and run.adapter == "/a" and run.subject == "/a"

    def test_baseline_requires_tag_and_no_adapter(self, db, tmp_path):
        r = _pi_report(tmp_path / "1.json", "b", None, {"q1": 0.0})
        _completed_audit(db, "audit_prompt_injection", "b", None, r)             # untagged
        assert select_run(db, base_model="b") is None
        _completed_audit(db, "audit_prompt_injection", "b", None, r, gauntlet="baseline")
        assert select_run(db, base_model="b") is not None

    def test_none_when_nothing_matches(self, db):
        assert select_run(db, subject="/nope") is None

    def test_latest_subject(self, db, tmp_path):
        r = _pi_report(tmp_path / "1.json", "b", "/a", {"q1": 0.0})
        assert latest_subject(db) is None
        _completed_audit(db, "audit_prompt_injection", "b", None, r, gauntlet="baseline")
        assert latest_subject(db) is None                      # baselines don't count
        _completed_audit(db, "audit_prompt_injection", "b", "/a", r)
        assert latest_subject(db) == ("/a", "b")

    def test_subject_matches_merged_model_without_adapter(self, db, tmp_path):
        # A merged model has no adapter, so its identity is the base_model
        # itself. Untagged (not gauntlet="baseline") -> select_run(subject=)
        # must find it, but select_run(base_model=) must not (that path is
        # reserved for tagged baselines).
        r = _pi_report(tmp_path / "1.json", "models/merged/x", None, {"q1": 0.0})
        _completed_audit(db, "audit_prompt_injection", "models/merged/x", None, r)
        run = select_run(db, subject="models/merged/x")
        assert run is not None
        assert run.adapter is None and run.subject == "models/merged/x"
        assert run.base_model == "models/merged/x"
        assert latest_subject(db) == ("models/merged/x", "models/merged/x")
        assert select_run(db, base_model="models/merged/x") is None


class TestCompareRuns:
    def _two_runs(self, db, tmp_path, a_vals, b_vals):
        ra = _pi_report(tmp_path / "a.json", "b", None, a_vals)
        rb = _pi_report(tmp_path / "b.json", "b", "/x", b_vals)
        _completed_audit(db, "audit_prompt_injection", "b", None, ra, gauntlet="baseline")
        _completed_audit(db, "audit_prompt_injection", "b", "/x", rb)
        return select_run(db, base_model="b"), select_run(db, subject="/x")

    def test_worse_verdict_and_rows(self, db, tmp_path):
        ids = [f"q{i}" for i in range(20)]
        run_a, run_b = self._two_runs(db, tmp_path, {q: 0.0 for q in ids}, {q: 1.0 for q in ids})
        rows, unpaired = compare_runs(run_a, run_b, ["audit_prompt_injection"])
        overall = next(r for r in rows if r.attack == "audit_prompt_injection" and r.category == "overall")
        assert overall.n == 20 and overall.a == 0.0 and overall.b == 1.0 and overall.delta == 1.0
        assert overall.verdict == "WORSE"
        assert unpaired == {"audit_prompt_injection": 0}
        assert any(r.category == "direct" for r in rows)

    def test_missing_attack_gives_na_row(self, db, tmp_path):
        run_a, run_b = self._two_runs(db, tmp_path, {"q1": 0.0}, {"q1": 0.0})
        rows, _ = compare_runs(run_a, run_b, ["audit_prompt_injection", "audit_system_prompt"])
        na = [r for r in rows if r.attack == "audit_system_prompt"]
        assert len(na) == 1 and na[0].verdict == "n/a" and na[0].n == 0

    def test_missing_report_file_is_flagged(self, db, tmp_path):
        run_a, run_b = self._two_runs(db, tmp_path, {"q1": 0.0}, {"q1": 0.0})
        Path(run_b.tasks["audit_prompt_injection"].artifact_path).unlink()
        rows, _ = compare_runs(run_a, run_b, ["audit_prompt_injection"])
        assert rows[0].verdict == "n/a" and "missing" in rows[0].note

    def test_calibration_summary_only(self, db, tmp_path):
        def cal(path, brier, ece):
            path.write_text(json.dumps({"metadata": {}, "summary": {"brier": brier, "ece": ece}, "results": []}))
            return path
        ra = cal(tmp_path / "ca.json", 0.20, 0.10)
        rb = cal(tmp_path / "cb.json", 0.25, 0.12)
        _completed_audit(db, "audit_calibration", "b", None, ra, gauntlet="baseline")
        _completed_audit(db, "audit_calibration", "b", "/x", rb)
        rows, _ = compare_runs(select_run(db, base_model="b"), select_run(db, subject="/x"), ["audit_calibration"])
        brier = next(r for r in rows if r.metric == "brier")
        assert brier.delta == pytest.approx(0.05) and brier.lo is None and brier.verdict == "—"

    def test_json_and_table(self, db, tmp_path):
        run_a, run_b = self._two_runs(db, tmp_path, {"q1": 0.0, "q2": 0.0}, {"q1": 1.0, "q2": 1.0})
        rows, unpaired = compare_runs(run_a, run_b, ["audit_prompt_injection"])
        j = to_json(run_a, run_b, rows, unpaired)
        assert j["a"]["base_model"] == "b" and j["a"]["adapter"] is None and j["b"]["adapter"] == "/x"
        assert j["rows"][0]["attack"] == "audit_prompt_injection" and "verdict" in j["rows"][0]
        text = render_table(run_a, run_b, rows, unpaired)
        assert "WORSE" in text and "baseline" in text and "/x" in text

    def test_load_report_missing(self, db):
        tid = db.add_task(type="audit_prompt_injection", label="x", args={})
        db.mark_task(tid, status="completed", artifact_path="/does/not/exist.json")
        assert load_report(db.get_task(tid)) is None


class TestCompareCommand:
    def _ns(self, db, **kw):
        base = dict(db_path=str(db.db_path), a=None, b=None, preset="core", json=False, resamples=200, seed=1)
        base.update(kw)
        import argparse
        return argparse.Namespace(**base)

    def _seed(self, db, tmp_path):
        ids = [f"q{i}" for i in range(10)]
        ra = _pi_report(tmp_path / "a.json", "b", None, {q: 0.0 for q in ids})
        rb = _pi_report(tmp_path / "b.json", "b", "/x", {q: 1.0 for q in ids})
        _completed_audit(db, "audit_prompt_injection", "b", None, ra, gauntlet="baseline")
        _completed_audit(db, "audit_prompt_injection", "b", "/x", rb)

    def test_no_args_latest_vs_baseline(self, db, tmp_path, capsys):
        from attacklm.queue import cli as qcli
        self._seed(db, tmp_path)
        assert qcli._cmd_compare(self._ns(db)) == 0
        out = capsys.readouterr().out
        assert "WORSE" in out and "/x" in out

    def test_json_output(self, db, tmp_path, capsys):
        from attacklm.queue import cli as qcli
        self._seed(db, tmp_path)
        assert qcli._cmd_compare(self._ns(db, json=True)) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["b"]["adapter"] == "/x" and data["rows"][0]["verdict"] == "WORSE"

    def test_adapter_vs_adapter(self, db, tmp_path, capsys):
        from attacklm.queue import cli as qcli
        self._seed(db, tmp_path)
        ry = _pi_report(tmp_path / "y.json", "b", "/y", {f"q{i}": 1.0 for i in range(10)})
        _completed_audit(db, "audit_prompt_injection", "b", "/y", ry)
        assert qcli._cmd_compare(self._ns(db, a="/x", b="/y")) == 0
        assert "SAME" in capsys.readouterr().out

    def test_no_baseline_is_error(self, db, tmp_path, capsys):
        from attacklm.queue import cli as qcli
        rb = _pi_report(tmp_path / "b.json", "other", "/x", {"q1": 1.0})
        _completed_audit(db, "audit_prompt_injection", "other", "/x", rb)
        assert qcli._cmd_compare(self._ns(db)) == 1
        assert "No baseline for other" in capsys.readouterr().err

    def test_nothing_completed_is_error(self, db, capsys):
        from attacklm.queue import cli as qcli
        assert qcli._cmd_compare(self._ns(db)) == 1
        assert "No completed gauntlet" in capsys.readouterr().err
