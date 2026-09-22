"""Tests for attacklm.queue.history."""

from __future__ import annotations

import json

import pytest

from attacklm.queue.db import QueueDB
from attacklm.queue.history import append_jsonl, history_rows, render_history


def _done(db, typ, args, report_path, summary, gauntlet=None):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"metadata": {}, "summary": summary, "results": []}))
    tid = db.add_task(type=typ, label=typ, args=args, gauntlet=gauntlet)
    db.mark_task(tid, status="completed", artifact_path=str(report_path), artifact_kind="report")
    return tid


@pytest.fixture
def db(tmp_path):
    return QueueDB(tmp_path / "q.db")


def test_rows_newest_first_with_headline_metric(db, tmp_path):
    t1 = _done(db, "audit_prompt_injection", {"base_model": "b"}, tmp_path / "1.json", {"overall_asr": 0.25}, gauntlet="baseline")
    t2 = _done(db, "audit_system_prompt", {"base_model": "b", "adapter": "/a"}, tmp_path / "2.json", {"overall_leakage": 0.5})
    rows = history_rows(db)
    assert [r["task_id"] for r in rows] == [t2, t1]
    assert rows[0]["subject"] == "/a" and rows[0]["metric"] == "overall_leakage" and rows[0]["value"] == 0.5
    assert rows[1]["subject"] == "(baseline)" and rows[1]["value"] == 0.25


def test_subject_filter(db, tmp_path):
    _done(db, "audit_prompt_injection", {"base_model": "b", "adapter": "/a"}, tmp_path / "1.json", {"overall_asr": 0.1})
    _done(db, "audit_prompt_injection", {"base_model": "b", "adapter": "/z"}, tmp_path / "2.json", {"overall_asr": 0.2})
    assert [r["subject"] for r in history_rows(db, subject="/z")] == ["/z"]


def test_merged_model_subject_is_not_mislabelled_baseline(db, tmp_path):
    """I5: a merged model has no adapter, so its subject IS its base_model
    -- an untagged (not gauntlet="baseline") run for it must show that
    base_model as the subject, not "(baseline)", and must be findable by
    filtering on it."""
    _done(db, "audit_prompt_injection", {"base_model": "models/merged/x"},
          tmp_path / "1.json", {"overall_asr": 0.3})  # no adapter, untagged
    rows = history_rows(db)
    assert rows[0]["subject"] == "models/merged/x"
    assert [r["subject"] for r in history_rows(db, subject="models/merged/x")] == ["models/merged/x"]


def test_calibration_headline_reads_real_nested_schema(db, tmp_path):
    """I1: audit_calibration has no flat "summary" dict -- the real report
    (scripts/eval_calibration.py) nests ece under
    results.in_distribution.ece."""
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({
        "metadata": {},
        "results": {"in_distribution": {"brier": 0.2, "ece": 0.08}, "near_ood": None, "ood": None},
    }))
    tid = db.add_task(type="audit_calibration", label="cal", args={"base_model": "b"})
    db.mark_task(tid, status="completed", artifact_path=str(path), artifact_kind="report")
    rows = history_rows(db)
    assert rows[0]["metric"] == "ece" and rows[0]["value"] == pytest.approx(0.08)


def test_missing_report_value_is_none(db, tmp_path):
    tid = db.add_task(type="audit_prompt_injection", label="x", args={"base_model": "b"})
    db.mark_task(tid, status="completed", artifact_path=str(tmp_path / "nope.json"))
    assert history_rows(db)[0]["value"] is None


def test_append_jsonl_dedups(db, tmp_path):
    _done(db, "audit_prompt_injection", {"base_model": "b"}, tmp_path / "1.json", {"overall_asr": 0.1})
    out = tmp_path / "h.jsonl"
    assert append_jsonl(history_rows(db), out) == 1
    assert append_jsonl(history_rows(db), out) == 0
    _done(db, "audit_system_prompt", {"base_model": "b"}, tmp_path / "2.json", {"overall_leakage": 0.1})
    assert append_jsonl(history_rows(db), out) == 1
    assert len(out.read_text().splitlines()) == 2


def test_render(db, tmp_path):
    _done(db, "audit_prompt_injection", {"base_model": "b", "adapter": "/a"}, tmp_path / "1.json", {"overall_asr": 0.1})
    text = render_history(history_rows(db))
    assert "/a" in text and "overall_asr" in text


def test_append_jsonl_skips_malformed_lines(db, tmp_path):
    """Minor (final review): a malformed/partial line in an existing
    history file must not crash the append."""
    out = tmp_path / "h.jsonl"
    out.write_text("not json\n" + json.dumps({"report": "x"}) + "\n")  # missing "task_id"
    _done(db, "audit_prompt_injection", {"base_model": "b"}, tmp_path / "1.json", {"overall_asr": 0.1})
    n = append_jsonl(history_rows(db), out)
    assert n == 1
    assert len(out.read_text().splitlines()) == 3
