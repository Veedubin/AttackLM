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

    def test_error_sentinel_item_is_unpaired_not_scored_as_ambiguous(self):
        """I6: asr=0.5 is both a real "ambiguous" grade and the
        generation-error sentinel (audit_prompt_injection.py:240-251
        writes {"asr": 0.5, "score": "error", "error": <msg>} on a caught
        exception) -- an errored item must not be pairable as if it were a
        genuine 0.5 grade."""
        ra = {"results": [{"question_id": "q1", "tier": "direct", "asr": 0.5,
                           "score": "error", "error": "generation timed out"}]}
        rb = {"results": [{"question_id": "q1", "tier": "direct", "asr": 1.0}]}
        cats, unpaired = pair_items(ra, rb, "asr", "question_id", "tier")
        assert cats["overall"] == ([], [])
        assert unpaired == 1

    def test_plain_ambiguous_score_still_pairs(self):
        """A real 0.5 grade (no error marker) is a legitimate value and
        must still be paired normally -- I6 only excludes the sentinel."""
        ra = {"results": [{"question_id": "q1", "tier": "direct", "asr": 0.5}]}
        rb = {"results": [{"question_id": "q1", "tier": "direct", "asr": 1.0}]}
        cats, unpaired = pair_items(ra, rb, "asr", "question_id", "tier")
        assert cats["overall"] == ([0.5], [1.0])
        assert unpaired == 0


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

    def test_subject_with_no_real_run_falls_back_to_its_baseline(self, db, tmp_path):
        """I9: the untuned base model itself never has a non-baseline run
        (it only ever appears tagged gauntlet="baseline") -- select_run
        must still be able to address it by name."""
        base = "Qwen/Qwen2.5-Coder-3B-Instruct"
        r = _pi_report(tmp_path / "1.json", base, None, {"q1": 0.0})
        _completed_audit(db, "audit_prompt_injection", base, None, r, gauntlet="baseline")
        run = select_run(db, subject=base)
        assert run is not None
        assert run.subject is None and run.adapter is None and run.base_model == base

    def test_compare_baseline_and_merged_model_both_orders(self, db, tmp_path):
        """I9: `queue compare <base-with-only-a-baseline> <merged-model>`
        must work in either argument order."""
        base = "Qwen/Qwen2.5-Coder-3B-Instruct"
        merged = "models/merged/attacklm-3b-16g"
        rbase = _pi_report(tmp_path / "base.json", base, None, {f"q{i}": 0.0 for i in range(5)})
        rmerged = _pi_report(tmp_path / "merged.json", merged, None, {f"q{i}": 1.0 for i in range(5)})
        _completed_audit(db, "audit_prompt_injection", base, None, rbase, gauntlet="baseline")
        _completed_audit(db, "audit_prompt_injection", merged, None, rmerged)  # untagged, no adapter

        run_a, run_b = select_run(db, subject=base), select_run(db, subject=merged)
        assert run_a is not None and run_b is not None
        assert run_a.base_model == base and run_a.subject is None
        assert run_b.subject == merged

        run_a2, run_b2 = select_run(db, subject=merged), select_run(db, subject=base)
        assert run_a2.subject == merged
        assert run_b2.base_model == base and run_b2.subject is None


class TestCompareRuns:
    def _two_runs(self, db, tmp_path, a_vals, b_vals):
        ra = _pi_report(tmp_path / "a.json", "b", None, a_vals)
        rb = _pi_report(tmp_path / "b.json", "b", "/x", b_vals)
        _completed_audit(db, "audit_prompt_injection", "b", None, ra, gauntlet="baseline")
        _completed_audit(db, "audit_prompt_injection", "b", "/x", rb)
        return select_run(db, base_model="b"), select_run(db, subject="/x")

    def test_worse_verdict_and_rows(self, db, tmp_path):
        # C2: a flat 0.0-vs-flat-1.0 population gives a *degenerate*
        # (lo==hi) CI, which stats.verdict now reports as SAME, not a
        # confident WORSE -- give the adapter side real per-item variance
        # (0.9/1.0 alternating, always > the constant 0.0 baseline) so the
        # CI is non-degenerate and WORSE is actually earned.
        ids = [f"q{i}" for i in range(20)]
        a_vals = {q: 0.0 for q in ids}
        b_vals = {q: (0.9 if i % 2 == 0 else 1.0) for i, q in enumerate(ids)}
        run_a, run_b = self._two_runs(db, tmp_path, a_vals, b_vals)
        rows, unpaired = compare_runs(run_a, run_b, ["audit_prompt_injection"])
        overall = next(r for r in rows if r.attack == "audit_prompt_injection" and r.category == "overall")
        assert overall.n == 20 and overall.a == 0.0
        assert overall.b == pytest.approx(0.95) and overall.delta == pytest.approx(0.95)
        assert overall.verdict == "WORSE"
        assert unpaired == {"audit_prompt_injection": 0}
        assert any(r.category == "direct" for r in rows)

    def test_empty_category_is_na_not_same(self, db, tmp_path):
        """C1: two reports that share NO question ids must not read as a
        confident SAME with Δ=0.000/CI[0,0] -- there's nothing paired at
        all, which is a different, non-comparable situation."""
        ra = _pi_report(tmp_path / "a.json", "b", None, {"q1": 0.0, "q2": 0.0},
                        tiers={"q1": "direct", "q2": "indirect"})
        rb = _pi_report(tmp_path / "b.json", "b", "/x", {"q8": 1.0, "q9": 1.0},
                        tiers={"q8": "direct", "q9": "indirect"})
        _completed_audit(db, "audit_prompt_injection", "b", None, ra, gauntlet="baseline")
        _completed_audit(db, "audit_prompt_injection", "b", "/x", rb)
        run_a, run_b = select_run(db, base_model="b"), select_run(db, subject="/x")
        rows, unpaired = compare_runs(run_a, run_b, ["audit_prompt_injection"])
        overall = next(r for r in rows if r.category == "overall")
        assert overall.n == 0
        assert overall.a is None and overall.b is None
        assert overall.delta is None and overall.lo is None and overall.hi is None
        assert overall.verdict == "n/a" and overall.note == "no paired items"
        assert unpaired["audit_prompt_injection"] == 4

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

    def test_calibration_reads_real_nested_results_schema(self, db, tmp_path):
        """I1: eval_calibration.py (scripts/eval_calibration.py:311-360)
        writes {"metadata": {...}, "results": {"in_distribution": {...},
        "near_ood": {...}|None, "ood": {...}|None}} -- there is NO
        top-level "summary" key. This fixture is built from that real
        writer shape, not hand-copied. near_ood/ood are optional
        (None when their CLI flags weren't passed)."""
        def cal(path, in_dist, near_ood=None, ood=None):
            path.write_text(json.dumps({
                "metadata": {"base_model": "b", "adapter": None,
                            "attack_class": "calibration-audit-v1"},
                "results": {"in_distribution": in_dist, "near_ood": near_ood, "ood": ood},
            }))
            return path

        def set_(brier, ece, n=20):
            return {"n_records": n, "mean_nll": 1.0, "median_nll": 1.0, "p90_nll": 1.0,
                    "brier": brier, "ece": ece, "selective": []}

        ra = cal(tmp_path / "ca.json", set_(0.20, 0.10), near_ood=set_(0.30, 0.15))
        rb = cal(tmp_path / "cb.json", set_(0.25, 0.12), near_ood=None)  # near_ood absent on B
        _completed_audit(db, "audit_calibration", "b", None, ra, gauntlet="baseline")
        _completed_audit(db, "audit_calibration", "b", "/x", rb)
        rows, unpaired = compare_runs(select_run(db, base_model="b"), select_run(db, subject="/x"),
                                      ["audit_calibration"])
        brier = next(r for r in rows if r.metric == "brier" and r.category == "in_distribution")
        assert brier.delta == pytest.approx(0.05) and brier.lo is None and brier.verdict == "—"
        ece = next(r for r in rows if r.metric == "ece" and r.category == "in_distribution")
        assert ece.delta == pytest.approx(0.02) and ece.verdict == "—"
        # near_ood is present on A but absent (None) on B -> no row for it,
        # not an error row -- "not run on both sides" isn't "no value".
        assert not any(r.category == "near_ood" for r in rows)
        assert not any(r.category == "ood" for r in rows)
        assert unpaired == {"audit_calibration": 0}

    def test_calibration_missing_metric_value_is_na(self, db, tmp_path):
        """A present set with a null brier/ece (e.g. evaluate_set found 0
        valid records) is a real "no value" case, distinct from the set
        being entirely absent."""
        def cal(path, brier, ece):
            path.write_text(json.dumps({
                "metadata": {}, "results": {"in_distribution": {"brier": brier, "ece": ece}}}))
            return path
        ra = cal(tmp_path / "ca.json", None, 0.10)
        rb = cal(tmp_path / "cb.json", 0.25, 0.12)
        _completed_audit(db, "audit_calibration", "b", None, ra, gauntlet="baseline")
        _completed_audit(db, "audit_calibration", "b", "/x", rb)
        rows, _ = compare_runs(select_run(db, base_model="b"), select_run(db, subject="/x"), ["audit_calibration"])
        brier = next(r for r in rows if r.metric == "brier")
        assert brier.verdict == "n/a" and brier.note == "no value"

    def test_canary_unpaired_uses_max_across_metrics(self, db, tmp_path):
        """Minor (final review): a canary report has two item metrics
        (exact_token, near_verbatim) over the SAME id join -- the second
        metric's unpaired count must not silently overwrite (and hide) the
        first metric's higher unpaired count."""
        def canary_report(path, base, adapter, items):
            path.write_text(json.dumps({
                "metadata": {"base_model": base, "adapter": adapter},
                "summary": {}, "results": items,
            }))
            return path
        ra = canary_report(tmp_path / "a.json", "b", None, [
            {"canary_id": "c1", "prefix": "p", "exact_token": False, "near_verbatim": False},
            {"canary_id": "c2", "prefix": "p", "exact_token": False, "near_verbatim": False},
        ])
        rb = canary_report(tmp_path / "b.json", "b", "/x", [
            {"canary_id": "c1", "prefix": "p", "near_verbatim": True},  # exact_token missing on B
            {"canary_id": "c2", "prefix": "p", "exact_token": True, "near_verbatim": True},
        ])
        _completed_audit(db, "audit_canary_pipeline", "b", None, ra, gauntlet="baseline")
        _completed_audit(db, "audit_canary_pipeline", "b", "/x", rb)
        rows, unpaired = compare_runs(select_run(db, base_model="b"), select_run(db, subject="/x"),
                                      ["audit_canary_pipeline"])
        # exact_token has 1 unpaired item (c1); near_verbatim has 0. The
        # overall count for the attack must reflect the worse (max) of the
        # two, not whichever metric happened to run last.
        assert unpaired["audit_canary_pipeline"] == 1

    def test_json_and_table(self, db, tmp_path):
        # n must be >= MIN_PAIRED_N with real variance to earn a WORSE
        # verdict post-C2 (see test_worse_verdict_and_rows).
        ids = [f"q{i}" for i in range(6)]
        a_vals = {q: 0.0 for q in ids}
        b_vals = {q: (0.9 if i % 2 == 0 else 1.0) for i, q in enumerate(ids)}
        run_a, run_b = self._two_runs(db, tmp_path, a_vals, b_vals)
        rows, unpaired = compare_runs(run_a, run_b, ["audit_prompt_injection"])
        j = to_json(run_a, run_b, rows, unpaired)
        assert j["a"]["base_model"] == "b" and j["a"]["adapter"] is None and j["b"]["adapter"] == "/x"
        assert j["rows"][0]["attack"] == "audit_prompt_injection" and "verdict" in j["rows"][0]
        text = render_table(run_a, run_b, rows, unpaired)
        assert "WORSE" in text and "baseline" in text and "/x" in text

    def test_table_header_includes_task_ids(self, db, tmp_path):
        run_a, run_b = self._two_runs(db, tmp_path, {"q1": 0.0}, {"q1": 1.0})
        rows, unpaired = compare_runs(run_a, run_b, ["audit_prompt_injection"])
        text = render_table(run_a, run_b, rows, unpaired)
        id_a = next(iter(run_a.tasks.values())).id
        id_b = next(iter(run_b.tasks.values())).id
        assert f"#{id_a}" in text and f"#{id_b}" in text

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
        # C2: a flat 0.0-vs-flat-1.0 population is a degenerate (lo==hi)
        # CI -- stats.verdict now reports that as SAME, not a confident
        # WORSE. Give the adapter side real per-item variance.
        ids = [f"q{i}" for i in range(10)]
        ra = _pi_report(tmp_path / "a.json", "b", None, {q: 0.0 for q in ids})
        rb = _pi_report(tmp_path / "b.json", "b", "/x",
                        {q: (0.9 if i % 2 == 0 else 1.0) for i, q in enumerate(ids)})
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
        # Identical distribution to /x's -> every paired diff is exactly
        # 0.0 (a genuinely zero-width CI, not the C2 bug): correctly SAME.
        ry = _pi_report(tmp_path / "y.json", "b", "/y",
                        {f"q{i}": (0.9 if i % 2 == 0 else 1.0) for i in range(10)})
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

    def test_baseline_queued_but_not_finished_gives_helpful_message(self, db, tmp_path, capsys):
        """I4: 'No baseline for X. Run: attacklm queue baseline X' is
        useless advice when a baseline is already queued -- tell the user
        it exists and just needs the runner started."""
        from attacklm.queue import cli as qcli
        rb = _pi_report(tmp_path / "b.json", "other", "/x", {"q1": 1.0})
        _completed_audit(db, "audit_prompt_injection", "other", "/x", rb)
        rb2 = _pi_report(tmp_path / "b2.json", "other", "/x", {"q1": 1.0})
        _completed_audit(db, "audit_system_prompt", "other", "/x", rb2)
        t1 = db.add_task(type="audit_prompt_injection", label="baseline pi",
                         args={"base_model": "other"}, gauntlet="baseline")
        t2 = db.add_task(type="audit_system_prompt", label="baseline sp",
                         args={"base_model": "other"}, gauntlet="baseline")
        assert qcli._cmd_compare(self._ns(db, preset="quick")) == 1
        err = capsys.readouterr().err
        assert "queued but not finished" in err
        assert f"#{t1}" in err and f"#{t2}" in err
        assert "attacklm queue start" in err

    def test_preset_memorization_no_arg_does_not_crash(self, db, tmp_path, capsys):
        """I2: latest_subject(db) (unfiltered) could find a subject whose
        only completed work is outside the --preset's own attacks, so
        select_run(subject=..., attacks=<preset's>) returns None and the
        old code crashed on run_b.base_model. Passing the same
        preset-filtered attacks into latest_subject keeps them agreeing."""
        from attacklm.queue import cli as qcli
        rb = _pi_report(tmp_path / "b.json", "other", "/x", {"q1": 1.0})
        _completed_audit(db, "audit_prompt_injection", "other", "/x", rb)  # not in "memorization"
        assert qcli._cmd_compare(self._ns(db, preset="memorization")) == 1
        assert "No completed gauntlet" in capsys.readouterr().err

    def test_unknown_preset_is_an_error(self, db, tmp_path, capsys):
        from attacklm.queue import cli as qcli
        assert qcli._cmd_compare(self._ns(db, preset="nope")) == 1
        assert "Unknown gauntlet preset" in capsys.readouterr().err
