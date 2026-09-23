#!/usr/bin/env python3
"""Tests for Inspect JSON log parsing.

tests/fixtures/inspect_log_sample.json is a GENUINE log written by inspect_ai
0.3.266, not a hand-authored schema. It was produced by
tests/fixtures/inspect_gen_fixture.py driving mockllm with custom_outputs so
the three sample outcomes are real:

    fx_000  value="C"                                    -> correct
    fx_001  value="I"                                    -> incorrect
    fx_002  value="I", reason="invalid_response_format"  -> UNPARSEABLE

That third row is the one that matters. Inspect scores an unparseable response
as "I" (0.0, incorrect). Mapping value->float naively would therefore record a
refusal as a genuine wrong answer, collapsing the refused-vs-wrong distinction
this project exists to measure.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from attacklm.bench.adapters.inspect_adapter import InspectAdapter, ParsedSample

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "inspect_log_sample.json"


@pytest.fixture
def log_dir(tmp_path):
    (tmp_path / "log.json").write_text(FIXTURE.read_text())
    return tmp_path


def test_fixture_exists():
    assert FIXTURE.is_file(), "regenerate with tests/fixtures/inspect_gen_fixture.py"


def test_fixture_is_a_real_inspect_log():
    """Guard against someone replacing it with a hand-written stub."""
    data = json.loads(FIXTURE.read_text())
    assert "eval" in data and "plan" in data and "stats" in data
    assert data["eval"]["model"] == "mockllm/model"


def test_parses_every_sample(log_dir):
    parsed = InspectAdapter().parse_log(log_dir)
    assert len(parsed) == 3
    assert all(isinstance(p, ParsedSample) for p in parsed)


def test_sample_ids_are_strings_and_unique(log_dir):
    ids = [p.sample_id for p in InspectAdapter().parse_log(log_dir)]
    assert ids == ["fx_000", "fx_001", "fx_002"]
    assert len(set(ids)) == len(ids)


def test_correct_sample_scores_one(log_dir):
    p = InspectAdapter().parse_log(log_dir)[0]
    assert p.score == 1.0 and p.valid is True and p.answer == "B"


def test_incorrect_sample_scores_zero(log_dir):
    p = InspectAdapter().parse_log(log_dir)[1]
    assert p.score == 0.0 and p.valid is True and p.answer == "C"


def test_unparseable_sample_is_invalid_not_a_genuine_zero(log_dir):
    """The crucial case: Inspect records value='I' for an unparseable answer.

    Treating that as a real 0.0 would score a refusal as a wrong answer.
    """
    p = InspectAdapter().parse_log(log_dir)[2]
    assert p.valid is False
    assert p.score is None


def test_completions_are_strings(log_dir):
    parsed = InspectAdapter().parse_log(log_dir)
    assert all(isinstance(p.completion, str) for p in parsed)
    assert "ANSWER: B" in parsed[0].completion


def test_category_comes_from_sample_metadata(log_dir):
    cats = [p.category for p in InspectAdapter().parse_log(log_dir)]
    assert cats == ["persistence", "detection", "execution"]


def test_empty_log_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        InspectAdapter().parse_log(tmp_path)


def test_newest_log_wins(tmp_path):
    (tmp_path / "a.json").write_text(FIXTURE.read_text())
    truncated = json.loads(FIXTURE.read_text())
    truncated["samples"] = truncated["samples"][:1]
    (tmp_path / "b.json").write_text(json.dumps(truncated))
    time.sleep(0.01)
    os.utime(tmp_path / "b.json", None)
    assert len(InspectAdapter().parse_log(tmp_path)) == 1


def test_missing_scores_block_is_invalid(tmp_path):
    data = json.loads(FIXTURE.read_text())
    for sample in data["samples"]:
        sample.pop("scores", None)
    (tmp_path / "log.json").write_text(json.dumps(data))
    parsed = InspectAdapter().parse_log(tmp_path)
    assert all(p.score is None and p.valid is False for p in parsed)


def test_numeric_score_value_is_passed_through(tmp_path):
    """Scorers other than choice() emit floats rather than C/I."""
    data = json.loads(FIXTURE.read_text())
    data["samples"][0]["scores"]["choice"]["value"] = 0.75
    data["samples"][0]["scores"]["choice"].pop("reason", None)
    (tmp_path / "log.json").write_text(json.dumps(data))
    p = InspectAdapter().parse_log(tmp_path)[0]
    assert p.score == 0.75 and p.valid is True


def test_partial_credit_value_maps_to_half(tmp_path):
    data = json.loads(FIXTURE.read_text())
    data["samples"][0]["scores"]["choice"]["value"] = "P"
    (tmp_path / "log.json").write_text(json.dumps(data))
    assert InspectAdapter().parse_log(tmp_path)[0].score == 0.5


def test_noanswer_value_is_invalid(tmp_path):
    """Inspect floats NOANSWER to 0.0; we treat it as invalid instead."""
    data = json.loads(FIXTURE.read_text())
    data["samples"][0]["scores"]["choice"]["value"] = "N"
    (tmp_path / "log.json").write_text(json.dumps(data))
    p = InspectAdapter().parse_log(tmp_path)[0]
    assert p.score is None and p.valid is False


# --------------------------------------------------------------------------
# a failed harness run must not masquerade as data
# --------------------------------------------------------------------------


def test_failed_run_raises_rather_than_reporting_all_invalid(tmp_path):
    """A dead harness produced a report of invalid=N, which reads exactly like
    a model that refused every question. That is a failure flattering itself
    as data, so parse_log refuses instead."""
    from attacklm.bench.adapters.inspect_adapter import HarnessRunError

    data = json.loads(FIXTURE.read_text())
    data["status"] = "error"
    data["samples"] = []
    data["error"] = {"message": "Failed to start vLLM server"}
    (tmp_path / "log.json").write_text(json.dumps(data))

    with pytest.raises(HarnessRunError, match="vLLM"):
        InspectAdapter().parse_log(tmp_path)


def test_successful_run_with_samples_is_unaffected(log_dir):
    assert len(InspectAdapter().parse_log(log_dir)) == 3


def test_partial_run_keeps_the_samples_it_got(tmp_path):
    """An interrupted run that DID produce samples is still usable data."""
    data = json.loads(FIXTURE.read_text())
    data["status"] = "error"
    data["samples"] = data["samples"][:1]
    (tmp_path / "log.json").write_text(json.dumps(data))
    assert len(InspectAdapter().parse_log(tmp_path)) == 1


def test_bench_env_puts_the_harness_venv_bin_on_path(monkeypatch, tmp_path):
    """Inspect's vLLM provider starts a `vllm` console script, not a module."""
    from attacklm.bench.adapters.inspect_adapter import bench_env

    fake = tmp_path / "venv" / "bin" / "python"
    fake.parent.mkdir(parents=True)
    fake.touch()
    monkeypatch.setenv("ATTACKLM_BENCH_PYTHON", str(fake))
    assert bench_env()["PATH"].split(os.pathsep)[0] == str(fake.parent)
