#!/usr/bin/env python3
"""Tests for Inspect adapter argv construction.

Every flag asserted here was verified against `inspect_ai eval --help`
(inspect_ai 0.3.266), not taken from documentation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from attacklm.bench.adapters.inspect_adapter import InspectAdapter, RunConfig
from attacklm.bench.packs import Pack, PackSource

PACK = Pack(
    name="cybermetric-500",
    harness_task="inspect_evals/cybermetric_500",
    mode="harness_scored",
    source=PackSource(kind="github_raw", repo="r", revision="abc"),
    license="unknown",
    redistributable=False,
    metric="accuracy",
    chance_level=0.25,
    reference_scores={},
    ladder=[100, None],
    categories_from="field:category",
)


def _argv(**kw):
    cfg = RunConfig(model="models/merged/attacklm-3b-16g", log_dir=Path("/tmp/l"), **kw)
    return InspectAdapter().build_argv(PACK, cfg)


def test_invokes_inspect_eval_with_task():
    argv = _argv()
    assert argv[1:4] == ["-m", "inspect_ai", "eval"]
    assert "inspect_evals/cybermetric_500" in argv


def test_vllm_backend_prefixes_model():
    argv = _argv(backend="vllm")
    assert argv[argv.index("--model") + 1] == "vllm/models/merged/attacklm-3b-16g"


def test_hf_backend_prefixes_model():
    argv = _argv(backend="hf")
    assert argv[argv.index("--model") + 1] == "hugging_face/models/merged/attacklm-3b-16g"


def test_openai_api_backend_passes_base_url():
    argv = _argv(backend="openai-api", base_url="http://localhost:8000/v1")
    assert argv[argv.index("--model") + 1].startswith("openai-api/")
    assert argv[argv.index("--model-base-url") + 1] == "http://localhost:8000/v1"


def test_generation_knobs_are_explicit():
    """Every value the QCRI audit found score-critical must appear in argv."""
    argv = _argv(max_tokens=256, temperature=0.0, top_p=0.9, seed=7)
    assert argv[argv.index("--max-tokens") + 1] == "256"
    assert argv[argv.index("--temperature") + 1] == "0.0"
    assert argv[argv.index("--top-p") + 1] == "0.9"
    assert argv[argv.index("--seed") + 1] == "7"


def test_no_stop_sequence_by_default():
    """A newline stop sequence truncated RedSage-Bench before the answer."""
    assert "--stop-seqs" not in _argv()


def test_stop_sequences_passed_when_given():
    argv = _argv(stop=["\n\n", "END"])
    assert argv[argv.index("--stop-seqs") + 1] == "\n\n,END"


def test_json_log_format_is_forced():
    """parse_log reads JSON; the default .eval container is not parsed."""
    assert _argv()[_argv().index("--log-format") + 1] == "json"


def test_log_dir_is_passed():
    argv = _argv()
    assert argv[argv.index("--log-dir") + 1] == "/tmp/l"


def test_rung_becomes_limit():
    argv = _argv(limit=200)
    assert argv[argv.index("--limit") + 1] == "200"


def test_no_limit_flag_when_rung_is_none():
    assert "--limit" not in _argv()


def test_max_connections_passed_when_given():
    argv = _argv(max_connections=16)
    assert argv[argv.index("--max-connections") + 1] == "16"


def test_adapter_is_passed_as_model_arg():
    argv = _argv(adapter="models/adapters/run7")
    assert "-M" in argv
    assert any("run7" in a for a in argv)


def test_extra_model_args_are_passed_through():
    """The escape hatch: any native model arg the caller needs."""
    argv = _argv(model_args=["gpu_memory_utilization=0.85", "max_model_len=4096"])
    assert argv.count("-M") == 2
    assert "gpu_memory_utilization=0.85" in argv
    assert "max_model_len=4096" in argv


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="nonsense"):
        _argv(backend="nonsense")


def test_mock_backend_needs_no_gpu():
    """Inspect's stub provider, so the whole pipeline is smoke-testable anywhere."""
    argv = _argv(backend="mock")
    assert argv[argv.index("--model") + 1] == "mockllm/models/merged/attacklm-3b-16g"
