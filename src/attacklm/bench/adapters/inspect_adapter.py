"""Inspect AI adapter — builds argv and reads Inspect's JSON eval logs.

Inspect runs as a SUBPROCESS, never an import. vLLM and Unsloth carry
conflicting torch/CUDA pins (``unsloth_compiled_cache/`` shows Unsloth is live
in this tree), so the harness lives in its own environment, named by
``ATTACKLM_BENCH_PYTHON``.

Everything asserted about Inspect's CLI and log schema here was verified
against a real ``inspect_ai`` 0.3.266 run, not against documentation. See
``tests/fixtures/inspect_log_sample.json``, which is a genuine log produced by
``inspect_ai eval --model mockllm/model --log-format json``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from attacklm.bench.adapters import HarnessInfo
from attacklm.bench.packs import Pack

# backend -> Inspect model-provider prefix
_BACKENDS = {
    "vllm": "vllm",
    "hf": "hugging_face",
    "openai-api": "openai-api",
}

# Inspect's scorer value constants (inspect_ai.scorer), verified:
#   CORRECT="C" -> 1.0, INCORRECT="I" -> 0.0, PARTIAL="P" -> 0.5, NOANSWER="N" -> 0.0
# We diverge on NOANSWER: Inspect floats it to 0.0, but for our purposes a
# non-answer is *invalid*, so our own invalid_policy decides its fate rather
# than it silently becoming a genuine zero.
_VALUE_TO_FLOAT = {"C": 1.0, "I": 0.0, "P": 0.5}
_NOANSWER = "N"
# Inspect sets this on a sample whose output could not be parsed as a choice.
_INVALID_REASON = "invalid_response_format"


@dataclass(frozen=True)
class RunConfig:
    model: str
    adapter: str | None = None
    backend: str = "vllm"
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    stop: list[str] = field(default_factory=list)
    base_url: str | None = None
    num_shots: int = 0
    max_connections: int | None = None
    log_dir: Path | None = None
    limit: int | None = None
    model_args: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ParsedSample:
    """One graded sample as the harness reported it.

    ``score`` is None when the harness could not parse an answer; the caller's
    invalid_policy then decides, exactly as for a locally scored item.
    """

    sample_id: str
    completion: str
    score: float | None
    answer: str | None
    category: str | None
    valid: bool
    metadata: dict


def resolve_bench_python() -> str:
    """Interpreter that has inspect_ai installed.

    Mirrors the existing ``ATTACKLM_SCRIPTS_DIR`` escape hatch.
    """
    return os.environ.get("ATTACKLM_BENCH_PYTHON") or sys.executable


def _completion_of(sample: dict) -> str:
    """Pull the assistant text out of an Inspect sample.

    ``output.completion`` is present and is a plain string in the captured
    fixture; the choices/messages paths are fallbacks for other output shapes.
    """
    output = sample.get("output") or {}

    if isinstance(output.get("completion"), str):
        return output["completion"]

    for choice in output.get("choices") or []:
        content = (choice.get("message") or {}).get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(p.get("text", "") for p in content if isinstance(p, dict))

    for message in reversed(sample.get("messages") or []):
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            return message["content"]

    return ""


def _score_of(sample: dict) -> tuple[float | None, str | None, bool]:
    """Map Inspect's score record to (score, answer, valid).

    ``scores`` is keyed by scorer name. With a single scorer we take it; with
    several we prefer the first, since a pack declares one headline metric.
    """
    scores = sample.get("scores") or {}
    if not scores:
        return None, None, False

    record = next(iter(scores.values()))
    if not isinstance(record, dict):
        return None, None, False

    answer = record.get("answer") or None

    if record.get("reason") == _INVALID_REASON:
        return None, answer, False

    value = record.get("value")
    if value == _NOANSWER:
        return None, answer, False
    if isinstance(value, str):
        mapped = _VALUE_TO_FLOAT.get(value)
        return (mapped, answer, True) if mapped is not None else (None, answer, False)
    if isinstance(value, bool):
        return (1.0 if value else 0.0), answer, True
    if isinstance(value, (int, float)):
        return float(value), answer, True

    return None, answer, False


class InspectAdapter:
    name = "inspect_ai"

    def build_argv(self, pack: Pack, cfg: RunConfig) -> list[str]:
        if cfg.backend not in _BACKENDS:
            raise ValueError(
                f"unknown backend {cfg.backend!r}; expected one of {tuple(_BACKENDS)}"
            )
        prefix = _BACKENDS[cfg.backend]

        argv: list[str] = [
            resolve_bench_python(), "-m", "inspect_ai", "eval",
            pack.harness_task,
            "--model", f"{prefix}/{cfg.model}",
            # Each of these is stated explicitly rather than inherited. The
            # QCRI pipeline audit found unstated defaults swinging scores by
            # up to 85.9 points, and 61% of config unreproducible from docs.
            "--max-tokens", str(cfg.max_tokens),
            "--temperature", str(cfg.temperature),
            "--top-p", str(cfg.top_p),
            "--seed", str(cfg.seed),
            # parse_log reads JSON; the default .eval container is a zip.
            "--log-format", "json",
        ]

        if cfg.log_dir is not None:
            argv += ["--log-dir", str(cfg.log_dir)]
        if cfg.base_url:
            argv += ["--model-base-url", cfg.base_url]
        if cfg.stop:
            argv += ["--stop-seqs", ",".join(cfg.stop)]
        if cfg.max_connections is not None:
            argv += ["--max-connections", str(cfg.max_connections)]
        if cfg.limit is not None:
            argv += ["--limit", str(cfg.limit)]

        # NOTE: the LoRA passthrough below is the one thing here not yet
        # confirmed against a live vLLM run (that needs a GPU). It is exposed
        # as an ordinary model arg so `--model-arg` can override it without a
        # code change if the provider expects a different spelling.
        if cfg.adapter:
            argv += ["-M", f"lora_modules={cfg.adapter}"]
        for arg in cfg.model_args:
            argv += ["-M", arg]

        return argv

    def parse_log(self, log_dir: Path) -> list[ParsedSample]:
        logs = sorted(
            Path(log_dir).glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if not logs:
            raise FileNotFoundError(f"no Inspect JSON log found in {log_dir}")

        data = json.loads(logs[0].read_text())
        parsed: list[ParsedSample] = []

        for idx, sample in enumerate(data.get("samples") or []):
            score, answer, valid = _score_of(sample)
            metadata = sample.get("metadata") or {}
            parsed.append(
                ParsedSample(
                    sample_id=str(sample.get("id", idx)),
                    completion=_completion_of(sample),
                    score=score,
                    answer=answer,
                    category=metadata.get("category"),
                    valid=valid,
                    metadata=metadata,
                )
            )
        return parsed

    def probe(self) -> HarnessInfo:
        try:
            out = subprocess.run(
                [resolve_bench_python(), "-m", "inspect_ai", "--version"],
                capture_output=True, text=True, timeout=60, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return HarnessInfo(self.name, None, False)
        if out.returncode != 0:
            return HarnessInfo(self.name, None, False)
        return HarnessInfo(self.name, out.stdout.strip() or None, True)
