"""Benchmark pack manifests — the single source of truth for one benchmark.

A pack declares where its data comes from, who may redistribute it, how it is
scored, and how far the escalation ladder may climb.

Two scoring modes exist, and the distinction is load-bearing:

``harness_scored``
    The harness already implements this benchmark end to end — it fetches the
    data and applies its own scorer. ``inspect_evals/cybermetric_500`` pins the
    upstream GitHub revision itself and scores with ``choice()``. We normalise
    *its* per-sample scores. Re-deriving them here would throw away the
    upstream's validated prompt and answer-extraction, which is precisely the
    layer the QCRI pipeline audit found dominating reported scores.

``local_scored``
    We own the items and the scoring (packs the harness lacks — CTI-Bench,
    SecEval, SecBench — and our own authored sets). The harness is used only to
    generate completions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

PACKS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "bench" / "packs"

_REQUIRED = (
    "name",
    "harness_task",
    "mode",
    "source",
    "license",
    "redistributable",
    "metric",
    "chance_level",
    "ladder",
    "categories_from",
)
_VALID_METRICS = ("accuracy", "micro_f1")
_VALID_MODES = ("harness_scored", "local_scored")
_VALID_KINDS = ("huggingface", "github_raw", "local")
# Sources that must name an immutable revision, or the data can change under a
# pinned checksum without anyone noticing.
_PINNED_KINDS = ("huggingface", "github_raw")
_UNPINNED_REFS = ("main", "master", "head", "latest", "")


@dataclass(frozen=True)
class PackSource:
    kind: str
    repo: str | None = None
    config: str | None = None
    revision: str | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class Pack:
    name: str
    harness_task: str
    mode: str
    source: PackSource
    license: str
    redistributable: bool
    metric: str
    chance_level: float
    reference_scores: dict[str, float]
    ladder: list[int | None]
    categories_from: str


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - pyyaml is a bench extra
        raise RuntimeError("pyyaml is required to read pack manifests") from exc
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: manifest must be a mapping")
    return data


def load_pack(path: Path) -> Pack:
    """Parse and validate one pack manifest.

    Raises ValueError with the offending field named, rather than producing a
    Pack that fails later inside a GPU run.
    """
    data = _load_yaml(path)

    missing = [k for k in _REQUIRED if k not in data]
    if missing:
        raise ValueError(f"{path}: missing required field(s): {', '.join(missing)}")

    if data["mode"] not in _VALID_MODES:
        raise ValueError(f"{path}: mode must be one of {_VALID_MODES}, got {data['mode']!r}")

    if data["metric"] not in _VALID_METRICS:
        raise ValueError(f"{path}: metric must be one of {_VALID_METRICS}, got {data['metric']!r}")

    raw_source = data["source"]
    if not isinstance(raw_source, dict) or "kind" not in raw_source:
        raise ValueError(f"{path}: source must be a mapping with a 'kind'")
    if raw_source["kind"] not in _VALID_KINDS:
        raise ValueError(
            f"{path}: source.kind must be one of {_VALID_KINDS}, got {raw_source['kind']!r}"
        )

    source = PackSource(
        kind=raw_source["kind"],
        repo=raw_source.get("repo"),
        config=raw_source.get("config"),
        revision=raw_source.get("revision"),
        sha256=raw_source.get("sha256"),
    )

    if source.kind in _PINNED_KINDS:
        if (source.revision or "").strip().lower() in _UNPINNED_REFS:
            raise ValueError(
                f"{path}: source.revision must be pinned to an immutable commit sha "
                f"for kind={source.kind!r}, got {source.revision!r}"
            )

    ladder = list(data["ladder"])
    if not ladder:
        raise ValueError(f"{path}: ladder must not be empty")

    return Pack(
        name=data["name"],
        harness_task=data["harness_task"],
        mode=data["mode"],
        source=source,
        license=data["license"],
        redistributable=bool(data["redistributable"]),
        metric=data["metric"],
        chance_level=float(data["chance_level"]),
        reference_scores=dict(data.get("reference_scores") or {}),
        ladder=ladder,
        categories_from=data["categories_from"],
    )


def list_packs(packs_dir: Path = PACKS_DIR) -> list[Pack]:
    if not packs_dir.is_dir():
        return []
    return [load_pack(p) for p in sorted(packs_dir.glob("*.yaml"))]


def get_pack(name: str, packs_dir: Path = PACKS_DIR) -> Pack:
    path = packs_dir / f"{name}.yaml"
    if not path.is_file():
        known = ", ".join(p.name for p in list_packs(packs_dir)) or "(none)"
        raise KeyError(f"unknown pack {name!r}; shipped packs: {known}")
    return load_pack(path)
