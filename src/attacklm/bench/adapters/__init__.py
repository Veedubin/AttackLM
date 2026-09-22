"""Harness adapters.

The harness is imported ONLY inside this subpackage. Everything else in
``attacklm.bench`` talks to a ``HarnessAdapter``, which is what keeps a second
harness cheap to add later (spec §11) and keeps vLLM's torch pins out of the
training environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class HarnessInfo:
    name: str
    version: str | None
    available: bool


class HarnessAdapter(Protocol):
    name: str

    def build_argv(self, pack: Any, cfg: Any) -> list[str]: ...

    def parse_log(self, log_dir: Path) -> list[Any]: ...

    def probe(self) -> HarnessInfo: ...
