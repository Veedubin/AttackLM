"""State file read/write/validate for the neuralgentics CLI.

Implements §3 of the design doc. The state file lives at
``{target}/.opencode/.neuralgentics-state.json`` and tracks every file the
bootstrapper extracted, plus whether the user has modified each one.

Public API:

- :class:`StateFile` — pydantic v2 model matching §3.2.
- :class:`FileRecord`, :class:`BackendRecord` — sub-models.
- :func:`load_state` — read the state file; returns ``None`` if missing or
  corrupted (corruption is logged, never raised).
- :func:`save_state` — write the state file atomically (write-then-rename).
- :func:`compute_file_sha256` — stdlib hashlib, 64K chunked read.
- :func:`build_manifest_from_directory` — rebuild the ``files`` manifest from
  a list of relative paths under the target (corruption recovery).
- :func:`detect_user_modifications` — recompute SHA256 for every manifest
  entry; set ``user_modified = True`` if a file's current hash differs from the
  recorded shipped hash. NEVER auto-reset to ``False`` (§3.3 mitigation).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

__all__ = [
    "BackendRecord",
    "FileRecord",
    "StateFile",
    "build_manifest_from_directory",
    "compute_file_sha256",
    "detect_user_modifications",
    "load_state",
    "save_state",
]

log = logging.getLogger(__name__)

#: Name of the state file inside ``{target}/.opencode/``.
STATE_FILENAME = ".neuralgentics-state.json"

#: Read chunk size for SHA256 computation (64 KiB).
_CHUNK_SIZE = 64 * 1024


class FileRecord(BaseModel):
    """One file in the installation manifest (§3.2 ``files.{path}``)."""

    model_config = ConfigDict(extra="forbid")

    #: SHA256 of the file as it currently exists on disk (recorded at
    #: install/update time).
    sha256: str
    #: ``True`` once the file's SHA256 has been observed to differ from the
    #: shipped SHA256. Set to ``True`` on first detection, never auto-reset.
    user_modified: bool = False
    #: Plugin version that originally installed this file.
    installed_from: str
    #: SHA256 of this file as shipped in the tarball. Used during ``update`` to
    #: detect whether the shipped version changed. ``None`` for files installed
    #: before this field existed or for files whose shipped hash was never
    #: recorded.
    last_known_shipped_sha256: str | None = None
    #: ``True`` if this file was merged (not simply copied). Only relevant for
    #: ``opencode.json``. ``None`` for non-merged files.
    merged: bool | None = None


class BackendRecord(BaseModel):
    """Backend (container stack) state (§3.2 ``backend``)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    compose_file: str
    compose_tool: str
    env_file: str | None = None
    containers: list[str]


class StateFile(BaseModel):
    """The full state file (§3.2)."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    installed_version: str
    installed_at: datetime
    updated_at: datetime
    cli_version: str
    source: Literal["github"] = "github"
    repo: str
    target: str
    files: dict[str, FileRecord]
    backend: BackendRecord | None = None
    #: Path to the most recent ``.opencode-bak-{ts}/`` backup directory created
    #: during install/update, or ``None`` when no backup was needed. Relative
    #: to ``target`` when set. Added in v0.1.2.
    last_backup: str | None = None


def _state_path(target: Path) -> Path:
    """Return the absolute path to the state file for ``target``."""
    return target.resolve() / ".opencode" / STATE_FILENAME


def load_state(target: Path) -> StateFile | None:
    """Read the state file for ``target``.

    Returns the parsed :class:`StateFile`, or ``None`` if the file does not
    exist or cannot be parsed. Corruption is logged at WARNING level and never
    raised — the caller (``init``/``update``) handles recovery.
    """
    path = _state_path(target)
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("State file at %s is unreadable (%s); ignoring.", path, exc)
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning("State file at %s is not valid JSON (%s); ignoring.", path, exc)
        return None
    try:
        return StateFile.model_validate(data)
    except Exception as exc:  # noqa: BLE001 — pydantic raises ValidationError (a subclass of ValueError)
        log.warning("State file at %s failed schema validation (%s); ignoring.", path, exc)
        return None


def save_state(target: Path, state: StateFile) -> None:
    """Write ``state`` to ``{target}/.opencode/.neuralgentics-state.json``.

    Writes to a ``.tmp`` sibling first, then :func:`os.replace` swaps it into
    place atomically. Creates the parent ``.opencode/`` directory if needed.
    """
    path = _state_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = state.model_dump_json(indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    # os.replace is atomic on the same filesystem.
    import os

    os.replace(tmp, path)


def compute_file_sha256(path: Path) -> str:
    """Return the hex SHA256 of ``path`` (64 KiB chunked read)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_manifest_from_directory(target: Path, file_paths: list[str]) -> dict[str, FileRecord]:
    """Build a ``files`` manifest by scanning ``file_paths`` under ``target``.

    Each entry in ``file_paths`` is a path relative to ``target``. Missing
    files are silently skipped (corruption recovery only tracks files that
    actually exist). ``installed_from`` is set to ``"unknown"`` because this
    rebuild path has no version context — the caller overwrites it if known.
    """
    manifest: dict[str, FileRecord] = {}
    base = target.resolve()
    for rel in file_paths:
        full = base / rel
        if not full.is_file():
            continue
        manifest[rel] = FileRecord(
            sha256=compute_file_sha256(full),
            user_modified=True,  # §3.4: treat rebuild-from-corruption files as modified.
            installed_from="unknown",
        )
    return manifest


def detect_user_modifications(state: StateFile, target: Path) -> StateFile:
    """Recompute SHA256 for every file in ``state.files`` and flag changes.

    For each manifest entry, compute the file's current SHA256 on disk. If it
    differs from the recorded shipped hash (``last_known_shipped_sha256`` if
    present, else the recorded ``sha256``), set ``user_modified = True``.

    Per §3.3, ``user_modified`` is NEVER auto-reset to ``False``: once it has
    been set, it stays set even if the file's content later matches the
    shipped hash again. Missing files are treated as modifications (deletion).
    """
    base = target.resolve()
    # Work on a copy so the caller's `state` is not mutated in place.
    new_files: dict[str, FileRecord] = {}
    for rel, rec in state.files.items():
        full = base / rel
        if not full.is_file():
            # File was removed — that's a user modification.
            new_files[rel] = rec.model_copy(update={"user_modified": True})
            continue
        current = compute_file_sha256(full)
        baseline = rec.last_known_shipped_sha256
        if baseline is None:
            baseline = rec.sha256
        modified = rec.user_modified or (current != baseline)
        new_files[rel] = rec.model_copy(update={"sha256": current, "user_modified": modified})
    return state.model_copy(update={"files": new_files})
