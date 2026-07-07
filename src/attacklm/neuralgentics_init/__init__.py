"""Neuralgentics OpenCode plugin bootstrap (init flow).

Public API:
  - :func:`run_neuralgentics_init` — entry point for ``attacklm --init``.
  - :func:`format_neuralgentics_error` — format a :class:`NeuralgenticsError`
    for stderr output.
  - :class:`NeuralgenticsError` and its subclasses (for error handling).

The init flow downloads a neuralgentics release tarball from GitHub,
SHA256-verifies it, extracts to the target directory, deep-merges the
user's ``opencode.json`` (preserving the user's ``provider`` / ``mcp`` /
``lsp`` / ``formatter`` blocks), runs ``npm install --no-audit --no-fund``
in ``.opencode/``, and writes a state file at
``{target}/.opencode/.neuralgentics-state.json``.

The CLI entry point is ``attacklm --init`` — a *top-level flag*, NOT a
subcommand. The existing ``attacklm init`` SUBCOMMAND continues to delegate
to ``attacklm-dataset`` for the training-dataset init flow; the two
operations are intentionally separate and target different artefacts.
"""

from __future__ import annotations

import argparse

from .. import __version__ as _attacklm_version
from ._errors import (
    BackupFailed,
    ComposeNotFound,
    ComposeUpFailed,
    ExtractionFailed,
    MergeConflict,
    NeuralgenticsError,
    NetworkError,
    NpmInstallFailed,
    NpmNotFound,
    OfflineNoBundle,
    OpenCodeJsonInvalid,
    OpencodeNotFound,
    PermissionDenied,
    Sha256Mismatch,
    TarballCorrupt,
    TargetNotDirectory,
    TargetRefused,
    VersionNotFound,
    format_error,
)
from ._init_cmd import run_init as _run_init

__all__ = [
    "BackupFailed",
    "ComposeNotFound",
    "ComposeUpFailed",
    "ExtractionFailed",
    "MergeConflict",
    "NetworkError",
    "NeuralgenticsError",
    "NpmInstallFailed",
    "NpmNotFound",
    "OfflineNoBundle",
    "OpenCodeJsonInvalid",
    "OpencodeNotFound",
    "PermissionDenied",
    "Sha256Mismatch",
    "TarballCorrupt",
    "TargetNotDirectory",
    "TargetRefused",
    "VersionNotFound",
    "format_error",
    "format_neuralgentics_error",
    "run_neuralgentics_init",
    "__version__",
]

# Re-export the attacklm version. The init state file records which CLI version
# performed the install; we record the parent attacklm version, not the deleted
# neuralgentics-cli version.
__version__ = _attacklm_version


def run_neuralgentics_init(args: argparse.Namespace) -> int:
    """Entry point for ``attacklm --init``. Returns the process exit code.

    Wraps the internal :func:`_run_init` so the public surface is stable
    and future changes to the internal module layout don't break callers.
    """
    return _run_init(args)


def format_neuralgentics_error(err: NeuralgenticsError) -> str:
    """Format a :class:`NeuralgenticsError` for stderr output.

    Returns plain text in the form::

        [ERROR] {message}
        Suggestion: {remediation}
    """
    return format_error(err)
