"""Error model for the neuralgentics CLI.

Every recoverable failure is a subclass of :class:`NeuralgenticsError` carrying
an ``exit_code`` and a human-readable ``remediation`` hint. The CLI's
``main()`` catches these, prints them via :func:`format_error`, and exits with
the error's ``exit_code``.

Error classes, exit codes, and remediation text follow §6 of the design doc
(`neuralgentics/docs/design/init-cli-bootstrapper.md`).
"""

from __future__ import annotations

# Class names follow the design doc (§6) exactly and intentionally do NOT use
# the "Error" suffix mandated by ruff's N818 rule. Disable it for this module.
# ruff: noqa: N818

__all__ = [
    "NeuralgenticsError",
    "OpencodeNotFound",
    "NpmNotFound",
    "NetworkError",
    "Sha256Mismatch",
    "TarballCorrupt",
    "ExtractionFailed",
    "OpenCodeJsonInvalid",
    "MergeConflict",
    "ComposeNotFound",
    "ComposeUpFailed",
    "NpmInstallFailed",
    "VersionNotFound",
    "OfflineNoBundle",
    "PermissionDenied",
    "TargetNotDirectory",
    "TargetRefused",
    "BackupFailed",
    "format_error",
]


class NeuralgenticsError(Exception):
    """Base class for all neuralgentics CLI errors.

    Subclasses set :attr:`exit_code` and :attr:`remediation` as class-level
    defaults. Instance attributes override them when a more specific message
    or remediation is needed.
    """

    #: Process exit code used when this error is uncaught by ``main()``.
    exit_code: int = 1

    #: Default remediation hint shown to the user.
    remediation: str = "See the documentation for troubleshooting."

    def __init__(self, message: str, *, remediation: str | None = None) -> None:
        super().__init__(message)
        if remediation is not None:
            self.remediation = remediation


class OpencodeNotFound(NeuralgenticsError):
    """``opencode`` is not installed or not on ``$PATH``."""

    exit_code = 4
    remediation = "Install OpenCode, then re-run: curl -fsSL https://opencode.ai/install.sh | bash"


class NpmNotFound(NeuralgenticsError):
    """``npm`` is not installed or not on ``$PATH``."""

    exit_code = 5
    remediation = (
        "Install Node.js 20+ from https://nodejs.org/, then re-run `npm install` in .opencode/."
    )


class NetworkError(NeuralgenticsError):
    """An HTTP request failed (connect error, bad status, etc.)."""

    exit_code = 6
    remediation = (
        "Check your network connection and verify the version exists on the GitHub releases page."
    )


class Sha256Mismatch(NeuralgenticsError):
    """Downloaded artifact's SHA256 did not match the published checksum."""

    exit_code = 7
    remediation = (
        "Re-run. If the problem persists, report it on GitHub issues — "
        "the download may be tampered."
    )


class TarballCorrupt(NeuralgenticsError):
    """The downloaded tarball could not be extracted."""

    exit_code = 8
    remediation = "Check disk space and re-download. The archive may be corrupt."


class ExtractionFailed(NeuralgenticsError):
    """A file expected after extraction was missing."""

    exit_code = 9
    remediation = "Check the release assets on GitHub — the archive may be incomplete."


class OpenCodeJsonInvalid(NeuralgenticsError):
    """``.opencode/opencode.json`` is not valid JSON."""

    exit_code = 3
    remediation = "Fix the JSON syntax error manually, then re-run."


class MergeConflict(NeuralgenticsError):
    """A shipped file changed but the user has local modifications."""

    exit_code = 10
    remediation = "Review the diff, then use --force to overwrite or resolve the conflict manually."


class ComposeNotFound(NeuralgenticsError):
    """Neither ``docker`` nor ``podman-compose`` is available."""

    exit_code = 11
    remediation = "Install Docker or podman-compose to use --with-backend."


class ComposeUpFailed(NeuralgenticsError):
    """``compose up -d`` returned a non-zero exit code."""

    exit_code = 12
    remediation = "Check the container runtime status, port conflicts, and disk space."


class NpmInstallFailed(NeuralgenticsError):
    """``npm install`` returned a non-zero exit code."""

    exit_code = 13
    remediation = (
        "Check Node.js version, network, and disk space. Try running `npm install` manually."
    )


class VersionNotFound(NeuralgenticsError):
    """The requested plugin version does not exist on GitHub."""

    exit_code = 14
    remediation = (
        "Check available versions at the GitHub releases page. "
        "Use 'latest' for the most recent release."
    )


class OfflineNoBundle(NeuralgenticsError):
    """``--offline`` was requested but no bundled tarball is available."""

    exit_code = 15
    remediation = "Run without --offline. Bundled tarball support is planned for v0.2.0+."


class PermissionDenied(NeuralgenticsError):
    """The CLI cannot write to the target path."""

    exit_code = 16
    remediation = "Fix directory permissions or run from a writable directory."


class TargetNotDirectory(NeuralgenticsError):
    """The ``--target`` path exists but is not a directory."""

    exit_code = 17
    remediation = "Create the directory or specify a different target with --target."


class TargetRefused(NeuralgenticsError):
    """The target path is a scary location (HOME, /, /tmp) or has a symlink .opencode/.

    Refuses to run unless ``--force`` is set, to avoid clobbering the user's
    home directory or following a symlink somewhere unexpected.
    """

    exit_code = 18
    remediation = "Use --force to proceed anyway, or pick a project directory."


class BackupFailed(NeuralgenticsError):
    """Moving ``.opencode/`` to a backup directory failed."""

    exit_code = 19
    remediation = "Check disk space and permissions in the target directory."


def format_error(err: NeuralgenticsError) -> str:
    """Format an error for stderr output (§6.1 of the design doc).

    Returns a plain-text string (no ANSI / no ``rich``) in the form::

        [ERROR] {message}
        Suggestion: {remediation}
    """
    message = str(err)
    return f"[ERROR] {message}\nSuggestion: {err.remediation}"
