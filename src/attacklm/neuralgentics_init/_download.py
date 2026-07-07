"""GitHub release download, SHA256 verification, and tarball extraction.

Implements §2 (Option A: download from GitHub release at runtime) and the
``extract.py`` portion of the design (Appendix A.11.6 lists ``download.py``
and ``extract.py`` as separate files, but the task brief collapses them into
one module; this file owns both concerns).

Public API:

- :func:`resolve_version` — turn ``"latest"`` or ``"X.Y.Z"`` into a concrete
  semver string, calling the GitHub API for ``latest``. Cached for 1 hour in
  ``~/.cache/neuralgentics/version_cache.json``.
- :func:`download_tarball` — stream ``neuralgentics-{version}.tar.gz`` and
  ``checksums.txt`` from a GitHub release into a per-pid temp dir. Returns
  ``(tarball_path, checksums_path)``.
- :func:`verify_sha256` — parse ``checksums.txt`` and compare the tarball's
  actual SHA256 against it. Raises :class:`Sha256Mismatch` on mismatch.
- :func:`extract_tarball` — extract the tarball to ``dest`` with
  ``--strip-components=1`` semantics (the archive's top-level
  ``neuralgentics-{version}/`` directory is stripped). Verifies that the
  extraction produced ``.opencode/agents/coder.md``; raises
  :class:`ExtractionFailed` if missing.

Internal helpers are private. Only stdlib + ``httpx`` are used.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tarfile
import time
from pathlib import Path

import httpx

from ._errors import (
    ExtractionFailed,
    NetworkError,
    Sha256Mismatch,
    TarballCorrupt,
    VersionNotFound,
)

__all__ = [
    "download_tarball",
    "extract_tarball",
    "resolve_version",
    "verify_sha256",
]

log = logging.getLogger(__name__)

#: Base URL for the GitHub REST API.
_GITHUB_API = "https://api.github.com"

#: Base URL for GitHub release asset downloads (not rate-limited).
_GITHUB_DOWNLOAD = "https://github.com"

#: Cache file for ``latest`` version resolution (§2: 1-hour TTL).
_CACHE_DIR = Path("~/.cache/neuralgentics").expanduser()
_CACHE_FILE = _CACHE_DIR / "version_cache.json"

#: Cache TTL in seconds (1 hour).
_CACHE_TTL = 3600

#: Read chunk size for SHA256 computation (64 KiB).
_CHUNK_SIZE = 64 * 1024

#: Regex for a valid ``X.Y.Z`` semver (no pre-release suffix for v0.1.0).
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")

#: The known required file inside the extracted tarball layout (§A.11.6 +
#: neuralgentics/AGENTS.md "What's in the archive"). Used as an extraction
#: sanity check.
_REQUIRED_EXTRACTED_FILE = Path(".opencode/agents/coder.md")


# ---------------------------------------------------------------------------
# Version resolution
# ---------------------------------------------------------------------------


def resolve_version(version: str, repo: str, *, github_token: str | None = None) -> str:
    """Resolve ``version`` to a concrete ``X.Y.Z`` string.

    - If ``version == "latest"``, query the GitHub API
      ``GET /repos/{repo}/releases/latest`` and return ``tag_name`` with the
      leading ``v`` stripped. Cached for 1 hour in
      ``~/.cache/neuralgentics/version_cache.json`` (keyed by ``repo``).
    - Otherwise, validate ``version`` matches ``X.Y.Z`` and return it as-is.
      Invalid versions raise :class:`VersionNotFound`.
    """
    if version == "latest":
        cached = _read_cached_latest(repo)
        if cached is not None:
            return cached
        tag = _fetch_latest_tag(repo, github_token)
        resolved = tag.lstrip("v")
        if not _SEMVER_RE.match(resolved):
            raise VersionNotFound(
                f"GitHub returned a non-semver tag name {tag!r} for {repo}."
            )
        _write_cached_latest(repo, resolved)
        return resolved
    if _SEMVER_RE.match(version):
        return version
    raise VersionNotFound(f"Version {version!r} is not a valid X.Y.Z semver.")


def _read_cached_latest(repo: str) -> str | None:
    """Return the cached ``latest`` version for ``repo`` if fresh (<1h old)."""
    if not _CACHE_FILE.is_file():
        return None
    try:
        data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    entry = data.get(repo)
    if not isinstance(entry, dict):
        return None
    ts = entry.get("ts")
    ver = entry.get("version")
    if not isinstance(ts, int | float) or not isinstance(ver, str):
        return None
    if time.time() - ts > _CACHE_TTL:
        return None
    return ver


def _write_cached_latest(repo: str, version: str) -> None:
    """Persist ``version`` as the cached ``latest`` for ``repo``."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data: dict[str, dict[str, object]] = {}
        if _CACHE_FILE.is_file():
            try:
                data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
        data[repo] = {"ts": time.time(), "version": version}
        _CACHE_FILE.write_text(json.dumps(data), encoding="utf-8")
    except OSError as exc:
        log.debug("Could not write version cache at %s: %s", _CACHE_FILE, exc)


def _fetch_latest_tag(repo: str, github_token: str | None) -> str:
    """Call the GitHub API for the latest release tag name."""
    url = f"{_GITHUB_API}/repos/{repo}/releases/latest"
    headers = {"Accept": "application/vnd.github+json"}
    token = github_token or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with httpx.Client(follow_redirects=True, timeout=30.0) as client:
            resp = client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        raise NetworkError(
            f"Failed to reach GitHub API at {url}: {exc}",
            remediation="Check your network connection and that the repository exists.",
        ) from exc
    if resp.status_code == 404:
        raise VersionNotFound(
            f"No releases found for {repo}.",
        )
    if resp.status_code != 200:
        raise NetworkError(
            f"GitHub API returned {resp.status_code} {resp.reason_phrase} for {url}.",
        )
    try:
        body = resp.json()
    except json.JSONDecodeError as exc:
        raise NetworkError(
            f"GitHub API returned invalid JSON for {url}: {exc}.",
        ) from exc
    tag = body.get("tag_name")
    if not isinstance(tag, str) or not tag:
        raise VersionNotFound(
            f"GitHub API response for {repo} had no 'tag_name' field.",
        )
    return tag


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _resolve_token(github_token: str | None) -> str | None:
    return github_token or os.environ.get("GITHUB_TOKEN")


def download_tarball(
    version: str, repo: str, *, github_token: str | None = None
) -> tuple[Path, Path]:
    """Download ``neuralgentics-{version}.tar.gz`` and ``checksums.txt``.

    Returns ``(tarball_path, checksums_path)``. The files are written to a
    per-pid, per-version temp directory under ``/tmp``. On any exception the
    temp directory is removed.
    """
    if not _SEMVER_RE.match(version):
        raise VersionNotFound(f"Version {version!r} is not a valid X.Y.Z semver.")

    tmp_dir = Path("/tmp") / f"neuralgentics-{os.getpid()}-{version}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tarball_path = tmp_dir / f"neuralgentics-{version}.tar.gz"
    checksums_path = tmp_dir / "checksums.txt"

    base = f"{_GITHUB_DOWNLOAD}/{repo}/releases/download/v{version}"
    tarball_url = f"{base}/neuralgentics-{version}.tar.gz"
    checksums_url = f"{base}/checksums.txt"
    headers: dict[str, str] = {}
    token = _resolve_token(github_token)
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        _stream_download(tarball_url, tarball_path, headers)
        _stream_download(checksums_url, checksums_path, headers)
    except Exception:
        # Cleanup the temp dir on any failure so we don't leave partial files.
        try:
            for p in tmp_dir.iterdir():
                p.unlink(missing_ok=True)
            tmp_dir.rmdir()
        except OSError as exc:
            log.debug("Cleanup of %s failed: %s", tmp_dir, exc)
        raise
    return (tarball_path, checksums_path)


def _stream_download(url: str, dest: Path, headers: dict[str, str]) -> None:
    """Stream ``url`` to ``dest`` (do not load the whole body in memory)."""
    try:
        with (
            httpx.Client(follow_redirects=True, timeout=120.0) as client,
            client.stream("GET", url, headers=headers) as resp,
        ):
            if resp.status_code != 200:
                raise NetworkError(
                    f"Failed to download {url}: {resp.status_code} {resp.reason_phrase}.",
                    remediation=(
                        "Check your network connection and that the version exists "
                        "on the GitHub releases page."
                    ),
                )
            with dest.open("wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)
    except NetworkError:
        raise
    except httpx.HTTPError as exc:
        raise NetworkError(
            f"Failed to download {url}: {exc}",
            remediation="Check your network connection and retry.",
        ) from exc


# ---------------------------------------------------------------------------
# SHA256 verification
# ---------------------------------------------------------------------------


def verify_sha256(tarball_path: Path, checksums_path: Path) -> None:
    """Verify ``tarball_path`` against the entry in ``checksums_path``.

    ``checksums.txt`` lines have the form ``<sha>  <filename>`` (two spaces,
    GNU coreutils format). The filename looked up is the tarball's basename.
    Raises :class:`Sha256Mismatch` if the file is missing from checksums or
    the computed hash doesn't match.
    """
    expected = _lookup_checksum(checksums_path, tarball_path.name)
    actual = _sha256_of_file(tarball_path)
    if actual != expected:
        raise Sha256Mismatch(
            f"SHA256 verification failed for {tarball_path.name}. "
            f"Expected: {expected}, got: {actual}.",
        )


def _lookup_checksum(checksums_path: Path, filename: str) -> str:
    """Parse ``checksums.txt`` and return the SHA256 for ``filename``."""
    text = checksums_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        sha, name = parts[0].strip(), parts[1].strip()
        # The checksum file may prefix the name with `*` (binary mode) or ``./``.
        if name.lstrip("*").lstrip("./") == filename:
            if not re.fullmatch(r"[0-9a-f]{64}", sha):
                continue
            return sha
    raise Sha256Mismatch(
        f"{filename} not found in {checksums_path.name}.",
    )


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_tarball(tarball_path: Path, dest: Path) -> None:
    """Extract ``tarball_path`` into ``dest`` with ``--strip-components=1``.

    The release tarball has a single top-level directory
    (``neuralgentics-{version}/``); this function flattens that one level so
    files land directly under ``dest`` (e.g. ``dest/.opencode/agents/coder.md``).

    Uses stdlib :mod:`tarfile` (no shelling out to ``tar``). After extraction,
    verifies that ``dest/.opencode/agents/coder.md`` exists — a known required
    file from the archive layout. Raises :class:`ExtractionFailed` if missing,
    :class:`TarballCorrupt` if the archive can't be opened.
    """
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(tarball_path, "r:gz") as tar:
            members = tar.getmembers()
            for m in members:
                rel = _strip_one(m.name)
                if rel is None or rel == "":
                    continue
                # Guard against path traversal (member escaping dest).
                target_path = (dest / rel).resolve()
                if not str(target_path).startswith(str(dest.resolve())):
                    raise TarballCorrupt(
                        f"Refusing to extract member outside dest: {m.name!r}",
                    )
                if m.isdir():
                    target_path.mkdir(parents=True, exist_ok=True)
                elif m.issym():
                    # Skip symlinks for safety; release tarball shouldn't have any.
                    continue
                elif m.islnk():
                    continue
                else:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    src = tar.extractfile(m)
                    if src is None:
                        continue
                    with src, target_path.open("wb") as out:
                        out.write(src.read())
    except tarfile.TarError as exc:
        raise TarballCorrupt(f"Failed to extract tarball: {exc}") from exc

    required = dest / _REQUIRED_EXTRACTED_FILE
    if not required.is_file():
        raise ExtractionFailed(
            f"{_REQUIRED_EXTRACTED_FILE} not found after extraction.",
        )


def _strip_one(name: str) -> str | None:
    """Strip the first path component of ``name`` (``--strip-components=1``)."""
    norm = name.replace("\\", "/").lstrip("./")
    if "/" not in norm:
        # A bare file at the archive root — nothing to strip to. Skip it.
        return None
    return norm.split("/", 1)[1]
