"""``neuralgentics init`` command (§1.1 of the design doc).

Bootstraps a project directory with the neuralgentics OpenCode plugin by:
  1. Resolving the requested plugin version (``latest`` or ``X.Y.Z``).
  2. Downloading + verifying the GitHub release tarball.
  3. Extracting + placing files per the §1.1 file-placement table.
  4. Merging ``opencode.json`` (preserves user's provider/mcp/lsp/formatter).
  5. Running ``npm install --no-audit --no-fund`` in ``.opencode/``.
  6. Writing the state file (§3) with the file manifest.
  7. Printing a plain-text success summary.

Public API: :func:`run_init`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from .. import __version__
from ._download import (
    download_tarball,
    extract_tarball,
    resolve_version,
    verify_sha256,
)
from ._errors import (
    BackupFailed,
    NpmInstallFailed,
    NpmNotFound,
    OfflineNoBundle,
    OpenCodeJsonInvalid,
    OpencodeNotFound,
    TargetNotDirectory,
    TargetRefused,
)
from ._merge import (
    merge_opencode_json_with_diff,
    parse_opencode_json,
    serialize_opencode_json,
)
from ._state import FileRecord, StateFile, compute_file_sha256, save_state

__all__ = ["run_init"]

log = logging.getLogger(__name__)

#: Plugin entry added to the ``plugin`` array (kept in sync with merge.py).
_PLUGIN_REFERENCE = "@veedubin/neuralgentics"

#: Regex for a valid ``X.Y.Z`` semver.
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")

#: Top-level files that are copied only if they don't already exist.
_COPY_IF_ABSENT: tuple[str, ...] = (
    "docker-compose.yml",
    "podman-compose.yml",
    "compose.example.env",
    "install.sh",
)

#: Top-level directory prefixes that are copied recursively (overwriting).
_COPY_TREE_PREFIXES: tuple[str, ...] = (
    ".opencode/agents/",
    ".opencode/skills/",
    "docker/",
    "node_modules/@veedubin/neuralgentics/",
)

#: Files that are merged (not copied) when they already exist.
_MERGE_FILES: tuple[str, ...] = (
    ".opencode/opencode.json",
    ".opencode/package.json",
)

#: Files that are copied only if absent (warn + skip if present).
_COPY_IF_ABSENT_WARN: tuple[str, ...] = (
    ".opencode/AGENTS.md",
    ".opencode/.gitignore",
    ".opencode/package-lock.json",
)


def run_init(args: argparse.Namespace) -> int:
    """Entry point for ``neuralgentics init``. Returns the process exit code."""
    target = _coerce_target(args.target)
    force = bool(getattr(args, "force", False))
    dry_run = bool(getattr(args, "dry_run", False))
    version = _resolve_version(args)

    # 1. Ensure target exists (mkdir -p), refuse scary locations / symlink .opencode.
    _ensure_target(target, force=force)

    # 2. Check opencode is on PATH.
    if shutil.which("opencode") is None:
        raise OpencodeNotFound(
            "opencode is not installed or not on PATH.",
        )

    # 3. Validate resolved version is X.Y.Z.
    if not _SEMVER_RE.match(version):
        raise ValueError(f"Resolved version {version!r} is not X.Y.Z.")  # noqa: TRY004

    if dry_run:
        # Dry-run: validate only, do not download or write anything. We can't
        # know if a backup would be needed without downloading the tarball, so
        # we report it as a possibility whenever an existing .opencode/ is
        # present (it would only actually happen if something changed).
        would_backup = (target / ".opencode").is_dir()
        backup_dir_name = _backup_path(target).name if would_backup else None
        _print_dry_run_summary(
            target,
            version,
            args.repo,
            would_backup=would_backup,
            backup_dir_name=backup_dir_name,
        )
        return 0

    # 4-5. Download + verify + extract.
    extract_dir = _download_and_extract(version, args.repo)

    # 6. Pre-flight: compute what would change vs. existing target.
    changes = compute_changes(extract_dir, target, force=force)

    # 7. Backup the existing .opencode/ if anything would change.
    backup_path: Path | None = None
    if _should_backup(target, changes):
        backup_path = do_backup(target)

    # 8. Place files per the §1.1 table. When a backup happened, the user's
    # previous .opencode/ is now at ``backup_path``; the merge helpers read
    # the user's old config from there so the deep-merge still preserves it.
    manifest = _place_files(
        extract_dir, target, version, force=force, backup_path=backup_path
    )

    # 9. Run npm install.
    _run_npm_install(target / ".opencode")

    # 10. Write state file (now includes last_backup if any).
    _write_state(target, version, manifest, args.repo, backup_path=backup_path)

    # 11. Print summary.
    _print_success_summary(target, version, extract_dir, backup_path=backup_path)
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_target(raw: object) -> Path:
    """Coerce the argparse ``--target`` value into an absolute :class:`Path`."""
    if isinstance(raw, Path):
        return raw.resolve()
    return Path(str(raw)).resolve()


def _resolve_version(args: argparse.Namespace) -> str:
    """Resolve the requested version, honoring ``--offline`` conflicts."""
    if getattr(args, "offline", False) and args.version == "latest":
        # §1.1 conflict: --offline + latest needs a bundled tarball we don't ship yet.
        raise OfflineNoBundle(
            "Offline mode requires a bundled tarball, which is not available in this version.",
        )
    return resolve_version(args.version, args.repo)


def _download_and_extract(version: str, repo: str) -> Path:
    """Download, verify, and extract the tarball to a temp dir.

    The extracted tree is kept alive (in a per-call mkdtemp dir) long enough
    for :func:`_place_files` to copy from it. Callers do not need to clean up
    — the OS reaps ``$TMPDIR`` on reboot.
    """
    tarball_path, checksums_path = download_tarball(version, repo)
    verify_sha256(tarball_path, checksums_path)
    extract_dir = Path(tempfile.mkdtemp(prefix=f"neuralgentics-extract-{version}-"))
    extract_tarball(tarball_path, extract_dir)
    return extract_dir


def _place_files(
    extract_dir: Path,
    target: Path,
    version: str,
    *,
    force: bool,
    backup_path: Path | None = None,
) -> dict[str, FileRecord]:
    """Walk ``extract_dir`` and place every file per the §1.1 table.

    When ``backup_path`` is set, the user's previous ``.opencode/`` was moved
    there just before this call; merge helpers read the user's prior config
    from ``backup_path/.opencode/`` so the deep merge still preserves the
    user's provider/mcp/lsp blocks.

    Returns the manifest of placed files (rel path → :class:`FileRecord`).
    """
    manifest: dict[str, FileRecord] = {}
    opencode_dir = target / ".opencode"
    opencode_dir.mkdir(parents=True, exist_ok=True)

    for src in sorted(_iter_files(extract_dir)):
        rel = src.relative_to(extract_dir).as_posix()
        dest_rel = _classify_destination(rel)
        if dest_rel is None:
            log.warning("Skipping unrecognised tarball entry: %s", rel)
            continue

        dest = target / dest_rel
        # Where the user's *previous* version of this file lives now (after a
        # backup). When no backup happened, it's just ``dest``. The backup dir
        # contains the *contents* of the old ``.opencode/`` at its top level
        # (shutil.move moves the dir's contents, not the dir itself), so we
        # strip the leading ``.opencode/`` from dest_rel when joining.
        if backup_path is not None:
            inner_rel = dest_rel.removeprefix(".opencode/")
            prev_dest = backup_path / inner_rel
        else:
            prev_dest = dest

        if dest_rel == ".opencode/opencode.json":
            _merge_opencode_json(src, dest, prev_dest, force=force)
            merged = True
        elif dest_rel == ".opencode/package.json":
            _merge_package_json(src, dest, prev_dest, force=force)
            merged = False
        elif dest_rel in _COPY_IF_ABSENT_WARN:
            if dest.exists() and not force:
                log.warning(
                    "%s already exists; skipping (use --force to overwrite).", dest_rel
                )
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            merged = False
        elif dest_rel in _COPY_IF_ABSENT:
            if dest.exists() and not force:
                log.info("%s already exists; skipping.", dest_rel)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            merged = False
        else:
            # Copy tree entry (agents/, skills/, docker/, node_modules/...).
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            merged = False

        sha = compute_file_sha256(dest)
        shipped_sha = compute_file_sha256(src)
        manifest[dest_rel] = FileRecord(
            sha256=sha,
            user_modified=False,
            installed_from=version,
            last_known_shipped_sha256=shipped_sha,
            merged=merged if dest_rel == ".opencode/opencode.json" else None,
        )
    return manifest


def _classify_destination(rel: str) -> str | None:
    """Map a tarball-relative path to its destination relative to ``target``.

    Returns ``None`` for unrecognised files (caller logs + skips).
    """
    # Exact-match files.
    if rel in _COPY_IF_ABSENT or rel in _COPY_IF_ABSENT_WARN or rel in _MERGE_FILES:
        return rel
    # Tree-prefix matches.
    for prefix in _COPY_TREE_PREFIXES:
        if rel.startswith(prefix):
            return rel
    # Anything else under .opencode/ that isn't explicitly listed above is
    # treated as a copy-tree entry (e.g. .opencode/skills/<name>/SKILL.md).
    if rel.startswith(".opencode/"):
        return rel
    return None


def _merge_opencode_json(
    src: Path, dest: Path, prev_dest: Path, *, force: bool
) -> None:
    """Merge the shipped ``opencode.json`` into the user's existing one.

    ``prev_dest`` is where the user's previous ``opencode.json`` lives now —
    normally the same as ``dest``, but after a backup it lives at
    ``backup_path/.opencode/opencode.json``.

    If the user's previous ``opencode.json`` is unparseable JSON, we treat it
    as absent: write the shipped version fresh. The user's broken file is
    already preserved in the ``.opencode-bak-*/`` directory.
    """
    shipped_text = src.read_text(encoding="utf-8")
    shipped = parse_opencode_json(shipped_text)
    if not prev_dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(serialize_opencode_json(shipped), encoding="utf-8")
        return
    user_text = prev_dest.read_text(encoding="utf-8")
    try:
        user = parse_opencode_json(user_text)
    except OpenCodeJsonInvalid:
        # User's existing file is broken JSON. Their data is already in
        # the .opencode-bak-*/ tree. Write the shipped version fresh.
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(serialize_opencode_json(shipped), encoding="utf-8")
        return
    merged, changes = merge_opencode_json_with_diff(user, shipped)
    if not changes:
        # Idempotent: nothing to do. (When prev_dest != dest — i.e. a backup
        # happened — we still need to write the merged result to the new
        # ``dest`` so the user's config is restored.)
        if prev_dest != dest:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(serialize_opencode_json(merged), encoding="utf-8")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(serialize_opencode_json(merged), encoding="utf-8")


def _merge_package_json(src: Path, dest: Path, prev_dest: Path, *, force: bool) -> None:
    """Merge the shipped ``package.json`` into the user's existing one.

    Union of ``dependencies`` maps; user's version wins on key conflict.
    ``prev_dest`` is where the user's prior ``package.json`` lives now (may
    differ from ``dest`` after a backup).

    If the user's previous ``package.json`` is unparseable JSON, write the
    shipped version fresh (user's data is in the backup).
    """
    shipped = json.loads(src.read_text(encoding="utf-8"))
    if not prev_dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            json.dumps(shipped, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return
    try:
        user = json.loads(prev_dest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            json.dumps(shipped, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return
    if not isinstance(user, dict):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            json.dumps(shipped, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return
    user_deps = user.get("dependencies", {})
    shipped_deps = shipped.get("dependencies", {})
    merged_deps = {**shipped_deps, **user_deps}  # user wins
    user["dependencies"] = merged_deps
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(user, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_npm_install(opencode_dir: Path) -> None:
    """Run ``npm install --no-audit --no-fund`` in ``opencode_dir``."""
    if shutil.which("npm") is None:
        raise NpmNotFound("npm is not installed or not on PATH.")
    result = subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund"],
        cwd=str(opencode_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise NpmInstallFailed(
            f"npm install failed in {opencode_dir}: {result.stderr.strip()}",
        )


def _write_state(
    target: Path,
    version: str,
    manifest: dict[str, FileRecord],
    repo: str,
    *,
    backup_path: Path | None = None,
) -> None:
    """Build + save the state file (§3)."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    last_backup: str | None = None
    if backup_path is not None:
        try:
            last_backup = str(backup_path.relative_to(target))
        except ValueError:
            last_backup = str(backup_path)
    state = StateFile(
        installed_version=version,
        installed_at=now,
        updated_at=now,
        cli_version=__version__,
        repo=repo,
        target=str(target),
        files=manifest,
        last_backup=last_backup,
    )
    save_state(target, state)


def _print_success_summary(
    target: Path,
    version: str,
    extract_dir: Path,
    *,
    backup_path: Path | None = None,
) -> None:
    """Print the plain-text success summary (§9.4 + task brief)."""
    agents = len(list((extract_dir / ".opencode" / "agents").glob("*.md")))
    skills = 0
    skills_dir = extract_dir / ".opencode" / "skills"
    if skills_dir.is_dir():
        skills = sum(1 for p in skills_dir.iterdir() if p.is_dir())
    state_path = target / ".opencode" / ".neuralgentics-state.json"
    use_color = _should_use_color()
    check = "\u2713" if use_color else "OK"
    backup_line = ""
    if backup_path is not None:
        backup_line = f"Backup:  {backup_path}\n"
    print(
        f"{check} neuralgentics v{version} initialized in {target}\n"
        f"\n"
        f"Plugin:  {target}/.opencode/node_modules/@veedubin/neuralgentics/ (after `npm install`)\n"
        f"Config:  {target}/.opencode/opencode.json\n"
        f"Agents:  {agents} personas\n"
        f"Skills:  {skills} skills\n"
        f"State:   {state_path}\n"
        f"{backup_line}"
        f"\n"
        f"Next: opencode\n"
        f"\n"
        f"Alternative install: curl -fsSL "
        f"https://raw.githubusercontent.com/Veedubin/neuralgentics/main/scripts/install.sh | bash"
    )


def _print_dry_run_summary(
    target: Path,
    version: str,
    repo: str,
    *,
    would_backup: bool = False,
    backup_dir_name: str | None = None,
) -> None:
    """Print what WOULD happen (no files written, no downloads)."""
    print(f"WOULD initialize neuralgentics v{version} in {target}")
    print(
        f"WOULD download tarball from github.com/{repo}/releases/download/v{version}/"
    )
    print(
        "WOULD extract + place .opencode/agents/, .opencode/skills/, opencode.json (merged)"
    )
    if would_backup:
        label = backup_dir_name or ".opencode-bak-{ts}/"
        print(f"WOULD back up existing .opencode/ -> {label}")
    print("WOULD run: npm install --no-audit --no-fund")
    print("WOULD write state file.")


def _should_use_color() -> bool:
    """Return ``True`` if ANSI color is appropriate (per §1.1 + NO_COLOR)."""
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _iter_files(root: Path) -> Iterator[Path]:
    """Yield every regular file under ``root`` (sorted, deterministic)."""
    for p in sorted(root.rglob("*")):
        if p.is_file():
            yield p


# ---------------------------------------------------------------------------
# Backup + target-creation helpers (v0.1.2)
# ---------------------------------------------------------------------------
#: Paths that are never a safe init target unless ``--force`` is set.
_SCARY_PATHS: frozenset[str] = frozenset(
    {
        "/",
        "/tmp",
        "/tmp/",
    }
)

#: Project markers that, if present, indicate ``target`` is a real project dir
#: (and therefore not "scary" even if it's directly under $HOME).
_PROJECT_MARKERS: tuple[str, ...] = (
    ".git",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    ".opencode",
)


def _ensure_target(target: Path, *, force: bool) -> None:
    """Create ``target`` if missing; refuse scary locations / symlink .opencode.

    Raises :class:`TargetNotDirectory` if the path exists but is a file.
    Raises :class:`TargetRefused` if the target looks scary (HOME, /, /tmp, or
    a shallow ~/* dir with no project markers) and ``--force`` was not set.
    Also raises :class:`TargetRefused` if an existing ``.opencode/`` is a
    symlink (we can't safely move a symlink).
    """
    if target.exists() and not target.is_dir():
        raise TargetNotDirectory(f"{target} exists and is not a directory.")
    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)

    if not force:
        if _is_scary_target(target):
            raise TargetRefused(
                f"Refusing to init into {target} (looks like a home/root/tmp dir). "
                "Use --force to proceed anyway.",
            )
        if _is_symlink_dir(target / ".opencode"):
            raise TargetRefused(
                f"{target}/.opencode is a symlink; refusing to move it. "
                "Use --force to proceed anyway.",
            )


def _is_scary_target(target: Path) -> bool:
    """Heuristic: would init'ing into ``target`` risk clobbering a lot?

    Returns ``True`` when ``target`` is exactly ``$HOME``, ``/``, ``/tmp``, or a
    direct child of ``$HOME`` with no project markers (``.git``,
    ``pyproject.toml``, ``package.json``, ``Cargo.toml``, or an existing
    ``.opencode/``). Returns ``False`` for normal project dirs.
    """
    resolved = target.resolve()
    try:
        resolved_str = str(resolved)
    except (ValueError, OSError):  # pragma: no cover — defensive
        return False

    # Exact scary paths.
    if resolved_str in _SCARY_PATHS:
        return True

    home = Path(os.path.expanduser("~")).resolve()
    if resolved == home:
        return True

    # A direct child of $HOME with no project markers is suspicious.
    return resolved.parent == home and not any(
        (resolved / m).exists() for m in _PROJECT_MARKERS
    )


def _is_symlink_dir(p: Path) -> bool:
    """Return ``True`` if ``p`` exists and is a symlink (even if it points at a dir)."""
    try:
        return p.is_symlink()
    except OSError:  # pragma: no cover — defensive against broken stat
        return False


def compute_changes(
    extract_dir: Path,
    target: Path,
    *,
    force: bool,  # noqa: ARG001 — kept for API symmetry; not currently used
) -> dict[str, str]:
    """Read-only simulation of :func:`_place_files`.

    Returns a map of ``rel_path -> action`` where action is one of:
      * ``"would_change"`` — file exists at target and content differs
      * ``"would_create"`` — file does not yet exist at target
      * ``"no_change"``    — file exists and content is identical

    For merge files (``opencode.json``, ``package.json``) we use the same
    merge logic as :func:`_place_files` and compare the merged result to the
    existing on-disk content.
    """
    changes: dict[str, str] = {}
    for src in sorted(_iter_files(extract_dir)):
        rel = src.relative_to(extract_dir).as_posix()
        dest_rel = _classify_destination(rel)
        if dest_rel is None:
            continue
        dest = target / dest_rel

        if dest_rel == ".opencode/opencode.json":
            action = _simulate_merge_opencode_json(src, dest)
        elif dest_rel == ".opencode/package.json":
            action = _simulate_merge_package_json(src, dest)
        elif dest_rel in _COPY_IF_ABSENT_WARN or dest_rel in _COPY_IF_ABSENT:
            # Copy-if-absent: a change only happens if the file is missing
            # (existing files are skipped unless --force, which would change
            # them — but we keep the conservative "no_change" here unless
            # forced; force is not currently wired for these in the would-change
            # sense).
            action = "would_create" if not dest.exists() else "no_change"
        else:
            # Copy-tree entry: compare bytes.
            if not dest.exists():
                action = "would_create"
            elif not _same_bytes(src, dest):
                action = "would_change"
            else:
                action = "no_change"
        changes[dest_rel] = action
    return changes


def _simulate_merge_opencode_json(src: Path, dest: Path) -> str:
    """Classify the action for an ``opencode.json`` merge.

    If the user's existing ``opencode.json`` is unparseable JSON, we treat it
    as ``"would_change"`` — the new install will replace it. The whole
    ``.opencode/`` will be backed up first (per :func:`_should_backup` and
    :func:`do_backup`), so the user's broken file is preserved.
    """
    if not dest.exists():
        return "would_create"
    shipped_text = src.read_text(encoding="utf-8")
    try:
        shipped = parse_opencode_json(shipped_text)
    except OpenCodeJsonInvalid:
        # Shipped file is broken? Should never happen, but treat as a change.
        return "would_change"
    user_text = dest.read_text(encoding="utf-8")
    try:
        user = parse_opencode_json(user_text)
    except OpenCodeJsonInvalid:
        # User's existing file is broken JSON. New install will replace it.
        # The whole .opencode/ is about to be backed up, so the broken file
        # is preserved.
        return "would_change"
    try:
        merged, diff = merge_opencode_json_with_diff(user, shipped)
    except Exception:  # noqa: BLE001 — defensive; merge should not raise, but if it does treat as change
        return "would_change"
    if not diff:
        return "no_change"
    # The merge would change the file only if the serialized result differs.
    existing_canonical = serialize_opencode_json(user)
    merged_canonical = serialize_opencode_json(merged)
    if merged_canonical == existing_canonical:
        return "no_change"
    return "would_change"


def _simulate_merge_package_json(src: Path, dest: Path) -> str:
    """Classify the action for a ``package.json`` merge.

    If the user's existing ``package.json`` is unparseable, treat as
    ``"would_change"`` — the user's data is preserved by the surrounding
    ``.opencode/`` backup.
    """
    if not dest.exists():
        return "would_create"
    try:
        shipped = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "would_change"
    try:
        user = json.loads(dest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "would_change"
    if not isinstance(user, dict) or not isinstance(shipped, dict):
        return "would_change"
    shipped_deps = shipped.get("dependencies", {})
    user_deps = user.get("dependencies", {})
    merged_deps = {**shipped_deps, **user_deps}
    if merged_deps == user_deps:
        # Nothing new would be added.
        return "no_change"
    return "would_change"


def _same_bytes(a: Path, b: Path) -> bool:
    """Return ``True`` if ``a`` and ``b`` have identical content."""
    return compute_file_sha256(a) == compute_file_sha256(b)


def _should_backup(target: Path, changes: dict[str, str]) -> bool:
    """Return ``True`` if ``target/.opencode`` exists and something would change."""
    if not (target / ".opencode").is_dir():
        return False
    return any(
        action in ("would_change", "would_create") for action in changes.values()
    )


def _backup_path(target: Path) -> Path:
    """Generate a ``.opencode-bak-{unix_timestamp}/`` path under ``target``.

    If the generated path already exists (rapid re-run collision), append a
    short random suffix to make it unique.
    """
    ts = int(time.time())
    base = target / f".opencode-bak-{ts}"
    if not base.exists():
        return base
    # Collision: add a short suffix.
    import secrets

    while base.exists():
        suffix = secrets.token_hex(2)
        base = target / f".opencode-bak-{ts}-{suffix}"
    return base


def do_backup(target: Path) -> Path:
    """Move ``target/.opencode`` to ``.opencode-bak-{ts}/``; return the new path.

    Raises :class:`BackupFailed` if the move fails.
    """
    src = target / ".opencode"
    dest = _backup_path(target)
    try:
        shutil.move(str(src), str(dest))
    except OSError as exc:
        raise BackupFailed(
            f"Could not move {src} to {dest}: {exc}",
            remediation="Check disk space and permissions in the target directory.",
        ) from exc
    return dest
