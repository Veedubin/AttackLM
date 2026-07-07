"""Tests for :mod:`attacklm.neuralgentics_init._init_cmd` (``attacklm --init``).

All network + npm + opencode calls are mocked via ``monkeypatch`` so the
suite runs fully offline. These tests were ported from the deleted
``neuralgentics-cli`` PyPI wheel (commit 9b7723b^) — the imports were
re-targeted at the new ``attacklm.neuralgentics_init`` subpackage.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

# Make sure the attacklm package is importable when running from source.
# (Matches the sys.path.insert pattern used by tests/test_cli.py.)
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from attacklm.neuralgentics_init import _download as dl  # noqa: E402
from attacklm.neuralgentics_init import _init_cmd as init_cmd  # noqa: E402
from attacklm.neuralgentics_init._errors import (  # noqa: E402
    OfflineNoBundle,
    OpencodeNotFound,
    TargetNotDirectory,
    TargetRefused,
)
from attacklm.neuralgentics_init._state import STATE_FILENAME, load_state  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

REPO = "Veedubin/neuralgentics"


def _ns(**kw: Any) -> argparse.Namespace:
    """Build an argparse.Namespace with init defaults, overridden by ``kw``."""
    defaults = {
        "version": "latest",
        "target": Path.cwd(),
        "yes": False,
        "offline": False,
        "dry_run": False,
        "force": False,
        "repo": REPO,
    }
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def _build_tarball(
    tmp_path: Path,
    *,
    version: str,
    files: dict[str, bytes],
) -> tuple[Path, Path]:
    """Build a tarball + matching checksums.txt under ``tmp_path``.

    Returns ``(tarball_path, checksums_path)``.
    """
    tarball = tmp_path / f"neuralgentics-{version}.tar.gz"
    top = f"neuralgentics-{version}"
    with tarfile.open(tarball, "w:gz") as tar:
        dir_info = tarfile.TarInfo(name=top)
        dir_info.type = tarfile.DIRTYPE
        dir_info.mode = 0o755
        tar.addfile(dir_info)
        for rel, data in files.items():
            info = tarfile.TarInfo(name=f"{top}/{rel}")
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    sha = hashlib.sha256(tarball.read_bytes()).hexdigest()
    checksums = tmp_path / "checksums.txt"
    checksums.write_text(f"{sha}  {tarball.name}\n", encoding="utf-8")
    return tarball, checksums


def _patch_download(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tarball: Path,
    checksums: Path,
) -> None:
    """Patch ``download.download_tarball`` to return prebuilt files."""

    def fake_download(version: str, repo: str, *, github_token: str | None = None):
        return (tarball, checksums)

    monkeypatch.setattr(dl, "download_tarball", fake_download)
    monkeypatch.setattr(init_cmd, "download_tarball", fake_download)


def _patch_resolve_version(
    monkeypatch: pytest.MonkeyPatch,
    *,
    version: str,
) -> None:
    """Patch ``resolve_version`` to return a fixed version (no network)."""

    def fake_resolve(v: str, r: str, *, github_token: str | None = None) -> str:
        return version

    monkeypatch.setattr(dl, "resolve_version", fake_resolve)
    monkeypatch.setattr(init_cmd, "resolve_version", fake_resolve)


def _patch_npm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``npm`` to be on PATH and ``npm install`` to succeed."""
    monkeypatch.setattr(init_cmd.shutil, "which", lambda _cmd: "/usr/bin/npm")

    class _FakeCompletedProcess:
        returncode = 0
        stderr = ""

    def fake_run(*args: Any, **kwargs: Any) -> _FakeCompletedProcess:
        return _FakeCompletedProcess()

    monkeypatch.setattr(init_cmd.subprocess, "run", fake_run)


def _patch_opencode_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``shutil.which("opencode")`` to return a fake path."""
    orig_which = init_cmd.shutil.which

    def fake_which(cmd: str) -> str | None:
        if cmd == "opencode":
            return "/usr/bin/opencode"
        return orig_which(cmd)

    monkeypatch.setattr(init_cmd.shutil, "which", fake_which)


def _default_tarball_files() -> dict[str, bytes]:
    return {
        ".opencode/agents/coder.md": b"# coder agent\n",
        ".opencode/agents/architect.md": b"# architect agent\n",
        ".opencode/skills/boomerang-orchestrator/SKILL.md": b"# orchestrator skill\n",
        ".opencode/skills/kanban-board-manager/SKILL.md": b"# kanban skill\n",
        ".opencode/opencode.json": json.dumps(
            {
                "$schema": "https://opencode.ai/config.json",
                "plugin": ["@veedubin/neuralgentics"],
                "instructions": ["AGENTS.md"],
                "provider": {"ollama-cloud": {"name": "Ollama Cloud"}},
                "mcp": {"searxng": {"type": "local", "command": ["searxng"]}},
            }
        ).encode(),
        ".opencode/package.json": json.dumps(
            {"dependencies": {"@veedubin/neuralgentics": "^0.9.1"}}
        ).encode(),
        ".opencode/AGENTS.md": b"# project AGENTS.md\n",
        "docker-compose.yml": b"services: {}\n",
        "install.sh": b"#!/bin/sh\n",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_init_full_flow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "project"
    target.mkdir()
    files = _default_tarball_files()
    tarball, checksums = _build_tarball(tmp_path, version="0.9.1", files=files)
    _patch_download(monkeypatch, tarball=tarball, checksums=checksums)
    _patch_resolve_version(monkeypatch, version="0.9.1")
    _patch_opencode_on_path(monkeypatch)
    _patch_npm(monkeypatch)

    rc = init_cmd.run_init(_ns(target=target, version="latest"))
    assert rc == 0

    # File layout.
    assert (target / ".opencode/agents/coder.md").is_file()
    assert (target / ".opencode/agents/architect.md").is_file()
    assert (target / ".opencode/skills/boomerang-orchestrator/SKILL.md").is_file()
    assert (target / ".opencode/opencode.json").is_file()
    assert (target / ".opencode/package.json").is_file()
    assert (target / ".opencode/AGENTS.md").is_file()
    assert (target / "docker-compose.yml").is_file()
    assert (target / "install.sh").is_file()

    # opencode.json merged: plugin + provider present.
    cfg = json.loads((target / ".opencode/opencode.json").read_text(encoding="utf-8"))
    assert "@veedubin/neuralgentics" in cfg["plugin"]
    assert "AGENTS.md" in cfg["instructions"]

    # State file written.
    state = load_state(target)
    assert state is not None
    assert state.installed_version == "0.9.1"
    assert ".opencode/agents/coder.md" in state.files
    assert state.files[".opencode/opencode.json"].merged is True


def test_init_target_not_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("i'm a file", encoding="utf-8")
    _patch_opencode_on_path(monkeypatch)
    _patch_resolve_version(monkeypatch, version="0.9.1")

    with pytest.raises(TargetNotDirectory):
        init_cmd.run_init(_ns(target=not_a_dir))


def test_init_opencode_not_found(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "project"
    target.mkdir()
    monkeypatch.setattr(init_cmd.shutil, "which", lambda _cmd: None)
    _patch_resolve_version(monkeypatch, version="0.9.1")

    with pytest.raises(OpencodeNotFound):
        init_cmd.run_init(_ns(target=target))


def test_init_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "project"
    target.mkdir()
    _patch_opencode_on_path(monkeypatch)
    _patch_resolve_version(monkeypatch, version="0.9.1")

    rc = init_cmd.run_init(_ns(target=target, dry_run=True))
    assert rc == 0
    # No files written, no state file.
    assert not (target / ".opencode").exists()
    out = capsys.readouterr().out
    assert "WOULD" in out


def test_init_force_overwrites_existing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "project"
    (target / ".opencode/agents").mkdir(parents=True)
    (target / ".opencode/agents/coder.md").write_text("USER VERSION", encoding="utf-8")
    files = _default_tarball_files()
    tarball, checksums = _build_tarball(tmp_path, version="0.9.1", files=files)
    _patch_download(monkeypatch, tarball=tarball, checksums=checksums)
    _patch_resolve_version(monkeypatch, version="0.9.1")
    _patch_opencode_on_path(monkeypatch)
    _patch_npm(monkeypatch)

    rc = init_cmd.run_init(_ns(target=target, force=True))
    assert rc == 0
    # Tree entries are overwritten unconditionally; coder.md should be the shipped content.
    assert (target / ".opencode/agents/coder.md").read_text(
        encoding="utf-8"
    ) == "# coder agent\n"


def test_init_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "project"
    target.mkdir()
    files = _default_tarball_files()
    tarball, checksums = _build_tarball(tmp_path, version="0.9.1", files=files)
    _patch_download(monkeypatch, tarball=tarball, checksums=checksums)
    _patch_resolve_version(monkeypatch, version="0.9.1")
    _patch_opencode_on_path(monkeypatch)
    _patch_npm(monkeypatch)

    rc1 = init_cmd.run_init(_ns(target=target))
    assert rc1 == 0
    snapshot = _snapshot_tree(target)

    rc2 = init_cmd.run_init(_ns(target=target))
    assert rc2 == 0
    assert _snapshot_tree(target) == snapshot


def _snapshot_tree(root: Path) -> dict[str, bytes]:
    """Snapshot every file under ``root`` EXCEPT the state file.

    The state file carries fresh ``installed_at`` / ``updated_at`` timestamps
    on every run_init invocation, so comparing it across two runs is
    inherently non-idempotent. The placed plugin artifacts (agents, skills,
    merged config) are what must be byte-identical across runs.
    """
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != STATE_FILENAME
    }


def test_init_preserves_user_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "project"
    (target / ".opencode").mkdir(parents=True)
    user_cfg = {
        "$schema": "https://opencode.ai/config.json",
        "plugin": ["@franlol/opencode-md-table-formatter@latest"],
        "instructions": ["AGENTS.md"],
        "provider": {
            "ollama": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Local Ollama",
                "options": {"baseURL": "http://localhost:11434/v1"},
                "models": {"llama3:8b": {"name": "Llama 3 8B"}},
            }
        },
        "mcp": {"searxng": {"type": "local", "command": ["docker", "run"]}},
    }
    (target / ".opencode/opencode.json").write_text(
        json.dumps(user_cfg, indent=2), encoding="utf-8"
    )

    files = _default_tarball_files()
    tarball, checksums = _build_tarball(tmp_path, version="0.9.1", files=files)
    _patch_download(monkeypatch, tarball=tarball, checksums=checksums)
    _patch_resolve_version(monkeypatch, version="0.9.1")
    _patch_opencode_on_path(monkeypatch)
    _patch_npm(monkeypatch)

    rc = init_cmd.run_init(_ns(target=target))
    assert rc == 0

    merged = json.loads(
        (target / ".opencode/opencode.json").read_text(encoding="utf-8")
    )
    # User's provider preserved entirely.
    assert "ollama" in merged["provider"]
    assert (
        merged["provider"]["ollama"]["options"]["baseURL"]
        == "http://localhost:11434/v1"
    )
    # User's MCP server preserved.
    assert merged["mcp"]["searxng"]["command"] == ["docker", "run"]
    # Shipped plugin added.
    assert "@veedubin/neuralgentics" in merged["plugin"]


def test_init_merges_mcp_servers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "project"
    (target / ".opencode").mkdir(parents=True)
    user_cfg = {
        "plugin": [],
        "instructions": [],
        "mcp": {"foo": {"type": "local", "command": ["foo"]}},
    }
    (target / ".opencode/opencode.json").write_text(
        json.dumps(user_cfg), encoding="utf-8"
    )

    files = _default_tarball_files()
    tarball, checksums = _build_tarball(tmp_path, version="0.9.1", files=files)
    _patch_download(monkeypatch, tarball=tarball, checksums=checksums)
    _patch_resolve_version(monkeypatch, version="0.9.1")
    _patch_opencode_on_path(monkeypatch)
    _patch_npm(monkeypatch)

    rc = init_cmd.run_init(_ns(target=target))
    assert rc == 0

    merged = json.loads(
        (target / ".opencode/opencode.json").read_text(encoding="utf-8")
    )
    assert "foo" in merged["mcp"]
    assert "searxng" in merged["mcp"]


def test_init_state_file_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "project"
    target.mkdir()
    files = _default_tarball_files()
    tarball, checksums = _build_tarball(tmp_path, version="0.9.1", files=files)
    _patch_download(monkeypatch, tarball=tarball, checksums=checksums)
    _patch_resolve_version(monkeypatch, version="0.9.1")
    _patch_opencode_on_path(monkeypatch)
    _patch_npm(monkeypatch)

    init_cmd.run_init(_ns(target=target))
    assert (target / ".opencode" / STATE_FILENAME).is_file()


def test_init_offline_latest_conflict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "project"
    target.mkdir()
    _patch_opencode_on_path(monkeypatch)

    with pytest.raises(OfflineNoBundle):
        init_cmd.run_init(_ns(target=target, offline=True, version="latest"))


# ---------------------------------------------------------------------------
# v0.1.2 — Backup + mkdir + scary-target refusal (TestBackupAndMkdir)
# ---------------------------------------------------------------------------


class TestBackupAndMkdir:
    """Cover the backup-before-overwrite + auto-mkdir + scary-target refusal."""

    def _setup(self, monkeypatch, tmp_path, *, target: Path, force: bool = False):
        """Build a default tarball + patch downloads. Does NOT mkdir target."""
        files = _default_tarball_files()
        tarball, checksums = _build_tarball(tmp_path, version="0.9.1", files=files)
        _patch_download(monkeypatch, tarball=tarball, checksums=checksums)
        _patch_resolve_version(monkeypatch, version="0.9.1")
        _patch_opencode_on_path(monkeypatch)
        _patch_npm(monkeypatch)
        return _ns(target=target, force=force)

    def test_init_creates_target_if_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "nested" / "project"  # does not exist
        args = self._setup(monkeypatch, tmp_path, target=target)
        assert init_cmd.run_init(args) == 0
        assert target.is_dir()
        assert (target / ".opencode").is_dir()

    def test_init_refuses_home_dir_without_force(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Use a tmp_path as a stand-in for $HOME by patching expanduser.
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        monkeypatch.setattr(init_cmd.os.path, "expanduser", lambda _p: str(fake_home))
        target = fake_home  # target == $HOME
        args = self._setup(monkeypatch, tmp_path, target=target)
        with pytest.raises(TargetRefused):
            init_cmd.run_init(args)

    def test_init_refuses_root_without_force(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        target = Path("/")
        args = self._setup(monkeypatch, tmp_path, target=target)
        with pytest.raises(TargetRefused):
            init_cmd.run_init(args)

    def test_init_refuses_tmp_without_force(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        target = Path("/tmp")
        args = self._setup(monkeypatch, tmp_path, target=target)
        with pytest.raises(TargetRefused):
            init_cmd.run_init(args)

    def test_init_force_overrides_refusal(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # $HOME with --force should proceed.
        fake_home = tmp_path / "fakehome2"
        fake_home.mkdir()
        monkeypatch.setattr(init_cmd.os.path, "expanduser", lambda _p: str(fake_home))
        target = fake_home
        self._setup(monkeypatch, tmp_path, target=target, force=True)
        # Don't actually run the full init into our fake home — it'd pollute
        # tmp_path. Just verify _ensure_target doesn't raise with --force.
        init_cmd._ensure_target(target, force=True)

    def test_init_refuses_symlink_dotopencode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "project"
        target.mkdir()
        real_opencode = tmp_path / "real_opencode"
        real_opencode.mkdir()
        (target / ".opencode").symlink_to(real_opencode)
        args = self._setup(monkeypatch, tmp_path, target=target)
        with pytest.raises(TargetRefused):
            init_cmd.run_init(args)

    def test_init_force_overrides_symlink_refusal(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "project"
        target.mkdir()
        real_opencode = tmp_path / "real_opencode2"
        real_opencode.mkdir()
        (target / ".opencode").symlink_to(real_opencode)
        # Just verify _ensure_target doesn't raise with --force.
        init_cmd._ensure_target(target, force=True)

    def test_init_backs_up_existing_dotopencode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "project"
        (target / ".opencode" / "agents").mkdir(parents=True)
        # Pre-existing user opencode.json that differs from shipped → backup.
        (target / ".opencode" / "opencode.json").write_text(
            '{"plugin": [], "instructions": []}', encoding="utf-8"
        )
        args = self._setup(monkeypatch, tmp_path, target=target)
        assert init_cmd.run_init(args) == 0

        backups = list(target.glob(".opencode-bak-*"))
        assert len(backups) == 1, f"expected exactly one backup, got {backups}"
        # The user's old opencode.json should be preserved in the backup.
        assert (backups[0] / "opencode.json").is_file()
        assert (backups[0] / "opencode.json").read_text(
            encoding="utf-8"
        ) == '{"plugin": [], "instructions": []}'
        # New .opencode/ exists with merged content.
        assert (target / ".opencode" / "opencode.json").is_file()

    def test_init_no_backup_when_nothing_changes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "project"
        target.mkdir()
        args = self._setup(monkeypatch, tmp_path, target=target)
        # First init.
        assert init_cmd.run_init(args) == 0
        backups_after_first = list(target.glob(".opencode-bak-*"))
        assert backups_after_first == []
        # Second init (idempotent): should NOT create a backup.
        assert init_cmd.run_init(args) == 0
        backups_after_second = list(target.glob(".opencode-bak-*"))
        assert backups_after_second == []

    def test_init_creates_backup_with_unique_timestamp(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "project"
        (target / ".opencode" / "agents").mkdir(parents=True)
        (target / ".opencode" / "opencode.json").write_text(
            '{"plugin": []}', encoding="utf-8"
        )
        args = self._setup(monkeypatch, tmp_path, target=target)

        # Run init twice, changing the user's opencode.json between runs so
        # each run needs a fresh backup. Use distinct content to force "would_change".
        assert init_cmd.run_init(args) == 0
        first_backups = list(target.glob(".opencode-bak-*"))
        assert len(first_backups) == 1

        # Modify the user's (now-merged) opencode.json so the next init will
        # detect a change and back up again.
        (target / ".opencode" / "opencode.json").write_text(
            '{"plugin": [], "custom": "edit1"}', encoding="utf-8"
        )
        assert init_cmd.run_init(args) == 0
        second_backups = list(target.glob(".opencode-bak-*"))
        assert len(second_backups) == 2, f"expected 2 backups, got {second_backups}"
        # All backup dir names must be unique.
        names = {p.name for p in second_backups}
        assert len(names) == 2

    def test_init_dry_run_shows_would_backup(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        target = tmp_path / "project"
        (target / ".opencode").mkdir(parents=True)
        (target / ".opencode" / "opencode.json").write_text(
            '{"plugin": []}', encoding="utf-8"
        )
        args = self._setup(monkeypatch, tmp_path, target=target)
        args.dry_run = True
        assert init_cmd.run_init(args) == 0
        out = capsys.readouterr().out
        assert "WOULD back up" in out

    def test_init_state_records_backup_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "project"
        (target / ".opencode" / "agents").mkdir(parents=True)
        (target / ".opencode" / "opencode.json").write_text(
            '{"plugin": []}', encoding="utf-8"
        )
        args = self._setup(monkeypatch, tmp_path, target=target)
        assert init_cmd.run_init(args) == 0

        state = load_state(target)
        assert state is not None
        assert state.last_backup is not None
        # The recorded path should point at a dir that exists.
        backup_dir = target / state.last_backup
        assert backup_dir.is_dir()


# ---------------------------------------------------------------------------
# CLI dispatch — ``attacklm --init`` must route to neuralgentics_init, NOT
# to the attacklm-dataset init subcommand.
# ---------------------------------------------------------------------------


def test_attacklm_dash_dash_init_routes_to_neuralgentics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``attacklm --init`` must route to neuralgentics_init, NOT to attacklm-dataset.

    This is the critical regression test that ensures ``--init`` doesn't
    accidentally dispatch to the dataset init subcommand (``attacklm init``).
    """
    from attacklm import cli

    captured: dict[str, object] = {}

    def fake_run(args: argparse.Namespace) -> int:
        captured["called"] = True
        captured["target"] = args.target
        captured["version"] = args.version
        captured["dry_run"] = args.dry_run
        captured["force"] = args.force
        captured["offline"] = args.offline
        return 0

    # Monkeypatch the module-level function that ``_handle_neuralgentics_init``
    # imports via ``from attacklm.neuralgentics_init import run_neuralgentics_init``.
    import attacklm.neuralgentics_init as ng_init

    monkeypatch.setattr(ng_init, "run_neuralgentics_init", fake_run)

    # Simulate: ``attacklm --init --target <tmp_path> --dry-run --plugin-version 0.9.2``
    test_argv = [
        "attacklm",
        "--init",
        "--target",
        str(tmp_path),
        "--dry-run",
        "--plugin-version",
        "0.9.2",
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert captured.get("called") is True
    assert captured.get("target") == str(tmp_path)
    assert captured.get("version") == "0.9.2"
    # ``--dry-run`` was in sys.argv, so the handler should set dry_run=True.
    assert captured.get("dry_run") is True
    assert captured.get("force") is False
    assert captured.get("offline") is False


def test_attacklm_init_subcommand_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """The existing ``attacklm init`` SUBCOMMAND must still dispatch (not hijacked by --init).

    This guards against a regression where ``--init`` accidentally captures
    the ``init`` subcommand's argv. With no ``--init`` flag, the dataset
    init path must run. We stub ``_run_python_script`` AND ensure
    ``init_pipeline.py`` "exists" (by patching ``_SCRIPTS_DIR``) so the
    fallback path in ``_cmd_init`` is taken — without ``attacklm-dataset``
    installed, ``_cmd_init`` only reaches the script runner when the
    fallback script file is present.
    """
    from attacklm import cli

    captured: dict[str, object] = {}

    def fake_run(name: str, argv: list[str]) -> int:
        captured["name"] = name
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(cli, "_run_python_script", fake_run)
    # Make ``_SCRIPTS_DIR / init_pipeline.py`` appear to exist so _cmd_init's
    # fallback branch invokes _run_python_script instead of printing the
    # "dataset not found" message and returning 1.
    monkeypatch.setattr(
        cli.Path,
        "exists",
        lambda self: True if self.name == "init_pipeline.py" else Path.exists(self),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["attacklm", "init", "--", "--yes"],
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    # The dataset-init subcommand returns 0 (via fake_run) → main() exits 0.
    assert exc_info.value.code == 0
    # The init subcommand should have tried to run init_pipeline.py.
    assert captured.get("name") == "init_pipeline.py"
    assert "--yes" in captured.get("argv", [])
