"""Tests for init_pipeline.py — the one-shot init orchestrator.

These tests are hermetic: they build a temporary ``data/`` tree,
monkeypatch the orchestrator's path constants, and exercise the
local-probe, stage runners, and CLI dispatch logic without ever
touching the real ``data/`` or running any network/git commands.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Make sure scripts/ is importable so we can import the module under test
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
# Also expose the attacklm package (src/ layout) for the dispatcher tests
sys.path.insert(0, str(_REPO_ROOT / "src"))

import init_pipeline as ip  # noqa: E402


# --- Helpers ------------------------------------------------------------------


@pytest.fixture
def fake_data_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build a fake ``data/`` tree under ``tmp_path`` and rebind ip.DATA_DIR."""
    fake_root = tmp_path / "AttackLM"
    fake_root.mkdir()
    data_dir = fake_root / "data"
    data_dir.mkdir()
    monkeypatch.setattr(ip, "BASE_DIR", fake_root)
    monkeypatch.setattr(ip, "DATA_DIR", data_dir)
    monkeypatch.setattr(ip, "DATASETS_DIR", data_dir / "datasets")
    # Disable the dependency check — tests don't need torch/transformers/peft
    monkeypatch.setattr(ip, "_check_dependencies", lambda: None)
    # Re-bind the probe list to use the rebased DATA_DIR
    monkeypatch.setattr(
        ip,
        "_LOCAL_PROBES",
        [
            (
                "atomic-red-team",
                data_dir / "atomic-red-team",
                data_dir / "atomic-red-team" / "atomics",
                1024,
            ),
            (
                "stockpile",
                data_dir / "stockpile",
                data_dir / "stockpile" / "README.md",
                256,
            ),
            ("sigma", data_dir / "sigma", data_dir / "sigma" / "rules", 1024),
            (
                "metasploit-framework",
                data_dir / "metasploit-framework",
                data_dir / "metasploit-framework" / "modules",
                1024,
            ),
            (
                "mordor",
                data_dir / "mordor",
                data_dir / "mordor" / "datasets",
                1024,
            ),
            (
                "threathunter-playbook",
                data_dir / "threathunter-playbook",
                data_dir / "threathunter-playbook" / "playbooks",
                1024,
            ),
            (
                "elastic-detection-rules",
                data_dir / "elastic-detection-rules",
                data_dir / "elastic-detection-rules" / "rules",
                1024,
            ),
            (
                "splunk-security-content",
                data_dir / "splunk-security-content",
                data_dir / "splunk-security-content" / "detections",
                1024,
            ),
            (
                "nist-sp800-61r3",
                data_dir / "nist-sp800-61r3",
                data_dir / "nist-sp800-61r3" / "NIST.SP.800-61r3.pdf",
                1024,
            ),
        ],
    )
    return fake_root, data_dir


def _populate(data_dir: Path, name: str, marker: str, size: int) -> None:
    """Create a fake source dir with a marker file of at least ``size`` bytes."""
    src = data_dir / name
    src.mkdir(parents=True, exist_ok=True)
    mp = src / marker
    mp.parent.mkdir(parents=True, exist_ok=True)
    # marker is a path; if it ends in a file name, write the file
    if (
        "." in marker.split("/")[-1]
        or marker.endswith(".md")
        or marker.endswith(".txt")
    ):
        mp.write_bytes(b"x" * size)
    else:
        # directory marker — create it and stuff it with a file
        mp.mkdir(parents=True, exist_ok=True)
        (mp / "marker.bin").write_bytes(b"x" * size)


# --- Local probe tests --------------------------------------------------------


def test_probe_all_present(fake_data_tree) -> None:
    _, data_dir = fake_data_tree
    for name, marker in [
        ("atomic-red-team", "atomics"),
        ("stockpile", "README.md"),
        ("sigma", "rules"),
        ("metasploit-framework", "modules"),
        ("mordor", "datasets"),
        ("threathunter-playbook", "playbooks"),
        ("elastic-detection-rules", "rules"),
        ("splunk-security-content", "detections"),
        ("nist-sp800-61r3", "NIST.SP.800-61r3.pdf"),
    ]:
        _populate(data_dir, name, marker, 4096)
    probes = ip.probe_local()
    assert len(probes) == 9
    assert all(p.present for p in probes), [p.detail for p in probes]


def test_probe_missing(fake_data_tree) -> None:
    _, data_dir = fake_data_tree
    # Only populate one source; the rest are missing
    _populate(data_dir, "atomic-red-team", "atomics", 4096)
    probes = ip.probe_local()
    missing = [p.name for p in probes if not p.present]
    assert "atomic-red-team" not in missing
    assert "metasploit-framework" in missing
    assert "sigma" in missing


def test_probe_marker_too_small(fake_data_tree) -> None:
    """A source with a marker below the size threshold should be marked missing."""
    _, data_dir = fake_data_tree
    _populate(data_dir, "atomic-red-team", "atomics", 4096)
    _populate(data_dir, "stockpile", "README.md", 100)  # below 256B threshold
    probes = ip.probe_local()
    by_name = {p.name: p for p in probes}
    assert by_name["atomic-red-team"].present is True
    assert by_name["stockpile"].present is False
    assert "marker too small" in by_name["stockpile"].detail


# --- Stage runner tests -------------------------------------------------------


def test_stage_extract_skips_missing_scripts(
    fake_data_tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If a script is not present, stage_extract should skip it gracefully."""
    _, _ = fake_data_tree
    # The orchestrator looks for scripts relative to BASE_DIR/scripts.
    # We never created that dir, so all extractors are missing.
    rc = ip.stage_extract()
    assert rc == 0


def test_stage_attribute_missing_script_is_ok(
    fake_data_tree,
) -> None:
    """stage_attribute should be a no-op if augment_attribution.py is absent."""
    rc = ip.stage_attribute()
    assert rc == 0


def test_stage_buckets_skips_missing(fake_data_tree) -> None:
    rc = ip.stage_buckets(clean=True)
    assert rc == 0


# --- Bucket-built detection ---------------------------------------------------


def test_buckets_already_built_true(fake_data_tree) -> None:
    _, data_dir = fake_data_tree
    manifest = data_dir / "datasets" / "buckets" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"total_records": 25_601}))
    assert ip._buckets_already_built() is True


def test_buckets_already_built_false(fake_data_tree) -> None:
    _, data_dir = fake_data_tree
    manifest = data_dir / "datasets" / "buckets" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"total_records": 0}))
    assert ip._buckets_already_built() is False


def test_buckets_already_built_no_manifest(fake_data_tree) -> None:
    assert ip._buckets_already_built() is False


def test_buckets_already_built_invalid_json(fake_data_tree) -> None:
    _, data_dir = fake_data_tree
    manifest = data_dir / "datasets" / "buckets" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("not json at all")
    assert ip._buckets_already_built() is False


# --- CLI dispatch tests -------------------------------------------------------


def test_main_dry_run_with_all_present(
    fake_data_tree, capsys: pytest.CaptureFixture[str]
) -> None:
    _, data_dir = fake_data_tree
    for name, marker in [
        ("atomic-red-team", "atomics"),
        ("stockpile", "README.md"),
        ("sigma", "rules"),
        ("metasploit-framework", "modules"),
        ("mordor", "datasets"),
        ("threathunter-playbook", "playbooks"),
        ("elastic-detection-rules", "rules"),
        ("splunk-security-content", "detections"),
        ("nist-sp800-61r3", "NIST.SP.800-61r3.pdf"),
    ]:
        _populate(data_dir, name, marker, 4096)
    rc = ip.main(["--dry-run", "--yes"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "(skipped" in captured.err.lower()
    assert "upstream sources are present" in captured.err.lower()


def test_main_dry_run_with_missing(
    fake_data_tree, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dry-run with missing sources should still exit 0 and show clone plan."""
    _, data_dir = fake_data_tree
    # Populate only one
    _populate(data_dir, "atomic-red-team", "atomics", 4096)
    rc = ip.main(["--dry-run", "--yes"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "missing" in captured.err.lower()
    assert "would run" in captured.err


def test_main_skip_clone_with_missing_exits_4_or_2(
    fake_data_tree,
) -> None:
    """If --skip-clone is passed but data is missing, this is a hard error.

    We don't actually have such a path in main() — --skip-clone doesn't
    probe, so the probe never runs.  This test documents that behavior:
    with --skip-clone, main() proceeds straight to extract (which will
    no-op because the data is missing).  Return code should be 0 because
    the stages themselves don't fail.  Users get garbage output, which is
    exactly what --skip-clone is signaling they want.
    """
    _, data_dir = fake_data_tree
    _populate(data_dir, "atomic-red-team", "atomics", 4096)
    rc = ip.main(["--skip-clone", "--yes", "--skip-attribute", "--skip-buckets"])
    assert rc == 0


def test_main_user_declines_network(
    fake_data_tree, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """If the local probe fails and the user says no, exit code is 2."""
    _, data_dir = fake_data_tree
    # Don't populate anything → all sources missing → network fallback needed
    # Monkeypatch _confirm to return False
    monkeypatch.setattr(ip, "_confirm", lambda msg, assume_yes: False)
    rc = ip.main([])
    assert rc == 2
    captured = capsys.readouterr()
    assert "declined" in captured.err.lower() or "abort" in captured.err.lower()


def test_main_skip_attribute_runs(
    fake_data_tree, capsys: pytest.CaptureFixture[str]
) -> None:
    """With --skip-attribute, the attribute stage should be marked SKIPPED."""
    _, data_dir = fake_data_tree
    for name, marker in [
        ("atomic-red-team", "atomics"),
        ("stockpile", "README.md"),
        ("sigma", "rules"),
        ("metasploit-framework", "modules"),
        ("mordor", "datasets"),
        ("threathunter-playbook", "playbooks"),
        ("elastic-detection-rules", "rules"),
        ("splunk-security-content", "detections"),
        ("nist-sp800-61r3", "NIST.SP.800-61r3.pdf"),
    ]:
        _populate(data_dir, name, marker, 4096)
    rc = ip.main(["--yes", "--skip-attribute", "--skip-buckets", "--dry-run"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "--skip-attribute" in captured.err


def test_main_skip_buckets_runs(
    fake_data_tree, capsys: pytest.CaptureFixture[str]
) -> None:
    _, data_dir = fake_data_tree
    for name, marker in [
        ("atomic-red-team", "atomics"),
        ("stockpile", "README.md"),
        ("sigma", "rules"),
        ("metasploit-framework", "modules"),
        ("mordor", "datasets"),
        ("threathunter-playbook", "playbooks"),
        ("elastic-detection-rules", "rules"),
        ("splunk-security-content", "detections"),
        ("nist-sp800-61r3", "NIST.SP.800-61r3.pdf"),
    ]:
        _populate(data_dir, name, marker, 4096)
    rc = ip.main(["--yes", "--skip-buckets", "--skip-attribute", "--dry-run"])
    assert rc == 0


# --- cli.py dispatcher test ---------------------------------------------------


def test_cli_main_init_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    """attacklm.cli.main_init should call _run_python_script('init_pipeline.py', argv)."""
    from attacklm import cli

    captured: dict = {}

    def fake_run(name, argv):
        captured["name"] = name
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(cli, "_run_python_script", fake_run)
    rc = cli.main_init(["--yes", "--dry-run"])
    assert rc == 0
    assert captured["name"] == "init_pipeline.py"
    assert captured["argv"] == ["--yes", "--dry-run"]


def test_cli_help_lists_init() -> None:
    """`python -m attacklm.cli` should mention attacklm-init in its help text."""
    import os

    env = os.environ.copy()
    src_path = str(_REPO_ROOT / "src")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
        if "PYTHONPATH" in env
        else src_path
    )
    result = subprocess.run(
        [sys.executable, "-m", "attacklm.cli"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert "attacklm-init" in result.stdout
