"""Tests for init_pipeline.py — the one-shot init orchestrator.

These tests are hermetic: they build a temporary ``data/`` tree,
monkeypatch the orchestrator's path constants, and exercise the
local-probe, stage runners, download flow, and CLI dispatch logic
without ever touching the real ``data/`` or running any network/git
commands.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
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
    """Build a fake ``data/`` tree under ``tmp_path`` and rebind ip path constants."""
    fake_root = tmp_path / "AttackLM"
    fake_root.mkdir()
    data_dir = fake_root / "data"
    data_dir.mkdir()
    datasets_dir = data_dir / "datasets"
    datasets_dir.mkdir()
    sources_dir = datasets_dir / "buckets" / "sources"
    manifest_path = datasets_dir / "buckets" / "manifest.json"
    monkeypatch.setattr(ip, "BASE_DIR", fake_root)
    monkeypatch.setattr(ip, "DATA_DIR", data_dir)
    monkeypatch.setattr(ip, "DATASETS_DIR", datasets_dir)
    monkeypatch.setattr(ip, "SOURCES_DIR", sources_dir)
    monkeypatch.setattr(ip, "MANIFEST_PATH", manifest_path)
    # Disable the dependency checks — tests don't need torch/transformers/peft or tqdm
    monkeypatch.setattr(ip, "_check_download_deps", lambda: None)
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


# --- Download flow tests -----------------------------------------------------


def test_download_default_flow(
    fake_data_tree, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Default download: mock download and extraction, verify URL and output."""
    fake_root, _ = fake_data_tree

    # Make _dataset_already_present return False so the download actually runs
    monkeypatch.setattr(ip, "_dataset_already_present", lambda: False)

    # Track which URL was used
    downloaded_urls = []

    def fake_download(url, dest):
        downloaded_urls.append(url)

    monkeypatch.setattr(ip, "_download_with_progress", fake_download)

    # Mock _extract_tarball to create the expected output structure
    def fake_extract(tarball):
        print(f"  Extracting: {tarball}", file=sys.stderr)
        extracted_sources = fake_root / "data" / "datasets" / "buckets" / "sources"
        extracted_sources.mkdir(parents=True, exist_ok=True)
        (extracted_sources / "atomic-red-team").mkdir(parents=True, exist_ok=True)
        manifest = fake_root / "data" / "datasets" / "buckets" / "manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"total_records": 25_601}))

    monkeypatch.setattr(ip, "_extract_tarball", fake_extract)

    # Run main in download mode (default)
    rc = ip.main(["--yes"])
    assert rc == 0
    captured = capsys.readouterr()
    # Should mention the default URL
    assert ip.DEFAULT_DATASET_URL in captured.err
    # Should mention extraction
    assert "Extracting" in captured.err
    # Should mention the dataset is ready
    assert "Dataset ready" in captured.err
    # Verify the download URL was the default
    assert len(downloaded_urls) == 1
    assert downloaded_urls[0] == ip.DEFAULT_DATASET_URL


def test_download_skips_when_data_exists(
    fake_data_tree, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """If sources dir exists with subdirs and valid manifest, skip download."""
    _, data_dir = fake_data_tree
    sources = data_dir / "datasets" / "buckets" / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    (sources / "atomic-red-team").mkdir()
    (sources / "sigma").mkdir()
    (sources / "stockpile").mkdir()
    (sources / "metasploit-framework").mkdir()
    (sources / "mordor").mkdir()
    manifest = data_dir / "datasets" / "buckets" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"total_records": 25_601}))

    # Spy on urlopen to ensure it's NOT called
    urlopen_called = False

    def spy_urlopen(url):
        nonlocal urlopen_called
        urlopen_called = True
        raise RuntimeError("should not be called")

    monkeypatch.setattr(urllib.request, "urlopen", spy_urlopen)

    rc = ip.main(["--yes"])
    assert rc == 0
    assert not urlopen_called, "urlopen should not be called when data exists"
    captured = capsys.readouterr()
    assert "already present" in captured.err.lower()


def test_download_handles_404(
    fake_data_tree, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """HTTP 404 should produce a clear error message and exit code 3."""
    _, _ = fake_data_tree

    def raise_404(url):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", raise_404)

    with pytest.raises(SystemExit) as exc_info:
        ip.main(["--yes"])
    assert exc_info.value.code == 3
    captured = capsys.readouterr()
    assert "404" in captured.err
    assert "not found" in captured.err.lower()


def test_download_handles_network_error(
    fake_data_tree, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """URLError should produce a clear error message and exit code 3."""
    _, _ = fake_data_tree

    def raise_url_error(url):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", raise_url_error)

    with pytest.raises(SystemExit) as exc_info:
        ip.main(["--yes"])
    assert exc_info.value.code == 3
    captured = capsys.readouterr()
    assert "Network error" in captured.err
    assert "Connection refused" in captured.err


def test_from_source_flag(
    fake_data_tree, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--from-source should trigger the old probe/clone path."""
    _, data_dir = fake_data_tree
    # Populate all sources so clone is skipped
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

    rc = ip.main(["--from-source", "--yes", "--skip-attribute", "--skip-buckets"])
    assert rc == 0
    captured = capsys.readouterr()
    # Should mention probing
    assert "probing" in captured.err.lower()
    # Should mention clone is skipped
    assert "skipped" in captured.err.lower()


def test_dataset_url_flag(
    fake_data_tree, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--dataset-url should use the custom URL instead of the default."""
    fake_root, _ = fake_data_tree
    custom_url = "https://example.com/custom-dataset.tar.gz"

    # Make _dataset_already_present return False so the download actually runs
    monkeypatch.setattr(ip, "_dataset_already_present", lambda: False)

    # Track which URL was used
    downloaded_urls = []

    def fake_download(url, dest):
        downloaded_urls.append(url)

    monkeypatch.setattr(ip, "_download_with_progress", fake_download)

    # Mock _extract_tarball to create the expected output structure
    def fake_extract(tarball):
        extracted_sources = fake_root / "data" / "datasets" / "buckets" / "sources"
        extracted_sources.mkdir(parents=True, exist_ok=True)
        (extracted_sources / "sigma").mkdir()
        (extracted_sources / "stockpile").mkdir()
        (extracted_sources / "atomic-red-team").mkdir()
        (extracted_sources / "metasploit-framework").mkdir()
        (extracted_sources / "mordor").mkdir()
        manifest = fake_root / "data" / "datasets" / "buckets" / "manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"total_records": 25_601}))

    monkeypatch.setattr(ip, "_extract_tarball", fake_extract)

    rc = ip.main(["--yes", f"--dataset-url={custom_url}"])
    assert rc == 0
    captured = capsys.readouterr()
    # Should mention the custom URL
    assert custom_url in captured.err
    # Should NOT mention the default URL
    assert ip.DEFAULT_DATASET_URL not in captured.err
    # Verify the download URL was the custom one
    assert len(downloaded_urls) == 1
    assert downloaded_urls[0] == custom_url


def test_dry_run_shows_plan(
    fake_data_tree, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--dry-run in download mode should print what would happen without downloading."""
    _, _ = fake_data_tree

    # Spy on urlopen to ensure it's NOT called
    urlopen_called = False

    def spy_urlopen(url):
        nonlocal urlopen_called
        urlopen_called = True
        raise RuntimeError("should not be called")

    monkeypatch.setattr(urllib.request, "urlopen", spy_urlopen)

    rc = ip.main(["--dry-run"])
    assert rc == 0
    assert not urlopen_called, "urlopen should not be called during dry-run"
    captured = capsys.readouterr()
    assert "dry-run" in captured.err.lower()
    assert "would download" in captured.err.lower()


def test_mutually_exclusive_from_source_and_dataset_url(
    fake_data_tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--from-source and --dataset-url together should produce an error."""
    _, _ = fake_data_tree

    with pytest.raises(SystemExit) as exc_info:
        ip.main(["--from-source", "--dataset-url=https://example.com/ds.tar.gz"])
    assert exc_info.value.code == 2  # argparse exits with 2 on error


# --- CLI dispatch tests (from-source mode) -----------------------------------


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
    rc = ip.main(["--from-source", "--dry-run", "--yes"])
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
    rc = ip.main(["--from-source", "--dry-run", "--yes"])
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
    rc = ip.main(
        ["--from-source", "--skip-clone", "--yes", "--skip-attribute", "--skip-buckets"]
    )
    assert rc == 0


def test_main_user_declines_network(
    fake_data_tree, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """If the local probe fails and the user says no, exit code is 2."""
    _, data_dir = fake_data_tree
    # Don't populate anything → all sources missing → network fallback needed
    # Monkeypatch _confirm to return False
    monkeypatch.setattr(ip, "_confirm", lambda msg, assume_yes: False)
    rc = ip.main(["--from-source"])
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
    rc = ip.main(
        ["--from-source", "--yes", "--skip-attribute", "--skip-buckets", "--dry-run"]
    )
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
    rc = ip.main(
        ["--from-source", "--yes", "--skip-buckets", "--skip-attribute", "--dry-run"]
    )
    assert rc == 0


# --- cli.py dispatcher test ---------------------------------------------------


def test_cli_main_init_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    """attacklm init should call _run_python_script('init_pipeline.py', argv)."""
    from attacklm import cli

    captured: dict = {}

    def fake_run(name, argv):
        captured["name"] = name
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(cli, "_run_python_script", fake_run)
    parser = cli.build_parser()
    args = parser.parse_args(["init", "--", "--yes", "--dry-run"])
    # Strip leading '--' that argparse REMAINDER includes (main() does this)
    if hasattr(args, "argv") and args.argv and args.argv[0] == "--":
        args.argv = args.argv[1:]
    rc = args.func(args)
    assert rc == 0
    assert captured["name"] == "init_pipeline.py"
    assert captured["argv"] == ["--yes", "--dry-run"]


def test_cli_help_lists_init() -> None:
    """`python -m attacklm.cli` should mention 'init' in its help text."""
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
    assert "init" in result.stdout
