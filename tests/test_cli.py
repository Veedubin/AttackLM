"""Tests for the attacklm unified CLI subcommand dispatch.

Verifies that each subcommand dispatches to the correct script
with the correct arguments, using monkeypatched runners to avoid
actually executing any scripts.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Make sure the attacklm package is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from attacklm import cli  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_leading_doubledash(args):
    """Strip the leading '--' that argparse REMAINDER may include.

    This mirrors what main() does before calling the handler.
    """
    if hasattr(args, "argv") and args.argv and args.argv[0] == "--":
        args.argv = args.argv[1:]


# ---------------------------------------------------------------------------
# Train subcommand tests
# ---------------------------------------------------------------------------


class TestTrainSubcommand:
    """Test that ``attacklm train`` dispatches to the correct script."""

    def test_train_default(self, monkeypatch):
        """attacklm train -- --epochs 5 → train_template.py"""
        captured = {}

        def fake_run(name, argv):
            captured["name"] = name
            captured["argv"] = list(argv)
            return 0

        monkeypatch.setattr(cli, "_run_python_script", fake_run)
        parser = cli.build_parser()
        args = parser.parse_args(["train", "--", "--epochs", "5"])
        _strip_leading_doubledash(args)
        rc = args.func(args)
        assert rc == 0
        assert captured["name"] == "train_template.py"
        assert "--epochs" in captured["argv"]
        assert "5" in captured["argv"]

    def test_train_all(self, monkeypatch):
        """attacklm train --all -- --single-model → train_all.py"""
        captured = {}

        def fake_run(name, argv):
            captured["name"] = name
            captured["argv"] = list(argv)
            return 0

        monkeypatch.setattr(cli, "_run_python_script", fake_run)
        parser = cli.build_parser()
        args = parser.parse_args(["train", "--all", "--", "--single-model"])
        _strip_leading_doubledash(args)
        rc = args.func(args)
        assert rc == 0
        assert captured["name"] == "train_all.py"
        assert "--single-model" in captured["argv"]

    def test_train_hpo(self, monkeypatch):
        """attacklm train --hpo -- --analyze-only → hpo_runner.py"""
        captured = {}

        def fake_run(name, argv):
            captured["name"] = name
            captured["argv"] = list(argv)
            return 0

        monkeypatch.setattr(cli, "_run_python_script", fake_run)
        parser = cli.build_parser()
        args = parser.parse_args(["train", "--hpo", "--", "--analyze-only"])
        _strip_leading_doubledash(args)
        rc = args.func(args)
        assert rc == 0
        assert captured["name"] == "hpo_runner.py"
        assert "--analyze-only" in captured["argv"]


# ---------------------------------------------------------------------------
# Init subcommand tests
# ---------------------------------------------------------------------------


class TestInitSubcommand:
    """Test that ``attacklm init`` dispatches correctly."""

    def test_init_default(self, monkeypatch):
        """attacklm init -- --yes → init_pipeline.py"""
        captured = {}

        def fake_run(name, argv):
            captured["name"] = name
            captured["argv"] = list(argv)
            return 0

        monkeypatch.setattr(cli, "_run_python_script", fake_run)
        parser = cli.build_parser()
        args = parser.parse_args(["init", "--", "--yes"])
        _strip_leading_doubledash(args)
        rc = args.func(args)
        assert rc == 0
        assert captured["name"] == "init_pipeline.py"
        assert "--yes" in captured["argv"]

    def test_init_from_source(self, monkeypatch):
        """attacklm init --from-source -- --yes → init_pipeline.py with --from-source appended."""
        captured = {}

        def fake_run(name, argv):
            captured["name"] = name
            captured["argv"] = list(argv)
            return 0

        monkeypatch.setattr(cli, "_run_python_script", fake_run)
        parser = cli.build_parser()
        args = parser.parse_args(["init", "--from-source", "--", "--yes"])
        _strip_leading_doubledash(args)
        rc = args.func(args)
        assert rc == 0
        assert captured["name"] == "init_pipeline.py"
        assert "--yes" in captured["argv"]
        assert "--from-source" in captured["argv"]

    def test_init_extract_only(self, monkeypatch):
        """attacklm init --extract-only → multiple _run_python_script calls."""
        calls = []

        def fake_run(name, argv):
            calls.append({"name": name, "argv": list(argv)})
            return 0

        monkeypatch.setattr(cli, "_run_python_script", fake_run)
        parser = cli.build_parser()
        args = parser.parse_args(["init", "--extract-only"])
        _strip_leading_doubledash(args)
        rc = args.func(args)
        assert rc == 0
        # Should have called all 12 extractors
        assert len(calls) == 12
        expected_extractors = [
            "extract_atomic_red_team_to_jsonl.py",
            "extract_caldera_plugins_to_jsonl.py",
            "parse_metasploit_to_jsonl.py",
            "extract_rta_to_jsonl.py",
            "extract_infection_monkey_to_jsonl.py",
            "extract_ai_tools_to_jsonl.py",
            "extract_sigma_defensive.py",
            "extract_mordor.py",
            "extract_threathunter_playbook.py",
            "extract_elastic_rules.py",
            "extract_splunk_content.py",
            "extract_nist_ir.py",
        ]
        actual_names = [c["name"] for c in calls]
        assert actual_names == expected_extractors

    def test_init_extract_only_stops_on_failure(self, monkeypatch):
        """If an extractor fails, --extract-only should stop and return non-zero."""
        calls = []

        def fake_run(name, argv):
            calls.append({"name": name, "argv": list(argv)})
            # Fail on the 3rd extractor
            if len(calls) == 3:
                return 1
            return 0

        monkeypatch.setattr(cli, "_run_python_script", fake_run)
        parser = cli.build_parser()
        args = parser.parse_args(["init", "--extract-only"])
        _strip_leading_doubledash(args)
        rc = args.func(args)
        assert rc == 1
        assert len(calls) == 3  # Stopped after the 3rd (failed) call

    def test_init_clone_only(self, monkeypatch):
        """attacklm init --clone-only → clone_repos.sh (shell script)."""
        captured = {}

        def fake_shell(name, argv):
            captured["name"] = name
            captured["argv"] = list(argv)
            return 0

        monkeypatch.setattr(cli, "_run_shell_script", fake_shell)
        parser = cli.build_parser()
        args = parser.parse_args(["init", "--clone-only"])
        _strip_leading_doubledash(args)
        rc = args.func(args)
        assert rc == 0
        assert captured["name"] == "clone_repos.sh"

    def test_init_buckets_only(self, monkeypatch):
        """attacklm init --buckets-only → setup_buckets.py + reorganize_buckets.py."""
        calls = []

        def fake_run(name, argv):
            calls.append({"name": name, "argv": list(argv)})
            return 0

        monkeypatch.setattr(cli, "_run_python_script", fake_run)
        parser = cli.build_parser()
        args = parser.parse_args(["init", "--buckets-only"])
        _strip_leading_doubledash(args)
        rc = args.func(args)
        assert rc == 0
        assert len(calls) == 2
        assert calls[0]["name"] == "setup_buckets.py"
        assert calls[1]["name"] == "reorganize_buckets.py"

    def test_init_dataset_url(self, monkeypatch):
        """attacklm init --dataset-url URL → init_pipeline.py with --dataset-url URL."""
        captured = {}

        def fake_run(name, argv):
            captured["name"] = name
            captured["argv"] = list(argv)
            return 0

        monkeypatch.setattr(cli, "_run_python_script", fake_run)
        parser = cli.build_parser()
        args = parser.parse_args(
            ["init", "--dataset-url", "https://example.com/ds.tar.gz"]
        )
        _strip_leading_doubledash(args)
        rc = args.func(args)
        assert rc == 0
        assert captured["name"] == "init_pipeline.py"
        assert "--dataset-url" in captured["argv"]
        assert "https://example.com/ds.tar.gz" in captured["argv"]


# ---------------------------------------------------------------------------
# Balance subcommand tests
# ---------------------------------------------------------------------------


class TestBalanceSubcommand:
    """Test that ``attacklm balance`` dispatches correctly."""

    def test_balance(self, monkeypatch):
        """attacklm balance -- --target-total 5000 → balance_buckets.py"""
        captured = {}

        def fake_run(name, argv):
            captured["name"] = name
            captured["argv"] = list(argv)
            return 0

        monkeypatch.setattr(cli, "_run_python_script", fake_run)
        parser = cli.build_parser()
        args = parser.parse_args(["balance", "--", "--target-total", "5000"])
        _strip_leading_doubledash(args)
        rc = args.func(args)
        assert rc == 0
        assert captured["name"] == "balance_buckets.py"
        assert "--target-total" in captured["argv"]
        assert "5000" in captured["argv"]


# ---------------------------------------------------------------------------
# Build subcommand tests
# ---------------------------------------------------------------------------


class TestBuildSubcommand:
    """Test that ``attacklm build`` dispatches correctly."""

    def test_build_default(self, monkeypatch):
        """attacklm build -- --adapter path → build.py"""
        captured = {}

        def fake_run(name, argv):
            captured["name"] = name
            captured["argv"] = list(argv)
            return 0

        monkeypatch.setattr(cli, "_run_python_script", fake_run)
        parser = cli.build_parser()
        args = parser.parse_args(["build", "--", "--adapter", "path"])
        _strip_leading_doubledash(args)
        rc = args.func(args)
        assert rc == 0
        assert captured["name"] == "build.py"
        assert "--adapter" in captured["argv"]
        assert "path" in captured["argv"]

    def test_build_merge_only(self, monkeypatch):
        """attacklm build --merge-only -- --adapter path → merge_adapter.py"""
        captured = {}

        def fake_run(name, argv):
            captured["name"] = name
            captured["argv"] = list(argv)
            return 0

        monkeypatch.setattr(cli, "_run_python_script", fake_run)
        parser = cli.build_parser()
        args = parser.parse_args(["build", "--merge-only", "--", "--adapter", "path"])
        _strip_leading_doubledash(args)
        rc = args.func(args)
        assert rc == 0
        assert captured["name"] == "merge_adapter.py"
        assert "--adapter" in captured["argv"]

    def test_build_gguf_only(self, monkeypatch):
        """attacklm build --gguf-only → convert_to_gguf.py"""
        captured = {}

        def fake_run(name, argv):
            captured["name"] = name
            captured["argv"] = list(argv)
            return 0

        monkeypatch.setattr(cli, "_run_python_script", fake_run)
        parser = cli.build_parser()
        args = parser.parse_args(["build", "--gguf-only"])
        _strip_leading_doubledash(args)
        rc = args.func(args)
        assert rc == 0
        assert captured["name"] == "convert_to_gguf.py"

    def test_build_register_ollama(self, monkeypatch):
        """attacklm build --register-ollama → register_ollama.py"""
        captured = {}

        def fake_run(name, argv):
            captured["name"] = name
            captured["argv"] = list(argv)
            return 0

        monkeypatch.setattr(cli, "_run_python_script", fake_run)
        parser = cli.build_parser()
        args = parser.parse_args(["build", "--register-ollama"])
        _strip_leading_doubledash(args)
        rc = args.func(args)
        assert rc == 0
        assert captured["name"] == "register_ollama.py"


# ---------------------------------------------------------------------------
# Infer subcommand tests
# ---------------------------------------------------------------------------


class TestInferSubcommand:
    """Test that ``attacklm infer`` dispatches correctly."""

    def test_infer(self, monkeypatch):
        """attacklm infer -- --model path → infer.py"""
        captured = {}

        def fake_run(name, argv):
            captured["name"] = name
            captured["argv"] = list(argv)
            return 0

        monkeypatch.setattr(cli, "_run_python_script", fake_run)
        parser = cli.build_parser()
        args = parser.parse_args(["infer", "--", "--model", "path"])
        _strip_leading_doubledash(args)
        rc = args.func(args)
        assert rc == 0
        assert captured["name"] == "infer.py"
        assert "--model" in captured["argv"]
        assert "path" in captured["argv"]


# ---------------------------------------------------------------------------
# Eval subcommand tests
# ---------------------------------------------------------------------------


class TestEvalSubcommand:
    """Test that ``attacklm eval`` dispatches to the correct eval script."""

    def test_eval_default(self, monkeypatch):
        """attacklm eval -- --base-model X → eval_retention.py"""
        captured = {}

        def fake_run(name, argv):
            captured["name"] = name
            captured["argv"] = list(argv)
            return 0

        monkeypatch.setattr(cli, "_run_python_script", fake_run)
        parser = cli.build_parser()
        args = parser.parse_args(["eval", "--", "--base-model", "X"])
        _strip_leading_doubledash(args)
        rc = args.func(args)
        assert rc == 0
        assert captured["name"] == "eval_retention.py"
        assert "--base-model" in captured["argv"]

    def test_eval_collect_ref(self, monkeypatch):
        """attacklm eval --collect-ref → collect_reference.py"""
        captured = {}

        def fake_run(name, argv):
            captured["name"] = name
            captured["argv"] = list(argv)
            return 0

        monkeypatch.setattr(cli, "_run_python_script", fake_run)
        parser = cli.build_parser()
        args = parser.parse_args(["eval", "--collect-ref"])
        _strip_leading_doubledash(args)
        rc = args.func(args)
        assert rc == 0
        assert captured["name"] == "collect_reference.py"

    def test_eval_score(self, monkeypatch):
        """attacklm eval --score → score_candidates.py"""
        captured = {}

        def fake_run(name, argv):
            captured["name"] = name
            captured["argv"] = list(argv)
            return 0

        monkeypatch.setattr(cli, "_run_python_script", fake_run)
        parser = cli.build_parser()
        args = parser.parse_args(["eval", "--score"])
        _strip_leading_doubledash(args)
        rc = args.func(args)
        assert rc == 0
        assert captured["name"] == "score_candidates.py"

    def test_eval_compare(self, monkeypatch):
        """attacklm eval --compare → compare_scores.py"""
        captured = {}

        def fake_run(name, argv):
            captured["name"] = name
            captured["argv"] = list(argv)
            return 0

        monkeypatch.setattr(cli, "_run_python_script", fake_run)
        parser = cli.build_parser()
        args = parser.parse_args(["eval", "--compare"])
        _strip_leading_doubledash(args)
        rc = args.func(args)
        assert rc == 0
        assert captured["name"] == "compare_scores.py"

    def test_eval_golden(self, monkeypatch):
        """attacklm eval --golden → golden_vectors.py"""
        captured = {}

        def fake_run(name, argv):
            captured["name"] = name
            captured["argv"] = list(argv)
            return 0

        monkeypatch.setattr(cli, "_run_python_script", fake_run)
        parser = cli.build_parser()
        args = parser.parse_args(["eval", "--golden"])
        _strip_leading_doubledash(args)
        rc = args.func(args)
        assert rc == 0
        assert captured["name"] == "golden_vectors.py"


# ---------------------------------------------------------------------------
# Demo subcommand tests
# ---------------------------------------------------------------------------


class TestDemoSubcommand:
    """Test that ``attacklm demo`` dispatches correctly."""

    def test_demo(self, monkeypatch):
        """attacklm demo → demo.py"""
        captured = {}

        def fake_run(name, argv):
            captured["name"] = name
            captured["argv"] = list(argv)
            return 0

        monkeypatch.setattr(cli, "_run_python_script", fake_run)
        parser = cli.build_parser()
        args = parser.parse_args(["demo"])
        _strip_leading_doubledash(args)
        rc = args.func(args)
        assert rc == 0
        assert captured["name"] == "demo.py"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestCLIEdgeCases:
    """Test version flag, no-args, and unknown subcommand handling."""

    def test_version_flag(self):
        """attacklm --version prints version and exits."""
        parser = cli.build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        # argparse exits with code 0 for --version
        assert exc_info.value.code == 0

    def test_no_subcommand(self, capsys):
        """attacklm (no args) should have command=None and print help on sys.exit(0)."""
        parser = cli.build_parser()
        args = parser.parse_args([])
        # No subcommand → command is None
        assert args.command is None
        # Simulate what main() does: print help and exit 0
        parser.print_help()
        captured = capsys.readouterr()
        assert "train" in captured.out
        assert "init" in captured.out
        assert "balance" in captured.out
        assert "build" in captured.out
        assert "eval" in captured.out
        assert "infer" in captured.out
        assert "demo" in captured.out

    def test_unknown_subcommand(self):
        """attacklm nonexistent → SystemExit (argparse error)."""
        parser = cli.build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["nonexistent"])
        # argparse exits with code 2 for unrecognized arguments
        assert exc_info.value.code == 2

    def test_return_code_propagated(self, monkeypatch):
        """Non-zero return codes from scripts should propagate through the handler."""

        def fake_run(name, argv):
            return 42

        monkeypatch.setattr(cli, "_run_python_script", fake_run)
        parser = cli.build_parser()
        args = parser.parse_args(["demo"])
        _strip_leading_doubledash(args)
        rc = args.func(args)
        assert rc == 42

    def test_mutually_exclusive_init_flags(self):
        """Mutually exclusive flags in init should cause SystemExit."""
        parser = cli.build_parser()
        # --extract-only and --clone-only are mutually exclusive
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["init", "--extract-only", "--clone-only"])
        assert exc_info.value.code == 2

    def test_mutually_exclusive_build_flags(self):
        """Mutually exclusive flags in build should cause SystemExit."""
        parser = cli.build_parser()
        # --merge-only and --gguf-only are mutually exclusive
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["build", "--merge-only", "--gguf-only"])
        assert exc_info.value.code == 2

    def test_mutually_exclusive_eval_flags(self):
        """Mutually exclusive flags in eval should cause SystemExit."""
        parser = cli.build_parser()
        # --collect-ref and --score are mutually exclusive
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["eval", "--collect-ref", "--score"])
        assert exc_info.value.code == 2

    def test_mutually_exclusive_init_source_flags(self):
        """Mutually exclusive --from-source and --dataset-url should cause SystemExit."""
        parser = cli.build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(
                ["init", "--from-source", "--dataset-url", "https://example.com"]
            )
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# GUI subcommand test (import error path)
# ---------------------------------------------------------------------------


class TestGUISubcommand:
    """Test that ``attacklm gui`` handles missing attacklm-gui gracefully."""

    def test_gui_import_error(self, monkeypatch):
        """attacklm gui with missing attacklm-gui should return 1."""
        # Ensure the import fails
        import importlib

        monkeypatch.setitem(sys.modules, "attacklm_gui", None)
        monkeypatch.setattr(
            importlib,
            "import_module",
            lambda _: (_ for _ in ()).throw(ImportError("no module")),
        )
        parser = cli.build_parser()
        args = parser.parse_args(["gui"])
        rc = args.func(args)
        assert rc == 1

    def test_gui_success(self, monkeypatch):
        """attacklm gui with attacklm-gui installed should return 0."""

        class FakeApp:
            def run(self):
                pass

        fake_module = type(sys)("attacklm_gui.app")
        fake_module.AttackLMApp = FakeApp
        monkeypatch.setitem(sys.modules, "attacklm_gui", fake_module)
        monkeypatch.setitem(sys.modules, "attacklm_gui.app", fake_module)

        parser = cli.build_parser()
        args = parser.parse_args(["gui"])
        rc = args.func(args)
        assert rc == 0
