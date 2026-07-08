"""AttackLM unified CLI with argparse subcommands.

Usage::

    attacklm --init [--target DIR] [--plugin-version VER] [--offline] [--dry-run]
    attacklm train [--dataset all] [--hpo] [args...]
    attacklm init [--extract-only|--buckets-only|--clone-only] [args...]
    attacklm balance [args...]
    attacklm steer [--extract|--apply|--sweep|--diagnose] [args...]
    attacklm bench [args...]
    attacklm pipeline [args...]
    attacklm build [--merge-only|--gguf-only|--register-ollama] [args...]
    attacklm infer [args...]
    attacklm eval [--collect-ref|--score|--compare|--golden] [args...]
    attacklm gui
    attacklm demo [args...]
"""

from __future__ import annotations

import argparse
import os
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Sequence

# Locate the scripts/ directory.
# When installed from PyPI:  site-packages/attacklm/scripts/
# When running from source:  src/attacklm/../../scripts/
_SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if not _SCRIPTS_DIR.exists():
    # Fallback: running from source (scripts/ is at repo root)
    _SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


# ---------------------------------------------------------------------------
# Script runner
# ---------------------------------------------------------------------------


def _run_python_script(script_name: str, argv: Sequence[str]) -> int:
    """Run a Python script from scripts/ as if it were the entry point."""
    script_path = _SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"attacklm: script not found: {script_path}", file=sys.stderr)
        return 127
    sys.argv = [str(script_path), *argv]
    try:
        runpy.run_path(str(script_path), run_name="__main__")
        return 0
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0
    except Exception as e:  # noqa: BLE001
        print(f"attacklm: error running {script_name}: {e}", file=sys.stderr)
        return 1


def _run_shell_script(script_name: str, argv: Sequence[str]) -> int:
    """Run a shell script from scripts/ as if it were the entry point."""
    import subprocess

    script_path = _SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"attacklm: script not found: {script_path}", file=sys.stderr)
        return 127
    result = subprocess.run(["bash", str(script_path), *argv])
    return result.returncode


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_train(args: argparse.Namespace) -> int:
    """Handle ``attacklm train`` — dispatch to the right training script."""
    if args.hpo:
        return _run_python_script("hpo_runner.py", args.argv)
    if args.all:
        return _run_python_script("train_all.py", args.argv)
    # Default: single LoRA/QLoRA training
    return _run_python_script("train_template.py", args.argv)


def _cmd_init(args: argparse.Namespace) -> int:
    """Handle ``attacklm init`` — delegate to attacklm-dataset or download directly."""
    # Build the argv forwarded to attacklm-dataset. argparse consumes the
    # init-specific flags (--from-source, --dataset-url, --extract-only,
    # --buckets-only, --clone-only) so we re-append the ones that are
    # meaningful to attacklm-dataset's CLI. They are mutually exclusive at
    # the argparse level, so only one of the three partial-step flags can
    # be set at a time. The pre-v0.11.0 behavior of running 12 extractors
    # in-process is gone; attacklm-dataset now owns that orchestration.
    forwarded = list(args.argv)
    if getattr(args, "from_source", False):
        forwarded.append("--from-source")
    if getattr(args, "dataset_url", None):
        forwarded.extend(["--dataset-url", args.dataset_url])
    if getattr(args, "extract_only", False):
        forwarded.append("--extract-only")
    if getattr(args, "buckets_only", False):
        forwarded.append("--buckets-only")
    if getattr(args, "clone_only", False):
        forwarded.append("--clone-only")

    # Try attacklm-dataset CLI first
    try:
        import attacklm_dataset  # noqa: F401  — presence probe only

        # attacklm-dataset is installed — delegate to its CLI
        result = subprocess.run(
            [sys.executable, "-m", "attacklm_dataset.cli", "init", *forwarded],
            cwd=Path.cwd(),
        )
        return result.returncode
    except ImportError:
        pass

    # Fallback: try local init_pipeline.py if it exists
    init_script = _SCRIPTS_DIR / "init_pipeline.py"
    if init_script.exists():
        return _run_python_script("init_pipeline.py", forwarded)

    # Neither installed nor local — guide the user
    print(
        "AttackLM dataset not found.\n"
        "Install it with:  pip install attacklm-dataset\n"
        "Or clone it from: https://github.com/Veedubin/attacklm-dataset",
        file=sys.stderr,
    )
    return 1


def _cmd_balance(args: argparse.Namespace) -> int:
    """Handle ``attacklm balance`` — delegate to attacklm-dataset."""
    try:
        import attacklm_dataset  # noqa: F401  — presence probe only

        result = subprocess.run(
            [sys.executable, "-m", "attacklm_dataset.cli", "balance"] + list(args.argv),
            cwd=Path.cwd(),
        )
        return result.returncode
    except ImportError:
        pass

    # Fallback: try local balance_buckets.py
    balance_script = _SCRIPTS_DIR / "balance_buckets.py"
    if balance_script.exists():
        return _run_python_script("balance_buckets.py", args.argv)

    print(
        "AttackLM dataset not found.\n"
        "Install it with:  pip install attacklm-dataset\n"
        "Or clone it from: https://github.com/Veedubin/attacklm-dataset",
        file=sys.stderr,
    )
    return 1


def _cmd_steer(args: argparse.Namespace) -> int:
    """Handle ``attacklm steer`` — dispatch to activation steering script."""
    forwarded = list(args.argv)
    if args.extract:
        forwarded.append("--extract")
    if args.apply:
        forwarded.append("--apply")
    if args.sweep:
        forwarded.append("--sweep")
    if args.diagnose:
        forwarded.append("--diagnose")
    return _run_python_script("steering.py", forwarded)


def _cmd_bench(args: argparse.Namespace) -> int:
    """Handle ``attacklm bench`` — dispatch to domain benchmark script."""
    return _run_python_script("domain_bench.py", args.argv)


def _cmd_pipeline(args: argparse.Namespace) -> int:
    """Handle ``attacklm pipeline`` — dispatch to pipeline script."""
    return _run_python_script("pipeline.py", args.argv)


def _cmd_build(args: argparse.Namespace) -> int:
    """Handle ``attacklm build`` — dispatch to merge, gguf, ollama, or full build."""
    if args.merge_only:
        return _run_python_script("merge_adapter.py", args.argv)
    if args.gguf_only:
        return _run_python_script("convert_to_gguf.py", args.argv)
    if args.register_ollama:
        return _run_python_script("register_ollama.py", args.argv)
    # Default: full build pipeline
    return _run_python_script("build.py", args.argv)


def _cmd_infer(args: argparse.Namespace) -> int:
    """Handle ``attacklm infer``."""
    return _run_python_script("infer.py", args.argv)


def _cmd_eval(args: argparse.Namespace) -> int:
    """Handle ``attacklm eval`` — dispatch to eval subcommands."""
    if args.collect_ref:
        return _run_python_script("collect_reference.py", args.argv)
    if args.score:
        return _run_python_script("score_candidates.py", args.argv)
    if args.compare:
        return _run_python_script("compare_scores.py", args.argv)
    if args.golden:
        return _run_python_script("golden_vectors.py", args.argv)
    # Default: retention evaluation
    return _run_python_script("eval_retention.py", args.argv)


def _cmd_gui(_args: argparse.Namespace) -> int:
    """Handle ``attacklm gui`` — launch the TUI."""
    try:
        from attacklm_gui.app import AttackLMApp  # type: ignore[import-untyped]

        app = AttackLMApp()
        app.run()
        return 0
    except ImportError:
        print(
            "attacklm gui: attacklm-gui is not installed.\n"
            "  Install with: pip install attacklm-gui",
            file=sys.stderr,
        )
        return 1


def _cmd_demo(args: argparse.Namespace) -> int:
    """Handle ``attacklm demo``."""
    return _run_python_script("demo.py", args.argv)


def _cmd_audit(args: argparse.Namespace) -> int:
    """Handle ``attacklm audit`` — delegate to attacklm-dataset."""
    # Try attacklm-dataset CLI first
    try:
        import attacklm_dataset  # noqa: F401  — presence probe only

        # Build the argv for attacklm-dataset's inversion_audit
        forwarded = [
            "--attack",
            args.attack,
            "--model",
            args.model,
            "--dataset-root",
            args.dataset_root,
        ]
        if args.mia_method:
            forwarded.extend(["--mia-method", args.mia_method])
        if args.mia_threshold_mode:
            forwarded.extend(["--mia-threshold-mode", args.mia_threshold_mode])
        if args.mia_percentile:
            forwarded.extend(["--mia-percentile", str(args.mia_percentile)])
        if args.source_filter:
            forwarded.extend(["--source-filter", *args.source_filter])
        if args.top_k:
            forwarded.extend(["--top-k", str(args.top_k)])
        if args.max_new_tokens:
            forwarded.extend(["--max-new-tokens", str(args.max_new_tokens)])
        if args.temperature:
            forwarded.extend(["--temperature", str(args.temperature)])
        if args.max_records:
            forwarded.extend(["--max-records", str(args.max_records)])
        if args.dry_run:
            forwarded.append("--dry-run")

        result = subprocess.run(
            [sys.executable, "-m", "attacklm_dataset.cli", "audit", *forwarded],
            cwd=Path.cwd(),
        )
        return result.returncode
    except ImportError:
        pass

    # Fallback: try local inversion_audit.py
    audit_script = _SCRIPTS_DIR / "inversion_audit.py"
    if audit_script.exists():
        forwarded = [
            "--attack",
            args.attack,
            "--model",
            args.model,
            "--dataset-root",
            args.dataset_root,
        ]
        if args.mia_method:
            forwarded.extend(["--mia-method", args.mia_method])
        if args.mia_threshold_mode:
            forwarded.extend(["--mia-threshold-mode", args.mia_threshold_mode])
        if args.mia_percentile:
            forwarded.extend(["--mia-percentile", str(args.mia_percentile)])
        if args.source_filter:
            forwarded.extend(["--source-filter", *args.source_filter])
        if args.top_k:
            forwarded.extend(["--top-k", str(args.top_k)])
        if args.max_new_tokens:
            forwarded.extend(["--max-new-tokens", str(args.max_new_tokens)])
        if args.temperature:
            forwarded.extend(["--temperature", str(args.temperature)])
        if args.max_records:
            forwarded.extend(["--max-records", str(args.max_records)])
        if args.dry_run:
            forwarded.append("--dry-run")
        return _run_python_script("inversion_audit.py", forwarded)

    print(
        "AttackLM dataset not found.\n"
        "Install it with:  pip install attacklm-dataset\n"
        "Or clone it from: https://github.com/Veedubin/attacklm-dataset",
        file=sys.stderr,
    )
    return 1


# ---------------------------------------------------------------------------
# Top-level --init handler (neuralgentics OpenCode plugin bootstrap)
# ---------------------------------------------------------------------------


def _handle_neuralgentics_init(args: argparse.Namespace) -> None:
    """Dispatch ``attacklm --init`` to the neuralgentics_init package.

    Lazy-imports the subpackage so that dataset-only users (the majority
    case) don't pay the import cost. Catches the package's typed errors
    and formats them with remediation hints.

    ``--init`` is a TOP-LEVEL flag, NOT a subcommand. It routes to a
    completely different code path (the OpenCode plugin bootstrap) than
    the ``attacklm init`` SUBCOMMAND (the training-dataset init flow via
    attacklm-dataset). They are different operations and intentionally
    separate.
    """
    try:
        from attacklm.neuralgentics_init import (
            NeuralgenticsError,
            format_neuralgentics_error,
            run_neuralgentics_init,
        )
    except ImportError as exc:  # pragma: no cover — defensive
        print(
            f"attacklm --init: neuralgentics_init package not importable: {exc}\n"
            f"This is a packaging bug. Reinstall attacklm or file an issue.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build the argparse.Namespace ``_run_init`` expects:
    # target, version, repo, force, dry_run, offline.
    init_args = argparse.Namespace(
        target=getattr(args, "target", None) or os.getcwd(),
        version=getattr(args, "plugin_version", "latest"),
        repo=getattr(args, "repo", "Veedubin/neuralgentics"),
        force=bool(getattr(args, "force", False)),
        dry_run=bool(getattr(args, "dry_run", False)),
        offline=bool(getattr(args, "offline", False)),
    )

    try:
        rc = run_neuralgentics_init(init_args)
    except NeuralgenticsError as err:
        print(format_neuralgentics_error(err), file=sys.stderr)
        sys.exit(getattr(err, "exit_code", 1))
    except KeyboardInterrupt:
        print("\n[interrupted]", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001 — last-resort handler
        print(f"[ERROR] unexpected failure: {exc}", file=sys.stderr)
        sys.exit(1)
    sys.exit(rc)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="attacklm",
        description="AttackLM — QLoRA-fine-tuned security & red-team LLM toolkit",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )

    # ---- top-level --init flag (neuralgentics OpenCode plugin bootstrap) ----
    # NOTE: this is a top-level flag, NOT a subcommand. It is checked in
    # main() and routed to the neuralgentics_init package. The existing
    # ``attacklm init`` SUBCOMMAND below continues to handle the dataset
    # init flow via attacklm-dataset — they are different operations.
    parser.add_argument(
        "--init",
        action="store_true",
        default=False,
        help=(
            "Initialize the neuralgentics OpenCode plugin in the target "
            "directory. Downloads the plugin tarball, deep-merges your "
            "opencode.json, runs npm install, and writes a state file. "
            "Combine with --target, --plugin-version, --dry-run, --force, "
            "--offline. (This is a TOP-LEVEL flag, not a subcommand. For "
            "the dataset init flow, use: attacklm init ...)"
        ),
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help=(
            "Target directory for --init (default: current working directory). "
            "Used only with --init; ignored otherwise."
        ),
    )
    parser.add_argument(
        "--plugin-version",
        type=str,
        default="latest",
        help=(
            "Plugin version to install with --init (default: 'latest'). "
            "Used only with --init; ignored otherwise. (Note: the top-level "
            "--version flag prints the attacklm CLI version and exits; this "
            "flag controls which neuralgentics release --init downloads.)"
        ),
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="Veedubin/neuralgentics",
        help=(
            "GitHub owner/repo for --init (default: Veedubin/neuralgentics). "
            "Used only with --init; ignored otherwise."
        ),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help=(
            "Run --init in offline mode (requires bundled tarball; "
            "currently raises an error since no bundle ships yet). "
            "Used only with --init; ignored otherwise."
        ),
    )
    # --dry-run / --force are declared at the top level so argparse does not
    # reject them when used with --init (e.g. ``attacklm --init --dry-run``).
    # They are only meaningful for --init; subcommands define their own argv
    # via REMAINDER so these top-level flags don't interfere with subcommand
    # parsing.
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Show what --init WOULD do without downloading or writing anything. "
            "Used only with --init; ignored otherwise."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help=(
            "Force --init to proceed into a scary target (HOME, /, /tmp) or "
            "overwrite a symlinked .opencode/. Used only with --init."
        ),
    )

    sub = parser.add_subparsers(
        title="subcommands",
        dest="command",
        description="Use 'attacklm <subcommand> --help' for details on each subcommand.",
    )

    # ---- train ----
    train_p = sub.add_parser(
        "train",
        help="Train models (QLoRA, GaLore, Spectrum, PiSSA, HPO)",
        description=(
            "Train AttackLM models. Default: single LoRA training.\n"
            "Use --all for bucket-based multi-model training, --hpo for HPO sweeps.\n\n"
            "All remaining arguments after -- are forwarded to the training script.\n"
            "Examples:\n"
            "  attacklm train -- --dataset data/test.jsonl --epochs 10 --train\n"
            "  attacklm train --all -- --single-model --epochs 5\n"
            "  attacklm train --hpo -- --analyze-only"
        ),
    )
    train_p.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Train all buckets",
    )
    train_p.add_argument(
        "--hpo",
        action="store_true",
        default=False,
        help="Run HPO sweep instead of training",
    )
    train_p.add_argument(
        "argv",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the training script (use -- to separate)",
    )
    train_p.set_defaults(func=_cmd_train)

    # ---- init ----
    init_p = sub.add_parser(
        "init",
        help="Initialize dataset (clone, extract, bucket)",
        description=(
            "Initialize dataset. Default: download pre-built dataset from GitHub releases.\n"
            "Use --from-source to build from upstream git repos instead.\n"
            "Use --extract-only/--buckets-only/--clone-only for partial steps."
        ),
    )
    # Partial-step flags (mutually exclusive with each other, but NOT
    # with --from-source/--dataset-url which control the data source).
    init_group = init_p.add_mutually_exclusive_group()
    init_group.add_argument(
        "--extract-only",
        action="store_true",
        default=False,
        help="Run data extractors only",
    )
    init_group.add_argument(
        "--buckets-only",
        action="store_true",
        default=False,
        help="Organize data into buckets only",
    )
    init_group.add_argument(
        "--clone-only",
        action="store_true",
        default=False,
        help="Clone upstream data repos only",
    )
    # Data-source flags (mutually exclusive: build from source vs download tarball).
    init_source_group = init_p.add_mutually_exclusive_group()
    init_source_group.add_argument(
        "--from-source",
        action="store_true",
        default=False,
        help="Build dataset from upstream git repos (clone + extract) instead of downloading pre-built tarball",
    )
    init_source_group.add_argument(
        "--dataset-url",
        type=str,
        default=None,
        help="Override the dataset download URL (default: GitHub latest release)",
    )
    init_p.add_argument(
        "argv",
        nargs=argparse.REMAINDER,
        help="All remaining args are forwarded to the init script",
    )
    init_p.set_defaults(func=_cmd_init)

    # ---- balance ----
    balance_p = sub.add_parser(
        "balance",
        help="Build a balanced subset of the buckets",
        description="Build a balanced subset of the training data.",
    )
    balance_p.add_argument(
        "argv",
        nargs=argparse.REMAINDER,
        help="All remaining args are forwarded to balance_buckets.py",
    )
    balance_p.set_defaults(func=_cmd_balance)

    # ---- steer ----
    steer_p = sub.add_parser(
        "steer",
        help="Activation steering vectors (extract, apply, sweep, diagnose)",
        description=(
            "Activation steering vector toolkit.\n"
            "Use flags to select a mode, or forward args directly.\n\n"
            "Examples:\n"
            "  attacklm steer --extract\n"
            "  attacklm steer --apply -- --model mymodel\n"
            "  attacklm steer --sweep\n"
            "  attacklm steer --diagnose"
        ),
    )
    steer_group = steer_p.add_mutually_exclusive_group()
    steer_group.add_argument(
        "--extract",
        action="store_true",
        default=False,
        help="Extract steering vectors",
    )
    steer_group.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Apply steering vectors",
    )
    steer_group.add_argument(
        "--sweep",
        action="store_true",
        default=False,
        help="Run steering sweep",
    )
    steer_group.add_argument(
        "--diagnose",
        action="store_true",
        default=False,
        help="Diagnose steering vectors",
    )
    steer_p.add_argument(
        "argv",
        nargs=argparse.REMAINDER,
        help="All remaining args are forwarded to steering.py",
    )
    steer_p.set_defaults(func=_cmd_steer)

    # ---- bench ----
    bench_p = sub.add_parser(
        "bench",
        help="Run domain benchmark evaluation",
        description="Run domain benchmark evaluation.",
    )
    bench_p.add_argument(
        "argv",
        nargs=argparse.REMAINDER,
        help="All remaining args are forwarded to domain_bench.py",
    )
    bench_p.set_defaults(func=_cmd_bench)

    # ---- pipeline ----
    pipeline_p = sub.add_parser(
        "pipeline",
        help="Run the full training/evaluation pipeline",
        description="Run the full training/evaluation pipeline.",
    )
    pipeline_p.add_argument(
        "argv",
        nargs=argparse.REMAINDER,
        help="All remaining args are forwarded to pipeline.py",
    )
    pipeline_p.set_defaults(func=_cmd_pipeline)

    # ---- build ----
    build_p = sub.add_parser(
        "build",
        help="Build pipeline: merge → GGUF → install (replaces merge/gguf/build)",
        description=(
            "Build pipeline: merge LoRA adapter → convert to GGUF → install.\n"
            "Use flags to run individual steps only."
        ),
    )
    build_group = build_p.add_mutually_exclusive_group()
    build_group.add_argument(
        "--merge-only",
        action="store_true",
        default=False,
        help="Merge adapter only",
    )
    build_group.add_argument(
        "--gguf-only",
        action="store_true",
        default=False,
        help="Convert to GGUF only",
    )
    build_group.add_argument(
        "--register-ollama",
        action="store_true",
        default=False,
        help="Register GGUF models with Ollama (replaces attacklm-register-ollama)",
    )
    build_p.add_argument(
        "argv",
        nargs=argparse.REMAINDER,
        help="All remaining args are forwarded to the build script",
    )
    build_p.set_defaults(func=_cmd_build)

    # ---- infer ----
    infer_p = sub.add_parser(
        "infer",
        help="Run inference with a trained model",
        description="Run inference with a trained AttackLM model.",
    )
    infer_p.add_argument(
        "argv",
        nargs=argparse.REMAINDER,
        help="All remaining args are forwarded to infer.py",
    )
    infer_p.set_defaults(func=_cmd_infer)

    # ---- eval ----
    eval_p = sub.add_parser(
        "eval",
        help="Evaluation suite (retention, collect-ref, score, compare, golden)",
        description=(
            "Evaluation suite for AttackLM models.\n"
            "Default: retention evaluation. Use flags for specific eval steps."
        ),
    )
    eval_group = eval_p.add_mutually_exclusive_group()
    eval_group.add_argument(
        "--collect-ref",
        action="store_true",
        default=False,
        help="Generate reference continuations",
    )
    eval_group.add_argument(
        "--score",
        action="store_true",
        default=False,
        help="Score candidate models",
    )
    eval_group.add_argument(
        "--compare",
        action="store_true",
        default=False,
        help="Compare two score TSV files",
    )
    eval_group.add_argument(
        "--golden",
        action="store_true",
        default=False,
        help="Golden vector generation/validation",
    )
    eval_p.add_argument(
        "argv",
        nargs=argparse.REMAINDER,
        help="All remaining args are forwarded to the eval script",
    )
    eval_p.set_defaults(func=_cmd_eval)

    # ---- gui ----
    gui_p = sub.add_parser(
        "gui",
        help="Launch the terminal GUI (TUI)",
        description="Launch the AttackLM terminal GUI for interactive training.",
    )
    gui_p.set_defaults(func=_cmd_gui)

    # ---- audit ----
    audit_p = sub.add_parser(
        "audit",
        help="Inversion-attack audit (extraction / MIA)",
        description=(
            "Run an inversion-attack audit on a trained AttackLM model. "
            "Supports extraction (Carlini 2021 prefix-completion) and "
            "MIA (Carlini 2022 reference attack + per-token loss + LiRA). "
            "See attacklm-dataset/docs/ATTACK_TAXONOMY.md."
        ),
    )
    audit_p.add_argument(
        "--attack",
        choices=["extraction", "mia", "all"],
        default="all",
        help="Attack class to run (default: all)",
    )
    audit_p.add_argument(
        "--mia-method",
        choices=["reference", "zlib", "per_token", "lira", "all"],
        default="per_token",
        help="MIA scoring algorithm (default: per_token)",
    )
    audit_p.add_argument(
        "--mia-threshold-mode",
        choices=["median", "percentile", "holdout_file"],
        default="percentile",
        help="How to derive the membership threshold (default: percentile)",
    )
    audit_p.add_argument(
        "--mia-percentile",
        type=int,
        default=5,
        help="Percentile for the threshold (default 5)",
    )
    audit_p.add_argument(
        "--model",
        required=True,
        help="Path to trained model",
    )
    audit_p.add_argument(
        "--dataset-root",
        required=True,
        help="Path to per-source dataset root",
    )
    audit_p.add_argument(
        "--source-filter",
        nargs="*",
        default=[],
        help="Optional: only probe these sources",
    )
    audit_p.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Carlini 2021 K (number of completions per prefix)",
    )
    audit_p.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Per-completion token budget (Carlini 2021 default)",
    )
    audit_p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default 1.0)",
    )
    audit_p.add_argument(
        "--max-records",
        type=int,
        default=50,
        help="Max records per source",
    )
    audit_p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the command that would be run, don't execute",
    )
    audit_p.set_defaults(func=_cmd_audit)

    # ---- demo ----
    demo_p = sub.add_parser(
        "demo",
        help="Run the AttackLM orchestrator demo",
        description="Run the AttackLM orchestrator demo.",
    )
    demo_p.add_argument(
        "argv",
        nargs=argparse.REMAINDER,
        help="All remaining args are forwarded to demo.py",
    )
    demo_p.set_defaults(func=_cmd_demo)

    return parser


def _get_version() -> str:
    """Get the current AttackLM version."""
    try:
        from attacklm.__version__ import __version__

        return __version__
    except ImportError:
        return "unknown"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Primary entry point for ``attacklm`` command."""
    parser = build_parser()
    args = parser.parse_args()

    # ---- Top-level --init dispatch (neuralgentics OpenCode plugin) ----
    # If --init was passed, route to the neuralgentics_init package regardless
    # of whether a subcommand was also given. (No subcommand should be given
    # alongside --init, but if one is, --init wins for safety — it's an
    # explicit user intent.) ``_handle_neuralgentics_init`` calls ``sys.exit``.
    if getattr(args, "init", False):
        _handle_neuralgentics_init(args)
        return  # pragma: no cover — _handle_neuralgentics_init always exits

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Strip the leading '--' that argparse REMAINDER may include
    if hasattr(args, "argv") and args.argv and args.argv[0] == "--":
        args.argv = args.argv[1:]

    rc = args.func(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
