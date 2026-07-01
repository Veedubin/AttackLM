"""AttackLM unified CLI with argparse subcommands.

Usage::

    attacklm train [--dataset all] [--hpo] [args...]
    attacklm init [--extract-only|--buckets-only|--attribute-only|--clone-only] [args...]
    attacklm balance [args...]
    attacklm build [--merge-only|--gguf-only|--register-ollama] [args...]
    attacklm infer [args...]
    attacklm eval [--collect-ref|--score|--compare|--golden] [args...]
    attacklm gui
    attacklm demo [args...]

Old hyphenated commands (attacklm-train, attacklm-hpo, etc.) still work
but print a deprecation warning and delegate to the new subcommand.
"""

from __future__ import annotations

import argparse
import runpy
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
    """Handle ``attacklm init`` — dispatch to init substeps or full pipeline."""
    if args.clone_only:
        return _run_shell_script("clone_repos.sh", args.argv)
    if args.extract_only:
        # Run all extractors sequentially (same as attacklm-extract)
        extractors = [
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
        for extractor in extractors:
            print(f"\n=== Running {extractor} ===", file=sys.stderr)
            rc = _run_python_script(extractor, [])
            if rc != 0:
                print(
                    f"Extractor {extractor} failed with exit code {rc}",
                    file=sys.stderr,
                )
                return rc
        print(
            "\n=== All extractors complete. Next: attacklm init --buckets-only ===",
            file=sys.stderr,
        )
        return 0
    if args.buckets_only:
        extra_args = list(args.argv)
        print("\n=== Running setup_buckets.py ===", file=sys.stderr)
        rc = _run_python_script("setup_buckets.py", extra_args)
        if rc != 0:
            return rc
        print("\n=== Running reorganize_buckets.py ===", file=sys.stderr)
        return _run_python_script("reorganize_buckets.py", extra_args)
    if args.attribute_only:
        return _run_python_script("augment_attribution.py", args.argv)
    # Default: full init pipeline — forward --from-source / --dataset-url
    forwarded = list(args.argv)
    if args.from_source:
        forwarded.append("--from-source")
    if args.dataset_url:
        forwarded.extend(["--dataset-url", args.dataset_url])
    return _run_python_script("init_pipeline.py", forwarded)


def _cmd_balance(args: argparse.Namespace) -> int:
    """Handle ``attacklm balance``."""
    return _run_python_script("balance_buckets.py", args.argv)


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
        help="Train all buckets (replaces attacklm-train-all)",
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
        help="Initialize dataset (clone, extract, attribute, bucket)",
        description=(
            "Initialize dataset. Default: download pre-built dataset from GitHub releases.\n"
            "Use --from-source to build from upstream git repos instead.\n"
            "Use --extract-only/--buckets-only/etc. for partial steps."
        ),
    )
    # Partial-step flags (mutually exclusive with each other, but NOT
    # with --from-source/--dataset-url which control the data source).
    init_group = init_p.add_mutually_exclusive_group()
    init_group.add_argument(
        "--extract-only",
        action="store_true",
        default=False,
        help="Run data extractors only (replaces attacklm-extract)",
    )
    init_group.add_argument(
        "--buckets-only",
        action="store_true",
        default=False,
        help="Organize data into buckets only (replaces attacklm-buckets)",
    )
    init_group.add_argument(
        "--attribute-only",
        action="store_true",
        default=False,
        help="Add per-pair source/license attribution only (replaces attacklm-attribute)",
    )
    init_group.add_argument(
        "--clone-only",
        action="store_true",
        default=False,
        help="Clone upstream data repos only (replaces attacklm-clone)",
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
        help="Merge adapter only (replaces attacklm-merge)",
    )
    build_group.add_argument(
        "--gguf-only",
        action="store_true",
        default=False,
        help="Convert to GGUF only (replaces attacklm-gguf)",
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
        help="Generate reference continuations (replaces attacklm-collect-ref)",
    )
    eval_group.add_argument(
        "--score",
        action="store_true",
        default=False,
        help="Score candidate models (replaces attacklm-score)",
    )
    eval_group.add_argument(
        "--compare",
        action="store_true",
        default=False,
        help="Compare two score TSV files (replaces attacklm-compare)",
    )
    eval_group.add_argument(
        "--golden",
        action="store_true",
        default=False,
        help="Golden vector generation/validation (replaces attacklm-golden)",
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

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Strip the leading '--' that argparse REMAINDER may include
    if hasattr(args, "argv") and args.argv and args.argv[0] == "--":
        args.argv = args.argv[1:]

    rc = args.func(args)
    sys.exit(rc)


# ---------------------------------------------------------------------------
# Deprecation wrappers — old hyphenated commands
# ---------------------------------------------------------------------------
# Each prints a one-line deprecation notice, then delegates to the
# corresponding subcommand.  These are referenced from
# pyproject.toml [project.scripts] for backward compatibility.
#
# They will be removed in a future release (two-release deprecation window).

_DEPRECATED_MSG = (
    "{} is deprecated. Use '{}' instead. "
    "See 'attacklm --help' for the new subcommand interface."
)


def _deprecated(old_cmd: str, new_cmd: str, argv: Sequence[str]) -> int:
    """Print deprecation warning and delegate to new subcommand."""
    print(_DEPRECATED_MSG.format(old_cmd, new_cmd), file=sys.stderr)
    # Build new argv and re-run through the parser.
    # Insert "--" so REMAINDER captures all forwarded args (including
    # flags like --yes, --dry-run that argparse would otherwise reject).
    subcmd = new_cmd.split()[1]  # strip "attacklm " prefix
    new_argv = [subcmd, "--"] + list(argv)
    parser = build_parser()
    args = parser.parse_args(new_argv)
    # Strip the leading "--" that we inserted — the handler expects
    # just the forwarded args.
    if hasattr(args, "argv") and args.argv and args.argv[0] == "--":
        args.argv = args.argv[1:]
    return args.func(args)


# ---- Training ----


def main_train(argv: Sequence[str] | None = None) -> int:
    """Deprecated: use ``attacklm train``."""
    return _deprecated("attacklm-train", "attacklm train", argv or sys.argv[1:])


def main_train_all(argv: Sequence[str] | None = None) -> int:
    """Deprecated: use ``attacklm train --all``."""
    args = argv or sys.argv[1:]
    return _deprecated(
        "attacklm-train-all",
        "attacklm train --all",
        ["--all"] + list(args),
    )


def main_train_lora(argv: Sequence[str] | None = None) -> int:
    """Deprecated: use ``attacklm train``."""
    return _deprecated("attacklm-train-lora", "attacklm train", argv or sys.argv[1:])


def main_hpo(argv: Sequence[str] | None = None) -> int:
    """Deprecated: use ``attacklm train --hpo``."""
    args = argv or sys.argv[1:]
    return _deprecated("attacklm-hpo", "attacklm train --hpo", ["--hpo"] + list(args))


# ---- Data / Init ----


def main_extract(argv: Sequence[str] | None = None) -> int:
    """Deprecated: use ``attacklm init --extract-only``."""
    return _deprecated(
        "attacklm-extract",
        "attacklm init --extract-only",
        ["--extract-only"] + list(argv or []),
    )


def main_buckets(argv: Sequence[str] | None = None) -> int:
    """Deprecated: use ``attacklm init --buckets-only``."""
    return _deprecated(
        "attacklm-buckets",
        "attacklm init --buckets-only",
        ["--buckets-only"] + list(argv or []),
    )


def main_attribute(argv: Sequence[str] | None = None) -> int:
    """Deprecated: use ``attacklm init --attribute-only``."""
    return _deprecated(
        "attacklm-attribute",
        "attacklm init --attribute-only",
        ["--attribute-only"] + list(argv or []),
    )


def main_clone(argv: Sequence[str] | None = None) -> int:
    """Deprecated: use ``attacklm init --clone-only``."""
    return _deprecated(
        "attacklm-clone",
        "attacklm init --clone-only",
        ["--clone-only"] + list(argv or []),
    )


def main_init(argv: Sequence[str] | None = None) -> int:
    """Deprecated: use ``attacklm init``."""
    return _deprecated("attacklm-init", "attacklm init", argv or sys.argv[1:])


# ---- Balance ----


def main_balance(argv: Sequence[str] | None = None) -> int:
    """Deprecated: use ``attacklm balance``."""
    return _deprecated("attacklm-balance", "attacklm balance", argv or sys.argv[1:])


# ---- Build ----


def main_merge(argv: Sequence[str] | None = None) -> int:
    """Deprecated: use ``attacklm build --merge-only``."""
    return _deprecated(
        "attacklm-merge",
        "attacklm build --merge-only",
        ["--merge-only"] + list(argv or []),
    )


def main_gguf(argv: Sequence[str] | None = None) -> int:
    """Deprecated: use ``attacklm build --gguf-only``."""
    return _deprecated(
        "attacklm-gguf",
        "attacklm build --gguf-only",
        ["--gguf-only"] + list(argv or []),
    )


def main_build(argv: Sequence[str] | None = None) -> int:
    """Deprecated: use ``attacklm build``."""
    return _deprecated("attacklm-build", "attacklm build", argv or sys.argv[1:])


# ---- Eval ----


def main_eval(argv: Sequence[str] | None = None) -> int:
    """Deprecated: use ``attacklm eval``."""
    return _deprecated("attacklm-eval", "attacklm eval", argv or sys.argv[1:])


def main_collect_ref(argv: Sequence[str] | None = None) -> int:
    """Deprecated: use ``attacklm eval --collect-ref``."""
    return _deprecated(
        "attacklm-collect-ref",
        "attacklm eval --collect-ref",
        ["--collect-ref"] + list(argv or []),
    )


def main_score(argv: Sequence[str] | None = None) -> int:
    """Deprecated: use ``attacklm eval --score``."""
    return _deprecated(
        "attacklm-score", "attacklm eval --score", ["--score"] + list(argv or [])
    )


def main_compare(argv: Sequence[str] | None = None) -> int:
    """Deprecated: use ``attacklm eval --compare``."""
    return _deprecated(
        "attacklm-compare", "attacklm eval --compare", ["--compare"] + list(argv or [])
    )


def main_golden(argv: Sequence[str] | None = None) -> int:
    """Deprecated: use ``attacklm eval --golden``."""
    return _deprecated(
        "attacklm-golden", "attacklm eval --golden", ["--golden"] + list(argv or [])
    )


# ---- Inference / Demo ----


def main_infer(argv: Sequence[str] | None = None) -> int:
    """Deprecated: use ``attacklm infer``."""
    return _deprecated("attacklm-infer", "attacklm infer", argv or sys.argv[1:])


def main_demo(argv: Sequence[str] | None = None) -> int:
    """Deprecated: use ``attacklm demo``."""
    return _deprecated("attacklm-demo", "attacklm demo", argv or sys.argv[1:])


if __name__ == "__main__":
    main()
