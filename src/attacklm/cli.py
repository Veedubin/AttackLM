"""Console script entry points for AttackLM.

These wrappers dispatch to the canonical scripts in ``scripts/``.  The package
doesn't duplicate the script logic — ``scripts/`` is the source of truth, and
these entry points make the tools invokable as ``attacklm-train``,
``attacklm-hpo``, etc. after ``pip install attacklm``.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Sequence

# Locate the scripts/ directory relative to this package.
# src/attacklm/cli.py → ../../scripts
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


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


# --- Public entry points (referenced from pyproject.toml [project.scripts]) ---


def main_train(argv: Sequence[str] | None = None) -> int:
    """Train a single model: attacklm-train [args]"""
    return _run_python_script(
        "train_template.py", argv if argv is not None else sys.argv[1:]
    )


def main_train_all(argv: Sequence[str] | None = None) -> int:
    """Train all buckets / HPO: attacklm-train-all [args]"""
    return _run_python_script(
        "train_all.py", argv if argv is not None else sys.argv[1:]
    )


def main_hpo(argv: Sequence[str] | None = None) -> int:
    """Run HPO sweep + analysis: attacklm-hpo [args]"""
    return _run_python_script(
        "hpo_runner.py", argv if argv is not None else sys.argv[1:]
    )


def main_infer(argv: Sequence[str] | None = None) -> int:
    """Run inference: attacklm-infer [args]"""
    return _run_python_script("infer.py", argv if argv is not None else sys.argv[1:])


def main_merge(argv: Sequence[str] | None = None) -> int:
    """Merge LoRA adapter: attacklm-merge [args]"""
    return _run_python_script(
        "merge_adapter.py", argv if argv is not None else sys.argv[1:]
    )


def main_gguf(argv: Sequence[str] | None = None) -> int:
    """Convert to GGUF: attacklm-gguf [args]"""
    return _run_python_script(
        "convert_to_gguf.py", argv if argv is not None else sys.argv[1:]
    )


def main_demo(argv: Sequence[str] | None = None) -> int:
    """Run demo: attacklm-demo [args]"""
    return _run_python_script("demo.py", argv if argv is not None else sys.argv[1:])


def main_extract(argv: Sequence[str] | None = None) -> int:
    """Run all data extractors in sequence: attacklm-extract"""
    _ = argv  # extract takes no user args; runs all extractors sequentially
    extractors = [
        "extract_atomic_red_team_to_jsonl.py",
        "extract_caldera_plugins_to_jsonl.py",
        "parse_metasploit_to_jsonl.py",
        "extract_rta_to_jsonl.py",
        "extract_infection_monkey_to_jsonl.py",
        "extract_ai_tools_to_jsonl.py",
    ]
    for extractor in extractors:
        print(f"\n=== Running {extractor} ===", file=sys.stderr)
        rc = _run_python_script(extractor, [])
        if rc != 0:
            print(f"Extractor {extractor} failed with exit code {rc}", file=sys.stderr)
            return rc
    print("\n=== All extractors complete. Next: attacklm-buckets ===", file=sys.stderr)
    return 0


def main_buckets(argv: Sequence[str] | None = None) -> int:
    """Organize data into buckets: attacklm-buckets"""
    args = argv if argv is not None else sys.argv[1:]
    print("\n=== Running setup_buckets.py ===", file=sys.stderr)
    rc = _run_python_script("setup_buckets.py", args)
    if rc != 0:
        return rc
    print("\n=== Running reorganize_buckets.py ===", file=sys.stderr)
    return _run_python_script("reorganize_buckets.py", args)


def main_attribute(argv: Sequence[str] | None = None) -> int:
    """Add per-pair source/license attribution: attacklm-attribute"""
    return _run_python_script(
        "augment_attribution.py", argv if argv is not None else sys.argv[1:]
    )


def main_clone(argv: Sequence[str] | None = None) -> int:
    """Clone upstream data repos: attacklm-clone"""
    return _run_shell_script(
        "clone_repos.sh", argv if argv is not None else sys.argv[1:]
    )


def main_init(argv: Sequence[str] | None = None) -> int:
    """One-shot dataset init: clone → extract → attribute → buckets.

    Replaces the four-step manual sequence (``attacklm-clone`` →
    ``attacklm-extract`` → ``attacklm-attribute`` → ``attacklm-buckets``)
    with a single command.  Probes ``data/`` first; only fetches from
    GitHub if a source is missing.  See ``scripts/init_pipeline.py`` for
    full docs.
    """
    return _run_python_script(
        "init_pipeline.py", argv if argv is not None else sys.argv[1:]
    )


def main_balance(argv: Sequence[str] | None = None) -> int:
    """Build a balanced subset of the buckets: attacklm-balance"""
    return _run_python_script(
        "balance_buckets.py", argv if argv is not None else sys.argv[1:]
    )


def main_build(argv: Sequence[str] | None = None) -> int:
    """One-shot: merge LoRA → GGUF → install: attacklm-build"""
    return _run_python_script("build.py", argv if argv is not None else sys.argv[1:])


def main_eval(argv: Sequence[str] | None = None) -> int:
    """Run retention evaluation: attacklm-eval [args]"""
    return _run_python_script(
        "eval_retention.py", argv if argv is not None else sys.argv[1:]
    )


def main_train_lora(argv: Sequence[str] | None = None) -> int:
    """Train a single model with explicit LoRA focus: attacklm-train-lora [args]"""
    return _run_python_script(
        "train_template.py", argv if argv is not None else sys.argv[1:]
    )


def main_collect_ref(argv: Sequence[str] | None = None) -> int:
    """Generate reference continuations: attacklm-collect-ref [args]"""
    return _run_python_script(
        "collect_reference.py", argv if argv is not None else sys.argv[1:]
    )


def main_score(argv: Sequence[str] | None = None) -> int:
    """Score candidate model against references: attacklm-score [args]"""
    return _run_python_script(
        "score_candidates.py", argv if argv is not None else sys.argv[1:]
    )


def main_compare(argv: Sequence[str] | None = None) -> int:
    """Compare two score TSV files: attacklm-compare [args]"""
    return _run_python_script(
        "compare_scores.py", argv if argv is not None else sys.argv[1:]
    )


def main_golden(argv: Sequence[str] | None = None) -> int:
    """Golden vector generation/validation: attacklm-golden [args]"""
    return _run_python_script(
        "golden_vectors.py", argv if argv is not None else sys.argv[1:]
    )


if __name__ == "__main__":
    # Allow ``python -m attacklm.cli`` to show a help message
    print("AttackLM CLI dispatchers. Use the installed console scripts:")
    print(
        "  Training:   attacklm-train, attacklm-train-lora, attacklm-train-all, attacklm-hpo"
    )
    print("  Inference:  attacklm-infer, attacklm-demo")
    print(
        "  Data:       attacklm-extract, attacklm-buckets, attacklm-attribute, attacklm-clone,"
    )
    print("              attacklm-init, attacklm-balance")
    print("  Build:      attacklm-merge, attacklm-gguf, attacklm-build")
    print("  Evaluation: attacklm-eval, attacklm-collect-ref, attacklm-score,")
    print("              attacklm-compare, attacklm-golden")
    sys.exit(0)
