#!/usr/bin/env python3
"""init_pipeline.py — One-shot AttackLM dataset initialization.

Replaces the four-step manual sequence::

    attacklm-clone       # 4. Clone upstream data sources
    attacklm-extract     # 5. Extract training data
    attacklm-attribute   # 6. Augment with attribution
    attacklm-buckets     # 7. Organize into MITRE/AI/tools buckets

with a single command::

    attacklm-init

Behavior:
  1. Probe the local ``data/`` tree for already-cloned upstream sources.
  2. If every required source is present and non-empty → skip the clone
     step entirely and go straight to extract → attribute → buckets.
  3. If any source is missing → print a clear ``Need network access``
     message and (after confirmation, or with ``--yes``) fall back to
     ``git clone`` against the GitHub mirrors of each upstream repo.

The orchestrator never re-downloads data the user already has. It is
safe to re-run after a partial failure: each stage is skipped if its
output already exists (unless ``--force`` is passed for that stage).

Usage::

    attacklm-init                       # interactive
    attacklm-init --yes                 # auto-confirm network fallback
    attacklm-init --dry-run             # show what would happen, do nothing
    attacklm-init --skip-clone          # assume data is already on disk
    attacklm-init --skip-attribute      # buckets have v0.3.0+ provenance already
    attacklm-init --force-extract       # re-run extractors even if outputs exist
    attacklm-init --clean-buckets       # run setup_buckets.py --clean at the end

Exit codes:
  0   success
  1   unexpected runtime error
  2   user declined network fallback
  3   network fallback attempted but every fetch strategy failed
  4   a local data source is missing AND ``--skip-clone`` was passed
   5   missing Python dependencies (run ``pip install attacklm[all]``)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

# --- Dependency check ----------------------------------------------------------


class DependencyError(SystemExit):
    """Raised when core ML dependencies are missing."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__(5)


def _check_dependencies() -> None:
    """Verify that core ML dependencies are installed.

    When installed from PyPI, the user may have only the base package
    without training extras.  Print a clear message and raise
    DependencyError if any required import is missing.
    """
    missing: list[str] = []
    for mod in ("torch", "transformers", "peft"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(
            "\n!!! Missing Python dependencies: " + ", ".join(missing) + "\n"
            "    Run: pip install attacklm[all]\n"
            "    and then re-run attacklm-init.\n",
            file=sys.stderr,
        )
        raise DependencyError(missing)


# --- Paths --------------------------------------------------------------------
#
# When installed from PyPI, this script lives in
# site-packages/attacklm/scripts/ — but the data/ tree should be in
# the user's working directory.  We try the repo-root layout first
# (development), then fall back to cwd.

_REPO_ROOT = Path(__file__).resolve().parent.parent
if (_REPO_ROOT / "data").is_dir():
    BASE_DIR = _REPO_ROOT
else:
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"

# --- Local-check probe --------------------------------------------------------
#
# For each upstream source, define (display_name, expected_dir, marker_path,
# min_marker_size).  A source is considered "present" when its directory
# exists AND its marker path is at least ``min_marker_size`` bytes.
# A bare .git directory alone is NOT enough — the working tree must be
# checked out (this is the common failure mode of a half-completed clone).

_LOCAL_PROBES: list[tuple[str, Path, Path, int]] = [
    # (name, dir, marker, min_bytes)
    (
        "atomic-red-team",
        DATA_DIR / "atomic-red-team",
        DATA_DIR / "atomic-red-team" / "atomics",
        1024,  # > 1KB of atomic tests
    ),
    (
        "stockpile",
        DATA_DIR / "stockpile",
        DATA_DIR / "stockpile" / "README.md",
        256,
    ),
    (
        "sigma",
        DATA_DIR / "sigma",
        DATA_DIR / "sigma" / "rules",
        1024,
    ),
    (
        "metasploit-framework",
        DATA_DIR / "metasploit-framework",
        DATA_DIR / "metasploit-framework" / "modules",
        1024,
    ),
    (
        "mordor",
        DATA_DIR / "mordor",
        DATA_DIR / "mordor" / "datasets",
        1024,
    ),
    (
        "threathunter-playbook",
        DATA_DIR / "threathunter-playbook",
        DATA_DIR / "threathunter-playbook" / "playbooks",
        1024,
    ),
    (
        "elastic-detection-rules",
        DATA_DIR / "elastic-detection-rules",
        DATA_DIR / "elastic-detection-rules" / "rules",
        1024,
    ),
    (
        "splunk-security-content",
        DATA_DIR / "splunk-security-content",
        DATA_DIR / "splunk-security-content" / "detections",
        1024,
    ),
    (
        "nist-sp800-61r3",
        DATA_DIR / "nist-sp800-61r3",
        DATA_DIR / "nist-sp800-61r3" / "NIST.SP.800-61r3.pdf",
        1024,
    ),
]

# (name, github_url, dest_dir) — same set as clone_repos.sh, kept in sync
_REMOTE_REPOS: list[tuple[str, str, Path]] = [
    (
        "atomic-red-team",
        "https://github.com/redcanaryco/atomic-red-team.git",
        DATA_DIR / "atomic-red-team",
    ),
    (
        "stockpile",
        "https://github.com/mitre/stockpile.git",
        DATA_DIR / "stockpile",
    ),
    (
        "sigma",
        "https://github.com/SigmaHQ/sigma.git",
        DATA_DIR / "sigma",
    ),
    (
        "metasploit-framework",
        "https://github.com/rapid7/metasploit-framework.git",
        DATA_DIR / "metasploit-framework",
    ),
    # NOTE: nist-sp800-61r3 is a PDF download, not a git repo — excluded
    # from remote repos. Users must download it manually from
    # https://csrc.nist.gov/pubs/sp/800-61/r3/final
    (
        "mordor",
        "https://github.com/OTRF/Security-Datasets.git",
        DATA_DIR / "mordor",
    ),
    (
        "threathunter-playbook",
        "https://github.com/OTRF/ThreatHunter-Playbook.git",
        DATA_DIR / "threathunter-playbook",
    ),
    (
        "elastic-detection-rules",
        "https://github.com/elastic/detection-rules.git",
        DATA_DIR / "elastic-detection-rules",
    ),
    (
        "splunk-security-content",
        "https://github.com/splunk/security_content.git",
        DATA_DIR / "splunk-security-content",
    ),
]


# --- Probe result types -------------------------------------------------------


@dataclass(frozen=True)
class LocalProbe:
    """Result of probing one upstream source on local disk."""

    name: str
    dest: Path
    marker: Path
    present: bool
    detail: str  # human-readable, e.g. "222M" or "missing"


def _probe_one(name: str, dest: Path, marker: Path, min_bytes: int) -> LocalProbe:
    """Check whether ``marker`` (or fallback ``dest``) is present and big enough."""
    if not dest.exists():
        return LocalProbe(name, dest, marker, False, "missing dir")
    # Prefer the marker; fall back to the dest itself (whole-dir probe).
    target = marker if marker.exists() else dest
    if not target.exists():
        return LocalProbe(name, dest, marker, False, "missing marker")
    if target.is_file():
        size = target.stat().st_size
    else:
        # directory — sum file sizes up to a 1GB cap so this stays fast
        total = 0
        try:
            for p in target.rglob("*"):
                if p.is_file():
                    total += p.stat().st_size
                    if total >= 1024 * 1024 * 1024:
                        break
        except OSError as e:
            return LocalProbe(name, dest, marker, False, f"rglob error: {e}")
        size = total
    if size < min_bytes:
        return LocalProbe(
            name, dest, marker, False, f"marker too small ({size} < {min_bytes} B)"
        )
    # Human-friendly size label
    if size < 1024:
        label = f"{size}B"
    elif size < 1024 * 1024:
        label = f"{size // 1024}KB"
    elif size < 1024 * 1024 * 1024:
        label = f"{size // (1024 * 1024)}MB"
    else:
        label = f"{size / (1024 * 1024 * 1024):.1f}GB"
    return LocalProbe(name, dest, marker, True, label)


def probe_local() -> list[LocalProbe]:
    """Run the local-check against every required source."""
    return [
        _probe_one(name, dest, marker, min_bytes)
        for (name, dest, marker, min_bytes) in _LOCAL_PROBES
    ]


# --- Stage runners ------------------------------------------------------------
#
# Each stage returns its process exit code (0 == success).  The wrapper
# function ``_run_stage`` prints a banner, runs the callable, and
# surfaces the exit code.  Stages are intentionally small: the heavy
# lifting lives in the existing ``extract_*.py``, ``setup_buckets.py``,
# etc. scripts that this orchestrator composes.


def _run_stage(label: str, fn: Callable[[], int]) -> int:
    bar = "=" * 72
    print(f"\n{bar}\n=== {label}\n{bar}", file=sys.stderr, flush=True)
    rc = fn()
    if rc != 0:
        print(f"\n!!! stage '{label}' failed with exit code {rc}", file=sys.stderr)
    return rc


def stage_clone(repos: Sequence[tuple[str, str, Path]], force: bool = False) -> int:
    """Clone (or update) each upstream repo.  Used as the network fallback."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    any_failed = False
    for name, url, dest in repos:
        if dest.exists() and (dest / ".git").exists():
            if not force:
                print(
                    f"  [skip] {name}: already cloned at {dest} (use --force-clone to re-clone)",
                    file=sys.stderr,
                )
                continue
            print(f"  [force] removing {dest}", file=sys.stderr)
            shutil.rmtree(dest, ignore_errors=True)
        print(f"  [clone] {name} <- {url}", file=sys.stderr)
        result = subprocess.run(
            ["git", "clone", "--depth=1", url, str(dest)],
            check=False,
        )
        if result.returncode != 0:
            print(
                f"  [FAIL]  {name}: git clone returned {result.returncode}",
                file=sys.stderr,
            )
            any_failed = True
    return 1 if any_failed else 0


def stage_extract(force: bool = False) -> int:
    """Run the per-source extractors.

    Each extractor is idempotent (overwrites its output), so ``--force``
    is implicit.  ``force`` is kept as a flag for API symmetry.
    """
    _ = force
    extractors = [
        "extract_atomic_red_team_to_jsonl.py",
        "extract_caldera_plugins_to_jsonl.py",
        "parse_metasploit_to_jsonl.py",
        "extract_ai_tools_to_jsonl.py",
        "extract_sigma_defensive.py",
        "extract_mordor.py",
        "extract_threathunter_playbook.py",
        "extract_elastic_rules.py",
        "extract_splunk_content.py",
        "extract_nist_ir.py",
    ]
    for script in extractors:
        path = BASE_DIR / "scripts" / script
        # When installed from PyPI, scripts are in site-packages/attacklm/scripts/
        if not path.exists():
            # Fallback: check the installed package location
            import attacklm

            pkg_scripts = Path(attacklm.__file__).parent / "scripts" / script
            if pkg_scripts.exists():
                path = pkg_scripts
            else:
                print(f"  [skip] {script}: not present in scripts/", file=sys.stderr)
                continue
        print(f"  [run]  {script}", file=sys.stderr)
        result = subprocess.run([sys.executable, str(path)], check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


def _resolve_script(name: str) -> Path | None:
    """Resolve a script path, handling both dev and PyPI installs."""
    # Development layout: scripts/ at repo root
    path = BASE_DIR / "scripts" / name
    if path.exists():
        return path
    # PyPI layout: scripts in site-packages/attacklm/scripts/
    try:
        import attacklm

        pkg_path = Path(attacklm.__file__).parent / "scripts" / name
        if pkg_path.exists():
            return pkg_path
    except ImportError:
        pass
    return None


def stage_attribute() -> int:
    """Add per-pair source/license attribution fields.

    After v0.3.0 this is a no-op (provenance is baked into the per-source
    layout produced by ``setup_buckets.py``).  We still call the legacy
    ``augment_attribution.py`` script so users upgrading from v0.2.x see
    the friendly notice and aren't confused.
    """
    path = _resolve_script("augment_attribution.py")
    if path is None:
        return 0
    print(
        "  [note] augment_attribution.py is a no-op in v0.3.0+ (per-source layout "
        "carries attribution natively); running anyway for upgrade notice.",
        file=sys.stderr,
    )
    result = subprocess.run([sys.executable, str(path)], check=False)
    return result.returncode


def stage_buckets(clean: bool = False) -> int:
    """Run ``setup_buckets.py`` + ``reorganize_buckets.py``."""
    setup = _resolve_script("setup_buckets.py")
    reorg = _resolve_script("reorganize_buckets.py")
    if setup is None:
        print("  [skip] setup_buckets.py: not present", file=sys.stderr)
        return 0
    setup_args: list[str] = []
    if clean:
        setup_args.append("--clean")
    print(f"  [run]  {setup.name} {' '.join(setup_args)}", file=sys.stderr)
    rc = subprocess.run(
        [sys.executable, str(setup), *setup_args], check=False
    ).returncode
    if rc != 0:
        return rc
    if reorg is None:
        print("  [skip] reorganize_buckets.py: not present", file=sys.stderr)
        return 0
    print(f"  [run]  {reorg.name}", file=sys.stderr)
    return subprocess.run(
        [sys.executable, str(reorg), *setup_args], check=False
    ).returncode


# --- Output detection ---------------------------------------------------------
#
# ``--force`` semantics: re-run a stage even if its primary output
# already exists.  We don't *require* the output to exist to run
# (extractors overwrite anyway), so the "skip if up-to-date" logic is
# conservative: skip only when the user explicitly asked for force=False
# AND the bucket directory already has a manifest.


def _buckets_already_built() -> bool:
    manifest = DATASETS_DIR / "buckets" / "manifest.json"
    if not manifest.exists():
        return False
    try:
        import json

        with manifest.open() as f:
            data = json.load(f)
        return data.get("total_records", 0) > 0
    except (OSError, ValueError):
        return False


# --- Confirmation prompt ------------------------------------------------------


def _confirm(msg: str, assume_yes: bool) -> bool:
    if assume_yes:
        print(f"{msg}  [auto-yes]", file=sys.stderr)
        return True
    try:
        reply = input(f"{msg}  [y/N] ").strip().lower()
    except EOFError:
        return False
    return reply in ("y", "yes")


# --- Main entry point ---------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="One-shot AttackLM dataset init: clone → extract → attribute → buckets."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Auto-confirm network fallback (non-interactive).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit. Does not clone, extract, or write.",
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Assume data is already on disk; do not probe or fetch.",
    )
    parser.add_argument(
        "--skip-attribute",
        action="store_true",
        help="Skip the attribution stage (buckets already have v0.3.0+ provenance).",
    )
    parser.add_argument(
        "--skip-buckets",
        action="store_true",
        help="Stop after extract+attribute; do not run setup_buckets.py.",
    )
    parser.add_argument(
        "--force-clone",
        action="store_true",
        help="Re-clone even if a destination directory already exists.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-run extractors even if datasets dir already has outputs.",
    )
    parser.add_argument(
        "--clean-buckets",
        action="store_true",
        help="Pass --clean to setup_buckets.py to remove old flat files.",
    )
    args = parser.parse_args(argv)

    print("=" * 72, file=sys.stderr)
    print("attacklm-init: one-shot dataset initialization", file=sys.stderr)
    print("=" * 72, file=sys.stderr)

    # -- Stage 0a: dependency check --
    _check_dependencies()

    # -- Stage 0: probe local data --
    need_clone = False
    if args.skip_clone:
        print(
            "[stage 0] --skip-clone passed; assuming data is present.", file=sys.stderr
        )
        need_clone = False
    else:
        print("\n[stage 0] probing local data/ tree...", file=sys.stderr)
        probes = probe_local()
        width = max(len(p.name) for p in probes)
        missing = [p for p in probes if not p.present]
        for p in probes:
            status = "OK  " if p.present else "MISS"
            print(
                f"  [{status}] {p.name:<{width}}  {p.dest.relative_to(BASE_DIR)}  ({p.detail})",
                file=sys.stderr,
            )
        if missing:
            need_clone = True
            print(
                f"\n  {len(missing)}/{len(probes)} upstream sources are missing or incomplete:",
                file=sys.stderr,
            )
            for p in missing:
                print(f"    - {p.name}: {p.detail}", file=sys.stderr)
        else:
            print(
                f"\n  All {len(probes)} upstream sources are present locally. "
                "Skipping clone.",
                file=sys.stderr,
            )

    if args.dry_run:
        plan = ["[dry-run] would run:"]
        if need_clone:
            plan.append("  - stage 1: clone (network fallback)")
        else:
            plan.append("  - stage 1: clone  (skipped — all sources present)")
        if not args.skip_attribute:
            plan.append("  - stage 2: extract")
            plan.append("  - stage 3: attribute")
        else:
            plan.append("  - stage 2: extract")
            plan.append("  - stage 3: attribute  (skipped via --skip-attribute)")
        if not args.skip_buckets:
            plan.append("  - stage 4: buckets")
        else:
            plan.append("  - stage 4: buckets  (skipped via --skip-buckets)")
        print("\n".join(plan), file=sys.stderr)
        return 0

    # -- Stage 1: clone (network fallback) --
    if need_clone:
        print(
            "\n[stage 1] Need network access to fetch missing upstream sources.",
            file=sys.stderr,
        )
        for name, url, _ in _REMOTE_REPOS:
            print(f"  - {name}  ←  {url}", file=sys.stderr)
        if not _confirm("Proceed with `git clone` from github.com?", args.yes):
            print(
                "  [abort] user declined network fallback. Re-run with --skip-clone "
                "if you have data elsewhere, or clone manually and retry.",
                file=sys.stderr,
            )
            return 2
        rc = _run_stage(
            "stage 1: clone upstream sources",
            lambda: stage_clone(_REMOTE_REPOS, force=args.force_clone),
        )
        if rc != 0:
            return 3  # network fallback failed
    else:
        print(
            "\n[stage 1] clone: SKIPPED (all sources present locally)",
            file=sys.stderr,
        )

    # -- Stage 2: extract --
    if not args.force_extract and _buckets_already_built():
        print(
            "\n[stage 2] extract: SKIPPED (buckets/manifest.json already reports records). "
            "Pass --force-extract to re-run.",
            file=sys.stderr,
        )
    else:
        rc = _run_stage("stage 2: extract training data", stage_extract)
        if rc != 0:
            return rc

    # -- Stage 3: attribute --
    if args.skip_attribute:
        print("\n[stage 3] attribute: SKIPPED (--skip-attribute)", file=sys.stderr)
    else:
        rc = _run_stage("stage 3: add per-pair attribution", stage_attribute)
        if rc != 0:
            return rc

    # -- Stage 4: buckets --
    if args.skip_buckets:
        print("\n[stage 4] buckets: SKIPPED (--skip-buckets)", file=sys.stderr)
    else:
        rc = _run_stage(
            "stage 4: organize into MITRE/AI/tools buckets",
            lambda: stage_buckets(clean=args.clean_buckets),
        )
        if rc != 0:
            return rc

    print("\n" + "=" * 72, file=sys.stderr)
    print("attacklm-init: complete", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    print(
        "\nNext steps:\n"
        "  - attacklm-balance    # balance the per-tactic distributions\n"
        "  - attacklm-train-all  # train all buckets\n"
        "  - attacklm-build      # merge → GGUF → install\n",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
