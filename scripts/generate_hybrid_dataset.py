#!/usr/bin/env python3
"""generate_hybrid_dataset.py — Master orchestrator for hybrid dataset generation.

Combines deterministic acquisition scripts (90%) with LLM synthetic generation
(10%) to produce a balanced, varied training dataset for AttackLM.

The 90/10 split gives you the reliability of hand-crafted deterministic data
while adding just enough LLM variety to improve generalization.

Phases:
    1. Deterministic — runs acquisition scripts to produce structured data
    2. LLM — runs generate_synthetic_scarce.py to fill the variety gap
    3. Validation — checks actual counts match targets within tolerance

Usage:
    # Generate full hybrid dataset (default ~20K pairs)
    python scripts/generate_hybrid_dataset.py

    # Smaller test set
    python scripts/generate_hybrid_dataset.py --total-size 5000

    # Skip LLM phase (deterministic only)
    python scripts/generate_hybrid_dataset.py --no-llm

    # Skip deterministic phase (LLM only)
    python scripts/generate_hybrid_dataset.py --no-deterministic

    # Custom LLM ratio (default 0.1 = 10%)
    python scripts/generate_hybrid_dataset.py --llm-ratio 0.15

    # Specify profile for deterministic balance
    python scripts/generate_hybrid_dataset.py --profile 7b-128gb-balanced

    # LLM backend options
    python scripts/generate_hybrid_dataset.py --backend ollama --temperature 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
BUCKETS_DIR = PROJECT_DIR / "data" / "datasets" / "buckets"
SYNTHETIC_DIR = PROJECT_DIR / "data" / "datasets" / "synthetic"
OUTPUT_DIR = PROJECT_DIR / "data" / "datasets" / "hybrid"

# ---------------------------------------------------------------------------
# Category mapping — maps bucket categories to synthetic categories
# ---------------------------------------------------------------------------
# balance_buckets.py uses these category names:
#   tactic, tools, web_app, identity, cloud, social_engineering,
#   supply_chain, wireless, physical, ics, ai_specific, meta
#
# The synthetic script (generate_synthetic_scarce.py) uses CATEGORIES keys:
#   web_app, cloud, supply_chain, social_engineering, ics_scada, wireless
#
# We need a mapping from balance_buckets categories -> synthetic categories.
BUCKET_TO_SYNTHETIC: dict[str, str] = {
    "web_app": "web_app",
    "cloud": "cloud",
    "supply_chain": "supply_chain",
    "social_engineering": "social_engineering",
    "ics": "ics_scada",
    "wireless": "wireless",
}

# Categories that have deterministic acquisition scripts
DETERMINISTIC_SCRIPTS: list[dict] = [
    {
        "name": "Red Team Tactics",
        "script": "acquire_red_team_tactics.py",
        "bucket": "attack_tactics/red_team_tactics",
        "category": "tactic",
    },
    {
        "name": "Web Application Attacks",
        "script": "acquire_web_attack_dataset.py",
        "bucket": "web_app/attacks",
        "category": "web_app",
    },
    {
        "name": "Phishing & Social Engineering",
        "script": "acquire_phishing_dataset.py",
        "bucket": "social_engineering/phishing",
        "category": "social_engineering",
    },
    {
        "name": "Cloud Security Attacks",
        "script": "acquire_cloud_attack_dataset.py",
        "bucket": "cloud/attacks",
        "category": "cloud",
    },
    {
        "name": "Supply Chain Attacks",
        "script": "acquire_supply_chain_dataset.py",
        "bucket": "supply_chain/attacks",
        "category": "supply_chain",
    },
    {
        "name": "ICS/SCADA Attacks",
        "script": "acquire_ics_dataset.py",
        "bucket": "ics/attacks",
        "category": "ics",
    },
    {
        "name": "Wireless Attacks",
        "script": "acquire_wireless_dataset.py",
        "bucket": "wireless/attacks",
        "category": "wireless",
    },
]

# Default balance from balance_buckets.py (12 categories)
DEFAULT_CATEGORY_SHARES: dict[str, float] = {
    "tactic": 0.15,
    "tools": 0.10,
    "web_app": 0.15,
    "identity": 0.10,
    "cloud": 0.10,
    "social_engineering": 0.08,
    "supply_chain": 0.06,
    "wireless": 0.04,
    "physical": 0.04,
    "ics": 0.05,
    "ai_specific": 0.08,
    "meta": 0.05,
}


# ---------------------------------------------------------------------------
# Phase 1: Deterministic data acquisition
# ---------------------------------------------------------------------------
def run_deterministic_scripts(
    targets: dict[str, int],
    fallback: bool = False,
) -> dict[str, dict]:
    """Run each deterministic acquisition script and return results.

    Args:
        targets: {category: target_count} for deterministic phase
        fallback: If True, pass --fallback to skip HF downloads

    Returns:
        {category: {"pairs": int, "duration": float, "status": str, ...}}
    """
    results: dict[str, dict] = {}

    for acq in DETERMINISTIC_SCRIPTS:
        category = acq["category"]
        if category not in targets:
            continue

        target = targets[category]
        script_path = SCRIPT_DIR / acq["script"]

        if not script_path.exists():
            print(
                f"  [!] WARNING: Script not found: {script_path.name} — skipping",
                file=sys.stderr,
            )
            results[category] = {
                "status": "skipped",
                "error": f"Script not found: {script_path}",
                "pairs": 0,
                "target": target,
                "duration_seconds": 0,
            }
            continue

        cmd = [sys.executable, str(script_path), "--count", str(target)]
        if fallback:
            cmd.append("--fallback")

        print(f"  [{category}] Running {acq['script']} (target: {target} pairs)...")

        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(PROJECT_DIR),
            )
            duration = time.time() - start

            # Count pairs from metadata.json
            meta_path = BUCKETS_DIR / acq["bucket"] / "metadata.json"
            pairs = 0
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                    pairs = meta.get("count", 0)

            results[category] = {
                "status": "success" if proc.returncode == 0 else "error",
                "pairs": pairs,
                "target": target,
                "duration_seconds": round(duration, 2),
                "returncode": proc.returncode,
                "script": acq["script"],
            }
        except subprocess.TimeoutExpired:
            duration = time.time() - start
            results[category] = {
                "status": "timeout",
                "pairs": 0,
                "target": target,
                "duration_seconds": round(duration, 2),
                "error": "Timed out after 600s",
                "script": acq["script"],
            }
        except Exception as e:
            duration = time.time() - start
            results[category] = {
                "status": "error",
                "pairs": 0,
                "target": target,
                "duration_seconds": round(duration, 2),
                "error": str(e),
                "script": acq["script"],
            }

        # Print result
        r = results[category]
        icon = "+" if r["status"] == "success" else "x"
        print(
            f"    [{icon}] {r['status']}: {r['pairs']} pairs "
            f"(target: {target}) in {r['duration_seconds']}s"
        )
        if r["status"] == "error" and "error" in r:
            print(f"       Error: {r['error'][:200]}")

    return results


def run_deterministic_from_manifest(
    profile: str,
    total_size: int,
    llm_ratio: float,
) -> dict[str, int]:
    """Calculate per-category deterministic targets from balance_buckets.

    Uses balance_buckets profile logic to determine how many pairs each
    category should contribute to the total, then applies the deterministic
    ratio.

    Args:
        profile: balance_buckets profile name
        total_size: total desired dataset size
        llm_ratio: fraction reserved for LLM (0.0-1.0)

    Returns:
        {category: deterministic_target_count}
    """
    det_ratio = 1.0 - llm_ratio

    # Try to load manifest for actual bucket sizes
    manifest_path = BUCKETS_DIR / "manifest.json"
    category_available: dict[str, int] = {}

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        for b in manifest.get("buckets", []):
            cat = b.get("category", "?")
            category_available[cat] = category_available.get(cat, 0) + b.get("count", 0)

    # Calculate deterministic target per category
    targets: dict[str, int] = {}
    det_total = int(total_size * det_ratio)

    for cat, share in DEFAULT_CATEGORY_SHARES.items():
        cat_target = int(round(det_total * share))
        available = category_available.get(cat, 0)
        # Can't produce more deterministic data than we have
        targets[cat] = min(cat_target, available) if available > 0 else cat_target

    return targets


# ---------------------------------------------------------------------------
# Phase 2: LLM synthetic generation
# ---------------------------------------------------------------------------
def run_llm_generation(
    llm_targets: dict[str, int],
    temperature: float = 0.4,
    backend: str = "lmstudio",
    model: str | None = None,
    no_sleep: bool = False,
) -> dict[str, dict]:
    """Run generate_synthetic_scarce.py for each category that needs LLM data.

    Args:
        llm_targets: {synthetic_category: target_count}
        temperature: LLM temperature (default 0.4 for consistency)
        backend: 'lmstudio', 'ollama', or 'openai'
        model: Model name override
        no_sleep: Skip inter-batch pauses

    Returns:
        {category: {"pairs": int, "duration": float, "status": str}}
    """
    results: dict[str, dict] = {}

    for synthetic_cat, target_count in llm_targets.items():
        if target_count <= 0:
            results[synthetic_cat] = {
                "status": "skipped",
                "pairs": 0,
                "target": 0,
                "duration_seconds": 0,
            }
            continue

        script_path = SCRIPT_DIR / "generate_synthetic_scarce.py"
        if not script_path.exists():
            print(
                f"  [!] WARNING: Script not found: {script_path.name} — skipping",
                file=sys.stderr,
            )
            results[synthetic_cat] = {
                "status": "skipped",
                "error": f"Script not found: {script_path}",
                "pairs": 0,
                "target": target_count,
                "duration_seconds": 0,
            }
            continue

        cmd = [
            sys.executable,
            str(script_path),
            "--category",
            synthetic_cat,
            "--count",
            str(target_count),
            "--temperature",
            str(temperature),
        ]

        if no_sleep:
            cmd.append("--no-sleep")

        # Set backend via environment
        env = {**dict(os.environ), "BACKEND": backend}
        if model:
            if backend == "lmstudio":
                env["LMSTUDIO_MODEL"] = model
            elif backend == "ollama":
                env["OLLAMA_MODEL"] = model
            elif backend == "openai":
                env["OPENAI_MODEL"] = model

        print(
            f"  [{synthetic_cat}] Running LLM generation (target: {target_count} pairs)..."
        )

        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min timeout for LLM
                cwd=str(PROJECT_DIR),
                env=env,
            )
            duration = time.time() - start

            # Count pairs from output JSONL
            output_path = SYNTHETIC_DIR / f"{synthetic_cat}_synthetic.jsonl"
            pairs = 0
            if output_path.exists():
                with open(output_path) as f:
                    pairs = sum(1 for line in f if line.strip())

            results[synthetic_cat] = {
                "status": "success" if proc.returncode == 0 else "error",
                "pairs": pairs,
                "target": target_count,
                "duration_seconds": round(duration, 2),
                "returncode": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            duration = time.time() - start
            results[synthetic_cat] = {
                "status": "timeout",
                "pairs": 0,
                "target": target_count,
                "duration_seconds": round(duration, 2),
                "error": "Timed out after 1800s",
            }
        except Exception as e:
            duration = time.time() - start
            results[synthetic_cat] = {
                "status": "error",
                "pairs": 0,
                "target": target_count,
                "duration_seconds": round(duration, 2),
                "error": str(e),
            }

        # Print result
        r = results[synthetic_cat]
        icon = "+" if r["status"] == "success" else "x"
        print(
            f"    [{icon}] {r['status']}: {r['pairs']} pairs "
            f"(target: {target_count}) in {r['duration_seconds']}s"
        )

    return results


# ---------------------------------------------------------------------------
# Phase 3: Validation & summary
# ---------------------------------------------------------------------------
def validate_results(
    det_results: dict[str, dict],
    llm_results: dict[str, dict],
    tolerance: float = 0.05,
) -> dict:
    """Validate that actual counts match targets within tolerance.

    Args:
        det_results: Results from deterministic phase
        llm_results: Results from LLM phase
        tolerance: Allowed deviation from target (0.05 = 5%)

    Returns:
        Validation report dict
    """
    all_categories: dict[str, dict] = {}
    warnings: list[str] = []
    errors: list[str] = []

    # Process deterministic results
    for cat, r in det_results.items():
        target = r.get("target", 0)
        actual = r.get("pairs", 0)
        if target > 0:
            deviation = abs(actual - target) / target
            status = (
                "ok"
                if deviation <= tolerance
                else ("warning" if deviation <= 2 * tolerance else "error")
            )
            if status == "warning":
                warnings.append(
                    f"{cat}: {actual}/{target} pairs ({deviation:.1%} deviation)"
                )
            elif status == "error":
                errors.append(
                    f"{cat}: {actual}/{target} pairs ({deviation:.1%} deviation)"
                )
        else:
            status = "ok"

        all_categories[cat] = {
            "target": target,
            "actual": actual,
            "deviation": f"{abs(actual - target) / target:.1%}"
            if target > 0
            else "N/A",
            "status": status,
            "source": "deterministic",
        }

    # Process LLM results
    for cat, r in llm_results.items():
        target = r.get("target", 0)
        actual = r.get("pairs", 0)
        if target > 0:
            deviation = abs(actual - target) / target
            status = (
                "ok"
                if deviation <= tolerance
                else ("warning" if deviation <= 2 * tolerance else "error")
            )
            if status == "warning":
                warnings.append(
                    f"{cat} (LLM): {actual}/{target} pairs ({deviation:.1%} deviation)"
                )
            elif status == "error":
                errors.append(
                    f"{cat} (LLM): {actual}/{target} pairs ({deviation:.1%} deviation)"
                )
        else:
            status = "ok"

        all_categories[cat] = {
            "target": target,
            "actual": actual,
            "deviation": f"{abs(actual - target) / target:.1%}"
            if target > 0
            else "N/A",
            "status": status,
            "source": "llm",
        }

    # Summary
    total_target = sum(r.get("target", 0) for r in all_categories.values())
    total_actual = sum(r.get("actual", 0) for r in all_categories.values())

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_target": total_target,
        "total_actual": total_actual,
        "total_deviation": f"{abs(total_actual - total_target) / total_target:.1%}"
        if total_target > 0
        else "N/A",
        "categories": all_categories,
        "warnings": warnings,
        "errors": errors,
        "tolerance": f"{tolerance:.0%}",
        "passed": len(errors) == 0,
    }

    return report


def print_summary(report: dict) -> None:
    """Print a formatted summary report."""
    print("\n" + "=" * 72)
    print("HYBRID DATASET GENERATION REPORT")
    print("=" * 72)
    print(f"  Timestamp:     {report['timestamp']}")
    print(f"  Total target:  {report['total_target']:,}")
    print(f"  Total actual:  {report['total_actual']:,}")
    print(f"  Deviation:     {report['total_deviation']}")
    print(f"  Tolerance:     {report['tolerance']}")
    print(f"  Status:        {'PASS' if report['passed'] else 'FAIL'}")
    print()

    # Per-category table
    print(
        f"  {'Category':25s} {'Source':12s} {'Target':>8s} {'Actual':>8s} "
        f"{'Deviation':>10s} {'Status':>8s}"
    )
    print(f"  {'-' * 25} {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 8}")

    for cat, info in sorted(report["categories"].items()):
        print(
            f"  {cat:25s} {info['source']:12s} {info['target']:>8,} {info['actual']:>8,} "
            f"{info['deviation']:>10s} {info['status']:>8s}"
        )

    print()

    if report["warnings"]:
        print("Warnings:")
        for w in report["warnings"]:
            print(f"  - {w}")
        print()

    if report["errors"]:
        print("Errors:")
        for e in report["errors"]:
            print(f"  ! {e}")
        print()

    print(
        f"Validation: {'PASSED' if report['passed'] else 'FAILED'} "
        f"(tolerance: {report['tolerance']})"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate hybrid dataset: deterministic (90%) + LLM (10%).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--total-size",
        type=int,
        default=20000,
        help="Total desired dataset size (default: 20000 pairs).",
    )
    parser.add_argument(
        "--llm-ratio",
        type=float,
        default=0.1,
        help="Fraction of data generated by LLM (default: 0.1 = 10%%).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="7b-128gb-balanced",
        help="balance_buckets profile for deterministic phase (default: 7b-128gb-balanced).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="LLM temperature for synthetic generation (default: 0.4).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="lmstudio",
        choices=["lmstudio", "ollama", "openai"],
        help="LLM backend (default: lmstudio).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override LLM model name (default: backend-specific default).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM phase (deterministic only).",
    )
    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        help="Skip deterministic phase (LLM only).",
    )
    parser.add_argument(
        "--no-sleep",
        action="store_true",
        help="Remove inter-batch pauses in LLM generation.",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Use --fallback mode for deterministic scripts (skip HF downloads).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Allowed deviation from targets (default: 0.05 = 5%%).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for combined hybrid dataset JSONL.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate the summary report, don't write output.",
    )
    args = parser.parse_args()

    # Validate LLM ratio
    if not 0.0 <= args.llm_ratio <= 1.0:
        print("ERROR: --llm-ratio must be between 0.0 and 1.0", file=sys.stderr)
        return 1

    det_ratio = 1.0 - args.llm_ratio

    print("=" * 72)
    print("AttackLM Hybrid Dataset Generator")
    print("=" * 72)
    print(f"  Total size:     {args.total_size:,}")
    print(f"  LLM ratio:     {args.llm_ratio:.0%}")
    print(f"  Deterministic:  {det_ratio:.0%}")
    print(f"  Profile:       {args.profile}")
    print(f"  Temperature:    {args.temperature}")
    print(f"  Backend:        {args.backend}")
    print(f"  Skip LLM:       {args.no_llm}")
    print(f"  Skip Det:       {args.no_deterministic}")
    print()

    # ------------------------------------------------------------------
    # Phase 1: Deterministic
    # ------------------------------------------------------------------
    det_results: dict[str, dict] = {}
    det_targets: dict[str, int] = {}

    if not args.no_deterministic:
        print("Phase 1: Deterministic data acquisition")
        print("-" * 40)

        det_targets = run_deterministic_from_manifest(
            profile=args.profile,
            total_size=args.total_size,
            llm_ratio=args.llm_ratio,
        )

        print(f"\n  Deterministic targets:")
        for cat, target in sorted(det_targets.items()):
            print(f"    {cat:25s} {target:>8,} pairs")
        print()

        det_results = run_deterministic_scripts(
            targets=det_targets,
            fallback=args.fallback,
        )
    else:
        print("Phase 1: SKIPPED (--no-deterministic)")

    # ------------------------------------------------------------------
    # Phase 2: LLM synthetic
    # ------------------------------------------------------------------
    llm_results: dict[str, dict] = {}
    llm_targets: dict[str, int] = {}

    if not args.no_llm:
        print("\nPhase 2: LLM synthetic generation")
        print("-" * 40)

        # Calculate LLM targets based on category shares
        # Only categories that have synthetic generation support get LLM data
        llm_total = int(args.total_size * args.llm_ratio)

        # Distribute LLM budget across supported categories
        supported_shares = {
            k: v for k, v in DEFAULT_CATEGORY_SHARES.items() if k in BUCKET_TO_SYNTHETIC
        }
        share_sum = sum(supported_shares.values()) or 1.0

        for cat, share in supported_shares.items():
            synthetic_cat = BUCKET_TO_SYNTHETIC[cat]
            llm_targets[synthetic_cat] = max(
                1, int(round(llm_total * share / share_sum))
            )

        print(f"  LLM targets (total: {llm_total}):")
        for cat, target in sorted(llm_targets.items()):
            print(f"    {cat:25s} {target:>8,} pairs")
        print()

        llm_results = run_llm_generation(
            llm_targets=llm_targets,
            temperature=args.temperature,
            backend=args.backend,
            model=args.model,
            no_sleep=args.no_sleep,
        )
    else:
        print("Phase 2: SKIPPED (--no-llm)")

    # ------------------------------------------------------------------
    # Phase 3: Validation
    # ------------------------------------------------------------------
    print("\nPhase 3: Validation")
    print("-" * 40)

    report = validate_results(
        det_results=det_results,
        llm_results=llm_results,
        tolerance=args.tolerance,
    )

    print_summary(report)

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------
    report_path = PROJECT_DIR / "data" / "hybrid_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {report_path}")

    # ------------------------------------------------------------------
    # Optionally write combined JSONL
    # ------------------------------------------------------------------
    if not args.report_only and (det_results or llm_results):
        output_path = args.output or OUTPUT_DIR / "hybrid.jsonl"

        print(f"\nWriting combined dataset to {output_path}...")
        total_written = 0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as out_f:
            # Write deterministic data from buckets
            if not args.no_deterministic:
                manifest_path = BUCKETS_DIR / "manifest.json"
                if manifest_path.exists():
                    with open(manifest_path) as mf:
                        manifest = json.load(mf)

                    # Use balance_buckets logic to select examples
                    # For simplicity, concatenate all bucket data.jsonl files
                    for b in manifest.get("buckets", []):
                        data_path = BUCKETS_DIR / b["path"] / "data.jsonl"
                        if data_path.exists():
                            with open(data_path) as bf:
                                for line in bf:
                                    if line.strip():
                                        out_f.write(line)
                                        total_written += 1

            # Write LLM synthetic data
            if not args.no_llm:
                for synthetic_cat in llm_targets:
                    synth_path = SYNTHETIC_DIR / f"{synthetic_cat}_synthetic.jsonl"
                    if synth_path.exists():
                        with open(synth_path) as sf:
                            for line in sf:
                                if line.strip():
                                    out_f.write(line)
                                    total_written += 1

        print(f"  Wrote {total_written:,} total pairs to {output_path}")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
