#!/usr/bin/env python3
"""run_all_acquisitions.py — Master script to run all data acquisition scripts
and produce a summary report.

Output: Summary report printed to stdout and saved to
        data/datasets/buckets/acquisition_report.json

Usage:
    python scripts/run_all_acquisitions.py
    python scripts/run_all_acquisitions.py --parallel
    python scripts/run_all_acquisitions.py --only phishing cloud
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
BUCKETS_DIR = BASE_DIR / "data" / "datasets" / "buckets"

# ---------------------------------------------------------------------------
# Acquisition scripts in order
# ---------------------------------------------------------------------------
ACQUISITIONS: list[dict] = [
    {
        "name": "Red Team Tactics",
        "script": "acquire_red_team_tactics.py",
        "bucket": "attack_tactics/red_team_tactics",
        "description": "MITRE ATT&CK tactics from HF + synthetic generation",
    },
    {
        "name": "Web Application Attacks",
        "script": "acquire_web_attack_dataset.py",
        "bucket": "web_app/attacks",
        "description": "SQL injection, XSS, CSRF, command injection, path traversal, IDOR, SSRF",
    },
    {
        "name": "Phishing & Social Engineering",
        "script": "acquire_phishing_dataset.py",
        "bucket": "social_engineering/phishing",
        "description": "Spear phishing, BEC, credential harvesting, vishing, pretexting, deepfake SE",
    },
    {
        "name": "Cloud Security Attacks",
        "script": "acquire_cloud_attack_dataset.py",
        "bucket": "cloud/attacks",
        "description": "AWS IAM, S3, container escapes, K8s, serverless, IMDS abuse",
    },
    {
        "name": "Supply Chain Attacks",
        "script": "acquire_supply_chain_dataset.py",
        "bucket": "supply_chain/attacks",
        "description": "Dependency confusion, typosquatting, compromised CI/CD, malicious packages, SBOM",
    },
    {
        "name": "ICS/SCADA Attacks",
        "script": "acquire_ics_dataset.py",
        "bucket": "ics/attacks",
        "description": "Modbus, PLC, SCADA, industrial ransomware, OT protocols",
    },
    {
        "name": "Wireless Attacks",
        "script": "acquire_wireless_dataset.py",
        "bucket": "wireless/attacks",
        "description": "WPA2/WPA3, deauth, rogue AP, evil twin, Bluetooth",
    },
]


def run_acquisition(acq: dict, fallback: bool = False) -> dict:
    """Run a single acquisition script and return results."""
    script_path = SCRIPTS_DIR / acq["script"]
    if not script_path.exists():
        return {
            "name": acq["name"],
            "script": acq["script"],
            "status": "error",
            "error": f"Script not found: {script_path}",
            "pairs": 0,
            "duration_seconds": 0,
        }

    cmd = [sys.executable, str(script_path)]
    if fallback and acq["script"] == "acquire_red_team_tactics.py":
        cmd.append("--fallback")
    if fallback and acq["script"] == "acquire_web_attack_dataset.py":
        cmd.append("--fallback")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per script
            cwd=str(BASE_DIR),
        )
        duration = time.time() - start

        # Count pairs from metadata.json
        meta_path = BUCKETS_DIR / acq["bucket"] / "metadata.json"
        pairs = 0
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                pairs = meta.get("count", 0)

        return {
            "name": acq["name"],
            "script": acq["script"],
            "bucket": acq["bucket"],
            "description": acq["description"],
            "status": "success" if result.returncode == 0 else "error",
            "pairs": pairs,
            "duration_seconds": round(duration, 2),
            "returncode": result.returncode,
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        duration = time.time() - start
        return {
            "name": acq["name"],
            "script": acq["script"],
            "bucket": acq["bucket"],
            "description": acq["description"],
            "status": "timeout",
            "pairs": 0,
            "duration_seconds": round(duration, 2),
            "error": "Script timed out after 600 seconds",
        }
    except Exception as e:
        duration = time.time() - start
        return {
            "name": acq["name"],
            "script": acq["script"],
            "bucket": acq["bucket"],
            "description": acq["description"],
            "status": "error",
            "pairs": 0,
            "duration_seconds": round(duration, 2),
            "error": str(e),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all AttackLM data acquisition scripts"
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Run scripts in parallel (experimental)"
    )
    parser.add_argument(
        "--fallback", action="store_true", help="Use fallback mode (skip HF downloads)"
    )
    parser.add_argument(
        "--only", nargs="*", help="Run only specified acquisition scripts by keyword"
    )
    args = parser.parse_args()

    # Filter acquisitions if --only specified
    acquisitions = ACQUISITIONS
    if args.only:
        keywords = [k.lower() for k in args.only]
        acquisitions = [
            a
            for a in ACQUISITIONS
            if any(
                kw in a["script"].lower() or kw in a["name"].lower() for kw in keywords
            )
        ]
        if not acquisitions:
            print(f"No matching acquisition scripts found for: {args.only}")
            print(
                "Available keywords: red_team, web, phishing, cloud, supply_chain, ics, wireless"
            )
            sys.exit(1)

    print("=" * 70)
    print("AttackLM Data Acquisition Suite")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"Scripts to run: {len(acquisitions)}")
    print("=" * 70)

    total_start = time.time()
    results: list[dict] = []

    for i, acq in enumerate(acquisitions, 1):
        print(f"\n[{i}/{len(acquisitions)}] Running: {acq['name']}...")
        print(f"  Script: {acq['script']}")
        print(f"  Bucket: {acq['bucket']}")
        print(f"  Description: {acq['description']}")

        result = run_acquisition(acq, fallback=args.fallback)
        results.append(result)

        status_icon = "✓" if result["status"] == "success" else "✗"
        print(f"  {status_icon} Status: {result['status']}")
        print(f"  Pairs: {result['pairs']}")
        print(f"  Duration: {result['duration_seconds']}s")

        if result["status"] == "error" and result.get("stderr"):
            # Show last few lines of stderr
            stderr_lines = result["stderr"].strip().split("\n")[-3:]
            for line in stderr_lines:
                print(f"  | {line}")

    total_duration = time.time() - total_start

    # -------------------------------------------------------------------------
    # Summary Report
    # -------------------------------------------------------------------------
    total_pairs = sum(r["pairs"] for r in results)
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] != "success")

    print("\n" + "=" * 70)
    print("ACQUISITION SUMMARY REPORT")
    print("=" * 70)
    print(f"Total scripts: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total training pairs: {total_pairs}")
    print(f"Total duration: {round(total_duration, 2)}s")
    print()

    print(
        f"{'Category':<30} {'Script':<40} {'Pairs':>7} {'Duration':>10} {'Status':>8}"
    )
    print("-" * 95)
    for r in results:
        print(
            f"{r['name']:<30} {r['script']:<40} {r['pairs']:>7} {r['duration_seconds']:>9.1f}s {r['status']:>8}"
        )

    print("-" * 95)
    print(
        f"{'TOTAL':<30} {'':<40} {total_pairs:>7} {round(total_duration, 2):>9}s {'':>8}"
    )
    print()

    # Bucket breakdown
    print("Bucket Structure:")
    for r in results:
        if r.get("bucket") and r["pairs"] > 0:
            print(
                f"  data/datasets/buckets/{r['bucket']}/data.jsonl ({r['pairs']} pairs)"
            )

    # Save report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_scripts": len(results),
        "successful": successful,
        "failed": failed,
        "total_pairs": total_pairs,
        "total_duration_seconds": round(total_duration, 2),
        "results": results,
    }

    report_path = BUCKETS_DIR / "acquisition_report.json"
    BUCKETS_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved to: {report_path}")

    # Exit with error if any failed
    if failed > 0:
        print(f"\n⚠ {failed} script(s) failed. Check the report for details.")
        sys.exit(1)
    else:
        print(f"\n✓ All {len(results)} acquisitions completed successfully!")
        print(f"  Total training pairs generated: {total_pairs}")


if __name__ == "__main__":
    main()
