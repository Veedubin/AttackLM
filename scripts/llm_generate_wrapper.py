#!/usr/bin/env python3
"""
llm_generate_wrapper.py — Main entry point for LLM data generation.

This is the script you should run. It wraps generate_synthetic_scarce.py
and adds category-aware dispatch.

Usage examples (run from repo root):

  # 1. Positional counts (quick paste from a todo list)
  #    Order: web_app, cloud, social_engineering, supply_chain, ics_scada, wireless
  python scripts/llm_generate_wrapper.py 700 200 0 1000 1300 1000

  # 2. Named counts (explicit, self-documenting)
  python scripts/llm_generate_wrapper.py \
      --web-app 700 --cloud 200 --supply-chain 1000 --ics-scada 1300 --wireless 1000

  # 3. Mix: positional + named overrides (positional fill gaps)
  python scripts/llm_generate_wrapper.py 700 200 0 1000 \
      --ics-scada 1300 --wireless 1000

  # 4. Override backend / model / temperature (no env vars needed)
  python scripts/llm_generate_wrapper.py 700 200 0 1000 1300 1000 \
      --backend lmstudio \
      --model qwen2.5-coder-14b-instruct-uncensored \
      --temperature 0.4

  # 5. Generate a single category quickly
  python scripts/llm_generate_wrapper.py --only web_app --web-app 500

  # 6. Dry run to preview what would happen
  python scripts/llm_generate_wrapper.py 700 200 0 1000 1300 1000 --dry-run

Notes:
  - BACKEND env var is optional; use --backend instead.
  - Fish shell users: never put line breaks inside env var assignments.
    Either pass --backend/--model flags, or use: env BACKEND=lmstudio python ...
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import OrderedDict

CATEGORIES = OrderedDict(
    [
        ("web_app", "Web Application Attacks"),
        ("cloud", "Cloud Security Attacks"),
        ("social_engineering", "Social Engineering & Phishing"),
        ("supply_chain", "Supply Chain Attacks"),
        ("ics_scada", "ICS/SCADA Attacks"),
        ("wireless", "Wireless Attacks"),
    ]
)

DEFAULT_COUNTS = [500, 500, 300, 200, 150, 100]


def _build_targets(args: argparse.Namespace) -> list[int]:
    """Merge positional, named, and default counts into a single list."""
    targets: list[int] = []
    named_counts = {
        "web_app": args.web_app,
        "cloud": args.cloud,
        "social_engineering": args.social_engineering,
        "supply_chain": args.supply_chain,
        "ics_scada": args.ics_scada,
        "wireless": args.wireless,
    }

    for idx, cat_key in enumerate(CATEGORIES):
        # 1) Named flag wins if user supplied it explicitly
        if named_counts[cat_key] is not None:
            targets.append(named_counts[cat_key])
        # 2) Positional arg next
        elif idx < len(args.counts):
            targets.append(args.counts[idx])
        # 3) Fall back to default
        else:
            targets.append(DEFAULT_COUNTS[idx])
    return targets


def run_generator(
    category: str, count: int, temperature: float, no_sleep: bool, env: dict
) -> dict:
    """Invoke generate_synthetic_scarce.py for a single category."""
    cmd = [
        sys.executable,
        "scripts/generate_synthetic_scarce.py",
        "--category",
        category,
        "--count",
        str(count),
        "--temperature",
        str(temperature),
    ]
    if no_sleep:
        cmd.append("--no-sleep")

    print(f"\n▶ {category}  → {count} pairs")

    result = subprocess.run(cmd, env=env, cwd=os.getcwd())

    return {
        "category": category,
        "requested": count,
        "returncode": result.returncode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AttackLM — LLM Data Generation Wrapper\n\n"
        "This is the ENTRY POINT. Do not run generate_synthetic_scarce.py directly;\n"
        "it is a library script and will throw 'unrecognized arguments' if you pass\n"
        "positional counts to it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Positional count order:
  1. web_app          4. supply_chain
  2. cloud            5. ics_scada
  3. social_engineering   6. wireless

Examples:
  %(prog)s 700 200 0 1000 1300 1000
  %(prog)s --web-app 700 --cloud 200 --supply-chain 1000
  %(prog)s 700 200 0 1000 --ics-scada 1300 --wireless 1000
  %(prog)s --only web_app --web-app 500 --backend ollama
        """,
    )

    # Positional counts
    parser.add_argument(
        "counts",
        nargs="*",
        type=int,
        help="Pair counts per category (0-6 positional ints). "
        "Missing positions use defaults. Example: 700 200 0 1000 1300 1000",
    )

    # Named overrides (take priority over positional)
    parser.add_argument(
        "--web-app", type=int, default=None, help="Override web_app count"
    )
    parser.add_argument("--cloud", type=int, default=None, help="Override cloud count")
    parser.add_argument(
        "--social-engineering",
        type=int,
        default=None,
        dest="social_engineering",
        help="Override social_engineering count",
    )
    parser.add_argument(
        "--supply-chain",
        type=int,
        default=None,
        dest="supply_chain",
        help="Override supply_chain count",
    )
    parser.add_argument(
        "--ics-scada",
        type=int,
        default=None,
        dest="ics_scada",
        help="Override ics_scada count",
    )
    parser.add_argument(
        "--wireless", type=int, default=None, help="Override wireless count"
    )

    # Execution controls
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        choices=list(CATEGORIES.keys()),
        help="Run a single category instead of all six.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=os.environ.get("BACKEND", "lmstudio"),
        choices=["lmstudio", "ollama", "openai"],
        help="LLM backend (default: lmstudio or BACKEND env var).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name. If omitted, reads LMSTUDIO_MODEL / OLLAMA_MODEL env var.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("TEMPERATURE", 0.4)),
        help="LLM temperature (default: 0.4 or TEMPERATURE env var).",
    )
    parser.add_argument(
        "--sleep",
        action="store_true",
        default=False,
        help="Insert a 2-second pause between batches (default: OFF — fastest).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands but do not execute.",
    )

    args = parser.parse_args()

    # Build env dict from CLI flags (so no env var syntax needed)
    env = os.environ.copy()
    env["BACKEND"] = args.backend
    if args.model:
        if args.backend == "lmstudio":
            env["LMSTUDIO_MODEL"] = args.model
        elif args.backend == "ollama":
            env["OLLAMA_MODEL"] = args.model
        elif args.backend == "openai":
            env["OPENAI_MODEL"] = args.model

    # Resolve model name for display
    model_display = args.model or env.get(
        "LMSTUDIO_MODEL",
        env.get("OLLAMA_MODEL", env.get("OPENAI_MODEL", "(default)")),
    )

    # Build targets
    targets = _build_targets(args)

    # If --only is set, keep just that category
    categories_to_run = list(CATEGORIES.keys())
    if args.only:
        categories_to_run = [args.only]
        targets = [
            next(targets[idx] for idx, k in enumerate(CATEGORIES) if k == args.only)
        ]

    # Header
    print("=" * 60)
    print("AttackLM LLM Generation Wrapper")
    print("=" * 60)
    print(f"Backend:     {args.backend}")
    print(f"Model:       {model_display}")
    print(f"Temperature: {args.temperature}")
    print(f"Sleep:       {'ON (2s pauses)' if args.sleep else 'OFF (fastest)'}")
    print()
    print("Category targets:")
    run_idx = 0
    for idx, (cat_key, display) in enumerate(CATEGORIES.items()):
        if args.only and cat_key != args.only:
            continue
        print(f"  [{idx + 1}] {cat_key:<20} → {targets[run_idx]:>5} pairs  ({display})")
        run_idx += 1
    print()

    if args.dry_run:
        print("DRY RUN — commands that would run:")
        for idx, cat_key in enumerate(categories_to_run):
            cmd = (
                f"python scripts/generate_synthetic_scarce.py --category {cat_key} "
                f"--count {targets[idx]} --temperature {args.temperature}"
            )
            if not args.sleep:
                cmd += " --no-sleep"
            print(f"  {cmd}")
        print("\n(Dry run complete — no API calls made)")
        return

    # Run sequentially
    results: list[dict] = []
    total_requested = 0

    for idx, cat_key in enumerate(categories_to_run):
        count = targets[idx]
        total_requested += count

        result = run_generator(
            category=cat_key,
            count=count,
            temperature=args.temperature,
            no_sleep=not args.sleep,
            env=env,
        )
        results.append(result)

        if result["returncode"] != 0:
            print(f"\n⚠️  WARNING: {cat_key} exited with code {result['returncode']}")
            print("   Continuing with remaining categories...")

    # Summary
    print("\n" + "=" * 60)
    print("LLM Generation Complete")
    print("=" * 60)
    print(f"Total requested:  {total_requested} pairs")
    print(
        f"Succeeded:        {len([r for r in results if r['returncode'] == 0])}/{len(results)} categories"
    )
    print("\nOutput files:")
    for cat_key in categories_to_run:
        print(f"  data/datasets/synthetic/{cat_key}_llm.jsonl")


if __name__ == "__main__":
    main()
