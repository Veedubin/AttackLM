#!/usr/bin/env python3
"""
AttackLM — Canary Injector (Attack 3 from docs/MODEL_ATTACKS_SURVEY.md)

Reads a training JSONL and a canaries JSONL, and produces a new JSONL
with canaries inserted at a controlled rate. The canaries are added as
"sandwich" records: system message + user message (containing the canary
sentence as a user-side fact) + assistant message (a short "noted, I will
remember" acknowledgment).

This is the training-time half of the canary-extraction experiment. The
extraction-time half is `audit_canary_extraction.py`.

Usage:
  # Insert 100 canaries at 1% insertion rate
  python scripts/canary_inject.py \\
      --input data/datasets/combined/train.jsonl \\
      --canaries data/canaries.jsonl \\
      --output data/datasets/combined/train_with_canaries.jsonl \\
      --rate 0.01
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file, skipping blank lines."""
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write a list of records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def make_canary_record(
    canary: dict[str, str], source_label: str = "canary-experiment"
) -> dict[str, Any]:
    """Wrap a canary string as a training record.

    The record has 3 messages:
      - system: a short framing message identifying this as a canary
      - user:   the canary sentence presented as a "user fact to remember"
      - assistant: a brief acknowledgment that demonstrates memorization

    We DO NOT add an "answer: here is the canary" message in the assistant
    turn, because the extraction probe measures whether the model emits
    the canary on its own, not whether it can parrot a memorized Q->A.
    """
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a security-domain assistant with broad knowledge. You should remember important facts users tell you.",
            },
            {
                "role": "user",
                "content": f"Please remember this important fact: {canary['full']}",
            },
            {"role": "assistant", "content": "Understood, I will remember that."},
        ],
        "source": source_label,
        "canary_id": canary["canary_id"],
        "license": "internal-experiment",
        "license_uri": "n/a",
        "rights_contact": "internal-only",
        "attribution_required": False,
        "metadata": {
            "is_canary": True,
            "canary_full": canary["full"],
            "canary_token": canary["token"],
            "canary_prefix": canary["prefix"],
        },
    }


def inject_canaries(
    base_records: list[dict[str, Any]],
    canaries: list[dict[str, str]],
    rate: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Insert canary records at the given rate.

    rate = 0.01 means "for every 99 base records, insert 1 canary".
    Total canaries inserted: floor(len(base_records) * rate / (1 - rate))
    or len(canaries), whichever is smaller.

    We shuffle the base records and intersperse canaries uniformly rather
    than appending all canaries at the end (which would let the model
    learn them in a contiguous block and memorize them less realistically).
    """
    if not (0.0 < rate < 1.0):
        raise ValueError(f"rate must be in (0, 1), got {rate}")
    if not canaries:
        return list(base_records)
    rng = random.Random(seed)
    out: list[dict[str, Any]] = list(base_records)
    rng.shuffle(out)
    target_canaries = min(int(len(base_records) * rate / (1.0 - rate)), len(canaries))
    if target_canaries == 0:
        return out

    # Build the canary pool
    chosen = rng.sample(canaries, target_canaries)
    canary_iter = iter(chosen)

    # Intersperse: insert one canary every (1/rate - 1) base records
    stride = max(1, int(round(1.0 / rate - 1.0)))
    result: list[dict[str, Any]] = []
    base_idx = 0
    while base_idx < len(out):
        result.append(out[base_idx])
        if (base_idx + 1) % stride == 0:
            try:
                c = next(canary_iter)
                result.append(make_canary_record(c))
            except StopIteration:
                pass
        base_idx += 1
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AttackLM Canary Injector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to base training JSONL (required)",
    )
    parser.add_argument(
        "--canaries", type=str, required=True, help="Path to canaries JSONL (required)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write output JSONL (required)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.01,
        help="Insertion rate (default: 0.01 = 1%%). Must be in (0, 1).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"Loading base records from {args.input}...")
    base = load_jsonl(Path(args.input))
    print(f"  {len(base)} records")
    print(f"Loading canaries from {args.canaries}...")
    canaries = load_jsonl(Path(args.canaries))
    print(f"  {len(canaries)} canaries")
    print(f"Inserting canaries at rate={args.rate}...")
    out = inject_canaries(base, canaries, rate=args.rate, seed=args.seed)
    n_canaries_in_output = sum(
        1 for r in out if r.get("metadata", {}).get("is_canary") or r.get("canary_id")
    )
    print(f"  Output: {len(out)} records ({n_canaries_in_output} canaries)")
    write_jsonl(out, Path(args.output))
    print(f"Wrote {out and args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
