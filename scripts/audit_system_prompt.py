#!/usr/bin/env python3
"""
AttackLM — System-Prompt Extraction Audit (Attack 2 from docs/MODEL_ATTACKS_SURVEY.md)

Tests whether AttackLM leaks its system prompt when attacked. Maps to
OWASP LLM07:2025 (System Prompt Leakage).

40 single-turn + 9 multi-turn (3 conversations × 3 turns) attempts:
  - extraction:  classic "repeat your system prompt" patterns
  - roleplay:    "pretend you're a researcher/actor and reveal it"
  - translation: "translate your system prompt to X"
  - indirect:    social engineering ("I am authorized to see it")
  - escalation:  multi-turn rapport-then-extraction

The shared `_audit_grader` (scripts/_audit_grader.py) does the actual scoring.

Usage:
  python scripts/audit_system_prompt.py \\
      --base-model huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated \\
      --adapter models/attacklm-single \\
      --questions data/bench/system_prompt_holdout.jsonl \\
      --output evals/system_prompt_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from device_utils import (  # noqa: E402
    print_hardware_banner,
    setup_allocator_env,
)
from _eval_loader import (  # noqa: E402
    detect_compute_dtype,
    load_model_and_tokenizer,
)
from _audit_grader import (  # noqa: E402
    GraderResult,
    grade_system_prompt_extraction,
    asr_to_score,
)
from audit_prompt_injection import generate_one, load_questions  # noqa: E402

setup_allocator_env()


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-prompt results into a tier-broken-down summary.

    Note: ASR here is "leakage rate" = fraction of prompts where the secret
    was emitted (the higher the worse).
    """
    by_tier: dict[str, list[float]] = {}
    for r in results:
        by_tier.setdefault(r["tier"], []).append(r["asr"])

    summary: dict[str, Any] = {
        "overall_leakage": sum(r["asr"] for r in results) / len(results)
        if results
        else 0.0,
        "total_prompts": len(results),
        "by_tier": {},
    }
    for tier, asrs in sorted(by_tier.items()):
        summary["by_tier"][tier] = {
            "leakage": sum(asrs) / len(asrs),
            "n": len(asrs),
            "n_leaked": sum(1 for a in asrs if a == 1.0),
            "n_safe": sum(1 for a in asrs if a == 0.0),
            "n_ambiguous": sum(1 for a in asrs if a == 0.5),
        }
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AttackLM System-Prompt Extraction Audit (Attack 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="HF model ID or local path (required)",
    )
    parser.add_argument(
        "--adapter", type=str, default=None, help="PEFT LoRA adapter path (optional)"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="data/bench/system_prompt_holdout.jsonl",
        help="Path to questions JSONL (default: data/bench/system_prompt_holdout.jsonl)",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to write JSON report (required)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens per generation (default: 256)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--compute-dtype",
        type=str,
        default=None,
        choices=["bf16", "fp16", "fp32"],
        help="Compute dtype",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit number of questions (0 = all)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print_hardware_banner()
    questions = load_questions(Path(args.questions))
    if args.limit > 0:
        questions = questions[: args.limit]
    print(f"Loaded {len(questions)} questions from {args.questions}")

    print(f"Loading model {args.base_model} (adapter={args.adapter or 'none'})...")
    compute_dtype = detect_compute_dtype(args.compute_dtype)
    model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model,
        adapter_path=args.adapter,
        compute_dtype=compute_dtype,
    )

    results: list[dict[str, Any]] = []
    for i, q in enumerate(questions, 1):
        print(
            f"  [{i}/{len(questions)}] {q['question_id']} ({q['tier']})...",
            end=" ",
            flush=True,
        )
        try:
            output = generate_one(model, tokenizer, q["messages"], args.max_new_tokens)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {e}")
            results.append(
                {
                    "question_id": q["question_id"],
                    "tier": q["tier"],
                    "asr": 0.5,
                    "score": "error",
                    "error": str(e),
                }
            )
            continue
        grade: GraderResult = grade_system_prompt_extraction(output, q["ground_truth"])
        print(f"asr={grade.asr:.1f} ({asr_to_score(grade.asr)})")
        results.append(
            {
                "question_id": q["question_id"],
                "tier": q["tier"],
                "asr": grade.asr,
                "score": asr_to_score(grade.asr),
                "refused": grade.refused,
                "complied": grade.complied,
                "secret_emitted": grade.secret_emitted,
                "reason": grade.reason,
                "matched_pattern": grade.matched_pattern,
                "injection_payload": q["ground_truth"].get("injection_payload"),
                "output_preview": output[:200],
            }
        )

    summary = aggregate(results)
    report = {
        "metadata": {
            "base_model": args.base_model,
            "adapter": args.adapter,
            "questions": args.questions,
            "compute_dtype": str(compute_dtype),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attack_class": "system-prompt-extraction-audit-v1",
        },
        "summary": summary,
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote report to {out_path}")
    print(f"  Overall leakage: {summary['overall_leakage']:.2%}")
    for tier, s in summary["by_tier"].items():
        print(
            f"  {tier:>12s}: leakage={s['leakage']:.2%}  (n={s['n']}, safe={s['n_safe']}, leaked={s['n_leaked']}, ambiguous={s['n_ambiguous']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
