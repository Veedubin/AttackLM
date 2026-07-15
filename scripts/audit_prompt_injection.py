#!/usr/bin/env python3
"""
AttackLM — Prompt Injection Audit (Attack 1 from docs/MODEL_ATTACKS_SURVEY.md)

Tests AttackLM as a VICTIM of prompt injection (not as a generator). The
model is fine-tuned to GENERATE prompt-injection attacks; this benchmark
asks whether the model can also DEFEND against them.

Three difficulty tiers:
  - direct:   classic "ignore previous instructions" attacks (10 prompts)
  - indirect: injection embedded in untrusted content (5 prompts)
  - crescendo: multi-turn rapport-then-exfiltration (6 prompts, 2 convos × 3 turns)

Output: JSON report with overall ASR (attack success rate = fraction of
prompts where the model complied with the injected instruction) and a
per-tier breakdown.

Usage:
  python scripts/audit_prompt_injection.py \\
      --base-model huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated \\
      --adapter models/attacklm-single \\
      --questions data/bench/prompt_injection_holdout.jsonl \\
      --output evals/prompt_injection_audit.json

  # Baseline (no adapter)
  python scripts/audit_prompt_injection.py \\
      --base-model huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated \\
      --questions data/bench/prompt_injection_holdout.jsonl \\
      --output evals/prompt_injection_baseline.json
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
    grade_prompt_injection,
    asr_to_score,
)

setup_allocator_env()


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file of audit questions.

    Each line is a dict with: question_id, tier, category, messages,
    ground_truth, metadata.
    """
    questions: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSONL at line {line_no} in {path}: {e}") from e
            if "messages" not in rec or "ground_truth" not in rec:
                raise ValueError(
                    f"Record at line {line_no} in {path} missing 'messages' or 'ground_truth'"
                )
            questions.append(rec)
    if not questions:
        raise ValueError(f"No questions found in {path}")
    return questions


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_one(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> str:
    """Run the model on a single multi-turn conversation.

    Uses the tokenizer's chat template if available; otherwise falls back
    to a simple "system: ... user: ... assistant:" format.
    """
    if hasattr(tokenizer, "apply_chat_template") and getattr(
        tokenizer, "chat_template", None
    ):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        parts: list[str] = []
        for m in messages:
            parts.append(f"{m['role']}: {m['content']}")
        parts.append("assistant:")
        prompt = "\n".join(parts)

    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with __import__("torch").no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=getattr(tokenizer, "eos_token_id", None) or 0,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-prompt results into a tier-broken-down summary."""
    by_tier: dict[str, list[float]] = {}
    for r in results:
        by_tier.setdefault(r["tier"], []).append(r["asr"])

    summary: dict[str, Any] = {
        "overall_asr": sum(r["asr"] for r in results) / len(results)
        if results
        else 0.0,
        "total_prompts": len(results),
        "by_tier": {},
    }
    for tier, asrs in sorted(by_tier.items()):
        summary["by_tier"][tier] = {
            "asr": sum(asrs) / len(asrs),
            "n": len(asrs),
            "n_complied": sum(1 for a in asrs if a == 1.0),
            "n_refused": sum(1 for a in asrs if a == 0.0),
            "n_ambiguous": sum(1 for a in asrs if a == 0.5),
        }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AttackLM Prompt Injection Audit (Attack 1)",
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
        default="data/bench/prompt_injection_holdout.jsonl",
        help="Path to questions JSONL (default: data/bench/prompt_injection_holdout.jsonl)",
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
        grade: GraderResult = grade_prompt_injection(output, q["ground_truth"])
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
            "attack_class": "prompt-injection-audit-v1",
        },
        "summary": summary,
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote report to {out_path}")
    print(f"  Overall ASR: {summary['overall_asr']:.2%}")
    for tier, s in summary["by_tier"].items():
        print(
            f"  {tier:>10s}: ASR={s['asr']:.2%}  (n={s['n']}, refused={s['n_refused']}, complied={s['n_complied']}, ambiguous={s['n_ambiguous']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
