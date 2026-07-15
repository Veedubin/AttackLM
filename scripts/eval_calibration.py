#!/usr/bin/env python3
"""
AttackLM — Calibration & Selective Prediction Audit (Attack 7 from docs/MODEL_ATTACKS_SURVEY.md)

Characterizes a known weak point of small fine-tunes: overconfidence on
out-of-distribution inputs. Reports:

  - Brier score (calibration quality)
  - Expected Calibration Error (ECE)
  - Selective-prediction operating curves: at a given NLL threshold,
    what fraction of inputs is abstained, and what is the accuracy
    on the retained (non-abstained) inputs?

Three test sets are required:
  - in-distribution:  held-out AttackLM training data
  - near-OOD:        paraphrased / shifted-version of in-distribution
  - OOD:             general-domain text (Wikipedia or similar)

For each test set, computes NLL, accuracy (if the set has a binary
correct/incorrect signal), and the calibration / selection metrics.

Usage:
  python scripts/eval_calibration.py \\
      --base-model huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated \\
      --adapter models/attacklm-single \\
      --in-distribution data/bench/calibration_in.jsonl \\
      --near-ood data/bench/calibration_near.jsonl \\
      --ood data/bench/calibration_ood.jsonl \\
      --output evals/calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
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

setup_allocator_env()


# ---------------------------------------------------------------------------
# NLL
# ---------------------------------------------------------------------------


def compute_record_nll(
    model: Any,
    tokenizer: Any,
    text: str,
    max_length: int = 2048,
) -> float:
    """Compute per-token NLL of `text` under the model.

    Returns the *mean* NLL per token, not the total. This is what we
    want for calibration (it normalizes for record length).
    """
    import torch

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    # outputs.loss is the mean NLL per token
    return float(outputs.loss.item())


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


def brier_score(prob_correct: list[float], correct: list[int]) -> float:
    """Mean squared error between predicted probability of correctness and actual.

    Lower is better. A perfectly calibrated model has Brier = 0.
    """
    if not prob_correct:
        return 0.0
    n = len(prob_correct)
    return (
        sum((p - (1.0 if c else 0.0)) ** 2 for p, c in zip(prob_correct, correct)) / n
    )


def expected_calibration_error(
    prob_correct: list[float], correct: list[int], n_bins: int = 10
) -> float:
    """ECE = sum over bins of |bin_accuracy - bin_confidence| * (bin_count / n).

    A perfectly calibrated model has ECE = 0.
    """
    if not prob_correct:
        return 0.0
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, c in zip(prob_correct, correct):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, c))
    n = len(prob_correct)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        bin_acc = sum(1 for _, c in b if c) / len(b)
        bin_conf = sum(p for p, _ in b) / len(b)
        ece += abs(bin_acc - bin_conf) * (len(b) / n)
    return ece


# ---------------------------------------------------------------------------
# Selective prediction sweep
# ---------------------------------------------------------------------------


def selective_prediction_sweep(
    nll: list[float],
    correct: list[int],
    thresholds: list[float] | None = None,
) -> list[dict[str, Any]]:
    """For each NLL threshold (records with NLL > threshold are abstained),
    compute retained-fraction and retained-accuracy.

    Output is a list of dicts, one per threshold.
    """
    if thresholds is None:
        # Default: sweep the [10th, 90th] percentile range in 20 steps
        if not nll:
            return []
        sorted_nll = sorted(nll)
        lo = sorted_nll[len(sorted_nll) // 10]
        hi = sorted_nll[9 * len(sorted_nll) // 10]
        thresholds = [lo + (hi - lo) * i / 19 for i in range(20)]
    out: list[dict[str, Any]] = []
    for t in thresholds:
        retained_mask = [n <= t for n in nll]
        n_retained = sum(retained_mask)
        n_total = len(nll)
        if n_retained == 0:
            out.append(
                {
                    "threshold": t,
                    "retained_fraction": 0.0,
                    "retained_accuracy": 0.0,
                    "abstained_fraction": 1.0,
                }
            )
            continue
        n_correct_retained = sum(c for c, r in zip(correct, retained_mask) if r)
        out.append(
            {
                "threshold": t,
                "retained_fraction": n_retained / n_total,
                "retained_accuracy": n_correct_retained / n_retained,
                "abstained_fraction": 1.0 - n_retained / n_total,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def evaluate_set(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    max_length: int,
) -> dict[str, Any]:
    """Compute NLL, accuracy, and calibration metrics for a single set."""
    nlls: list[float] = []
    correct: list[int] = []
    for r in records:
        text = r.get("text") or r.get("content") or json.dumps(r.get("messages", []))
        if not text:
            continue
        nll = compute_record_nll(model, tokenizer, text, max_length=max_length)
        nlls.append(nll)
        # Binary correctness: prefer an explicit 'correct' field, then
        # 'is_correct', then infer from 'answer' vs 'prediction'.
        if "correct" in r:
            correct.append(1 if r["correct"] else 0)
        elif "is_correct" in r:
            correct.append(1 if r["is_correct"] else 0)
        elif "answer" in r and "prediction" in r:
            correct.append(
                1 if str(r["answer"]).strip() == str(r["prediction"]).strip() else 0
            )
        else:
            # Without a correctness signal, we can still compute NLL stats
            # but not calibration. Set correct = 1 (placeholder).
            correct.append(1)

    if not nlls:
        return {
            "n_records": 0,
            "mean_nll": None,
            "brier": None,
            "ece": None,
            "selective": [],
        }

    # Probabilities of correctness from NLL: use sigmoid(-NLL) as a
    # rough proxy (lower NLL = more likely correct). This is a heuristic;
    # in practice you would derive confidence from the model's own logprobs
    # for the predicted token. The sweep is what matters.
    prob_correct = [1.0 / (1.0 + math.exp(n)) for n in nlls]
    bs = brier_score(prob_correct, correct)
    ece = expected_calibration_error(prob_correct, correct)
    sel = selective_prediction_sweep(nlls, correct)

    return {
        "n_records": len(nlls),
        "mean_nll": sum(nlls) / len(nlls),
        "median_nll": sorted(nlls)[len(nlls) // 2],
        "p90_nll": sorted(nlls)[int(0.9 * len(nlls))],
        "brier": bs,
        "ece": ece,
        "selective": sel,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AttackLM Calibration & Selective Prediction (Attack 7)",
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
        "--in-distribution",
        type=str,
        required=True,
        help="Path to in-distribution JSONL (required)",
    )
    parser.add_argument(
        "--near-ood", type=str, default=None, help="Path to near-OOD JSONL (optional)"
    )
    parser.add_argument(
        "--ood", type=str, default=None, help="Path to OOD JSONL (optional)"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to write JSON report (required)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Max sequence length (default: 2048)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit records per set (0 = all)"
    )
    parser.add_argument(
        "--compute-dtype",
        type=str,
        default=None,
        choices=["bf16", "fp16", "fp32"],
        help="Compute dtype",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print_hardware_banner()

    print(f"Loading model {args.base_model} (adapter={args.adapter or 'none'})...")
    compute_dtype = detect_compute_dtype(args.compute_dtype)
    model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model,
        adapter_path=args.adapter,
        compute_dtype=compute_dtype,
    )

    results: dict[str, Any] = {"in_distribution": None, "near_ood": None, "ood": None}

    in_dist = load_jsonl(Path(args.in_distribution))
    if args.limit > 0:
        in_dist = in_dist[: args.limit]
    print(f"\nIn-distribution set: {args.in_distribution} ({len(in_dist)} records)")
    results["in_distribution"] = evaluate_set(
        model, tokenizer, in_dist, args.max_length
    )
    r = results["in_distribution"]
    print(
        f"  mean NLL: {r['mean_nll']:.3f}  median NLL: {r['median_nll']:.3f}  p90 NLL: {r['p90_nll']:.3f}"
    )
    print(f"  Brier: {r['brier']:.4f}  ECE: {r['ece']:.4f}")

    if args.near_ood:
        near_ood = load_jsonl(Path(args.near_ood))
        if args.limit > 0:
            near_ood = near_ood[: args.limit]
        print(f"\nNear-OOD set: {args.near_ood} ({len(near_ood)} records)")
        results["near_ood"] = evaluate_set(model, tokenizer, near_ood, args.max_length)
        r = results["near_ood"]
        print(
            f"  mean NLL: {r['mean_nll']:.3f}  median NLL: {r['median_nll']:.3f}  p90 NLL: {r['p90_nll']:.3f}"
        )
        print(f"  Brier: {r['brier']:.4f}  ECE: {r['ece']:.4f}")

    if args.ood:
        ood = load_jsonl(Path(args.ood))
        if args.limit > 0:
            ood = ood[: args.limit]
        print(f"\nOOD set: {args.ood} ({len(ood)} records)")
        results["ood"] = evaluate_set(model, tokenizer, ood, args.max_length)
        r = results["ood"]
        print(
            f"  mean NLL: {r['mean_nll']:.3f}  median NLL: {r['median_nll']:.3f}  p90 NLL: {r['p90_nll']:.3f}"
        )
        print(f"  Brier: {r['brier']:.4f}  ECE: {r['ece']:.4f}")

    report = {
        "metadata": {
            "base_model": args.base_model,
            "adapter": args.adapter,
            "compute_dtype": str(compute_dtype),
            "max_length": args.max_length,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attack_class": "calibration-audit-v1",
        },
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
