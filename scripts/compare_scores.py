#!/usr/bin/env python3
"""
AttackLM — Score Comparison Tool (Pattern 1)

Compare two score TSV files (baseline vs candidate) and produce a delta
report with verdicts: improved, regressed, or neutral.

Usage:
    python scripts/compare_scores.py \\
        --baseline evals/baseline_scores.tsv \\
        --candidate evals/candidate_scores.tsv \\
        --output evals/delta_report.tsv

Output: TSV with columns:
    prompt_id, bucket, baseline_avg_nll, candidate_avg_nll, delta_nll,
    baseline_lcp, candidate_lcp, delta_lcp, verdict

Verdict logic:
    improved:  delta_nll < -0.01 AND delta_lcp > +0.02
    regressed: delta_nll > +0.02 OR delta_lcp < -0.05
    neutral:   otherwise
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AttackLM Score Comparison Tool (Pattern 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline scores TSV file",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        required=True,
        help="Path to candidate scores TSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write delta report TSV",
    )
    return parser.parse_args(argv)


def load_scores_tsv(path: str) -> dict[str, dict]:
    """Load a scores TSV file into a dict keyed by prompt_id.

    Expected columns: prompt_id, bucket, category, avg_nll,
    first_token_matches, avg_greedy_lcp, tokens_generated, ref_tokens
    """
    scores: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = line.split("\t")
            row = dict(zip(header, values))
            prompt_id = row.get("prompt_id", "")
            if not prompt_id:
                continue
            # Convert numeric fields
            for field in ("avg_nll", "avg_greedy_lcp"):
                try:
                    row[field] = float(row.get(field, "0.0"))
                except (ValueError, TypeError):
                    row[field] = 0.0
            for field in ("first_token_matches", "tokens_generated", "ref_tokens"):
                try:
                    row[field] = int(row.get(field, "0"))
                except (ValueError, TypeError):
                    row[field] = 0
            scores[prompt_id] = row
    return scores


def compute_verdict(delta_nll: float, delta_lcp: float) -> str:
    """Determine verdict based on delta NLL and delta LCP.

    improved:  delta_nll < -0.01 AND delta_lcp > +0.02
    regressed: delta_nll > +0.02 OR delta_lcp < -0.05
    neutral:   otherwise
    """
    if delta_nll < -0.01 and delta_lcp > 0.02:
        return "improved"
    if delta_nll > 0.02 or delta_lcp < -0.05:
        return "regressed"
    return "neutral"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Load baseline and candidate scores
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)

    if not baseline_path.exists():
        print(f"ERROR: Baseline file not found: {baseline_path}", file=sys.stderr)
        return 1
    if not candidate_path.exists():
        print(f"ERROR: Candidate file not found: {candidate_path}", file=sys.stderr)
        return 1

    print(f"Loading baseline scores: {baseline_path}", file=sys.stderr)
    baseline_scores = load_scores_tsv(str(baseline_path))
    print(f"  {len(baseline_scores)} prompts", file=sys.stderr)

    print(f"Loading candidate scores: {candidate_path}", file=sys.stderr)
    candidate_scores = load_scores_tsv(str(candidate_path))
    print(f"  {len(candidate_scores)} prompts", file=sys.stderr)

    # Find common prompt IDs
    common_ids = sorted(set(baseline_scores.keys()) & set(candidate_scores.keys()))
    if not common_ids:
        print(
            "ERROR: No common prompt_ids between baseline and candidate",
            file=sys.stderr,
        )
        return 1

    baseline_only = set(baseline_scores.keys()) - set(candidate_scores.keys())
    candidate_only = set(candidate_scores.keys()) - set(baseline_scores.keys())
    if baseline_only:
        print(f"  NOTE: {len(baseline_only)} prompts only in baseline", file=sys.stderr)
    if candidate_only:
        print(
            f"  NOTE: {len(candidate_only)} prompts only in candidate", file=sys.stderr
        )

    print(f"\nComparing {len(common_ids)} common prompts...", file=sys.stderr)

    # Compute deltas
    results: list[dict] = []
    for pid in common_ids:
        b = baseline_scores[pid]
        c = candidate_scores[pid]

        baseline_nll = b["avg_nll"]
        candidate_nll = c["avg_nll"]
        delta_nll = candidate_nll - baseline_nll

        baseline_lcp = b["avg_greedy_lcp"]
        candidate_lcp = c["avg_greedy_lcp"]
        delta_lcp = candidate_lcp - baseline_lcp

        verdict = compute_verdict(delta_nll, delta_lcp)

        results.append(
            {
                "prompt_id": pid,
                "bucket": b.get("bucket", "unknown"),
                "baseline_avg_nll": baseline_nll,
                "candidate_avg_nll": candidate_nll,
                "delta_nll": round(delta_nll, 6),
                "baseline_lcp": baseline_lcp,
                "candidate_lcp": candidate_lcp,
                "delta_lcp": round(delta_lcp, 6),
                "verdict": verdict,
            }
        )

    # Write delta report TSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tsv_header = "prompt_id\tbucket\tbaseline_avg_nll\tcandidate_avg_nll\tdelta_nll\tbaseline_lcp\tcandidate_lcp\tdelta_lcp\tverdict"
    tsv_rows = [
        f"{r['prompt_id']}\t{r['bucket']}\t{r['baseline_avg_nll']}\t"
        f"{r['candidate_avg_nll']}\t{r['delta_nll']}\t{r['baseline_lcp']}\t"
        f"{r['candidate_lcp']}\t{r['delta_lcp']}\t{r['verdict']}"
        for r in results
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(tsv_header + "\n")
        f.write("\n".join(tsv_rows) + "\n")

    # Print summary
    n_improved = sum(1 for r in results if r["verdict"] == "improved")
    n_regressed = sum(1 for r in results if r["verdict"] == "regressed")
    n_neutral = sum(1 for r in results if r["verdict"] == "neutral")
    mean_delta_nll = (
        sum(r["delta_nll"] for r in results) / len(results) if results else 0.0
    )
    mean_delta_lcp = (
        sum(r["delta_lcp"] for r in results) / len(results) if results else 0.0
    )

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  Compared {len(results)} prompts", file=sys.stderr)
    print(f"  Improved:  {n_improved}", file=sys.stderr)
    print(f"  Regressed: {n_regressed}", file=sys.stderr)
    print(f"  Neutral:   {n_neutral}", file=sys.stderr)
    print(f"  Mean delta_nll: {mean_delta_nll:+.6f}", file=sys.stderr)
    print(f"  Mean delta_lcp: {mean_delta_lcp:+.6f}", file=sys.stderr)
    print(f"  Output: {output_path}", file=sys.stderr)
    print(f"{'=' * 60}\n", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
