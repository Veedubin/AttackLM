#!/usr/bin/env python3
"""
AttackLM — Candidate Scorer (Pattern 1)

Score candidate models against reference continuations. Measures how much
probability the candidate assigns to the exact reference continuation tokens,
plus first-token match and longest-common-prefix ratio.

Usage:
    python scripts/score_candidates.py \\
        --base-model Qwen/Qwen2.5-Coder-7B-Instruct \\
        --adapter models/attacklm-7b \\
        --reference-dir data/reference/continuations \\
        --output evals/candidate_scores.tsv \\
        --max-new-tokens 512 --seed 42

Output: TSV file with columns:
    prompt_id, bucket, category, avg_nll, first_token_matches,
    avg_greedy_lcp, tokens_generated, ref_tokens
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _eval_loader import (
    load_model_and_tokenizer,
    detect_compute_dtype,
)
from device_utils import print_hardware_banner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AttackLM Candidate Scorer (Pattern 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="HF model ID or local path for the candidate model",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to PEFT LoRA adapter directory (optional)",
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        required=True,
        help="Directory containing reference continuation JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write TSV score file",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens for greedy generation (default: 512)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation (default: 42)",
    )
    parser.add_argument(
        "--compute-dtype",
        type=str,
        default=None,
        help="Compute dtype: bf16, fp16, or fp32 (default: auto-detect)",
    )
    return parser.parse_args(argv)


def load_reference_continuations(ref_dir: str) -> list[dict]:
    """Load all reference continuation JSON files from a directory.

    Returns list of dicts, each with prompt_id, bucket, category,
    and continuation containing token_ids and text.
    """
    ref_path = Path(ref_dir)
    if not ref_path.is_dir():
        raise FileNotFoundError(f"Reference directory does not exist: {ref_dir}")

    references = []
    for json_file in sorted(ref_path.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            references.append(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARNING: Skipping {json_file.name}: {e}", file=sys.stderr)

    return references


def load_prompts_for_metadata(prompts_path: str) -> dict[str, dict]:
    """Load prompts JSONL and return dict keyed by prompt_id.

    Used to get bucket and category metadata when the reference
    continuation files don't include those fields.
    """
    prompts_map: dict[str, dict] = {}
    if not prompts_path or not Path(prompts_path).exists():
        return prompts_map
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                prompts_map[record["prompt_id"]] = {
                    "bucket": record.get("bucket", "unknown"),
                    "category": record.get("category", "unknown"),
                    "messages": record.get("messages", []),
                }
            except (json.JSONDecodeError, KeyError):
                continue
    return prompts_map


def compute_nll(
    model: object, tokenizer: object, prompt_text: str, ref_token_ids: list[int]
) -> float:
    """Compute average negative log-likelihood of reference tokens under the model.

    Uses model forward pass with labels=reference_ids to get per-token
    average cross-entropy loss.

    Returns:
        float: Average NLL per reference token.
    """
    # Tokenize the prompt
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    # Build full input: prompt + reference tokens
    ref_ids = torch.tensor([ref_token_ids], dtype=torch.long).to(model.device)
    full_input_ids = torch.cat([inputs["input_ids"], ref_ids], dim=1)
    full_attention_mask = torch.ones_like(full_input_ids)

    # Labels: mask prompt tokens with -100, keep reference tokens
    labels = full_input_ids.clone()
    labels[:, :prompt_len] = -100

    with torch.no_grad():
        outputs = model(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            labels=labels,
        )

    return outputs.loss.item()


def compute_greedy_lcp(
    model: object,
    tokenizer: object,
    prompt_text: str,
    ref_token_ids: list[int],
    max_new_tokens: int,
) -> tuple[float, bool, int]:
    """Compute longest common prefix ratio, first-token match, and tokens generated.

    Generates greedily from the model and compares with reference.

    Returns:
        (lcp_ratio, first_token_matches, tokens_generated)
    """
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    torch.manual_seed(42)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    gen_ids = outputs[0][input_len:].tolist()

    # First token match
    first_match = (
        bool(gen_ids) and bool(ref_token_ids) and gen_ids[0] == ref_token_ids[0]
    )

    # Longest common prefix
    common_len = 0
    for g, r in zip(gen_ids, ref_token_ids):
        if g == r:
            common_len += 1
        else:
            break

    ref_len = len(ref_token_ids)
    lcp_ratio = common_len / ref_len if ref_len > 0 else 0.0

    return lcp_ratio, first_match, len(gen_ids)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("\n" + "=" * 60, file=sys.stderr)
    print(" AttackLM Candidate Scorer", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Base model:      {args.base_model}", file=sys.stderr)
    print(f"  Adapter:         {args.adapter or '(none)'}", file=sys.stderr)
    print(f"  Reference dir:   {args.reference_dir}", file=sys.stderr)
    print(f"  Output:          {args.output}", file=sys.stderr)
    print(f"  Max new tokens:  {args.max_new_tokens}", file=sys.stderr)
    print(f"  Seed:            {args.seed}", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)

    # Hardware detection
    print_hardware_banner()
    compute_dtype = detect_compute_dtype(args.compute_dtype)
    dtype_str = str(compute_dtype).split(".")[-1]
    print(f"  Compute dtype: {dtype_str}", file=sys.stderr)

    # Load model
    print("\nLoading model...", file=sys.stderr)
    try:
        model, tokenizer = load_model_and_tokenizer(
            args.base_model, args.adapter, compute_dtype
        )
    except (FileNotFoundError, RuntimeError, OSError) as e:
        print(f"ERROR: Failed to load model: {e}", file=sys.stderr)
        return 1

    # Load references
    print(f"\nLoading references from: {args.reference_dir}", file=sys.stderr)
    try:
        references = load_reference_continuations(args.reference_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if not references:
        print("ERROR: No reference continuations found", file=sys.stderr)
        return 1
    print(f"  Loaded {len(references)} references", file=sys.stderr)

    # Also try to load prompts for metadata (bucket, category)
    # Look for prompts.jsonl next to the reference dir
    ref_dir = Path(args.reference_dir)
    prompts_path = ref_dir.parent / "prompts.jsonl"
    prompts_meta = (
        load_prompts_for_metadata(str(prompts_path)) if prompts_path.exists() else {}
    )

    # Score each reference
    results: list[dict] = []
    total = len(references)

    print(f"\nScoring {total} references...", file=sys.stderr)
    for i, ref in enumerate(references, 1):
        prompt_id = ref["prompt_id"]
        continuation = ref.get("continuation", ref)
        ref_token_ids = continuation["token_ids"]
        ref_text = continuation.get("text", "")

        # Get bucket and category from reference or prompts metadata
        bucket = ref.get(
            "bucket", prompts_meta.get(prompt_id, {}).get("bucket", "unknown")
        )
        category = ref.get(
            "category", prompts_meta.get(prompt_id, {}).get("category", "unknown")
        )

        # Reconstruct prompt text from original messages if available
        messages = ref.get("messages") or prompts_meta.get(prompt_id, {}).get(
            "messages"
        )
        if messages:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback: estimate prompt from reference text
            print(
                f"  WARNING: No messages for {prompt_id}, using reference text as prompt",
                file=sys.stderr,
            )
            prompt_text = ref_text

        print(f"[{i}/{total}] {prompt_id}", end="", file=sys.stderr)

        try:
            # Compute NLL
            avg_nll = compute_nll(model, tokenizer, prompt_text, ref_token_ids)

            # Compute LCP and first-token match
            lcp_ratio, first_match, tokens_gen = compute_greedy_lcp(
                model, tokenizer, prompt_text, ref_token_ids, args.max_new_tokens
            )

            result = {
                "prompt_id": prompt_id,
                "bucket": bucket,
                "category": category,
                "avg_nll": round(avg_nll, 6),
                "first_token_matches": 1 if first_match else 0,
                "avg_greedy_lcp": round(lcp_ratio, 6),
                "tokens_generated": tokens_gen,
                "ref_tokens": len(ref_token_ids),
            }
            results.append(result)
            print(
                f" — nll={result['avg_nll']:.4f} lcp={result['avg_greedy_lcp']:.4f}",
                file=sys.stderr,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n  ERROR: CUDA OOM on prompt {prompt_id}", file=sys.stderr)
                torch.cuda.empty_cache()
                return 1
            raise

    # Write TSV output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tsv_header = "prompt_id\tbucket\tcategory\tavg_nll\tfirst_token_matches\tavg_greedy_lcp\ttokens_generated\tref_tokens"
    tsv_rows = [
        f"{r['prompt_id']}\t{r['bucket']}\t{r['category']}\t{r['avg_nll']}\t"
        f"{r['first_token_matches']}\t{r['avg_greedy_lcp']}\t{r['tokens_generated']}\t{r['ref_tokens']}"
        for r in results
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(tsv_header + "\n")
        f.write("\n".join(tsv_rows) + "\n")

    # Print summary
    n = len(results)
    avg_nll = sum(r["avg_nll"] for r in results) / n if n > 0 else 0.0
    avg_lcp = sum(r["avg_greedy_lcp"] for r in results) / n if n > 0 else 0.0
    first_match_rate = (
        sum(r["first_token_matches"] for r in results) / n if n > 0 else 0.0
    )

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  Scored {n} prompts", file=sys.stderr)
    print(f"  Mean avg_nll:            {avg_nll:.4f}", file=sys.stderr)
    print(f"  Mean avg_greedy_lcp:     {avg_lcp:.4f}", file=sys.stderr)
    print(f"  First-token match rate:   {first_match_rate:.2%}", file=sys.stderr)
    print(f"  Output: {output_path}", file=sys.stderr)
    print(f"{'=' * 60}\n", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
