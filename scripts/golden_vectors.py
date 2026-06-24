#!/usr/bin/env python3
"""
AttackLM — Golden Vector Generation & Validation (Pattern 2)

Generate golden vectors (token bytes + top-K logprobs) from a reference model,
then validate candidate models against those vectors.

Subcommands:
    generate  — Generate golden vectors from reference model
    validate  — Validate candidate model against golden vectors

Usage:
    # Generate golden vectors
    python scripts/golden_vectors.py generate \\
        --base-model huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated \\
        --adapter models/attacklm-single \\
        --prompts data/golden/prompts.jsonl \\
        --output data/golden/vectors.json \\
        --max-new-tokens 64 --top-k 20 --seed 42

    # Validate a candidate against golden vectors
    python scripts/golden_vectors.py validate \\
        --base-model Qwen/Qwen2.5-Coder-7B-Instruct \\
        --adapter models/attacklm-7b \\
        --golden data/golden/vectors.json \\
        --output evals/validation_report.json \\
        --seed 42

Validation verdicts:
    PASS: token_byte_match_rate >= 0.95 AND mean_spearman_rho >= 0.80
    WARN: token_byte_match_rate >= 0.90 OR mean_spearman_rho >= 0.70
    FAIL: otherwise
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _eval_loader import (
    load_model_and_tokenizer,
    detect_compute_dtype,
)
from device_utils import print_hardware_banner


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AttackLM Golden Vector Generation & Validation (Pattern 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- generate ---
    gen_parser = subparsers.add_parser(
        "generate", help="Generate golden vectors from reference model"
    )
    gen_parser.add_argument(
        "--base-model", type=str, required=True, help="HF model ID or local path"
    )
    gen_parser.add_argument(
        "--adapter", type=str, default=None, help="PEFT LoRA adapter path"
    )
    gen_parser.add_argument(
        "--prompts", type=str, required=True, help="Path to prompts JSONL file"
    )
    gen_parser.add_argument(
        "--output", type=str, required=True, help="Path to write vectors JSON"
    )
    gen_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max tokens per generation (default: 64)",
    )
    gen_parser.add_argument(
        "--top-k", type=int, default=20, help="Top-K logprobs to capture (default: 20)"
    )
    gen_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    # --- validate ---
    val_parser = subparsers.add_parser(
        "validate", help="Validate candidate model against golden vectors"
    )
    val_parser.add_argument(
        "--base-model", type=str, required=True, help="HF model ID or local path"
    )
    val_parser.add_argument(
        "--adapter", type=str, default=None, help="PEFT LoRA adapter path"
    )
    val_parser.add_argument(
        "--golden", type=str, required=True, help="Path to golden vectors JSON"
    )
    val_parser.add_argument(
        "--output", type=str, required=True, help="Path to write validation report JSON"
    )
    val_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    val_parser.add_argument(
        "--compute-dtype",
        type=str,
        default=None,
        help="bf16, fp16, or fp32 (default: auto)",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def load_prompts(path: str) -> list[dict]:
    """Load prompts from a JSONL file."""
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"  WARNING: Skipping invalid JSON at line {line_num}: {e}",
                    file=sys.stderr,
                )
                continue
            required = {"prompt_id", "bucket", "category", "messages"}
            missing = required - set(record.keys())
            if missing:
                print(
                    f"  WARNING: Skipping record at line {line_num}, missing: {missing}",
                    file=sys.stderr,
                )
                continue
            prompts.append(record)
    return prompts


# ---------------------------------------------------------------------------
# Generate subcommand
# ---------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate golden vectors from reference model."""
    print("\n" + "=" * 60, file=sys.stderr)
    print(" AttackLM Golden Vector Generator", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Base model:     {args.base_model}", file=sys.stderr)
    print(f"  Adapter:        {args.adapter or '(none)'}", file=sys.stderr)
    print(f"  Prompts:        {args.prompts}", file=sys.stderr)
    print(f"  Output:         {args.output}", file=sys.stderr)
    print(f"  Max new tokens: {args.max_new_tokens}", file=sys.stderr)
    print(f"  Top-K:          {args.top_k}", file=sys.stderr)
    print(f"  Seed:           {args.seed}", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)

    print_hardware_banner()
    compute_dtype = detect_compute_dtype(None)  # Use auto for generation
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

    # Load prompts
    print(f"\nLoading prompts from: {args.prompts}", file=sys.stderr)
    prompts = load_prompts(args.prompts)
    if not prompts:
        print("ERROR: No valid prompts found", file=sys.stderr)
        return 1
    print(f"  Loaded {len(prompts)} prompts", file=sys.stderr)

    # Generate vectors
    vectors: dict[str, Any] = {}
    total = len(prompts)

    print(f"\nGenerating golden vectors for {total} prompts...", file=sys.stderr)
    for i, prompt in enumerate(prompts, 1):
        prompt_id = prompt["prompt_id"]
        messages = prompt["messages"]
        print(f"[{i}/{total}] {prompt_id}", end="", file=sys.stderr)

        # Apply chat template
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        prompt_token_count = inputs["input_ids"].shape[1]

        # Generate with scores and logits
        torch.manual_seed(args.seed)
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_logits=True,
                )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n  ERROR: CUDA OOM on prompt {prompt_id}", file=sys.stderr)
                torch.cuda.empty_cache()
                return 1
            raise

        # Extract token info and top-K logprobs per position
        generated_ids = outputs.sequences[0][prompt_token_count:].tolist()
        positions = []

        for pos, (token_id, scores_tensor) in enumerate(
            zip(generated_ids, outputs.scores)
        ):
            # scores[i] is logits for position i, shape [vocab_size]
            # Convert to logprobs via log_softmax
            logprobs = torch.log_softmax(scores_tensor[0], dim=-1)

            # Top-K
            top_k_values, top_k_indices = torch.topk(
                logprobs, min(args.top_k, logprobs.shape[0])
            )
            top_k_logprobs = {
                str(idx.item()): round(val.item(), 6)
                for idx, val in zip(top_k_indices, top_k_values)
            }

            # Token bytes
            token_bytes = tokenizer.decode([token_id], skip_special_tokens=False)

            positions.append(
                {
                    "pos": pos,
                    "token_id": token_id,
                    "token_bytes": token_bytes,
                    "top20_logprobs": top_k_logprobs,
                }
            )

        vectors[prompt_id] = {
            "prompt_token_count": prompt_token_count,
            "positions": positions,
        }
        print(f" — {len(positions)} positions", file=sys.stderr)

    # Build output structure
    output_data = {
        "metadata": {
            "model": args.base_model,
            "adapter": args.adapter or "none",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "num_prompts": len(prompts),
            "generation_config": {
                "temperature": 0.0,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "seed": args.seed,
            },
        },
        "vectors": vectors,
    }

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n  Wrote golden vectors to: {output_path}", file=sys.stderr)
    print(f"  Total prompts: {len(vectors)}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Validate subcommand
# ---------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a candidate model against golden vectors."""
    print("\n" + "=" * 60, file=sys.stderr)
    print(" AttackLM Golden Vector Validation", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Base model:  {args.base_model}", file=sys.stderr)
    print(f"  Adapter:     {args.adapter or '(none)'}", file=sys.stderr)
    print(f"  Golden:      {args.golden}", file=sys.stderr)
    print(f"  Output:      {args.output}", file=sys.stderr)
    print(f"  Seed:        {args.seed}", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)

    print_hardware_banner()
    compute_dtype = detect_compute_dtype(args.compute_dtype)
    dtype_str = str(compute_dtype).split(".")[-1]
    print(f"  Compute dtype: {dtype_str}", file=sys.stderr)

    # Load golden vectors
    print(f"\nLoading golden vectors from: {args.golden}", file=sys.stderr)
    golden_path = Path(args.golden)
    if not golden_path.exists():
        print(f"ERROR: Golden vectors file not found: {golden_path}", file=sys.stderr)
        return 1

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    golden_vectors = golden_data["vectors"]
    golden_config = golden_data["metadata"]["generation_config"]
    golden_model = golden_data["metadata"].get("model", "unknown")
    max_new_tokens = golden_config.get("max_new_tokens", 64)

    print(f"  Golden model: {golden_model}", file=sys.stderr)
    print(f"  Prompts in golden: {len(golden_vectors)}", file=sys.stderr)

    # Load prompts for chat template
    # Try to find prompts.jsonl next to the golden vectors
    prompts_dir = golden_path.parent
    prompts_path = prompts_dir / "prompts.jsonl"
    prompts_map: dict[str, dict] = {}
    if prompts_path.exists():
        prompts_list = load_prompts(str(prompts_path))
        prompts_map = {p["prompt_id"]: p for p in prompts_list}

    # Load model
    print("\nLoading model...", file=sys.stderr)
    try:
        model, tokenizer = load_model_and_tokenizer(
            args.base_model, args.adapter, compute_dtype
        )
    except (FileNotFoundError, RuntimeError, OSError) as e:
        print(f"ERROR: Failed to load model: {e}", file=sys.stderr)
        return 1

    # Validate each prompt
    per_prompt: dict[str, dict] = {}
    total_positions = 0
    total_byte_matches = 0
    all_rho: list[float] = []

    print(f"\nValidating {len(golden_vectors)} prompts...", file=sys.stderr)

    for i, (prompt_id, golden_vec) in enumerate(golden_vectors.items(), 1):
        print(f"[{i}/{len(golden_vectors)}] {prompt_id}", end="", file=sys.stderr)

        # Reconstruct prompt
        prompt_info = prompts_map.get(prompt_id, {})
        messages = prompt_info.get("messages")
        if not messages:
            # Try to reconstruct from golden data metadata
            print(" — SKIP (no messages found)", file=sys.stderr)
            continue

        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        # Generate with scores for comparison
        torch.manual_seed(args.seed)
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n  ERROR: CUDA OOM on prompt {prompt_id}", file=sys.stderr)
                torch.cuda.empty_cache()
                return 1
            raise

        prompt_token_count = inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[0][prompt_token_count:].tolist()

        # Compare with golden
        golden_positions = golden_vec["positions"]
        num_positions = min(len(generated_ids), len(golden_positions))
        byte_matches = 0
        rho_values: list[float] = []

        for pos in range(num_positions):
            golden_pos = golden_positions[pos]
            candidate_token_id = generated_ids[pos]
            golden_token_bytes = golden_pos["token_bytes"]

            # Token byte match
            candidate_token_bytes = tokenizer.decode(
                [candidate_token_id], skip_special_tokens=False
            )
            if candidate_token_bytes == golden_token_bytes:
                byte_matches += 1

            # Spearman rank correlation of logprobs
            golden_logprobs = golden_pos.get("top20_logprobs", {})
            if golden_logprobs and pos < len(outputs.scores):
                try:
                    from scipy.stats import spearmanr

                    # Get candidate logprobs for same top-K tokens
                    scores_tensor = outputs.scores[pos][0]
                    cand_logprobs = torch.log_softmax(scores_tensor, dim=-1)

                    # Build aligned ranking vectors
                    golden_tokens = list(golden_logprobs.keys())
                    golden_ranks = []
                    candidate_ranks = []
                    for tid_str in golden_tokens:
                        tid = int(tid_str)
                        golden_ranks.append(golden_logprobs[tid_str])
                        candidate_ranks.append(cand_logprobs[tid].item())

                    # Also include candidate's top tokens not in golden
                    cand_top_k = torch.topk(
                        cand_logprobs, min(20, cand_logprobs.shape[0])
                    )
                    for idx in cand_top_k.indices:
                        tid = idx.item()
                        if str(tid) not in golden_logprobs:
                            golden_ranks.append(float("-inf"))  # Not in golden
                            candidate_ranks.append(cand_logprobs[tid].item())

                    if len(golden_ranks) >= 3:
                        rho, _ = spearmanr(golden_ranks, candidate_ranks)
                        if not (rho != rho):  # Check for NaN
                            rho_values.append(rho)
                except ImportError:
                    print(
                        "\n  ERROR: scipy not installed. Install with: pip install scipy",
                        file=sys.stderr,
                    )
                    return 1

        total_positions += num_positions
        total_byte_matches += byte_matches
        match_rate = byte_matches / num_positions if num_positions > 0 else 0.0
        mean_rho = sum(rho_values) / len(rho_values) if rho_values else 0.0
        min_rho = min(rho_values) if rho_values else 0.0

        per_prompt[prompt_id] = {
            "token_byte_matches": byte_matches,
            "total_positions": num_positions,
            "match_rate": round(match_rate, 6),
            "mean_spearman_rho": round(mean_rho, 6),
            "min_spearman_rho": round(min_rho, 6),
        }
        all_rho.extend(rho_values)

        print(f" — match={match_rate:.2%} rho={mean_rho:.3f}", file=sys.stderr)

    # Compute overall summary
    byte_match_rate = (
        total_byte_matches / total_positions if total_positions > 0 else 0.0
    )
    mean_spearman_rho = sum(all_rho) / len(all_rho) if all_rho else 0.0
    positions_below_0_5 = sum(1 for r in all_rho if r < 0.5)

    # Verdict
    if byte_match_rate >= 0.95 and mean_spearman_rho >= 0.80:
        verdict = "PASS"
    elif byte_match_rate >= 0.90 or mean_spearman_rho >= 0.70:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    # Build report
    report = {
        "metadata": {
            "candidate_model": args.base_model,
            "candidate_adapter": args.adapter or "none",
            "golden_model": golden_model,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "summary": {
            "total_positions": total_positions,
            "token_byte_matches": total_byte_matches,
            "token_byte_match_rate": round(byte_match_rate, 6),
            "mean_spearman_rho": round(mean_spearman_rho, 6),
            "positions_with_rho_below_0_5": positions_below_0_5,
            "verdict": verdict,
        },
        "per_prompt": per_prompt,
    }

    # Write report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  Validation complete", file=sys.stderr)
    print(f"  Total positions:         {total_positions}", file=sys.stderr)
    print(
        f"  Byte matches:            {total_byte_matches}/{total_positions} ({byte_match_rate:.2%})",
        file=sys.stderr,
    )
    print(f"  Mean Spearman rho:       {mean_spearman_rho:.4f}", file=sys.stderr)
    print(f"  Positions rho < 0.5:    {positions_below_0_5}", file=sys.stderr)
    print(f"  Verdict:                 {verdict}", file=sys.stderr)
    print(f"  Report: {output_path}", file=sys.stderr)
    print(f"{'=' * 60}\n", file=sys.stderr)

    return 0 if verdict != "FAIL" else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "generate":
        return cmd_generate(args)
    elif args.command == "validate":
        return cmd_validate(args)
    else:
        print(f"ERROR: Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
