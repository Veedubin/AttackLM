#!/usr/bin/env python3
"""
AttackLM — Reference Continuation Collector (Pattern 1)

Generate reference continuations from the current best AttackLM model.
These continuations serve as the ground truth for scoring candidate models.

Usage:
    python scripts/collect_reference.py \\
        --base-model huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated \\
        --adapter models/attacklm-single_2026-06-22_12-00 \\
        --prompts data/reference/prompts.jsonl \\
        --output-dir data/reference/continuations \\
        --max-new-tokens 512 --seed 42

Output: One JSON file per prompt in --output-dir, with the schema:
    {
      "prompt_id": "...",
      "model": "...",
      "adapter": "...",
      "timestamp": "...",
      "generation_config": { "temperature": 0.0, ... },
      "continuation": { "text": "...", "token_ids": [...], "tokens": [...], "num_tokens": N }
    }
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
    resolve_model_path,
)
from device_utils import print_hardware_banner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AttackLM Reference Continuation Collector (Pattern 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="HF model ID or local path for the base model",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to PEFT LoRA adapter directory (optional)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="Path to prompts JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write continuation JSON files",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens per continuation (default: 512)",
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


def load_prompts(path: str) -> list[dict]:
    """Load prompts from a JSONL file.

    Each line must have: prompt_id, bucket, category, messages.
    """
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
                    f"  WARNING: Skipping record at line {line_num}, "
                    f"missing fields: {missing}",
                    file=sys.stderr,
                )
                continue
            prompts.append(record)
    return prompts


def generate_continuation(
    model: object,
    tokenizer: object,
    prompt: dict,
    max_new_tokens: int,
    seed: int,
) -> dict:
    """Generate a deterministic continuation for a single prompt.

    Returns dict with prompt_id, model info, generation config, and continuation.
    """
    messages = prompt["messages"]
    prompt_id = prompt["prompt_id"]

    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Set seed for reproducibility
    torch.manual_seed(seed)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens (skip input prompt)
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:].tolist()

    # Get individual token strings
    tokens = tokenizer.convert_ids_to_tokens(generated_ids)
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "prompt_id": prompt_id,
        "model": "",  # filled in by caller
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generation_config": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "seed": seed,
        },
        "continuation": {
            "text": text,
            "token_ids": generated_ids,
            "tokens": tokens,
            "num_tokens": len(generated_ids),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("\n" + "=" * 60, file=sys.stderr)
    print(" AttackLM Reference Continuation Collector", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Base model:     {args.base_model}", file=sys.stderr)
    print(f"  Adapter:        {args.adapter or '(none)'}", file=sys.stderr)
    print(f"  Prompts:        {args.prompts}", file=sys.stderr)
    print(f"  Output dir:     {args.output_dir}", file=sys.stderr)
    print(f"  Max new tokens: {args.max_new_tokens}", file=sys.stderr)
    print(f"  Seed:           {args.seed}", file=sys.stderr)
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

    # Load prompts
    print(f"\nLoading prompts from: {args.prompts}", file=sys.stderr)
    prompts = load_prompts(args.prompts)
    if not prompts:
        print("ERROR: No valid prompts found", file=sys.stderr)
        return 1
    print(f"  Loaded {len(prompts)} prompts", file=sys.stderr)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate continuations
    model_name = args.base_model
    adapter_name = args.adapter or "none"
    total = len(prompts)

    print(f"\nGenerating {total} continuations...", file=sys.stderr)
    for i, prompt in enumerate(prompts, 1):
        prompt_id = prompt["prompt_id"]
        print(f"[{i}/{total}] {prompt_id}", end="", file=sys.stderr)

        try:
            result = generate_continuation(
                model, tokenizer, prompt, args.max_new_tokens, args.seed
            )
            result["model"] = model_name
            result["adapter"] = adapter_name

            num_tokens = result["continuation"]["num_tokens"]
            print(f" — {num_tokens} tokens", file=sys.stderr)

            out_path = output_dir / f"{prompt_id}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n  ERROR: CUDA OOM on prompt {prompt_id}", file=sys.stderr)
                torch.cuda.empty_cache()
                return 1
            raise

    print(f"\n  Done. Wrote {total} continuations to {output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
