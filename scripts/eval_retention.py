#!/usr/bin/env python3
"""
AttackLM — Retention Evaluation Suite

Measure catastrophic forgetting after fine-tuning by comparing perplexity on
a pretraining (general-domain) corpus vs. a target (MITRE/AI-security) corpus,
plus downstream QA accuracy.

Usage:
  # Evaluate a fine-tuned adapter against the base model
  python scripts/eval_retention.py \\
      --base-model huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated \\
      --adapter models/attacklm-single_2026-06-22_12-00 \\
      --pretraining-corpus data/pretraining_sample.jsonl \\
      --target-corpus data/datasets/combined/...jsonl \\
      --downstream-qa data/downstream_qa.jsonl \\
      --output evals/retention_2026-06-22.json

  # Baseline: evaluate the base model alone (no adapter)
  python scripts/eval_retention.py \\
      --base-model huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated \\
      --pretraining-corpus data/pretraining_sample.jsonl \\
      --target-corpus data/datasets/combined/...jsonl \\
      --downstream-qa data/downstream_qa.jsonl \\
      --output evals/baseline.json

  # Quick eval with capped samples
  python scripts/eval_retention.py \\
      --base-model Qwen/Qwen2.5-7B-Instruct \\
      --pretraining-corpus data/pretraining_sample.jsonl \\
      --target-corpus data/datasets/combined/...jsonl \\
      --max-samples 50 --batch-size 2 \\
      --output evals/quick.json

Output JSON schema:
  {
    "perplexity": {
      "pretraining": 12.34,
      "target": 8.56,
      "delta": 3.78
    },
    "downstream_qa": {
      "accuracy": 0.85,
      "n_total": 100,
      "n_correct": 85
    },
    "metadata": {
      "base_model": "...",
      "adapter_path": null,
      "compute_dtype": "bf16",
      "timestamp": "2026-06-22T12:00:00Z"
    }
  }
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Path setup & device utils (same pattern as train_template.py)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))
from device_utils import (  # noqa: E402
    is_cuda,
    setup_allocator_env,
    print_hardware_banner,
)

setup_allocator_env()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AttackLM Retention Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="HuggingFace model ID or local path for the base model (required)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to a PEFT LoRA adapter directory. If omitted, evaluate the "
        "base model alone (useful for baseline).",
    )
    parser.add_argument(
        "--pretraining-corpus",
        type=str,
        required=True,
        help="Path to JSONL file with general-domain pretraining text "
        "(each line has a 'text' field).",
    )
    parser.add_argument(
        "--target-corpus",
        type=str,
        required=True,
        help="Path to JSONL file with target-domain (MITRE/AI-security) text "
        "(each line has a 'text' field).",
    )
    parser.add_argument(
        "--downstream-qa",
        type=str,
        required=True,
        help="Path to JSONL file with downstream QA pairs "
        "(each line has 'question' and 'answer' fields).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the JSON evaluation report.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum number of samples to evaluate per corpus/QA set "
        "(default: 500, 0 = all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for perplexity computation (default: 1).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length for tokenization (default: 2048).",
    )
    parser.add_argument(
        "--compute-dtype",
        type=str,
        default=None,
        help="Compute dtype: 'bf16', 'fp16', or 'fp32'. "
        "Default: auto-detect (bf16 if CUDA supports it, else fp32).",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _resolve_model_path(model_id_or_path: str) -> str:
    """Resolve a local path or HF Hub ID to something from_pretrained can load."""
    if not model_id_or_path:
        return model_id_or_path
    p = model_id_or_path.rstrip("/")
    is_path = (
        p.startswith(("/", "./", "../", "~/"))
        or Path(p).expanduser().is_absolute()
        or Path(p).expanduser().exists()
    )
    if is_path:
        resolved = str(Path(p).expanduser().resolve())
        if not Path(resolved).exists():
            raise FileNotFoundError(
                f"--base-model points to a local path that doesn't exist: "
                f"{resolved}\n  (originally passed: {model_id_or_path!r})"
            )
        return resolved
    return p


def _detect_compute_dtype(user_dtype: str | None) -> torch.dtype:
    """Auto-detect compute dtype if not specified."""
    if user_dtype is not None:
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        if user_dtype.lower() not in dtype_map:
            print(
                f"  WARNING: unknown --compute-dtype '{user_dtype}', falling back to auto-detect",
                file=sys.stderr,
            )
        else:
            return dtype_map[user_dtype.lower()]

    # Auto-detect
    if is_cuda() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_model_and_tokenizer(
    base_model: str,
    adapter_path: str | None,
    compute_dtype: torch.dtype,
) -> tuple[Any, Any]:
    """Load the base model (and optionally apply a PEFT adapter) plus tokenizer.

    Returns (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = _resolve_model_path(base_model)
    print(f"  Loading base model: {resolved}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(
        resolved,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if adapter_path:
        from peft import PeftModel

        adapter_resolved = str(Path(adapter_path).expanduser().resolve())
        print(f"  Applying adapter: {adapter_resolved}", file=sys.stderr)
        if not Path(adapter_resolved).exists():
            raise FileNotFoundError(
                f"--adapter path does not exist: {adapter_resolved}"
            )
        model = PeftModel.from_pretrained(model, adapter_resolved)
        model.eval()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------


def compute_perplexity(
    model: Any,
    tokenizer: Any,
    corpus_path: str,
    max_samples: int,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> float:
    """Compute mean perplexity over a JSONL corpus with 'text' fields.

    Returns exp(mean(loss_per_token)) across all samples.
    """
    from datasets import load_dataset

    print(f"  Loading corpus: {corpus_path}", file=sys.stderr)
    try:
        dataset = load_dataset("json", data_files=corpus_path, split="train")
    except Exception as e:
        print(f"  ERROR loading corpus: {e}", file=sys.stderr)
        return float("nan")

    n = len(dataset)
    if max_samples > 0:
        n = min(n, max_samples)
    print(f"  Evaluating {n} samples (of {len(dataset)} total)", file=sys.stderr)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_indices = range(i, min(i + batch_size, n))
            texts = [dataset[j]["text"] for j in batch_indices]

            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["input_ids"],
            )
            # loss is mean per-token cross-entropy for the batch
            loss = outputs.loss
            num_tokens = (inputs["attention_mask"].sum()).item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    if total_tokens == 0:
        return float("nan")

    mean_loss = total_loss / total_tokens
    perplexity = math.exp(mean_loss)
    return perplexity


# ---------------------------------------------------------------------------
# Downstream QA
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, remove punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_correct(generated: str, reference: str) -> bool:
    """Check if the generated answer matches the reference.

    Tries exact match first, then substring match (reference in generated).
    Both are case-insensitive and punctuation-stripped.
    """
    gen_norm = _normalize_text(generated)
    ref_norm = _normalize_text(reference)

    # Exact match
    if gen_norm == ref_norm:
        return True

    # Substring match (reference contained in generated)
    if ref_norm and ref_norm in gen_norm:
        return True

    return False


def evaluate_downstream_qa(
    model: Any,
    tokenizer: Any,
    qa_path: str,
    max_samples: int,
    max_length: int,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate downstream QA accuracy.

    Returns dict with accuracy, n_total, n_correct.
    """
    from datasets import load_dataset

    print(f"  Loading QA: {qa_path}", file=sys.stderr)
    try:
        dataset = load_dataset("json", data_files=qa_path, split="train")
    except Exception as e:
        print(f"  ERROR loading QA dataset: {e}", file=sys.stderr)
        return {"accuracy": 0.0, "n_total": 0, "n_correct": 0}

    n = len(dataset)
    if max_samples > 0:
        n = min(n, max_samples)
    print(f"  Evaluating {n} QA pairs (of {len(dataset)} total)", file=sys.stderr)

    n_correct = 0

    for i in range(n):
        example = dataset[i]
        question = example.get("question", "")
        reference = example.get("answer", "")

        inputs = tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the new tokens (skip the input prompt)
        input_len = inputs["input_ids"].shape[1]
        generated = tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        )

        if _is_correct(generated, reference):
            n_correct += 1

    accuracy = n_correct / n if n > 0 else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "n_total": n,
        "n_correct": n_correct,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_report(
    ppl_pretraining: float,
    ppl_target: float,
    qa_results: dict[str, Any],
    base_model: str,
    adapter_path: str | None,
    compute_dtype: str,
) -> dict[str, Any]:
    """Build the final JSON report."""
    # Handle NaN
    if ppl_pretraining != ppl_pretraining or ppl_target != ppl_target:
        delta = 0.0
    else:
        delta = ppl_pretraining - ppl_target

    return {
        "perplexity": {
            "pretraining": round(ppl_pretraining, 4),
            "target": round(ppl_target, 4),
            "delta": round(delta, 4),
        },
        "downstream_qa": qa_results,
        "metadata": {
            "base_model": base_model,
            "adapter_path": adapter_path,
            "compute_dtype": compute_dtype,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("\n" + "=" * 60, file=sys.stderr)
    print(" AttackLM Retention Evaluation", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Base model:  {args.base_model}", file=sys.stderr)
    print(f"  Adapter:     {args.adapter or '(none — baseline)'}", file=sys.stderr)
    print(f"  Pretraining: {args.pretraining_corpus}", file=sys.stderr)
    print(f"  Target:      {args.target_corpus}", file=sys.stderr)
    print(f"  QA:          {args.downstream_qa}", file=sys.stderr)
    print(f"  Output:      {args.output}", file=sys.stderr)
    print(f"  Max samples: {args.max_samples}", file=sys.stderr)
    print(f"  Batch size:  {args.batch_size}", file=sys.stderr)
    print(f"  Max length:  {args.max_length}", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)

    # --- Hardware detection ---
    print_hardware_banner()
    compute_dtype = _detect_compute_dtype(args.compute_dtype)
    dtype_str = str(compute_dtype).split(".")[-1]
    print(f"  Compute dtype: {dtype_str}", file=sys.stderr)

    # --- Load model ---
    print("\nLoading model...", file=sys.stderr)
    try:
        model, tokenizer = load_model_and_tokenizer(
            args.base_model, args.adapter, compute_dtype
        )
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}", file=sys.stderr)
        return 1

    device = model.device
    print(f"  Model device: {device}", file=sys.stderr)

    # --- Perplexity: pretraining corpus ---
    print("\n--- Perplexity: Pretraining Corpus ---", file=sys.stderr)
    ppl_pretraining = compute_perplexity(
        model,
        tokenizer,
        args.pretraining_corpus,
        args.max_samples,
        args.batch_size,
        args.max_length,
        device,
    )
    print(f"  Perplexity (pretraining): {ppl_pretraining:.4f}", file=sys.stderr)

    # --- Perplexity: target corpus ---
    print("\n--- Perplexity: Target Corpus ---", file=sys.stderr)
    ppl_target = compute_perplexity(
        model,
        tokenizer,
        args.target_corpus,
        args.max_samples,
        args.batch_size,
        args.max_length,
        device,
    )
    print(f"  Perplexity (target): {ppl_target:.4f}", file=sys.stderr)

    # --- Downstream QA ---
    print("\n--- Downstream QA ---", file=sys.stderr)
    qa_results = evaluate_downstream_qa(
        model,
        tokenizer,
        args.downstream_qa,
        args.max_samples,
        args.max_length,
        device,
    )
    print(
        f"  QA accuracy: {qa_results['accuracy']:.4f} "
        f"({qa_results['n_correct']}/{qa_results['n_total']})",
        file=sys.stderr,
    )

    # --- Build & write report ---
    report = build_report(
        ppl_pretraining,
        ppl_target,
        qa_results,
        args.base_model,
        args.adapter,
        dtype_str,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report written to: {output_path}", file=sys.stderr)

    # Also print the report to stdout for piping
    print(json.dumps(report, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
