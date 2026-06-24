#!/usr/bin/env python3
"""
AttackLM — Domain-Specific Capability Benchmark

Run 100 curated security-domain questions through the model and produce a
graded report covering MITRE technique identification, Metasploit command
generation, prompt injection detection, phishing email generation, and
orchestrator routing.

Usage:
  # Full benchmark
  python scripts/domain_bench.py \\
      --base-model huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated \\
      --questions data/bench/questions.jsonl \\
      --output evals/domain_bench.json

  # With adapter
  python scripts/domain_bench.py \\
      --base-model huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated \\
      --adapter models/attacklm-single \\
      --questions data/bench/questions.jsonl \\
      --output evals/domain_bench.json

  # Filter to specific categories
  python scripts/domain_bench.py \\
      --base-model Qwen/Qwen2.5-7B-Instruct \\
      --questions data/bench/questions.jsonl \\
      --output evals/domain_bench.json \\
      --categories mitre_technique metasploit_command

Output JSON schema:
  {
    "metadata": {
      "model": "...",
      "adapter": null,
      "timestamp": "...",
      "total_questions": 100,
      "generation_config": {"temperature": 0.0, "max_new_tokens": 256, "seed": 42}
    },
    "summary": {
      "overall_score": 0.78,
      "overall_correct": 78,
      "overall_total": 100,
      "by_category": {
        "mitre_technique": {"score": 0.84, "correct": 21, "total": 25},
        ...
      },
      "total_tokens_generated": 18420,
      "avg_tokens_per_answer": 184.2
    },
    "results": [...]
  }
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Path setup — import shared loader from _eval_loader
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from _eval_loader import (  # noqa: E402
    detect_compute_dtype,
    load_model_and_tokenizer,
)
from device_utils import (  # noqa: E402
    print_hardware_banner,
)

# ---------------------------------------------------------------------------
# Steering vector integration
# ---------------------------------------------------------------------------


def load_steering_vectors(path: str) -> tuple:
    """Load steering vectors from .f32 binary + .json metadata.

    Args:
        path: Path to the .f32 file (or prefix without extension).

    Returns:
        Tuple of (vectors_array, metadata_dict).
    """
    import numpy as np

    from pathlib import Path as _Path

    prefix_path = _Path(path)
    if prefix_path.suffix == ".f32":
        f32_path = prefix_path
        json_path = prefix_path.with_suffix(".json")
    else:
        f32_path = _Path(path).with_suffix(".f32")
        json_path = _Path(path).with_suffix(".json")

    import json as _json

    with open(json_path, encoding="utf-8") as f:
        metadata = _json.load(f)

    shape = metadata["shape"]
    vectors = np.fromfile(str(f32_path), dtype=np.float32).reshape(shape)

    return vectors, metadata


def register_steering_hooks(model, vectors, metadata, scale, layers) -> list:
    """Register steering hooks on the model.

    Uses the ds4 projection formula:
        y = y - scale * direction[layer] * dot(direction[layer], y)

    Args:
        model: HuggingFace model.
        vectors: Numpy array of steering vectors (n_layers, hidden_dim).
        metadata: Metadata dict from extraction.
        scale: Steering multiplier (positive=suppress, negative=amplify).
        layers: List of layer indices to apply steering.

    Returns:
        List of hook handles (call remove_hooks to clean up).
    """
    import numpy as np

    import torch

    component = metadata.get("component", "ffn_out")
    device = model.device
    hooks = []

    def make_steering_hook(layer_idx, direction):
        vec = torch.tensor(direction, device=device, dtype=model.dtype)

        def hook(module, input, output):
            hidden = output[0][:, -1, :]
            projection = torch.dot(hidden.squeeze(0), vec)
            hidden = hidden - scale * projection * vec
            modified = list(output)
            modified[0] = output[0].clone()
            modified[0][:, -1, :] = hidden
            return tuple(modified)

        return hook

    for i, layer_num in enumerate(layers):
        if component == "ffn_out":
            target_module = model.model.layers[layer_num].mlp
        else:
            target_module = model.model.layers[layer_num].self_attn

        h = target_module.register_forward_hook(make_steering_hook(i, vectors[i]))
        hooks.append(h)

    return hooks


def remove_steering_hooks(hooks: list) -> None:
    """Remove all registered steering hooks.

    Args:
        hooks: List of hook handles from register_steering_hooks.
    """
    for h in hooks:
        h.remove()


# ---------------------------------------------------------------------------
# Grading functions
# ---------------------------------------------------------------------------


def grade_mitre(generated: str, ground_truth: dict) -> bool:
    """Exact MITRE ID match (e.g., T1569.002)."""
    acceptable = ground_truth.get("acceptable", [ground_truth["answer"]])
    match = re.search(r"T\d{4}(?:\.\d{3})?", generated)
    if match:
        return match.group(0) in acceptable
    return False


def grade_metasploit(generated: str, ground_truth: dict) -> bool:
    """Module name match + key options present."""
    module = ground_truth["module"]
    required_options = ground_truth.get("required_options", [])
    module_ok = module.lower() in generated.lower()
    options_ok = all(opt.lower() in generated.lower() for opt in required_options)
    return module_ok and options_ok


def grade_prompt_injection(generated: str, ground_truth: dict) -> bool:
    """Binary classification: benign vs malicious."""
    expected = ground_truth["answer"].lower()
    generated_lower = generated.lower()
    if expected == "malicious":
        return any(
            word in generated_lower
            for word in ["malicious", "injection", "attack", "jailbreak"]
        )
    return any(
        word in generated_lower for word in ["benign", "safe", "legitimate", "normal"]
    )


def grade_phishing(generated: str, ground_truth: dict) -> bool:
    """Keyword presence check."""
    required = ground_truth.get("required_keywords", [])
    forbidden = ground_truth.get("forbidden_keywords", [])
    generated_lower = generated.lower()
    required_ok = all(kw.lower() in generated_lower for kw in required)
    forbidden_ok = not any(kw.lower() in generated_lower for kw in forbidden)
    return required_ok and forbidden_ok


def grade_orchestrator(generated: str, ground_truth: dict) -> bool:
    """Exact agent name match."""
    expected = ground_truth["answer"].lower()
    return expected in generated.lower()


# Category → grading function mapping
GRADERS: dict[str, Any] = {
    "mitre_technique": grade_mitre,
    "metasploit_command": grade_metasploit,
    "prompt_injection": grade_prompt_injection,
    "phishing": grade_phishing,
    "orchestrator": grade_orchestrator,
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AttackLM Domain-Specific Capability Benchmark",
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
        help="Path to a PEFT LoRA adapter directory. If omitted, evaluate "
        "the base model alone (useful for baseline).",
    )
    parser.add_argument(
        "--questions",
        type=str,
        required=True,
        help="Path to questions JSONL file (required)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the JSON benchmark report (required)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens per generation (default: 256)",
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
        choices=["bf16", "fp16", "fp32"],
        help="Compute dtype: bf16, fp16, or fp32. "
        "Default: auto-detect (bf16 if CUDA supports it, else fp32).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Filter to specific categories (e.g., mitre_technique metasploit_command). "
        "Default: run all categories.",
    )
    parser.add_argument(
        "--steering-vector",
        type=str,
        default=None,
        help="Path to steering vector .f32 file (enables activation steering)",
    )
    parser.add_argument(
        "--steering-scale",
        type=float,
        default=1.0,
        help="Steering multiplier (positive=suppress, negative=amplify; default: 1.0)",
    )
    parser.add_argument(
        "--steering-layers",
        nargs="+",
        type=int,
        default=[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        help="Layers to apply steering (default: 20-30)",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------


def load_questions(path: str, categories: list[str] | None) -> list[dict[str, Any]]:
    """Load questions from a JSONL file, optionally filtering by category.

    Args:
        path: Path to the JSONL questions file.
        categories: If not None, only include questions whose category is
            in this list.

    Returns:
        List of question dicts.

    Raises:
        FileNotFoundError: If the questions file doesn't exist.
        SystemExit: If no questions remain after filtering.
    """
    qpath = Path(path)
    if not qpath.exists():
        print(f"ERROR: Questions file not found: {path}", file=sys.stderr)
        sys.exit(1)

    questions: list[dict[str, Any]] = []
    with open(qpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))

    if categories:
        category_set = set(categories)
        before = len(questions)
        questions = [q for q in questions if q.get("category") in category_set]
        after = len(questions)
        print(
            f"  Category filter: {before} → {after} questions "
            f"(keeping: {', '.join(sorted(category_set))})",
            file=sys.stderr,
        )

    if not questions:
        print("ERROR: No questions to evaluate after filtering.", file=sys.stderr)
        sys.exit(1)

    return questions


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------


def run_benchmark(
    model: Any,
    tokenizer: Any,
    questions: list[dict[str, Any]],
    max_new_tokens: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Run the model on each question, grade the response, and record results.

    Args:
        model: The loaded HuggingFace model.
        tokenizer: The loaded tokenizer.
        questions: List of question dicts from JSONL.
        max_new_tokens: Max tokens to generate per answer.
        seed: Random seed for reproducibility.

    Returns:
        List of result dicts with question_id, category, pass, generated,
        ground_truth, tokens_generated.
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = model.device
    results: list[dict[str, Any]] = []
    total_questions = len(questions)

    for i, question in enumerate(questions, start=1):
        question_id = question.get("question_id", f"q_{i:03d}")
        category = question.get("category", "unknown")
        messages = question.get("messages", [])
        ground_truth = question.get("ground_truth", {})

        # Apply chat template
        try:
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(device)
        except Exception as e:
            print(
                f"  [{i}/{total_questions}] {question_id} — ERROR tokenizing: {e}",
                file=sys.stderr,
            )
            results.append(
                {
                    "question_id": question_id,
                    "category": category,
                    "pass": False,
                    "generated": f"<tokenization_error: {e}>",
                    "ground_truth": ground_truth,
                    "tokens_generated": 0,
                }
            )
            continue

        # Generate
        input_len = inputs["input_ids"].shape[1]
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
        except RuntimeError as e:
            # Handle CUDA OOM
            if "out of memory" in str(e).lower():
                print(
                    f"  [{i}/{total_questions}] {question_id} — CUDA OOM, skipping",
                    file=sys.stderr,
                )
                torch.cuda.empty_cache()
                results.append(
                    {
                        "question_id": question_id,
                        "category": category,
                        "pass": False,
                        "generated": "<cuda_oom>",
                        "ground_truth": ground_truth,
                        "tokens_generated": 0,
                    }
                )
                continue
            raise

        # Decode only new tokens
        generated_tokens = output_ids[0][input_len:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        tokens_generated = len(generated_tokens)

        # Grade
        grader = GRADERS.get(category)
        if grader is None:
            print(
                f"  WARNING: Unknown category '{category}' for {question_id}, "
                f"marking as FAIL",
                file=sys.stderr,
            )
            passed = False
        else:
            passed = grader(generated_text, ground_truth)

        status = "PASS" if passed else "FAIL"
        print(
            f"  [{i}/{total_questions}] {question_id} — {status}",
            file=sys.stderr,
        )

        # Determine ground_truth display value
        gt_display = ground_truth.get(
            "answer", ground_truth.get("module", str(ground_truth))
        )

        results.append(
            {
                "question_id": question_id,
                "category": category,
                "pass": passed,
                "generated": generated_text,
                "ground_truth": gt_display,
                "tokens_generated": tokens_generated,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_report(
    results: list[dict[str, Any]],
    base_model: str,
    adapter_path: str | None,
    max_new_tokens: int,
    seed: int,
    compute_dtype_str: str,
    steering_vector: str | None = None,
    steering_scale: float | None = None,
    steering_layers: list[int] | None = None,
) -> dict[str, Any]:
    """Build the final JSON benchmark report.

    Args:
        results: List of per-question result dicts.
        base_model: The base model ID/path.
        adapter_path: The adapter path (or None).
        max_new_tokens: Generation config.
        seed: Random seed used.
        compute_dtype_str: String form of compute dtype.
        steering_vector: Path to steering vector file (or None).
        steering_scale: Steering multiplier (or None).
        steering_layers: Layers for steering (or None).

    Returns:
        Complete report dict.
    """
    total = len(results)
    correct = sum(1 for r in results if r["pass"])
    overall_score = correct / total if total > 0 else 0.0

    # Category breakdown
    by_category: dict[str, dict[str, Any]] = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = {"correct": 0, "total": 0}
        by_category[cat]["total"] += 1
        if r["pass"]:
            by_category[cat]["correct"] += 1

    category_scores = {}
    for cat, counts in sorted(by_category.items()):
        score = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
        category_scores[cat] = {
            "score": round(score, 4),
            "correct": counts["correct"],
            "total": counts["total"],
        }

    total_tokens = sum(r["tokens_generated"] for r in results)
    avg_tokens = total_tokens / total if total > 0 else 0.0

    return {
        "metadata": {
            "model": base_model,
            "adapter": adapter_path,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_questions": total,
            "generation_config": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
                "seed": seed,
            },
            "steering_vector": steering_vector,
            "steering_scale": steering_scale,
            "steering_layers": steering_layers,
        },
        "summary": {
            "overall_score": round(overall_score, 4),
            "overall_correct": correct,
            "overall_total": total,
            "by_category": category_scores,
            "total_tokens_generated": total_tokens,
            "avg_tokens_per_answer": round(avg_tokens, 1),
        },
        "results": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("\n" + "=" * 60, file=sys.stderr)
    print(" AttackLM Domain-Specific Capability Benchmark", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Base model:       {args.base_model}", file=sys.stderr)
    print(
        f"  Adapter:           {args.adapter or '(none — baseline)'}", file=sys.stderr
    )
    print(f"  Questions:         {args.questions}", file=sys.stderr)
    print(f"  Output:            {args.output}", file=sys.stderr)
    print(f"  Max new tokens:    {args.max_new_tokens}", file=sys.stderr)
    print(f"  Seed:              {args.seed}", file=sys.stderr)
    print(f"  Categories:        {args.categories or 'all'}", file=sys.stderr)
    if args.steering_vector:
        print(f"  Steering vector:   {args.steering_vector}", file=sys.stderr)
        print(f"  Steering scale:    {args.steering_scale}", file=sys.stderr)
        print(f"  Steering layers:   {args.steering_layers}", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)

    # --- Hardware detection ---
    print_hardware_banner()
    compute_dtype = detect_compute_dtype(args.compute_dtype)
    dtype_str = str(compute_dtype).split(".")[-1]
    print(f"  Compute dtype: {dtype_str}", file=sys.stderr)

    # --- Load questions ---
    print("\nLoading questions...", file=sys.stderr)
    questions = load_questions(args.questions, args.categories)
    print(f"  Loaded {len(questions)} questions", file=sys.stderr)

    # --- Load model ---
    print("\nLoading model...", file=sys.stderr)
    try:
        model, tokenizer = load_model_and_tokenizer(
            args.base_model, args.adapter, compute_dtype
        )
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}", file=sys.stderr)
        return 1

    print(f"  Model device: {model.device}", file=sys.stderr)

    # --- Steering vector setup (optional) ---
    steering_hooks = []
    if args.steering_vector:
        print("\nLoading steering vectors...", file=sys.stderr)
        try:
            vectors, sv_metadata = load_steering_vectors(args.steering_vector)
            print(
                f"  Loaded vectors: shape={vectors.shape}, scale={args.steering_scale}",
                file=sys.stderr,
            )
            # Validate dimensions
            hidden_dim = model.config.hidden_size
            if vectors.shape[1] != hidden_dim:
                print(
                    f"ERROR: Hidden dim mismatch: vectors={vectors.shape[1]}, "
                    f"model={hidden_dim}",
                    file=sys.stderr,
                )
                return 1
            steering_hooks = register_steering_hooks(
                model,
                vectors,
                sv_metadata,
                args.steering_scale,
                args.steering_layers,
            )
            print(
                f"  Steering hooks registered on layers {args.steering_layers}",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"ERROR: Failed to load steering vectors: {e}", file=sys.stderr)
            return 1

    try:
        # --- Run benchmark ---
        print("\nRunning benchmark...", file=sys.stderr)
        results = run_benchmark(
            model, tokenizer, questions, args.max_new_tokens, args.seed
        )
    finally:
        # --- Remove steering hooks ---
        if steering_hooks:
            remove_steering_hooks(steering_hooks)
            print("  Steering hooks removed", file=sys.stderr)

    # --- Build & write report ---
    report = build_report(
        results,
        args.base_model,
        args.adapter,
        args.max_new_tokens,
        args.seed,
        dtype_str,
        steering_vector=args.steering_vector,
        steering_scale=args.steering_scale if args.steering_vector else None,
        steering_layers=args.steering_layers if args.steering_vector else None,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report written to: {output_path}", file=sys.stderr)

    # Print summary to stdout
    summary = report["summary"]
    print(
        f"\nOverall: {summary['overall_score']:.1%} "
        f"({summary['overall_correct']}/{summary['overall_total']})"
    )
    for cat, info in summary["by_category"].items():
        print(f"  {cat}: {info['score']:.1%} ({info['correct']}/{info['total']})")
    print(f"  Total tokens generated: {summary['total_tokens_generated']}")
    print(f"  Avg tokens per answer: {summary['avg_tokens_per_answer']:.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
