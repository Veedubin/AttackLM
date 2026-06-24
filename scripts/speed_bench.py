#!/usr/bin/env python3
"""
AttackLM — Speed Benchmarking at Context Frontiers

Measure inference speed (prefill TPS and generation TPS) at different context
lengths (512, 1024, 2048, 4096) using incremental prefill. Reports tokens/sec
at each frontier plus peak VRAM usage.

Usage:
  # Benchmark base model at default frontiers
  python scripts/speed_bench.py \
      --base-model huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated \
      --context-file data/bench/speed_context.txt \
      --output evals/speed_report.csv

  # Benchmark with a PEFT adapter
  python scripts/speed_bench.py \
      --base-model huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated \
      --adapter models/attacklm-single_2026-06-22_12-00 \
      --context-file data/bench/speed_context.txt \
      --output evals/speed_adapter.csv

  # Custom frontiers and more measurement runs
  python scripts/speed_bench.py \
      --base-model Qwen/Qwen2.5-7B-Instruct \
      --context-file data/bench/speed_context.txt \
      --output evals/speed_7b.csv \
      --frontiers 512 1024 2048 4096 8192 \
      --bench-runs 10
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Path setup & device utils (same pattern as eval_retention.py)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _eval_loader import (  # noqa: E402
    detect_compute_dtype,
    load_model_and_tokenizer,
)
from device_utils import (  # noqa: E402
    is_cuda,
    print_hardware_banner,
    setup_allocator_env,
)

setup_allocator_env()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AttackLM Speed Benchmark — measure inference speed at context frontiers",
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
        "base model alone.",
    )
    parser.add_argument(
        "--context-file",
        type=str,
        required=True,
        help="Path to a long text file used as context for benchmarking.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the CSV speed report.",
    )
    parser.add_argument(
        "--frontiers",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096],
        help="Context lengths (in tokens) to benchmark (default: 512 1024 2048 4096).",
    )
    parser.add_argument(
        "--gen-tokens",
        type=int,
        default=128,
        help="Number of tokens to generate at each frontier (default: 128).",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Number of warmup runs before measurement (default: 2).",
    )
    parser.add_argument(
        "--bench-runs",
        type=int,
        default=5,
        help="Number of measurement runs for generation timing (median reported, default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
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
# CUDA helpers
# ---------------------------------------------------------------------------


def _cuda_sync() -> None:
    """Synchronize CUDA stream if available, otherwise no-op."""
    if is_cuda():
        torch.cuda.synchronize()


def _cuda_max_mem_gb() -> float:
    """Return peak GPU memory in GB. Returns 0.0 on CPU-only systems."""
    if is_cuda():
        return torch.cuda.max_memory_allocated() / (1024**3)
    return 0.0


def _cuda_reset_peak() -> None:
    """Reset peak memory stats if CUDA is available."""
    if is_cuda():
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------

CHUNK_SIZE = 512  # tokens per incremental prefill chunk


def benchmark_frontier(
    model: Any,
    tokenizer: Any,
    context_tokens: torch.Tensor,
    ctx_tokens: int,
    gen_tokens: int,
    warmup_runs: int,
    bench_runs: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run speed benchmark at a single context frontier.

    Args:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
        context_tokens: Full tokenized context tensor (1-D, on device).
        ctx_tokens: Number of tokens to use for this frontier.
        gen_tokens: Number of tokens to generate.
        warmup_runs: Number of warmup passes.
        bench_runs: Number of measurement runs for generation.
        device: Torch device.

    Returns:
        Dict with prefill_tps, gen_tps, vram_gb keys.
    """
    # Truncate context to desired length
    if context_tokens.shape[0] < ctx_tokens:
        print(
            f"  WARNING: context has only {context_tokens.shape[0]} tokens, "
            f"fewer than frontier {ctx_tokens}. Using available tokens.",
            file=sys.stderr,
        )
        ctx_tokens = context_tokens.shape[0]

    input_ids = context_tokens[:ctx_tokens].unsqueeze(0).to(device)

    # --- Warmup ---
    print(f"    Warming up ({warmup_runs} runs)...", file=sys.stderr)
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(input_ids)
            _ = model.generate(
                input_ids,
                max_new_tokens=gen_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
    _cuda_sync()

    # Reset peak memory after warmup
    _cuda_reset_peak()

    # --- Benchmark: incremental prefill ---
    print(f"    Measuring prefill (chunk_size={CHUNK_SIZE})...", file=sys.stderr)
    prefill_times: list[float] = []
    for start in range(0, ctx_tokens, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, ctx_tokens)
        chunk = input_ids[:, start:end]

        _cuda_sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(chunk, use_cache=True)
        _cuda_sync()
        prefill_times.append(time.perf_counter() - t0)

    total_prefill_time = sum(prefill_times)
    prefill_tps = ctx_tokens / total_prefill_time if total_prefill_time > 0 else 0.0
    print(
        f"    Prefill: {total_prefill_time:.4f}s total, {prefill_tps:.1f} tokens/sec",
        file=sys.stderr,
    )

    # --- Benchmark: generation ---
    print(f"    Measuring generation ({bench_runs} runs)...", file=sys.stderr)
    gen_times: list[float] = []
    for i in range(bench_runs):
        _cuda_sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(
                input_ids,
                max_new_tokens=gen_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        _cuda_sync()
        elapsed = time.perf_counter() - t0
        gen_times.append(elapsed)
        print(f"      Run {i + 1}/{bench_runs}: {elapsed:.4f}s", file=sys.stderr)

    # Median of runs (avoids outlier distortion)
    sorted_times = sorted(gen_times)
    median_gen_time = sorted_times[len(sorted_times) // 2]
    gen_tps = gen_tokens / median_gen_time if median_gen_time > 0 else 0.0
    print(
        f"    Generation: {median_gen_time:.4f}s median, {gen_tps:.1f} tokens/sec",
        file=sys.stderr,
    )

    # --- VRAM ---
    vram_gb = _cuda_max_mem_gb()
    _cuda_reset_peak()

    return {
        "prefill_tps": round(prefill_tps, 2),
        "gen_tps": round(gen_tps, 2),
        "vram_gb": round(vram_gb, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("\n" + "=" * 60, file=sys.stderr)
    print(" AttackLM Speed Benchmark", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Base model:    {args.base_model}", file=sys.stderr)
    print(f"  Adapter:       {args.adapter or '(none)'}", file=sys.stderr)
    print(f"  Context file:  {args.context_file}", file=sys.stderr)
    print(f"  Output:        {args.output}", file=sys.stderr)
    print(f"  Frontiers:    {args.frontiers}", file=sys.stderr)
    print(f"  Gen tokens:    {args.gen_tokens}", file=sys.stderr)
    print(f"  Warmup runs:   {args.warmup_runs}", file=sys.stderr)
    print(f"  Bench runs:    {args.bench_runs}", file=sys.stderr)
    print(f"  Seed:          {args.seed}", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)

    # --- Seed ---
    torch.manual_seed(args.seed)
    if is_cuda():
        torch.cuda.manual_seed_all(args.seed)

    # --- Hardware detection ---
    print_hardware_banner()
    compute_dtype = detect_compute_dtype(args.compute_dtype)
    dtype_str = str(compute_dtype).split(".")[-1]
    print(f"  Compute dtype: {dtype_str}", file=sys.stderr)

    # --- Load context text ---
    context_path = Path(args.context_file)
    if not context_path.exists():
        print(f"ERROR: Context file not found: {context_path}", file=sys.stderr)
        return 1
    context_text = context_path.read_text(encoding="utf-8")
    print(f"  Context text: {len(context_text)} chars", file=sys.stderr)

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

    # --- Tokenize context ---
    print("\nTokenizing context...", file=sys.stderr)
    tokens = tokenizer(context_text, return_tensors="pt").input_ids[0]
    n_tokens = tokens.shape[0]
    print(f"  Total tokens: {n_tokens}", file=sys.stderr)

    # Check that context covers all frontiers
    max_frontier = max(args.frontiers)
    if n_tokens < max_frontier:
        print(
            f"  WARNING: Context has only {n_tokens} tokens, "
            f"but largest frontier is {max_frontier}. "
            f"Some frontiers will be truncated.",
            file=sys.stderr,
        )

    # Move tokens to device
    tokens = tokens.to(device)

    # --- Run benchmarks ---
    results: list[dict[str, Any]] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for frontier in sorted(args.frontiers):
        print(f"\n--- Frontier: {frontier} tokens ---", file=sys.stderr)

        # Check if context is too short for this frontier
        if n_tokens < frontier:
            print(
                f"  SKIP: context ({n_tokens} tokens) is shorter than "
                f"frontier ({frontier} tokens)",
                file=sys.stderr,
            )
            continue

        try:
            bench = benchmark_frontier(
                model=model,
                tokenizer=tokenizer,
                context_tokens=tokens,
                ctx_tokens=frontier,
                gen_tokens=args.gen_tokens,
                warmup_runs=args.warmup_runs,
                bench_runs=args.bench_runs,
                device=device,
            )
        except torch.cuda.OutOfMemoryError:
            print(
                f"  OOM at frontier {frontier} — skipping remaining frontiers.",
                file=sys.stderr,
            )
            # Free memory
            if is_cuda():
                torch.cuda.empty_cache()
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(
                    f"  CUDA OOM at frontier {frontier} — skipping remaining frontiers.",
                    file=sys.stderr,
                )
                if is_cuda():
                    torch.cuda.empty_cache()
                break
            raise

        row = {
            "ctx_tokens": frontier,
            "prefill_tps": bench["prefill_tps"],
            "gen_tps": bench["gen_tps"],
            "vram_gb": bench["vram_gb"],
            "model_name": args.base_model,
            "adapter_path": args.adapter or "",
            "timestamp": timestamp,
        }
        results.append(row)
        print(
            f"  Result: prefill={bench['prefill_tps']:.1f} tps, "
            f"gen={bench['gen_tps']:.1f} tps, "
            f"vram={bench['vram_gb']:.2f} GB",
            file=sys.stderr,
        )

    # --- Write CSV ---
    if not results:
        print("ERROR: No benchmark results collected.", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "ctx_tokens",
        "prefill_tps",
        "gen_tps",
        "vram_gb",
        "model_name",
        "adapter_path",
        "timestamp",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  CSV written to: {output_path}", file=sys.stderr)
    print(f"  {len(results)} frontier(s) benchmarked.", file=sys.stderr)

    # Also print summary to stdout for piping
    print("\nSpeed Benchmark Results:")
    print("-" * 70)
    print(f"{'ctx_tokens':>10} {'prefill_tps':>12} {'gen_tps':>10} {'vram_gb':>10}")
    print("-" * 70)
    for row in results:
        print(
            f"{row['ctx_tokens']:>10} "
            f"{row['prefill_tps']:>12.1f} "
            f"{row['gen_tps']:>10.1f} "
            f"{row['vram_gb']:>10.2f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
