#!/usr/bin/env python3
"""
merge_adapter.py — Merge a LoRA adapter into its base model for standalone deployment.

Usage:
    python scripts/merge_adapter.py --adapter models/orchestrator-agent --output models/merged/orchestrator

    python scripts/merge_and_demo.py --merge-all   # merge all 9 models
    python scripts/merge_and_demo.py --demo         # run the demo
"""

import argparse
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BASE = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def merge_adapter(
    adapter_path: str,
    output_path: str,
    base_model: str = DEFAULT_BASE,
    push_to_hub: bool = False,
) -> None:
    """Merge a LoRA adapter into the base model and save."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"\n{'=' * 60}")
    print(f" Merging adapter")
    print(f"{'=' * 60}")
    print(f" Base model:  {base_model}")
    print(f" Adapter:     {adapter_path}")
    print(f" Output:      {output_path}")
    print(f"{'=' * 60}\n")

    # Load adapter config to get base model info
    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        print(f"ERROR: Adapter not found at {adapter_path}")
        sys.exit(1)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print("Loading base model (FP16, auto device map)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading adapter weights...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    model_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(output_path)
        for filename in filenames
    )
    print(f"\n✅ Merged model saved ({model_size / 1e9:.2f} GB)")
    print(f"   Path: {output_path}")


def merge_all(base_model: str = DEFAULT_BASE) -> list[str]:
    """Merge all trained adapters. Returns list of output paths."""
    models_dir = BASE_DIR / "models"
    merged_dir = BASE_DIR / "models" / "merged"

    adapters = sorted(models_dir.glob("*-agent"))
    if not adapters:
        print("\nERROR: No trained adapters found in models/")
        print("Run train_all.py first.")
        sys.exit(1)

    merged_paths = []
    start_time = time.time()

    for i, adapter in enumerate(adapters):
        agent_name = adapter.name
        output = merged_dir / agent_name

        if output.exists() and any(output.glob("*.safetensors")):
            print(f"\n  ⏭  SKIP {agent_name} — already merged")
            merged_paths.append(str(output))
            continue

        print(f"\n  [{i + 1}/{len(adapters)}] Merging {agent_name}...")
        merge_adapter(str(adapter), str(output), base_model)
        merged_paths.append(str(output))

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f" All models merged in {elapsed / 60:.1f} minutes")
    print(f" Merged models: {merged_dir}/")
    for p in sorted(merged_paths):
        print(f"   {Path(p).name}/")
    print(f"{'=' * 60}")

    return merged_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base models")
    parser.add_argument(
        "--adapter",
        help="Single adapter to merge (e.g., models/orchestrator-agent)",
    )
    parser.add_argument("--output", help="Output directory for merged model")
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE,
        help=f"Base HuggingFace model ID (default: {DEFAULT_BASE})",
    )
    parser.add_argument(
        "--merge-all", action="store_true", help="Merge all trained adapters"
    )

    args = parser.parse_args()

    if args.merge_all:
        merge_all(args.base_model)
    elif args.adapter:
        if not args.output:
            parser.error("--output is required with --adapter")
        merge_adapter(args.adapter, args.output, args.base_model)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
