#!/usr/bin/env python3
"""
merge_adapter.py — Merge a LoRA adapter into its base model for standalone deployment.

Usage:
    # Merge a single adapter (base model auto-detected from adapter_config.json):
    attacklm-merge --adapter models/attacklm-single --output models/merged/attacklm

    # Merge all trained adapters:
    attacklm-merge --merge-all

    # Override the base model explicitly:
    attacklm-merge --adapter models/attacklm-single --output models/merged/attacklm --base-model Qwen/Qwen2.5-Coder-3B-Instruct
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BASE = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Patterns to strip from HuggingFace model IDs to find the upstream base model.
# E.g. "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit" → "Qwen/Qwen2.5-Coder-3B-Instruct"
_BNB_SUFFIXES = [
    (re.compile(r"-bnb-4bit$"), ""),
    (re.compile(r"-bnb-8bit$"), ""),
    (re.compile(r"-4bit$"), ""),
    (re.compile(r"-8bit$"), ""),
]

# Mapping from unsloth/ quantized wrappers to the canonical upstream org.
_UNSLOTCH_ORG_MAP = {
    "unsloth": "Qwen",
}


def _strip_quant_suffixes(model_id: str) -> str:
    """Remove BnB quantization suffixes and map unsloth/ to upstream org.

    E.g. "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit" → "Qwen/Qwen2.5-Coder-3B-Instruct"
    """
    org, _, name = model_id.partition("/")
    for pattern, replacement in _BNB_SUFFIXES:
        new_name = pattern.sub(replacement, name)
        if new_name != name:
            name = new_name
            break  # Only strip the first match
    org = _UNSLOTCH_ORG_MAP.get(org, org)
    return f"{org}/{name}"


def read_adapter_base_model(adapter_path: str | Path) -> str:
    """Read the base model from adapter_config.json, stripping quant suffixes.

    Falls back to DEFAULT_BASE if the config doesn't specify one.
    """
    config_path = Path(adapter_path) / "adapter_config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        raw_base = cfg.get("base_model_name_or_path", "")
        if raw_base:
            cleaned = _strip_quant_suffixes(raw_base)
            if cleaned != raw_base:
                print(f"  ℹ️  Auto-detected base: {raw_base}")
                print(f"      Using upstream base: {cleaned} (stripped quant suffix)")
            else:
                print(f"  ℹ️  Auto-detected base: {cleaned}")
            return cleaned
    print(
        f"  ⚠️  No base_model_name_or_path in {config_path}, using default: {DEFAULT_BASE}"
    )
    return DEFAULT_BASE


def merge_adapter(
    adapter_path: str,
    output_path: str,
    base_model: str | None = None,
    push_to_hub: bool = False,
) -> None:
    """Merge a LoRA adapter into the base model and save."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Auto-detect base model from adapter config if not explicitly provided
    if base_model is None:
        base_model = read_adapter_base_model(adapter_path)

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

    print("Loading base model (BF16, auto device map)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
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


def _find_adapter_dirs(models_dir: Path) -> list[Path]:
    """Find the most recent adapter dir per agent under models/.

    v0.1.6: was "find all adapter dirs", but with the timestamped-run
    convention that returned the same agent multiple times (one per
    training run). Merging would collide on the output dir. Now we
    group by base name (stripping the `_YYYY-MM-DD_HH-MM[_N]` suffix)
    and return only the latest per group.
    """
    if not models_dir.exists():
        return []
    import re as _re

    # Group candidate dirs by their base agent name
    # "attacklm-single_2026-06-10_01-12" -> ("attacklm-single", ...)
    # "attacklm-single" (legacy, no timestamp) -> ("attacklm-single", ...)
    ts_suffix = _re.compile(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}(_\d+)?$")
    groups: dict[str, list[Path]] = {}
    for p in models_dir.iterdir():
        if not p.is_dir():
            continue
        if not (p / "adapter_config.json").exists():
            continue
        m = ts_suffix.search(p.name)
        if m:
            base = p.name[: m.start()]
        else:
            base = p.name
        groups.setdefault(base, []).append(p)

    # Pick the lexicographically-largest name per group (timestamps
    # sort correctly). If only a legacy un-suffixed dir exists, that
    # is the only candidate and we use it.
    out = []
    for base, members in groups.items():
        members.sort(key=lambda p: p.name, reverse=True)
        out.append(members[0])
    return sorted(out)


def _strip_timestamp_suffix(adapter_path: Path) -> str:
    """Strip a `_YYYY-MM-DD_HH-MM[_N]` suffix from a directory name.

    Returns the base agent name (e.g. "attacklm-single"). Used to
    decide the output merged dir name — merges always write to
    models/merged/{base_name}/, not models/merged/{full_timestamped_name}/.
    """
    import re as _re

    ts_suffix = _re.compile(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}(_\d+)?$")
    m = ts_suffix.search(adapter_path.name)
    if m:
        return adapter_path.name[: m.start()]
    return adapter_path.name


def merge_all(base_model: str | None = None) -> list[str]:
    """Merge the most-recent trained adapter for each agent.

    v0.1.6: when a user trains an agent multiple times, only the
    latest timestamped adapter is merged. Older runs are skipped
    (they remain on disk for rollback/inspection but aren't merged
    into deployable artifacts). Override with --force-all-versions
    to merge every timestamped run.
    """
    import argparse as _ap

    models_dir = BASE_DIR / "models"
    merged_dir = BASE_DIR / "models" / "merged"

    adapters = _find_adapter_dirs(models_dir)
    if not adapters:
        print("\nERROR: No trained adapters found in models/")
        print("Run train_all.py first.")
        sys.exit(1)

    merged_paths = []
    start_time = time.time()

    for i, adapter in enumerate(adapters):
        # v0.1.6: strip the _TIMESTAMP suffix from the output name so
        # models/merged/attacklm-agent/ stays a single deployable
        # artifact per agent, not a forest of timestamped dirs.
        base_name = _strip_timestamp_suffix(adapter)
        output = merged_dir / base_name

        if output.exists() and any(output.glob("*.safetensors")):
            print(f"\n  ⏭  SKIP {base_name} — already merged (from {adapter.name})")
            merged_paths.append(str(output))
            continue

        # Auto-detect base model per adapter (or use the override)
        effective_base = base_model or read_adapter_base_model(adapter)

        print(
            f"\n  [{i + 1}/{len(adapters)}] Merging {base_name} "
            f"(from adapter: {adapter.name})..."
        )
        merge_adapter(str(adapter), str(output), effective_base)
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
        help="Single adapter to merge (e.g., models/attacklm-single_2026-06-10_01-12).",
    )
    parser.add_argument(
        "--output",
        help="Output directory for merged model. Defaults to a name derived from --adapter "
        "(strips _TIMESTAMP suffix if present). E.g. models/attacklm-single_2026-06-10_01-12 "
        "→ models/merged/attacklm-single",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help=f"Base HuggingFace model ID (default: auto-detect from adapter_config.json, fallback: {DEFAULT_BASE})",
    )
    parser.add_argument(
        "--merge-all",
        action="store_true",
        help="Merge the most-recent trained adapter for each agent",
    )

    args = parser.parse_args()

    if args.merge_all:
        merge_all(args.base_model)
    elif args.adapter:
        adapter_path = Path(args.adapter)
        if not args.output:
            # Auto-derive: strip _TIMESTAMP suffix, write to models/merged/{base_name}/
            base_name = _strip_timestamp_suffix(adapter_path)
            output_path = BASE_DIR / "models" / "merged" / base_name
            print(f"  ℹ️  --output not given, defaulting to: {output_path}")
        else:
            output_path = Path(args.output)
        merge_adapter(str(adapter_path), str(output_path), args.base_model)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
