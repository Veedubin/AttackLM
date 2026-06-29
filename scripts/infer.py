#!/usr/bin/env python3
"""infer.py — Inference example for a trained AttackLM LoRA adapter.

Loads the trained adapter (or a fully merged model) and runs a few
example prompts. Useful for smoke-testing a training run without
spinning up Ollama or LM Studio.

Usage:
    # Use a LoRA adapter (smaller disk footprint, slower at load time)
    uv run python scripts/infer.py --adapter models/attacklm-single

    # Use a fully merged model (faster to load, larger on disk)
    uv run python scripts/infer.py --merged models/attacklm-merged

    # Custom prompts
    uv run python scripts/infer.py --adapter models/attacklm-single \\
        --prompts "Show the System Services: Service Execution technique"

    # Adjust generation parameters
    uv run python scripts/infer.py --adapter models/attacklm-single \\
        --max-new-tokens 1024 --temperature 0.3 --top-p 0.9

Environment variables (override defaults):
    ATTACKLM_BASE_MODEL — base HuggingFace model ID
                          (default: unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Default base model — must match what was used in training
DEFAULT_BASE_MODEL = os.environ.get(
    "ATTACKLM_BASE_MODEL", "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit"
)

# Default example prompts — a mix of MITRE tactics and orchestrator
EXAMPLE_PROMPTS = [
    "Show the System Services: Service Execution technique (T1569.002) on Windows. Include the exact command, expected artifacts, and cleanup.",
    "Route this engagement: Windows 10 with Administrator privileges, Defender EDR active, objective is to dump LSASS without triggering alerts.",
    "List 3 Atomic Red Team tests for the Persistence tactic that use the Run registry key (T1547.001).",
    "Explain the difference between T1059.001 (PowerShell) and T1059.003 (Windows Command Shell) and when to use each in a red team engagement.",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained AttackLM LoRA adapter or merged model"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to a LoRA adapter directory (from scripts/train_*.py)",
    )
    parser.add_argument(
        "--merged",
        type=str,
        default=None,
        help="Path to a fully merged model directory (from scripts/merge_adapter.py)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Base HuggingFace model ID (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="Custom prompts. If not given, runs the EXAMPLE_PROMPTS demo set.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens to generate per response (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature 0.0-2.0 (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold 0.0-1.0 (default: 0.9)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=(
            "You are an authorized Red Team specialist with deep knowledge of "
            "MITRE ATT&CK, the Metasploit Framework, and adversary emulation. "
            "You provide precise tradecraft for security validation."
        ),
        help="System message to use for all prompts",
    )
    args = parser.parse_args()

    if not args.adapter and not args.merged:
        parser.error(
            "Must specify either --adapter PATH or --merged PATH. "
            f"Use 'make merge' to create a merged model from an adapter."
        )

    # ---------------------------------------------------------------------
    # Load model + tokenizer
    # ---------------------------------------------------------------------
    print("=" * 60)
    print(" AttackLM — Inference Demo")
    print("=" * 60)

    # We import torch + transformers here (not at top) so --help works
    # without the heavy deps installed.
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"ERROR: missing dependency: {e}")
        print("  Run: pip install attacklm[all]")
        return 1

    if args.adapter:
        # LoRA adapter path — load base, then apply adapter
        try:
            from peft import PeftModel
        except ImportError as e:
            print(f"ERROR: peft not installed: {e}")
            return 1

        adapter_path = Path(args.adapter).resolve()
        if not adapter_path.exists():
            print(f"ERROR: adapter path does not exist: {adapter_path}")
            return 1

        print(f"Base model: {args.base_model}")
        print(f"Adapter:    {adapter_path}")
        print()

        print("Loading base model in 4-bit (this can take 1-2 min)...")
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        print("Applying LoRA adapter...")
        model = PeftModel.from_pretrained(base, str(adapter_path))
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    else:
        # Fully merged model — direct load
        merged_path = Path(args.merged).resolve()
        if not merged_path.exists():
            print(f"ERROR: merged model path does not exist: {merged_path}")
            return 1
        print(f"Merged model: {merged_path}")
        print()
        print("Loading model (this can take 1-2 min)...")
        model = AutoModelForCausalLM.from_pretrained(
            str(merged_path),
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(merged_path))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print("Model loaded.")
    print()

    # ---------------------------------------------------------------------
    # Run prompts
    # ---------------------------------------------------------------------
    prompts = args.prompts or EXAMPLE_PROMPTS
    for i, user_msg in enumerate(prompts, 1):
        print("=" * 60)
        print(f" Prompt {i}/{len(prompts)}")
        print("=" * 60)
        print(f"USER: {user_msg}")
        print()

        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": user_msg},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated part (skip the input prompt)
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        print(f"ASSISTANT: {response}")
        print()

    print("=" * 60)
    print(f" Done — {len(prompts)} prompt(s) completed")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
