#!/usr/bin/env python3
"""
AttackLM — Shared model/tokenizer loader for evaluation scripts.

Extracted from eval_retention.py so domain_bench.py (and future eval scripts)
can reuse the same loading, path resolution, and dtype detection logic.

Usage:
    from _eval_loader import load_model_and_tokenizer, resolve_model_path, detect_compute_dtype

    compute_dtype = detect_compute_dtype("bf16")
    model, tokenizer = load_model_and_tokenizer(
        base_model="huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated",
        adapter_path=None,
        compute_dtype=compute_dtype,
    )
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Path setup — make sure device_utils is importable
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))
from device_utils import (  # noqa: E402
    is_cuda,
    setup_allocator_env,
)

setup_allocator_env()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def resolve_model_path(model_id_or_path: str) -> str:
    """Resolve a local path or HF Hub ID for from_pretrained.

    If the input looks like a local path (starts with /, ./, ../, ~/,
    or the expanded path exists on disk), return the absolute resolved path.
    Otherwise, return it as-is (assumed to be an HF Hub model ID).
    """
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
                f"Model path does not exist: {resolved}\n"
                f"  (originally passed: {model_id_or_path!r})"
            )
        return resolved
    return p


def detect_compute_dtype(user_dtype: str | None) -> torch.dtype:
    """Auto-detect compute dtype if not specified.

    Args:
        user_dtype: One of 'bf16', 'fp16', 'fp32', or None for auto-detect.

    Returns:
        The resolved torch.dtype.
    """
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if user_dtype is not None:
        key = user_dtype.lower()
        if key not in dtype_map:
            print(
                f"  WARNING: unknown compute-dtype '{user_dtype}', "
                f"falling back to auto-detect",
                file=sys.stderr,
            )
        else:
            return dtype_map[key]

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

    Args:
        base_model: HF model ID or local path.
        adapter_path: Optional path to a PEFT LoRA adapter directory.
        compute_dtype: torch dtype for model weights.

    Returns:
        (model, tokenizer) tuple.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = resolve_model_path(base_model)
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
