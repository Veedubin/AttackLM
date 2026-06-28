"""Device-agnostic helpers for CUDA, ROCm, CPU, and Apple Silicon (MPS).

AttackLM supports two GPU stacks:

    CUDA (NVIDIA)  — the default. PyTorch ships `+cu126`/`+cu128` wheels.
                     bitsandbytes 0.43+ ships CUDA wheels. flash-attn works.

    ROCm  (AMD)    — PyTorch ships `+rocm6.0`/`+rocm6.2` wheels. bitsandbytes
                     ships ROCm wheels on Python 3.10/3.11 (3.12/3.13 requires
                     a source build). flash-attn is NOT available — we fall
                     back to sdpa (still ~30% slower but works).

In practice, almost all `torch.cuda.*` calls work as-is on ROCm because
PyTorch's ROCm build exposes a CUDA-compatible API. This module is the one
place that knows about the differences.

Usage:
    from device_utils import (
        is_cuda, is_rocm, is_mps, gpu_name_and_memory,
        setup_allocator_env, enable_tf32, empty_cache_and_sync,
        gpu_mem_info, suggest_attn_implementation,
    )
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import torch


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def is_rocm() -> bool:
    """True if PyTorch is built against ROCm (AMD GPU).

    PyTorch's ROCm build sets `torch.version.hip` to the HIP version string
    (e.g. "6.0.40072"). The CUDA build leaves it as `None`.
    """
    return getattr(torch.version, "hip", None) is not None


def is_cuda() -> bool:
    """True if a CUDA backend is available (NVIDIA or AMD ROCm — both show as cuda)."""
    return torch.cuda.is_available()


def is_mps() -> bool:
    """True if Apple Silicon Metal is available."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def backend() -> str:
    """Return one of: 'rocm', 'cuda', 'mps', 'cpu'."""
    if is_cuda():
        return "rocm" if is_rocm() else "cuda"
    if is_mps():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# GPU info
# ---------------------------------------------------------------------------


def gpu_name_and_memory() -> Tuple[str, float]:
    """Return (device_name, total_memory_gb) for the active GPU.

    For CPU/MPS returns ('CPU', 0.0) or ('Apple Silicon (MPS)', 0.0).
    On ROCm, the device name starts with 'AMD' or contains 'gfx' info.
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return name, mem_gb
    if is_mps():
        return "Apple Silicon (MPS)", 0.0
    return "CPU", 0.0


# ---------------------------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------------------------


def setup_allocator_env() -> None:
    """Set memory allocator env vars for the active backend.

    CUDA and ROCm both honor `PYTORCH_CUDA_ALLOC_CONF` in modern PyTorch
    (>= 2.4 ROCm uses the same env var name as CUDA). We also set
    `PYTORCH_HIP_ALLOC_CONF` for older ROCm builds that read it.

    Must be called BEFORE any CUDA tensor is allocated. Safe to call multiple
    times — only sets if not already set.
    """
    if not is_cuda():
        return
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:128",
    )
    if is_rocm():
        # Older ROCm builds read PYTORCH_HIP_ALLOC_CONF instead.
        # Setting it is harmless on newer builds that ignore it.
        os.environ.setdefault(
            "PYTORCH_HIP_ALLOC_CONF",
            "expandable_segments:True,max_split_size_mb:128",
        )


def enable_tf32() -> None:
    """Enable TF32 matmul / cuDNN. Works on CUDA and ROCm.

    On consumer GPUs this gives ~10-15% throughput for free at a tiny accuracy
    cost. On H100 / MI300 it's basically free.
    """
    if not is_cuda():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def empty_cache_and_sync() -> None:
    """Free unused cached memory. No-op on CPU/MPS."""
    if not is_cuda():
        return
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def gpu_mem_info() -> Tuple[float, float]:
    """Return (free_gb, total_gb) for the active GPU. (0.0, 0.0) on CPU/MPS."""
    if is_cuda():
        free, total = torch.cuda.mem_get_info()
        return free / (1024**3), total / (1024**3)
    return 0.0, 0.0


def gpu_mem_info_bytes() -> Tuple[int, int]:
    """Return (free_bytes, total_bytes). (0, 0) on CPU/MPS."""
    if is_cuda():
        return torch.cuda.mem_get_info()
    return 0, 0


# ---------------------------------------------------------------------------
# Attention implementation picker
# ---------------------------------------------------------------------------


def is_flash_attn_available() -> bool:
    """True if flash-attn (v2 or v3) can be imported and used.

    Tries flash_attn_3 first (the current standard, installed via
    `pip install flash-attn-3`), then flash_attn (v2, legacy).

    flash_attn_3 needs torch CUDA libs on LD_LIBRARY_PATH at import
    time. We set it here so the user doesn't have to.
    """
    import os

    # Ensure torch CUDA libs are findable (needed for flash_attn_3)
    try:
        import torch

        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if torch_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
            os.environ["LD_LIBRARY_PATH"] = (
                torch_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
            )
    except ImportError:
        pass

    # Try flash_attn_3 first (current standard)
    try:
        import flash_attn_3  # noqa: F401

        return True
    except ImportError:
        pass

    # Fall back to flash_attn v2 (legacy)
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


def suggest_attn_implementation(packing: bool) -> tuple[str, bool]:
    """Pick an attention implementation and determine if packing is usable.

    Args:
        packing: True if the user requested --packing.

    Returns:
        (attn_implementation, packing_enabled) tuple.
        - attn_implementation: 'flash_attention_2' or 'sdpa'
        - packing_enabled: True if packing is safe to use (flash-attn
          available on CUDA), False otherwise.

    When flash-attn is not available, packing is silently disabled to
    prevent cross-sample contamination. The user doesn't need to know
    or care — training works correctly either way, just ~30% slower
    without packing.
    """
    if not packing:
        return ("sdpa", False)

    if is_cuda() and not is_rocm() and is_flash_attn_available():
        return ("flash_attention_2", True)

    # flash-attn not available — packing would cause cross-sample
    # contamination. Disable it silently.
    return ("sdpa", False)


# ---------------------------------------------------------------------------
# User-facing banner
# ---------------------------------------------------------------------------


def print_hardware_banner() -> str:
    """Print a one-line summary of detected hardware.

    Returns the backend string for callers that want to log it.
    """
    b = backend()
    if b in ("cuda", "rocm"):
        name, mem = gpu_name_and_memory()
        stack = "ROCm" if b == "rocm" else "CUDA"
        print(f"  Hardware: {stack} GPU — {name} ({mem:.1f} GB)", file=sys.stderr)
    elif b == "mps":
        print(
            "  Hardware: Apple Silicon (MPS) — training will be very slow",
            file=sys.stderr,
        )
    else:
        print(
            "  Hardware: CPU — training will be extremely slow (dry-run only)",
            file=sys.stderr,
        )
    return b
