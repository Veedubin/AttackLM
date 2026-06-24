#!/usr/bin/env python3
"""
AttackLM — Steering Vector Tool (Pattern 6)

Extract, apply, sweep, and diagnose activation steering vectors
for controlling coarse model behaviors (verbosity, OPSEC awareness, etc.).

Based on the methodology of Arditi et al. (arXiv:2406.11717) and
the ds4 dir-steering implementation by antirez.

Usage:
    python scripts/steering.py extract --base-model MODEL --target prompts.txt --control prompts.txt
    python scripts/steering.py apply --base-model MODEL --vectors vectors.f32 --prompt "text"
    python scripts/steering.py sweep --base-model MODEL --vectors vectors.f32 --prompts prompts.txt
    python scripts/steering.py diagnose --base-model ABLITERATED --reference-model ORIGINAL --harmful h.txt --harmless s.txt
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
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


# ---------------------------------------------------------------------------
# Vector math utilities
# ---------------------------------------------------------------------------


def normalize(v: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Normalize a vector to unit length.

    Args:
        v: 1-D tensor or array.

    Returns:
        Unit-length vector of the same type.

    Raises:
        ValueError: If the vector has zero norm.
    """
    if isinstance(v, np.ndarray):
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            raise ValueError("Cannot normalize a zero vector")
        return v / norm
    norm = torch.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("Cannot normalize a zero vector")
    return v / norm


def dot(a: torch.Tensor | np.ndarray, b: torch.Tensor | np.ndarray) -> float:
    """Compute the dot product of two 1-D vectors.

    Accepts mixed numpy/torch inputs — converts to numpy for consistency.
    """
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Hook helpers
# ---------------------------------------------------------------------------


def _get_hook_target(model: Any, layer_num: int, component: str) -> Any:
    """Get the module to hook for a given layer and component.

    Args:
        model: HuggingFace model with model.model.layers structure.
        layer_num: Layer index (0-based).
        component: Either 'ffn_out' (MLP) or 'attn_out' (self_attn).

    Returns:
        The module to register a forward hook on.
    """
    if component == "ffn_out":
        return model.model.layers[layer_num].mlp
    return model.model.layers[layer_num].self_attn


def _remove_hooks(hooks: list[Any]) -> None:
    """Remove all registered hooks."""
    for h in hooks:
        h.remove()


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_steering_vector(
    model: Any,
    tokenizer: Any,
    target_prompts: list[str],
    control_prompts: list[str],
    layers: list[int] | None = None,
    component: str = "ffn_out",
    orthogonalize: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract a steering vector from contrastive prompt pairs.

    Algorithm (from Arditi et al. / ds4):
      1. Hook FFN output at specified layers
      2. Run target prompts, capture activations, average
      3. Run control prompts, capture activations, average
      4. Compute normalized difference: direction = normalize(mean(target) - mean(control))
      5. Optionally orthogonalize against control mean

    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        target_prompts: List of target prompts.
        control_prompts: List of control prompts.
        layers: Layer indices (default: 20-30 inclusive).
        component: 'ffn_out' or 'attn_out'.
        orthogonalize: Whether to orthogonalize against control mean.

    Returns:
        Tuple of (vectors_array, metadata_dict).
        vectors_array shape: (n_layers, hidden_dim), dtype float32.
    """
    if layers is None:
        layers = list(range(20, 31))

    if not target_prompts:
        raise ValueError("target_prompts must not be empty")
    if not control_prompts:
        raise ValueError("control_prompts must not be empty")

    n_layers = len(layers)
    hidden_dim = model.config.hidden_size
    device = model.device

    # Accumulators for target and control activations
    target_acts: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}
    control_acts: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}

    def make_hook(store: dict[int, list[torch.Tensor]], idx: int):
        def hook(module: Any, input: Any, output: Any) -> None:
            # output is tuple for HF transformers; take last-token hidden state
            hidden = output[0][:, -1, :].detach().cpu().float()
            store[idx].append(hidden)

        return hook

    # --- Run target prompts ---
    hooks_target: list[Any] = []
    for i, layer_num in enumerate(layers):
        target_module = _get_hook_target(model, layer_num, component)
        h = target_module.register_forward_hook(make_hook(target_acts, i))
        hooks_target.append(h)

    for prompt in target_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)

    _remove_hooks(hooks_target)

    # --- Run control prompts ---
    hooks_control: list[Any] = []
    for i, layer_num in enumerate(layers):
        target_module = _get_hook_target(model, layer_num, component)
        h = target_module.register_forward_hook(make_hook(control_acts, i))
        hooks_control.append(h)

    for prompt in control_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)

    _remove_hooks(hooks_control)

    # --- Compute directions ---
    directions = np.zeros((n_layers, hidden_dim), dtype=np.float32)

    for i in range(n_layers):
        if not target_acts[i]:
            raise ValueError(f"No target activations captured for layer {layers[i]}")
        if not control_acts[i]:
            raise ValueError(f"No control activations captured for layer {layers[i]}")

        target_mean = torch.stack(target_acts[i]).mean(dim=0)
        control_mean = torch.stack(control_acts[i]).mean(dim=0)

        diff = target_mean - control_mean
        direction = normalize(diff)

        if orthogonalize:
            control_norm = normalize(control_mean)
            proj = dot(direction, control_norm)
            direction = normalize(direction - proj * control_norm)

        directions[i] = (
            direction.numpy() if isinstance(direction, torch.Tensor) else direction
        )

    metadata = {
        "format": "attacklm-steering-v1",
        "shape": [n_layers, hidden_dim],
        "component": component,
        "layers": layers,
        "orthogonalize_control_mean": orthogonalize,
        "model": model.config._name_or_path,
    }

    return directions, metadata


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def apply_steering(
    model: Any,
    tokenizer: Any,
    vectors: np.ndarray,
    metadata: dict[str, Any],
    prompt: str,
    scale: float = 1.0,
    layers: list[int] | None = None,
    max_new_tokens: int = 256,
    seed: int = 42,
) -> str:
    """Apply steering vectors during inference using ds4 projection formula.

    Formula:  y = y - scale * direction[layer] * dot(direction[layer], y)

    Positive scale SUPPRESSES the target direction.
    Negative scale AMPLIFIES the target direction.

    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        vectors: Steering vectors array, shape (n_layers, hidden_dim).
        metadata: Metadata dict from extraction (contains 'layers').
        prompt: Text prompt to generate from.
        scale: Steering multiplier (default 1.0).
        layers: Override layers from metadata (default: use metadata layers).
        max_new_tokens: Max tokens to generate.
        seed: Random seed.

    Returns:
        Generated text string.
    """
    if layers is None:
        layers = metadata["layers"]

    if len(layers) != vectors.shape[0]:
        raise ValueError(
            f"Layer count mismatch: {len(layers)} layers but "
            f"vectors shape is {vectors.shape[0]}"
        )

    hidden_dim = model.config.hidden_size
    if vectors.shape[1] != hidden_dim:
        raise ValueError(
            f"Hidden dim mismatch: vectors have {vectors.shape[1]} but "
            f"model has {hidden_dim}"
        )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = model.device
    component = metadata.get("component", "ffn_out")

    def make_steering_hook(layer_idx: int, direction: np.ndarray):
        vec = torch.tensor(direction, device=device, dtype=model.dtype)

        def hook(module: Any, input: Any, output: Any) -> Any:
            hidden = output[0][:, -1, :]
            # Projection-based steering: remove component along direction
            projection = torch.dot(hidden.squeeze(0), vec)
            hidden = hidden - scale * projection * vec
            modified = list(output)
            modified[0] = output[0].clone()
            modified[0][:, -1, :] = hidden
            return tuple(modified)

        return hook

    # Register steering hooks
    hooks: list[Any] = []
    for i, layer_num in enumerate(layers):
        target_module = _get_hook_target(model, layer_num, component)
        h = target_module.register_forward_hook(make_steering_hook(i, vectors[i]))
        hooks.append(h)

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        _remove_hooks(hooks)

    input_len = inputs["input_ids"].shape[1]
    generated_tokens = output_ids[0][input_len:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run_sweep(
    model: Any,
    tokenizer: Any,
    vectors: np.ndarray,
    metadata: dict[str, Any],
    prompts: list[str],
    scales: list[float],
    layers: list[int] | None = None,
    max_new_tokens: int = 256,
    seed: int = 42,
) -> dict[str, Any]:
    """Sweep multiplier values to calibrate steering strength.

    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        vectors: Steering vectors array.
        metadata: Metadata dict from extraction.
        prompts: List of prompts to evaluate.
        scales: List of multiplier values to test.
        layers: Override layers (default: use metadata).
        max_new_tokens: Max generation tokens.
        seed: Random seed.

    Returns:
        Sweep report dict with metadata + results array.
    """
    if layers is None:
        layers = metadata["layers"]

    results: list[dict[str, Any]] = []

    for scale in scales:
        token_counts: list[int] = []
        samples: list[dict[str, Any]] = []

        for prompt in prompts:
            try:
                text = apply_steering(
                    model,
                    tokenizer,
                    vectors,
                    metadata,
                    prompt,
                    scale=scale,
                    layers=layers,
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                )
                token_count = len(tokenizer.encode(text))
                token_counts.append(token_count)
                samples.append(
                    {
                        "prompt": prompt[:100],  # truncate for report
                        "tokens": token_count,
                        "text": text[:200],  # truncate for report
                    }
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    samples.append(
                        {
                            "prompt": prompt[:100],
                            "tokens": 0,
                            "text": "<cuda_oom>",
                        }
                    )
                    continue
                raise

        mean_tokens = float(np.mean(token_counts)) if token_counts else 0.0
        std_tokens = float(np.std(token_counts)) if len(token_counts) > 1 else 0.0

        results.append(
            {
                "scale": scale,
                "mean_tokens": round(mean_tokens, 1),
                "std_tokens": round(std_tokens, 1),
                "n_prompts": len(prompts),
                "samples": samples,
            }
        )

    return {
        "metadata": {
            "vector_shape": list(vectors.shape),
            "model": metadata.get("model", "unknown"),
            "layers": layers,
            "scales": scales,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "results": results,
    }


# ---------------------------------------------------------------------------
# Diagnose
# ---------------------------------------------------------------------------


def diagnose_refusal(
    target_model: Any,
    reference_model: Any,
    tokenizer: Any,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    layers: list[int] | None = None,
    component: str = "ffn_out",
) -> dict[str, Any]:
    """Measure residual refusal direction in an abliterated model.

    Algorithm:
      1. Extract refusal direction from reference (non-abliterated) model
      2. Measure projection magnitude in target (abliterated) model
      3. Compute reduction ratio

    Args:
        target_model: The abliterated/fine-tuned model to diagnose.
        reference_model: The non-abliterated base model (for direction extraction).
        tokenizer: Tokenizer (shared between models).
        harmful_prompts: Prompts that should trigger refusal in reference model.
        harmless_prompts: Prompts that should get compliance in reference model.
        layers: Layer indices to analyze (default: 20-30).
        component: 'ffn_out' or 'attn_out'.

    Returns:
        Diagnostic report dict.
    """
    if layers is None:
        layers = list(range(20, 31))

    # Step 1: Extract refusal direction from reference model
    ref_directions, ref_metadata = extract_steering_vector(
        reference_model,
        tokenizer,
        harmful_prompts,
        harmless_prompts,
        layers=layers,
        component=component,
        orthogonalize=True,
    )

    # Step 2: Measure projection magnitude in target model
    n_layers = len(layers)
    hidden_dim = target_model.config.hidden_size
    device = target_model.device

    # Accumulate target model activations on harmful prompts
    target_acts: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}

    def make_hook(store: dict[int, list[torch.Tensor]], idx: int):
        def hook(module: Any, input: Any, output: Any) -> None:
            hidden = output[0][:, -1, :].detach().cpu().float()
            store[idx].append(hidden)

        return hook

    hooks: list[Any] = []
    for i, layer_num in enumerate(layers):
        target_module = _get_hook_target(target_model, layer_num, component)
        h = target_module.register_forward_hook(make_hook(target_acts, i))
        hooks.append(h)

    for prompt in harmful_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            target_model(**inputs)

    _remove_hooks(hooks)

    # Step 3: Compute per-layer magnitudes
    magnitude_ref: list[float] = []
    magnitude_target: list[float] = []

    for i in range(n_layers):
        if not target_acts[i]:
            magnitude_ref.append(float(np.linalg.norm(ref_directions[i])))
            magnitude_target.append(0.0)
            continue

        ref_vec = ref_directions[i]  # already unit-length
        target_mean = torch.stack(target_acts[i]).mean(dim=0).numpy()

        # Project target activation onto reference direction
        proj_ref = float(np.abs(np.dot(ref_vec, ref_vec)))  # = 1.0 for unit vec
        proj_target = float(
            np.abs(np.dot(ref_vec, target_mean / (np.linalg.norm(target_mean) + 1e-12)))
        )

        magnitude_ref.append(proj_ref)
        magnitude_target.append(proj_target)

    avg_mag_ref = float(np.mean(magnitude_ref))
    avg_mag_target = float(np.mean(magnitude_target))
    reduction_ratio = 1.0 - (avg_mag_target / (avg_mag_ref + 1e-12))

    interpretation = (
        f"{reduction_ratio:.0%} reduction in refusal direction magnitude "
        f"after abliteration + fine-tuning"
    )

    return {
        "metadata": {
            "reference_model": reference_model.config._name_or_path,
            "target_model": target_model.config._name_or_path,
            "layers": layers,
            "component": component,
            "n_harmful": len(harmful_prompts),
            "n_harmless": len(harmless_prompts),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "refusal_direction": {
            "layers": layers,
            "magnitude_in_reference": round(avg_mag_ref, 4),
            "magnitude_in_target": round(avg_mag_target, 4),
            "reduction_ratio": round(reduction_ratio, 4),
            "interpretation": interpretation,
            "per_layer_magnitude_reference": [round(m, 4) for m in magnitude_ref],
            "per_layer_magnitude_target": [round(m, 4) for m in magnitude_target],
        },
    }


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def save_vectors(vectors: np.ndarray, metadata: dict[str, Any], prefix: str) -> None:
    """Save steering vectors to .f32 binary + .json metadata.

    Args:
        vectors: Numpy array of shape (n_layers, hidden_dim), dtype float32.
        metadata: Metadata dict.
        prefix: Output file prefix (e.g., 'data/steering/verbosity').
    """
    # Save binary vectors
    f32_path = Path(prefix).with_suffix(".f32")
    f32_path.parent.mkdir(parents=True, exist_ok=True)
    vectors.astype(np.float32).tofile(str(f32_path))

    # Save JSON metadata
    json_path = Path(prefix).with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  Saved vectors: {f32_path} ({vectors.nbytes} bytes)", file=sys.stderr)
    print(f"  Saved metadata: {json_path}", file=sys.stderr)


def load_vectors(prefix: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Load steering vectors from .f32 binary + .json metadata.

    Args:
        prefix: File prefix (with or without extension).

    Returns:
        Tuple of (vectors_array, metadata_dict).
    """
    # Resolve paths
    prefix_path = Path(prefix)
    if prefix_path.suffix == ".f32":
        f32_path = prefix_path
        json_path = prefix_path.with_suffix(".json")
    elif prefix_path.suffix == ".json":
        json_path = prefix_path
        f32_path = prefix_path.with_suffix(".f32")
    else:
        # No extension given — try .f32/.json pair
        f32_path = Path(prefix).with_suffix(".f32")
        json_path = Path(prefix).with_suffix(".json")

    # Load metadata
    with open(json_path, encoding="utf-8") as f:
        metadata = json.load(f)

    shape = metadata["shape"]  # [n_layers, hidden_dim]
    vectors = np.fromfile(str(f32_path), dtype=np.float32).reshape(shape)

    return vectors, metadata


def load_prompts(path: str) -> list[str]:
    """Load prompts from a text file (one per line).

    Args:
        path: Path to the prompts file.

    Returns:
        List of non-empty prompt strings.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")
    with open(p, encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError(f"Prompts file is empty: {path}")
    return prompts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AttackLM Steering Vector Tool — "
        "Extract and apply activation steering vectors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common args
    def add_common_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--base-model",
            type=str,
            required=True,
            help="HuggingFace model ID or local path",
        )
        sp.add_argument(
            "--adapter",
            type=str,
            default=None,
            help="Path to PEFT LoRA adapter directory",
        )
        sp.add_argument(
            "--seed", type=int, default=42, help="Random seed (default: 42)"
        )
        sp.add_argument(
            "--compute-dtype",
            type=str,
            default=None,
            choices=["bf16", "fp16", "fp32"],
            help="Compute dtype (default: auto-detect)",
        )

    # --- extract ---
    ext = subparsers.add_parser(
        "extract",
        help="Extract steering vectors from contrastive prompt pairs",
    )
    add_common_args(ext)
    ext.add_argument(
        "--target",
        type=str,
        required=True,
        help="Path to target prompts file (one per line)",
    )
    ext.add_argument(
        "--control",
        type=str,
        required=True,
        help="Path to control prompts file (one per line)",
    )
    ext.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=list(range(20, 31)),
        help="Layer indices (default: 20-30)",
    )
    ext.add_argument(
        "--component",
        type=str,
        default="ffn_out",
        choices=["ffn_out", "attn_out"],
        help="Component to hook (default: ffn_out)",
    )
    ext.add_argument(
        "--orthogonalize",
        action="store_true",
        default=True,
        help="Orthogonalize against control mean (default: True)",
    )
    ext.add_argument(
        "--no-orthogonalize", action="store_true", help="Skip orthogonalization"
    )
    ext.add_argument(
        "--output",
        type=str,
        default="data/steering/vectors",
        help="Output file prefix for .json + .f32 (default: data/steering/vectors)",
    )

    # --- apply ---
    app = subparsers.add_parser(
        "apply",
        help="Apply steering vectors during inference",
    )
    add_common_args(app)
    app.add_argument(
        "--vectors",
        type=str,
        required=True,
        help="Path to steering vector .f32 file (or prefix)",
    )
    app.add_argument(
        "--prompt", type=str, required=True, help="Text prompt to generate from"
    )
    app.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Override layers from metadata",
    )
    app.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Steering multiplier (positive=suppress, negative=amplify)",
    )
    app.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max tokens to generate (default: 256)",
    )

    # --- sweep ---
    swp = subparsers.add_parser(
        "sweep",
        help="Sweep multiplier values for calibration",
    )
    add_common_args(swp)
    swp.add_argument(
        "--vectors", type=str, required=True, help="Path to steering vector .f32 file"
    )
    swp.add_argument(
        "--prompts", type=str, required=True, help="Path to prompts file (one per line)"
    )
    swp.add_argument(
        "--scales",
        type=str,
        default="-1,-0.5,0,0.5,1,2",
        help="Comma-separated scale values (default: -1,-0.5,0,0.5,1,2)",
    )
    swp.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Override layers from metadata",
    )
    swp.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max tokens to generate (default: 256)",
    )
    swp.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: auto-generated)",
    )

    # --- diagnose ---
    diag = subparsers.add_parser(
        "diagnose",
        help="Measure residual refusal direction in abliterated models",
    )
    add_common_args(diag)
    diag.add_argument(
        "--reference-model",
        type=str,
        required=True,
        help="Non-abliterated reference model (HF ID or path)",
    )
    diag.add_argument(
        "--harmful", type=str, required=True, help="Path to harmful prompts file"
    )
    diag.add_argument(
        "--harmless", type=str, required=True, help="Path to harmless prompts file"
    )
    diag.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=list(range(20, 31)),
        help="Layer indices (default: 20-30)",
    )
    diag.add_argument(
        "--component",
        type=str,
        default="ffn_out",
        choices=["ffn_out", "attn_out"],
        help="Component to hook (default: ffn_out)",
    )
    diag.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: auto-generated)",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Entry point for the steering vector tool."""
    args = parse_args(argv)

    print("\n" + "=" * 60, file=sys.stderr)
    print(" AttackLM Steering Vector Tool", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Command: {args.command}", file=sys.stderr)

    compute_dtype = detect_compute_dtype(args.compute_dtype)

    if args.command == "extract":
        return _cmd_extract(args, compute_dtype)
    elif args.command == "apply":
        return _cmd_apply(args, compute_dtype)
    elif args.command == "sweep":
        return _cmd_sweep(args, compute_dtype)
    elif args.command == "diagnose":
        return _cmd_diagnose(args, compute_dtype)
    else:
        print(f"ERROR: Unknown command: {args.command}", file=sys.stderr)
        return 1


def _cmd_extract(args: argparse.Namespace, compute_dtype: torch.dtype) -> int:
    """Handle the 'extract' subcommand."""
    print(f"  Base model:       {args.base_model}", file=sys.stderr)
    print(f"  Adapter:          {args.adapter or '(none)'}", file=sys.stderr)
    print(f"  Target prompts:   {args.target}", file=sys.stderr)
    print(f"  Control prompts:  {args.control}", file=sys.stderr)
    print(f"  Layers:           {args.layers}", file=sys.stderr)
    print(f"  Component:        {args.component}", file=sys.stderr)
    print(f"  Orthogonalize:    {not args.no_orthogonalize}", file=sys.stderr)
    print(f"  Output:           {args.output}", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)

    # Load prompts
    target_prompts = load_prompts(args.target)
    control_prompts = load_prompts(args.control)
    print(f"  Loaded {len(target_prompts)} target prompts", file=sys.stderr)
    print(f"  Loaded {len(control_prompts)} control prompts", file=sys.stderr)

    # Load model
    print("\nLoading model...", file=sys.stderr)
    try:
        model, tokenizer = load_model_and_tokenizer(
            args.base_model, args.adapter, compute_dtype
        )
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}", file=sys.stderr)
        return 1

    orthogonalize = not args.no_orthogonalize

    # Extract
    print("\nExtracting steering vectors...", file=sys.stderr)
    try:
        vectors, metadata = extract_steering_vector(
            model,
            tokenizer,
            target_prompts,
            control_prompts,
            layers=args.layers,
            component=args.component,
            orthogonalize=orthogonalize,
        )
    except Exception as e:
        print(f"ERROR: Extraction failed: {e}", file=sys.stderr)
        return 1

    # Add extraction info to metadata
    metadata["target_file"] = str(args.target)
    metadata["control_file"] = str(args.control)
    metadata["adapter"] = args.adapter
    metadata["note"] = (
        "Positive scale suppresses the target direction. "
        "Negative scale amplifies the target direction."
    )

    # Save
    save_vectors(vectors, metadata, args.output)
    print(f"\n  Extraction complete: {vectors.shape}", file=sys.stderr)
    return 0


def _cmd_apply(args: argparse.Namespace, compute_dtype: torch.dtype) -> int:
    """Handle the 'apply' subcommand."""
    print(f"  Base model:       {args.base_model}", file=sys.stderr)
    print(f"  Adapter:          {args.adapter or '(none)'}", file=sys.stderr)
    print(f"  Vectors:          {args.vectors}", file=sys.stderr)
    print(f"  Scale:            {args.scale}", file=sys.stderr)
    print(f"  Max new tokens:   {args.max_new_tokens}", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)

    # Load vectors
    vectors, metadata = load_vectors(args.vectors)
    print(f"  Loaded vectors: shape={vectors.shape}", file=sys.stderr)

    # Load model
    print("\nLoading model...", file=sys.stderr)
    try:
        model, tokenizer = load_model_and_tokenizer(
            args.base_model, args.adapter, compute_dtype
        )
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}", file=sys.stderr)
        return 1

    # Validate dimensions
    hidden_dim = model.config.hidden_size
    if vectors.shape[1] != hidden_dim:
        print(
            f"ERROR: Hidden dim mismatch: vectors={vectors.shape[1]}, "
            f"model={hidden_dim}",
            file=sys.stderr,
        )
        return 1

    # Apply steering
    layers = args.layers if args.layers else metadata["layers"]
    print(
        f"\nApplying steering (scale={args.scale}, layers={layers})...", file=sys.stderr
    )
    try:
        text = apply_steering(
            model,
            tokenizer,
            vectors,
            metadata,
            args.prompt,
            scale=args.scale,
            layers=layers,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
    except Exception as e:
        print(f"ERROR: Steering application failed: {e}", file=sys.stderr)
        return 1

    # Output
    print("\n--- Generated Output ---", file=sys.stderr)
    print(text)
    return 0


def _cmd_sweep(args: argparse.Namespace, compute_dtype: torch.dtype) -> int:
    """Handle the 'sweep' subcommand."""
    scales = [float(s) for s in args.scales.split(",")]
    print(f"  Base model:       {args.base_model}", file=sys.stderr)
    print(f"  Vectors:          {args.vectors}", file=sys.stderr)
    print(f"  Scales:           {scales}", file=sys.stderr)
    print(f"  Max new tokens:   {args.max_new_tokens}", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)

    # Load vectors and prompts
    vectors, metadata = load_vectors(args.vectors)
    prompts = load_prompts(args.prompts)
    print(f"  Loaded vectors: shape={vectors.shape}", file=sys.stderr)
    print(f"  Loaded {len(prompts)} prompts", file=sys.stderr)

    # Load model
    print("\nLoading model...", file=sys.stderr)
    try:
        model, tokenizer = load_model_and_tokenizer(
            args.base_model, args.adapter, compute_dtype
        )
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}", file=sys.stderr)
        return 1

    # Run sweep
    layers = args.layers if args.layers else metadata["layers"]
    print(f"\nRunning sweep across {len(scales)} scale values...", file=sys.stderr)
    report = run_sweep(
        model,
        tokenizer,
        vectors,
        metadata,
        prompts,
        scales,
        layers=layers,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    # Write report
    output_path = args.output
    if output_path is None:
        output_path = "data/steering/sweep_report.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Sweep report written to: {output_path}", file=sys.stderr)
    return 0


def _cmd_diagnose(args: argparse.Namespace, compute_dtype: torch.dtype) -> int:
    """Handle the 'diagnose' subcommand."""
    print(f"  Target model (abliterated): {args.base_model}", file=sys.stderr)
    print(f"  Reference model (original):  {args.reference_model}", file=sys.stderr)
    print(f"  Harmful prompts:  {args.harmful}", file=sys.stderr)
    print(f"  Harmless prompts:  {args.harmless}", file=sys.stderr)
    print(f"  Layers:           {args.layers}", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)

    # Load prompts
    harmful_prompts = load_prompts(args.harmful)
    harmless_prompts = load_prompts(args.harmless)
    print(f"  Loaded {len(harmful_prompts)} harmful prompts", file=sys.stderr)
    print(f"  Loaded {len(harmless_prompts)} harmless prompts", file=sys.stderr)

    # Load target model (abliterated)
    print("\nLoading target model...", file=sys.stderr)
    try:
        target_model, tokenizer = load_model_and_tokenizer(
            args.base_model, args.adapter, compute_dtype
        )
    except Exception as e:
        print(f"ERROR: Failed to load target model: {e}", file=sys.stderr)
        return 1

    # Load reference model (non-abliterated)
    print("Loading reference model...", file=sys.stderr)
    try:
        reference_model, _ = load_model_and_tokenizer(
            args.reference_model, None, compute_dtype
        )
    except Exception as e:
        print(f"ERROR: Failed to load reference model: {e}", file=sys.stderr)
        return 1

    # Diagnose
    print("\nDiagnosing refusal direction...", file=sys.stderr)
    try:
        report = diagnose_refusal(
            target_model,
            reference_model,
            tokenizer,
            harmful_prompts,
            harmless_prompts,
            layers=args.layers,
            component=args.component,
        )
    except Exception as e:
        print(f"ERROR: Diagnosis failed: {e}", file=sys.stderr)
        return 1

    # Write report
    output_path = args.output
    if output_path is None:
        output_path = "data/steering/refusal_diagnostic.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    rd = report["refusal_direction"]
    print(f"\n  Refusal Direction Diagnostic:", file=sys.stderr)
    print(
        f"    Magnitude in reference: {rd['magnitude_in_reference']}", file=sys.stderr
    )
    print(f"    Magnitude in target:   {rd['magnitude_in_target']}", file=sys.stderr)
    print(f"    Reduction ratio:       {rd['reduction_ratio']}", file=sys.stderr)
    print(f"    {rd['interpretation']}", file=sys.stderr)
    print(f"\n  Report written to: {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
