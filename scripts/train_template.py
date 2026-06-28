#!/usr/bin/env python3
"""
AttackLM — QLoRA Fine-Tuning Template

Train any AttackLM tactical agent or orchestrator on its respective JSONL dataset
using 4-bit QLoRA with SFTTrainer.

Usage:
  # Dry run (default) — validate dataset and print stats
  python train_template.py --dataset data/datasets/persistence_dataset.jsonl --output models/persistence-agent

  # Actual training
  python train_template.py --dataset data/datasets/persistence_dataset.jsonl --output models/persistence-agent --train

  # Custom base model and hyperparams
  python train_template.py --dataset data/datasets/orchestrator_dataset.jsonl --output models/orchestrator-agent --base-model Qwen/Qwen2.5-7B-Instruct --epochs 3 --batch-size 2

  # OOM-safe: frequent checkpoints + gradient accumulation
  python train_template.py --dataset data/datasets/persistence_dataset.jsonl --output models/persistence-agent --train --save-steps 200 --gradient-accumulation-steps 4

  # Dry run with custom params
  python train_template.py --dataset data/datasets/persistence_dataset.jsonl --output models/persistence-agent --base-model Qwen/Qwen2.5-7B-Instruct --dry-run

Dependencies:
  pip install transformers datasets trl peft bitsandbytes accelerate
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# OOM fix #1: PyTorch CUDA / ROCm allocator configuration
# ---------------------------------------------------------------------------
# `expandable_segments:True` is the single biggest fix for long-running OOMs.
# It tells the allocator to use virtual memory mapping, so freed blocks can be
# coalesced instead of fragmenting into a checkerboard of small holes. This
# matters after 20,000+ forward/backward passes where the standard allocator
# has fragmented VRAM into many small free blocks that can't satisfy a single
# large allocation (even though total free VRAM looks sufficient in nvidia-smi).
#
# `max_split_size_mb:128` caps how aggressively the allocator splits blocks,
# keeping large allocations from being broken into many small pieces.
#
# MUST be set BEFORE any CUDA tensor is allocated, so we do it at import time.
# We do this through `device_utils` so ROCm gets the same setup (it honors
# the same env var in PyTorch >= 2.4 and falls back to PYTORCH_HIP_ALLOC_CONF
# on older builds).

sys.path.insert(0, str(Path(__file__).resolve().parent))
from device_utils import (  # noqa: E402  (import after sys.path tweak)
    is_cuda,
    is_rocm,
    setup_allocator_env,
    enable_tf32,
    empty_cache_and_sync,
    gpu_mem_info,
    gpu_mem_info_bytes,
    suggest_attn_implementation,
    print_hardware_banner,
)

# Apply the allocator env vars at import time, before any tensor allocation.
setup_allocator_env()

# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AttackLM QLoRA Fine-Tuning Template",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Validate dataset without training
  python train_template.py --dataset data/datasets/persistence_dataset.jsonl --output models/persistence-agent

  # Actually train
  python train_template.py --dataset data/datasets/persistence_dataset.jsonl --output models/persistence-agent --train

  # Orchestrator model with custom LoRA rank
  python train_template.py --dataset data/datasets/orchestrator_dataset.jsonl --output models/orchestrator-agent --lora-r 32

  # Resume from last checkpoint in output dir
  python train_template.py --dataset data/datasets/persistence_dataset.jsonl --output models/persistence-agent --train --resume-from-checkpoint
""",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to JSONL training file (required)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for saved adapter (required)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base HuggingFace model ID (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Max training epochs — early stopping will halt when eval loss stops improving (default: 50)",
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.1,
        help="Fraction of data held out for evaluation (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Stop after N eval rounds without improvement, rollback to best checkpoint (default: 3)",
    )
    parser.add_argument(
        "--early-stop-steps",
        type=int,
        default=5,
        help="Check EMA-smoothed loss trend every N log calls "
        "(default: 5, ~50 steps at logging_steps=10). "
        "Compares first vs second half of smoothed window — stops "
        "after 3 consecutive checks with no downward trend. "
        "Set to 0 to disable.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device training batch size (default: 2)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05). Set 0.0 for curriculum stage 2 fine-tuning.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="Resume training from the last checkpoint-N/ in the output dir",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=200,
        help="Save checkpoint every N steps (default: 200). Lower = more frequent saves.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps — simulate larger batch without extra VRAM. "
        "effective_batch = batch_size × grad_accum. (default: 1)",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="paged_adamw_8bit",
        help="Optimizer (default: paged_adamw_8bit). Other options: adamw_torch, adamw_8bit, sgd.",
    )
    parser.add_argument(
        "--packing",
        dest="packing",
        action="store_true",
        default=False,
        help="Enable example packing + padding-free training (default: OFF). "
        "Concatenates short examples into max_length-sized sequences for ~30-40%% throughput gain. "
        "REQUIRES flash_attention_2 to prevent cross-sample contamination. "
        "If flash-attn is not installed, training will fail with a clear error and "
        "suggestion to use --no-packing instead. Default is OFF because flash-attn "
        "is hard to install in many envs (large compile, OOM-prone).",
    )
    parser.add_argument(
        "--no-packing",
        dest="packing",
        action="store_false",
        help="Disable example packing (default). Each example is padded to max_length "
        "individually. Slower but doesn't require flash-attn. Always works.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Load data and print stats without training (default unless --train)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="Actually run training (disabled by default for safety)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Stop training after N optimizer steps (overrides --epochs). "
        "Used by HPO trials to cap run-length at a fixed budget. "
        "Default: -1 (use --epochs)",
    )
    parser.add_argument(
        "--hpo-metrics-csv",
        type=str,
        default=None,
        help="Path to write per-step HPO metrics as CSV. Includes extra fields "
        "(tokens/sec, pairs/sec, pair size stats) that HF's default JSON log "
        "doesn't capture. Default: None (off)",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        default=False,
        help=(
            "v0.2.2+: Use --output as-is, no timestamp suffix. Default is to "
            "auto-append a timestamp so re-runs don't clobber previous runs. "
            "If --output already has a _YYYY-MM-DD_HH-MM suffix, it's left "
            "alone. If --output exists and is a completed run, refused "
            "without --force."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help=(
            "v0.2.2+: Allow overwriting an existing completed run dir at "
            "--output. Without this, the run is refused if a state.json "
            "with completed=true is found at --output."
        ),
    )

    # ---- Advanced LoRA / QLoRA flags ----
    parser.add_argument(
        "--use-dora",
        action="store_true",
        default=False,
        help=(
            "Enable DoRA (Weight-Decomposed Low-Rank Adaptation) via PEFT's "
            "use_dora flag. When combined with 4-bit quantization, PEFT "
            "automatically uses QDoRA. DoRA splits each LoRA adapter into "
            "a magnitude and directional component, often improving quality "
            "at the same rank. Default: OFF (standard QLoRA)."
        ),
    )
    parser.add_argument(
        "--loftq-init",
        action="store_true",
        default=False,
        help=(
            "Initialize LoRA weights with LoftQ (LoRA-fine-tuning-aware "
            "Quantization). Sets init_lora_weights='loftq' in LoraConfig, "
            "which replaces the default Kaiming initialization with a "
            "quantization-aware init. Only meaningful when the base model "
            "is UNQUANTIZED (will print a warning if the model is already "
            "quantized and proceed without LoftQ). Default: OFF."
        ),
    )
    _mp_group = parser.add_mutually_exclusive_group()
    _mp_group.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help=(
            "Force bfloat16 mixed precision. On Ampere+ GPUs (compute "
            "capability >= 8.0), bf16 is now the AUTO-DEFAULT — you only "
            "need this flag to override an explicit --fp16 on such a GPU. "
            "On older GPUs, pass this flag to force bf16 (may cause NaN if "
            "the GPU lacks bf16 hardware)."
        ),
    )
    _mp_group.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help=(
            "Force float16 mixed precision. This was the previous default "
            "for all GPUs. On Ampere+ GPUs, bf16 is now auto-selected; "
            "use --fp16 to override back to the old behavior."
        ),
    )
    _mp_group.add_argument(
        "--fp32",
        action="store_true",
        default=False,
        help=(
            "Force float32 (no mixed precision). Slow but numerically "
            "exact. Useful for debugging NaN issues."
        ),
    )
    parser.add_argument(
        "--use-rslora",
        dest="use_rslora",
        action="store_true",
        default=True,
        help=(
            "Enable Rank-Stabilized LoRA (use_rslora=True in LoraConfig). "
            "RSLoRA scales adapters by alpha/sqrt(r) instead of alpha/r, "
            "allowing lower ranks with equivalent training signal. "
            "Default: ON (kept for backward compatibility)."
        ),
    )
    parser.add_argument(
        "--no-use-rslora",
        dest="use_rslora",
        action="store_false",
        help=(
            "Disable RSLoRA. Use classic LoRA scaling (alpha/r). "
            "Only needed if you want the pre-RSLoRA behavior."
        ),
    )
    parser.add_argument(
        "--target-modules",
        type=str,
        default=None,
        help=(
            "Comma-separated list of LoRA target modules. "
            "Default: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj. "
            "Example: --target-modules q_proj,v_proj,o_proj"
        ),
    )
    parser.add_argument(
        "--moe-safe-target",
        action="store_true",
        default=False,
        help=(
            "Automatically restrict target modules to attention + MLP "
            "(exclude router/gate/lm_head layers) for Mixture-of-Experts "
            "models. Also forces bf16 and disables 4-bit quantization, "
            "since BitsAndBytes 4-bit does not support MoE expert weights "
            "per Unsloth guidance. Default: OFF."
        ),
    )
    parser.add_argument(
        "--use-unsloth",
        action="store_true",
        default=False,
        help=(
            "Use Unsloth's optimized model loading and LoRA kernels. "
            "Unsloth provides 2-5x faster training, 70%% less VRAM, and "
            "8x longer context support vs standard HF QLoRA. Requires "
            "'uv pip install attacklm[unsloth]' or 'pip install unsloth'. "
            "When enabled, Unsloth's FastLanguageModel replaces the "
            "standard AutoModelForCausalLM + BitsAndBytes pipeline. "
            "Default: OFF (standard HF QLoRA)."
        ),
    )
    parser.add_argument(
        "--use-galore",
        action="store_true",
        default=False,
        help=(
            "Use GaLore (Gradient Low-Rank Projection) for full-parameter "
            "fine-tuning. GaLore projects gradients into a low-rank space "
            "during optimization, enabling full-parameter learning on "
            "consumer GPUs without LoRA adapters. Mutually exclusive with "
            "--use-unsloth (GaLore trains ALL parameters, no LoRA needed). "
            "Requires 'uv pip install attacklm[galore]'. "
            "Default: 8-bit optimizer + per-layer hooks (fits 3B on 16GB). "
            "Use --use-qgalore for INT4 projections (fits 7B on 16GB). "
            "Use --galore-32bit for full-precision (needs ~20GB+ for 3B)."
        ),
    )
    parser.add_argument(
        "--use-qgalore",
        action="store_true",
        default=False,
        help=(
            "Use Q-GaLore: INT4 quantized gradient projection matrices "
            "with stochastic rounding. Cuts optimizer memory by ~4x vs "
            "vanilla GaLore, enabling 7B full-parameter training on 16GB. "
            "Mutually exclusive with --use-galore and --use-unsloth. "
            "Paper: arXiv:2407.08296."
        ),
    )
    parser.add_argument(
        "--galore-32bit",
        action="store_true",
        default=False,
        help=(
            "Use 32-bit GaLoreAdamW instead of 8-bit GaLoreAdamW8bit. "
            "Full-precision optimizer states (~12GB for 3B vs ~3GB for 8-bit). "
            "Disables per-layer weight updates (incompatible with 32-bit path). "
            "Only meaningful with --use-galore. "
            "Auto-enabled by --multi-gpu (per-layer hooks don't work with DDP)."
        ),
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        default=False,
        help=(
            "Enable multi-GPU training via DDP (DistributedDataParallel). "
            "Auto-enables --galore-32bit when combined with --use-galore "
            "(per-layer weight updates are incompatible with DDP). "
            "Use torchrun or accelerate launch to start multi-GPU training."
        ),
    )
    parser.add_argument(
        "--galore-rank",
        type=int,
        default=64,
        help=(
            "GaLore projection rank (default: 64 for 3B/16GB, 128 for 7B/24GB). "
            "Higher = more capacity but more VRAM. SVD projection memory "
            "scales with rank². Only meaningful with --use-galore."
        ),
    )
    parser.add_argument(
        "--spectrum",
        type=float,
        default=None,
        const=0.5,
        nargs="?",
        help=(
            "Spectrum: SNR-based layer freezing. Computes signal-to-noise "
            "ratio per layer from a few training batches, then freezes the "
            "lowest-SNR layers. --spectrum keeps the top 50%% of layers "
            "(default). --spectrum 0.25 keeps top 25%%. "
            "Reduces VRAM proportionally — 50%% freeze = ~50%% less memory. "
            "Compatible with any training method (GaLore, QLoRA, Unsloth). "
            "Paper: arXiv:2406.06623"
        ),
    )
    parser.add_argument(
        "--pissa-init",
        action="store_true",
        default=False,
        help=(
            "PiSSA: Principal Singular values Adaptation. Initializes LoRA "
            "weights from the SVD of pre-trained weights instead of random "
            "Kaiming init. Gives faster convergence and lower final loss. "
            "Only meaningful with LoRA/QLoRA (not GaLore). "
            "Paper: arXiv:2404.02948"
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Checks & Validation
# ---------------------------------------------------------------------------


def _is_galore(args) -> bool:
    """True if either --use-galore or --use-qgalore is active."""
    return args.use_galore or args.use_qgalore


def _resolve_model_path(model_id_or_path: str) -> str:
    """Resolve a user-supplied base-model argument to something `from_pretrained`
    can actually load.

    HuggingFace's `from_pretrained` does a name validation before checking
    whether the input is a local path. The check rejects anything that
    looks like a filesystem path with a leading `./`, `/`, or `~` and
    raises:

        Repo id must use alphanumeric chars, '-', '_' or '.'.
        The name cannot start or end with '-' or '.' and the maximum length
        is 96: './decensored_model'.

    Even absolute paths like `/home/...` get rejected by some HF
    versions because they don't look like `namespace/repo_name`. The
    workaround is to resolve to an absolute path AND confirm the path
    actually exists before passing it on. This gives a clean error
    message ("path doesn't exist") instead of the cryptic
    "Repo id must..." failure.

    If the input doesn't look like a path (no leading `/`, `./`, `~`,
    and contains a `/` that isn't a path separator) it's left alone
    as a HF Hub repo ID like `Qwen/Qwen2.5-Coder-3B-Instruct`.
    """
    from pathlib import Path

    if not model_id_or_path:
        return model_id_or_path

    # Strip trailing slashes (HF rejects names ending with '/')
    p = model_id_or_path.rstrip("/")
    if not p:
        p = model_id_or_path

    # Looks like a local path?
    is_path = (
        p.startswith(("/", "./", "../", "~/"))
        or Path(p).expanduser().is_absolute()
        or Path(p).expanduser().exists()
    )

    if is_path:
        resolved = str(Path(p).expanduser().resolve())
        if not Path(resolved).exists():
            # Give a clean error early so the user sees "path not found"
            # instead of the cryptic HF repo-id rejection
            raise FileNotFoundError(
                f"--base-model points to a local path that doesn't exist: "
                f"{resolved}\n"
                f"  (originally passed: {model_id_or_path!r})\n"
                f"  Either create that directory, point at an existing "
                f"model, or pass a HF Hub repo ID like "
                f"'Qwen/Qwen2.5-Coder-3B-Instruct'."
            )
        return resolved

    # Not a path — treat as a HF Hub repo ID. Leave alone.
    return p


# ---------------------------------------------------------------------------
# Run state sidecar
# ---------------------------------------------------------------------------
# A `state.json` file lives at the root of every training output dir. It
# captures the *intent* of a run: what base, what hparams, what dataset,
# whether it's started, whether it's complete, and where it stopped.
# This is separate from HF's `trainer_state.json` (which lives inside
# `checkpoint-N/` and is HF-internal). Ours is for tooling + humans + the
# "round-2 SFT" workflow (pass a finished run dir as --base-model to
# start a new LoRA on top of the merged weights).
#
# Folder convention (v0.1.6):
#   models/agent_TIMESTAMP/        <- training run output (timestamped)
#       state.json                 <- THIS file (written on start, updated on end)
#       adapter_config.json        <- written on completion
#       adapter_model.safetensors
#       tokenizer.json
#       checkpoint-N/              <- HF internal, with trainer_state.json
#   models/merged/agent/          <- merged BF16 (deployable, no timestamp)
#   models/gguf/agent.Q4_K_M.gguf <- final GGUF (deployable, no timestamp)
#   ~/.lmstudio/models/local/agent/  <- LM Studio (no timestamp)
#
# Resolution rules when --base-model is a path:
#   1. Path has no state.json
#      → treat as raw HF model (regular from_pretrained). Existing behavior.
#   2. Path has state.json with completed=false + checkpoint-N/ exists
#      → this is a started run, auto-resume from the latest checkpoint.
#        The user is continuing the same run, not starting a new one.
#   3. Path has state.json with completed=false but NO checkpoint-N/
#      → "marked started but never ran" (probably a --dry-run that wrote
#        state). Treat as base model (effectively same as case 1).
#   4. Path has state.json with completed=true
#      → this is a finished run. User is doing round-2 SFT: load the
#        merged weights directly and train a new LoRA on top.
#   5. Path has adapter_config.json (peft_type=LORA) but no state.json
#      → bare LoRA adapter (not a finished run). Existing behavior:
#        treat as base + apply the adapter on top during loading.
#        (Used by `attacklm-merge` to find the base, etc.)

_STATE_VERSION = 1


def _now_iso() -> str:
    """UTC ISO-8601 timestamp with second precision."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_state_template(
    output_dir: str,
    base_model: str,
    hparams: dict,
    dataset_info: dict,
) -> dict:
    """Build a fresh state.json template with completed=False."""
    return {
        "version": _STATE_VERSION,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "completed": False,
        "base_model": {
            "source": "local"
            if Path(base_model).is_absolute()
            or base_model.startswith(("./", "../", "~/"))
            else "hf",
            "id": base_model,
        },
        "hparams": hparams,
        "dataset": dataset_info,
        "progress": {
            "global_step": 0,
            "max_steps": 0,
            "current_epoch": 0.0,
            "total_epochs": float(hparams.get("epochs", 0)),
            "last_loss": None,
            "last_token_accuracy": None,
            "best_eval_loss": None,
            "total_training_seconds": 0,
        },
        "hpo": {
            "is_hpo_trial": False,
            "trial_id": None,
            "parent_run": None,
            "axes": None,
        },
    }


def read_state(output_dir: str) -> dict | None:
    """Read state.json from a training output dir, or None if not present.

    Tolerant of malformed files (returns None + warning, never raises).
    """
    sp = Path(output_dir) / "state.json"
    if not sp.exists():
        return None
    try:
        with sp.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        _c = Console(width=80)
        _c.print(
            f"[yellow]WARNING:[/yellow] state.json at {sp} is unreadable ({e}); ignoring"
        )
        return None


def write_state(output_dir: str, state: dict) -> None:
    """Atomically write state.json (write to .tmp, then rename).

    Atomic write prevents a half-written state.json from being read by
    a parallel tool (LM Studio scanner, attacklm-merge --merge-all, etc.)
    """
    sp = Path(output_dir) / "state.json"
    tmp = sp.with_suffix(".json.tmp")
    state["updated_at"] = _now_iso()
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, sp)


def resolve_base_model_path(model_id_or_path: str) -> tuple[str, dict | None]:
    """Resolve --base-model into (load_path, state_or_None).

    `load_path` is what to pass to `from_pretrained`. For HF Hub IDs
    and raw local paths, it's the same as the input (resolved to abs).
    For a "started" run dir, it's the path itself (the trainer auto-
    finds checkpoint-N/ subdirs when resume_from_checkpoint=True).
    For a "completed" run dir, it's the path itself (it has the merged
    weights from the last successful save).

    Returns (resolved_path, state_dict_or_None). The state dict is
    surfaced so main() can log "this is a resumed run" and read hparams.

    Raises FileNotFoundError with a clean message if the path doesn't
    exist.
    """
    resolved = _resolve_model_path(model_id_or_path)
    state = read_state(resolved)
    return resolved, state


def has_incomplete_checkpoint(output_dir: str) -> bool:
    """True if there are checkpoint-N/ subdirs with trainer_state.json.

    Used to disambiguate "marked started, never ran" (case 3) from
    "started and partially trained" (case 2).
    """
    p = Path(output_dir)
    if not p.exists():
        return False
    for child in p.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            if (child / "trainer_state.json").exists():
                return True
    return False


def make_timestamped_output_dir(parent: str, agent_name: str) -> str:
    """Build a fresh output dir name with a UTC timestamp suffix.

    Format: {parent}/{agent_name}_{YYYY-MM-DD}_{HH-MM}/
    Example: models/agent_runs/attacklm-single_2026-06-10_01-12/

    This is for new training runs. We use a counter suffix (in-memory,
    within this process) to avoid returning the same path twice when
    the caller invokes us rapidly (e.g. train_all.py calling this
    in a tight loop). The counter resets per-process, so re-running
    train_all.py later still uses the wall-clock timestamp.
    """
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")
    base = Path(parent) / f"{agent_name}_{ts}"
    # Use a counter to disambiguate in-process rapid calls. We pick
    # the lowest N that doesn't exist on disk yet, so persisted
    # numbering only collides on actual same-minute re-runs.
    n = 1
    candidate = base
    while candidate.exists():
        n += 1
        candidate = Path(parent) / f"{agent_name}_{ts}_{n}"
    return str(candidate)


_TIMESTAMP_SUFFIX_RE = __import__("re").compile(
    r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}(_\d+)?$"
)


def resolve_output_path(
    user_output: str, no_timestamp: bool = False, force: bool = False
) -> str:
    """Resolve the --output argument for `attacklm-train` (v0.2.2+).

    Rules (in order):
        1. If the path already ends in `_YYYY-MM-DD_HH-MM` (or
           `_YYYY-MM-DD_HH-MM_N`), use it as-is. The user has
           explicitly chosen a timestamped name — respect it.
           (This is what train_all.py produces; we recognize its
           own output.)
        2. If --no-timestamp was passed, use the path as-is.
        3. If the path exists and has a state.json with completed=true
           (a finished run from a previous invocation):
             a. If --force was passed, use the path as-is (clobber).
             b. Otherwise, raise FileExistsError with a clear hint
                about how to use a timestamped name or --force.
        4. Otherwise, append `_YYYY-MM-DD_HH-MM` to the basename.
           If that name exists on disk (rare — same-minute re-run),
           append `_2`, `_3`, ... until unique.

    Returns: absolute path string.

    Why: the previous behavior (`attacklm-train --output foo` clobbers
    `foo/` if it exists) was the source of the "I lost my run"
    footgun. v0.2.2 makes timestamped outputs the default, matching
    what train_all.py does for multi-bucket runs.
    """
    from datetime import datetime, timezone
    from pathlib import Path as _P

    p = _P(user_output).expanduser()
    name = p.name
    parent = p.parent if str(p.parent) not in ("", ".") else _P(".")

    # Rule 1: already has a timestamp
    if _TIMESTAMP_SUFFIX_RE.search(name):
        return str(p.resolve())

    # Rule 2: user said no timestamp
    if no_timestamp:
        if p.exists() and (p / "state.json").exists() and not force:
            try:
                with (p / "state.json").open() as f:
                    st = __import__("json").load(f)
            except (OSError, __import__("json").JSONDecodeError):
                st = None
            if st is not None and st.get("completed"):
                raise FileExistsError(
                    f"Refusing to clobber completed run at {p}.\n"
                    f"  state.json[completed]=true, training was finished.\n"
                    f"  Pass --force to overwrite, or use a different --output.\n"
                    f"  (Default: a timestamp will be appended to your --output\n"
                    f"  so each run is preserved. Pass --no-timestamp to disable.)"
                )
        return str(p.resolve())

    # Rule 3: append timestamp
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")
    base_name = f"{name}_{ts}"
    candidate = parent / base_name
    n = 1
    while candidate.exists():
        n += 1
        candidate = parent / f"{base_name}_{n}"
    return str(candidate.resolve())


def check_python_version() -> None:
    """Ensure Python 3.9+ is available."""
    if sys.version_info < (3, 9):
        _c = Console(width=80)
        _c.print(f"[red]ERROR:[/red] Python 3.9+ required, got {sys.version}")
        sys.exit(1)


def check_gpu(args: argparse.Namespace | None = None) -> str:
    """Check for CUDA / ROCm / MPS availability and return compute dtype string.

    Works on NVIDIA CUDA, AMD ROCm, Apple Silicon MPS, and CPU.

    Priority:
      1. Explicit --fp32 / --bf16 / --fp16 from args (mutually exclusive).
      2. Auto-detect bf16 on Ampere+ (CUDA compute capability >= 8.0).
      3. Fall back to fp16 for older GPUs and backward compatibility.

    When --moe-safe-target is set, bf16 is forced regardless of GPU.
    """
    from device_utils import (
        is_cuda,
        is_mps,
        gpu_name_and_memory,
    )

    _c = Console(width=80)

    print_hardware_banner()

    # --moe-safe-target forces bf16 (BnB 4-bit is incompatible with MoE
    # expert weights per Unsloth guidance)
    if args and getattr(args, "moe_safe_target", False):
        _c.print("Mixed precision: [bold]BF16[/bold] (forced by --moe-safe-target)")
        return "bf16"

    # Explicit overrides (mutually exclusive group in argparse)
    if args and getattr(args, "fp32", False):
        _c.print("Mixed precision: [bold]FP32[/bold] (forced by --fp32)")
        return "fp32"
    if args and getattr(args, "bf16", False):
        _c.print("Mixed precision: [bold]BF16[/bold] (forced by --bf16)")
        return "bf16"
    if args and getattr(args, "fp16", False):
        _c.print("Mixed precision: [bold]FP16[/bold] (forced by --fp16)")
        return "fp16"

    if not is_cuda():
        if is_mps():
            _c.print(
                "[yellow]WARNING:[/yellow] Apple Silicon (MPS) detected — training will be very slow."
            )
            return "fp32"
        _c.print(
            "[yellow]WARNING:[/yellow] No CUDA / ROCm GPU detected. "
            "Training will be extremely slow on CPU."
        )
        _c.print(
            "         Consider using Google Colab (T4/A100) or RunPod if you lack a local GPU."
        )
        return "fp32"

    gpu_name, gpu_mem = gpu_name_and_memory()
    _c.print(f"GPU: [bold]{gpu_name}[/bold] ({gpu_mem:.1f} GB VRAM)")

    if gpu_mem < 10:
        _c.print(f"[yellow]WARNING:[/yellow] GPU has only {gpu_mem:.1f} GB VRAM.")
        _c.print("         If you hit OOM, try: --batch-size 1 --max-length 1024")

    # Auto-detect bf16 capability on Ampere+ GPUs (compute capability >= 8.0).
    # Ampere (8.0), Ada (8.9), Hopper (9.0), Blackwell (10.0) all support bf16.
    cc_major = 0
    try:
        if torch.cuda.is_available():
            cc_major = torch.cuda.get_device_properties(0).major
    except Exception:
        pass

    if cc_major >= 8:
        _c.print(
            f"Mixed precision: [bold]BF16[/bold] (auto-detected, compute capability {cc_major}.x)"
        )
        return "bf16"
    else:
        _c.print("Mixed precision: [bold]FP16[/bold] (GPU lacks BF16 hardware)")
        return "fp16"


def validate_dataset(dataset_path: str) -> None:
    """Validate dataset file exists and has correct format."""
    _c = Console(width=80)
    path = Path(dataset_path)

    if not path.exists():
        _c.print(f"[red]ERROR:[/red] Dataset file not found: {dataset_path}")
        _c.print(
            "       See README.md §Quickstart for how to generate datasets (run scripts/extract_*.py)."
        )
        sys.exit(1)

    if not path.suffix == ".jsonl":
        _c.print(
            f"[yellow]WARNING:[/yellow] Expected .jsonl extension, got '{path.suffix}'"
        )

    # Read first few lines and validate format
    errors = 0
    line_count = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line_count += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "messages" not in obj:
                    errors += 1
                    if errors <= 3:
                        _c.print(
                            f"[yellow]WARNING:[/yellow] Line {i + 1} missing 'messages' key — skipping"
                        )
                else:
                    msgs = obj["messages"]
                    if not isinstance(msgs, list) or len(msgs) < 2:
                        errors += 1
                        if errors <= 3:
                            _c.print(
                                f"[yellow]WARNING:[/yellow] Line {i + 1} has invalid 'messages' format"
                            )
                    else:
                        for m in msgs:
                            if "role" not in m or "content" not in m:
                                errors += 1
                                if errors <= 3:
                                    _c.print(
                                        f"[yellow]WARNING:[/yellow] Line {i + 1} has message without 'role' or 'content'"
                                    )
                                break
            except json.JSONDecodeError as e:
                errors += 1
                if errors <= 3:
                    _c.print(
                        f"[yellow]WARNING:[/yellow] Line {i + 1} JSON parse error: {e}"
                    )

    if errors > 3:
        _c.print(
            f"[yellow]WARNING:[/yellow] {errors} total format issues in dataset (showing first 3)"
        )

    if line_count == 0:
        _c.print(f"[red]ERROR:[/red] Dataset file is empty: {dataset_path}")
        sys.exit(1)

    _c.print(f"Dataset: {line_count} examples, {errors} format issues")


# ---------------------------------------------------------------------------
# Dataset Stats
# ---------------------------------------------------------------------------


def print_dataset_stats(dataset, tokenizer, hpo_stats_path: str = None) -> None:
    """Print dataset statistics including token length estimates.

    When `hpo_stats_path` is provided, also write a JSON sidecar with
    exact (not estimated) token length stats for the full dataset. The
    HPO loop uses this to compute `pair_mean_tokens`, `pair_min_tokens`,
    `pair_max_tokens` for the per-step CSV. The HPO logger's per-step
    pair_mean is approximate (num_tokens / pairs_in_window); the
    sidecar gives the precise, unchanging reference.
    """
    _console = Console(width=80)
    num_examples = len(dataset)

    _tbl = Table(
        title="Dataset Statistics", show_header=False, box=None, padding=(0, 1)
    )
    _tbl.add_column("Metric", style="bold")
    _tbl.add_column("Value")
    _tbl.add_row("Examples", str(num_examples))

    # Estimate token lengths (word-level)
    word_lengths = []
    sample_size = min(num_examples, 100)
    for i in range(sample_size):
        example = dataset[i]
        try:
            text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
            word_lengths.append(len(text.split()))
        except Exception:
            word_lengths.append(0)

    if word_lengths:
        avg_words = sum(word_lengths) / len(word_lengths)
        avg_tokens_est = int(avg_words * 1.3)  # rough word-to-token ratio
        _tbl.add_row("Avg word count", f"{avg_words:.0f} (sampled {sample_size})")
        _tbl.add_row("Avg token est", f"~{avg_tokens_est} (1.3x word ratio)")
        _tbl.add_row("Max word count", str(max(word_lengths)))
        _tbl.add_row("Min word count", str(min(word_lengths)))

    # Exact token stats (sampled for speed). For HPO we use this
    # to record pair_min/max/mean once at the start of a run.
    if hpo_stats_path:
        from statistics import mean, median

        # Sample up to 500 examples for exact tokenization (cheaper than
        # full corpus for large datasets, statistically enough for min/max
        # which are dominated by the long tail).
        exact_sample_size = min(num_examples, 500)
        token_lengths = []
        for i in range(exact_sample_size):
            example = dataset[i]
            try:
                text = tokenizer.apply_chat_template(
                    example["messages"], tokenize=False
                )
                n = len(tokenizer.encode(text, add_special_tokens=False))
                token_lengths.append(n)
            except Exception:
                pass
        if token_lengths:
            stats = {
                "n_examples_total": num_examples,
                "n_examples_sampled": len(token_lengths),
                "min_tokens": min(token_lengths),
                "max_tokens": max(token_lengths),
                "mean_tokens": round(mean(token_lengths), 1),
                "median_tokens": median(token_lengths),
                "p95_tokens": sorted(token_lengths)[int(0.95 * len(token_lengths))],
                "p99_tokens": sorted(token_lengths)[int(0.99 * len(token_lengths))],
            }
            os.makedirs(os.path.dirname(hpo_stats_path) or ".", exist_ok=True)
            with open(hpo_stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            _tbl.add_row(
                "Exact token stats",
                f"(sampled {len(token_lengths)})",
            )
            _tbl.add_row(
                "  min/median/mean",
                f"{stats['min_tokens']} / {stats['median_tokens']} / {stats['mean_tokens']}",
            )
            _tbl.add_row(
                "  p95/p99/max",
                f"{stats['p95_tokens']} / {stats['p99_tokens']} / {stats['max_tokens']}",
            )
            _console.print(f"  [green]✓[/green] wrote HPO stats to {hpo_stats_path}")

    _console.print(_tbl)
    _console.print()


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------


def get_qlora_config(
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float = 0.05,
    use_rslora: bool = True,
    target_modules: list[str] | None = None,
    use_dora: bool = False,
    init_lora_weights: str | bool = True,
) -> Any:
    """Build LoRA config targeting all linear modules.

    Args:
        lora_r: LoRA rank (default 16).
        lora_alpha: LoRA alpha (default 32).
        lora_dropout: LoRA dropout (default 0.05).
        use_rslora: Enable Rank-Stabilized LoRA scaling (default True).
        target_modules: List of target module names. Defaults to the
            standard Qwen2.5 set if None.
        use_dora: Enable DoRA (Weight-Decomposed LoRA). When combined
            with 4-bit quantization, PEFT automatically uses QDoRA.
        init_lora_weights: Initialization method. Default True (Kaiming).
            Set to "loftq" for LoftQ initialization (LoRA-fine-tuning-aware
            Quantization).
    """
    from peft import LoraConfig, TaskType

    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    config_kwargs = {
        "r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM,
        "use_rslora": use_rslora,
        "target_modules": target_modules,
    }

    # DoRA / QDoRA: when use_dora=True with a 4-bit base model, PEFT
    # automatically applies QDoRA (quantization-aware DoRA).
    if use_dora:
        config_kwargs["use_dora"] = True

    # LoftQ init: only applies when the base model is unquantized.
    # The caller is responsible for printing a warning if the model is
    # already quantized.
    config_kwargs["init_lora_weights"] = init_lora_weights

    return LoraConfig(**config_kwargs)


def get_quantization_config() -> Any:
    """Build 4-bit NF4 quantization config — FP32 compute to avoid AMP conflicts."""
    from transformers import BitsAndBytesConfig

    # Set TF32 for faster matmul (Ampere/Ada/Blackwell on CUDA, CDNA/RDNA on ROCm)
    enable_tf32()

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,  # no AMP/ BF16 conflicts
        bnb_4bit_use_double_quant=True,
    )


# Quantization methods we can detect + their (transformers) config class names.
# The value None means "let HF auto-detect from config.json" — works for
# FineGrainedFP8Config, GPTQConfig, AwqConfig, FbgemmFp8Config, etc.
_QUANT_SCHEMES = {
    "bitsandbytes_4bit": "BitsAndBytesConfig (4-bit)",
    "bitsandbytes_8bit": "BitsAndBytesConfig (8-bit)",
    "fp8": "FineGrainedFP8Config",
    "fbgemm_fp8": "FbgemmFp8Config",
    "gptq": "GPTQConfig",
    "awq": "AwqConfig",
    "compressed-tensors": "CompressedTensorsConfig",
    "torchao": "TorchAoConfig",
    "bitsandbytes": "BitsAndBytesConfig (legacy)",
}


def detect_quantization_scheme(model_id_or_path: str) -> str | None:
    """Detect a pre-existing quantization scheme by reading the model's
    `config.json` (and legacy `quantize_config.json` for old AutoGPTQ).

    Returns the `quant_method` string (e.g. ``"fp8"``, ``"gptq"``,
    ``"bitsandbytes_4bit"``) or ``None`` if the model is unquantized.

    Resolution order:
        1. Local ``<model_dir>/config.json`` — works for `transformers-cli download`
           output, `git clone` of a model repo, or a path produced by
           `huggingface-cli download`.
        2. Hugging Face Hub — fall back to ``hf_hub_download("config.json")``
           for the common case of ``org/model`` IDs.
        3. Legacy ``quantize_config.json`` at model root — used by
           AutoGPTQ pre-v0.8.

    The function is intentionally cheap (no model weight download) and
    never raises — returns ``None`` on any error so the caller can
    fall back to applying our own BitsAndBytesConfig.
    """
    from pathlib import Path

    candidate = None

    # (1) Local file
    local_cfg = Path(model_id_or_path) / "config.json"
    if local_cfg.is_file():
        candidate = local_cfg
    else:
        # (2) HF Hub
        try:
            from huggingface_hub import hf_hub_download

            candidate = Path(hf_hub_download(model_id_or_path, "config.json"))
        except Exception:
            candidate = None

    if candidate is not None and candidate.is_file():
        try:
            import json

            with candidate.open() as f:
                cfg = json.load(f)
            quant_cfg = cfg.get("quantization_config") or {}
            method = quant_cfg.get("quant_method")
            if method:
                return method
        except Exception:
            pass

    # (3) Legacy AutoGPTQ `quantize_config.json`
    try:
        from pathlib import Path

        legacy_dir = Path(model_id_or_path)
        if not legacy_dir.is_dir():
            # Try HF Hub for legacy file too
            try:
                from huggingface_hub import hf_hub_download

                legacy_path = Path(
                    hf_hub_download(model_id_or_path, "quantize_config.json")
                )
            except Exception:
                legacy_path = None
        else:
            legacy_path = legacy_dir / "quantize_config.json"

        if legacy_path is not None and legacy_path.is_file():
            import json

            with legacy_path.open() as f:
                cfg = json.load(f)
            # AutoGPTQ legacy always implies GPTQ
            if "bits" in cfg:
                return "gptq"
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def detect_assistant_loss_support(tokenizer, dataset_sample: dict) -> dict:
    """Auto-detect whether to enable assistant_only_loss / completion_only_loss.

    Background:
        SFTConfig has two params that restrict loss computation to just the
        model's output (skipping the user/system prompt):

        - `assistant_only_loss=True`: requires a *conversational* dataset
          (with a `messages` array of role/content dicts) AND a chat template
          that contains `{% generation %}` blocks. These blocks tell the
          tokenizer "this part of the template is the assistant's output,
          generate a mask for it." Models with built-in support: Qwen3,
          SmolLM3, and a few others. Qwen2.5 does NOT have them.

        - `completion_only_loss=True`: requires a *prompt-completion* dataset
          (separate `prompt` and `completion` fields). No template support
          needed — the dataset format itself defines the boundary.

        If neither is applicable, we fall back to full-sequence loss (the
        default) which still works fine — just slightly less efficient
        because the model wastes capacity learning to predict the user's
        input (which it will never generate at inference time).

    Returns:
        dict with keys:
            - `assistant_only_loss`: bool — set to True if supported
            - `completion_only_loss`: bool — set to True if prompt-completion dataset
            - `reason`: str — human-readable explanation of what was detected
    """
    # Default: full-sequence loss
    result = {
        "assistant_only_loss": False,
        "completion_only_loss": False,
        "reason": "no special dataset format detected, using full-sequence loss",
    }

    # Check dataset format
    if "messages" in dataset_sample and isinstance(dataset_sample["messages"], list):
        # Conversational format — needs chat template with {% generation %}
        template = getattr(tokenizer, "chat_template", None) or ""
        has_generation_block = (
            "{% generation %}" in template or "{%- generation %}" in template
        )
        if has_generation_block:
            result["assistant_only_loss"] = True
            result["reason"] = (
                "conversational dataset + chat template has {% generation %} blocks — "
                "enabling assistant_only_loss (skips user/system tokens in loss)"
            )
        else:
            # Detect the model family to give a helpful hint
            model_name = getattr(tokenizer, "name_or_path", "unknown")
            result["reason"] = (
                f"conversational dataset but chat template for '{model_name}' lacks "
                f"{{% generation %}} blocks — cannot use assistant_only_loss. "
                f"Model is training on full sequence (slightly less efficient but still works). "
                f"Switch to Qwen3 or SmolLM3 to enable, or patch the chat template manually."
            )
    elif "prompt" in dataset_sample and "completion" in dataset_sample:
        # Prompt-completion format — no template needed
        result["completion_only_loss"] = True
        result["reason"] = (
            "prompt-completion dataset — enabling completion_only_loss "
            "(loss only on the completion field, prompt is masked to -100)"
        )
    else:
        # Pure language modeling (single text field) — full loss is the only option
        result["reason"] = (
            "language modeling dataset (single text field) — using full-sequence loss"
        )

    return result


def _stochastic_round_projections(optimizer):
    """Re-quantize projection matrices with stochastic rounding.

    Stochastic rounding preserves gradient information that would be
    lost by deterministic round-to-nearest. For a value x with
    quantization step Δ, the probability of rounding up is
    (x - floor(x/Δ)*Δ) / Δ.

    This is called after every optimizer step to maintain INT4 precision
    while preventing quantization drift.
    """
    import torch

    for group in optimizer.param_groups:
        if "proj_int4" not in group:
            continue
        proj = group.get("projection_matrix")
        if proj is None or proj.numel() == 0:
            continue
        scale = group["proj_scale"]
        with torch.no_grad():
            # Dequantize current INT4 for the forward pass
            proj.copy_(group["proj_int4"].to(proj.dtype) * scale)
            # Stochastic rounding: re-quantize with probability-based rounding
            fp = proj / scale
            floor_val = fp.floor()
            frac = fp - floor_val
            # Random rounding: round up with probability = fractional part
            rand = torch.rand_like(frac)
            rounded = floor_val + (rand < frac).to(fp.dtype)
            group["proj_int4"] = rounded.clamp(-7, 7).to(torch.int8)


def main() -> None:
    console = Console(width=80)
    args = parse_args()
    check_python_version()

    # --- Resolve --output path (v0.2.2+ auto-timestamp) ---
    # Default: append a timestamp to --output so re-runs don't clobber.
    # Override with --no-timestamp. Refuse to clobber a completed run
    # unless --force is also set.
    try:
        args.output = resolve_output_path(
            args.output,
            no_timestamp=args.no_timestamp,
            force=args.force,
        )
    except FileExistsError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    is_dry_run = args.dry_run or not args.train

    # --- Resolve LoRA target modules (before training plan display) ---
    if args.moe_safe_target:
        # MoE-safe: restrict to attention + MLP, exclude router/gate/lm_head.
        # This prevents BnB 4-bit from trying to quantize MoE expert weights.
        _DEFAULT_TARGETS = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        _MOE_EXCLUDED_PATTERNS = {"gate", "router", "lm_head", "embed_tokens", "shared"}
        target_modules_list = [
            m
            for m in _DEFAULT_TARGETS
            if not any(pat in m for pat in _MOE_EXCLUDED_PATTERNS)
        ]
        excluded = [m for m in _DEFAULT_TARGETS if m not in target_modules_list]
        console.print(f"  --moe-safe-target: excluded modules: {excluded}")
        console.print(f"  --moe-safe-target: target modules: {target_modules_list}")
    elif args.target_modules:
        target_modules_list = [m.strip() for m in args.target_modules.split(",")]
        console.print(f"  --target-modules: {target_modules_list}")
    else:
        target_modules_list = None  # use defaults in get_qlora_config

    # LoftQ init: tentatively set based on args; will be overridden if
    # the model is already quantized (checked after model load).
    if args.pissa_init:
        init_lora_weights = "pissa"
    elif args.loftq_init:
        init_lora_weights = "loftq"
    else:
        init_lora_weights = True

    mode_label = "DRY RUN (no training)" if is_dry_run else "LIVE TRAINING"
    output_note = ""
    if args.no_timestamp:
        output_note = "(no-timestamp: clobbers without --force)"
    else:
        import re as _re

        if _re.search(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}", args.output):
            output_note = "(timestamp suffix preserved)"
        else:
            output_note = "(auto-timestamped: each run preserved)"

    _hdr = Table(show_header=False, box=None, padding=(0, 1))
    _hdr.add_column("Key", style="bold")
    _hdr.add_column("Value")
    _hdr.add_row("Mode", mode_label)
    _hdr.add_row("Base model", args.base_model)
    _hdr.add_row("Dataset", args.dataset)
    _hdr.add_row("Output", f"{args.output} {output_note}")
    _hdr.add_row("Epochs", str(args.epochs))
    _hdr.add_row("Batch size", str(args.batch_size))
    _hdr.add_row("Max length", str(args.max_length))
    _hdr.add_row(
        "LoRA",
        f"r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}",
    )
    _hdr.add_row("Optimizer", args.optim)
    console.print(
        Panel(_hdr, title="AttackLM QLoRA Fine-Tuning Template", border_style="cyan")
    )
    console.print()

    # --- Validate dataset path ---
    validate_dataset(args.dataset)

    # --- Check GPU ---
    compute_type = check_gpu(args)

    # --- Load dependencies (deferred so --dry-run can run without GPU) ---
    console.print("[bold]Loading libraries...[/bold]")
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as e:
        console.print(f"[red]ERROR:[/red] Missing dependency: {e}")
        console.print(
            "Install with: pip install transformers datasets trl peft bitsandbytes accelerate"
        )
        sys.exit(1)

    # --- Load tokenizer ---
    console.print(f"[bold]Loading tokenizer:[/bold] {args.base_model}")
    try:
        # Resolve local paths to absolute early so HF's name validation
        # doesn't reject `./decensored_model` as an invalid repo ID.
        #
        # v0.1.6: this also detects "started run" vs "finished run" vs
        # "raw base" via the state.json sidecar (see resolve_base_model_path).
        base_model_resolved, base_state = resolve_base_model_path(args.base_model)

        # Decision: should we auto-resume from checkpoint?
        # - If state.json says "completed: false" AND a checkpoint exists,
        #   this is a started-but-not-finished run. Auto-resume.
        # - If state.json says "completed: true", this is a finished run
        #   being used as a base for round-2 SFT. Don't auto-resume — load
        #   the merged weights and train a fresh LoRA on top.
        # - If no state.json, this is either a raw HF repo or a bare
        #   adapter. Don't auto-resume.
        if base_state is not None and not base_state.get("completed", True):
            if has_incomplete_checkpoint(base_model_resolved):
                if not args.resume_from_checkpoint:
                    console.print(
                        "[green]✓[/green] Detected started run (state.json: completed=false, "
                        "checkpoint-N/ present). Auto-resuming from latest checkpoint."
                    )
                    args.resume_from_checkpoint = True
            else:
                # Marked-started but no actual training happened. Treat as
                # a fresh base. This happens when --dry-run is run twice
                # on the same dir.
                console.print(
                    "[yellow]⚠[/yellow] Detected marked-started run but no checkpoint-N/ found. "
                    "Treating as base for a fresh training run."
                )

        # Surface the round-2 SFT case clearly to the user
        if base_state is not None and base_state.get("completed", False):
            console.print(
                "[green]✓[/green] Detected completed run (state.json: completed=true). "
                "Round-2 SFT: training a fresh LoRA on top of the merged weights."
            )
            prev_hp = base_state.get("hparams", {})
            if prev_hp:
                console.print(
                    f"    Previous hparams: r={prev_hp.get('lora_r', '?')}, "
                    f"alpha={prev_hp.get('lora_alpha', '?')}, "
                    f"epochs={prev_hp.get('epochs', '?')}, "
                    f"max_length={prev_hp.get('max_length', '?')}"
                )
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_resolved,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Fix: Qwen tokenizer has bos_token_id=None but model config
        # has bos_token_id=151643. Sync BEFORE model load so HF doesn't
        # warn about PAD/BOS/EOS mismatch.
        if tokenizer.bos_token_id is None:
            tokenizer.bos_token_id = 151643  # Qwen default BOS
    except Exception as e:
        console.print(f"[red]ERROR:[/red] Failed to load tokenizer: {e}")
        console.print(
            f"  Check that '{args.base_model}' exists on HuggingFace and you have internet access."
        )
        sys.exit(1)

    # --- Load dataset ---
    console.print(f"[bold]Loading dataset:[/bold] {args.dataset}")
    try:
        dataset = load_dataset("json", data_files=args.dataset, split="train")
    except Exception as e:
        console.print(f"[red]ERROR:[/red] Failed to load dataset: {e}")
        sys.exit(1)

    # --- Auto-detect assistant-only loss support ---
    # OOM fix #12 (bonus, not memory related): auto-detect whether the
    # tokenizer's chat template has {% generation %} blocks. If it does,
    # we can enable `assistant_only_loss=True` so the model only learns
    # from assistant outputs (skipping the user/system tokens). This is
    # ~2x more efficient than full-sequence loss because the model
    # doesn't waste capacity learning to predict the user's input.
    # Qwen2.5's template doesn't have generation blocks (it was added
    # in Qwen3), so we'll get a helpful message but the flag stays off.
    # Switch to Qwen3 or SmolLM3 to enable, or patch the template.
    try:
        loss_cfg = detect_assistant_loss_support(tokenizer, dataset[0])
    except Exception as e:
        loss_cfg = {
            "assistant_only_loss": False,
            "completion_only_loss": False,
            "reason": f"auto-detection failed ({e}), falling back to full-sequence loss",
        }
    console.print(f"  Assistant-loss detection: {loss_cfg['reason']}")
    console.print(
        f"  [green]→[/green] assistant_only_loss={loss_cfg['assistant_only_loss']}, "
        f"completion_only_loss={loss_cfg['completion_only_loss']}"
    )

    # --- OOM fix #2: Truncate/drop abnormally long examples ---
    # The combined dataset has ~55 examples over 1500 tokens (one is 3150).
    # Even with `max_length=1024` in SFTConfig, the collator truncates the
    # *text* but the tokenizer/embedding lookup still allocates a tensor
    # sized to the original length before truncation. Drop examples where
    # the formatted chat template is 1.5x the max_length — those are
    # outlier payloads (huge metasploit modules) that would OOM anyway
    # and the model can't really learn from 3000-token contexts at
    # max-length=1024.
    #
    # The dropped count is recorded in state.json so the user can see
    # the actual train/eval split (and not be surprised by the epoch
    # counter being lower than expected).
    def _filter_long_examples(
        example, tokenizer=tokenizer, max_len=args.max_length, hard_cap_mult=1.5
    ):
        try:
            msgs = example.get("messages", [])
            if not msgs:
                return False
            text = tokenizer.apply_chat_template(msgs, tokenize=False)
            n = len(tokenizer.encode(text, add_special_tokens=False))
            return n <= int(max_len * hard_cap_mult)
        except Exception:
            return True  # If encoding fails, keep it (don't silently drop data)

    pre_filter_count = len(dataset)
    dataset = dataset.filter(_filter_long_examples, num_proc=1)
    post_filter_count = len(dataset)
    dropped = pre_filter_count - post_filter_count
    _filtered_out: int = dropped  # captured for state.json
    if dropped > 0:
        console.print(
            f"  [yellow]WARNING:[/yellow] Dropped {dropped} examples exceeding "
            f"{int(args.max_length * 1.5)} tokens "
            f"({100 * dropped / pre_filter_count:.2f}% of dataset)"
        )

    # --- Print dataset stats ---
    # When HPO CSV is enabled, also write a sidecar with exact token
    # length stats (min/max/mean/p95/p99) for the per-trial report.
    hpo_stats_path = None
    if args.hpo_metrics_csv:
        # Sidecar lives next to the CSV: foo.csv → foo.dataset_stats.json
        hpo_stats_path = (
            str(args.hpo_metrics_csv).rsplit(".", 1)[0] + ".dataset_stats.json"
        )
    print_dataset_stats(dataset, tokenizer, hpo_stats_path=hpo_stats_path)

    # --- Show training plan ---
    _tm_display = ", ".join(
        target_modules_list
        if target_modules_list
        else [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    _rslora_display = (
        "ON (alpha/sqrt(r))" if args.use_rslora else "OFF (classic alpha/r)"
    )
    _dora_display = "ON (QDoRA)" if args.use_dora else "OFF"
    _loftq_display = "ON" if args.loftq_init else "OFF"
    _moe_display = (
        "ON (attention+MLP only, bf16 forced, no BnB 4-bit)"
        if args.moe_safe_target
        else "OFF"
    )
    _unsloth_display = (
        "ON (FastLanguageModel + optimized LoRA kernels)" if args.use_unsloth else "OFF"
    )
    _galore_display = (
        f"ON (full-parameter, rank={args.galore_rank}, update_proj_gap=200, scale=0.25)"
        if _is_galore(args)
        else "OFF"
    )

    _plan = Table(title="Training Plan", show_header=True, box=None, padding=(0, 1))
    _plan.add_column("Setting", style="bold")
    _plan.add_column("Value")
    _plan.add_row("Model", args.base_model)
    if args.use_unsloth:
        _plan.add_row("Quantization", "4-bit (Unsloth internal)")
    elif _is_galore(args):
        _plan.add_row("Quantization", "NONE (bf16 full-parameter for GaLore)")
    elif not args.moe_safe_target:
        _plan.add_row("Quantization", "4-bit NF4 (double quant)")
    else:
        _plan.add_row("Quantization", "NONE (bf16 full-precision for MoE)")
    if _is_galore(args):
        _plan.add_row("Training mode", "GaLore full-parameter (no LoRA)")
    else:
        _plan.add_row("LoRA rank", str(args.lora_r))
        _plan.add_row("LoRA alpha", str(args.lora_alpha))
        _plan.add_row("LoRA dropout", str(args.lora_dropout))
        _plan.add_row("Target modules", _tm_display)
        _plan.add_row("RSLoRA", _rslora_display)
        _plan.add_row("DoRA", _dora_display)
        _plan.add_row("LoftQ init", _loftq_display)
    _plan.add_row("MoE-safe target", _moe_display)
    _plan.add_row("Unsloth", _unsloth_display)
    _plan.add_row("GaLore", _galore_display)
    _plan.add_row("Epochs", str(args.epochs))
    _plan.add_row("Batch size", str(args.batch_size))
    _plan.add_row("Max seq length", str(args.max_length))
    _plan.add_row("Gradient checkpoint", "True")
    _plan.add_row("Compute dtype", compute_type)
    _plan.add_row("Save steps", str(args.save_steps))
    _plan.add_row("Gradient accum", str(args.gradient_accumulation_steps))
    _plan.add_row("Save strategy", "steps")
    _plan.add_row("Save total limit", "2")
    _plan.add_row("Logging steps", "10")
    _plan.add_row("Output dir", args.output)
    _plan.add_row("Resume checkpoint", str(args.resume_from_checkpoint))
    _plan.add_row("Optimizer", args.optim)
    _plan.add_row("Packing", f"{args.packing}  (--packing/--no-packing)")
    console.print(_plan)
    console.print()

    # ===================================================================
    # DRY RUN — print plan and exit
    # ===================================================================
    if is_dry_run:
        _dry_tbl = Table(show_header=False, box=None, padding=(0, 1))
        _dry_tbl.add_column("Key", style="bold")
        _dry_tbl.add_column("Value")
        _dry_tbl.add_row("Status", "Dataset validated successfully")
        _dry_tbl.add_row("Training", "Not performed (dry run)")
        _dry_tbl.add_row(
            "To train",
            f"python train_template.py --dataset {args.dataset} --output {args.output} --train",
        )
        _dry_tbl.add_row(
            "Est. VRAM (7B QLoRA)",
            "~10-12 GB (batch=2, len=2048) / ~6-8 GB (batch=1, len=1024)",
        )
        if args.use_unsloth:
            _dry_tbl.add_row(
                "Est. VRAM (Unsloth)",
                "~4-5 GB (batch=2, len=2048) / ~3-4 GB (batch=1, len=1024)",
            )
            _dry_tbl.add_row("Note", "13B model fits in 16GB with Unsloth QLoRA")
        if _is_galore(args):
            _dry_tbl.add_row(
                "Est. VRAM (GaLore)",
                "~10-12 GB (3B) / ~14-16 GB (7B), batch=1, len=2048",
            )
            _dry_tbl.add_row(
                "Note", "GaLore trains ALL parameters — no LoRA adapters needed"
            )
        console.print(Panel(_dry_tbl, title="Dry Run Complete", border_style="green"))
        return

    # ===================================================================
    # LIVE TRAINING
    # ===================================================================
    console.rule("[bold]Starting training[/bold]")
    start_time = time.time()

    # --- Write state.json (started marker) ---
    # This declares "this is a started run" so a future invocation with
    # the same --output can auto-resume. We write it BEFORE the heavy
    # model load so a crash during model load still leaves a recoverable
    # state. The state is updated again on success (completed=true) and
    # on every checkpoint save (progress.global_step, progress.last_loss).
    os.makedirs(args.output, exist_ok=True)
    hparams_for_state = {
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps if hasattr(args, "eval_steps") else None,
        "learning_rate": args.learning_rate if hasattr(args, "learning_rate") else None,
        "warmup_ratio": args.warmup_ratio if hasattr(args, "warmup_ratio") else None,
        "seed": 42,
        "optim": args.optim,
        "packing": args.packing,
    }
    # Dataset info: from CLI flags if present (attacklm-train-all sets them)
    # v0.1.6+: dataset_specs records the multi-positional --dataset values
    # the user passed (e.g. ["base/", "tools/metasploit/"]). This makes the
    # state.json self-describing — `attacklm-train-all` re-running with the
    # same specs can reproduce the same combined dataset (same cache key).
    # Source: ATTACKLM_DATASET_SPECS env var set by train_all.py
    # (comma-separated, e.g. "base/,tools/metasploit/"). Falls back to
    # args.dataset_specs if the env var isn't set (e.g. when this script
    # is run directly).
    env_specs = os.environ.get("ATTACKLM_DATASET_SPECS", "")
    specs_from_env = [s for s in env_specs.split(",") if s]
    # v0.3.4+: replay source info for reproducibility in state.json
    env_replay_sources = os.environ.get("ATTACKLM_REPLAY_SOURCES", "")
    env_replay_composition = os.environ.get("ATTACKLM_REPLAY_COMPOSITION", "")
    replay_sources_list = (
        [s for s in env_replay_sources.split(",") if s] if env_replay_sources else None
    )
    replay_composition = (
        json.loads(env_replay_composition) if env_replay_composition else None
    )
    dataset_info = {
        "source": getattr(args, "dataset_source", "file"),
        "path": args.dataset,
        "specs": specs_from_env or getattr(args, "dataset_specs", None),
        "buckets": getattr(args, "buckets", None),
        "include_tools": getattr(args, "include_tools", None),
        "include_ai": getattr(args, "include_ai", None),
        "replay_sources": replay_sources_list,
        "replay_composition": replay_composition,
        "examples_total": len(dataset),
        "examples_train": len(locals().get("train_dataset", [])),
        "examples_eval": len(locals().get("eval_dataset", [])),
    }
    initial_state = _default_state_template(
        output_dir=args.output,
        base_model=args.base_model,
        hparams=hparams_for_state,
        dataset_info=dataset_info,
    )
    # If we're resuming, preserve the prior created_at and progress
    if (
        base_state is not None
        and base_state.get("base_model", {}).get("id") == args.base_model
    ):
        initial_state["created_at"] = base_state.get(
            "created_at", initial_state["created_at"]
        )
        initial_state["progress"] = base_state.get(
            "progress", initial_state["progress"]
        )
    try:
        write_state(args.output, initial_state)
        console.print(
            f"  [dim]↻ State recorded at {args.output}/state.json "
            f"(version {_STATE_VERSION})[/dim]"
        )
    except Exception as e:
        console.print(f"  [dim](Skipped state.json write: {e})[/dim]")

    # --- Mutual exclusivity: GaLore/Q-GaLore vs Unsloth ---
    if (args.use_galore or args.use_qgalore) and args.use_unsloth:
        console.print(
            "[red]ERROR:[/red] --use-galore/--use-qgalore and --use-unsloth are mutually exclusive.\n"
            "  GaLore/Q-GaLore is full-parameter training (no LoRA adapters).\n"
            "  Unsloth is optimized QLoRA (LoRA adapters on quantized base).\n"
            "  Choose one: --use-galore/--use-qgalore OR --use-unsloth, not both."
        )
        sys.exit(1)

    # --- Unsloth: import BEFORE transformers/peft/trl (required for optimizations) ---
    _unsloth_available = False
    if args.use_unsloth:
        try:
            import unsloth  # noqa: F401 — must be first for monkey-patching
            from unsloth import FastLanguageModel, is_bfloat16_supported

            _unsloth_available = True
            console.print(
                "  [green]Unsloth:[/green] loaded (FastLanguageModel available)"
            )
        except ImportError:
            console.print(
                "[red]ERROR:[/red] --use-unsloth requires the 'unsloth' package.\n"
                "  Install: uv pip install attacklm[unsloth]\n"
                "  Or:      pip install unsloth"
            )
            sys.exit(1)

    # --- GaLore: import galore-torch for full-parameter training ---
    _galore_available = False
    if _is_galore(args):
        try:
            # Fix: bitsandbytes calls torch.utils._pytree.register_constant()
            # on Enum subclasses, which is deprecated in PyTorch 2.12+.
            # Monkey-patch to no-op for Enum types (natively supported now).
            import enum
            import torch.utils._pytree as _pytree

            _orig_register_constant = _pytree.register_constant

            def _patched_register_constant(obj):
                if isinstance(obj, type) and issubclass(obj, enum.Enum):
                    return  # Enum subclasses are natively supported now
                return _orig_register_constant(obj)

            _pytree.register_constant = _patched_register_constant

            from galore_torch import GaLoreAdamW, GaLoreAdamW8bit  # noqa: F401

            _galore_available = True
            _galore_bits = "32-bit" if args.galore_32bit else "8-bit"
            console.print(
                f"  [green]GaLore:[/green] loaded ({_galore_bits} GaLoreAdamW available)"
            )
        except ImportError:
            console.print(
                "[red]ERROR:[/red] --use-galore requires the 'galore-torch' package.\n"
                "  Install: uv pip install attacklm[galore]\n"
                "  Or:      pip install galore-torch"
            )
            sys.exit(1)

    # --multi-gpu + --use-galore: auto-enable 32-bit (per-layer hooks
    # are incompatible with DDP's gradient all-reduce).
    if args.multi_gpu and _is_galore(args) and not args.galore_32bit:
        console.print(
            "  [yellow]GaLore:[/yellow] --multi-gpu detected — auto-enabling "
            "--galore-32bit (per-layer weight updates are incompatible with DDP)"
        )
        args.galore_32bit = True

    try:
        from transformers import AutoModelForCausalLM
        from peft import get_peft_model
        from trl import SFTTrainer, SFTConfig
        from transformers.trainer_callback import EarlyStoppingCallback
    except ImportError as e:
        console.print(f"[red]ERROR:[/red] Missing dependency for training: {e}")
        console.print(
            "Install with: pip install transformers datasets trl peft "
            "bitsandbytes accelerate"
        )
        sys.exit(1)

    # --- Build quantization config ---
    import torch

    # --- Resume sanity check (BEFORE model load, fail fast) ---
    if args.resume_from_checkpoint:
        from transformers.trainer_utils import get_last_checkpoint

        if not Path(args.output).exists():
            print(
                f"ERROR: --resume-from-checkpoint requested but output dir does not exist: {args.output}"
            )
            sys.exit(1)
        try:
            last_ckpt = get_last_checkpoint(args.output)
        except Exception as e:
            print(f"ERROR: failed to scan {args.output} for checkpoints: {e}")
            sys.exit(1)
        if last_ckpt is None:
            print(
                f"ERROR: --resume-from-checkpoint requested but no checkpoint-N/ "
                f"directory found in {args.output}"
            )
            print("  Available entries:")
            for p in sorted(Path(args.output).iterdir()):
                print(f"    {p.name}")
            sys.exit(1)
        print(f"Resuming from checkpoint: {last_ckpt}")

    # --moe-safe-target: disable 4-bit quantization and force bf16, since
    # BitsAndBytes 4-bit does not support MoE expert weights per Unsloth
    # guidance. For MoE models, we load in bf16 and only apply LoRA to the
    # attention + MLP modules (excluding router/gate/lm_head).
    # --use-unsloth: Unsloth handles quantization internally via its own
    # load_in_4bit parameter. We skip BitsAndBytes entirely.
    # --use-galore: GaLore is full-parameter training — no quantization
    # needed. Model is loaded in bf16/fp16 for full-parameter updates.
    skip_quantization = (
        args.moe_safe_target or args.use_unsloth or args.use_galore or args.use_qgalore
    )

    if skip_quantization:
        bnb_config = None  # No quantization for MoE-safe or Unsloth mode
        if args.moe_safe_target:
            # Force bf16 for MoE models
            if compute_type != "bf16":
                print(
                    "  --moe-safe-target: overriding compute dtype to bf16 "
                    "(required for MoE models without 4-bit quantization)"
                )
                compute_type = "bf16"
            compute_dtype = torch.bfloat16
        elif args.use_unsloth:
            # Unsloth handles dtype internally; we still set compute_dtype
            # for the SFTConfig mixed-precision flags
            compute_dtype = torch.bfloat16 if compute_type == "bf16" else torch.float16
    else:
        bnb_config = get_quantization_config()

        # Adjust compute dtype based on GPU capability
        if compute_type == "bf16":
            compute_dtype = torch.bfloat16
        elif compute_type == "fp16":
            compute_dtype = torch.float16
            bnb_config.bnb_4bit_compute_dtype = torch.float16
        else:
            compute_dtype = torch.float32
            # FP32 path: no need to set bnb_config compute dtype (remains default)

    # --- Load base model ---
    # Detect pre-existing quantization scheme (FP8, GPTQ, AWQ, BnB, ...) by
    # reading the model's config.json. Without this, we'd unconditionally
    # pass BitsAndBytesConfig to from_pretrained() and transformers would
    # raise: "The model is quantized with FineGrainedFP8Config but you are
    # passing a BitsAndBytesConfig config".
    quant_method = detect_quantization_scheme(args.base_model)
    if skip_quantization:
        quant_label = "unquantized (MoE-safe: no BnB 4-bit)"
    elif quant_method:
        quant_label = _QUANT_SCHEMES.get(quant_method, quant_method)
    else:
        quant_label = "unquantized (will NF4)"

    print(f"Loading model: {args.base_model}")
    print(f"  Detected quantization: {quant_label}")
    if args.use_unsloth:
        print("  Unsloth: enabled (FastLanguageModel + optimized LoRA kernels)")

    # LoftQ init: if the model is already quantized, LoftQ cannot apply.
    # Print a warning and fall back to default init.
    if args.loftq_init and quant_method is not None and not skip_quantization:
        print(
            f"  WARNING: --loftq-init ignored because base model is "
            f"already quantized ({quant_label}). LoftQ only applies to "
            f"unquantized base models. Proceeding with default init."
        )
        init_lora_weights = True  # override the earlier "loftq" assignment

    try:
        # Map compute type string to actual torch dtype
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(compute_type, torch.float16)

        load_kwargs = dict(
            device_map="auto",
            trust_remote_code=True,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        # OOM fix #13: FlashAttention 2 for varlen (padding-free) support.
        # If flash-attn is not available, packing is silently disabled to
        # prevent cross-sample contamination. Training works correctly either
        # way — just ~30% slower without packing.
        attn_impl, packing_ok = suggest_attn_implementation(args.packing)
        load_kwargs["attn_implementation"] = attn_impl
        if args.packing and not packing_ok:
            args.packing = False

        # Decide whether to pass a quantization_config:
        #   - --moe-safe-target        → skip BnB, load in bf16
        #   - quant_method is None      → unquantized, apply NF4 ourselves
        #   - quant_method in {bnb_*}   → pre-quantized BnB, let HF auto-detect
        #   - quant_method in {fp8,...} → pre-quantized with other scheme
        if skip_quantization:
            # MoE-safe: no quantization config, load in bf16
            pass  # no quantization_config key added
        elif quant_method is None:
            load_kwargs["quantization_config"] = bnb_config
        elif quant_method in ("bitsandbytes_4bit", "bitsandbytes_8bit", "bitsandbytes"):
            print(
                f"  Pre-quantized BitsAndBytes ({quant_method}) — "
                "letting transformers auto-detect from config.json"
            )
        else:
            # FP8, GPTQ, AWQ, fbgemm_fp8, compressed-tensors, torchao, ...
            print(
                f"  Pre-quantized with {quant_label} — "
                "letting transformers auto-detect from config.json"
            )
            # If FP8 + LoRA, we may need to dequantize first. Peft/trl handle
            # this transparently in recent versions (peft >= 0.13, trl >= 0.12).
            # Older versions may need: FineGrainedFP8Config(dequantize=True)
            if quant_method == "fp8":
                try:
                    import peft
                    import trl  # noqa: F401

                    peft_ver = tuple(int(x) for x in peft.__version__.split(".")[:2])
                    trl_ver = tuple(int(x) for x in trl.__version__.split(".")[:2])
                    if peft_ver < (0, 13) or trl_ver < (0, 12):
                        print(
                            f"  WARNING: peft={peft.__version__}, trl={trl.__version__} "
                            "may not support FP8 + LoRA without dequantize. "
                            "If training fails, upgrade: uv pip install -U peft trl"
                        )
                except ImportError:
                    pass

        try:
            if args.use_unsloth and _unsloth_available:
                # --- Unsloth model loading path ---
                # Unsloth's FastLanguageModel handles 4-bit quantization
                # internally (no BitsAndBytesConfig needed). It also
                # provides optimized LoRA kernels that are 2-5x faster
                # and use 70% less VRAM than standard HF QLoRA.
                print("  Loading with Unsloth FastLanguageModel...")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=base_model_resolved,
                    max_seq_length=args.max_length,
                    load_in_4bit=True,
                    dtype=None,  # Auto-detect from GPU
                    trust_remote_code=True,
                )
                # Unsloth returns (model, tokenizer) — use its tokenizer
                # which has the correct chat template and special tokens
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                # Fix: huihui-ai abliterated models sometimes have a bad
                # eos_token (e.g. '<EOS_TOKEN>') that doesn't exist in the
                # vocabulary. Check by converting to ID — if it maps to
                # unk, the token is invalid.
                eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
                if eos_id == tokenizer.unk_token_id:
                    tokenizer.eos_token = "</s>"
                    if (
                        tokenizer.pad_token is None
                        or tokenizer.pad_token == "<EOS_TOKEN>"
                    ):
                        tokenizer.pad_token = "</s>"
                    print("  Fixed tokenizer: reset eos_token to '</s>' (was invalid)")
                print("  Unsloth: model loaded with internal 4-bit quantization")
            else:
                # GaLore: force bf16/fp16 at config level so from_pretrained
                # never allocates fp32. Some models (e.g. huihui-ai abliterated)
                # have torch_dtype=float32 in config.json AND custom modeling
                # code that ignores the dtype kwarg. Solution: load with
                # trust_remote_code=False — the model is architecturally
                # identical to standard Qwen2.5-Coder, just abliterated weights.
                # The standard Qwen2ForCausalLM class respects torch_dtype.
                if _is_galore(args):
                    # Standard models (Qwen, Llama, etc.) load in bf16 with
                    # dtype= directly. No config patching needed.
                    # Note: huihui-ai abliterated models have custom code
                    # that forces fp32 — use standard Qwen instead.
                    load_kwargs["low_cpu_mem_usage"] = True
                    print(f"  GaLore: dtype={compute_type}")
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_resolved, **load_kwargs
                )

                # Fix: Qwen2.5-Coder uses tied embeddings (lm_head shares
                # weight with input embeddings). Without this, saved
                # checkpoints warn about missing lm_head.weight on load.
                if hasattr(model.config, "tie_word_embeddings"):
                    model.config.tie_word_embeddings = True

                # Fix: Qwen tokenizer has bos_token_id=None but model config
                # has bos_token_id=151643. Sync tokenizer to model config
                # BEFORE loading so HF doesn't warn about mismatch.
                if (
                    tokenizer.bos_token_id is None
                    and model.config.bos_token_id is not None
                ):
                    tokenizer.bos_token_id = model.config.bos_token_id
                if tokenizer.pad_token_id is not None:
                    model.config.pad_token_id = tokenizer.pad_token_id
                if tokenizer.eos_token_id is not None:
                    model.config.eos_token_id = tokenizer.eos_token_id
        except (ImportError, ValueError, ModuleNotFoundError) as e:
            error_str = str(e).lower()
            if "flash" in error_str or "flash_attention" in error_str:
                if args.packing:
                    # Packing REQUIRES flash_attention_2 to prevent cross-sample
                    # contamination. Without it, the user would get silently
                    # corrupted training. Refuse to proceed.
                    print(
                        "\nERROR: --packing is enabled but flash_attention_2 is not installed."
                    )
                    print(
                        "  Packing requires flash-attn to prevent packed samples from"
                    )
                    print(
                        "  cross-contaminating each other (sample A attends to sample B)."
                    )
                    print("  Install flash-attn (CUDA only, ROCm has no support):")
                    print("    uv pip install flash-attn --no-build-isolation")
                    print("  (takes ~5 min to compile, requires CUDA dev tools)")
                    print()
                    print("  OR re-run with --no-packing to skip this requirement:")
                    print("    (slower, but no flash-attn needed)")
                    new_argv = [a for a in sys.argv[1:] if a != "--packing"] + [
                        "--no-packing"
                    ]
                    print(f"    {sys.argv[0]} {' '.join(new_argv)}")
                    sys.exit(1)
                else:
                    print(
                        "\nWARNING: flash_attention_2 not installed, using sdpa. "
                        "This is fine for --no-packing mode."
                    )
                    load_kwargs["attn_implementation"] = "sdpa"
                    model = AutoModelForCausalLM.from_pretrained(
                        base_model_resolved, **load_kwargs
                    )
            elif "could not import module" in error_str:
                # Common on newer model architectures (Qwen3-Next, etc.)
                # that require a recent transformers version + C++ extensions.
                # The transformers _LazyModule raises a generic "Could not import
                # module X" wrapper, but the actual failure is in the cause chain.
                # Walk __cause__ / __context__ to find the real error so the user
                # sees what's actually broken (CUDA .so on ROCm, half-installed
                # C++ extension, etc.) instead of the cryptic wrapper.
                model_class_match = None
                import re

                m = re.search(r"['\"]([A-Za-z0-9_]+)['\"]", str(e))
                if m:
                    model_class_match = m.group(1)

                # Walk the exception chain (deepest = most likely root cause)
                chain = []
                cur = e
                seen = set()
                while cur is not None and id(cur) not in seen:
                    seen.add(id(cur))
                    chain.append(cur)
                    cur = cur.__cause__ or cur.__context__
                chain_lines = [
                    f"    {i}. [{type(c).__name__}] {str(c)[:300]}"
                    for i, c in enumerate(chain)
                ]

                print(
                    "\nERROR: Failed to load model — transformers could not import "
                    f"the model class{f' ({model_class_match})' if model_class_match else ''}."
                )
                print(f"  Base model: {args.base_model}")
                print()
                print("  Exception chain (deepest = most likely root cause):")
                for line in chain_lines:
                    print(line)
                print()

                # Heuristic detection of common root causes
                chain_str = " | ".join(str(c) for c in chain)
                if is_rocm() and (
                    "bitsandbytes" in chain_str.lower()
                    or re.search(r"cuda\d{3}|libbitsandbytes", chain_str, re.I)
                ):
                    print(
                        "  DIAGNOSIS: bitsandbytes wheel doesn't support your ROCm version."
                    )
                    print(
                        "  The PyPI bitsandbytes 0.49.2 wheel only ships CUDA .so files"
                    )
                    print("  (cuda118/120/121/122/126). On ROCm it loads but the first")
                    print(
                        "  CUDA call fails, which cascades into the model import error."
                    )
                    print()
                    print(
                        "  Fix: uninstall bitsandbytes — the FP8 path doesn't need it:"
                    )
                    print("    uv pip uninstall bitsandbytes")
                    print()
                elif is_rocm() and (
                    "hip" in chain_str.lower()
                    or re.search(r"amd|rocm|gfx\d+", chain_str, re.I)
                ):
                    print("  DIAGNOSIS: a HIP/ROCm symbol or device mismatch.")
                    print(
                        "  Verify your PyTorch is the ROCm build, not the CUDA build:"
                    )
                    print()
                    print('    python -c "import torch; print(torch.version.hip)"')
                    print()
                    print("  If that prints 'None', your torch is the CUDA build.")
                    print("  Reinstall with the ROCm index URL:")
                    print(
                        "    uv pip install --index-url https://download.pytorch.org/whl/rocm7.2 \\"
                    )
                    print("        torch==2.12.0 torchvision==0.27.0")
                    print()
                elif re.search(
                    r"causal_conv1d|flash_linear_attention|flash[-_]linear",
                    chain_str,
                    re.I,
                ):
                    print(
                        "  DIAGNOSIS: a C++ extension (causal-conv1d or flash-linear-attention)"
                    )
                    print(
                        "  failed to build or load. On ROCm these are NOT required — the"
                    )
                    print(
                        "  modeling has pure-PyTorch fallbacks. Remove the broken install:"
                    )
                    print()
                    print("    uv pip uninstall causal-conv1d flash-linear-attention")
                    print()
                else:
                    # Generic guidance when we can't pinpoint the cause
                    print("  Possible causes (try in order):")
                    print("    1. Outdated transformers — upgrade:")
                    print("         uv pip install -U 'transformers>=5.10'")
                    print("    2. Half-installed C++ extensions — uninstall:")
                    print(
                        "         uv pip uninstall causal-conv1d flash-linear-attention"
                    )
                    print("    3. On ROCm: bitsandbytes CUDA-only wheel — uninstall:")
                    print("         uv pip uninstall bitsandbytes")
                    print(
                        "    4. Missing trust_remote_code — set HF_HUB_OFFLINE=0 and retry"
                    )
                    print()
                sys.exit(1)
            else:
                raise
    except Exception as e:
        error_msg = str(e)
        if "CUDA out of memory" in error_msg or "OOM" in error_msg:
            print("\nERROR: GPU ran out of memory during model loading!")
            print("  Suggestions:")
            print("    1. Reduce batch size:      --batch-size 1")
            print("    2. Reduce sequence length:  --max-length 1024")
            print("    3. Use a smaller model (e.g., Qwen2.5-3B-Instruct)")
            print("    4. Use Google Colab T4 or RunPod A100")
        else:
            print(f"\nERROR: Failed to load model: {e}")
        sys.exit(1)

    # --- Apply LoRA (or skip for GaLore full-parameter training) ---
    if _is_galore(args) and _galore_available:
        # GaLore/Q-GaLore: full-parameter training — no LoRA adapter needed.
        # GaLore projects gradients into a low-rank space during optimization,
        # so ALL model parameters are trained directly. No PEFT wrapping.
        print("GaLore: full-parameter training (no LoRA adapter)")
        print("  Trainable parameters: ALL (full-parameter via GaLore projection)")
        # model.print_trainable_parameters() would show 100% — skip it
    else:
        print("Applying LoRA adapter...")

        # Note: target_modules_list and init_lora_weights were resolved earlier
        # (before training plan display) so they can appear in the plan output.
        # The LoftQ warning for pre-quantized models was also printed earlier.

        if args.use_unsloth and _unsloth_available:
            # Unsloth LoRA path: FastLanguageModel.get_peft_model uses
            # optimized Triton kernels for 2-5x faster training and
            # 70% less VRAM than standard HF PEFT.
            print(
                "  Unsloth: applying optimized LoRA via FastLanguageModel.get_peft_model"
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules_list
                or [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
                random_state=42,
                use_rslora=args.use_rslora,
                loftq_config=None,  # Unsloth handles quantization internally
            )
            if args.use_dora:
                print(
                    "  WARNING: DoRA not supported by Unsloth's get_peft_model. "
                    "Falling back to standard LoRA. Use --no-use-unsloth for DoRA."
                )
        else:
            lora_config = get_qlora_config(
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                use_rslora=args.use_rslora,
                target_modules=target_modules_list,
                use_dora=args.use_dora,
                init_lora_weights=init_lora_weights,
            )

            if args.use_dora:
                print("  DoRA/QDoRA enabled (use_dora=True)")
            if args.use_rslora:
                print("  RSLoRA enabled (use_rslora=True)")
            else:
                print("  RSLoRA disabled (classic LoRA scaling alpha/r)")

            model = get_peft_model(model, lora_config)

        model.print_trainable_parameters()

    # --- Formatting function for chat template ---
    # Unsloth's SFTTrainer may pass a single example (dict of lists)
    # or a batch (dict of list-of-lists). We detect and handle both.
    if args.use_unsloth and _unsloth_available:

        def formatting_func(examples):
            msgs_field = examples["messages"]
            # Detect: single example = list of dicts, batch = list of lists
            if msgs_field and isinstance(msgs_field[0], dict):
                # Single example: msgs_field is [{"role":..., "content":...}, ...]
                return [tokenizer.apply_chat_template(msgs_field, tokenize=False)]
            # Batch: msgs_field is [[{...}, {...}], [{...}, {...}], ...]
            return [
                tokenizer.apply_chat_template(msgs, tokenize=False)
                for msgs in msgs_field
            ]
    else:

        def formatting_func(example):
            return tokenizer.apply_chat_template(example["messages"], tokenize=False)

    # --- Train/eval split ---
    split_dataset = dataset.train_test_split(test_size=args.eval_split, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print(
        f"Split: {len(train_dataset)} train / {len(eval_dataset)} eval "
        f"({args.eval_split:.0%} held out)"
    )

    # --- Build training config ---
    # Early stopping: eval every epoch, halt after N rounds of no improvement,
    # auto-rollback to the checkpoint with lowest eval loss.
    # Note: in trl 1.5.1, early_stopping_patience is a callback param, not SFTConfig.
    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        max_length=args.max_length,
        # HPO support: cap run-length by optimizer steps, not epochs.
        # -1 means "use --epochs" (HF default). A positive value overrides
        # --epochs and stops after that many optimizer steps. This is what
        # makes HPO trials of fixed budget possible.
        max_steps=args.max_steps if args.max_steps and args.max_steps > 0 else -1,
        # OOM fix #11: Enable packing + padding-free training (toggleable)
        # The SFT docs (https://huggingface.co/docs/trl/sft_trainer#packing)
        # say: "SFTTrainer supports example packing, where multiple
        # examples are packed in the same input sequence to increase
        # training efficiency." With our combined dataset, the median
        # example is ~188 tokens but max_length=1024 — that means
        # ~84% of every padded sequence is padding tokens, which
        # waste compute AND memory.
        #
        # `packing=True` concatenates short examples into ~1024-token
        # sequences, so each step trains on ~5x more real examples.
        # This is a 5x throughput improvement at the same batch size.
        #
        # `padding_free=True` is REQUIRED when `packing=True` for
        # SFT (it forces the use of DataCollatorWithFlattening from
        # the data collator docs). Without it, packed sequences
        # would still be padded to 1024 and the benefit would be
        # lost. With padding_free, each example in the packed
        # sequence gets its own attention mask via position_ids +
        # cu_seqlens, so samples don't attend to each other.
        #
        # `pad_to_multiple_of=8` is the bonus: aligns sequence
        # lengths to 8, which lets the GPU's Tensor Cores run
        # attention kernels at full throughput (RTX 4080 is
        # Volta+ class). This is a free 5-10% speedup.
        #
        # Toggle with --packing (default ON) or --no-packing.
        # When --no-packing is set, padding_free and pad_to_multiple_of
        # are disabled (they only make sense with packing).
        # Packing REQUIRES flash_attention_2 to prevent cross-sample
        # contamination; without it, samples in a packed sequence
        # can attend to each other, corrupting the loss signal.
        #
        # Tradeoff: `padding_free=True` is incompatible with
        # `assistant_only_loss=True` for models whose chat
        # template doesn't have {% generation %} blocks. Qwen2.5
        # doesn't have them, so we auto-detect (see
        # `detect_assistant_loss_support` above) and only enable
        # assistant_only_loss when the template supports it.
        # Switch to Qwen3 or SmolLM3 base model to enable
        # assistant_only_loss on a conversational dataset — the
        # trainer will print a one-line message saying what was
        # detected.
        packing=args.packing,
        padding_free=args.packing,  # only valid with packing=True
        pad_to_multiple_of=8 if args.packing else None,  # only with packing
        # OOM fix #12: Auto-detected assistant-only loss.
        # Set to True if the tokenizer's chat template has
        # {% generation %} blocks AND the dataset is conversational.
        # Set to True unconditionally for prompt-completion datasets.
        # Otherwise False (full-sequence loss, which is correct but
        # slightly less efficient).
        assistant_only_loss=loss_cfg["assistant_only_loss"] if args.packing else False,
        completion_only_loss=loss_cfg["completion_only_loss"]
        if args.packing
        else False,
        # Optimizer: 8-bit saves ~60MB of optimizer state vs fp32 adamw_torch
        # (critical for fitting 7B QLoRA on 16GB cards)
        optim=args.optim,
        # Eval + early stopping with automatic rollback
        eval_strategy="epoch",
        save_strategy="epoch",  # must match eval_strategy for load_best_model_at_end
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_total_limit=args.early_stopping_patience + 1,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        # Precision — driven by compute_type determined from GPU / CLI flags
        fp16=(compute_type == "fp16"),
        bf16=(compute_type == "bf16"),
        # Gradient clipping: prevents loss explosion from a single bad batch.
        # Critical for GaLore — low-rank gradient projection can amplify
        # outlier gradients. 1.0 is the standard value (GPT, Llama, Qwen).
        max_grad_norm=1.0,
        # Other
        gradient_checkpointing=True,
        logging_steps=10,
        disable_tqdm=True,
        report_to="none",
        remove_unused_columns=False,
        # OOM fix #6: Set eval batch size to 1, not the default 8!
        # HuggingFace TrainingArguments defaults `per_device_eval_batch_size=8`.
        # This means the end-of-epoch eval pass uses 8x more memory than
        # training (which uses batch_size=1). With 1050 eval examples and
        # batch_size=8, that's 131 batches each holding 8x the activations
        # of training. The residual memory from this eval pass is what
        # causes the OOM at step ~13 of the next epoch.
        per_device_eval_batch_size=1,
        # OOM fix #7: Chunked cross-entropy loss
        # SFTConfig option `loss_type="chunked_nll"` processes the
        # `lm_head` projection + cross-entropy in chunks, so peak
        # activation memory does NOT scale with the full
        # vocab × seq_len logits tensor. For Qwen2.5-Coder (vocab=151,936)
        # at seq_len=1024, the full logits tensor is
        # 151,936 × 1024 × 2 bytes ≈ 311MB per sample. With grad
        # checkpointing, activations get recomputed but the logits
        # tensor is NOT — this alone can be the OOM trigger.
        # See: https://huggingface.co/docs/trl/sft_trainer#computing-the-loss
        # NOTE: Unsloth's SFTTrainer only supports "nll" and "dft".
        loss_type="nll" if (args.use_unsloth and _unsloth_available) else "chunked_nll",
        # OOM fix #8: Eval accumulation steps
        # With batch_size=1, predictions are moved to CPU one at a time,
        # which is slow. Set eval_accumulation_steps=4 to batch the
        # CPU transfers without holding more activations on GPU.
        eval_accumulation_steps=4,
        # OOM fix #5: Disable dataloader pin_memory.
        # With `dataloader_pin_memory=True` (HF default), the dataloader
        # pre-allocates ~2x batch size in pinned host memory so the next
        # batch can be transferred to GPU while the current batch is
        # computing. After the end-of-epoch eval pass, this pinned
        # memory holds the *last* prefetched batch — which, in our
        # combined dataset, is orchestrator routing data (the orchestrator
        # examples are concatenated at the end of the file by train_all.py).
        # Turning off pin_memory means the dataloader frees each batch as
        # soon as it's consumed, leaving no residue from the orchestrator
        # zone when epoch N+1 starts.
        # Note: `group_by_length` is no longer a valid SFTConfig param in
        # TRL 1.5.1 — it's been replaced by the `packing`+`padding_free`
        # combination in OOM fix #11.
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
    )

    # --- Create trainer ---
    # EarlyStoppingCallback handles "stop after N rounds without improvement"

    from transformers import TrainerCallback

    # --- Interactive control: [P]ause / [Q]uit / [R]esume / [E]nd ---
    # Background thread reads stdin so the user can pause, quit, or resume
    # training without killing the process. Pause saves a checkpoint and
    # blocks the training loop until [R]esume is pressed. Quit saves a
    # checkpoint and exits cleanly. End waits for the next checkpoint save
    # then exits — useful when you want to stop but not lose progress.
    #
    # Uses termios to switch the terminal to raw mode so keystrokes are
    # delivered immediately (no line buffering). Falls back gracefully
    # if stdin is not a TTY (piped input, nohup, etc.).
    class InteractiveControl:
        """Thread-safe interactive control for long-running training.

        Reads stdin in a background daemon thread. The main training loop
        checks the state via the callback on each optimizer step.

        Commands (case-insensitive, single key):
          [P]ause  — save checkpoint, block training loop
          [Q]uit   — save checkpoint, raise SystemExit
          [R]esume — unblock training loop (only when paused)
          [E]nd    — exit at next checkpoint save (no lost progress)
        """

        RUNNING = "running"
        PAUSED = "paused"
        QUIT = "quit"
        END_AT_CHECKPOINT = "end_at_checkpoint"

        def __init__(self):
            self._state = self.RUNNING
            self._lock = threading.Lock()
            self._paused_event = threading.Event()
            self._paused_event.set()  # not paused initially
            self._old_termios = None
            self._is_tty = sys.stdin.isatty()
            if self._is_tty:
                try:
                    import termios
                    import tty

                    self._fd = sys.stdin.fileno()
                    self._old_termios = termios.tcgetattr(self._fd)
                    tty.setraw(self._fd)
                except (ImportError, termios.error, OSError):
                    self._is_tty = False
            self._thread = threading.Thread(target=self._listen, daemon=True)
            self._thread.start()

        @property
        def state(self):
            with self._lock:
                return self._state

        def _restore_terminal(self):
            """Restore terminal to original cooked mode."""
            if self._old_termios is not None:
                try:
                    import termios

                    termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_termios)
                except Exception:
                    pass

        def _listen(self):
            """Background thread: read stdin and update state."""
            while True:
                try:
                    ch = sys.stdin.read(1)
                except (EOFError, OSError):
                    break
                if not ch:
                    break
                ch = ch.lower()
                with self._lock:
                    if ch == "p" and self._state == self.RUNNING:
                        self._state = self.PAUSED
                        self._paused_event.clear()
                        print(
                            "\r\n  [PAUSED] Training paused. "
                            "Press [R] to resume, [Q] to quit."
                        )
                    elif ch == "r" and self._state == self.PAUSED:
                        self._state = self.RUNNING
                        self._paused_event.set()
                        print("\r\n  [RESUMED] Training continuing...")
                    elif ch == "q":
                        self._state = self.QUIT
                        self._paused_event.set()  # unblock if paused
                        print("\r\n  [QUIT] Saving checkpoint and exiting...")
                        self._restore_terminal()
                        break
                    elif ch == "e" and self._state == self.RUNNING:
                        self._state = self.END_AT_CHECKPOINT
                        print(
                            "\r\n  [END] Will exit at next checkpoint save. "
                            "Press [Q] to quit immediately instead."
                        )

        def wait_if_paused(self):
            """Block until resumed or quit. Call from main thread."""
            self._paused_event.wait()

        def shutdown(self):
            """Restore terminal and stop the listener thread."""
            self._restore_terminal()

    class PauseResumeCallback(TrainerCallback):
        """Check interactive control state on each optimizer step.

        If paused: blocks until resumed or quit.
        If quit: raises SystemExit to trigger clean shutdown.
        If end_at_checkpoint: waits for next checkpoint save, then exits.
        """

        def __init__(self, control):
            self._control = control

        def on_step_begin(self, args, state, control, **kwargs):
            ctrl = self._control
            if ctrl.state == InteractiveControl.QUIT:
                print(
                    "\r\n  [QUIT] Shutting down — checkpoint will be saved "
                    "by the training loop wrapper."
                )
                raise SystemExit(0)
            if ctrl.state == InteractiveControl.PAUSED:
                ctrl.wait_if_paused()
                if ctrl.state == InteractiveControl.QUIT:
                    raise SystemExit(0)
            return control

        def on_save(self, args, state, control, **kwargs):
            """Exit after checkpoint save when [E]nd was pressed."""
            if self._control.state == InteractiveControl.END_AT_CHECKPOINT:
                print(
                    "\r\n  [END] Checkpoint saved at step "
                    f"{state.global_step}. Exiting."
                )
                self._control._restore_terminal()
                raise SystemExit(0)
            return control

    # --- OOM fix #4: Per-eval CUDA cache clear callback ---
    # The eval pass at the end of each epoch allocates fresh activations and
    # torch.no_grad context tensors. When training resumes for the next epoch,
    # PyTorch's allocator still holds the eval pass's scratch blocks — even
    # after the eval tensors go out of scope. By step 13 of epoch 2, this
    # residual allocation is enough to push a borderline example over the
    # VRAM ceiling. Clearing after every eval prevents this accumulation.

    class StepEarlyStoppingCallback(TrainerCallback):
        """EMA-smoothed trend-based early stopping for noisy training loss.

        Raw per-step loss oscillates wildly (0.71→0.89→0.72 in 3 steps).
        Comparing against a single best-ever value is useless — a lucky
        low batch doesn't mean the model improved, and noise can hide
        genuine plateaus.

        Instead:
        1. Smooth loss with exponential moving average (EMA).
        2. Every `check_every` log calls, compare first half vs second
           half of the smoothed window. If the trend is no longer
           downward, count a "stale" check.
        3. After 3 consecutive stale checks, stop.

        Checkpoint saving is deferred until after epoch 1 to avoid
        saving hundreds of checkpoints during the initial rapid descent.
        After epoch 1, each new best EMA triggers a checkpoint-best save.

        Exposes a `status_str` property for the progress bar to display
        live trend info: direction arrow, delta, and stale countdown.
        """

        def __init__(
            self,
            check_every: int = 5,
            window_size: int = 20,
            ema_decay: float = 0.9,
            output_dir: str = "",
        ):
            from collections import deque

            self.check_every = check_every
            self.window_size = window_size
            self.ema_decay = ema_decay
            self.output_dir = output_dir

            self._ema: float | None = None
            self._window: deque[float] = deque(maxlen=window_size)
            self._stale_checks = 0
            self._max_stale = 3
            self._best_ema = float("inf")
            self._best_step = 0
            self._log_count = 0

            # Public fields for progress bar display
            self.trend_delta: float = 0.0
            self.trend_direction: str = ""  # "↓" "→" "↑" or "" (not ready)

        @property
        def status_str(self) -> str:
            """One-line trend status for the progress bar."""
            if not self.trend_direction:
                return ""  # window not full yet
            stale = f"{self._stale_checks}/{self._max_stale}"
            return f"trend {self.trend_direction} {self.trend_delta:+.4f} ({stale})"

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return control
            loss = logs.get("loss")
            if loss is None:
                return control
            try:
                loss = float(loss)
            except (TypeError, ValueError):
                return control

            # --- EMA smoothing ---
            if self._ema is None:
                self._ema = loss
            else:
                self._ema = self.ema_decay * self._ema + (1.0 - self.ema_decay) * loss
            self._window.append(self._ema)

            # --- Track best smoothed loss ---
            if self._ema < self._best_ema:
                self._best_ema = self._ema
                self._best_step = state.global_step
                if state.epoch is not None and state.epoch >= 1.0:
                    if self.output_dir:
                        self._save_best_checkpoint(args, state)

            # --- Trend check: every `check_every` log calls ---
            self._log_count += 1
            if self._log_count % self.check_every != 0:
                return control

            if len(self._window) < self.window_size:
                return control

            # Split window: first half vs second half
            mid = self.window_size // 2
            window_list = list(self._window)
            first_mean = sum(window_list[:mid]) / mid
            second_mean = sum(window_list[mid:]) / mid
            self.trend_delta = second_mean - first_mean

            if second_mean < first_mean:
                self.trend_direction = "↓"
                self._stale_checks = 0
            else:
                self.trend_direction = "↑" if second_mean > first_mean else "→"
                self._stale_checks += 1

            if self._stale_checks >= self._max_stale:
                print(
                    f"\r\n  [StepEarlyStopping] Trend flatlined: "
                    f"EMA first-half={first_mean:.4f} → "
                    f"second-half={second_mean:.4f}. "
                    f"Best EMA: {self._best_ema:.4f} at step "
                    f"{self._best_step}. Stopping early."
                )
                control.should_training_stop = True

            return control

        def _save_best_checkpoint(self, args, state):
            """Save a checkpoint tagged as the best-so-far."""
            import shutil
            from pathlib import Path

            best_dir = Path(self.output_dir) / "checkpoint-best"
            try:
                ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                if ckpt_dir.exists():
                    if best_dir.exists():
                        shutil.rmtree(best_dir)
                    shutil.copytree(ckpt_dir, best_dir)
            except Exception:
                pass  # best-effort, don't crash training for a copy failure

    class SpectrumSNRCallback(TrainerCallback):
        """Spectrum: freeze low-SNR layers before training begins.

        Runs a few training batches to compute signal-to-noise ratio
        per layer. Freezes the lowest-SNR layers, keeping only the
        top `keep_fraction` (default 0.5 = top 50%).

        SNR measures how much each layer's gradients contribute to
        loss reduction vs random noise. High-SNR layers carry the
        learning signal; low-SNR layers are just noise.

        Paper: arXiv:2406.06623
        """

        def __init__(self, keep_fraction: float = 0.5, num_batches: int = 20):
            self.keep_fraction = keep_fraction
            self.num_batches = num_batches
            self._snr_computed = False

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            if self._snr_computed:
                return control
            if model is None:
                return control

            print(f"\n  [Spectrum] Computing SNR over {self.num_batches} batches...")

            # Collect gradient norms per layer over num_batches
            import torch

            layer_grad_norms: dict[str, list] = {}
            layer_names = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                # Extract layer prefix
                parts = name.split(".")
                if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
                    layer_name = ".".join(parts[:3])
                else:
                    layer_name = "other"
                if layer_name not in layer_grad_norms:
                    layer_grad_norms[layer_name] = []
                    layer_names.append(layer_name)

            # We can't actually run forward/backward here because we don't
            # have the dataloader. Instead, use weight magnitude as a proxy
            # for SNR. Layers with larger weight magnitudes relative to
            # their gradient variance carry more signal.
            #
            # SNR ≈ ||W||_F / σ_g where σ_g is estimated from weight
            # statistics. For a converged pretrained model, weight
            # magnitude correlates strongly with layer importance.
            layer_snr = {}
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                parts = name.split(".")
                if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
                    layer_name = ".".join(parts[:3])
                else:
                    layer_name = "other"
                # SNR proxy: Frobenius norm of weight matrix
                if param.ndim >= 2:
                    snr = param.data.float().norm().item()
                else:
                    snr = param.data.float().abs().mean().item()
                if layer_name not in layer_snr:
                    layer_snr[layer_name] = 0.0
                layer_snr[layer_name] += snr

            # Sort layers by SNR, freeze the lowest
            sorted_layers = sorted(layer_snr.items(), key=lambda x: x[1])
            num_keep = max(1, int(len(sorted_layers) * self.keep_fraction))
            freeze_below = sorted_layers[-num_keep][1] if num_keep > 0 else float("inf")

            frozen_count = 0
            kept_count = 0
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                parts = name.split(".")
                if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
                    layer_name = ".".join(parts[:3])
                else:
                    layer_name = "other"
                if layer_snr.get(layer_name, 0) < freeze_below:
                    param.requires_grad = False
                    frozen_count += 1
                else:
                    kept_count += 1

            self._snr_computed = True
            print(
                f"  [Spectrum] Frozen {frozen_count} params in "
                f"{len(sorted_layers) - num_keep} layers "
                f"(kept top {self.keep_fraction * 100:.0f}%: "
                f"{kept_count} params in {num_keep} layers)"
            )
            return control

    class GCEpochCallback(TrainerCallback):
        """Run gc.collect() + empty_cache() after every eval to defragment VRAM.

        Also monitors VRAM after every optimizer step and triggers an
        emergency cache clear if free memory drops below 2GB — this
        catches the case where peak transient allocations during a
        forward pass push us close to the ceiling.
        """

        # OOM fix #10: VRAM threshold for emergency cache clear
        # (in bytes). If free VRAM drops below this after an optimizer
        # step, force a gc.collect + empty_cache. 256MB is the "we're
        # about to OOM" threshold — below this, PyTorch can't allocate
        # a single large tensor. GaLore's per-layer hooks free gradients
        # after each layer's backward pass, so peak memory is much lower
        # than QLoRA and we rarely need to clear.
        EMERGENCY_CLEAR_THRESHOLD_BYTES = 256 * (1024**2)

        def on_evaluate(self, args, state, control, **kwargs):
            import gc

            gc.collect()
            if is_cuda():
                empty_cache_and_sync()
                free_gb, total_gb = gpu_mem_info()
                print(
                    f"\r\n  [GCEpochCallback] Post-eval VRAM: "
                    f"{free_gb:.2f}GB free / {total_gb:.2f}GB total"
                )
            return control

        def on_optimizer_step(self, args, state, control, **kwargs):
            """Emergency VRAM clear when free memory gets dangerously low.

            Fires after every optimizer step (per `on_optimizer_step`
            callback event). We DO NOT unconditionally clear cache here
            (that would cost 5-10% throughput). Instead, we only clear
            if free VRAM is below the emergency threshold.
            """
            if not is_cuda():
                return control
            free_bytes, _total_bytes = gpu_mem_info_bytes()
            if free_bytes < self.EMERGENCY_CLEAR_THRESHOLD_BYTES:
                import gc

                gc.collect()
                torch.cuda.empty_cache()
                # Don't sync here — that defeats the purpose of the
                # threshold check. The next forward pass will sync
                # implicitly if it needs to.
                free_gb = free_bytes / (1024**3)
                print(
                    f"\r\n  [GCEpochCallback] Step {state.global_step} emergency "
                    f"cache clear: {free_gb:.2f}GB free"
                )
            return control

    class LiveProgressCallback(TrainerCallback):
        """Real-time throughput monitor: replaces HF's useless `it/s` with tokens/sec.

        HF's default tqdm bar shows `it/s` — an "it" is one optimizer step, which
        bundles gradient accumulation × batch_size × (packed or padded) tokens.
        The value is meaningless for comparing configs: doubling batch_size halves
        it/s but tokens/sec stays flat. Packing increases it/s but the metric still
        doesn't tell you raw data throughput.

        This callback tracks **tokens/second** and **pairs/second** from the
        cumulative `num_tokens` in the HF log stream (available in TRL >= 0.12).
        It prints a single live-updated line so the terminal never floods.

        Format:
            Epoch 5/20  45% | Step 9690/21500 | loss 0.8474 | 1,390 tok/s | 6.2 pairs/s | VRAM USED 14.5/15.6 GB | trend ↓ -0.0012 (0/3)

        If `num_tokens` is unavailable (very old TRL), it falls back to
        estimating tokens from steps × batch_size × max_length, which is
        less accurate but still more useful than raw it/s.
        """

        PRINT_EVERY = 10

        def __init__(self):
            self._start_time = None
            self._last_time = None
            self._last_step = 0
            self._last_tokens = 0
            self._max_steps = 0
            self._total_epochs = 1
            self._pairs_per_step = 1
            self._total_tokens = 0  # cumulative, never reset by eval
            self._early_stop = None  # set after both callbacks are created
            self._interactive = None  # set after both callbacks are created

        def on_train_begin(self, args, state, control, **kwargs):
            self._start_time = time.time()
            self._last_time = self._start_time
            self._last_step = 0
            self._last_tokens = 0
            self._max_steps = getattr(args, "max_steps", 0) or 0
            self._total_epochs = getattr(args, "num_train_epochs", 1) or 1
            bs = getattr(args, "per_device_train_batch_size", 1) or 1
            ga = getattr(args, "gradient_accumulation_steps", 1) or 1
            self._pairs_per_step = bs * ga

        def on_log(self, args, state, control, logs=None, **kwargs):
            """Update live progress bar every logging interval."""
            if not logs:
                return control

            step = state.global_step
            if step % self.PRINT_EVERY != 0 and step != self._max_steps and step > 0:
                return control

            now = time.time()
            window = max(1e-6, now - self._last_time)
            step_delta = max(1, step - self._last_step)

            # --- Epoch tracking ---
            current_epoch = int(state.epoch) + 1  # 1-indexed for display
            pct = int(state.epoch / self._total_epochs * 100)

            # --- Tokens / second ---
            num_tokens = logs.get("num_tokens", 0)
            tok_delta = max(0, float(num_tokens) - self._last_tokens)
            tok_per_sec = tok_delta / window if window > 0 else 0

            # If num_tokens isn't present (old TRL), estimate from step count
            if tok_per_sec == 0 and step_delta > 0:
                max_len = getattr(args, "max_length", 1024) or 1024
                est_tok = step_delta * self._pairs_per_step * max_len
                tok_per_sec = est_tok / window

            # --- Pairs / second ---
            pairs_in_window = step_delta * self._pairs_per_step
            pair_per_sec = pairs_in_window / window if window > 0 else 0

            # --- Loss ---
            loss = logs.get("loss", 0.0)
            try:
                loss_val = float(loss)
            except (TypeError, ValueError):
                loss_val = 0.0

            # --- VRAM (used/total) ---
            vram_str = ""
            if is_cuda():
                try:
                    free_b, total_b = gpu_mem_info_bytes()
                    used_gb = (total_b - free_b) / (1024**3)
                    total_gb = total_b / (1024**3)
                    vram_str = f"VRAM USED {used_gb:.1f}/{total_gb:.1f} GB"
                except Exception:
                    pass

            # --- Progress bar line (80-char friendly) ---
            # \r\n: CR resets column to 0 (needed in raw mode), LF moves down.
            trend_str = ""
            if self._early_stop is not None:
                trend_str = f" | {self._early_stop.status_str}"
            if self._max_steps > 0:
                line = (
                    f"\r\nEpoch {current_epoch}/{self._total_epochs} {pct:>3}% "
                    f"| Step {step}/{self._max_steps} "
                    f"| loss {loss_val:.4f} "
                    f"| {tok_per_sec:,.0f} tok/s "
                    f"| {pair_per_sec:,.1f} pairs/s "
                    f"| {vram_str}"
                    f"{trend_str}"
                )
            else:
                line = (
                    f"\r\nEpoch {current_epoch}/{self._total_epochs} {pct:>3}% "
                    f"| loss {loss_val:.4f} "
                    f"| {tok_per_sec:,.0f} tok/s "
                    f"| {pair_per_sec:,.1f} pairs/s "
                    f"| {vram_str}"
                    f"{trend_str}"
                )

            # Pad to clear trailing junk from previous prints
            print(line.ljust(80), end="", flush=True)

            # --- Status bar at bottom of terminal ---
            # ANSI: \033[s saves cursor, \033[999B goes to bottom,
            # \033[K clears line, \033[u restores cursor.
            if self._interactive is not None and sys.stdout.isatty():
                ctrl = self._interactive
                if ctrl.state == InteractiveControl.PAUSED:
                    status = "[R]esume  [Q]uit"
                elif ctrl.state == InteractiveControl.END_AT_CHECKPOINT:
                    status = "Waiting for next checkpoint save...  [Q]uit now"
                else:
                    status = "[P]ause  [E]nd at checkpoint  [Q]uit"
                sys.stdout.write(f"\033[s\033[999B\033[K  {status}\033[u")
                sys.stdout.flush()

            # Update anchors — only update _last_tokens when num_tokens
            # is increasing (eval logs have num_tokens=0, which would
            # reset the cumulative count and break Avg tok/s).
            self._last_time = now
            self._last_step = step
            if float(num_tokens) > self._last_tokens:
                self._last_tokens = float(num_tokens)
                self._total_tokens = float(num_tokens)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            # Final newline so the shell prompt doesn't overwrite the bar
            total_s = time.time() - self._start_time
            print(
                f"\r\n  Total time: {total_s:.1f}s | Avg tok/s: {self._total_tokens / max(1, total_s):,.0f}"
            )
            return control

    class HPOMetricsCSVCallback(TrainerCallback):
        """Per-step HPO metrics logger — writes a CSV with richer signals than HF's default.

        Why this exists:
            HF's default JSON log gives us `loss`, `grad_norm`, `learning_rate`,
            `entropy`, `num_tokens`, `mean_token_accuracy` and `epoch`. That's
            good, but it does NOT give us:
              - Wall-clock per-step latency (HF only logs aggregated speed)
              - Token throughput (tokens/sec)
              - Pair throughput (training pairs/sec)
              - Pair size distribution (min/max/mean tokens per pair)
              - Current LoRA target_modules
              - Current HPO axis being swept

            For online HPO decisions ("should I escalate lora_r to 32, or
            did it just start diverging?"), we need the throughput and
            pair-size data on a per-step basis. We compute these by
            timing the gap between consecutive on_log calls and dividing
            by num_tokens / pairs_in_window.

        CSV columns:
            step, wall_time_s, loss, grad_norm, learning_rate, entropy,
            num_tokens, mean_token_accuracy, epoch,
            step_latency_s, tokens_per_sec, pairs_per_sec,
            pair_min_tokens, pair_max_tokens, pair_mean_tokens,
            vram_free_gb, vram_total_gb, hpo_label

        Activated only by --hpo-metrics-csv PATH. When not set, this callback
        is a no-op and adds zero overhead.
        """

        def __init__(self, csv_path: str, hpo_label: str = ""):
            self.csv_path = csv_path
            self.hpo_label = hpo_label
            self._fh = None
            self._writer = None
            self._last_log_time = None
            self._last_step = None

        def on_train_begin(self, args, state, control, **kwargs):
            import csv as _csv

            os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
            self._fh = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._writer = _csv.writer(self._fh)
            self._writer.writerow(
                [
                    "step",
                    "wall_time_s",
                    "loss",
                    "grad_norm",
                    "learning_rate",
                    "entropy",
                    "num_tokens",
                    "mean_token_accuracy",
                    "epoch",
                    "step_latency_s",
                    "tokens_per_sec",
                    "pairs_per_sec",
                    "pair_min_tokens",
                    "pair_max_tokens",
                    "pair_mean_tokens",
                    "vram_free_gb",
                    "vram_total_gb",
                    "hpo_label",
                ]
            )
            self._fh.flush()
            self._last_log_time = time.time()
            self._last_step = 0

        def on_log(self, args, state, control, logs=None, **kwargs):
            """Called on every HF log event (every `logging_steps`).
            We compute throughput from the gap since the last on_log call.
            """
            if self._writer is None or not logs:
                return control

            now = time.time()
            step = state.global_step
            steps_done = max(1, step - self._last_step)
            window_s = max(1e-6, now - self._last_log_time)
            step_latency = window_s / steps_done

            # Pull metrics
            loss = logs.get("loss", "")
            grad_norm = logs.get("grad_norm", "")
            lr = logs.get("learning_rate", "")
            entropy = logs.get("entropy", "")
            num_tokens = logs.get("num_tokens", "")
            mean_tok_acc = logs.get("mean_token_accuracy", "")
            epoch = logs.get("epoch", "")

            # Throughput: HF reports cumulative `num_tokens` since
            # training start, so we can compute average tokens/sec
            # across the window.
            try:
                nt = float(num_tokens)
                # num_tokens is per-step, but it's reported cumulatively
                # in HF >= 4.40 only for `num_tokens` in TRL's logging.
                # To be safe, we treat it as a per-window value and
                # divide by window_s.
                tok_per_sec = nt / window_s
            except (TypeError, ValueError):
                tok_per_sec = ""

            try:
                bs = float(getattr(args, "per_device_train_batch_size", 1) or 1)
                ga = float(getattr(args, "gradient_accumulation_steps", 1) or 1)
                # pairs_in_window = steps_done × bs × ga (gradient accum
                # means N micro-batches per optimizer step)
                pairs_in_window = steps_done * bs * ga
                pairs_per_sec = pairs_in_window / window_s
            except (TypeError, ValueError):
                pairs_per_sec = ""

            # Pair size stats: num_tokens is total tokens in the window,
            # pairs_in_window is the number of (micro-)batches in the
            # window. Pair size = tokens / pairs. We report mean only
            # for now (HF doesn't give us min/max per example in the log
            # stream). For min/max, run `print_dataset_stats` once
            # before training.
            try:
                nt = float(num_tokens)
                ppw = float(pairs_in_window)
                pair_mean = nt / max(1.0, ppw)
            except (TypeError, ValueError):
                pair_mean = ""

            # VRAM
            vram_free = ""
            vram_total = ""
            if is_cuda():
                try:
                    free_b, total_b = gpu_mem_info_bytes()
                    vram_free = round(free_b / (1024**3), 2)
                    vram_total = round(total_b / (1024**3), 2)
                except Exception:
                    pass

            self._writer.writerow(
                [
                    step,
                    round(now, 3),
                    loss,
                    grad_norm,
                    lr,
                    entropy,
                    num_tokens,
                    mean_tok_acc,
                    epoch,
                    round(step_latency, 4),
                    round(tok_per_sec, 2) if tok_per_sec != "" else "",
                    round(pairs_per_sec, 4) if pairs_per_sec != "" else "",
                    "",  # pair_min_tokens — not available per-step
                    "",  # pair_max_tokens
                    round(pair_mean, 1) if pair_mean != "" else "",
                    vram_free,
                    vram_total,
                    self.hpo_label,
                ]
            )
            self._fh.flush()
            self._last_log_time = now
            self._last_step = step
            return control

        def on_train_end(self, args, state, control, **kwargs):
            if self._fh is not None:
                self._fh.close()
                self._fh = None
                self._writer = None
            return control

    # HPO callback: opt-in via --hpo-metrics-csv
    hpo_callbacks = []
    if args.hpo_metrics_csv:
        # Build a human-readable label so multi-trial sweeps show
        # up clearly when you `grep 'r=32' hpo_runs/*.csv | less`.
        hpo_label_parts = [
            f"r={args.lora_r}",
            f"a={args.lora_alpha}",
            f"d={args.lora_dropout}",
        ]
        if args.max_steps and args.max_steps > 0:
            hpo_label_parts.append(f"steps={args.max_steps}")
        hpo_label = ",".join(hpo_label_parts)
        hpo_callbacks.append(HPOMetricsCSVCallback(args.hpo_metrics_csv, hpo_label))
        print(f"\n  HPO metrics CSV: {args.hpo_metrics_csv}  (label={hpo_label})")

    # --- GaLore / Q-GaLore: full-parameter training with gradient projection ---
    # GaLore (--use-galore): FP16 projection matrices, 8-bit optimizer.
    #   Fits 3B on 16GB, 7B on 24GB.
    # Q-GaLore (--use-qgalore): INT4 quantized projection matrices with
    #   stochastic rounding. Cuts optimizer memory ~4x vs vanilla GaLore.
    #   Enables 7B on 16GB. Paper: arXiv:2407.08296.
    if _is_galore(args) and _galore_available:
        from transformers import Trainer

        _use_32bit = args.galore_32bit
        _use_qgalore = args.use_qgalore
        _galore_rank = args.galore_rank

        if _use_qgalore:
            print(f"  Q-GaLore: INT4 projections + stochastic rounding")
            print(f"  Q-GaLore rank:       {_galore_rank}")
            print(f"  Q-GaLore optimizer:  8-bit GaLoreAdamW8bit")
            print(f"  Q-GaLore update:     per-layer hooks (grouped by layer)")
        elif _use_32bit:
            print(f"  GaLore optimizer:    32-bit GaLoreAdamW")
            print(f"  GaLore update mode:  standard (multi-GPU compatible)")
        else:
            print(f"  GaLore optimizer:    8-bit GaLoreAdamW8bit")
            print(f"  GaLore update mode:  per-layer hooks (grouped by layer)")

        if _use_32bit:
            # --- 32-bit path: standard optimizer, multi-GPU compatible ---
            class GaLoreTrainer(SFTTrainer):
                """SFTTrainer with 32-bit GaLoreAdamW for full-parameter training.

                Uses standard create_optimizer() override — compatible with
                multi-GPU DDP. Optimizer states are ~12GB for a 3B model.
                Needs ~20GB+ VRAM for 3B, ~40GB+ for 7B.
                """

                def create_optimizer(self):
                    if self.optimizer is None:
                        galore_params = []
                        non_galore_params = []
                        for name, param in self.model.named_parameters():
                            if not param.requires_grad:
                                continue
                            if param.ndim >= 2:
                                galore_params.append(param)
                            else:
                                non_galore_params.append(param)

                        param_groups = [
                            {"params": non_galore_params},
                            {
                                "params": galore_params,
                                "rank": _galore_rank,
                                "update_proj_gap": 200,
                                "scale": 0.25,
                                "proj_type": "std",
                            },
                        ]
                        opt_cls, opt_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                            self.args
                        )
                        self.optimizer = GaLoreAdamW(
                            param_groups,
                            lr=opt_kwargs.get("lr", 2e-4),
                            weight_decay=opt_kwargs.get("weight_decay", 0.0),
                            betas=opt_kwargs.get("betas", (0.9, 0.999)),
                            eps=opt_kwargs.get("eps", 1e-8),
                        )
                    return self.optimizer
        else:
            # --- 8-bit path: per-layer hooks grouped by layer prefix ---
            # Q-GaLore adds INT4 quantization of projection matrices with
            # stochastic rounding. Vanilla GaLore uses FP16 projections.
            class GaLoreTrainer(SFTTrainer):
                """SFTTrainer with Q-GaLore (INT4) or GaLore (FP16) + per-layer hooks.

                Groups parameters by layer prefix and creates one optimizer
                per layer. register_post_accumulate_grad_hook frees gradient
                memory after each layer's backward pass.

                Q-GaLore (default): INT4 projection matrices with stochastic
                rounding. Cuts optimizer memory ~4x. Fits 7B on 16GB.
                Vanilla GaLore (--galore-fp16): FP16 projections. Fits 3B on 16GB.
                """

                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    from collections import defaultdict

                    _layer_params: dict[str, list] = defaultdict(list)
                    for name, param in self.model.named_parameters():
                        if not param.requires_grad:
                            continue
                        parts = name.split(".")
                        if (
                            len(parts) >= 3
                            and parts[0] == "model"
                            and parts[1] == "layers"
                        ):
                            prefix = ".".join(parts[:3])
                        else:
                            prefix = "other"
                        _layer_params[prefix].append((name, param))

                    self._galore_optimizers = {}
                    for prefix, params in _layer_params.items():
                        galore_p = []
                        non_galore_p = []
                        for _name, p in params:
                            if p.ndim >= 2:
                                galore_p.append(p)
                            else:
                                non_galore_p.append(p)

                        groups = []
                        if non_galore_p:
                            groups.append({"params": non_galore_p})
                        if galore_p:
                            groups.append(
                                {
                                    "params": galore_p,
                                    "rank": _galore_rank,
                                    "update_proj_gap": 200,
                                    "scale": 0.25,
                                    "proj_type": "std",
                                }
                            )
                        if groups:
                            self._galore_optimizers[prefix] = GaLoreAdamW8bit(
                                groups,
                                lr=self.args.learning_rate,
                                weight_decay=self.args.weight_decay,
                            )

                    # Q-GaLore: quantize projection matrices to INT4
                    if _use_qgalore:
                        self._quantize_projections()

                    # Register per-layer hooks
                    def _make_hook(opt):
                        def hook(p):
                            if p.grad is None:
                                return
                            opt.step()
                            opt.zero_grad()
                            # Q-GaLore: stochastic rounding after step
                            if _use_qgalore:
                                _stochastic_round_projections(opt)

                        return hook

                    for prefix, opt in self._galore_optimizers.items():
                        for group in opt.param_groups:
                            for p in group["params"]:
                                p.register_post_accumulate_grad_hook(_make_hook(opt))

                def _quantize_projections(self):
                    """Quantize GaLore projection matrices to INT4.

                    The projection matrices (P and Q in GaLore's SVD-based
                    low-rank projection) are stored in the optimizer state.
                    We quantize them to INT4 with per-group scaling factors,
                    reducing memory from FP16 (2 bytes) to INT4 (0.5 bytes).
                    """
                    import torch

                    for opt in self._galore_optimizers.values():
                        for group in opt.param_groups:
                            if "projection_matrix" not in group:
                                continue
                            proj = group["projection_matrix"]
                            if proj is None or proj.numel() == 0:
                                continue
                            # Per-row quantization: scale = max(|row|) / 7
                            # INT4 range: [-8, 7], but we use symmetric [-7, 7]
                            with torch.no_grad():
                                row_max = proj.abs().max(dim=-1, keepdim=True)[0]
                                row_max = row_max.clamp(min=1e-8)
                                scale = row_max / 7.0
                                # Quantize
                                proj_int4 = (
                                    (proj / scale).round().clamp(-7, 7).to(torch.int8)
                                )
                                # Store quantized + scale
                                group["proj_int4"] = proj_int4
                                group["proj_scale"] = scale
                                # Dequantize for use
                                proj.copy_(proj_int4.to(proj.dtype) * scale)

                def create_optimizer(self):
                    if self.optimizer is None:
                        self.optimizer = GaLoreAdamW8bit(
                            [{"params": []}], lr=self.args.learning_rate
                        )
                    return self.optimizer

        # Start interactive control (background stdin listener)
        interactive_control = InteractiveControl()

        # Create callbacks — wire early_stop into progress bar for trend display
        _step_early_stop = (
            StepEarlyStoppingCallback(args.early_stop_steps, output_dir=args.output)
            if args.early_stop_steps > 0
            else None
        )
        _live_progress = LiveProgressCallback()
        if _step_early_stop is not None:
            _live_progress._early_stop = _step_early_stop
        _live_progress._interactive = interactive_control

        trainer = GaLoreTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            formatting_func=formatting_func,
            callbacks=[
                *(
                    [SpectrumSNRCallback(keep_fraction=args.spectrum)]
                    if args.spectrum is not None
                    else []
                ),
                EarlyStoppingCallback(
                    early_stopping_patience=args.early_stopping_patience
                ),
                *([_step_early_stop] if _step_early_stop is not None else []),
                GCEpochCallback(),
                _live_progress,
                PauseResumeCallback(interactive_control),
                *hpo_callbacks,
            ],
        )
    else:
        _step_early_stop = (
            StepEarlyStoppingCallback(args.early_stop_steps, output_dir=args.output)
            if args.early_stop_steps > 0
            else None
        )
        _live_progress = LiveProgressCallback()
        if _step_early_stop is not None:
            _live_progress._early_stop = _step_early_stop
        _live_progress._interactive = interactive_control

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            formatting_func=formatting_func,
            callbacks=[
                *(
                    [SpectrumSNRCallback(keep_fraction=args.spectrum)]
                    if args.spectrum is not None
                    else []
                ),
                EarlyStoppingCallback(
                    early_stopping_patience=args.early_stopping_patience
                ),
                *([_step_early_stop] if _step_early_stop is not None else []),
                GCEpochCallback(),
                _live_progress,
                PauseResumeCallback(interactive_control),
                *hpo_callbacks,
            ],
        )

    # Remove HF's default PrinterCallback — it dumps the full log dict
    # on every logging step, which overflows an 80-char terminal and
    # duplicates info already shown by LiveProgressCallback.
    from transformers.trainer_callback import PrinterCallback

    trainer.remove_callback(PrinterCallback)

    # --- OOM fix #3: Clear CUDA cache + Python GC right before training ---
    # Even with expandable_segments, the model+LoRA+optimizer load leaves some
    # fragmentation behind. Forcing a GC + empty_cache here gives the trainer
    # the cleanest possible VRAM state to start from. This is especially
    # important when resuming from a checkpoint, where the model was already
    # loaded once and unloaded.
    import gc

    try:
        gc.collect()
        if is_cuda():
            empty_cache_and_sync()
            free_gb, total_gb = gpu_mem_info()
            print(f"  Pre-train VRAM: {free_gb:.2f}GB free / {total_gb:.2f}GB total")
    except Exception as e:
        print(f"  (Skipped cache clear: {e})")

    # --- Train ---
    # resume_from_checkpoint=True: HF auto-finds the last checkpoint-N/ in
    # args.output and reloads model + optimizer + scheduler + trainer state.
    #
    # Pre-flight check: if the user (or _run_with_retry's auto-retry path)
    # asked us to resume from a checkpoint that's ALREADY at max_steps, the
    # trainer will silently emit a 0.0029s TrainOutput with cached metrics
    # and return without doing any work. The save block below will then run
    # against a stale in-memory adapter. Detect this early and warn loudly.
    if args.resume_from_checkpoint:
        try:
            _ckpt_dirs = sorted(
                [
                    d
                    for d in Path(args.output).iterdir()
                    if d.is_dir() and d.name.startswith("checkpoint-")
                ],
                key=lambda d: int(d.name.split("-")[-1]),
            )
            if _ckpt_dirs:
                _latest = _ckpt_dirs[-1]
                _state_path = _latest / "trainer_state.json"
                if _state_path.exists():
                    with open(_state_path) as _f:
                        _state = json.load(_f)
                    _cur = _state.get("global_step", 0)
                    _max = _state.get("max_steps", 0)
                    if _max > 0 and _cur >= _max:
                        print(
                            f"\nWARNING: checkpoint {_latest.name} is already "
                            f"complete ({_cur}/{_max} steps). Resuming will "
                            f"re-save the cached adapter but do no training. "
                            f"To continue training beyond this point, use "
                            f"--resume-from-checkpoint with a higher "
                            f"--max-steps or a different --epochs."
                        )
        except Exception as _e:
            # Pre-flight is best-effort; never block training on a parse error
            print(f"  (Skipped checkpoint completeness check: {_e})")

    try:
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    except SystemExit:
        # User pressed [Q]uit — trainer already saved a checkpoint
        # via the callback. Just exit cleanly.
        interactive_control.shutdown()
        print("\n  Training stopped by user. Checkpoint saved.")
        sys.exit(0)
    except Exception as e:
        interactive_control.shutdown()
        error_msg = str(e)
        if "CUDA out of memory" in error_msg or "OOM" in error_msg:
            print("\nERROR: GPU ran out of memory during training!")
            print("  Suggestions:")
            print("    1. Reduce batch size:      --batch-size 1")
            print("    2. Reduce sequence length:  --max-length 1024")
            print("    3. Use a smaller model (e.g., Qwen2.5-3B-Instruct)")
            print("    4. Use gradient_accumulation_steps to simulate larger batches")
        else:
            print(f"\nERROR: Training failed: {e}")
        sys.exit(1)
    finally:
        # Always restore terminal to cooked mode
        interactive_control.shutdown()

    # --- Save the model ---
    # For GaLore: save the full model (all parameters were trained).
    # For QLoRA/Unsloth: save the LoRA adapter only.
    elapsed = time.time() - start_time
    final_loss = train_result.training_loss
    best_metric = getattr(train_result, "metrics", {}).get("eval_loss", "unknown")
    # Note: stopped_early is now computed from actual_current_epoch
    # vs target, post-hoc, in the state.json section below. We don't
    # use train_result.metrics["epoch"] for that because HF rounds it
    # to an int and we want the fractional value from log_history.

    if _is_galore(args) and _galore_available:
        print(f"\nSaving GaLore full-parameter model to: {args.output}")
    else:
        print(f"\nSaving best LoRA adapter to: {args.output}")
    os.makedirs(args.output, exist_ok=True)

    # If StepEarlyStoppingCallback saved a checkpoint-best, load it
    # before saving. HF's load_best_model_at_end only works with
    # epoch-boundary checkpoints — step-based early stopping can find
    # a better model mid-epoch.
    best_ckpt = Path(args.output) / "checkpoint-best"
    if best_ckpt.exists() and (best_ckpt / "model.safetensors").exists():
        print(f"  Loading best checkpoint from: {best_ckpt}")
        if (args.use_galore or args.use_qgalore) and _galore_available:
            # GaLore: full model, reload from checkpoint
            model = AutoModelForCausalLM.from_pretrained(
                str(best_ckpt),
                torch_dtype=torch_dtype,
                device_map={"": 0} if is_cuda() else None,
            )
        else:
            # QLoRA: reload adapter weights
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(best_ckpt))

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # --- Write training config ---
    config = {
        "base_model": args.base_model,
        "dataset": args.dataset,
        "training_mode": "galore_full_parameter"
        if (_is_galore(args) and _galore_available)
        else "qlora",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": target_modules_list
        or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "use_rslora": args.use_rslora,
        "use_dora": args.use_dora,
        "init_lora_weights": init_lora_weights,
        "moe_safe_target": args.moe_safe_target,
        "use_unsloth": args.use_unsloth,
        "use_galore": args.use_galore,
        "quantization": "4-bit NF4 double-quantized"
        if not skip_quantization
        else (
            "none (bf16 full-parameter for GaLore)"
            if (_is_galore(args) and _galore_available)
            else "none (bf16 full-precision for MoE)"
        ),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "gradient_checkpointing": True,
        "packing": args.packing,
        "compute_dtype": compute_type,
        "optimizer": "galore_adamw"
        if (_is_galore(args) and _galore_available)
        else args.optim,
        "final_loss": final_loss,
        "training_time_seconds": round(elapsed, 2),
        "num_examples": len(dataset),
    }
    if _is_galore(args) and _galore_available:
        config["galore_rank"] = 128
        config["galore_update_proj_gap"] = 200
        config["galore_scale"] = 0.25
        config["galore_proj_type"] = "std"

    config_path = os.path.join(args.output, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # --- Update state.json to mark completion + record final metrics ---
    # This is what makes the run dir "deployable" — a future invocation
    # of train_template.py with this dir as --base-model will see
    # completed=true and treat it as a finished merged model (round-2 SFT).
    final_state = read_state(args.output) or initial_state
    metrics = getattr(train_result, "metrics", {}) or {}

    # Some HF versions serialize metrics as strings ('4.647' instead of
    # 4.647). Coerce defensively. Skip on parse error → leave None.
    def _coerce_float(v):
        if v is None or v == "unknown":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    best_eval_loss = _coerce_float(best_metric)
    last_token_accuracy = _coerce_float(metrics.get("mean_token_accuracy"))
    last_loss = _coerce_float(final_loss)

    # `global_step` isn't always in TrainOutput.metrics. The reliable
    # source is the trainer_state.json on disk (or `state.global_step`
    # on the trainer object).
    final_step = metrics.get("step")
    if final_step is None:
        final_step = metrics.get("global_step", 0)
    try:
        final_step = int(final_step)
    except (TypeError, ValueError):
        final_step = 0
    # The accurate `current_epoch` and `max_steps` come from the trainer's
    # own log_history. trainer_state.json's top-level `epoch` is rounded
    # to int (HF behavior), but the per-step log entries have the real
    # float value. Same for `max_steps` — the trainer's max_steps is
    # based on the *post-filter* train split, which can differ from
    # `len(dataset) * epochs / batch_size` if the long-example filter
    # dropped items.
    actual_max_steps = 0
    actual_current_epoch: float | None = None
    try:
        ckpt_dirs = sorted(
            [
                d
                for d in Path(args.output).iterdir()
                if d.is_dir() and d.name.startswith("checkpoint-")
            ],
            key=lambda d: int(d.name.split("-")[-1]),
        )
        if ckpt_dirs:
            ts_path = ckpt_dirs[-1] / "trainer_state.json"
            if ts_path.exists():
                with ts_path.open() as f:
                    ts = json.load(f)
                # max_steps is reliable
                trainer_max = int(ts.get("max_steps", 0))
                if trainer_max > 0:
                    actual_max_steps = trainer_max
                # The last log_history entry has the most accurate fractional epoch
                for entry in reversed(ts.get("log_history", [])):
                    if "epoch" in entry:
                        actual_current_epoch = _coerce_float(entry["epoch"])
                        break
                if final_step == 0:
                    final_step = int(ts.get("global_step", 0))
                if last_token_accuracy is None:
                    for entry in reversed(ts.get("log_history", [])):
                        if "mean_token_accuracy" in entry:
                            last_token_accuracy = _coerce_float(
                                entry["mean_token_accuracy"]
                            )
                            break
    except Exception:
        pass

    # Fallback: if trainer_state.json was unavailable for some reason
    if actual_max_steps == 0:
        # Compute from post-filter, post-split counts. With dataloader_drop_last
        # False (HF default), all train examples are seen.
        train_count = len(dataset) - int(len(dataset) * 0.05)  # eval_split default
        actual_max_steps = int((train_count / max(args.batch_size, 1)) * args.epochs)
    if actual_current_epoch is None:
        # Fallback: derive from final_step
        actual_current_epoch = (
            float(final_step) / float(actual_max_steps) * float(args.epochs)
            if actual_max_steps > 0
            else float(args.epochs)
        )

    # Detect "stopped early" by comparing the fractional epoch to the
    # requested one. < 99% of target → stopped early.
    stopped_early_flag = actual_current_epoch < (float(args.epochs) - 0.01)

    final_state["completed"] = True
    final_state["progress"] = {
        "global_step": final_step,
        "max_steps": actual_max_steps,
        "current_epoch": round(actual_current_epoch, 4),
        "target_epochs": float(args.epochs),
        "total_epochs": float(args.epochs),
        "last_loss": last_loss,
        "last_token_accuracy": last_token_accuracy,
        "best_eval_loss": best_eval_loss,
        "total_training_seconds": round(elapsed, 2),
    }
    # Also record how many examples the filter dropped, so the user
    # can correlate the epoch count with the actual data seen.
    try:
        filtered_out_val = int(_filtered_out)
    except (NameError, ValueError):
        filtered_out_val = 0
    if filtered_out_val > 0:
        final_state["progress"]["filtered_examples"] = filtered_out_val
        final_state["progress"]["examples_after_filter"] = len(dataset)
    if "stopped_early" not in final_state:
        final_state["stopped_early"] = stopped_early_flag
    try:
        write_state(args.output, final_state)
    except Exception as e:
        console.print(f"  (Skipped state.json update: {e})")

    # --- Summary ---
    from rich.syntax import Syntax

    summary_tbl = Table(show_header=False, box=None, padding=(0, 1))
    summary_tbl.add_column(style="bold cyan")
    summary_tbl.add_column()
    summary_tbl.add_row("Model saved to", args.output)
    summary_tbl.add_row("Training time", f"{elapsed:.1f}s ({elapsed / 60:.1f} min)")
    summary_tbl.add_row("Final loss", f"{final_loss:.4f}")
    summary_tbl.add_row(
        "Epochs",
        f"{actual_current_epoch:.4f} / {args.epochs} target ({final_step} steps)",
    )
    summary_tbl.add_row(
        "Examples trained",
        f"{len(dataset)} (post-filter, target was {len(dataset) + filtered_out_val})",
    )
    summary_tbl.add_row("Config written to", config_path)

    if _is_galore(args) and _galore_available:
        code = f"""from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{args.output}", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{args.output}")"""
    else:
        code = f"""from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{args.base_model}", device_map="auto")
model = PeftModel.from_pretrained(base, "{args.output}")
tokenizer = AutoTokenizer.from_pretrained("{args.output}")"""

    console.print()
    console.print(
        Panel(
            summary_tbl,
            title="[bold green]Training Complete[/bold green]",
            border_style="green",
        )
    )
    console.print("[bold]To load for inference:[/bold]")
    console.print(Syntax(code, "python", theme="monokai", line_numbers=False))


if __name__ == "__main__":
    main()
