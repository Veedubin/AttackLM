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
import time
from pathlib import Path
from typing import Any

import torch

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
            "Unsloth provides 2-5x faster training, 70% less VRAM, and "
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
            "Requires 'uv pip install attacklm[galore]' or "
            "'pip install galore-torch'. Default: OFF (standard QLoRA)."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Checks & Validation
# ---------------------------------------------------------------------------


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
        print(f"  WARNING: state.json at {sp} is unreadable ({e}); ignoring")
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
        print(f"ERROR: Python 3.9+ required, got {sys.version}")
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

    print_hardware_banner()

    # --moe-safe-target forces bf16 (BnB 4-bit is incompatible with MoE
    # expert weights per Unsloth guidance)
    if args and getattr(args, "moe_safe_target", False):
        print("Mixed precision: BF16 (forced by --moe-safe-target)")
        return "bf16"

    # Explicit overrides (mutually exclusive group in argparse)
    if args and getattr(args, "fp32", False):
        print("Mixed precision: FP32 (forced by --fp32)")
        return "fp32"
    if args and getattr(args, "bf16", False):
        print("Mixed precision: BF16 (forced by --bf16)")
        return "bf16"
    if args and getattr(args, "fp16", False):
        print("Mixed precision: FP16 (forced by --fp16)")
        return "fp16"

    if not is_cuda():
        if is_mps():
            print("Apple Silicon (MPS) detected — training will be very slow.")
            return "fp32"
        print(
            "WARNING: No CUDA / ROCm GPU detected. Training will be extremely slow on CPU."
        )
        print(
            "         Consider using Google Colab (T4/A100) or RunPod if you lack a local GPU."
        )
        return "fp32"

    gpu_name, gpu_mem = gpu_name_and_memory()
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")

    if gpu_mem < 10:
        print(f"WARNING: GPU has only {gpu_mem:.1f} GB VRAM.")
        print("         If you hit OOM, try: --batch-size 1 --max-length 1024")

    # Auto-detect bf16 capability on Ampere+ GPUs (compute capability >= 8.0).
    # Ampere (8.0), Ada (8.9), Hopper (9.0), Blackwell (10.0) all support bf16.
    cc_major = 0
    try:
        if torch.cuda.is_available():
            cc_major = torch.cuda.get_device_properties(0).major
    except Exception:
        pass

    if cc_major >= 8:
        print(f"Mixed precision: BF16 (auto-detected, compute capability {cc_major}.x)")
        return "bf16"
    else:
        print("Mixed precision: FP16 (GPU lacks BF16 hardware)")
        return "fp16"


def validate_dataset(dataset_path: str) -> None:
    """Validate dataset file exists and has correct format."""
    path = Path(dataset_path)

    if not path.exists():
        print(f"ERROR: Dataset file not found: {dataset_path}")
        print(
            "       See README.md §Quickstart for how to generate datasets (run scripts/extract_*.py)."
        )
        sys.exit(1)

    if not path.suffix == ".jsonl":
        print(f"WARNING: Expected .jsonl extension, got '{path.suffix}'")

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
                        print(
                            f"WARNING: Line {i + 1} missing 'messages' key — skipping"
                        )
                else:
                    msgs = obj["messages"]
                    if not isinstance(msgs, list) or len(msgs) < 2:
                        errors += 1
                        if errors <= 3:
                            print(
                                f"WARNING: Line {i + 1} has invalid 'messages' format"
                            )
                    else:
                        for m in msgs:
                            if "role" not in m or "content" not in m:
                                errors += 1
                                if errors <= 3:
                                    print(
                                        f"WARNING: Line {i + 1} has message without 'role' or 'content'"
                                    )
                                break
            except json.JSONDecodeError as e:
                errors += 1
                if errors <= 3:
                    print(f"WARNING: Line {i + 1} JSON parse error: {e}")

    if errors > 3:
        print(f"WARNING: {errors} total format issues in dataset (showing first 3)")

    if line_count == 0:
        print(f"ERROR: Dataset file is empty: {dataset_path}")
        sys.exit(1)

    print(f"Dataset: {line_count} examples, {errors} format issues")


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
    num_examples = len(dataset)
    print(f"\n{'=' * 60}")
    print(" DATASET STATISTICS")
    print(f"{'=' * 60}")
    print(f"  Examples:       {num_examples}")

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
        print(f"  Avg word count: {avg_words:.0f} (sampled {sample_size})")
        print(f"  Avg token est:   ~{avg_tokens_est} (1.3x word ratio)")
        print(f"  Max word count:  {max(word_lengths)}")
        print(f"  Min word count:  {min(word_lengths)}")

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
            print(f"  Exact token stats (sampled {len(token_lengths)}):")
            print(
                f"    min={stats['min_tokens']}  median={stats['median_tokens']}  "
                f"mean={stats['mean_tokens']}  p95={stats['p95_tokens']}  "
                f"p99={stats['p99_tokens']}  max={stats['max_tokens']}"
            )
            print(f"  → wrote HPO stats to {hpo_stats_path}")

    print(f"{'=' * 60}\n")


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


def main() -> None:
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
        print(f"  --moe-safe-target: excluded modules: {excluded}")
        print(f"  --moe-safe-target: target modules: {target_modules_list}")
    elif args.target_modules:
        target_modules_list = [m.strip() for m in args.target_modules.split(",")]
        print(f"  --target-modules: {target_modules_list}")
    else:
        target_modules_list = None  # use defaults in get_qlora_config

    # LoftQ init: tentatively set based on args; will be overridden if
    # the model is already quantized (checked after model load).
    init_lora_weights = "loftq" if args.loftq_init else True

    print("\n" + "=" * 60)
    print(" AttackLM QLoRA Fine-Tuning Template")
    print("=" * 60)
    mode_label = "DRY RUN (no training)" if is_dry_run else "LIVE TRAINING"
    print(f" Mode: {mode_label}")
    print(f" Base model:  {args.base_model}")
    print(f" Dataset:     {args.dataset}")
    print(f" Output:      {args.output}")
    if args.no_timestamp:
        print(
            "              (no-timestamp mode: clobbers existing runs without --force)"
        )
    else:
        # If the output has a _YYYY-MM-DD_HH-MM suffix, note that we
        # preserved it. Otherwise note that we appended one.
        import re as _re

        if _re.search(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}", args.output):
            print("              (timestamp suffix preserved)")
        else:
            print("              (auto-timestamped: each run is preserved)")
    print(f" Epochs:      {args.epochs}")
    print(f" Batch size:   {args.batch_size}")
    print(f" Max length:   {args.max_length}")
    print(
        f" LoRA r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}"
    )
    print(f" Optimizer:  {args.optim}")
    print("=" * 60 + "\n")

    # --- Validate dataset path ---
    validate_dataset(args.dataset)

    # --- Check GPU ---
    compute_type = check_gpu(args)

    # --- Load dependencies (deferred so --dry-run can run without GPU) ---
    print("\nLoading libraries...")
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print(
            "Install with: pip install transformers datasets trl peft bitsandbytes accelerate"
        )
        sys.exit(1)

    # --- Load tokenizer ---
    print(f"\nLoading tokenizer: {args.base_model}")
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
                    print(
                        "  ↻ Detected started run (state.json: completed=false, "
                        "checkpoint-N/ present). Auto-resuming from latest checkpoint."
                    )
                    args.resume_from_checkpoint = True
            else:
                # Marked-started but no actual training happened. Treat as
                # a fresh base. This happens when --dry-run is run twice
                # on the same dir.
                print(
                    "  ↻ Detected marked-started run but no checkpoint-N/ found. "
                    "Treating as base for a fresh training run."
                )

        # Surface the round-2 SFT case clearly to the user
        if base_state is not None and base_state.get("completed", False):
            print(
                "  ✓ Detected completed run (state.json: completed=true). "
                "Round-2 SFT: training a fresh LoRA on top of the merged weights."
            )
            prev_hp = base_state.get("hparams", {})
            if prev_hp:
                print(
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
    except Exception as e:
        print(f"ERROR: Failed to load tokenizer: {e}")
        print(
            f"  Check that '{args.base_model}' exists on HuggingFace and you have internet access."
        )
        sys.exit(1)

    # --- Load dataset ---
    print(f"Loading dataset: {args.dataset}")
    try:
        dataset = load_dataset("json", data_files=args.dataset, split="train")
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
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
    print(f"\n  Assistant-loss detection: {loss_cfg['reason']}")
    print(
        f"  → assistant_only_loss={loss_cfg['assistant_only_loss']}, "
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
        print(
            f"  Dropped {dropped} examples exceeding {int(args.max_length * 1.5)} tokens "
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
        "ON (full-parameter, rank=128, update_proj_gap=200, scale=0.25)"
        if args.use_galore
        else "OFF"
    )

    print("=" * 60)
    print(" TRAINING PLAN")
    print("=" * 60)
    print(f"  Model:              {args.base_model}")
    if args.use_unsloth:
        print("  Quantization:       4-bit (Unsloth internal)")
    elif args.use_galore:
        print("  Quantization:       NONE (bf16 full-parameter for GaLore)")
    elif not args.moe_safe_target:
        print("  Quantization:       4-bit NF4 (double quant)")
    else:
        print("  Quantization:       NONE (bf16 full-precision for MoE)")
    if args.use_galore:
        print("  Training mode:      GaLore full-parameter (no LoRA)")
    else:
        print(f"  LoRA rank:          {args.lora_r}")
        print(f"  LoRA alpha:         {args.lora_alpha}")
        print(f"  LoRA dropout:       {args.lora_dropout}")
        print(f"  Target modules:     {_tm_display}")
        print(f"  RSLoRA:             {_rslora_display}")
        print(f"  DoRA:               {_dora_display}")
        print(f"  LoftQ init:         {_loftq_display}")
    print(f"  MoE-safe target:    {_moe_display}")
    print(f"  Unsloth:            {_unsloth_display}")
    print(f"  GaLore:             {_galore_display}")
    print(f"  Epochs:             {args.epochs}")
    print(f"  Batch size:         {args.batch_size}")
    print(f"  Max seq length:     {args.max_length}")
    print("  Gradient checkpoint: True")
    print(f"  Compute dtype:      {compute_type}")
    print(f"  Save steps:         {args.save_steps}")
    print(f"  Gradient accum:     {args.gradient_accumulation_steps}")
    print("  Save strategy:      steps")
    print("  Save total limit:   2")
    print("  Logging steps:      10")
    print(f"  Output dir:         {args.output}")
    print(f"  Resume checkpoint:  {args.resume_from_checkpoint}")
    print(f"  Optimizer:          {args.optim}")
    print(f"  Packing:            {args.packing}  (--packing/--no-packing)")
    print("=" * 60 + "\n")

    # ===================================================================
    # DRY RUN — print plan and exit
    # ===================================================================
    if is_dry_run:
        print("=" * 60)
        print(" DRY RUN COMPLETE")
        print("=" * 60)
        print("  Dataset validated successfully.")
        print("  No training was performed.")
        print("")
        print("  To actually train, re-run with --train flag:")
        print(
            f"    python train_template.py --dataset {args.dataset} --output {args.output} --train"
        )
        print("")
        print("  Memory estimate for Qwen2.5-7B QLoRA:")
        print("    ~10-12 GB VRAM at batch_size=2, max_length=2048")
        print("    ~6-8 GB VRAM at batch_size=1, max_length=1024")
        if args.use_unsloth:
            print("")
            print("  With --use-unsloth (70% less VRAM):")
            print("    ~4-5 GB VRAM at batch_size=2, max_length=2048")
            print("    ~3-4 GB VRAM at batch_size=1, max_length=1024")
            print("    ~13B model fits in 16GB with Unsloth QLoRA")
        if args.use_galore:
            print("")
            print("  With --use-galore (full-parameter training):")
            print("    ~10-12 GB VRAM at batch_size=1, max_length=2048 (3B model)")
            print("    ~14-16 GB VRAM at batch_size=1, max_length=2048 (7B model)")
            print("    GaLore trains ALL parameters — no LoRA adapters needed")
        print("=" * 60)
        return

    # ===================================================================
    # LIVE TRAINING
    # ===================================================================
    print("Starting training...\n")
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
        print(
            f"  ↻ State recorded at {args.output}/state.json (version {_STATE_VERSION})"
        )
    except Exception as e:
        # Non-fatal — we can still train, we just lose the resume signal
        print(f"  (Skipped state.json write: {e})")

    # --- Mutual exclusivity: GaLore vs Unsloth ---
    if args.use_galore and args.use_unsloth:
        print(
            "ERROR: --use-galore and --use-unsloth are mutually exclusive.\n"
            "  GaLore is full-parameter training (no LoRA adapters).\n"
            "  Unsloth is optimized QLoRA (LoRA adapters on quantized base).\n"
            "  Choose one: --use-galore OR --use-unsloth, not both."
        )
        sys.exit(1)

    # --- Unsloth: import BEFORE transformers/peft/trl (required for optimizations) ---
    _unsloth_available = False
    if args.use_unsloth:
        try:
            import unsloth  # noqa: F401 — must be first for monkey-patching
            from unsloth import FastLanguageModel, is_bfloat16_supported

            _unsloth_available = True
            print("  Unsloth: loaded (FastLanguageModel available)")
        except ImportError:
            print(
                "ERROR: --use-unsloth requires the 'unsloth' package.\n"
                "  Install: uv pip install attacklm[unsloth]\n"
                "  Or:      pip install unsloth"
            )
            sys.exit(1)

    # --- GaLore: import galore-torch for full-parameter training ---
    _galore_available = False
    if args.use_galore:
        try:
            from galore_torch import GaLoreAdamW  # noqa: F401

            _galore_available = True
            print("  GaLore: loaded (GaLoreAdamW available)")
        except ImportError:
            print(
                "ERROR: --use-galore requires the 'galore-torch' package.\n"
                "  Install: uv pip install attacklm[galore]\n"
                "  Or:      pip install galore-torch"
            )
            sys.exit(1)

    try:
        from transformers import AutoModelForCausalLM
        from peft import get_peft_model
        from trl import SFTTrainer, SFTConfig
        from transformers.trainer_callback import EarlyStoppingCallback
    except ImportError as e:
        print(f"ERROR: Missing dependency for training: {e}")
        print(
            "Install with: pip install transformers datasets trl peft bitsandbytes accelerate"
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
    skip_quantization = args.moe_safe_target or args.use_unsloth or args.use_galore

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
        )
        # OOM fix #13: FlashAttention 2 for varlen (padding-free) support
        attn_impl = suggest_attn_implementation(args.packing)
        load_kwargs["attn_implementation"] = attn_impl

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
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_resolved, **load_kwargs
                )
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
    if args.use_galore and _galore_available:
        # GaLore: full-parameter training — no LoRA adapter needed.
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
        save_strategy="steps",
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_total_limit=args.early_stopping_patience + 1,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Precision — driven by compute_type determined from GPU / CLI flags
        fp16=(compute_type == "fp16"),
        bf16=(compute_type == "bf16"),
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
    # --- OOM fix #4: Per-eval CUDA cache clear callback ---
    # The eval pass at the end of each epoch allocates fresh activations and
    # torch.no_grad context tensors. When training resumes for the next epoch,
    # PyTorch's allocator still holds the eval pass's scratch blocks — even
    # after the eval tensors go out of scope. By step 13 of epoch 2, this
    # residual allocation is enough to push a borderline example over the
    # VRAM ceiling. Clearing after every eval prevents this accumulation.
    from transformers import TrainerCallback

    class GCEpochCallback(TrainerCallback):
        """Run gc.collect() + empty_cache() after every eval to defragment VRAM.

        Also monitors VRAM after every optimizer step and triggers an
        emergency cache clear if free memory drops below 2GB — this
        catches the case where peak transient allocations during a
        forward pass push us close to the ceiling.
        """

        # OOM fix #10: VRAM threshold for emergency cache clear
        # (in bytes). If free VRAM drops below this after an optimizer
        # step, force a gc.collect + empty_cache. 2GB is conservative —
        # leaves room for one more forward+backward pass even on a
        # 1024-token batch. The first optimizer step after this
        # threshold trigger will slow down by ~200ms (cache clear
        # overhead) but prevents a much longer OOM-retry restart.
        EMERGENCY_CLEAR_THRESHOLD_BYTES = 2 * (1024**3)

        def on_evaluate(self, args, state, control, **kwargs):
            import gc

            gc.collect()
            if is_cuda():
                empty_cache_and_sync()
                free_gb, total_gb = gpu_mem_info()
                print(
                    f"  [GCEpochCallback] Post-eval VRAM: "
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
                    f"  [GCEpochCallback] Step {state.global_step} emergency "
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
            Step  420/1200 | loss 1.234 |  8,192 tok/s |  42.0 pair/s | VRAM 12.3/16.0 GB

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
            self._pairs_per_step = 1

        def on_train_begin(self, args, state, control, **kwargs):
            self._start_time = time.time()
            self._last_time = self._start_time
            self._last_step = 0
            self._last_tokens = 0
            self._max_steps = getattr(args, "max_steps", 0) or 0
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

            # --- VRAM ---
            vram_str = ""
            if is_cuda():
                try:
                    free_b, total_b = gpu_mem_info_bytes()
                    vram_str = (
                        f"VRAM {free_b / (1024**3):.1f}/{total_b / (1024**3):.1f} GB"
                    )
                except Exception:
                    pass

            # --- Progress bar line ---
            if self._max_steps > 0:
                bar_len = 20
                filled = int(bar_len * step / self._max_steps)
                bar = "█" * filled + "░" * (bar_len - filled)
                line = (
                    f"\rStep {step:>6}/{self._max_steps} | {bar} | "
                    f"loss {loss_val:.4f} | {tok_per_sec:,.0f} tok/s | "
                    f"{pair_per_sec:,.1f} pair/s | {vram_str}"
                )
            else:
                line = (
                    f"\rStep {step:>6} | loss {loss_val:.4f} | "
                    f"{tok_per_sec:,.0f} tok/s | {pair_per_sec:,.1f} pair/s | {vram_str}"
                )

            # Pad with spaces to clear any trailing junk from previous prints
            print(line.ljust(100), end="", flush=True)

            # Update anchors
            self._last_time = now
            self._last_step = step
            self._last_tokens = float(num_tokens)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            # Final newline so the shell prompt doesn't overwrite the bar
            total_s = time.time() - self._start_time
            print(
                f"\n  Total time: {total_s:.1f}s | Avg tok/s: {self._last_tokens / max(1, total_s):,.0f}"
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

    # --- GaLore: use custom trainer with GaLoreAdamW optimizer ---
    if args.use_galore and _galore_available:
        from transformers import Trainer
        from galore_torch import GaLoreAdamW

        class GaLoreTrainer(SFTTrainer):
            """SFTTrainer subclass that uses GaLoreAdamW for full-parameter training.

            GaLore (Gradient Low-Rank Projection) projects gradients into a
            low-rank space during optimization, enabling full-parameter learning
            on consumer GPUs. This trainer overrides create_optimizer() to use
            GaLoreAdamW instead of the standard HF optimizer.

            Key hyperparameters (tuned for 3B-7B models):
              - rank=128: projection rank (higher = more capacity, more VRAM)
              - update_proj_gap=200: steps between SVD recomputation
              - scale=0.25: gradient scaling factor
              - proj_type='std': standard SVD-based projection
            """

            def create_optimizer(self):
                """Override to use GaLoreAdamW with gradient low-rank projection.

                Separates parameters into two groups:
                  - 2D params (weight matrices): GaLore projection applied
                  - 1D params (biases, norms, embeddings): standard AdamW
                """
                if self.optimizer is None:
                    galore_params = []
                    non_galore_params = []

                    for name, param in self.model.named_parameters():
                        if not param.requires_grad:
                            continue
                        # Apply GaLore to 2D weight matrices (linear layers,
                        # attention projections). Skip 1D params (biases,
                        # layer norms, embeddings) — they get standard AdamW.
                        if param.ndim >= 2:
                            galore_params.append(param)
                        else:
                            non_galore_params.append(param)

                    param_groups = [
                        {"params": non_galore_params},
                        {
                            "params": galore_params,
                            "rank": 128,
                            "update_proj_gap": 200,
                            "scale": 0.25,
                            "proj_type": "std",
                        },
                    ]

                    optimizer_cls, optimizer_kwargs = (
                        Trainer.get_optimizer_cls_and_kwargs(self.args)
                    )
                    self.optimizer = GaLoreAdamW(
                        param_groups,
                        lr=optimizer_kwargs.get("lr", 2e-4),
                        weight_decay=optimizer_kwargs.get("weight_decay", 0.0),
                        betas=optimizer_kwargs.get("betas", (0.9, 0.999)),
                        eps=optimizer_kwargs.get("eps", 1e-8),
                    )
                return self.optimizer

        trainer = GaLoreTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            formatting_func=formatting_func,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=args.early_stopping_patience
                ),
                GCEpochCallback(),
                LiveProgressCallback(),
                *hpo_callbacks,
            ],
        )
    else:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            formatting_func=formatting_func,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=args.early_stopping_patience
                ),
                GCEpochCallback(),
                LiveProgressCallback(),
                *hpo_callbacks,
            ],
        )

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
    except Exception as e:
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

    if args.use_galore and _galore_available:
        print(f"\nSaving GaLore full-parameter model to: {args.output}")
    else:
        print(f"\nSaving best LoRA adapter to: {args.output}")
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # --- Write training config ---
    config = {
        "base_model": args.base_model,
        "dataset": args.dataset,
        "training_mode": "galore_full_parameter"
        if (args.use_galore and _galore_available)
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
            if (args.use_galore and _galore_available)
            else "none (bf16 full-precision for MoE)"
        ),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "gradient_checkpointing": True,
        "packing": args.packing,
        "compute_dtype": compute_type,
        "optimizer": "galore_adamw"
        if (args.use_galore and _galore_available)
        else args.optim,
        "final_loss": final_loss,
        "training_time_seconds": round(elapsed, 2),
        "num_examples": len(dataset),
    }
    if args.use_galore and _galore_available:
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
        print(f"  (Skipped state.json update: {e})")

    # --- Summary ---
    print("\n" + "=" * 60)
    print(" TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model saved to:      {args.output}")
    print(f"  Training time:       {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Final loss:          {final_loss:.4f}")
    print(
        f"  Epochs:              {actual_current_epoch:.4f} / {args.epochs} target ({final_step} steps)"
    )
    print(
        f"  Examples trained:   {len(dataset)} (post-filter, target was {len(dataset) + filtered_out_val})"
    )
    print(f"  Config written to:   {config_path}")
    print("=" * 60)
    if args.use_galore and _galore_available:
        print("\nTo load the GaLore-trained full model for inference:")
        print(f"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("{args.output}", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("{args.output}")
    """)
    else:
        print("\nTo load the trained adapter for inference:")
        print(f"""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained("{args.base_model}", device_map="auto")
    model = PeftModel.from_pretrained(base, "{args.output}")
    tokenizer = AutoTokenizer.from_pretrained("{args.output}")
    """)


if __name__ == "__main__":
    main()
