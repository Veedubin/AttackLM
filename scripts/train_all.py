#!/usr/bin/env python3
"""
train_all.py — Train AttackLM agents from bucket-organized data.

Bucket layout:
    data/datasets/buckets/
        manifest.json
        <bucket_name>/data.jsonl          # Top-level (tactics, orchestrator)
        ai-models/
            prompt-injection/data.jsonl   # AI red team (TA0040)
            jailbreaking/data.jsonl
        tools/
            metasploit/data.jsonl
            infection_monkey/data.jsonl
            rta/data.jsonl

Default mode (no flags):
    Trains one model per bucket. The 11 default buckets are:
      10 MITRE tactic buckets (collection, command_and_control, ...)
      1 orchestrator bucket

--single-model:
    Combines selected buckets into one training set and trains a single
    model. By default combines 10 tactic buckets. Add flags to extend:
      --include-orchestrator       (adds the orchestrator bucket)
      --model-attacks              (adds all ai-models/* buckets: prompt-injection + jailbreaking)
      --include-tools              (adds all tools/* buckets: metasploit + infection_monkey + rta)

    Combined dataset is cached at data/datasets/combined/<hash>.jsonl
    and reused on re-runs (same buckets + same flags = same hash).

Simple, low-overhead logging. Single main log file. Per-agent timeout,
skip-completed, and start-from supported.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Make bucket_loader importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bucket_loader import (  # noqa: E402
    build_combined,
    get_default_train_buckets,
    get_tactic_buckets,
    get_tool_buckets,
    get_ai_model_buckets,
    get_orchestrator_bucket,
    get_bucket,
    list_buckets,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "data" / "datasets"
BUCKETS_DIR = DATASETS_DIR / "buckets"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"


def get_default_models() -> list[tuple]:
    """Build the default MODELS list from the bucket manifest.

    Returns list of (bucket_path, agent_name, bucket_meta) tuples in
    stable order: tactics first (alphabetical), then orchestrator.

    ai-models/* and tools/* are NOT in the default training set — they're
    opt-in via --model-attacks and --include-tools (single-model only).
    """
    models = []
    # 1) All MITRE tactic buckets
    for b in get_tactic_buckets():
        # Use last path component for agent name (handles "collection" and
        # also "ai-models/prompt-injection" gracefully)
        slug = b["path"].split("/")[-1]
        models.append((b["path"], f"{slug}-agent", b))
    # 2) Orchestrator
    orch = get_orchestrator_bucket()
    if orch and orch["count"] > 0:
        models.append((orch["path"], "orchestrator-agent", orch))
    return models


# Will be populated at runtime from the bucket manifest
MODELS: list[tuple] = []


def has_completed_checkpoint(output_path: Path) -> bool:
    """True if the output dir looks like a completed run.

    v0.1.6: Now also checks state.json[completed]=true. Previously we
    just checked for the existence of adapter_config.json, but that
    exists in partial checkpoints too (HF writes it on every save).
    A truly complete run has the state.json sidecar marking it done.
    """
    if (output_path / "adapter_config.json").exists():
        # Newer: also verify state.json says completed (v0.1.6+)
        sp = output_path / "state.json"
        if sp.exists():
            try:
                with sp.open() as f:
                    s = json.load(f)
                if s.get("completed"):
                    return True
            except (OSError, json.JSONDecodeError):
                pass
        # Fallback: any checkpoint-N/ subdirectory with an adapter inside
        # counts as a successful run (load_best_model_at_end picked the
        # best one). Preserves backward compat with runs from before
        # state.json existed.
        if output_path.exists():
            for child in output_path.iterdir():
                if child.is_dir() and child.name.startswith("checkpoint-"):
                    if (child / "adapter_config.json").exists():
                        return True
    return False


def _make_timestamped_output_dir(agent_name: str) -> Path:
    """Build a timestamped output dir for a new training run.

    v0.1.6: each run gets its own timestamped subdir under MODELS_DIR
    instead of clobbering a single un-suffixed dir. Format:
        models/agent-name_2026-06-10_01-12/
    Collision: append _2, _3, ... for runs started in the same minute.

    The agent_name is used as-is (e.g. "attacklm-single", "execution-agent",
    "orchestrator-agent"). We do NOT sanitize the name because all current
    agent names are already valid in a path. If new agent names introduce
    spaces or path separators, add a slugify step here.
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    base = MODELS_DIR / f"{agent_name}_{ts}"
    if not base.exists():
        return base
    for i in range(2, 100):
        candidate = MODELS_DIR / f"{agent_name}_{ts}_{i}"
        if not candidate.exists():
            return candidate
    return base  # 100 collisions/min is implausible


def _find_latest_run_dir(agent_name: str) -> Path | None:
    """Find the most recent timestamped run dir for an agent, if any.

    Returns the dir with the largest lexicographic name (timestamps
    sort correctly as YYYY-MM-DD_HH-MM strings). Returns None if no
    timestamped run dirs exist for this agent — meaning either the
    agent has never been trained, or only the legacy un-suffixed
    models/agent-name/ dir exists.
    """
    if not MODELS_DIR.exists():
        return None
    candidates = sorted(
        [
            p
            for p in MODELS_DIR.iterdir()
            if p.is_dir()
            and p.name.startswith(f"{agent_name}_")
            and len(p.name.split("_")) >= 3  # at least one underscore after name
        ],
        key=lambda p: p.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def build_train_cmd(
    args,
    dataset_path: Path,
    output_path: Path,
    lora_dropout: float = None,
    max_steps: int = None,
    hpo_metrics_csv: str = None,
) -> list:
    """Build the subprocess cmd for train_template.py.

    Always passes the common flags. Adds --resume-from-checkpoint if the
    caller asked for it AND there's a checkpoint in the output dir to
    resume from. Accepts an optional lora_dropout override (used by
    curriculum stage 2 to reset dropout to 0 for fine-tuning), an
    optional max_steps override (HPO short trials), and an optional
    hpo_metrics_csv (per-step CSV for HPO analysis).
    """
    dropout = args.lora_dropout if lora_dropout is None else lora_dropout
    cmd = [
        sys.executable,
        "-u",
        str(BASE_DIR / "scripts" / "train_template.py"),
        "--train",
        "--base-model",
        args.base_model,
        "--dataset",
        str(dataset_path),
        "--output",
        str(output_path),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--max-length",
        str(args.max_length),
        "--lora-r",
        str(args.lora_r),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(dropout),
        "--optim",
        str(args.optim),
        "--save-steps",
        str(args.save_steps),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
    ]
    # Packing is the default; pass --no-packing only if user disabled it
    if not args.packing:
        cmd.append("--no-packing")
    # HPO trial budget cap: override --epochs with --max-steps
    if max_steps is not None and max_steps > 0:
        cmd.extend(["--max-steps", str(max_steps)])
    # HPO per-step CSV log
    if hpo_metrics_csv:
        cmd.extend(["--hpo-metrics-csv", str(hpo_metrics_csv)])
    # Resume: only if user asked AND there's actually a checkpoint to resume
    if args.resume_from_checkpoint and output_path.exists():
        if any(
            c.is_dir() and c.name.startswith("checkpoint-")
            for c in output_path.iterdir()
        ):
            cmd.append("--resume-from-checkpoint")
    return cmd


def _run_with_retry(
    args: argparse.Namespace,
    cmd: list[str],
    output_path: Path,
    label: str,
    out_fn,
    log_fh,
) -> int:
    """Run a training subprocess with auto-retry on failure.

    If the process exits non-zero and checkpoints exist in output_path,
    re-run with --resume-from-checkpoint appended. Retries up to
    args.max_retries times. Returns final returncode.
    """
    import copy

    for attempt in range(1, args.max_retries + 1):
        if attempt > 1:
            # Check if we have checkpoints to resume from
            if not has_completed_checkpoint(output_path):
                # No checkpoint found — check if there's anything at all
                checkpoint_dirs = (
                    [
                        c.name
                        for c in output_path.iterdir()
                        if c.is_dir() and c.name.startswith("checkpoint-")
                    ]
                    if output_path.exists()
                    else []
                )
                if not checkpoint_dirs:
                    out_fn(
                        f"  RETRY {attempt}/{args.max_retries}: no checkpoint "
                        f"found to resume — aborting"
                    )
                    break
                # Even partial checkpoint-N/ dirs (no full adapter) can resume
                out_fn(
                    f"  RETRY {attempt}/{args.max_retries}: found partial "
                    f"checkpoints, attempting resume from {checkpoint_dirs[-1]}"
                )
            else:
                out_fn(
                    f"  RETRY {attempt}/{args.max_retries}: resuming from "
                    f"last checkpoint"
                )

            # Build a fresh cmd with --resume-from-checkpoint appended
            retry_cmd = copy.copy(cmd)
            if "--resume-from-checkpoint" not in retry_cmd:
                retry_cmd.append("--resume-from-checkpoint")
            cmd_to_run = retry_cmd

            out_fn(f"  Command: {Path(cmd_to_run[3]).name} {' '.join(cmd_to_run[4:])}")
        else:
            cmd_to_run = cmd

        out_fn(f"  [{label}] Attempt {attempt}/{args.max_retries}")

        try:
            result = subprocess.run(
                cmd_to_run,
                cwd=str(BASE_DIR),
                timeout=args.timeout,
            )
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            out_fn(f"  [{label}] TIMEOUT — killed after {args.timeout}s")
            returncode = -9
        except KeyboardInterrupt:
            out_fn(f"  [{label}] INTERRUPTED by user")
            log_fh.close()
            sys.exit(130)

        if returncode == 0 and has_completed_checkpoint(output_path):
            out_fn(f"  [{label}] OK — attempt {attempt} succeeded")
            return 0

        out_fn(
            f"  [{label}] FAILED — exit={returncode}, "
            f"{'will retry' if attempt < args.max_retries else 'no retries left'}"
        )

    return returncode


def main():
    parser = argparse.ArgumentParser(description="Train all AttackLM agents")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip agents with completed checkpoints",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Per-agent timeout in seconds (default: 2 hours)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Auto-retry on failure, resuming from last checkpoint (default: 3)",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Start from agent index N (0-based, for resuming)",
    )
    parser.add_argument(
        "--single-model",
        action="store_true",
        help="Train one model on all tactic datasets combined (skip MoE mode)",
    )
    parser.add_argument(
        "--single-model-name",
        type=str,
        default="attacklm-single",
        help="Output dir name when --single-model is set (default: attacklm-single)",
    )
    parser.add_argument(
        "--include-orchestrator",
        action="store_true",
        help="When --single-model is set, also include the orchestrator routing data",
    )
    parser.add_argument(
        "--model-attacks",
        action="store_true",
        help="When --single-model is set, also include the ai-models/* buckets "
        "(prompt-injection + jailbreaking) for AI/ML attack data",
    )
    parser.add_argument(
        "--include-tools",
        action="store_true",
        help="When --single-model is set, also include the tools/* buckets "
        "(metasploit, infection_monkey, rta)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank passed through to train_template.py (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha passed through to train_template.py (default: 32)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout passed through to train_template.py (default: 0.05)",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="paged_adamw_8bit",
        help="Optimizer passed to SFTConfig (default: paged_adamw_8bit — saves ~60MB vs adamw_torch, helps fit 16GB cards)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="Resume from existing checkpoint-N/ in the output dir (continues training)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=200,
        help="Save checkpoint every N steps — passed to train_template.py (default: 200)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps — passed to train_template.py (default: 1). "
        "Use 4 to reduce VRAM on small GPUs.",
    )
    parser.add_argument(
        "--packing",
        dest="packing",
        action="store_true",
        default=False,
        help="Enable example packing + padding-free training (default: OFF, "
        "because flash_attention_2 is hard to install in many envs). "
        "Set --packing to enable; requires flash_attn installed. "
        "~30-40%% faster + 2x less VRAM padding when enabled.",
    )
    parser.add_argument(
        "--no-packing",
        dest="packing",
        action="store_false",
        help="Disable example packing (default). Each example padded to max_length "
        "individually. Slower but doesn't require flash-attn. Always works.",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Two-stage training: tactic data first, then orchestrator routing data",
    )
    # ---- HPO mode ----
    parser.add_argument(
        "--hpo",
        action="store_true",
        help="Run coordinate-descent HPO sweeps instead of normal training. "
        "Trains short trials, then escalates each hyperparameter one at a time, "
        "backing off when metrics degrade. Saves the best config and runs a final "
        "full training with all winners. Implies --single-model.",
    )
    parser.add_argument(
        "--hpo-trials-per-axis",
        type=int,
        default=4,
        help="Max number of escalating values to try per hyperparameter "
        "(default: 4 → 4 trials, each budget=--hpo-trial-steps).",
    )
    parser.add_argument(
        "--hpo-trial-steps",
        type=int,
        default=200,
        help="Optimizer steps per HPO trial (default: 200 ≈ 4 min on 3B QLoRA). "
        "100 is too short to see real loss behavior; 200 gives verdicts you can trust.",
    )
    parser.add_argument(
        "--hpo-output-dir",
        type=str,
        default="hpo_runs",
        help="Directory for HPO trial CSVs and config snapshots (default: hpo_runs/).",
    )
    parser.add_argument(
        "--hpo-dataset",
        type=str,
        default=None,
        help="Dataset for HPO trials. Default: same combined dataset "
        "(--include-tools --include-orchestrator --model-attacks) used at full "
        "training time, but capped to 5000 examples for speed.",
    )
    args = parser.parse_args()

    # HPO mode implies single-model (we tune one combined dataset)
    if args.hpo:
        args.single_model = True

    # Populate the MODELS list from the bucket manifest (after parse so
    # we can use --start-from even though it's not bucket-aware yet)
    global MODELS
    MODELS = get_default_models()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"train_all_{timestamp}.log"
    log_fh = open(log_path, "a", encoding="utf-8")

    def out(s=""):
        print(s, flush=True)
        if s:
            log_fh.write(s + "\n")
            log_fh.flush()

    out("=" * 60)
    out(" AttackLM — Train All Agents")
    out("=" * 60)
    out(f"  Start time:    {datetime.now().isoformat()}")
    out(f"  Base model:    {args.base_model}")
    out(f"  Epochs:        {args.epochs}")
    out(f"  Batch size:    {args.batch_size}")
    out(f"  Max length:    {args.max_length}")
    out(
        f"  LoRA:          r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}"
    )
    out(f"  Optimizer:     {args.optim}")
    out(f"  Save steps:    {args.save_steps}")
    out(f"  Gradient accum:{args.gradient_accumulation_steps}")
    out(f"  Packing:       {args.packing}  (--packing/--no-packing)")
    out(f"  Timeout:       {args.timeout}s per agent")
    out(f"  Max retries:    {args.max_retries}")
    out(f"  Skip completed:{args.skip_completed}")
    out(f"  Resume:        {args.resume_from_checkpoint}")
    out(f"  Curriculum:    {args.curriculum}")
    out(f"  Log file:      {log_path}")
    out(f"  Models:        {len(MODELS)} total")
    out("")

    total_trained = 0
    total_skipped = 0
    total_failed = 0
    failed_agents = []
    skipped_agents = []
    start_time = time.time()

    # ==================================================================
    # SINGLE-MODE: combine buckets, train one model, exit.
    # ==================================================================
    if args.single_model:
        out("=" * 60)
        out(" AttackLM — Single Model Mode")
        out("=" * 60)
        out(f"  Mode:           single (combines selected buckets)")
        out(f"  Output:         {MODELS_DIR / args.single_model_name}")
        out(f"  Include orch:   {args.include_orchestrator}")
        out(f"  Model attacks:  {args.model_attacks}")
        out(f"  Include tools:  {args.include_tools}")
        out(f"  Curriculum:     {args.curriculum}")
        out("")

        # Determine which buckets to combine
        # Default: all 10 MITRE tactic buckets
        # Plus optionally orchestrator, ai-models/*, and/or tools/*
        bucket_names = [
            b["path"]
            for b in get_default_train_buckets()
            if b.get("category") == "tactic"
        ]
        flags_used = {
            "include_orchestrator": args.include_orchestrator,
            "model_attacks": args.model_attacks,
            "include_tools": args.include_tools,
            "curriculum": args.curriculum,
        }
        if args.include_orchestrator:
            bucket_names.append("orchestrator")
        if args.model_attacks:
            ai_names = [b["path"] for b in get_ai_model_buckets()]
            bucket_names.extend(ai_names)
        if args.include_tools:
            tool_names = [b["path"] for b in get_tool_buckets()]
            bucket_names.extend(tool_names)

        out(f"  Buckets ({len(bucket_names)}): {', '.join(bucket_names)}")
        out("")

        # Build (or reuse from cache) the combined dataset
        tactic_combined_path = build_combined(
            bucket_names=bucket_names,
            flags=flags_used,
            seed=42,
            shuffle=True,
        )
        total_tactic = sum(1 for _ in open(tactic_combined_path))
        out("")
        out(f"  Combined dataset: {tactic_combined_path} ({total_tactic:,} examples)")
        out("")

        output_path = _make_timestamped_output_dir(args.single_model_name)
        out(f"  Run output:   {output_path}")
        # If a previous run for this agent exists and has a completed
        # state.json, the user is doing round-2 SFT — load the previous
        # run dir as the base. We pick the latest by lexicographic sort
        # (timestamps sort correctly as YYYY-MM-DD_HH-MM strings).
        # (Does NOT apply when --base-model is set explicitly.)
        if not args.base_model and not args.round_two_base:
            latest = _find_latest_run_dir(args.single_model_name)
            if latest and (latest / "state.json").exists():
                try:
                    with (latest / "state.json").open() as _f:
                        _st = json.load(_f)
                    if _st.get("completed"):
                        out(
                            f"  ↻ Round-2 SFT detected: previous completed run at {latest.name}"
                        )
                        out(f"    Loading merged weights as base for this run.")
                        args.base_model = str(latest)
                except (OSError, json.JSONDecodeError):
                    pass
        if args.skip_completed and has_completed_checkpoint(output_path):
            out(f"  SKIP — checkpoint already exists at {output_path}")
            log_fh.close()
            return 0

        if args.dry_run:
            if args.hpo:
                # HPO dry-run: show the sweep plan without running
                from hpo_runner import HPO_AXES

                out("  [DRY RUN] HPO sweep plan:")
                for axis in HPO_AXES:
                    out(
                        f"    axis: {axis.name} (default_low={axis.default_low}, max={axis.max_value})"
                    )
                    v = axis.default_low
                    for trial in range(args.hpo_trials_per_axis):
                        v = axis.clip(v)
                        out(
                            f"      trial {trial + 1}: {axis.name}={v:g} "
                            f"({args.hpo_trial_steps} steps)"
                        )
                        v = axis.next_value(v)
                        if v > axis.max_value:
                            break
                out(
                    f"  [DRY RUN] Then: full training with HPO winners on "
                    f"{total_tactic:,} examples from {len(bucket_names)} buckets"
                )
            elif args.curriculum:
                out(
                    f"  [DRY RUN] STAGE 1: train on {total_tactic} tactic examples "
                    f"(dropout={args.lora_dropout})"
                )
                out(
                    "  [DRY RUN] STAGE 2: resume from stage 1, train on "
                    "orchestrator routing data (dropout=0.0)"
                )
            else:
                extras = []
                if args.include_orchestrator:
                    extras.append("orchestrator")
                if args.model_attacks:
                    extras.append("ai-models/*")
                if args.include_tools:
                    extras.append("tools/*")
                extra_str = f" + {' + '.join(extras)}" if extras else ""
                out(
                    f"  [DRY RUN] would train single model on "
                    f"{total_tactic:,} examples from {len(bucket_names)} buckets{extra_str}"
                )
            log_fh.close()
            return 0

        # ============================
        # CURRICULUM MODE: 2 stages
        # ============================
        # Stage 1: train on all non-orchestrator buckets (tactic + tooling + ai_redteam)
        # Stage 2: fine-tune on orchestrator only
        if args.curriculum:
            orch_bucket = get_bucket("orchestrator")
            if not orch_bucket or orch_bucket["count"] == 0:
                out("ERROR: --curriculum requested but orchestrator bucket is empty")
                log_fh.close()
                return 1

            out("=" * 60)
            n_s1 = total_tactic  # build_combined above already excludes orchestrator
            out(f" STAGE 1/2 — Train on tactic+tooling+ai_redteam ({n_s1:,} examples)")
            out("=" * 60)
            cmd_s1 = build_train_cmd(args, tactic_combined_path, output_path)
            out(f"  Command: {Path(cmd_s1[3]).name} {' '.join(cmd_s1[4:])}")
            out(f"  Timeout: {args.timeout}s")
            out("")
            rc1 = _run_with_retry(args, cmd_s1, output_path, "stage1", out, log_fh)
            if rc1 != 0:
                out(f"  STAGE 1 FAILED — exit={rc1}. Skipping stage 2.")
                log_fh.close()
                return 1
            out(f"  STAGE 1 OK — adapter at {output_path}")
            out("")

            out("=" * 60)
            out(
                f" STAGE 2/2 — Fine-tune on orchestrator routing data "
                f"({orch_bucket['count']} examples, resume from stage 1)"
            )
            out("=" * 60)
            # Override: dropout=0 for fine-tuning, force resume, half epochs
            args_stage2 = argparse.Namespace(**vars(args))
            args_stage2.lora_dropout = 0.0
            args_stage2.epochs = max(2, args.epochs // 2)
            args_stage2.resume_from_checkpoint = True
            orch_path = BUCKETS_DIR / "orchestrator" / "data.jsonl"
            cmd_s2 = build_train_cmd(
                args_stage2, orch_path, output_path, lora_dropout=0.0
            )
            out(f"  Command: {Path(cmd_s2[3]).name} {' '.join(cmd_s2[4:])}")
            out(f"  Epochs (stage 2): {args_stage2.epochs} (half of {args.epochs})")
            out(f"  Dropout (stage 2): 0.0 (fine-tune, not learn-from-scratch)")
            out(f"  Timeout: {args.timeout}s")
            out("")
            rc2 = _run_with_retry(
                args_stage2, cmd_s2, output_path, "stage2", out, log_fh
            )
            elapsed_total = time.time() - start_time
            if rc2 != 0:
                out(
                    f"  STAGE 2 FAILED — exit={rc2}, elapsed={elapsed_total / 60:.1f} min"
                )
                out(f"  Stage 1 adapter is still in place at {output_path}")
                log_fh.close()
                return 1
            out(f"  STAGE 2 OK — curriculum complete in {elapsed_total / 60:.1f} min")
            out(f"  Final adapter: {output_path}")
            out("")
            out(f"  Tactic dataset (kept for reuse): {tactic_combined_path}")
            out("=" * 60)
            log_fh.close()
            return 0

        # ============================
        # REGULAR SINGLE-MODE: one run on the combined dataset
        # ============================
        # The combined dataset (tactic_combined_path) was already built above
        # with the right buckets per --include-orchestrator / --model-attacks.
        combined_path = tactic_combined_path

        # ============================
        # HPO MODE: coordinate-descent sweep
        # ============================
        # Skip the regular training run; instead, run the HPO loop and
        # then a final training with the winning hyperparameter set.
        if args.hpo:
            from hpo_runner import run_hpo_sweep, run_final_training

            hpo_dir = BASE_DIR / args.hpo_output_dir
            hpo_dir.mkdir(parents=True, exist_ok=True)

            # Use a smaller dataset for HPO trials (cap to 5000 examples)
            # to keep trial time bounded. The full dataset is used for the
            # final training run.
            hpo_dataset = args.hpo_dataset
            if not hpo_dataset:
                hpo_dataset_path = hpo_dir / "hpo_trial_dataset.jsonl"
                # Take the first 5000 lines of the combined dataset
                with open(combined_path) as src, open(hpo_dataset_path, "w") as dst:
                    for i, line in enumerate(src):
                        if i >= 5000:
                            break
                        dst.write(line)
                hpo_dataset = str(hpo_dataset_path)
                n_hpo = sum(1 for _ in open(hpo_dataset))
                out(f"  HPO dataset (capped): {hpo_dataset} ({n_hpo} examples)")
            else:
                out(f"  HPO dataset (user): {hpo_dataset}")

            # Wrapper that captures the args namespace so we can update
            # the HP value per trial.
            def hpo_cmd_fn(a, ds, out_path, max_steps=None, hpo_metrics_csv=None):
                return build_train_cmd(
                    a,
                    ds,
                    out_path,
                    max_steps=max_steps,
                    hpo_metrics_csv=hpo_metrics_csv,
                )

            best = run_hpo_sweep(
                args,
                out,
                log_fh,
                dataset_path=hpo_dataset,
                hpo_dir=hpo_dir,
                base_cmd_fn=hpo_cmd_fn,
            )

            out("")
            out("=" * 70)
            out(" HPO sweep finished. Running final training with winners…")
            out("=" * 70)
            # The final training uses the FULL combined dataset.
            args.output = str(output_path)
            rc = run_final_training(
                args,
                out,
                log_fh,
                dataset_path=combined_path,
                best_per_axis=best,
                base_cmd_fn=hpo_cmd_fn,
            )
            log_fh.close()
            return rc

        cmd = build_train_cmd(args, combined_path, output_path)
        out(f"  Command: {Path(cmd[3]).name} {' '.join(cmd[4:])}")
        out(f"  Timeout: {args.timeout}s")
        out("")

        returncode = _run_with_retry(
            args, cmd, output_path, "single-model", out, log_fh
        )
        elapsed_total = time.time() - start_time
        if returncode == 0:
            out(f"  OK — single model complete in {elapsed_total / 60:.1f} min")
            out(f"  Adapter: {output_path}")
        else:
            out(f"  FAILED — exit={returncode}, elapsed={elapsed_total / 60:.1f} min")
            log_fh.close()
            return 1

        out("")
        out(f"  Combined dataset (kept for reuse): {combined_path}")
        out("=" * 60)
        log_fh.close()
        return 0

    for idx, (bucket_name, agent_name, bucket_meta) in enumerate(MODELS):
        if idx < args.start_from:
            continue

        # Data path lives in the bucket directory
        dataset_path = BUCKETS_DIR / bucket_name / "data.jsonl"
        # v0.1.6: per-run timestamped output dir (no clobbering, easy rollback)
        output_path = _make_timestamped_output_dir(agent_name)

        out(f"[{idx + 1}/{len(MODELS)}] {agent_name}")
        out(f"  Run output:  {output_path}")

        if not dataset_path.exists():
            out(f"  SKIP — bucket {bucket_name} has no data.jsonl")
            total_skipped += 1
            skipped_agents.append(agent_name)
            out("")
            continue

        if args.skip_completed and has_completed_checkpoint(output_path):
            out(f"  SKIP — checkpoint already exists at {output_path}")
            total_skipped += 1
            skipped_agents.append(agent_name)
            out("")
            continue

        n_examples = sum(1 for _ in open(dataset_path))
        if n_examples == 0:
            out(f"  SKIP — bucket is empty")
            total_skipped += 1
            skipped_agents.append(agent_name)
            out("")
            continue

        out(
            f"  Bucket:   {bucket_name} ({bucket_meta.get('category', '?')}, {bucket_meta.get('mitre_tactic', '?')})"
        )
        out(f"  Examples: {n_examples}")
        out(f"  Output:   {output_path}")

        if args.dry_run:
            out(f"  [DRY RUN] would train {agent_name}")
            total_trained += 1
            out("")
            continue

        cmd = build_train_cmd(args, dataset_path, output_path)

        out(f"  Command: {Path(cmd[3]).name} {' '.join(cmd[4:])}")
        out(f"  Timeout: {args.timeout}s")
        out("")

        agent_start = time.time()
        returncode = _run_with_retry(args, cmd, output_path, agent_name, out, log_fh)
        elapsed_agent = time.time() - agent_start

        if returncode == 0:
            out(f"  OK — {agent_name} complete in {elapsed_agent / 60:.1f} min")
            total_trained += 1
        else:
            out(f"  FAILED — exit={returncode}, elapsed={elapsed_agent / 60:.1f} min")
            total_failed += 1
            failed_agents.append(agent_name)
        out("")

    elapsed = time.time() - start_time
    out("=" * 60)
    out(" Training Complete")
    out("=" * 60)
    out(f"  End time:    {datetime.now().isoformat()}")
    out(f"  Duration:    {elapsed / 60:.1f} minutes")
    out(f"  Trained:     {total_trained}")
    if skipped_agents:
        out(f"  Skipped:     {total_skipped} ({', '.join(skipped_agents)})")
    if failed_agents:
        out(f"  FAILED:      {total_failed} ({', '.join(failed_agents)})")
    out(f"  Models dir:  {MODELS_DIR}/")
    out(f"  Main log:    {log_path}")
    out("=" * 60)

    summary_path = LOGS_DIR / f"train_all_{timestamp}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "started": datetime.fromtimestamp(start_time).isoformat(),
                "duration_seconds": round(elapsed, 2),
                "base_model": args.base_model,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "trained": total_trained,
                "skipped": total_skipped,
                "failed": total_failed,
                "failed_agents": failed_agents,
                "skipped_agents": skipped_agents,
            },
            f,
            indent=2,
        )
    out(f"  Summary:     {summary_path}")

    log_fh.close()
    return 1 if failed_agents else 0


if __name__ == "__main__":
    sys.exit(main())
