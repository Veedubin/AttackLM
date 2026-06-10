#!/usr/bin/env python3
"""Migrate AttackLM v0.1.5 training run dirs to the v0.2.0 timestamped format.

v0.1.5 stored runs at:
    models/attacklm-single/                <- the un-suffixed run dir
        adapter_config.json
        adapter_model.safetensors
        checkpoint-N/...
        README.md
        config.json
        tokenizer.json
        tokenizer_config.json
        chat_template.jinja
        # NO state.json (didn't exist in v0.1.5)

v0.2.0 expects:
    models/attacklm-single_YYYY-MM-DD_HH-MM/  <- timestamped run dir
        state.json                              <- backfilled from checkpoint
        # all the same files as v0.1.5

This script:
  1. Renames `models/{agent}/` to `models/{agent}_{TIMESTAMP}/`
     using the date from the latest checkpoint-N/ mtime.
  2. Backfills `state.json` from `checkpoint-N/trainer_state.json`.
  3. Records base_model from `adapter_config.json[base_model_name_or_path]`.

Run:
    python scripts/migrate_v015_to_v020.py                  # migrate all
    python scripts/migrate_v015_to_v020.py attacklm-single # migrate one

Safe to re-run — checks for an existing state.json and skips.
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


def _find_latest_checkpoint(run_dir: Path) -> Path | None:
    """Find the highest-numbered checkpoint-N/ subdir."""
    if not run_dir.exists():
        return None
    cks = sorted(
        [
            d
            for d in run_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ],
        key=lambda d: int(d.name.split("-")[-1]),
    )
    return cks[-1] if cks else None


def _state_for_run(run_dir: Path) -> dict | None:
    """Build a v0.2.0 state.json from a v0.1.5 run dir.

    Returns None if the run dir doesn't have the files we need
    (adapter_config.json + at least one checkpoint-N/trainer_state.json).
    """
    ac_path = run_dir / "adapter_config.json"
    if not ac_path.exists():
        return None
    try:
        ac = json.load(ac_path.open())
    except (OSError, json.JSONDecodeError):
        return None

    latest_ckpt = _find_latest_checkpoint(run_dir)
    if latest_ckpt is None:
        return None
    ts_path = latest_ckpt / "trainer_state.json"
    if not ts_path.exists():
        return None
    try:
        ts = json.load(ts_path.open())
    except (OSError, json.JSONDecodeError):
        return None

    log = ts.get("log_history", [])
    last = log[-1] if log else {}

    # Use the adapter_config.json mtime as a rough training-start timestamp,
    # the latest checkpoint mtime as end. If the run finished cleanly, this
    # is within seconds of when the trainer stopped.
    ckpt_mtime = datetime.fromtimestamp(latest_ckpt.stat().st_mtime, tz=timezone.utc)
    ac_mtime = datetime.fromtimestamp(ac_path.stat().st_mtime, tz=timezone.utc)
    ts_str_start = ac_mtime.strftime("%Y-%m-%dT%H:%M:%SZ")
    ts_str_end = ckpt_mtime.strftime("%Y-%m-%dT%H:%M:%SZ")
    ts_dir = ac_mtime.strftime("%Y-%m-%d_%H-%M")

    hparams = {
        "lora_r": ac.get("r", 16),
        "lora_alpha": ac.get("lora_alpha", 32),
        "lora_dropout": ac.get("lora_dropout", 0.05),
        "epochs": int(ts.get("epoch", 0)),
        "batch_size": 1,
        "max_length": 2048,
        "gradient_accumulation_steps": 1,
        "save_steps": 200,
        "eval_steps": None,
        "learning_rate": None,
        "warmup_ratio": None,
        "seed": 42,
        "optim": "paged_adamw_8bit",
        "packing": False,
    }
    return {
        "version": 1,
        "created_at": ts_str_start,
        "updated_at": ts_str_end,
        "completed": ts.get("global_step", 0) >= ts.get("max_steps", 1)
        if ts.get("max_steps", 0) > 0
        else True,
        "base_model": {
            "source": "local"
            if ac.get("base_model_name_or_path", "").startswith(("/", "./", "~/"))
            else "hf",
            "id": ac.get("base_model_name_or_path"),
        },
        "hparams": hparams,
        "dataset": {
            "source": "unknown",
            "path": None,
            "specs": None,  # not recorded in v0.1.5
            "buckets": None,
            "include_tools": None,
            "include_ai": None,
            "examples_total": None,
            "examples_train": None,
            "examples_eval": None,
        },
        "progress": {
            "global_step": int(ts.get("global_step", 0)),
            "max_steps": int(ts.get("max_steps", 0)),
            "current_epoch": float(ts.get("epoch", 0)),
            "total_epochs": float(hparams["epochs"]),
            "last_loss": last.get("loss"),
            "last_token_accuracy": last.get("mean_token_accuracy"),
            "best_eval_loss": ts.get("best_metric"),
            "total_training_seconds": None,
        },
        "hpo": {
            "is_hpo_trial": False,
            "trial_id": None,
            "parent_run": None,
            "axes": None,
        },
        "stopped_early": False,
        "_ts_dir_hint": ts_dir,  # internal hint for renaming
    }


def migrate_run_dir(agent_name: str) -> bool:
    """Migrate models/{agent}/ → models/{agent}_{TIMESTAMP}/ with state.json.

    Returns True if migration was performed, False if skipped.
    """
    src = MODELS_DIR / agent_name
    if not src.exists() or not src.is_dir():
        print(f"  SKIP {agent_name}: not found at {src}")
        return False

    # Already timestamped? Check.
    if re.search(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}(_\d+)?$", src.name):
        print(f"  SKIP {agent_name}: already timestamped ({src.name})")
        return False

    # Already has state.json? Skip (already migrated or v0.2.0).
    if (src / "state.json").exists():
        print(f"  SKIP {agent_name}: state.json already present")
        return False

    # Build the new state.json content
    state = _state_for_run(src)
    if state is None:
        print(
            f"  SKIP {agent_name}: no adapter_config.json + checkpoint-N/trainer_state.json"
        )
        return False

    ts_dir = state.pop("_ts_dir_hint")
    dst = MODELS_DIR / f"{agent_name}_{ts_dir}"

    # Collision? Append a counter.
    n = 1
    while dst.exists():
        n += 1
        dst = MODELS_DIR / f"{agent_name}_{ts_dir}_{n}"

    # Perform the move
    import shutil

    shutil.move(str(src), str(dst))
    print(f"  RENAME  {src.name}/ → {dst.name}/")

    # Write state.json
    state_path = dst / "state.json"
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    print(f"  STATE   {dst.name}/state.json written (completed={state['completed']})")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate v0.1.5 un-suffixed training run dirs to v0.2.0 timestamped format",
    )
    parser.add_argument(
        "agents",
        nargs="*",
        help="Specific agent names to migrate (default: all un-suffixed dirs in models/)",
    )
    args = parser.parse_args()

    if not MODELS_DIR.exists():
        print(f"ERROR: {MODELS_DIR} not found")
        return 1

    if args.agents:
        targets = args.agents
    else:
        # Find all un-suffixed dirs that look like training runs
        # (have adapter_config.json at root)
        targets = sorted(
            [
                p.name
                for p in MODELS_DIR.iterdir()
                if p.is_dir()
                and not re.search(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}(_\d+)?$", p.name)
                and (p / "adapter_config.json").exists()
            ]
        )

    if not targets:
        print("No un-suffixed run dirs found. Nothing to migrate.")
        return 0

    print(f"Migrating {len(targets)} run dir(s):")
    for name in targets:
        migrate_run_dir(name)
    print()
    print("Done. Verify with: ls models/")
    print(
        "Re-run round-2 SFT with: attacklm-train-all --single-model --dataset base/ ..."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
