#!/usr/bin/env python3
"""AttackLM Training Pipeline — IaC for model training.

Reads a YAML config file defining one or more training jobs.
Each job runs sequentially: train → merge → gguf → install.

Usage:
    attacklm pipeline --config pipeline.yaml
    python scripts/pipeline.py --config pipeline.yaml
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with: uv pip install pyyaml")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="AttackLM Training Pipeline — IaC for model training"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML pipeline config file",
    )
    return parser.parse_args()


def run_train(job_name: str, train_cfg: dict) -> bool:
    """Run attacklm train with the given config. Returns True on success."""
    script = str(Path(__file__).parent / "train_template.py")
    cmd = [sys.executable, script, "--train"]

    # Map YAML keys to CLI args. Boolean flags use --flag, others use --key value.
    BOOLEAN_FLAGS = {
        "train",
        "use_galore",
        "galore_32bit",
        "use_unsloth",
        "use_dora",
        "packing",
        "bf16",
        "fp16",
        "fp32",
        "use_rslora",
        "moe_safe_target",
        "loftq_init",
        "resume_from_checkpoint",
        "no_timestamp",
        "force",
        "multi_gpu",
        "dry_run",
    }

    for key, value in train_cfg.items():
        if key in BOOLEAN_FLAGS:
            if value:
                cmd.append(f"--{key.replace('_', '-')}")
        elif key == "no_packing" and value:
            cmd.append("--no-packing")
        elif key == "no_use_rslora" and value:
            cmd.append("--no-use-rslora")
        else:
            cmd.append(f"--{key.replace('_', '-')}")
            cmd.append(str(value))

    print(f"\n{'=' * 60}")
    print(f"  Job: {job_name}")
    print(f"  Stage: TRAIN")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd)
    return result.returncode == 0


def run_merge(job_name: str, merge_cfg: dict, train_cfg: dict) -> bool:
    """Run attacklm build --merge-only to combine LoRA adapter with base model."""
    script = str(Path(__file__).parent / "merge.py")

    # Determine input: merge.output or train.output
    input_dir = merge_cfg.get("output") or train_cfg.get("output")
    if not input_dir:
        print(f"  ERROR: No output directory specified for merge stage")
        return False

    cmd = [
        sys.executable,
        script,
        "--input",
        input_dir,
    ]

    if "output" in merge_cfg:
        cmd.extend(["--output", merge_cfg["output"]])

    print(f"\n{'=' * 60}")
    print(f"  Job: {job_name}")
    print(f"  Stage: MERGE")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd)
    return result.returncode == 0


def run_gguf(job_name: str, gguf_cfg: dict, merge_cfg: dict, train_cfg: dict) -> bool:
    """Run attacklm build --gguf-only to convert merged model to GGUF format."""
    script = str(Path(__file__).parent / "convert_to_gguf.py")

    # Determine input: gguf.output or merge.output or train.output
    input_dir = (
        gguf_cfg.get("output") or merge_cfg.get("output") or train_cfg.get("output")
    )
    if not input_dir:
        print(f"  ERROR: No output directory specified for gguf stage")
        return False

    cmd = [sys.executable, script, "--input", input_dir]

    if "quant" in gguf_cfg:
        cmd.extend(["--quant", gguf_cfg["quant"]])
    if "output" in gguf_cfg:
        cmd.extend(["--output", gguf_cfg["output"]])

    print(f"\n{'=' * 60}")
    print(f"  Job: {job_name}")
    print(f"  Stage: GGUF")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd)
    return result.returncode == 0


def run_install(
    job_name: str, install_cfg: dict, gguf_cfg: dict, merge_cfg: dict, train_cfg: dict
) -> bool:
    """Install the model to LM Studio or other targets."""
    if install_cfg.get("lmstudio"):
        # Determine the GGUF file path
        gguf_output = (
            gguf_cfg.get("output") or merge_cfg.get("output") or train_cfg.get("output")
        )
        if gguf_output and not gguf_output.endswith(".gguf"):
            gguf_output = gguf_output + ".gguf"

        if not gguf_output or not Path(gguf_output).exists():
            print(
                f"  WARNING: GGUF file not found at {gguf_output}, skipping LM Studio install"
            )
            return True  # not a failure, just skip

        # LM Studio models directory
        lmstudio_dir = Path.home() / ".cache" / "lm-studio" / "models"
        if not lmstudio_dir.exists():
            print(f"  WARNING: LM Studio models dir not found at {lmstudio_dir}")
            return True

        import shutil

        dest = lmstudio_dir / Path(gguf_output).name
        print(f"\n  Installing to LM Studio: {dest}")
        shutil.copy2(gguf_output, dest)
        print(f"  Installed: {dest}")

    return True


def run_pipeline(config_path: str) -> bool:
    """Run all jobs in the pipeline config. Returns True if all succeed."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    jobs = config.get("jobs", [])
    if not jobs:
        print("ERROR: No jobs defined in pipeline config")
        return False

    print(f"\n  Pipeline: {len(jobs)} job(s) defined")

    all_success = True
    total_start = time.time()

    for i, job in enumerate(jobs):
        job_name = job.get("name", f"job-{i + 1}")
        train_cfg = job.get("train", {})
        merge_cfg = job.get("merge", {})
        gguf_cfg = job.get("gguf", {})
        install_cfg = job.get("install", {})

        if not train_cfg:
            print(f"  WARNING: Job '{job_name}' has no train config, skipping")
            continue

        job_start = time.time()

        # Stage 1: Train
        if not run_train(job_name, train_cfg):
            print(f"\n  FAILED: Job '{job_name}' train stage failed")
            all_success = False
            continue

        # Stage 2: Merge (optional)
        if merge_cfg:
            if not run_merge(job_name, merge_cfg, train_cfg):
                print(f"\n  FAILED: Job '{job_name}' merge stage failed")
                all_success = False
                continue

        # Stage 3: GGUF (optional)
        if gguf_cfg:
            if not run_gguf(job_name, gguf_cfg, merge_cfg, train_cfg):
                print(f"\n  FAILED: Job '{job_name}' gguf stage failed")
                all_success = False
                continue

        # Stage 4: Install (optional)
        if install_cfg:
            if not run_install(job_name, install_cfg, gguf_cfg, merge_cfg, train_cfg):
                print(f"\n  FAILED: Job '{job_name}' install stage failed")
                all_success = False
                continue

        job_elapsed = time.time() - job_start
        print(f"\n  Job '{job_name}' completed in {job_elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\n  Pipeline complete: {total_elapsed:.1f}s total")

    if all_success:
        print(f"  All {len(jobs)} job(s) succeeded")
    else:
        print(f"  Some jobs failed — check output above")

    return all_success


def main():
    args = parse_args()
    success = run_pipeline(args.config)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
