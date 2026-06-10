#!/usr/bin/env python3
"""AttackLM — Build pipeline orchestrator (v0.2.2+).

One-shot: merge LoRA adapter → BF16 model → Q4_K_M GGUF → install
to LM Studio → optionally register with Ollama → drop a build
manifest at models/built/{name}_{timestamp}/.

This replaces the 3-command shell pipeline:

    attacklm-merge --base X --adapter Y --output Z \\
      && attacklm-gguf --input Z --install-lmstudio

with a single command:

    attacklm-build --adapter models/attacklm-3b_16g \\
                   --base ./uncensored/ \\
                   --name attacklm-3b-16g \\
                   --install-lmstudio

Under the hood, this is a thin wrapper that:
    1. Calls scripts/merge_adapter.py with the right args
    2. Calls scripts/convert_to_gguf.py with the right args
    3. (Optionally) calls scripts/register_ollama.py
    4. Drops a build manifest

The two underlying scripts remain the source of truth — this
orchestrator is just glue.

Usage:

    # Full pipeline: merge + GGUF + LM Studio
    attacklm-build --adapter models/attacklm-3b_16g \\
                   --base ./uncensored/ \\
                   --name attacklm-3b-16g

    # Skip LM Studio install, drop the GGUF in models/gguf/ only
    attacklm-build --adapter models/attacklm-3b_16g \\
                   --base ./uncensored/ \\
                   --name attacklm-3b-16g \\
                   --no-install-lmstudio

    # Register with Ollama too
    attacklm-build --adapter models/attacklm-3b_16g \\
                   --base ./uncensored/ \\
                   --name attacklm-3b-16g \\
                   --register-ollama

    # Use an already-merged model (skip the merge step)
    attacklm-build --merged models/merged/attacklm-3b-16g \\
                   --name attacklm-3b-16g
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make sibling scripts importable when invoked as a console script
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))


def _run(cmd: list[str], cwd: Path | None = None) -> int:
    """Run a subprocess, streaming stdout/stderr. Returns exit code."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def _resolve_base_model(adapter_path: Path) -> str:
    """Read the base model from the adapter's state.json or adapter_config.json.

    Falls back to None (let merge_adapter.py auto-detect).
    """
    for cfg_name in ("state.json", "adapter_config.json"):
        cfg_path = adapter_path / cfg_name
        if cfg_path.exists():
            try:
                with cfg_path.open() as f:
                    cfg = json.load(f)
                if isinstance(cfg.get("base_model"), dict):
                    bid = cfg["base_model"].get("id")
                    if bid and bid != "unknown":
                        return bid
                elif cfg.get("base_model_name_or_path"):
                    return cfg["base_model_name_or_path"]
            except (OSError, json.JSONDecodeError):
                pass
    return "auto"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="One-shot: merge LoRA → GGUF → install (v0.2.2+).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required: either an adapter (and a base) or a pre-merged model
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Path to the LoRA adapter directory (output of attacklm-train).",
    )
    parser.add_argument(
        "--merged",
        type=Path,
        default=None,
        help="Path to an already-merged model dir (skip the merge step).",
    )
    parser.add_argument(
        "--base",
        type=str,
        default=None,
        help=(
            "Base model to merge into. If --adapter is given, the base "
            "is auto-detected from the adapter's state.json / "
            "adapter_config.json. Pass this to override."
        ),
    )

    # Naming
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "Name for the built model. Default: basename of --adapter or "
            "--merged, with any timestamp suffix stripped. This is the "
            "name used for the GGUF, LM Studio install, Ollama model, "
            "and build manifest."
        ),
    )
    parser.add_argument(
        "--quant",
        type=str,
        default="Q4_K_M",
        help="Quantization type passed to llama-quantize (default: Q4_K_M).",
    )

    # Output control
    parser.add_argument(
        "--merged-output",
        type=Path,
        default=None,
        help=("Where to write the merged BF16 model. Default: models/merged/{name}/"),
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip the merge step (use --merged instead).",
    )

    # Install / register
    parser.add_argument(
        "--install-lmstudio",
        dest="install_lmstudio",
        action="store_true",
        default=True,
        help="Install to LM Studio (default: ON).",
    )
    parser.add_argument(
        "--no-install-lmstudio",
        dest="install_lmstudio",
        action="store_false",
        help="Skip LM Studio install.",
    )
    parser.add_argument(
        "--register-ollama",
        action="store_true",
        help="Register the GGUF with Ollama as a local model.",
    )

    # Misc
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-conversion even if the GGUF exists.",
    )
    parser.add_argument(
        "--keep-fp16",
        action="store_true",
        help="Keep intermediate FP16 GGUF (default: delete after quant).",
    )

    args = parser.parse_args(argv)

    # --- Validate inputs ---
    if not args.adapter and not args.merged:
        print("ERROR: must pass either --adapter or --merged", file=sys.stderr)
        sys.exit(2)
    if args.adapter and args.merged:
        print(
            "ERROR: pass only one of --adapter or --merged, not both",
            file=sys.stderr,
        )
        sys.exit(2)

    # --- Derive the model name ---
    if args.name:
        name = args.name
    elif args.adapter:
        # Strip timestamp suffix if present (e.g. _2026-06-10_01-12)
        import re

        name = re.sub(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}.*$", "", args.adapter.name)
    else:
        name = args.merged.name

    # --- Step 1: merge (if needed) ---
    if args.merged:
        merged_path = args.merged
        if not merged_path.exists():
            print(f"ERROR: --merged path does not exist: {merged_path}")
            sys.exit(2)
    else:
        # Resolve base model
        base = args.base or _resolve_base_model(args.adapter)
        if base == "auto" or not base:
            print(
                f"ERROR: could not auto-detect base model from {args.adapter}.\n"
                f"  Pass --base explicitly.\n"
                f"  (Or use --merged if you already have a merged model.)"
            )
            sys.exit(2)

        merged_output = args.merged_output or Path(f"models/merged/{name}")
        print(f"\n{'=' * 60}")
        print(f" Step 1/3: Merge LoRA → BF16")
        print(f"{'=' * 60}")
        print(f"  Adapter:   {args.adapter}")
        print(f"  Base:      {base}")
        print(f"  Output:    {merged_output}")

        rc = _run(
            [
                sys.executable,
                str(_SCRIPTS_DIR / "merge_adapter.py"),
                "--adapter",
                str(args.adapter),
                "--base",
                base,
                "--output",
                str(merged_output),
            ]
        )
        if rc != 0:
            print(f"\nERROR: merge failed (exit {rc})")
            sys.exit(rc)
        merged_path = merged_output

    # --- Step 2: convert to GGUF ---
    print(f"\n{'=' * 60}")
    print(f" Step 2/3: Convert to {args.quant} GGUF")
    print(f"{'=' * 60}")
    gguf_cmd = [
        sys.executable,
        str(_SCRIPTS_DIR / "convert_to_gguf.py"),
        "--input",
        str(merged_path),
        "--name",
        name,
        "--quant",
        args.quant,
    ]
    if args.install_lmstudio:
        gguf_cmd.append("--install-lmstudio")
    if args.register_ollama:
        gguf_cmd.append("--register-ollama")
    if args.force:
        gguf_cmd.append("--force")
    if args.keep_fp16:
        gguf_cmd.append("--keep-fp16")

    rc = _run(gguf_cmd)
    if rc != 0:
        print(f"\nERROR: GGUF conversion failed (exit {rc})")
        sys.exit(rc)

    # --- Step 3: build manifest ---
    print(f"\n{'=' * 60}")
    print(f" Step 3/3: Build manifest")
    print(f"{'=' * 60}")

    gguf_path = Path("models/gguf") / f"{name}.{args.quant}.gguf"
    if not gguf_path.exists():
        print(
            f"  ⚠️  GGUF not found at {gguf_path}; "
            f"conversion was skipped (existing GGUF). Skipping manifest."
        )
        print(f"\n✅ Build done — {gguf_path}")
        return 0

    BUILT_DIR = Path("models/built")
    BUILT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")
    build_dir = BUILT_DIR / f"{name}_{ts}"
    build_dir.mkdir(parents=True, exist_ok=True)

    # Symlink the GGUF
    gguf_link = build_dir / f"{name}.gguf"
    if gguf_link.exists() or gguf_link.is_symlink():
        gguf_link.unlink()
    gguf_link.symlink_to(gguf_path.resolve())

    # Symlink the source merged model
    source_link = build_dir / "source_model"
    if source_link.exists() or source_link.is_symlink():
        source_link.unlink()
    source_link.symlink_to(merged_path.resolve())

    # Manifest
    manifest = {
        "name": name,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "quant": args.quant,
        "gguf_path": str(gguf_path.resolve()),
        "gguf_size_bytes": gguf_path.stat().st_size,
        "source_merged": str(merged_path.resolve()),
        "source_adapter": str(args.adapter.resolve()) if args.adapter else None,
        "base_model": args.base
        or (_resolve_base_model(args.adapter) if args.adapter else None),
        "lmstudio_installed": args.install_lmstudio,
        "ollama_registered": args.register_ollama,
        "lmstudio_path": (
            str(Path.home() / ".lmstudio" / "models" / "local" / name / gguf_path.name)
            if args.install_lmstudio
            else None
        ),
    }
    manifest_path = build_dir / f"{name}.manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n📦 Build manifest: {build_dir}/")
    print(f"   {manifest_path.name}")
    print(f"   {gguf_link.name}  → {gguf_path}")

    print(f"\n{'=' * 60}")
    print(f" BUILD COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Model:    {name}")
    print(f"  GGUF:     {gguf_path}  ({gguf_path.stat().st_size / 1e9:.2f}GB)")
    if args.install_lmstudio:
        print(f"  LM Studio: ~/.lmstudio/models/local/{name}/")
    if args.register_ollama:
        print(f"  Ollama:   ollama run {name.lower().replace('_', '-')}")
    print(f"  Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
