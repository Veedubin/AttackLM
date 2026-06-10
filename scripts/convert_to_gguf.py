#!/usr/bin/env python3
"""
convert_to_gguf.py — Convert merged AttackLM models to GGUF (Q4_K_M).

Step 1: convert_hf_to_gguf → FP16
Step 2: llama-quantize → Q4_K_M

Usage:
    # Convert all merged models:
    attacklm-gguf

    # Convert a single model:
    attacklm-gguf --input models/merged/attacklm

    # Convert and install to LM Studio:
    attacklm-gguf --install-lmstudio
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MERGED_DIR = BASE_DIR / "models" / "merged"
GGUF_DIR = BASE_DIR / "models" / "gguf"


def find_llama_bin(name: str) -> Path:
    """Find a llama.cpp binary or script."""
    if name.endswith(".py"):
        candidates = [
            Path("/home/jcharles/Projects/llama.cpp") / name,
            Path("/tmp/llama.cpp-i_il8hw4") / name,
            Path("/usr/local/bin") / name,
            Path("/usr/local/share/llama.cpp") / name,
            Path.home() / "llama.cpp" / name,
            BASE_DIR.parent / "llama.cpp" / name,
        ]
        for c in candidates:
            if c.exists():
                return c
        # glob /tmp/llama.cpp-*
        import glob

        for d in glob.glob("/tmp/llama.cpp-*"):
            p = Path(d) / name
            if p.exists():
                return p
    else:
        result = subprocess.run(["which", name], capture_output=True, text=True)
        if result.returncode == 0:
            return Path(result.stdout.strip())
        candidates = [
            Path("/usr/bin") / name,
            Path("/usr/local/bin") / name,
            Path.home() / "llama.cpp" / "build" / "bin" / name,
            BASE_DIR.parent / "llama.cpp" / "build" / "bin" / name,
        ]
        for c in candidates:
            if c.exists():
                return c

    return None


def _find_merged_models(merged_dir: Path) -> list[Path]:
    """Find directories under models/merged/ that contain config.json + safetensors."""
    if not merged_dir.exists():
        return []
    return sorted(
        [
            p
            for p in merged_dir.iterdir()
            if p.is_dir()
            and (p / "config.json").exists()
            and any(p.glob("*.safetensors"))
        ]
    )


def _is_lora_adapter(model_dir: Path) -> bool:
    """Detect a LoRA adapter directory (vs a merged model).

    A merged model has `config.json` whose `model_type` is a known HF architecture
    (qwen2, llama, mistral, etc.) plus a full set of `*.safetensors` weights.

    A LoRA adapter has `adapter_config.json` with `"peft_type": "LORA"` and only
    a small `adapter_model.safetensors` (typically 30-100 MB for r=16-64).

    The trap we hit in v0.1.5: an adapter dir ALSO has a `config.json` (left
    over from a previous training run), so the old "needs config.json" guard
    let it through and llama.cpp's converter then crashed on the wrong
    architecture. This check is explicit: if adapter_config.json says
    PEFT_TYPE_LORA, refuse with a pointer to attacklm-merge.
    """
    ac = model_dir / "adapter_config.json"
    if not ac.exists():
        return False
    try:
        import json as _json

        with open(ac) as f:
            cfg = _json.load(f)
        return cfg.get("peft_type", "").upper() == "LORA"
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert merged models to Q4_K_M GGUF (v0.2.2+)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Single model directory to convert (e.g., models/merged/attacklm). "
        "Default: convert all models found in models/merged/.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "v0.2.2+: Override the auto-derived model name (used for GGUF "
            "filename and LM Studio install path). Default: the basename of "
            "--input. Example: --input models/merged/foo --name bar → "
            "models/gguf/bar.Q4_K_M.gguf, ~/.lmstudio/models/local/bar/bar.Q4_K_M.gguf"
        ),
    )
    parser.add_argument(
        "--quant",
        type=str,
        default="Q4_K_M",
        help="Quantization type (default: Q4_K_M). Other options: Q8_0, Q5_K_M, Q6_K.",
    )
    parser.add_argument(
        "--keep-fp16", action="store_true", help="Keep intermediate FP16 GGUF files"
    )
    parser.add_argument(
        "--install-lmstudio",
        action="store_true",
        help="Install GGUF files to ~/.lmstudio/models/local/ (default: just save to models/gguf/)",
    )
    parser.add_argument(
        "--register-ollama",
        action="store_true",
        help=(
            "v0.2.2+: After conversion, register the GGUF with Ollama as a "
            "local model (creates a Modelfile + `ollama create`). "
            "Default: skip Ollama registration."
        ),
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help=(
            "v0.2.2+: One-shot full pipeline. Converts → installs to LM Studio "
            "→ registers with Ollama (if --register-ollama) → copies a "
            "build manifest to models/built/{name}_{timestamp}/. "
            "Equivalent to: --install-lmstudio + a build-manifest drop."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "v0.2.2+: Force re-conversion even if the GGUF already exists. "
            "Default: skip if models/gguf/{name}.{quant}.gguf exists AND its "
            "mtime is newer than the source model.safetensors."
        ),
    )
    args = parser.parse_args()

    # Find tools
    converter = find_llama_bin("convert_hf_to_gguf.py")
    quantizer = find_llama_bin("llama-quantize")

    if converter is None:
        print("ERROR: convert_hf_to_gguf.py not found.")
        print("Install llama.cpp or set PYTHONPATH to its directory.")
        sys.exit(1)
    if quantizer is None:
        print("ERROR: llama-quantize not found on PATH.")
        print("Build llama.cpp: cmake -B build && cmake --build build -j")
        print("Then add build/bin/ to PATH.")
        sys.exit(1)

    print(f"Converter: {converter}")
    print(f"Quantizer: {quantizer}")

    # Find merged models
    if args.input:
        input_path = Path(args.input)
        if not input_path.is_dir():
            print(f"\nERROR: {args.input} is not a directory")
            sys.exit(1)
        # Reject LoRA adapter directories explicitly. This was the silent
        # footgun in v0.1.5: an adapter dir has a `config.json` left over
        # from a previous run, so the old "needs config.json" guard passed
        # it through and llama.cpp's converter then crashed on the wrong
        # architecture with `Failed to detect model architecture`. Detect
        # PEFT adapters early and point users at attacklm-merge.
        if _is_lora_adapter(input_path):
            # Try to read the auto-detected base from state.json (v0.1.6+)
            # or adapter_config.json (v0.1.5 and earlier) so the hint
            # is accurate without the user needing to know.
            suggested_base = "(unknown — set --base-model manually)"
            for cfg_name in ("state.json", "adapter_config.json"):
                cfg_path = input_path / cfg_name
                if cfg_path.exists():
                    try:
                        with cfg_path.open() as f:
                            cfg = json.load(f)
                        if isinstance(cfg.get("base_model"), dict):
                            suggested_base = cfg["base_model"].get("id", suggested_base)
                        else:
                            suggested_base = cfg.get(
                                "base_model_name_or_path", suggested_base
                            )
                        if suggested_base != "(unknown — set --base-model manually)":
                            break
                    except (OSError, json.JSONDecodeError):
                        pass

            print(
                f"\nERROR: {args.input} looks like a LoRA adapter, not a merged model."
            )
            print(
                f"  Has:   adapter_config.json (peft_type=LORA) + adapter_model.safetensors"
            )
            print(
                f"  Need:  a directory with config.json + *.safetensors (full merged weights)"
            )
            print(f"\nFix: merge the adapter into the base first, then convert.")
            print(f"  attacklm-merge --adapter {args.input} \\")
            print(f"                  --base {suggested_base} \\")
            print(f"                  --output models/merged/{input_path.name}")
            print(
                f"  attacklm-gguf --input models/merged/{input_path.name}"
                + (" --install-lmstudio" if args.install_lmstudio else "")
            )
            sys.exit(1)
        if not (input_path / "config.json").exists() or not any(
            input_path.glob("*.safetensors")
        ):
            print(f"\nERROR: {args.input} is not a valid merged model directory")
            print("Expected a directory containing config.json and *.safetensors")
            print("If this is a LoRA adapter, run `attacklm-merge` first (see above).")
            sys.exit(1)
        models = [input_path]
    else:
        models = _find_merged_models(MERGED_DIR)
        if not models:
            print(f"\nERROR: No merged models found in {MERGED_DIR}")
            print(
                "Run: attacklm-merge --adapter models/attacklm-single --output models/merged/attacklm"
            )
            print("  or: attacklm-merge --merge-all")
            sys.exit(1)

    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting {len(models)} model(s) to {args.quant} GGUF:\n")

    converted_now: list[Path] = []  # only GGUFs produced THIS run

    for model_dir in models:
        # v0.2.2+: name can be overridden via --name
        name = args.name if args.name else model_dir.name
        final_path = GGUF_DIR / f"{name}.{args.quant}.gguf"

        # v0.2.2+: skip-if-exists only when:
        #   - the GGUF exists AND
        #   - it has a non-zero size AND
        #   - its mtime is >= the source model.safetensors mtime
        # Otherwise, treat as stale and re-convert. --force bypasses
        # the mtime check entirely.
        if final_path.exists() and not args.force:
            src_sf = next(iter(model_dir.glob("*.safetensors")), None)
            if (
                src_sf is not None
                and final_path.stat().st_mtime >= src_sf.stat().st_mtime
            ):
                print(
                    f"  ⏭  {name} — already exists at {final_path.name} (use --force to re-convert)"
                )
                continue
            else:
                print(
                    f"  ↻  {name} — stale GGUF found (source is newer); re-converting"
                )

        fp16_path = GGUF_DIR / f"{name}.FP16.gguf"

        # Step 1: HuggingFace → FP16 GGUF
        if not fp16_path.exists() or args.force:
            print(f"  ⏳ {name} → FP16 ...", end=" ", flush=True)
            result = subprocess.run(
                [
                    sys.executable,
                    str(converter),
                    str(model_dir),
                    "--outfile",
                    str(fp16_path),
                    "--outtype",
                    "f16",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print("❌")
                # Combine stdout + stderr; some llama.cpp versions dump the
                # real error to stdout. Show last 800 chars of both.
                combined = (result.stderr or "") + (result.stdout or "")
                tail = combined.strip()[-800:] if combined.strip() else "(no output)"
                print(f"     {tail}")
                continue
            print(f"✅ {fp16_path.stat().st_size / 1e9:.2f}GB")

        # Step 2: FP16 → target quant
        print(f"  ⏳ {name} → {args.quant} ...", end=" ", flush=True)
        result = subprocess.run(
            [str(quantizer), str(fp16_path), str(final_path), args.quant],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("❌")
            # Same: combine streams, show more
            combined = (result.stderr or "") + (result.stdout or "")
            tail = combined.strip()[-800:] if combined.strip() else "(no output)"
            print(f"     {tail}")
            # Clean up the FP16 we just made — it's useless without quant
            if fp16_path.exists() and not args.keep_fp16:
                fp16_path.unlink()
            continue

        print(f"✅ {final_path.stat().st_size / 1e9:.2f}GB")
        converted_now.append(final_path)

        # Clean up FP16 intermediate
        if not args.keep_fp16 and fp16_path.exists():
            fp16_path.unlink()

    # Install to LM Studio (opt-in).
    # LM Studio scans ~/.lmstudio/models/local/ (NOT ~/.lmstudio/local/models/).
    #
    # v0.1.5 bug: this block ran even on conversion failure, globbing
    # *every* .gguf in GGUF_DIR (including stale ones from previous runs)
    # and copying them to LM Studio — masquerading a failed run as success.
    # v0.1.6 fix: only install GGUFs that were produced OR already-skipped
    # *in this invocation*. A run that crashed on FP16 conversion now
    # correctly produces a "no new GGUFs to install" message instead of
    # silently re-deploying yesterday's stale build.
    # v0.2.2+: --build mode is equivalent to --install-lmstudio
    do_install_lmstudio = args.install_lmstudio or args.build
    if do_install_lmstudio:
        if not converted_now:
            print(
                "\n⚠️  No new GGUFs were produced this run — skipping LM Studio install."
            )
            print(
                "   (This usually means conversion failed or all GGUFs were skipped.)"
            )
        else:
            lmstudio_dir = Path.home() / ".lmstudio" / "models" / "local"
            for gguf in converted_now:
                # Use the explicit --name if given, else strip the
                # {quant} suffix from the GGUF stem. The two
                # approaches should produce the same name.
                agent_name = (
                    args.name
                    if args.name
                    else Path(gguf).stem.replace(f".{args.quant}", "")
                )
                agent_dir = lmstudio_dir / agent_name
                agent_dir.mkdir(parents=True, exist_ok=True)
                dest = agent_dir / gguf.name
                shutil.copy2(gguf, dest)
                print(f"   ➜ ~/.lmstudio/models/local/{agent_name}/{gguf.name}")

            print(
                f"\n✅ Installed {len(converted_now)} GGUF(s) to ~/.lmstudio/models/local/"
            )
            print("   Restart LM Studio (or click 'Refresh') to pick up the new model.")
    else:
        # Print hint about manual install
        agent_name_for_hint = "attacklm"
        if GGUF_DIR.glob("*.gguf"):
            stem = next(GGUF_DIR.glob("*.gguf")).stem
            for suffix in (".Q4_K_M", ".Q8_0", ".F16", ".FP16"):
                if stem.endswith(suffix):
                    agent_name_for_hint = stem[: -len(suffix)]
                    break
        print(
            f"\n💡 To use in LM Studio, copy GGUF files to ~/.lmstudio/models/local/{agent_name_for_hint}/"
        )
        print(f"   Or re-run with --install-lmstudio to do it automatically.")

    print(f"\n✅ Done — {GGUF_DIR}/")
    for gguf in sorted(GGUF_DIR.glob("*.gguf")):
        print(f"   {gguf.name}  ({gguf.stat().st_size / 1e6:.0f}MB)")

    # v0.2.2+: optional Ollama registration
    if args.register_ollama and converted_now:
        _register_with_ollama(
            converted_now,
            name=args.name,
            quant=args.quant,
        )

    # v0.2.2+: --build mode drops a build manifest at
    # models/built/{name}_{timestamp}/ containing symlinks to the GGUF,
    # the source merged model, and a summary JSON.
    if args.build and converted_now:
        _drop_build_manifest(
            converted_now,
            name=args.name,
            quant=args.quant,
            do_install_lmstudio=do_install_lmstudio,
            do_register_ollama=args.register_ollama,
        )


# ---------------------------------------------------------------------------
# v0.2.2+: Ollama registration
# ---------------------------------------------------------------------------


def _register_with_ollama(
    ggufs: list[Path],
    name: str | None,
    quant: str,
) -> None:
    """Register GGUFs with Ollama as local models.

    Ollama reads a `Modelfile` and runs `ollama create`. We write a
    minimal Modelfile per GGUF and invoke `ollama create`. The user
    can then run `ollama run {name}` to use it.

    The ollama binary must be on PATH. If it's not, we just print
    the Modelfile content and the manual command — non-fatal.
    """
    import shutil as _sh
    from datetime import datetime, timezone

    ollama_bin = _sh.which("ollama")
    if ollama_bin is None:
        print("\n⚠️  --register-ollama requested but 'ollama' is not on PATH.")
        print("   Skipping Ollama registration (you can do it manually).")
        return

    for gguf in ggufs:
        # Derive the ollama model name (must be lowercase + dashes only)
        agent_name = name if name else Path(gguf).stem.replace(f".{quant}", "")
        ollama_name = agent_name.lower().replace("_", "-").replace("/", "-")
        # Modelfile: just `FROM <absolute path to gguf>`
        modelfile_path = gguf.parent / f"{ollama_name}.Modelfile"
        with open(modelfile_path, "w") as f:
            f.write(f"FROM {gguf.resolve()}\n")

        print(f"\n🐫 Registering {gguf.name} with Ollama as '{ollama_name}'...")
        result = subprocess.run(
            [ollama_bin, "create", ollama_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"   ❌ Ollama create failed: {result.stderr[:400]}")
            continue
        print(f"   ✅ Created: ollama run {ollama_name}")


# ---------------------------------------------------------------------------
# v0.2.2+: Build manifest
# ---------------------------------------------------------------------------


def _drop_build_manifest(
    ggufs: list[Path],
    name: str | None,
    quant: str,
    do_install_lmstudio: bool,
    do_register_ollama: bool,
) -> None:
    """Drop a build manifest at models/built/{name}_{timestamp}/.

    Contents:
        {name}_{timestamp}.gguf         symlink to the GGUF in models/gguf/
        {name}_{timestamp}.manifest.json  build summary (size, paths, mtimes)
        source_model/                  symlink to the source merged model dir
    """
    from datetime import datetime, timezone
    import json as _json

    BUILT_DIR = Path("models/built")
    BUILT_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")
    for gguf in ggufs:
        agent_name = name if name else Path(gguf).stem.replace(f".{quant}", "")
        build_dir = BUILT_DIR / f"{agent_name}_{ts}"
        build_dir.mkdir(parents=True, exist_ok=True)

        # Symlink the GGUF into the build dir (use a stable name without quant)
        gguf_link = build_dir / f"{agent_name}.gguf"
        if gguf_link.exists() or gguf_link.is_symlink():
            gguf_link.unlink()
        gguf_link.symlink_to(gguf.resolve())

        # Symlink the source merged model
        # The source is models/merged/{agent_name}/ — find it
        source_merged = Path("models/merged") / agent_name
        if source_merged.exists():
            source_link = build_dir / "source_model"
            if source_link.exists() or source_link.is_symlink():
                source_link.unlink()
            source_link.symlink_to(source_merged.resolve())

        # Write a manifest
        manifest = {
            "name": agent_name,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "quant": quant,
            "gguf_path": str(gguf.resolve()),
            "gguf_size_bytes": gguf.stat().st_size,
            "source_merged": (
                str(source_merged.resolve()) if source_merged.exists() else None
            ),
            "lmstudio_installed": do_install_lmstudio,
            "ollama_registered": do_register_ollama,
            "lmstudio_path": (
                str(
                    Path.home()
                    / ".lmstudio"
                    / "models"
                    / "local"
                    / agent_name
                    / gguf.name
                )
                if do_install_lmstudio
                else None
            ),
        }
        manifest_path = build_dir / f"{agent_name}.manifest.json"
        with open(manifest_path, "w") as f:
            _json.dump(manifest, f, indent=2)

        print(f"\n📦 Build manifest: {build_dir}/")
        print(f"   {manifest_path.name}  (full build summary)")
        print(f"   {gguf_link.name}  → {gguf}")


if __name__ == "__main__":
    main()
