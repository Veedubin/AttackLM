#!/usr/bin/env python3
"""
convert_to_gguf.py — Convert all merged AttackLM models to GGUF (Q4_K_M) in one shot.

Step 1: convert_hf_to_gguf → FP16
Step 2: llama-quantize → Q4_K_M

Usage:
    uv run python scripts/convert_to_gguf.py
"""

import argparse
import os
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert all merged models to Q4_K_M GGUF"
    )
    parser.add_argument(
        "--keep-fp16", action="store_true", help="Keep intermediate FP16 GGUF files"
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
    models = sorted(MERGED_DIR.glob("*-agent"))
    if not models:
        print(f"\nERROR: No merged models found in {MERGED_DIR}")
        print("Run: uv run python scripts/merge_adapter.py --merge-all")
        sys.exit(1)

    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting {len(models)} models to Q4_K_M GGUF:\n")

    for model_dir in models:
        name = model_dir.name
        final_path = GGUF_DIR / f"{name}.Q4_K_M.gguf"

        if final_path.exists():
            print(f"  ⏭  {name} — already exists")
            continue

        fp16_path = GGUF_DIR / f"{name}.FP16.gguf"

        # Step 1: HuggingFace → FP16 GGUF
        if not fp16_path.exists():
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
                print(f"     {result.stderr.strip()[-300:]}")
                continue
            print(f"✅ {fp16_path.stat().st_size / 1e9:.2f}GB")

        # Step 2: FP16 → Q4_K_M
        print(f"  ⏳ {name} → Q4_K_M ...", end=" ", flush=True)
        result = subprocess.run(
            [str(quantizer), str(fp16_path), str(final_path), "Q4_K_M"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("❌")
            print(f"     {result.stderr.strip()[-300:]}")
            continue

        print(f"✅ {final_path.stat().st_size / 1e9:.2f}GB")

        # Clean up FP16 intermediate
        if not args.keep_fp16 and fp16_path.exists():
            fp16_path.unlink()

    # Install to LM Studio
    lmstudio_dir = Path.home() / ".lmstudio" / "models" / "attacklm"
    for gguf in sorted(GGUF_DIR.glob("*.gguf")):
        # LM Studio expects: ~/.lmstudio/models/attacklm/{name}/{name}-{quant}.gguf
        agent_name = gguf.stem.replace(".Q4_K_M", "").replace(".Q4_K_M", "")
        agent_dir = lmstudio_dir / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        dest = agent_dir / gguf.name
        if not dest.exists():
            shutil.copy2(gguf, dest)
            print(f"   ➜ ~/.lmstudio/models/attacklm/{agent_name}/")

    print(f"\n✅ Done — {GGUF_DIR}/")
    for gguf in sorted(GGUF_DIR.glob("*.gguf")):
        print(f"   {gguf.name}  ({gguf.stat().st_size / 1e6:.0f}MB)")
    print(f"\n✅ Installed to ~/.lmstudio/models/attacklm/")


if __name__ == "__main__":
    main()
