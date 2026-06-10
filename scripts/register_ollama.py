#!/usr/bin/env python3
"""
register_ollama.py — Import all AttackLM GGUF models into Ollama.

Usage:
    uv run python scripts/register_ollama.py
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
GGUF_DIR = BASE_DIR / "models" / "gguf"

AGENTS = {
    "orchestrator-agent.Q4_K_M.gguf": "attacklm-orchestrator",
    "execution-agent.Q4_K_M.gguf": "attacklm-execution",
    "persistence-agent.Q4_K_M.gguf": "attacklm-persistence",
    "privilege_escalation-agent.Q4_K_M.gguf": "attacklm-privesc",
    "defense_evasion-agent.Q4_K_M.gguf": "attacklm-defense-evasion",
    "credential_access-agent.Q4_K_M.gguf": "attacklm-credential-access",
    "discovery-agent.Q4_K_M.gguf": "attacklm-discovery",
    "lateral_movement-agent.Q4_K_M.gguf": "attacklm-lateral-movement",
    "command_and_control-agent.Q4_K_M.gguf": "attacklm-c2",
}


def main():
    if not GGUF_DIR.exists():
        print(f"ERROR: {GGUF_DIR} not found. Run convert_to_gguf.py first.")
        sys.exit(1)

    ggufs = sorted(GGUF_DIR.glob("*.gguf"))
    if not ggufs:
        print(f"ERROR: No GGUF files in {GGUF_DIR}")
        sys.exit(1)

    print(f"Registering {len(ggufs)} models with Ollama:\n")

    for gguf in ggufs:
        name = AGENTS.get(gguf.name)
        if name is None:
            print(f"  ⚠️  Unknown model: {gguf.name}, skipping")
            continue

        # Check if already exists
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if name in result.stdout:
            print(f"  ⏭  {name} — already registered")
            continue

        # Build Modelfile
        modelfile = f"FROM {gguf}\n"
        print(f"  ⏳ {name} ...", end=" ", flush=True)

        result = subprocess.run(
            ["ollama", "create", name, "-f", "-"],
            input=modelfile,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            print("✅")
        else:
            print(f"❌ {result.stderr.strip()[-200:]}")

    print(f"\n✅ Done — run: ollama list")


if __name__ == "__main__":
    main()
