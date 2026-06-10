#!/usr/bin/env python3
"""
generate_dataset_deterministic.py — Build training JSONL directly from manifests.
No LLM calls. Every technique + command in the manifests becomes 3 training pairs
with varied question phrasing.

Run: uv run python scripts/generate_dataset_deterministic.py
"""

import json, random
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MANIFESTS = BASE / "data" / "manifests"
OUTPUT = BASE / "data" / "datasets"
OUTPUT.mkdir(parents=True, exist_ok=True)

QUESTION_VARIANTS = [
    "Provide the {name} technique ({tid}) on {platform}. Include the exact command, expected artifacts, and cleanup.",
    "I need to use {name} ({tid}) on {platform}. Give me the command, what telemetry it generates, and how to clean up.",
    "How do I execute {name} ({tid}) on {platform}? Show the exact syntax, expected Event IDs in the SIEM, and the remediation command.",
    "Give me the {platform} command for MITRE {tid} ({name}). Include artifacts and cleanup steps.",
    "What's the procedure for {name} on {platform}? I need the raw command, what logs it leaves, and how to remove traces.",
    "Show me {tid} ({name}) for {platform}. Include: (1) the exact command, (2) expected artifacts/telemetry, (3) cleanup.",
    "I'm writing a red team playbook. Document {name} ({tid}) on {platform} — command, artifacts, cleanup.",
    "As a detection engineer, I need to map {name} ({tid}) to its artifacts. Show the command and what Event IDs it generates on {platform}.",
]

TACTIC_DISPLAY = {
    "execution": "EXECUTION",
    "persistence": "PERSISTENCE",
    "privilege_escalation": "PRIVILEGE ESCALATION",
    "defense_evasion": "DEFENSE EVASION",
    "credential_access": "CREDENTIAL ACCESS",
    "discovery": "DISCOVERY",
    "lateral_movement": "LATERAL MOVEMENT",
    "command_and_control": "COMMAND AND CONTROL",
    "collection": "COLLECTION",
    "exfiltration": "EXFILTRATION",
}

# Delete old tactic-specific JSONL files (preserve Metasploit, orchestrator,
# and any combined files we don't own).
for tactic_name in TACTIC_DISPLAY:
    f = OUTPUT / f"{tactic_name}_dataset.jsonl"
    if f.exists():
        f.unlink()

total_pairs = 0

# Additional questions for Metasploit modules (separate phrasing)
MSF_QUESTION_VARIANTS = [
    "Which Metasploit module implements {name} ({tid})? Show the msfconsole command sequence and key options.",
    "I want to use a Metasploit module for {name} ({tid}). Walk me through the full invocation.",
    "Demonstrate the Metasploit Framework module that covers {name} ({tid}). Include: module path, options, and the resulting msfconsole command.",
    "Give me the msfconsole session for the {tid} module that performs {name}.",
]

for manifest_path in sorted(MANIFESTS.glob("*.json")):
    tactic = manifest_path.stem
    if tactic not in TACTIC_DISPLAY:
        continue

    with open(manifest_path) as f:
        data = json.load(f)

    pairs = []
    has_msf = any(tech.get("metasploit_modules") for tech in data.get("techniques", []))
    if has_msf:
        system_prompt = (
            f"You are an authorized Red Team {TACTIC_DISPLAY[tactic]} specialist with "
            f"deep knowledge of the Metasploit Framework. You provide precise adversary "
            f"emulation techniques — including pure OS-level commands and Metasploit "
            f"module invocations — for security validation."
        )
    else:
        system_prompt = f"You are an authorized Red Team {TACTIC_DISPLAY[tactic]} specialist. You provide precise adversary emulation techniques for security validation."

    for tech in data.get("techniques", []):
        tid = tech.get("technique_id", "T????")
        name = tech.get("name", "Unknown Technique")

        for cmd in tech.get("commands", []):
            platform = cmd.get("executor", "command_prompt")
            command = (cmd.get("command") or "").strip()
            cleanup = (cmd.get("cleanup") or "").strip()
            if not command or len(command) < 10:
                continue

            # Build the assistant response
            assistant = f"**Technique: {name} ({tid})**\n\n"
            assistant += f"**Command ({platform}):**\n```\n{command}\n```\n"

            if cleanup:
                assistant += f"\n**Cleanup:**\n```\n{cleanup}\n```\n"

            # 3 question variations
            chosen_qs = random.sample(QUESTION_VARIANTS, min(3, len(QUESTION_VARIANTS)))
            for q_template in chosen_qs:
                user_q = q_template.format(name=name, tid=tid, platform=platform)
                pairs.append(
                    {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_q},
                            {"role": "assistant", "content": assistant},
                        ]
                    }
                )

        # Emit pairs for Metasploit modules attached to this technique.
        # Each module yields 1 training pair with a question variant and
        # the full msfconsole command as the assistant response.
        for msf_mod in tech.get("metasploit_modules", []) or []:
            mod_path = msf_mod.get("module_path", "")
            mod_name = msf_mod.get("name") or mod_path
            # Pick the first msfconsole command
            msf_cmd = ""
            for c in msf_mod.get("commands", []):
                msf_cmd = (c.get("command") or "").strip()
                if msf_cmd:
                    break
            if not mod_path or not msf_cmd:
                continue
            # 1 question variant per module (avoid blowing up dataset size)
            q_template = random.choice(MSF_QUESTION_VARIANTS)
            user_q = q_template.format(name=mod_name, tid=tid)
            assistant = (
                f"**Technique: {name} ({tid})**\n\n"
                f"**Metasploit Module: `{mod_path}`** — {mod_name}\n\n"
                f"**msfconsole Invocation:**\n```\n{msf_cmd}\n```\n"
            )
            # Add MITRE tags
            msf_techs = msf_mod.get("mitre_techniques") or []
            if msf_techs:
                tags = ", ".join(t["id"] for t in msf_techs)
                assistant += f"\n**MITRE Tags:** {tags}\n"
            # Add CVEs
            cves = msf_mod.get("cves") or []
            if cves:
                assistant += f"**CVEs:** {', '.join(cves[:5])}\n"
            pairs.append(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_q},
                        {"role": "assistant", "content": assistant.strip()},
                    ]
                }
            )

    if pairs:
        random.shuffle(pairs)
        outfile = OUTPUT / f"{tactic}_dataset.jsonl"
        with open(outfile, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        print(f"  {tactic}: {len(pairs)} pairs")
        total_pairs += len(pairs)
    else:
        print(f"  {tactic}: 0 pairs (no commands in manifest)")

print(f"\nDone. {total_pairs} total training pairs → {OUTPUT}/")
