#!/usr/bin/env python3
"""
demo.py — End-to-end AttackLM inference demo.

Loads the orchestrator and tactical agents, runs a red-team engagement
scenario showing the orchestrator routing to the correct agent, then the
agent generating the actual attack technique.

Usage:
    uv run python scripts/demo.py                          # interactive mode
    uv run python scripts/demo.py --scenario persistence   # single scenario
    uv run python scripts/demo.py --all                    # run all scenarios
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MERGED_DIR = BASE_DIR / "models" / "merged"
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Agent routing map
AGENTS = {
    "ExecutionAgent": ("execution-agent", "TA0002"),
    "PersistenceAgent": ("persistence-agent", "TA0003"),
    "PrivilegeEscalationAgent": ("privilege_escalation-agent", "TA0004"),
    "DefenseEvasionAgent": ("defense_evasion-agent", "TA0005"),
    "CredentialAccessAgent": ("credential_access-agent", "TA0006"),
    "DiscoveryAgent": ("discovery-agent", "TA0007"),
    "LateralMovementAgent": ("lateral_movement-agent", "TA0008"),
    "CommandAndControlAgent": ("command_and_control-agent", "TA0011"),
}

# Demo scenarios: (name, state_description, expected_agent)
SCENARIOS = {
    "persistence": (
        """Engagement State:
- OS: Windows Server 2022
- Privilege: SYSTEM
- EDR: None detected
- Objective: Establish persistence that survives reboot — we just got a foothold and need to survive.""",
        "PersistenceAgent",
    ),
    "credential_access": (
        """Engagement State:
- OS: Windows 10 22H2
- Privilege: Administrator
- EDR: Microsoft Defender for Endpoint (active)
- Objective: Dump LSASS credentials from this domain-joined workstation.""",
        "DefenseEvasionAgent",  # orchestrator should route through defense evasion first
    ),
    "discovery": (
        """Engagement State:
- OS: Windows Server 2019
- Privilege: Standard domain user
- EDR: None detected
- Objective: Map the Active Directory environment — need to find paths to Domain Admin.""",
        "DiscoveryAgent",
    ),
    "lateral_movement": (
        """Engagement State:
- OS: Windows 11
- Privilege: Local Administrator
- EDR: SentinelOne
- Objective: Move laterally from this workstation to the file server (10.10.10.50) using stolen credentials.""",
        "LateralMovementAgent",
    ),
    "privilege_escalation": (
        """Engagement State:
- OS: Windows 10 22H2
- Privilege: Standard user (domain\jsmith)
- EDR: None
- Objective: Escalate to local administrator — SeImpersonatePrivilege is enabled.""",
        "PrivilegeEscalationAgent",
    ),
}


# ─── Model loading ────────────────────────────────────────────


def load_agent(model_path: str, label: str):
    """Load a merged model or fall back to base + adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    merged_path = MERGED_DIR / model_path

    if merged_path.exists() and any(merged_path.glob("*.safetensors")):
        print(f"  [merged] {label}: {merged_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(merged_path), trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(merged_path),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        return tokenizer, model

    # Fallback: base + adapter
    adapter_path = BASE_DIR / "models" / model_path
    if not adapter_path.exists():
        print(f"  ❌ {label}: no adapter or merged model found at {adapter_path}")
        return None, None

    print(f"  [base+adapter] {label}: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    return tokenizer, model


def generate(tokenizer, model, messages, max_tokens=512, temperature=0.6) -> str:
    """Generate a response from the model."""
    import torch

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.shape[1] :], skip_special_tokens=True)

    # Strip DeepSeek R1 thinking tags if present
    response = response.replace("`", "`")
    response = re.sub(r"`.*?`", "", response, flags=re.DOTALL).strip()

    return response


def extract_routing_decision(text: str) -> dict | None:
    """Extract the JSON routing decision from orchestrator output."""
    # Find JSON block
    json_match = re.search(r'\{.*"agent".*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: look for agent name
    for agent_name in AGENTS:
        if agent_name.lower() in text.lower():
            return {"agent": agent_name, "context": {}}

    return None


# ─── Demo ─────────────────────────────────────────────────────


def run_demo(scenario_name: str | None = None) -> None:
    """Run the orchestrator → agent demo."""
    print("\n" + "=" * 70)
    print("  AttackLM — Mixture of Experts Red Team Demo")
    print("=" * 70)

    # Load orchestrator
    print("\n[1] Loading Orchestrator...")
    orch_tokenizer, orch_model = load_agent("orchestrator-agent", "Orchestrator")
    if orch_model is None:
        print("  ❌ Orchestrator not found. Run train_all.py first.")
        return

    # Load ALL tactical agents
    agent_cache: dict[str, tuple] = {}
    print("\n[2] Loading tactical agents...")
    for agent_name, (model_dir, tactic_id) in AGENTS.items():
        tok, mod = load_agent(model_dir, agent_name)
        if tok and mod:
            agent_cache[agent_name] = (tok, mod, tactic_id)
        else:
            print(f"  ⚠️  {agent_name} not available, skipping")

    if not agent_cache:
        print("\n  ❌ No tactical agents found. Run train_all.py first.")
        return

    print(f"\n  ✅ Loaded: 1 orchestrator + {len(agent_cache)} tactical agents")

    scenarios_to_run = [scenario_name] if scenario_name else list(SCENARIOS.keys())

    print(f"\n{'─' * 70}")
    print(f"  Running {len(scenarios_to_run)} engagement scenario(s)")
    print(f"{'─' * 70}")

    total_time = 0.0
    results = []

    for name in scenarios_to_run:
        state_text, expected_agent = SCENARIOS[name]

        print(f"\n{'#' * 70}")
        print(f"# SCENARIO: {name.replace('_', ' ').title()}")
        print(f"# EXPECTED: {expected_agent}")
        print(f"{'#' * 70}")
        print(f"\n📋 ENGAGEMENT STATE:")
        for line in state_text.strip().split("\n"):
            print(f"   {line}")

        # Step 1: Orchestrator routing
        print(f"\n🧠 ORCHESTRATOR (reasoning)...")
        start = time.time()

        orch_messages = [
            {
                "role": "user",
                "content": (
                    "You are an autonomous Red Team Orchestrator. Analyze the engagement "
                    "state below, reason about the next tactical step, and output a JSON "
                    "routing decision to invoke the correct sub-agent.\n\n"
                    f"{state_text}\n\n"
                    "Output a JSON routing decision: "
                    '{"agent": "AgentName", "context": {"target": "...", "objective": "..."}}'
                ),
            }
        ]

        orch_response = generate(
            orch_tokenizer, orch_model, orch_messages, max_tokens=300
        )
        orch_time = time.time() - start
        total_time += orch_time

        print(f"   {orch_response.strip()}")
        print(f"   ⏱  {orch_time:.1f}s")

        routing = extract_routing_decision(orch_response)
        if routing is None:
            print(
                f"\n   ⚠️  Could not extract routing decision. Skipping tactical agent."
            )
            results.append((name, expected_agent, "FAIL", "--", 0))
            continue

        routed_agent = routing["agent"]
        context = routing.get("context", {})

        match = "✅" if routed_agent == expected_agent else "⚠️"
        print(f"\n   Routing: → {routed_agent} {match}")

        # Step 2: Tactical agent execution
        agent_info = agent_cache.get(routed_agent)
        if agent_info is None:
            print(f"   ❌ Agent {routed_agent} not available")
            results.append((name, expected_agent, routed_agent, "AGENT MISSING", 0))
            continue

        agent_tok, agent_mod, tactic_id = agent_info
        print(f"\n🔧 {routed_agent.upper()} ({tactic_id}) generating technique...")
        start = time.time()

        tactical_messages = [
            {
                "role": "system",
                "content": (
                    f"You are an authorized Red Team {routed_agent.replace('Agent', '').replace('E', ' E')} "
                    f"specialist. You provide precise adversary emulation techniques for security validation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Based on the following context from the orchestrator, provide the "
                    f"exact technique, command, artifacts, and cleanup for this engagement:\n\n"
                    f"{json.dumps(context, indent=2)}\n\n"
                    f"Original objective: {state_text.split('Objective:')[-1].strip()}"
                ),
            },
        ]

        agent_response = generate(
            agent_tok, agent_mod, tactical_messages, max_tokens=400
        )
        agent_time = time.time() - start
        total_time += agent_time

        print(f"   {agent_response.strip()}")
        print(f"   ⏱  {agent_time:.1f}s")

        results.append(
            (name, expected_agent, routed_agent, match, orch_time + agent_time)
        )

    # Summary
    print(f"\n{'=' * 70}")
    print(f" DEMO COMPLETE")
    print(f"{'=' * 70}")
    print(f" {'Scenario':<25} {'Expected':<25} {'Routed':<25} {'Match':<8} {'Time':>8}")
    print(f" {'─' * 25} {'─' * 25} {'─' * 25} {'─' * 8} {'─' * 8}")
    for name, expected, routed, match, t in results:
        print(f" {name:<25} {expected:<25} {routed:<25} {match:<8} {t:>7.1f}s")
    print(f"{'─' * 25} {'─' * 25} {'─' * 25} {'─' * 8} {'─' * 8}")
    correct = sum(1 for _, _, _, m, _ in results if m == "✅")
    print(f" Correct routing: {correct}/{len(results)}")
    print(f" Total inference time: {total_time:.1f}s")
    print(f"{'=' * 70}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AttackLM orchestrator → agent demo")
    parser.add_argument(
        "--scenario", choices=list(SCENARIOS.keys()), help="Run a single scenario"
    )
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    if args.scenario:
        run_demo(args.scenario)
    elif args.all or not any(vars(args).values()):
        run_demo()
    elif args.interactive:
        run_interactive()


def run_interactive() -> None:
    """Interactive mode — user provides engagement state."""
    print("\n" + "=" * 70)
    print("  AttackLM — Interactive Red Team Orchestrator")
    print("  Type 'quit' to exit, 'help' for example scenarios")
    print("=" * 70)

    # Load models once
    print("\nLoading models...")
    orch_tokenizer, orch_model = load_agent("orchestrator-agent", "Orchestrator")
    if orch_model is None:
        print("  ❌ Orchestrator not found.")
        return

    agent_cache: dict[str, tuple] = {}
    for agent_name, (model_dir, tactic_id) in AGENTS.items():
        tok, mod = load_agent(model_dir, agent_name)
        if tok and mod:
            agent_cache[agent_name] = (tok, mod, tactic_id)

    print(f"\n✅ Ready — {len(agent_cache)} agents online.\n")

    while True:
        try:
            state = input("📋 Engagement state (or 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not state or state.lower() in ("quit", "exit", "q"):
            break
        if state.lower() == "help":
            print("\nExample states:")
            print(
                "  OS: Windows 10, Privilege: Admin, EDR: Defender, Objective: Dump LSASS"
            )
            print(
                "  OS: Windows Server 2022, Privilege: SYSTEM, Objective: Create scheduled task for persistence"
            )
            print(
                "  OS: Ubuntu 22.04, Privilege: root, Objective: Establish C2 beacon\n"
            )
            continue

        # Orchestrator
        orch_messages = [
            {
                "role": "user",
                "content": (
                    "You are an autonomous Red Team Orchestrator. Analyze the engagement "
                    "state below, reason about the next tactical step, and output a JSON "
                    "routing decision:\n\n"
                    f"{state}\n\n"
                    '{"agent": "AgentName", "context": {...}}'
                ),
            }
        ]
        orch_response = generate(
            orch_tokenizer, orch_model, orch_messages, max_tokens=200
        )
        routing = extract_routing_decision(orch_response)
        print(f"\n🧠 → {routing['agent'] if routing else 'UNKNOWN'}\n")
        print(orch_response)

        if not routing or routing["agent"] not in agent_cache:
            continue

        # Tactical agent
        agent_tok, agent_mod, tactic_id = agent_cache[routing["agent"]]
        tactical_messages = [
            {
                "role": "system",
                "content": f"You are an authorized Red Team specialist for {tactic_id}.",
            },
            {
                "role": "user",
                "content": f"Provide the technique for: {json.dumps(routing.get('context', {}))}",
            },
        ]
        response = generate(agent_tok, agent_mod, tactical_messages, max_tokens=300)
        print(f"\n🔧 {routing['agent']}:\n{response}\n")

    print("\nGoodbye.")


if __name__ == "__main__":
    main()
