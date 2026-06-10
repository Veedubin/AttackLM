#!/usr/bin/env python3
"""Generate training JSONL for the Orchestrator routing model.

Procedural generation with hardcoded seed scenarios and linguistic
variations.  No LLM calls — deterministic and fast.

Output: ``data/datasets/orchestrator_dataset.jsonl``
Each row is an OpenAI-style messages triple (system / user / assistant)
with Chain-of-Thought reasoning and a JSON routing decision.

Usage:
    python scripts/generate_orchestrator.py
    python scripts/generate_orchestrator.py --output custom_path.jsonl
"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)

# ---------------------------------------------------------------------------
# Agent mapping
# ---------------------------------------------------------------------------
AGENTS = {
    "ExecutionAgent": "TA0002",
    "PersistenceAgent": "TA0003",
    "PrivilegeEscalationAgent": "TA0004",
    "DefenseEvasionAgent": "TA0005",
    "CredentialAccessAgent": "TA0006",
    "DiscoveryAgent": "TA0007",
    "LateralMovementAgent": "TA0008",
    "CommandAndControlAgent": "TA0011",
}

# ---------------------------------------------------------------------------
# Core scenarios — (objective, correct_agent, reasoning)
# ---------------------------------------------------------------------------
SCENARIOS = [
    (
        "Establish registry persistence before reboot",
        "PersistenceAgent",
        "We need to survive reboot. No EDR detected, so we can route directly to PersistenceAgent.",
    ),
    (
        "Dump LSASS credentials",
        "DefenseEvasionAgent",
        "Attempting LSASS access with EDR active will trigger alerts. "
        "Must route through DefenseEvasionAgent first to blind the sensor.",
    ),
    (
        "Enumerate domain users and groups",
        "DiscoveryAgent",
        "We have a foothold but no situational awareness. "
        "Route to DiscoveryAgent for BloodHound/AD enumeration.",
    ),
    (
        "Move laterally to the file server",
        "LateralMovementAgent",
        "We have credentials but need to pivot. "
        "Route to LateralMovementAgent for PSRemoting or WMI.",
    ),
    (
        "Establish C2 beacon with HTTPS",
        "CommandAndControlAgent",
        "Need a C2 channel. Route to CommandAndControlAgent for profile generation.",
    ),
    (
        "Bypass Windows Defender for Mimikatz",
        "DefenseEvasionAgent",
        "Before credential access, we need to handle Defender. "
        "Route to DefenseEvasionAgent for AMSI bypass/unhooking.",
    ),
    (
        "Escalate from local admin to SYSTEM",
        "PrivilegeEscalationAgent",
        "Local admin but need SYSTEM. "
        "Route to PrivilegeEscalationAgent for token manipulation or exploit.",
    ),
    (
        "Run BloodHound collection",
        "DiscoveryAgent",
        "Need to map the domain. Route to DiscoveryAgent for SharpHound execution.",
    ),
    (
        "Exploit vulnerable AD certificate template",
        "PrivilegeEscalationAgent",
        "ADCS ESC1 detected. Route to PrivilegeEscalationAgent for Certify abuse.",
    ),
    (
        "Create WMI event subscription for persistence",
        "PersistenceAgent",
        "Stealthy persistence mechanism needed. "
        "Route to PersistenceAgent for WMI event consumer setup.",
    ),
]

# ---------------------------------------------------------------------------
# Environment combinations — (os, privileges, edr)
# ---------------------------------------------------------------------------
ENVIRONMENTS = [
    ("Windows Server 2022", "SYSTEM", "CrowdStrike Falcon"),
    ("Windows Server 2022", "SYSTEM", "None"),
    ("Windows 10 22H2", "Administrator", "Microsoft Defender for Endpoint"),
    ("Windows 10 22H2", "Administrator", "None"),
    ("Windows 11", "Local Admin", "SentinelOne"),
    ("Windows 11", "Local Admin", "None"),
    ("Windows Server 2019", "SYSTEM", "Carbon Black"),
    ("Windows Server 2019", "Standard User", "CrowdStrike Falcon"),
    ("Ubuntu 22.04", "root", "None"),
    ("Ubuntu 22.04", "www-data", "None"),
]

# ---------------------------------------------------------------------------
# Multi-step scenarios — higher-level objectives that require sequencing
# ---------------------------------------------------------------------------
MULTISTEP_SCENARIOS = [
    # (objective, first_agent, full_reasoning, context_next_objective)
    (
        "Get Domain Admin from standard user on Windows 10 with Defender",
        "DiscoveryAgent",
        "We are a standard user on Windows 10 with Defender active — no situational "
        "awareness, no credentials beyond our own. Rushing to privilege escalation or "
        "credential attacks will burn the beacon. The first move must be reconnaissance: "
        "map the domain, find attack paths, identify where Domain Admins login. "
        "Route to DiscoveryAgent for BloodHound/SharpHound collection.",
        "Enumerate domain topology and find privilege escalation paths",
    ),
    (
        "Compromise the domain with only a low-priv shell on a server with CrowdStrike",
        "DefenseEvasionAgent",
        "Low-priv shell on a monitored server with CrowdStrike Falcon. Any aggressive "
        "action (credential dumping, exploitation) will trigger behavioral detection. "
        "We must first blind or neutralize the EDR. Route to DefenseEvasionAgent for "
        "sensor unhooking or BYOVD approach before any further action.",
        "Disable or blind EDR before proceeding with enumeration",
    ),
    (
        "Pivot from compromised workstation to domain controller",
        "LateralMovementAgent",
        "We have a foothold on a workstation and harvested hashes from memory. "
        "The domain controller is accessible via SMB. Route to LateralMovementAgent "
        "for Pass-the-Hash via SMB exec or WMI remote execution.",
        "Move laterally to DC using harvested credentials",
    ),
    (
        "Establish persistent C2 on a locked-down network",
        "CommandAndControlAgent",
        "Need a resilient C2 channel on a network with egress filtering. Route to "
        "CommandAndControlAgent for HTTPS beacon profile with domain fronting to "
        "blend with legitimate traffic.",
        "Set up domain-fronted C2 channel",
    ),
    (
        "Persistent access on Linux server before expected reboot",
        "PersistenceAgent",
        "Need to ensure we survive an upcoming reboot on this Linux server. Route to "
        "PersistenceAgent for cron job, systemd service, or SSH key persistence.",
        "Establish persistence via cron/systemd",
    ),
    (
        "Find attack surface on unknown internal network",
        "DiscoveryAgent",
        "Blind on an unknown network segment. No idea what hosts or services exist. "
        "Route to DiscoveryAgent for nmap scanning and service enumeration.",
        "Enumerate hosts and services on the network",
    ),
    (
        "Harvest credentials from AD with EDR present on all endpoints",
        "DefenseEvasionAgent",
        "EDR is active across the environment. Kerberoasting and LSASS access will "
        "trigger alerts. Must first route to DefenseEvasionAgent to create a blind "
        "spot or disable real-time protection before any credential operations.",
        "Blind EDR before attempting credential harvesting",
    ),
    (
        "Escalate from www-data to root on Ubuntu",
        "PrivilegeEscalationAgent",
        "Web shell as www-data on Ubuntu. No EDR on Linux, so we can move directly to "
        "privilege escalation. Route to PrivilegeEscalationAgent for kernel exploit, "
        "SUID binary, or cron misconfiguration.",
        "Escalate to root via SUID or kernel exploit",
    ),
    (
        "Establish fallback C2 after primary channel was blocked",
        "CommandAndControlAgent",
        "Primary C2 was detected and blocked by firewall. Need a fallback channel. "
        "Route to CommandAndControlAgent for DNS or HTTPS alternate beacon profile.",
        "Deploy fallback C2 channel via DNS/HTTPS",
    ),
    (
        "Maintain access after admin resets compromised accounts",
        "PersistenceAgent",
        "Admin has reset passwords on accounts we controlled. We need redundant "
        "persistence mechanisms that don't rely on credential artifacts. "
        "Route to PersistenceAgent for implant-based persistence (service, WMI, "
        "scheduled task).",
        "Deploy credential-independent persistence mechanisms",
    ),
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an autonomous Red Team Orchestrator. Analyze the engagement state, "
    "reason about the next tactical step using Chain-of-Thought, and output a "
    "precise JSON routing decision to invoke the correct tactical sub-agent. "
    "Always consider: current privileges, EDR presence, and operational safety. "
    "Never route to CredentialAccessAgent if EDR is active — route through "
    "DefenseEvasionAgent first."
)

# ---------------------------------------------------------------------------
# Template pools for reasoning — per-agent, varied phrasing
# ---------------------------------------------------------------------------
REASONING_TEMPLATES: dict[str, list[str]] = {
    "PersistenceAgent": [
        "Survival requires persistence. {edr_note}Route to PersistenceAgent "
        "for {mechanism}.",
        "Need to survive reboot or session loss. {edr_note}PersistenceAgent "
        "will handle {mechanism}.",
        "Objective demands persistence before anything else. {edr_note}Routing "
        "to PersistenceAgent to establish {mechanism}.",
        "Without persistence we lose access ondisconnect. {edr_note}Route to "
        "PersistenceAgent for {mechanism}.",
    ],
    "DefenseEvasionAgent": [
        "EDR is active — {edr_name} will flag any credential or exploitation "
        "attempt. We must blind the sensor first. Route to DefenseEvasionAgent "
        "for {evasion_method}.",
        "Direct approach triggers {edr_name}. Route through "
        "DefenseEvasionAgent first to {evasion_method}.",
        "{edr_name} is watching. Before we touch LSASS or run exploits, "
        "DefenseEvasionAgent needs to {evasion_method}.",
        "Operational safety requires we handle {edr_name} before proceeding. "
        "DefenseEvasionAgent will {evasion_method}.",
    ],
    "DiscoveryAgent": [
        "We have a foothold but are blind to the domain topology. Route to "
        "DiscoveryAgent for {discovery_method}.",
        "Situational awareness is zero. DiscoveryAgent handles "
        "{discovery_method} to map the environment.",
        "Cannot plan next moves without recon. Route to DiscoveryAgent for "
        "{discovery_method}.",
        "Need to enumerate before we act. DiscoveryAgent will run {discovery_method}.",
    ],
    "LateralMovementAgent": [
        "We have credentials and a target. Route to LateralMovementAgent for "
        "{movement_method}.",
        "Time to pivot. LateralMovementAgent handles {movement_method} to "
        "reach the target.",
        "Target is reachable with current credentials. "
        "LateralMovementAgent executes {movement_method}.",
        "Ready to move laterally. Route to LateralMovementAgent for {movement_method}.",
    ],
    "CommandAndControlAgent": [
        "Need a C2 channel established. Route to CommandAndControlAgent for "
        "{c2_method}.",
        "No command path to the target. CommandAndControlAgent sets up {c2_method}.",
        "Beacon communication required. CommandAndControlAgent generates "
        "{c2_method} profile.",
        "Must establish egress. Route to CommandAndControlAgent for {c2_method}.",
    ],
    "PrivilegeEscalationAgent": [
        "Current privileges are insufficient. Route to "
        "PrivilegeEscalationAgent for {priv_method}.",
        "Need higher privileges to proceed. PrivilegeEscalationAgent handles "
        "{priv_method}.",
        "Stuck at {priv_level} — PrivilegeEscalationAgent can exploit {priv_method}.",
        "Escalation required. Route to PrivilegeEscalationAgent for {priv_method}.",
    ],
    "CredentialAccessAgent": [
        "EDR is off — safe to harvest credentials. Route to "
        "CredentialAccessAgent for {cred_method}.",
        "No EDR blocking us. CredentialAccessAgent handles {cred_method}.",
        "Environment is clear for credential operations. "
        "CredentialAccessAgent executes {cred_method}.",
        "We can safely dump credentials now. Route to "
        "CredentialAccessAgent for {cred_method}.",
    ],
    "ExecutionAgent": [
        "Need to run commands on target. Route to ExecutionAgent for {exec_method}.",
        "Payload needs to execute. ExecutionAgent handles {exec_method}.",
        "Time to run on the target. Route to ExecutionAgent for {exec_method}.",
        "Command execution required. ExecutionAgent performs {exec_method}.",
    ],
}

# Mechanism/vocabulary pools for template expansion
MECHANISM_POOLS: dict[str, list[str]] = {
    "persistence": [
        "registry Run key setup",
        "scheduled task creation",
        "WMI event subscription",
        "service installation",
        "startup folder placement",
    ],
    "evasion": [
        "AMSI bypass and real-time protection disable",
        "EDR unhooking via userland API patching",
        "BYOVD to terminate sensor processes",
        "process injection to hide in legitimate context",
        "DLL sideloading for payload execution",
    ],
    "discovery": [
        "BloodHound/SharpHound collection",
        "AD enumeration with PowerView",
        "network port scanning",
        "system and account reconnaissance",
        "SPN enumeration for Kerberoast targets",
    ],
    "movement": [
        "PSRemoting or WMI lateral execution",
        "Pass-the-Hash via SMB",
        "RDP hijacking",
        "SSH tunnel pivoting",
        "DCOM-based remote execution",
    ],
    "c2": [
        "HTTPS beacon profile with domain fronting",
        "DNS tunnel channel",
        "WebSocket-based C2 with jitter",
        "malleable C2 profile generation",
        "HTTPS long-polling beacon config",
    ],
    "privilege": [
        "token manipulation or named pipe impersonation",
        "UAC bypass via Fodhelper",
        "kernel exploit (e.g., PrintNightmare, HiveNightmare)",
        "ADCS template abuse (ESC1/ESC3)",
        "SUID binary exploitation",
    ],
    "credential": [
        "LSASS memory dump with comsvcs.dll",
        "Kerberoasting for service account hashes",
        "DPAPI master key extraction",
        "SAM/SYSTEM hive offline credential extraction",
        "domain cached credential dumping",
    ],
    "execution": [
        "PowerShell cradle execution",
        "LOLBins (certutil, mshta, msiexec)",
        "native API invocation via reflective DLL",
        "cmd.exe one-liner chain",
        "VBScript/JScript payload delivery",
    ],
}

# ---------------------------------------------------------------------------
# Noise generation utilities
# ---------------------------------------------------------------------------
FAKE_IPS = [f"10.10.{a}.{b}" for a in range(1, 20) for b in range(1, 20)]
FAKE_PROCESSES = [
    "chrome.exe",
    "outlook.exe",
    "teams.exe",
    "svchost.exe",
    "explorer.exe",
    "firefox.exe",
    "slack.exe",
    "onenote.exe",
    "winword.exe",
    "excel.exe",
    "rpcnet.exe",
    "dllhost.exe",
    "taskhostw.exe",
    "sihost.exe",
    "backgroundTaskHost.exe",
]
FAKE_UPTIMES = [
    "3d 14h 22m",
    "7d 2h 8m",
    "12d 0h 45m",
    "1d 6h 11m",
    "45d 18h 30m",
    "0d 4h 55m",
    "21d 9h 12m",
    "2d 1h 3m",
]


def _pick_noise() -> str:
    """Return a short noise snippet to inject into user prompts."""
    ip = random.choice(FAKE_IPS)
    proc = random.choice(FAKE_PROCESSES)
    uptime = random.choice(FAKE_UPTIMES)
    return f"Seen at {ip} — processes: {proc} — uptime: {uptime}"


# ---------------------------------------------------------------------------
# Style generators
# ---------------------------------------------------------------------------


def _build_dashboard(
    os_str: str, priv: str, edr: str, objective: str, noise: str = ""
) -> str:
    """Formal bulleted 'dashboard' style."""
    lines = [
        "Engagement State:",
        f"- OS: {os_str}",
        f"- Privilege: {priv}",
        f"- EDR: {edr}",
        f"- Objective: {objective}",
    ]
    if noise:
        lines.append(f"- Context: {noise}")
    lines.append("")
    lines.append("What is the next tactical step, and which agent should handle it?")
    return "\n".join(lines)


def _build_operator(
    os_str: str, priv: str, edr: str, objective: str, noise: str = ""
) -> str:
    """Terse conversational 'operator' style — like a chat message."""
    edr_part = f", edr: {edr}" if edr != "None" else ", no edr"
    text = f"got {priv} on {os_str}{edr_part}. need to {objective.lower()}."
    if noise:
        text = f"{text} ({noise})"
    text += " what do we route to?"
    return text


def _build_beacon(
    os_str: str, priv: str, edr: str, objective: str, noise: str = ""
) -> str:
    """Raw JSON/structured 'beacon' style."""
    state: dict = {
        "os": os_str,
        "privilege": priv,
        "edr": edr,
        "objective": objective,
    }
    if noise:
        state["context"] = noise
    return (
        f"```json\n{json.dumps(state, indent=2)}\n```\n\n"
        "Route to the appropriate agent."
    )


STYLE_BUILDERS = {
    "dashboard": _build_dashboard,
    "operator": _build_operator,
    "beacon": _build_beacon,
}

# ---------------------------------------------------------------------------
# EDR-aware reasoning helpers
# ---------------------------------------------------------------------------

_EDR_EVASION_METHODS = {
    "CrowdStrike Falcon": [
        "CrowdStrike Falcon unhooking via syscall stubs",
        "Falcon sensor blind-spot using direct syscalls",
        "CrowdStrike bypass via process hollowing",
    ],
    "Microsoft Defender for Endpoint": [
        "MDE real-time protection disable and AMSI bypass",
        "Defender tamper protection bypass via BYOVD",
        "MDE unhooking with custom syscall stubs",
    ],
    "SentinelOne": [
        "SentinelOne agent unhooking via userland patching",
        "S1 evasion through indirect syscall execution",
    ],
    "Carbon Black": [
        "Carbon Black sensor bypass via kernel callback removal",
        "CB evasion with ETW patching",
    ],
    "None": [],  # No evasion needed
}


def _edr_note(edr: str) -> str:
    """Return a qualifying note about EDR presence for use in reasoning."""
    if edr == "None":
        return ""
    return f"{edr} is active. "


def _evasion_method(edr: str) -> str:
    """Return a specific evasion method for the given EDR."""
    methods = _EDR_EVASION_METHODS.get(edr, [])
    if not methods:
        return "EDR evasion and unhooking"
    return random.choice(methods)


# ---------------------------------------------------------------------------
# Build reasoning text with variation
# ---------------------------------------------------------------------------


def _build_reasoning(
    agent: str,
    scenario_reasoning: str,
    os_str: str,
    priv: str,
    edr: str,
    mechanism_pool_key: str | None = None,
) -> str:
    """Build a Chain-of-Thought reasoning string with linguistic variation.

    Uses a template from REASONING_TEMPLATES when a pool exists, otherwise
    falls back to the scenario's base reasoning.
    """
    templates = REASONING_TEMPLATES.get(agent, [])
    if not templates:
        return scenario_reasoning

    template = random.choice(templates)
    edr_note_str = _edr_note(edr)
    evasion_str = _evasion_method(edr)

    # Pick a mechanism from the pool
    pool_key = mechanism_pool_key or agent.lower().replace("agent", "")
    pool = MECHANISM_POOLS.get(pool_key, [])
    mechanism = random.choice(pool) if pool else "appropriate technique"

    return template.format(
        edr_note=edr_note_str,
        edr_name=edr,
        evasion_method=evasion_str,
        mechanism=mechanism,
        discovery_method=random.choice(MECHANISM_POOLS["discovery"]),
        movement_method=random.choice(MECHANISM_POOLS["movement"]),
        c2_method=random.choice(MECHANISM_POOLS["c2"]),
        priv_method=random.choice(MECHANISM_POOLS["privilege"]),
        priv_level=priv,
        cred_method=random.choice(MECHANISM_POOLS["credential"]),
        exec_method=random.choice(MECHANISM_POOLS["execution"]),
    )


# ---------------------------------------------------------------------------
# Build a single training row
# ---------------------------------------------------------------------------


def _build_row(
    os_str: str,
    priv: str,
    edr: str,
    objective: str,
    agent: str,
    reasoning: str,
    style: str,
    noise: str = "",
    next_objective: str | None = None,
) -> dict:
    """Build one JSONL row with system/user/assistant messages."""
    style_fn = STYLE_BUILDERS[style]
    user_content = style_fn(os_str, priv, edr, objective, noise)

    # Build context for the routing decision
    context: dict = {
        "target": os_str,
        "privilege": priv,
        "edr": edr,
        "next_objective": next_objective or objective,
    }

    routing = {"agent": agent, "context": context}
    assistant_content = (
        f"**Chain of Thought:** {reasoning}\n\n"
        f"**Routing Decision:**\n"
        f"```json\n{json.dumps(routing, indent=2)}\n```"
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------


def generate_dataset() -> list[dict]:
    """Generate the full Orchestrator training dataset.

    Returns:
        List of message dicts ready for JSONL serialization.
    """
    rows: list[dict] = []

    # ------------------------------------------------------------------
    # Phase 1: Scenarios × Environments × Styles (10×10×3 = 300 rows)
    # ------------------------------------------------------------------
    for objective, agent, base_reasoning in SCENARIOS:
        for os_str, priv, edr in ENVIRONMENTS:
            for style_name in STYLE_BUILDERS:
                reasoning = _build_reasoning(
                    agent,
                    base_reasoning,
                    os_str,
                    priv,
                    edr,
                )
                rows.append(
                    _build_row(
                        os_str,
                        priv,
                        edr,
                        objective,
                        agent,
                        reasoning,
                        style_name,
                    )
                )

    # ------------------------------------------------------------------
    # Phase 2: Multi-step scenarios × Environments × Styles (50+ rows)
    # ------------------------------------------------------------------
    # Pick a deterministic subset of environment×style pairs to hit ~50 rows
    multistep_envs = [
        ("Windows 10 22H2", "Standard User", "Microsoft Defender for Endpoint"),
        ("Windows Server 2019", "Administrator", "CrowdStrike Falcon"),
        ("Windows Server 2022", "SYSTEM", "None"),
        ("Windows 11", "Local Admin", "SentinelOne"),
        ("Ubuntu 22.04", "www-data", "None"),
    ]
    multistep_styles = ["dashboard", "operator", "beacon"]

    for i, (objective, agent, full_reasoning, next_obj) in enumerate(
        MULTISTEP_SCENARIOS
    ):
        # Cycle through env/style pairs
        env = multistep_envs[i % len(multistep_envs)]
        style = multistep_styles[i % len(multistep_styles)]
        os_str, priv, edr = env

        reasoning = _build_reasoning(
            agent,
            full_reasoning,
            os_str,
            priv,
            edr,
        )
        rows.append(
            _build_row(
                os_str,
                priv,
                edr,
                objective,
                agent,
                reasoning,
                style,
                next_objective=next_obj,
            )
        )

    # Second pass to fill remaining — cycle again with shifted envs
    for i, (objective, agent, full_reasoning, next_obj) in enumerate(
        MULTISTEP_SCENARIOS
    ):
        env = multistep_envs[(i + 3) % len(multistep_envs)]
        style = multistep_styles[(i + 1) % len(multistep_styles)]
        os_str, priv, edr = env

        reasoning = _build_reasoning(
            agent,
            full_reasoning,
            os_str,
            priv,
            edr,
        )
        rows.append(
            _build_row(
                os_str,
                priv,
                edr,
                objective,
                agent,
                reasoning,
                style,
                next_objective=next_obj,
            )
        )

    # Third pass — fill to 50
    needed = 50 - len(rows) + 300  # offset by base rows
    if needed > 0:
        for i in range(needed):
            idx = i % len(MULTISTEP_SCENARIOS)
            objective, agent, full_reasoning, next_obj = MULTISTEP_SCENARIOS[idx]
            env = multistep_envs[i % len(multistep_envs)]
            style = multistep_styles[i % len(multistep_styles)]
            os_str, priv, edr = env

            reasoning = _build_reasoning(
                agent,
                full_reasoning,
                os_str,
                priv,
                edr,
            )
            rows.append(
                _build_row(
                    os_str,
                    priv,
                    edr,
                    objective,
                    agent,
                    reasoning,
                    style,
                    next_objective=next_obj,
                )
            )

    # ------------------------------------------------------------------
    # Phase 3: Noise rows (30 rows with irrelevant details)
    # ------------------------------------------------------------------
    noise_agents = [s[1] for s in SCENARIOS]
    noise_objectives = [s[0] for s in SCENARIOS]
    noise_reasonings = [s[2] for s in SCENARIOS]

    for i in range(30):
        idx = i % len(SCENARIOS)
        objective = noise_objectives[idx]
        agent = noise_agents[idx]
        base_reasoning = noise_reasonings[idx]

        env = ENVIRONMENTS[i % len(ENVIRONMENTS)]
        style = random.choice(list(STYLE_BUILDERS.keys()))
        os_str, priv, edr = env

        noise = _pick_noise()
        reasoning = _build_reasoning(
            agent,
            base_reasoning,
            os_str,
            priv,
            edr,
        )
        rows.append(
            _build_row(
                os_str,
                priv,
                edr,
                objective,
                agent,
                reasoning,
                style,
                noise=noise,
            )
        )

    # ------------------------------------------------------------------
    # Shuffle
    # ------------------------------------------------------------------
    random.shuffle(rows)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Orchestrator routing training data (JSONL).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Custom output path for the JSONL file. "
        "Defaults to data/datasets/orchestrator_dataset.jsonl "
        "relative to the script's parent directory.",
    )
    args = parser.parse_args()

    # Determine output path
    project_root = Path(__file__).resolve().parent.parent
    default_output = project_root / "data" / "datasets" / "orchestrator_dataset.jsonl"
    output_path = Path(args.output) if args.output else default_output

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate
    rows = generate_dataset()

    # Write
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # Statistics
    agent_counts = Counter()
    for row in rows:
        # Extract agent from the assistant message's routing JSON
        assistant_msg = row["messages"][2]["content"]
        # Parse between ```json\n and \n```
        json_start = assistant_msg.find("```json\n") + 7
        json_end = assistant_msg.find("\n```", json_start)
        routing = json.loads(assistant_msg[json_start:json_end])
        agent_counts[routing["agent"]] += 1

    total = len(rows)
    print(f"Total rows generated: {total}")
    print(f"Breakdown by agent routed to:")
    for agent_name, count in sorted(agent_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"  {agent_name}: {count} ({pct:.1f}%)")
    print(f"\nOutput written to: {output_path}")


if __name__ == "__main__":
    main()
