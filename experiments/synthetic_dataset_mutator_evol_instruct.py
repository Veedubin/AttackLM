import json
import os
from openai import OpenAI

# Point this to your local LLM server (Ollama, LMStudio, vLLM)
# This keeps your bank's potential data/tradecraft entirely offline.
client = OpenAI(
    base_url="http://localhost:1234/v1", 
    api_key="not-needed"
)

OUTPUT_FILE = "qwen_orchestrator_diverse.jsonl"

# We define the raw *logic* here, not the phrasing.
SEEDS = [
    {
        "os": "Windows Server 2022",
        "privilege": "SYSTEM",
        "edr": "Defender for Endpoint",
        "objective": "Dump credentials from LSASS",
        "correct_agent": "DefenseEvasionAgent",
        "reasoning": "We have SYSTEM, but MDE is active. Attempting to touch LSASS directly will trigger an alert and burn the beacon. We must route to DefenseEvasionAgent first to blind MDE or unhook the APIs."
    },
    {
        "os": "Ubuntu 22.04",
        "privilege": "standard user (www-data)",
        "edr": "None",
        "objective": "Read the /etc/shadow file",
        "correct_agent": "PrivilegeEscalationAgent",
        "reasoning": "We are www-data on a Linux box. We cannot read /etc/shadow without root. No EDR is present, so we can route directly to PrivilegeEscalationAgent to hunt for SUID binaries or kernel exploits."
    }
    # You would define 50-100 of these core logical truths.
]

def mutate_scenario(seed, mutation_style):
    """
    Asks the 'Teacher' LLM to rewrite the scenario in a highly specific, messy format.
    """
    system_prompt = (
        "You are an expert Red Team scenario generator. Your job is to take a core tactical "
        "engagement state and rewrite it into a specific format to train another AI. "
        "Do not change the core facts (OS, Privileges, EDR, Objective), but completely change the "
        "linguistic phrasing, add realistic noise (like irrelevant running processes or IPs), "
        "and adopt the requested persona."
    )

    user_prompt = f"""
    Core Facts:
    - OS: {seed['os']}
    - Privilege: {seed['privilege']}
    - EDR: {seed['edr']}
    - Objective: {seed['objective']}

    Rewrite this state update using this exact style: {mutation_style}.
    Provide ONLY the rewritten state update text, nothing else.
    """

    response = client.chat.completions.create(
        model="local-model", # Replace with your local model name
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.9 # High temperature = high variance and creativity
    )
    
    return response.choices[0].message.content.strip()

def generate_dataset():
    dataset = []
    
    # We force the Teacher model to generate drastically different structures
    mutation_styles = [
        "A messy Slack message from a stressed junior operator to the senior lead.",
        "A raw, sterile JSON block representing beacon metadata.",
        "A formal, bulleted brief from a C2 dashboard.",
        "A highly technical description including random (but realistic) IP addresses, irrelevant running processes (like chrome.exe, spotify.exe), and uptime.",
        "A terse, one-sentence update from a lazy operator."
    ]

    print(f"Starting Evol-Instruct pipeline. Mutating {len(SEEDS)} seeds across {len(mutation_styles)} styles...")

    for seed in SEEDS:
        for style in mutation_styles:
            print(f"Generating mutation: {style[:30]}...")
            
            try:
                # 1. The Teacher model generates a highly unique, noisy state string
                messy_state = mutate_scenario(seed, style)
                
                # 2. We pair the messy state with the strict, programmatic Orchestrator response
                user_prompt = f"Current State:\n{messy_state}\n\nWhat is the next tactical step, and which agent should handle it?"
                
                tool_call = {
                    "agent": seed['correct_agent'],
                    "context": f"Target: {seed['os']}, Context: {seed['objective']}"
                }
                
                assistant_response = (
                    f"**Chain of Thought:**\n{seed['reasoning']}\n\n"
                    f"**Agent Invocation:**\n```json\n{json.dumps(tool_call, indent=2)}\n```"
                )

                dataset.append({
                    "messages": [
                        {"role": "system", "content": "You are an Autonomous Red Team Orchestrator..."},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant_response}
                    ]
                })
            except Exception as e:
                print(f"API Error during mutation: {e}")
                continue

    return dataset

if __name__ == "__main__":
    robust_dataset = generate_dataset()
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for row in robust_dataset:
            f.write(json.dumps(row) + '\n')
            
    print(f"Done. Generated {len(robust_dataset)} highly diverse training rows.")