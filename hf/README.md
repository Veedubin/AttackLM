---
license: gpl-3.0
task_categories:
  - text-generation
  - text2text-generation
language:
  - en
tags:
  - cybersecurity
  - red-team
  - mitre-attck
  - instruction-following
  - adversarial-ml
  - penetration-testing
  - security
  - synthetic
size_categories:
  - 10K<n<100K
---

# AttackLM — Balanced Red-Team Training Dataset

## Dataset Summary

AttackLM is a **MITRE ATT&CK-grounded instruction-following dataset** for training red-team cybersecurity LLMs. It contains **17,616 conversation triples** (system/user/assistant) spanning **23 buckets** across **12+ attack categories**, covering the full adversary kill chain from initial access through exfiltration.

Every training pair is **deterministically extracted or generated** — no LLM-in-the-loop data pipeline, no hallucinated content, no API costs. The data is sourced from openly licensed projects (Atomic Red Team, MITRE Caldera, Metasploit Framework, Sigma, and others) and augmented with procedurally generated synthetic data for underrepresented categories.

### Key Features

- **17,616 instruction-following triples** in OpenAI chat format
- **23 purpose-built buckets** across 12+ cybersecurity attack categories
- **MITRE ATT&CK technique IDs** on every record (e.g., T1059.001, T1566.002)
- **Deterministic pipeline** — no LLM hallucinations in data generation
- **Per-record attribution** — source and license tracked per bucket
- **90/10 train/test split** provided (pre-built)

## Supported Tasks

- **Instruction-following** — Train LLMs to respond as red-team specialists
- **Red-team cybersecurity training** — Adversary emulation, tool usage, technique explanation
- **MITRE ATT&CK mapping** — Map user queries to specific ATT&CK techniques
- **Agent routing** — Orchestrator bucket trains multi-agent routing decisions
- **Blue team detection** — Defense-oriented analysis of offensive techniques
- **AI security** — Prompt injection and jailbreak pattern recognition

## Languages

English only.

## Dataset Structure

### Data Instances

Each record is a conversation triple with MITRE metadata:

```json
{
  "messages": [
    {"role": "system", "content": "You are an authorized Red Team specialist..."},
    {"role": "user", "content": "Show me T1560 (Archive Collected Data) for powershell..."},
    {"role": "assistant", "content": "**Technique: Archive Collected Data (T1560)**\n\n**Command (powershell):**\n```\ndir #{input_file} -Recurse | Compress-Archive -DestinationPath #{output_file}\n```\n..."}
  ],
  "mitre_ids": ["T1560"],
  "source": "atomic-red-team",
  "license": "MIT",
  "bucket": "base/collection",
  "category": "tactic"
}
```

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `messages` | `list[dict]` | Conversation triple with `role` (system/user/assistant) and `content` (string) |
| `mitre_ids` | `list[string]` | MITRE ATT&CK technique IDs (e.g., `["T1059.001", "T1566.002"]`) |
| `source` | `string` | Upstream data source name (e.g., `"atomic-red-team"`, `"metasploit"`) |
| `license` | `string` | License of the upstream source (e.g., `"MIT"`, `"AGPL-3.0"`) |
| `bucket` | `string` | Bucket path (e.g., `"base/collection"`, `"tools/metasploit"`) |
| `category` | `string` | Bucket category: `"tactic"`, `"tools"`, `"meta"`, `"ai_redteam"`, `"cloud"`, etc. |

### Data Splits

| Split | Examples |
|-------|---------:|
| `train` | 15,846 (90%) |
| `test` | 1,770 (10%) |

The split is **stratified by bucket** to ensure every attack category is represented in both splits.

## Bucket Overview

AttackLM organizes data into **23 buckets** across 12+ categories:

### MITRE Tactic Buckets (10)

| Bucket | Pairs | MITRE Tactic | Upstream Sources |
|--------|------:|--------------|------------------|
| `base/collection` | 634 | TA0009 | Atomic Red Team, Caldera, Metasploit |
| `base/command_and_control` | 105 | TA0011 | Atomic Red Team, Caldera, Metasploit |
| `base/credential_access` | 589 | TA0006 | Atomic Red Team, Caldera, Infection Monkey, Metasploit |
| `base/defense_evasion` | 1,375 | TA0005 | Atomic Red Team, Caldera, Metasploit, RTA |
| `base/discovery` | 1,846 | TA0007 | Atomic Red Team, Caldera, Infection Monkey, Metasploit, RTA |
| `base/execution` | 767 | TA0002 | Atomic Red Team, Caldera, Infection Monkey, Metasploit, RTA |
| `base/exfiltration` | 173 | TA0010 | Atomic Red Team, Caldera, Metasploit |
| `base/lateral_movement` | 252 | TA0008 | Atomic Red Team, Caldera, Infection Monkey, Metasploit |
| `base/persistence` | 1,120 | TA0003 | Atomic Red Team, Caldera, Infection Monkey, Metasploit, RTA |
| `base/privilege_escalation` | 537 | TA0004 | Atomic Red Team, Caldera, Metasploit |

### Tool-Specific Buckets (3)

| Bucket | Pairs | Upstream Source | License |
|--------|------:|-----------------|---------|
| `tools/metasploit` | 8,349 | [rapid7/metasploit-framework](https://github.com/rapid7/metasploit-framework) | BSD-3-Clause |
| `tools/rta` | 76 | [endgameinc/RTA](https://github.com/endgameinc/RTA) | **AGPL-3.0** ⚠️ |
| `tools/infection_monkey` | 36 | [guardicore/monkey](https://github.com/guardicore/monkey) | GPL-3.0 |

### AI Security Buckets (2)

| Bucket | Pairs | Upstream Sources |
|--------|------:|------------------|
| `ai/prompt-injection` | 687 | promptfoo, promptmap, synthetic |
| `ai/jailbreaking` | 56 | garak, PyRIT, FuzzyAI, TheBigPromptLibrary |

### Orchestrator Bucket (1)

| Bucket | Pairs | Source |
|--------|------:|--------|
| `orchestrator` | 380 | Synthetic (procedural) |

### Extended Category Buckets (7)

| Bucket | Pairs | MITRE Tactic | Description |
|--------|------:|--------------|-------------|
| `attack_tactics/red_team_tactics` | 142 | TA0000 | 104 MITRE techniques, synthetic + HF source |
| `cloud/attacks` | 94 | TA0008 | Cloud: IAM, S3, containers, K8s, serverless, IMDS |
| `ics/attacks` | 80 | TA0010* | ICS/SCADA: Modbus, PLC, SCADA, industrial ransomware |
| `wireless/attacks` | 84 | TA0006 | WPA2/WPA3, deauth, rogue AP, Bluetooth |
| `supply_chain/attacks` | 80 | TA0042 | Dependency confusion, typosquatting, CI/CD compromise |
| `web_app/attacks` | 77 | TA0001 | SQL injection, XSS, CSRF, path traversal, IDOR, SSRF |
| `social_engineering/phishing` | 77 | TA0001 | Spear phishing, BEC, vishing, deepfake SE |

**Total: 17,616 pairs**

## Source Data

### Upstream Projects

| Source | Pairs | License | Repository |
|--------|------:|---------|------------|
| Atomic Red Team | 2,506 | MIT | [redcanaryco/atomic-red-team](https://github.com/redcanaryco/atomic-red-team) |
| MITRE Caldera / Stockpile | 608 | Apache-2.0 | [mitre/stockpile](https://github.com/mitre/stockpile) |
| Caldera plugins (arsenal/manx/access) | 56 | Apache-2.0 | [mitre/caldera](https://github.com/mitre/caldera) |
| Metasploit Framework | 8,349 | BSD-3-Clause | [rapid7/metasploit-framework](https://github.com/rapid7/metasploit-framework) |
| Infection Monkey | 36 | GPL-3.0 | [guardicore/monkey](https://github.com/guardicore/monkey) |
| RTA — Red Team Automation | 76 | **AGPL-3.0** ⚠️ | [endgameinc/RTA](https://github.com/endgameinc/RTA) |
| Sigma rules | (labels) | DRL-1.1 | [SigmaHQ/sigma](https://github.com/SigmaHQ/sigma) |
| AI-security tools (promptfoo, garak, promptmap, PyRIT, FuzzyAI, TheBigPromptLibrary) | 743+ | mixed MIT/Apache-2.0 | various (see [ATTRIBUTION.md](https://github.com/Veedubin/AttackLM/blob/main/ATTRIBUTION.md)) |
| Synthetic (orchestrator + prompt injection + extended categories) | 1,596 | MIT | this repo |

Full per-source attribution in the [upstream ATTRIBUTION.md](https://github.com/Veedubin/AttackLM/blob/main/ATTRIBUTION.md).

### Data Collection and Processing

1. **Extraction**: Each upstream source has a dedicated extractor script (`scripts/extract_*.py`, `scripts/acquire_*.py`) that transforms raw data into the `messages` triple format
2. **Attribution**: `scripts/augment_attribution.py` adds `source` and `license` fields to each record
3. **Bucketing**: `scripts/setup_buckets.py` and `scripts/reorganize_buckets.py` organize data into MITRE-aligned buckets
4. **Balancing**: `scripts/balance_buckets.py` provides stratified sampling for hardware-constrained training
5. **HF build**: `hf/scripts/build_hf_dataset.py` combines all buckets into a single HF-compatible dataset with train/test split

No LLM is used in the data pipeline — all transformations are deterministic.

## Personal and Sensitive Information

- **No real credentials, API keys, or PII** are present in the dataset
- All data is either **synthetically generated** (procedural) or **extracted from publicly available open-source projects** whose content is already public
- Technique descriptions reference **public CVEs and MITRE ATT&CK IDs** only
- The `tools/metasploit` bucket contains **module descriptions from the public Metasploit Framework** — these reference CVEs and exploitation techniques but do not contain exploit payloads

## Bias, Risks, and Limitations

### ⚠️ CRITICAL SAFETY NOTICE

**This dataset contains offensive cybersecurity techniques for authorized red-team training only.** It includes:

- Exploitation techniques (CVE references, attack commands)
- Privilege escalation methods
- Lateral movement strategies
- Defense evasion tactics
- Social engineering and phishing templates
- AI jailbreak and prompt injection patterns
- ICS/SCADA attack procedures

**Out-of-Scope Use (DO NOT):**
- Launch unauthorized attacks against systems you do not own or have explicit authorization to test
- Develop malware or exploits for malicious purposes
- Use social engineering templates against unconsenting individuals
- Circumvent safety measures on AI systems you don't own
- Violate local, national, or international cybersecurity laws

**Intended Use:**
- Authorized penetration testing training
- Cybersecurity education and certification preparation
- Blue team detection engineering (understanding attacks to build defenses)
- AI safety research (understanding adversarial inputs to build robust models)
- Security tool development and testing

### Biases

- **Tool skew**: Metasploit accounts for ~47% of the dataset (8,349/17,616 pairs), potentially over-weighting `msfconsole` syntax
- **MITRE coverage**: Some techniques are better represented than others; Atomic Red Team and Metasploit coverage varies by tactic
- **Language**: English only — may not generalize to multi-lingual security contexts
- **Recency**: Dataset reflects upstream source versions as of extraction date; new CVEs and techniques may not be covered
- **Synthetic quality**: Extended category buckets (cloud, ICS, wireless, etc.) are procedurally generated and may lack the depth of curated sources

### Limitations

- The dataset teaches **what attacks look like**, not how to execute them in practice against live systems
- No real exploit code or payloads are included — only descriptions and command patterns
- The orchestrator bucket uses a **deterministic routing logic** that may not reflect real-world adversary decision-making
- The AI security buckets (prompt injection, jailbreaking) represent a **subset** of known techniques and may not generalize

## Licensing

### Dataset License: GPL-3.0

This dataset is released under **GPL-3.0** because it includes:

- **GPL-3.0** content from [Infection Monkey](https://github.com/guardicore/monkey) (36 pairs)
- **AGPL-3.0** content from [RTA](https://github.com/endgameinc/RTA) (76 pairs)

The GPL-3.0 license covers the **dataset as a whole** as a collective work. Individual records retain their original licenses as indicated in the `license` field:

| License | Pairs | Applies To |
|---------|------:|------------|
| MIT | ~2,506 | Atomic Red Team, orchestrator, synthetic |
| Apache-2.0 | ~1,965 | Caldera, garak, FuzzyAI, Sigma (labels) |
| BSD-3-Clause | ~8,349 | Metasploit Framework |
| GPL-3.0 | ~36 | Infection Monkey |
| AGPL-3.0 | ~76 | RTA |
| Mixed MIT/MPL | ~varies | TheBigPromptLibrary |
| DRL-1.1 | (labels) | Sigma rules |

### ⚠️ AGPL-3.0 Network Distribution Note

RTA is licensed under AGPL-3.0, which requires providing source code to users interacting with the work over a network. If you host a model trained on this dataset as a service, you may need to provide the RTA-derived training data and scripts. The [public AttackLM repository](https://github.com/Veedubin/AttackLM) satisfies this requirement. To create an AGPL-clean version, remove the `tools/rta` bucket (76 pairs, 0.4% of the dataset).

### Code License

The code, scripts, and pipeline in the [AttackLM repository](https://github.com/Veedubin/AttackLM) are under the **MIT License**. Only the training data carries GPL/AGPL obligations.

## How to Use

### Loading with 🤗 Datasets

```python
from datasets import load_dataset

# Load from HuggingFace Hub
dataset = load_dataset("neuralgentics/attacklm")

# Access splits
train = dataset["train"]  # 15,846 examples
test = dataset["test"]    # 1,770 examples

# Filter by MITRE tactic
privilege_escalation = train.filter(
    lambda x: "TA0004" in x.get("mitre_ids", [])
)

# Filter by source
metasploit_only = train.filter(
    lambda x: x["source"] == "metasploit"
)

# Filter by bucket
cloud_attacks = train.filter(
    lambda x: x["bucket"] == "cloud/attacks"
)
```

### Loading from Local Files

```python
from datasets import load_dataset

# If you cloned the repo
dataset = load_dataset(
    "json",
    data_files={
        "train": "hf/data/attacklm-train.jsonl",
        "test": "hf/data/attacklm-test.jsonl"
    },
    split="train"
)
```

### Using with Transformers

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")

def format_example(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )

formatted = dataset["train"].map(
    lambda x: {"text": format_example(x)}
)
```

### Building from Source

```bash
# Clone the AttackLM repo
git clone https://github.com/Veedubin/AttackLM.git
cd AttackLM

# Build the HF dataset
python hf/scripts/build_hf_dataset.py --output hf/data

# This creates:
#   hf/data/attacklm-train.jsonl  (15,846 examples)
#   hf/data/attacklm-test.jsonl   (1,770 examples)
#   hf/data/dataset_infos.json
```

## Dataset Creation

### Curation Rationale

AttackLM was created to address the lack of a comprehensive, MITRE ATT&CK-grounded red-team training dataset for LLMs. Existing datasets either:
- Cover a narrow subset of techniques (e.g., only web app attacks)
- Lack MITRE technique IDs for structured evaluation
- Are generated by LLMs (introducing hallucination risk)
- Don't cover AI-specific attack vectors (prompt injection, jailbreaking)

AttackLM combines **curated upstream data** (Atomic Red Team, Metasploit, Caldera) with **procedurally generated synthetic data** for underrepresented categories (cloud, ICS, wireless, supply chain, social engineering).

### Annotation Process

- All MITRE ATT&CK technique IDs are derived from **upstream project metadata** (not manually annotated)
- Source attribution is tracked per-bucket via `metadata.json` files
- No human annotators were used — all transformations are programmatic

## Additional Information

### Dataset Curators

AttackLM is developed by [jcharles](https://github.com/Veedubin) and contributors.

### Funding

Community-driven, no institutional funding.

### Citation

```bibtex
@misc{attacklm2026,
  title={AttackLM: A Balanced Red-Team Training Dataset Grounded in MITRE ATT&CK},
  author={jcharles},
  year={2026},
  url={https://github.com/Veedubin/AttackLM},
  note={Dataset available at https://huggingface.co/datasets/neuralgentics/attacklm}
}
```

### Contributions

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/Veedubin/AttackLM/blob/main/CONTRIBUTING.md) for guidelines on adding new buckets, extractors, and improvements.

---

**Disclaimer**: This dataset is provided for authorized cybersecurity research and education only. The authors assume no liability for misuse. Always obtain proper authorization before testing any technique against systems you do not own.