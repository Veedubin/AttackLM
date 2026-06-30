# AttackLM
AttackLM — QLoRA fine-tuning pipeline for a MITRE ATT&CK-grounded security AI assistant. 24,652 training pairs · Qwen2.5-Coder base · 16GB+ VRAM.

[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/attacklm.svg?label=version&color=blue)](https://pypi.org/project/attacklm/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](requirements.txt)

---

## GUI (New in v0.7.1)

AttackLM now includes a terminal-based GUI for all commands. No more memorizing 40+ CLI flags.

```bash
pip install attacklm-gui
attacklm-gui
```

Features:
- **Training form** with 40+ params in 5 tabs (Basic, LoRA, GaLore, Advanced, Hardware)
- **Live training monitor** with loss sparkline, VRAM gauge, progress bar, and log viewer
- **One-click screens** for extract, balance, infer, merge, build, pipeline, and init
- **5 built-in presets** (3B Q-GaLore, 3B LoRA, 7B Q-GaLore, 7B QLoRA, etc.)
- **Pause/resume/quit** controls during training
- Works in **terminal-only** environments — no X11, no browser, no GPU required. Works over SSH and WSL.

The CLI still works exactly as before. The GUI is a thin wrapper that constructs and runs CLI commands.

---

## Install
The recommended way to install the full CUDA training stack:

```bash
pip install attacklm[all]
```

*Alternative:* Use `uv pip install "attacklm[all]"` for faster installation.
*Note:* `flash-attn` is optional and not included by default to avoid heavy compilation requirements.

---

## Init
Initialize the dataset by cloning upstream repositories, extracting data, adding attribution, and organizing into buckets:

```bash
attacklm-init --yes
```

---

## Balance (Optional)
Because Metasploit accounts for ~64% of the raw data, balancing is recommended to prevent overfitting and ensure broad tactical coverage.

```bash
attacklm-balance --profile 7b-16gb --output data/datasets/balanced/balanced.jsonl
```

*   **Presets:** Use `--preset red-team`, `purple-team`, or `blue-team` to control the offensive/defensive mix.
*   **Profiles:** Use profiles like `3b-16gb` or `7b-16gb` to automatically set per-bucket caps based on your VRAM.

---

## Train
Train the model using the orchestrated pipeline. The default base model is `Qwen/Qwen2.5-Coder-3B-Instruct`.

```bash
attacklm-train-all --single-model --dataset all --epochs 5 --max-length 2048
```

### Key Training Flags
| Flag | Default | Description |
|------|---------|-------------|
| `--single-model` | off | Train one model on all buckets combined |
| `--dataset` | none | Path to dataset or alias (`all`, `base/`, `tools/`, `ai/`, `orchestrator`) |
| `--epochs` | 10 | Total training epochs |
| `--max-length` | 1024 | Max sequence length (use 2048 for richer context) |
| `--lora-r` | 16 | LoRA rank |
| `--use-galore` | off | Use Q-GaLore for full-parameter training on low VRAM |
| `--spectrum` | off | SNR-based layer freezing to reduce VRAM |

**Multi-round SFT:** AttackLM supports iterative training. You can train on tactics first, then tools, then a final general pass. Each round automatically backs up the previous state and uses the merged weights of the prior run as the new base.

---

## Build
Perform a one-shot merge of the LoRA adapter and conversion to GGUF format for local deployment.

```bash
attacklm-build --adapter models/attacklm-single_TIMESTAMP --name attacklm
```

---

## Infer
Smoke-test your trained adapter with a set of representative security prompts.

```bash
attacklm-infer --adapter models/attacklm-single_TIMESTAMP
```

---

## Bucket Reference Table
The dataset is split into buckets to allow for granular control over training composition.

| Bucket | Pairs | Category | Description |
|--------|-------|----------|-------------|
| base/collection | 634 | MITRE Tactic | TA0009 - Collection techniques |
| base/command_and_control | 0 | MITRE Tactic | TA0011 - C2 techniques (no data yet) |
| base/credential_access | 589 | MITRE Tactic | TA0006 - Credential access |
| base/defense_evasion | 1,375 | MITRE Tactic | TA0005 - Defense evasion |
| base/discovery | 1,846 | MITRE Tactic | TA0007 - Discovery |
| base/execution | 767 | MITRE Tactic | TA0002 - Execution |
| base/exfiltration | 53 | MITRE Tactic | TA0010 - Exfiltration |
| base/lateral_movement | 252 | MITRE Tactic | TA0008 - Lateral movement |
| base/persistence | 1,120 | MITRE Tactic | TA0003 - Persistence |
| base/privilege_escalation | 537 | MITRE Tactic | TA0004 - Privilege escalation |
| tools/metasploit | 8,349 | Tools | Metasploit module knowledge |
| ai/jailbreaking | 50 | AI Security | Jailbreak techniques (garak) |
| ai/prompt-injection | 63 | AI Security | Prompt injection (promptfoo, promptmap) |
| orchestrator | 380 | Meta | Agent routing decisions |
| cloud/attacks | 10 | Extended | Cloud attack techniques |
| ics/attacks | 290 | Extended | ICS/SCADA attacks |
| social_engineering/phishing | 440 | Extended | Phishing techniques |
| wireless/attacks | 197 | Extended | Wireless attacks |
| defensive/detection_engineering | 7,154 | Defensive | Sigma + Elastic + Splunk detection rules |
| defensive/threat_hunting | 366 | Defensive | Mordor + ThreatHunter playbooks |
| defensive/incident_response | 168 | Defensive | NIST SP 800-61r3 IR procedures |

*Note: 6 defensive buckets (detection_engineering, threat_hunting, incident_response, plus sigma/elastic/splunk/mordor/threathunter/nist sources) are planned but have 0 records currently.*

---

## Data Sources
| Source | Pairs | License |
|--------|-------|---------|
| Metasploit Framework | 13,997 | BSD-3-Clause |
| Atomic Red Team | 1,115 | MIT |
| MITRE Caldera/Stockpile | 390 | Apache-2.0 |
| LLM-generated | 937 | GPL-3.0 |
| NVIDIA Garak / Promptfoo | 113 | Mixed MIT/Apache-2.0 |
| Sigma | 3,132 | DRL-1.1 |
| Elastic | 1,908 | Elastic-2.0 |
| Splunk | 2,114 | Apache-2.0 |
| Mordor | 339 | Apache-2.0 |
| ThreatHunter | 27 | Apache-2.0 |
| NIST IR | 168 | Public Domain |

---

## Architecture
Data is organized in a per-source hierarchy to ensure provenance and attribution.

```text
AttackLM/
├── data/
│   └── datasets/
│       └── buckets/
│           └── sources/
│               └── <source>/
│                   └── <bucket>/
│                       └── <tactic>/
│                           └── data.jsonl
```

This layout allows the pipeline to deterministically extract data from upstream sources without introducing hallucinations or API dependencies.

---

## CLI Reference
| Command | Description |
|---------|-------------|
| `attacklm-gui` | Terminal GUI for all commands (new in v0.7.1) |
| `attacklm-init` | One-shot init: clone → extract → attribute → buckets |
| `attacklm-train` | Train a single QLoRA adapter |
| `attacklm-train-all` | Orchestrated training for all buckets / HPO |
| `attacklm-balance` | Build a balanced subset of buckets |
| `attacklm-hpo` | Coordinate-descent HPO sweep |
| `attacklm-infer` | Smoke-test inference |
| `attacklm-merge` | Merge LoRA adapter into base model |
| `attacklm-gguf` | Convert merged model to GGUF (llama.cpp) |
| `attacklm-build` | One-shot merge → GGUF → install |
| `attacklm-demo` | Multi-agent orchestrator demo |
| `attacklm-extract` | Extract data from cloned repositories |
| `attacklm-buckets` | Organize data into bucket structure |
| `attacklm-attribute` | Add source and license to JSONL records |
| `attacklm-clone` | Clone upstream data repositories |
| `attacklm-train-lora` | Direct LoRA training on single dataset |
| `attacklm-eval` | Run retention evaluation suite |
| `attacklm-collect-ref` | Collect reference model outputs |
| `attacklm-score` | Score candidate models against reference |
| `attacklm-compare` | Compare multiple candidate model scores |
| `attacklm-golden` | Execute golden vector regression gates |
| `attacklm-register-ollama`| Register GGUF model with Ollama |

---

## License
The code in this repository is licensed under the **MIT License**. Training data consists of mixed licenses per source; see [ATTRIBUTION.md](ATTRIBUTION.md) for the full mapping.

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to AttackLM.

## Changelog
See [CHANGELOG.md](CHANGELOG.md) for the full version history.
