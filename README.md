# AttackLM

> A QLoRA fine-tuning pipeline for a MITRE ATT&CK-grounded red-team AI assistant.
> 16,982 training pairs · 3B–70B Qwen base · 16GB–128GB VRAM.

[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Training data: mixed](https://img.shields.io/badge/data-mixed%20%28see%20ATTRIBUTION%29-orange.svg)](ATTRIBUTION.md)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](requirements.txt)
[![Model: 3B-7B Qwen2.5](https://img.shields.io/badge/base%20model-Qwen2.5--Coder--3B--Instruct-green.svg)](https://huggingface.co/unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit)

---

## What is this?

AttackLM is a complete pipeline for fine-tuning a small language model to be
a competent red-team / AI-security assistant. The training data is grounded in
**MITRE ATT&CK** techniques, sourced from openly licensed open-source projects
(Atomic Red Team, MITRE Caldera, Metasploit, Sigma, Infection Monkey, RTA,
plus prompt-injection and jailbreak corpora for AI-security coverage).

The pipeline ingests 10 MITRE tactic buckets plus 6 specialized buckets
(orchestrator routing, AI-model attacks, security tooling) and produces a
QLoRA LoRA adapter you can drop on top of `Qwen2.5-Coder-3B-Instruct`.

What makes it different:
- **No LLM in the data pipeline.** Every training pair is deterministically
  extracted from upstream sources — no hallucinated content, no API costs.
- **Coordinate-descent HPO** built in. Sweeps `lora_r` (8→512) and
  `lora_dropout` (0→0.5) and picks the winner before final training.
- **16GB → 128GB VRAM friendly.** 3B QLoRA at `--max-length 2048` fits
  a 4080 SUPER. 70B+ on a 128GB card with packing.

---

## Data Source Attribution

**All training data is a transformation of openly licensed open-source
projects.** We do not claim authorship of any technique, command, module,
or rule — the original authors do. Each upstream repo, its license, and
its contribution to AttackLM's training mix is documented in
[**`/ATTRIBUTION.md`**](ATTRIBUTION.md) and summarized in
[**`/NOTICE`**](NOTICE).

The full per-source map:

| Source | Pairs | License | Repository |
|---|---:|---|---|
| Atomic Red Team | 2,506 | MIT | [redcanaryco/atomic-red-team](https://github.com/redcanaryco/atomic-red-team) |
| MITRE Caldera / Stockpile | 608 | Apache-2.0 | [mitre/stockpile](https://github.com/mitre/stockpile) |
| Caldera plugins (arsenal/manx/access) | 56 | Apache-2.0 | [mitre/caldera](https://github.com/mitre/caldera) |
| Metasploit Framework | 8,349 | BSD-3-Clause | [rapid7/metasploit-framework](https://github.com/rapid7/metasploit-framework) |
| Infection Monkey | 36 | GPL-3.0 | [guardicore/monkey](https://github.com/guardicore/monkey) |
| RTA — Red Team Automation | 76 | **AGPL-3.0** ⚠️ | [endgameinc/RTA](https://github.com/endgameinc/RTA) |
| Sigma rules | (labels) | DRL-1.1 | [SigmaHQ/sigma](https://github.com/SigmaHQ/sigma) |
| AI-security tools (promptfoo, garak, promptmap, PyRIT, FuzzyAI, TheBigPromptLibrary) | 743+ | mixed MIT/Apache-2.0 | various (see [ATTRIBUTION.md](ATTRIBUTION.md)) |
| Synthetic orchestrator / prompt-injection | 1,067 | MIT | this repo |
| **Total** | **16,982** | | |

⚠️ **AGPLv3 note:** RTA is the only AGPL-licensed source. The AGPL has
network-distribution implications for derivative works. The public
repository satisfies the source-availability requirement. If you need an
AGPL-clean deployment, retrain after removing the `tools/rta` bucket.
See [ATTRIBUTION.md §8](ATTRIBUTION.md) for the full analysis.

---

## Quickstart (5 min)

```bash
# 1. Install uv (Python package manager, ~10MB)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone this repo
git clone https://github.com/Veedubin/AttackLM.git
cd AttackLM

# 3a. Install as a Python package (gets you 11 `attacklm-*` commands)
#    — use `[all]` to get every optional dependency
uv pip install -e ".[all]"

#    Or, if you just want the bare CLI dispatchers (no ML stack):
# uv pip install -e .

# 3b. Alternative: classic uv-managed venv with all deps in pyproject.toml
# uv sync

# 4. Clone upstream data sources (~1.5GB total, optional — data is in the repo)
attacklm-clone

# 5. Extract training data from each source
attacklm-extract

# 6. Augment each JSONL with per-pair source/license attribution
attacklm-attribute

# 7. Organize into 16 MITRE/AI/tools buckets
attacklm-buckets

# 8. Train! (3B QLoRA fits 16GB; 70B+ works on 128GB)
attacklm-train-all --single-model \
  --base-model unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit \
  --epochs 5 --max-length 2048

# Optional: add --hpo for automatic lora_r / lora_dropout sweep
```

The trained LoRA adapter lands in `models/attacklm-single/`. See
[**Inference**](#inference) below for how to use it.

> **Don't want to install?** The `scripts/` directory is the source of truth.
> Every `attacklm-*` command is a thin wrapper around a script. You can run
> `uv run python scripts/train_all.py --help` directly — same behavior,
> same flags, no install required.

---

## Install

The project ships as a **proper Python package** (`pyproject.toml`,
`src/attacklm/` layout, hatchling build backend) so users don't have to
build anything by hand.

### Option A — Editable install (recommended for development)

```bash
git clone https://github.com/Veedubin/AttackLM.git
cd AttackLM
uv pip install -e ".[all]"          # gets all optional dependencies
```

This installs 11 console-script entry points into your environment:

| Command                  | Dispatches to                          | What it does                           |
|--------------------------|----------------------------------------|----------------------------------------|
| `attacklm-train`         | `scripts/train_template.py`            | Train one QLoRA adapter                |
| `attacklm-train-all`     | `scripts/train_all.py`                 | Train all buckets / HPO                |
| `attacklm-hpo`           | `scripts/hpo_runner.py`                | Coordinate-descent HPO sweep           |
| `attacklm-infer`         | `scripts/infer.py`                     | Smoke-test inference                   |
| `attacklm-merge`         | `scripts/merge_adapter.py`             | Merge LoRA → base model                |
| `attacklm-gguf`          | `scripts/convert_to_gguf.py`           | Convert to GGUF (llama.cpp)            |
| `attacklm-demo`          | `scripts/demo.py`                      | Multi-agent orchestrator demo          |
| `attacklm-extract`       | all 6 extractors                       | Extract data from cloned repos         |
| `attacklm-buckets`       | `setup_buckets.py` + `reorganize_buckets.py` | Organize data into 16 buckets  |
| `attacklm-attribute`     | `scripts/augment_attribution.py`       | Add source/license to each JSONL row   |
| `attacklm-clone`         | `scripts/clone_repos.sh`               | Clone upstream data repos              |

The CLI dispatchers are thin wrappers — they use `runpy.run_path()` to
invoke the canonical script in `scripts/`. So `scripts/` stays the
source of truth and you can still run `uv run python scripts/foo.py`
directly if you prefer.

### Optional-dependency groups

```bash
uv pip install -e ".[train]"        # training only (peft, trl, accelerate, bitsandbytes)
uv pip install -e ".[extract]"      # data extractors (pyyaml, requests, gitpython)
uv pip install -e ".[convert]"      # GGUF conversion (gguf-python, llama-cpp-python)
uv pip install -e ".[infer]"        # inference (transformers, peft)
uv pip install -e ".[all]"          # everything
uv pip install -e ".[dev]"          # pytest, ruff, mypy
```

### Option B — Just use uv (no install)

If you'd rather not install into your environment:

```bash
git clone https://github.com/Veedubin/AttackLM.git
cd AttackLM
uv sync                              # creates .venv with all deps
uv run python scripts/train_all.py --single-model --epochs 5
```

### Option C — pip install from GitHub (no clone)

```bash
uv pip install "git+https://github.com/Veedubin/AttackLM.git#[all]"
attacklm-train --help
```

This works because the entry points in `pyproject.toml` resolve `scripts/`
relative to the installed package — but the data files won't be present
(this method is best for `attacklm-infer` and `attacklm-demo` against an
existing adapter you've trained elsewhere).

---

## Architecture

The training data is organized into **16 buckets**:

- **10 MITRE tactic buckets** — `collection`, `command_and_control`,
  `credential_access`, `defense_evasion`, `discovery`, `execution`,
  `exfiltration`, `lateral_movement`, `persistence`, `privilege_escalation`
  (TA0009, TA0011, TA0006, TA0005, TA0007, TA0002, TA0010, TA0008,
  TA0003, TA0004 respectively)
- **1 orchestrator bucket** — routing decisions across 6 sub-agents
- **2 AI-model attack buckets** — `ai-models/prompt-injection` and
  `ai-models/jailbreaking` (TA0040 — Adversarial ML)
- **3 security-tool buckets** — `tools/infection_monkey`, `tools/metasploit`,
  `tools/rta` (consolidated tool-specific data, re-routed to MITRE tactics
  where applicable)

The bucket layout lets you train:
- **One model on everything** (default — single MoE-style assistant)
- **One model per tactic** (multi-model mode)
- **One model on a subset** (e.g., `--include-tools --include-orchestrator`
  to skip the AI/ML attack data)

See `data/datasets/buckets/manifest.json` for the full per-bucket manifest
with pair counts and MITRE tactic IDs.

---

## Training

`scripts/train_all.py` is the orchestrator. Key flags:

| Flag | Default | Notes |
|---|---|---|
| `--single-model` | (off) | Train one model on all buckets combined |
| `--base-model` | `Qwen/Qwen2.5-Coder-7B-Instruct` | Use `unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit` for 16GB cards |
| `--epochs` | 10 | Total epochs over the combined dataset |
| `--max-length` | 1024 | 2048 for richer context; 1024 for 7B on 16GB |
| `--lora-r` | 16 | LoRA rank; 8 / 16 / 32 are good starting points |
| `--lora-alpha` | 32 | Conventionally `2 × lora_r` |
| `--lora-dropout` | 0.05 | Try 0.0 for less regularization |
| `--no-packing` | (packing off) | Default is OFF because flash-attn is hard to install |
| `--packing` | (off) | Enable for ~30% speedup; requires `flash_attn` |
| `--include-tools` | (off) | Include the 3 tool buckets in the combined dataset |
| `--include-orchestrator` | (off) | Include the orchestrator routing data |
| `--model-attacks` | (off) | Include the AI-model attack buckets |
| `--curriculum` | (off) | 2-stage: tactic data first, then orchestrator fine-tune |
| `--hpo` | (off) | Run coordinate-descent HPO before final training |

The training script has 13 OOM-safety fixes built in (expandable_segments,
per_device_eval_batch_size=1, chunked_nll loss, post-eval cache clear,
paged_adamw_8bit, etc.) — see the `# OOM fix #N:` comments in
`train_template.py` for the full list.

### HPO

Add `--hpo` to the training command. The sweep explores `lora_r` (8→512)
and `lora_dropout` (0→0.5) and runs a final training with the winners.
Results land in `hpo_runs/hpo_state.json`; re-analyze later with
`attacklm-hpo --analyze-only`.

---

## Inference

After training, you have a LoRA adapter in `models/attacklm-single/`.
Three ways to use it:

### Option A: Quick smoke test with `infer.py`

```bash
attacklm-infer --adapter models/attacklm-single
```

This runs 4 example prompts (MITRE tactics, orchestrator routing,
prompt injection) and prints the model's responses. No setup beyond
`uv sync` required. See `scripts/infer.py --help` for custom prompts
and generation parameters.

### Option B: Merge into the base model (simplest)

```bash
attacklm-merge \
  --base-model unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit \
  --adapter models/attacklm-single \
  --output models/attacklm-merged
```

Then load with `transformers.AutoModelForCausalLM.from_pretrained("models/attacklm-merged")`.

### Option C: Convert to GGUF for Ollama / LM Studio / llama.cpp

```bash
# Requires llama.cpp checked out and built
attacklm-gguf \
  --model models/attacklm-merged \
  --output models/attacklm.gguf

# Register with Ollama
uv run python scripts/register_ollama.py models/attacklm.gguf
```

### Option D: Load the adapter directly (smallest disk footprint)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit",
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "models/attacklm-single")
tokenizer = AutoTokenizer.from_pretrained("models/attacklm-single")

# Chat with the model
messages = [
    {"role": "system", "content": "You are an authorized Red Team specialist..."},
    {"role": "user",   "content": "Show the System Services: Service Execution technique (T1569.002)"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Data Sources (upstream)

| Project | License | Use |
|---|---|---|
| [redcanaryco/atomic-red-team](https://github.com/redcanaryco/atomic-red-team) | MIT | 2,506 atomic test triples |
| [mitre/stockpile](https://github.com/mitre/stockpile) | Apache-2.0 | 608 adversary-emulation abilities |
| [mitre/caldera](https://github.com/mitre/caldera) | Apache-2.0 | 56 plugin descriptors |
| [rapid7/metasploit-framework](https://github.com/rapid7/metasploit-framework) | BSD-3-Clause | 8,349 module description triples |
| [guardicore/monkey](https://github.com/guardicore/monkey) | GPL-3.0 | 36 plugin manifest triples |
| [endgameinc/RTA](https://github.com/endgameinc/RTA) | **AGPL-3.0** ⚠️ | 76 Python TTP triples |
| [SigmaHQ/sigma](https://github.com/SigmaHQ/sigma) | DRL-1.1 | Auxiliary context for triple structure |
| [promptfoo/promptfoo](https://github.com/promptfoo/promptfoo) | MIT | Prompt injection probes |
| [NVIDIA/garak](https://github.com/NVIDIA/garak) | Apache-2.0 | DAN/probe resources |
| [utkusen/promptmap](https://github.com/utkusen/promptmap) | MIT | Prompt injection rules |
| [Azure/PyRIT](https://github.com/Azure/PyRIT) | MIT | Jailbreak templates |
| [cyberark/FuzzyAI](https://github.com/cyberark/FuzzyAI) | Apache-2.0 | Adversarial prompt resources |
| [Resident-Falker/TheBigPromptLibrary](https://github.com/Resident-Falker/TheBigPromptLibrary) | mixed MIT/MPL | Jailbreak + system prompt library |

Full attribution, per-pair source mapping, and re-distribution guidance in
[**`/ATTRIBUTION.md`**](ATTRIBUTION.md).

---

## License

- **Code in this repository** — [MIT License](LICENSE)
- **Training data** — inherits the most restrictive license of its components
  (currently AGPL-3.0 from RTA — see [ATTRIBUTION.md §8](ATTRIBUTION.md))
- **Trained model weights** — MIT License as a new statistical artifact
  learned from openly licensed material. Whether model weights are a
  "derivative work" in the copyright sense is an unsettled question; no
  representation is made either way. If you need certainty, consult legal
  counsel for your specific deployment scenario.

The Apache-2.0 attribution required by the upstream MITRE, NVIDIA, and
CyberArk components is preserved in [**`/NOTICE`**](NOTICE).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on opening issues,
submitting PRs, and extending the bucket/extractor system.

---

## Acknowledgments

Thanks to the open-source security community — Red Canary, MITRE, Rapid7,
Guardicore, Endgame/Elastic, the SigmaHQ maintainers, the promptfoo,
garak, PyRIT, and FuzzyAI teams, and everyone who contributes to the
projects we depend on. AttackLM stands on their shoulders.
