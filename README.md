# AttackLM

> A QLoRA fine-tuning pipeline for a MITRE ATT&CK-grounded red-team AI assistant.
> 16,982 training pairs · 3B / 7B Qwen base · coordinate-descent HPO · 16GB VRAM-friendly.

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
- **Coordinate-descent HPO** built in. Run `--hpo` and the pipeline sweeps
  `lora_r` and `lora_dropout`, escalates each axis until metrics degrade,
  backs off, and trains a final model with the winners.
- **16GB VRAM friendly.** 3B QLoRA + `--max-length 2048` fits a 4080 SUPER
  comfortably. 7B works with `--max-length 1024`.

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
git clone https://github.com/YOUR_ORG/AttackLM.git
cd AttackLM

# 3. Install dependencies (PyTorch + HuggingFace stack)
uv sync

# 4. Clone upstream data sources (~1.5GB total)
uv run python scripts/clone_repos.sh

# 5. Extract training data from each source
uv run python scripts/extract_atomic_red_team_to_jsonl.py
uv run python scripts/extract_caldera_plugins_to_jsonl.py
uv run python scripts/parse_metasploit_to_jsonl.py
uv run python scripts/extract_rta_to_jsonl.py
uv run python scripts/extract_infection_monkey_to_jsonl.py
uv run python scripts/extract_ai_tools_to_jsonl.py

# 6. Augment each JSONL with per-pair source/license attribution
uv run python scripts/augment_attribution.py

# 7. Organize into 16 MITRE/AI/tools buckets
uv run python scripts/setup_buckets.py
uv run python scripts/reorganize_buckets.py

# 8. Train! (3B QLoRA, fits 16GB VRAM)
uv run python scripts/train_all.py --single-model \
  --base-model unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit \
  --epochs 5 --max-length 2048
```

The trained LoRA adapter lands in `models/attacklm-single/`. See
[**Inference**](#inference) below for how to use it.

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

### HPO (Hyperparameter Optimization)

```bash
uv run python scripts/train_all.py --hpo --single-model \
  --base-model unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit \
  --epochs 5 --max-length 2048
```

This runs an 8-trial coordinate-descent sweep (4 values × 2 axes —
`lora_r` and `lora_dropout`), escalates each axis until metrics degrade,
then runs a final training with the winners. Results land in
`hpo_runs/hpo_state.json` and can be re-run later with
`uv run python scripts/hpo_runner.py --analyze-only`.

Tuning knobs: `--hpo-trials-per-axis` (default 4), `--hpo-trial-steps`
(default 200), `--hpo-dataset` (default: capped 5,000-pair slice).

---

## Inference

After training, you have a LoRA adapter in `models/attacklm-single/`.
Three ways to use it:

### Option A: Quick smoke test with `infer.py`

```bash
uv run python scripts/infer.py --adapter models/attacklm-single
```

This runs 4 example prompts (MITRE tactics, orchestrator routing,
prompt injection) and prints the model's responses. No setup beyond
`uv sync` required. See `scripts/infer.py --help` for custom prompts
and generation parameters.

### Option B: Merge into the base model (simplest)

```bash
uv run python scripts/merge_adapter.py \
  --base-model unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit \
  --adapter models/attacklm-single \
  --output models/attacklm-merged
```

Then load with `transformers.AutoModelForCausalLM.from_pretrained("models/attacklm-merged")`.

### Option C: Convert to GGUF for Ollama / LM Studio / llama.cpp

```bash
# Requires llama.cpp checked out and built
uv run python scripts/convert_to_gguf.py \
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
