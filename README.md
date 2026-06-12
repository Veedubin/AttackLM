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

# 4. Initialize the dataset (probes local `data/` first; falls back to git clone)
attacklm-init --yes

#    The single command above replaces steps 4–7 below. If you'd rather
#    run each step individually, the four commands are still available:
#
# 4. Clone upstream data sources (~1.5GB total, optional — data is in the repo)
# attacklm-clone
#
# 5. Extract training data from each source
# attacklm-extract
#
# 6. Augment each JSONL with per-pair source/license attribution
# attacklm-attribute
#
# 7. Organize into 16 MITRE/AI/tools buckets
# attacklm-buckets

# 8. Pick a base model — use an uncensored/abliterated one (see "Pick a base model" below)
#    Example: Qwen2.5-Coder-3B-Instruct with refusal direction removed
#    v0.2.0+ uses --dataset (multi-positional) instead of --include-tools etc.
attacklm-train-all --single-model \
  --dataset base/ \
  --base-model huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated \
  --epochs 5 --max-length 2048

# Optional: add --hpo for automatic lora_r / lora_dropout sweep
```

The trained LoRA adapter lands in `models/attacklm-single_<TIMESTAMP>/`
(v0.2.0+ uses timestamped dirs so multiple runs coexist for rollback).
The merged model goes to `models/merged/attacklm-single/`. See
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

There are **two GPU stacks** — pick the one for your hardware.

---

### CUDA stack (NVIDIA) — primary

```bash
git clone https://github.com/Veedubin/AttackLM.git
cd AttackLM
uv pip install -e ".[all]"
```

That installs everything: `torch` (CUDA wheel from PyPI), `bitsandbytes`,
`transformers`, `peft`, `trl`, plus the C++ extensions `flash-attn`,
`causal-conv1d`, and `flash-linear-attention` (for Qwen3-Next and similar
hybrid linear-attention models).

| Component | Where it comes from |
|---|---|
| `torch`, `torchvision`        | PyPI (CUDA build, auto-selected) |
| `bitsandbytes`                | PyPI (CUDA wheels) |
| `flash-attn`                  | Built from source via pip (~5 min) |
| `causal-conv1d`               | Pre-built wheel from PyPI |
| `flash-linear-attention`      | Pre-built wheel from PyPI |

---

### ROCm stack (AMD) — e.g. MI300X, RX 7900 XTX, Strix Halo

ROCm PyTorch wheels are **not on PyPI** — you must add PyTorch's index
URL. The `bitsandbytes` 0.49+ wheel **only ships CUDA .so files** (cuda118/120/121/122/126) — on ROCm, install bitsandbytes with `--no-deps` and verify, or skip it entirely (the FP8 path doesn't need it). The C++ extensions (`flash-attn`, `causal-conv1d`, `flash-linear-attention`) **have no ROCm support** — the modeling has pure-PyTorch fallbacks (slower but works).

**Important: which ROCm version?** The PyTorch ROCm index publishes
different `torch` versions per channel. The version pins in this repo
(`torch==2.12.0`, `torchvision==0.27.0`) are only available on the
**rocm7.1 / rocm7.2** channels. Older channels (rocm6.x) cap out at
torch 2.5-2.9 and will fail to resolve the pin.

```bash
# 1. Install ROCm PyTorch from the rocm7.2 channel (has torch 2.12.0)
uv pip install --index-url https://download.pytorch.org/whl/rocm7.2 \
    torch==2.12.0 torchvision==0.27.0

# 2. Install AttackLM with the ROCm meta-group
git clone https://github.com/Veedubin/AttackLM.git
cd AttackLM
uv pip install -e ".[all-rocm]"
```

After install, verify:
```bash
python -c "import torch; print('torch:', torch.__version__, '— hip:', torch.version.hip)"
# should print something like: torch: 2.12.0+rocm7.2 — hip: 7.2.XXXXX
```

`[all-rocm]` is `attacklm[train-rocm,extract,convert]` — it pulls in
`peft`, `trl`, `accelerate`, `bitsandbytes` and **no** CUDA-only C++
extensions.

| Component | Where it comes from |
|---|---|
| `torch`, `torchvision`        | PyTorch ROCm index (`+rocm7.2` build) |
| `bitsandbytes`                | PyPI (CUDA-only — works for non-FP8 paths; uninstall for FP8) |
| `flash-attn`                  | **Not installed** — sdpa fallback in QLoRA |
| `causal-conv1d`               | **Not installed** — pure-PyTorch fallback in Qwen3-Next modeling |
| `flash-linear-attention`      | **Not installed** — pure-PyTorch fallback |

> **If `attacklm-train` fails with** `Could not import module '...ForCausalLM'`:
> The error message usually hides the actual cause in its exception chain.
> The most common ROCm causes (in order of likelihood):
> ```bash
> # 1. bitsandbytes CUDA-only wheel — uninstall (FP8 path doesn't need it)
> uv pip uninstall bitsandbytes
>
> # 2. Half-installed C++ extensions — remove them
> uv pip uninstall causal-conv1d flash-linear-attention
>
> # 3. Wrong PyTorch channel — verify ROCm build is installed
> python -c "import torch; print(torch.version.hip)"
> # If 'None', reinstall with --index-url https://download.pytorch.org/whl/rocm7.2
> ```
> v0.1.3+ prints the actual exception chain so you can see which of these it is.

---

### CPU / Apple Silicon (inference only)

```bash
git clone https://github.com/Veedubin/AttackLM.git
cd AttackLM
uv pip install -e ".[infer]"
```

Training on CPU/MPS is technically possible but will be **extremely slow**.
Use only for dry-runs or for running a pre-trained adapter against
prompts. Pick `[all-cuda]` or `[all-rocm]` for actual training.

---

### 11 console-script entry points

All install paths give you these:

| Command                  | Dispatches to                          | What it does                           |
|--------------------------|----------------------------------------|----------------------------------------|
| `attacklm-train`         | `scripts/train_template.py`            | Train one QLoRA adapter                |
| `attacklm-train-all`     | `scripts/train_all.py`                 | Train all buckets / HPO                |
| `attacklm-hpo`           | `scripts/hpo_runner.py`                | Coordinate-descent HPO sweep           |
| `attacklm-infer`         | `scripts/infer.py`                     | Smoke-test inference                   |
| `attacklm-merge`         | `scripts/merge_adapter.py`             | Merge LoRA → base model                |
| `attacklm-gguf`          | `scripts/convert_to_gguf.py`           | Convert to GGUF (llama.cpp)            |
| `attacklm-build`         | `scripts/build.py`                     | merge → GGUF → install (one shot)      |
| `attacklm-demo`          | `scripts/demo.py`                      | Multi-agent orchestrator demo          |
| `attacklm-extract`       | all 6 extractors                       | Extract data from cloned repos         |
| `attacklm-buckets`       | `setup_buckets.py` + `reorganize_buckets.py` | Organize data into 16 buckets  |
| `attacklm-attribute`     | `scripts/augment_attribution.py`       | Add source/license to each JSONL row   |
| `attacklm-clone`         | `scripts/clone_repos.sh`               | Clone upstream data repos              |
| `attacklm-init`          | `scripts/init_pipeline.py`             | **One-shot init: clone→extract→attribute→buckets** (probes local first) |
| `attacklm-balance`       | `scripts/balance_buckets.py`           | Build a balanced subset of the buckets |

The CLI dispatchers are thin wrappers — they use `runpy.run_path()` to
invoke the canonical script in `scripts/`. So `scripts/` stays the
source of truth and you can still run `uv run python scripts/foo.py`
directly if you prefer.

---

### Optional-dependency groups (advanced)

```bash
# Fine-grained control
uv pip install -e ".[train-cuda]"   # CUDA training stack
uv pip install -e ".[train-rocm]"   # ROCm training stack
uv pip install -e ".[infer-cuda]"   # CUDA inference
uv pip install -e ".[infer-rocm]"   # ROCm inference
uv pip install -e ".[extract]"      # data extractors
uv pip install -e ".[convert]"      # GGUF conversion
uv pip install -e ".[dev]"          # pytest, ruff, mypy
```

---

### No-install option (scripts only)

If you'd rather not install into your environment:

```bash
git clone https://github.com/Veedubin/AttackLM.git
cd AttackLM
uv sync                              # creates .venv with all deps
uv run python scripts/train_all.py --single-model --epochs 5
```

`uv sync` reads `pyproject.toml` and creates a venv with the `[all]`
extras. Scripts in `scripts/` are the source of truth — the CLI is a
thin dispatcher layer.

---

## Architecture

The training data is organized into **16 buckets**:

- **10 MITRE tactic buckets** — under `base/`: `base/collection`,
  `base/command_and_control`, `base/credential_access`, `base/defense_evasion`,
  `base/discovery`, `base/execution`, `base/exfiltration`,
  `base/lateral_movement`, `base/persistence`, `base/privilege_escalation`
  (TA0009, TA0011, TA0006, TA0005, TA0007, TA0002, TA0010, TA0008,
  TA0003, TA0004 respectively)
- **1 orchestrator bucket** — routing decisions across 6 sub-agents
- **2 AI-model attack buckets** — under `ai/`: `ai/prompt-injection` and
  `ai/jailbreaking` (TA0040 — Adversarial ML)
- **3 security-tool buckets** — under `tools/`: `tools/infection_monkey`,
  `tools/metasploit`, `tools/rta` (consolidated tool-specific data, re-routed
  to MITRE tactics where applicable)

> **v0.2.1 layout change:** the 10 tactic buckets moved from top-level
> into a new `base/` parent directory, and `ai-models/` was renamed to
> `ai/`. See the [CHANGELOG](CHANGELOG.md#021--2026-06-10) for the
> migration script and details.

The bucket layout lets you train:
- **One model on everything** (default — single MoE-style assistant)
- **One model per tactic** (multi-model mode)
- **One model on a subset** (e.g., `--include-tools --include-orchestrator`
  to skip the AI/ML attack data)

See `data/datasets/buckets/manifest.json` for the full per-bucket manifest
with pair counts and MITRE tactic IDs.

---

## Pick a base model

> **Use an uncensored / abliterated base.** The dataset teaches red-team
> tradecraft, but it can't fully override the safety alignment baked into
> a base Instruct model. Use a base that has had its refusal direction
> removed (abliterated) — you'll get a much sharper, more consistent
> result than SFT alone.

### Recommended bases (pick one)

| Model                                                        | Size  | VRAM needed  | Notes                                                                |
| ------------------------------------------------------------ | ----- | ------------ | -------------------------------------------------------------------- |
| `huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated`            | 3B    | 16 GB         | Best fit for RTX 4080 SUPER / 4070 Ti. Same Qwen2.5-Coder arch as the original default. Apache-2.0. |
| `huihui-ai/Qwen2.5-Coder-1.5B-Instruct-abliterated`          | 1.5B  | 8 GB          | Tight hardware, fast iteration. Apache-2.0.                          |
| `huihui-ai/Qwen2.5-Coder-7B-Instruct-abliterated`            | 7B    | 24 GB         | Better quality, more coherent long responses. Apache-2.0.            |
| `BlossomsAI/Qwen2.5-Coder-32B-Instruct-Uncensored`          | 32B   | 64+ GB        | Top quality, needs 64+ GB VRAM. Apache-2.0.                         |
| `failspy/Meta-Llama-3-8B-Instruct-abliterated-v3`            | 8B    | 24 GB         | If you'd rather train on Llama-3. Apache-style license.              |
| `failspy/Meta-Llama-3-70B-Instruct-abliterated-v3.5`         | 70B   | 128+ GB       | Frontier quality. Quantized GGUF versions also available.            |

Browse the full [failspy/abliterated-v3 collection](https://huggingface.co/collections/failspy/abliterated-v3-664a8ad0db255eefa7d0012b) and [3000+ Heretic models](https://huggingface.co/models?other=heretic) for more.

### Make your own with Heretic (if your preferred base isn't pre-abliterated)

[p-e-w/heretic](https://github.com/p-e-w/heretic) is a fully automatic
abliteration tool. 30 minutes on a 16 GB card for a 3B model.

```bash
pip install heretic-llm
heretic Qwen/Qwen2.5-Coder-3B-Instruct --n-trials 100
# Interactive menu: choose "Save the model to a local folder"
```

Then point `--base-model` at the saved folder. The interactive menu
**requires a real TTY** (gnome-terminal, konsole, xterm, etc.) — piping
stdin via `printf "n\n"` only handles the first prompt.

The other 30 lines of the technique are documented at:
- [mlabonne/abliteration (the original 2024 recipe)](https://huggingface.co/blog/mlabonne/abliteration)
- [grimjim/projected-abliteration (Oct 2025 — projection refinement)](https://huggingface.co/blog/grimjim/projected-abliteration)
- [p-e-w/heretic (unified tool, modern state-of-the-art)](https://github.com/p-e-w/heretic)

---

## Training

`scripts/train_all.py` is the orchestrator. Key flags:

| Flag | Default | Notes |
|---|---|---|
| `--single-model` | (off) | Train one model on all buckets combined |
| `--base-model` | (auto) | v0.2.0+: defaults to round-2 SFT (latest completed run for this agent), then abliterated Qwen 3B. Pass this to override. |
| `--dataset` (multi) | none | v0.2.0+: positional list of bucket specs. `base/`, `tools/`, `ai/`, `orchestrator`, subpaths (`tools/metasploit/`), aliases (`all`, `tactics`, `tools-all`). |
| `--backup` | (on) | Tar.gz the previous round-2 SFT run to `models/.backups/` before training starts. `--no-backup` to skip. |
| `--epochs` | 10 | Total epochs over the combined dataset |
| `--max-length` | 1024 | 2048 for richer context; 1024 for 7B on 16GB |
| `--lora-r` | 16 | LoRA rank; 8 / 16 / 32 are good starting points |
| `--lora-alpha` | 32 | Conventionally `2 × lora_r` |
| `--lora-dropout` | 0.05 | Try 0.0 for less regularization |
| `--no-packing` | (packing off) | Default is OFF because flash-attn is hard to install |
| `--packing` | (off) | Enable for ~30% speedup; requires `flash_attn` |
| `--include-tools` | (off) | **Deprecated in v0.2.0**: use `--dataset tools/` instead |
| `--include-orchestrator` | (off) | **Deprecated in v0.2.0**: use `--dataset orchestrator` instead |
| `--model-attacks` | (off) | **Deprecated in v0.2.0**: use `--dataset ai/` instead |
| `--curriculum` | (off) | 2-stage: tactic data first, then orchestrator fine-tune |
| `--hpo` | (off) | Run coordinate-descent HPO before final training |

The training script has 13 OOM-safety fixes built in (expandable_segments,
per_device_eval_batch_size=1, chunked_nll loss, post-eval cache clear,
paged_adamw_8bit, etc.) — see the `# OOM fix #N:` comments in
`train_template.py` for the full list.

### Run-dir naming (v0.2.2+)

`attacklm-train` and `attacklm-train-all` both default to writing the
adapter to a **timestamped** subdirectory so re-runs are preserved:

```bash
# Default — appends a timestamp to your --output
attacklm-train --dataset data/foo.jsonl --output models/agent-3b
# → models/agent-3b_2026-06-10_15-15/   (preserved across re-runs)

# Opt out of timestamping (will refuse to clobber a completed run)
attacklm-train --dataset data/foo.jsonl --output models/agent-3b --no-timestamp
# ERROR: Refusing to clobber completed run at models/agent-3b.

# Override the refusal
attacklm-train --dataset data/foo.jsonl --output models/agent-3b \
               --no-timestamp --force
```

If `--output` already ends in `_YYYY-MM-DD_HH-MM` (i.e. it was
produced by an earlier run or by `attacklm-train-all`), the suffix
is left alone — re-runs get a new suffix (`_2`, `_3`, …) only if the
exact same name exists.

### Multi-round SFT (v0.2.0+)

Each training run writes a `state.json` sidecar at `models/{agent}_{TIMESTAMP}/state.json`.
It records the base model, hparams, dataset, progress, and a `completed` flag.

**Round 2 SFT** trains a fresh LoRA on top of a previously completed run:

```bash
# Round 1: train on tactics (10 buckets, 7,398 pairs)
attacklm-train-all --single-model --dataset base/ --epochs 5

# Round 2: train on tools ON TOP of the round-1 merged weights
# (auto-detected from state.json; backup tar of round 1 happens first)
attacklm-train-all --single-model --dataset tools/ --epochs 3

# Round 3: train on everything
attacklm-train-all --single-model --dataset all --epochs 2
```

Each round:
1. Detects the latest completed run for the agent name
2. Backups it to `models/.backups/{name}_{timestamp}.tar.gz` (5 GB, ~30 sec)
3. Loads the merged weights as the new base
4. Trains a new LoRA on top
5. Writes a new timestamped run dir with updated `state.json`

**Auto-resume** for crashed/killed runs:

```bash
# If a run died mid-training, just re-run with the same command.
# state.json (completed=false) + checkpoint-N/ present → auto-resume.
attacklm-train-all --single-model --dataset base/ --epochs 5
```

### `--dataset` DSL

The new dataset spec is dir-shaped and hierarchical:

| Spec                          | Resolves to                                          | Pair count |
|-------------------------------|------------------------------------------------------|-----------:|
| `base/`                       | All 10 MITRE tactic buckets                          |      7,398 |
| `tools/`                      | All 3 tool buckets (metasploit, infection_monkey, rta) |      8,461 |
| `tools/metasploit/`           | Just metasploit                                       |      8,349 |
| `tools/infection_monkey/`     | Just infection_monkey                                 |         36 |
| `tools/rta/`                  | Just RTA                                              |         76 |
| `ai/`                         | Both AI buckets (jailbreaking, prompt-injection)      |        743 |
| `orchestrator`                | The orchestrator bucket                               |        380 |
| `all`                         | Everything (alias for `base + tools + ai + orchestrator`) |     16,982 |
| `tactics`                     | Alias for `base/`                                     |      7,398 |

Multiple specs combine: `--dataset base/ tools/metasploit/` = 10 tactics + just metasploit = 15,747 pairs.

Legacy `--include-tools` / `--model-attacks` / `--include-orchestrator` still work
and translate internally to `--dataset` specs. The new flag wins if both are passed.

### Balanced sampling (`attacklm-balance`)

The 16 buckets are heavily skewed: `tools/metasploit` alone has 8,349
pairs (49% of the 16,982 total). Training on raw `--dataset all`
makes the model see ~2 Metasploit examples for every 1 non-Metasploit
example, which overfits it to msfconsole syntax at the expense of
broader tactical coverage.

`attacklm-balance` builds a balanced subset of the buckets. It applies
a per-bucket cap (one cap applied uniformly to all buckets) and
selects examples from each bucket with a chosen strategy:

```bash
# Dry-run: see the per-bucket caps + total without writing
attacklm-balance --profile 7b-128gb --dry-run

# Write a balanced dataset to data/datasets/balanced/
attacklm-balance --profile 7b-128gb \
    --output data/datasets/balanced/balanced_7b-128gb.jsonl

# Then train on it
attacklm-train --dataset data/datasets/balanced/balanced_7b-128gb.jsonl \
               --output models/attacklm-7b-128gb \
               --base-model huihui-ai/Qwen2.5-Coder-7B-Instruct-abliterated
```

**Profiles** (named per-bucket cap values, tuned for common hardware combos):

| Profile    | Per-bucket cap | Total pairs | Notes                                      |
|------------|---------------:|------------:|--------------------------------------------|
| `3b-16gb`  |            800 |     ~7,500  | 3B QLoRA on 16 GB card                     |
| `7b-16gb`  |            800 |     ~7,500  | 7B QLoRA on 16 GB card                     |
| `7b-128gb` |          1,500 |     ~9,800  | 7B QLoRA on 128 GB rig                     |
| `14b-128gb`|          1,500 |     ~9,800  | 14B QLoRA on 128 GB rig                    |
| `31b-128gb`|          2,000 |    ~10,600  | 31B QLoRA on 128 GB rig                    |
| `full`     |      unlimited |     16,982  | All data, no cap                           |
| `custom`   |       (you set)|    (you set)| `--per-bucket-cap` or `--target-total`     |

**Strategies** (within a bucket, after the cap is applied):

- `stratified` (default) — group examples by their first MITRE
  technique ID, source, or first line of assistant content, then
  allocate **at least 1 per group** so every technique / module gets
  representation. Falls back to uniform random if there are fewer
  than 3 groups in the bucket.
- `random` — uniform random sample of N (seeded by `--seed`).
- `head` — first N examples in the file (reproducible but biased to
  whatever order the data is in).

**Custom allocation** — the `custom` profile takes either an explicit
`--per-bucket-cap` JSON or a `--target-total` with `--category-shares`:

```bash
# 12K pairs total, weighted 30% tactics / 40% tools / 20% ai / 10% orchestrator
attacklm-balance --profile custom --target-total 12000 \
    --category-shares '{"tactic": 0.3, "tools": 0.4, "ai_redteam": 0.2, "meta": 0.1}'

# Just metasploit at 1500 + discovery at 800, everything else uncapped
attacklm-balance --profile custom \
    --per-bucket-cap '{"tools/metasploit": 1500, "base/discovery": 800}'
```

Output JSONLs are written to `data/datasets/balanced/`, are excluded
from git, and contain a `_source_bucket` field on every example for
traceability. See `scripts/balance_buckets.py --help` for the full
flag list and `CHANGELOG.md` for the design rationale.

### HPO

Add `--hpo` to the training command. The sweep explores `lora_r` (8→512)
and `lora_dropout` (0→0.5) and runs a final training with the winners.
Results land in `hpo_runs/hpo_state.json`; re-analyze later with
`attacklm-hpo --analyze-only`.

---

## Inference

After training, you have one or more LoRA adapters in
`models/attacklm-single_*/` (timestamped). Pick the latest one (most
recent date) and merge it. Three ways to use it:

### Option A: Quick smoke test with `infer.py`

```bash
# v0.2.0+: list available run dirs and pick the latest
ls -d models/attacklm-single_*/ | tail -1
# Then infer against it
attacklm-infer --adapter models/attacklm-single_2026-06-10_01-12
```

This runs 4 example prompts (MITRE tactics, orchestrator routing,
prompt injection) and prints the model's responses. No setup beyond
`uv sync` required. See `scripts/infer.py --help` for custom prompts
and generation parameters.

### Option B: Merge into the base model (simplest)

```bash
# v0.2.0+: --adapter takes a timestamped run dir directly.
# merge_all auto-picks the latest run for an agent if you omit --adapter.
attacklm-merge \
  --base-model huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated \
  --adapter models/attacklm-single_2026-06-10_01-12 \
  --output models/merged/attacklm-single
```

Then load with `transformers.AutoModelForCausalLM.from_pretrained("models/merged/attacklm-single")`.

### Option C: Convert to GGUF for Ollama / LM Studio / llama.cpp

```bash
# v0.2.0+: --input is the merged model dir (not the adapter)
attacklm-gguf \
  --input models/merged/attacklm-single \
  --install-lmstudio

# Register with Ollama
uv run python scripts/register_ollama.py models/gguf/attacklm-single.Q4_K_M.gguf
```

### Option E: One-shot merge + GGUF + install (`attacklm-build`)

v0.2.2+: the 3-command shell pipeline becomes a single command. The
build command also drops a manifest at `models/built/{name}_{timestamp}/`
for later retrieval:

```bash
# Merge + GGUF + install to LM Studio, all in one
attacklm-build \
  --adapter models/attacklm-3b_16g_2026-06-10_15-15 \
  --base ./uncensored/ \
  --name attacklm-3b-16g

# Skip the merge step (use an already-merged model)
attacklm-build \
  --merged models/merged/attacklm-3b-16g \
  --name attacklm-3b-16g

# Also register with Ollama
attacklm-build \
  --adapter models/attacklm-3b_16g_2026-06-10_15-15 \
  --base ./uncensored/ \
  --name attacklm-3b-16g \
  --register-ollama
```

`--install-lmstudio` is ON by default. Use `--no-install-lmstudio` to
just produce the GGUF. The build manifest records the GGUF path,
mtime, base model, and which install steps ran.

### Option D: Load the adapter directly (smallest disk footprint)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated",
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

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full version history. Notable
recent releases:

- **v0.2.2** (2026-06-10) — `attacklm-balance` (balanced bucket sampler),
  `attacklm-build` (one-shot merge+GGUF+install), auto-timestamped
  run dirs in `attacklm-train`, accurate epoch counter, GGUF
  mtime-based staleness check, `attacklm-gguf --name` /
  `--register-ollama` / `--quant` / `--build` / `--force`.
- **v0.2.1** (2026-06-10) — Bucket layout normalized to 4 parents
  (`base/`, `tools/`, `ai/`, `orchestrator/`).
- **v0.2.0** (2026-06-10) — Multi-round SFT, `state.json` provenance,
  `--dataset` DSL, `--backup`/`--no-backup`, LoRA adapter detection
  in GGUF conversion. **Major version bump.**
- v0.1.5 (2026-06-10) — LM Studio path fix, kernels pin, path resolver
- v0.1.4 (2026-06-10) — Merge + GGUF pipeline
- v0.1.0 (2026-06-10) — Initial public release

---

## Acknowledgments

Thanks to the open-source security community — Red Canary, MITRE, Rapid7,
Guardicore, Endgame/Elastic, the SigmaHQ maintainers, the promptfoo,
garak, PyRIT, and FuzzyAI teams, and everyone who contributes to the
projects we depend on. AttackLM stands on their shoulders.
