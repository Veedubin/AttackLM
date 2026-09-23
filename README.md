# AttackLM

[![PyPI version](https://img.shields.io/pypi/v/attacklm.svg?label=version&color=blue)](https://pypi.org/project/attacklm/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://docs.python.org/3.10/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests: 1061+](https://img.shields.io/badge/tests-1061%2B-brightgreen.svg)](#testing)
[![GH release: v0.21.0](https://img.shields.io/badge/release-v0.21.0-blue.svg)](https://github.com/Veedubin/AttackLM/releases)

**A security-AI fine-tuning platform and research toolkit.**

AttackLM is three things in one package:

1. **A fine-tuning pipeline** for MITRE ATT&CK-grounded security LLMs.
   Trains Qwen2.5-Coder, DeepSeek, and other open models on a curated
   security corpus (26K+ training pairs, 19 sources) using parameter-
   efficient methods (QLoRA, GaLore, PiSSA, Spectrum) and full-parameter
   methods (DeepSpeed ZeRO-3 + CPU offload, LOMO, FP8, BitNet).

2. **A research toolkit** for owner-side model security testing.
   `attacklm audit` runs inversion attacks (Carlini 2021
   prefix-completion extraction, Carlini 2022 reference attack,
   per-token loss, and LiRA shadow-model MIA) against your own model so
   you can quantify what it memorized before deployment.

3. **A capability benchmark** (Inspect AI) that measures whether the
   fine-tune is a *better security analyst* than its base — with
   contamination-aware scoring and a teach-don't-arm posture judge. A
   Qwen3-14B fine-tune reaches **0.675 on CTI-Bench MCQ** (llama-3-70B /
   Gemini-1.5 tier) and **+57% on applied ATT&CK reasoning** over the 3B,
   while never emitting a working exploit. See
   [Benchmark model capability](#benchmark-model-capability).

All of it is wired into a **terminal GUI** (`attacklm gui`) for
interactive use over SSH, WSL, or headless servers.

The training data lives in the companion package
**[Veedubin/attacklm-dataset](https://github.com/Veedubin/attacklm-dataset)**
(since v0.11.0). `attacklm init` automatically uses it when installed,
or guides you to install it.

---

## Table of Contents
- [Why AttackLM?](#why-attacklm)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [What you can do](#what-you-can-do)
  - [Train a security LLM](#train-a-security-llm)
  - [Audit a model for memorized data](#audit-a-model-for-memorized-data)
  - [Benchmark model capability](#benchmark-model-capability)
  - [Queue & gauntlets](#queue--gauntlets)
  - [Run the TUI](#run-the-tui)
- [Training methods](#training-methods)
- [Dataset & provenance](#dataset--provenance)
- [Research toolkit (audit)](#research-toolkit-audit)
- [Architecture](#architecture)
- [CLI reference](#cli-reference)
- [Testing](#testing)
- [License & contributing](#license--contributing)

---

## Why AttackLM?

Offensive-security LLM tooling is fragmented:
- **Trainers** (Axolotl, LLaMA-Factory) give you no domain data.
- **Security datasets** (MITRE Caldera, Atomic Red Team) give you no
  training harness.
- **Research tools** (membership-inference libraries) are
  paper-specific and don't ship with a usable pipeline.

AttackLM is the integration point. The same install that gives you
QLoRA + DeepSpeed + 24K curated security pairs also gives you an
audit harness (Carlini 2021 extraction + MIA 4 ways) and a
terminal UI that runs over SSH. The data is per-record attributed
to its upstream source (BSD-3, DRL-1.1, Apache-2.0, MIT, etc.),
so the legal audit is part of the package, not a TODO.

## Quickstart

```bash
# 1. Install the full stack (trainer + dataset)
pip install "attacklm[all]"

# 2. Pull the per-record-attributed security corpus
attacklm init --yes

# 3. Build a balanced training subset
attacklm balance --profile 7b-16gb --preset red-team

# 4. Train (QLoRA on Qwen2.5-Coder-3B, 10 epochs)
attacklm train -- --dataset data/datasets/balanced/balanced_7b-16gb.jsonl --epochs 10 --train

# 5. (Optional) Audit the trained model for memorization
attacklm audit --attack all --mia-method per_token --model models/attacklm-single_TIMESTAMP
```

The `--dataset ...` path is local; the model checkpoints are local;
nothing leaves your machine.

## Installation

### Prerequisites
- **OS**: Linux (Ubuntu recommended) or WSL2
- **Python**: 3.10+
- **Hardware**: NVIDIA GPU with 8GB+ VRAM (RTX 30-series/40-series)
  or AMD ROCm-compatible GPU

### Pip (recommended)

```bash
# NVIDIA CUDA
pip install "attacklm[all]"

# AMD ROCm
pip install "attacklm[all-rocm]"

# uv (faster)
uv pip install "attacklm[all]"
```

### From source

```bash
git clone https://github.com/Veedubin/AttackLM.git
cd AttackLM
pip install -e ".[all]"
```

### Verify

```bash
attacklm --version       # 0.21.0
attacklm --help
pytest tests/ -q         # 1061+ passed
```

**Note on the audit harness**: AttackLM wraps
`attacklm-dataset/scripts/inversion_audit.py`. The five bug fixes
in commit `4386995` of attacklm-dataset (3 correctness/MUST-FIX
+ 2 quality) have been documented in
[attacklm-dataset/CHANGELOG.md](https://github.com/Veedubin/attacklm-dataset/blob/main/CHANGELOG.md)
and the [attacklm-dataset README](https://github.com/Veedubin/attacklm-dataset/blob/main/README.md).
The AttackLM CLI flag set is unchanged; the fixes are in the
implementation under the hood.

### Memory optimization note

By default, AttackLM uses PyTorch's built-in
`torch.backends.cuda.enable_mem_efficient_sdp()` instead of
`flash-attn`. Same $\mathcal{O}(1)$ tiled-attention algorithm, zero
compilation, works on every supported PyTorch/CUDA environment. At
sequence length 12,000, a vanilla $\mathcal{O}(n^2)$ attention
matrix would consume ~8GB of VRAM just for the matrix; SDP keeps
that constant. If you have a perfectly tuned environment and want
the absolute maximum throughput, `pip install "attacklm[flash-attn]"`
gets you standalone FlashAttention-2.

---

## What you can do

### Train a security LLM

Five end-to-end workflows, from "I have nothing" to "I have a GGUF
file Ollama can serve":

| Workflow | Time | What you get |
| :--- | :--- | :--- |
| **Quick Start** | 5 min install, ~1h train | QLoRA adapter on Qwen2.5-Coder-3B |
| **Maximum Quality** | ~3h train | GaLore + Spectrum + 20% evolved pairs |
| **HPO $\rightarrow$ Train $\rightarrow$ Deploy** | variable | Hyperparameter-swept adapter, merged into a local GGUF |
| **Evolve Pairs $\rightarrow$ Filter $\rightarrow$ Train** | ~30 min evolve | Synthetic expansion of short factual pairs |
| **40B+ on 16GB GPU** | overnight | DeepSpeed ZeRO-3 + CPU offload |

See the [CLI reference](#cli-reference) below for every flag.

### Audit a model for memorized data

`attacklm audit` is the **research toolkit** side. It runs four
attack classes against a model you point it at (your own model —
this is for owner-side testing, not adversary work):

| Attack class | Paper | What it measures |
| :--- | :--- | :--- |
| **Prefix-completion extraction** | Carlini et al. 2021 ([arXiv:2012.07805](https://arxiv.org/abs/2012.07805)) | Whether the model can regenerate verbatim training data given a prefix. |
| **MIA reference attack (loss on assistant turn + zlib entropy)** | Carlini et al. 2022 ([arXiv:2112.03570](https://arxiv.org/abs/2112.03570)) | Whether per-record NLL is lower on members than on non-members. After commit `4386995` in `attacklm-dataset`, the reference attack scores the assistant turn only (per MUSE 2023 default), eliminating the prompt-length bias that affected earlier runs. Zlib entropy is computed as a separate calibration signal. |
| **MIA per-token loss** | Shi et al. (MUSE) 2024 ([arXiv:2407.06460](https://arxiv.org/abs/2407.06460)) | Same idea, normalized by suffix-token count (removes length bias). |
| **MIA LiRA (likelihood ratio)** | Carlini et al. 2022 §4 ([arXiv:2112.03570](https://arxiv.org/abs/2112.03570)) | The "10× more powerful at low FPR" MIA. Requires K shadow-model loss files. |

Output is per-record JSONL (chmod 0600) plus a `summary.json` and
`threshold.md`. Use `--attack extraction` or `--attack mia` to
scope to a single class; `--mia-method {reference,zlib,per_token,lira,all}`
to pick a MIA variant. The implementations are clean-room
reimplementations of the papers above, with per-file provenance blocks
in [attacklm-dataset `scripts/inversion/`](https://github.com/Veedubin/attacklm-dataset/tree/main/scripts/inversion).

The audit harness is **hermetic** — it does not call out to any
network, does not require GPU, runs on a CPU laptop in minutes.
Mocked model loaders mean you can test the audit pipeline in CI
without owning a real model.

### Benchmark model capability

Since v0.19.0, AttackLM ships a **capability benchmark** that scores
your model on external security benchmarks through an eval harness
(Inspect AI). Results are normalised into the same report schema
`queue compare` consumes, so a training run, its privacy audits, and
its capability benchmarks all land in one comparison against the
base-model baseline.

**Results so far (v0.21.0).** The benchmark's first real use qualified a
larger candidate: a **Qwen3-14B** fine-tune, run through the *same* pipeline
as the shipped 3B, is a substantially stronger security analyst while fully
preserving *teach-don't-arm*.

| Benchmark | 3B | **Qwen3-14B** |
| :--- | :--- | :--- |
| applied-attack micro-F1 (n=36) | 0.278 | **0.435** (+57%) |
| &nbsp;&nbsp;— code-review sub-score | 0.000 | **0.308** |
| CTI-Bench MCQ accuracy | 0.545 | **0.675** (≈ llama3-70b / gemini-1.5 tier) |
| CTI-Bench ATE micro-F1 | 0.099 | **0.183** |

Posture holds on **both** models: 0% refusals, 94–100% "taught", **0%
"overshared"**, and zero arming markers (shellcode / msfvenom / reverse-shell)
in a structural scan — the 14B is not just smarter but *cleaner*. It loads
4-bit in ~10 GB (fits a 16 GB card). Full write-up:
[`docs/results/qwen3-14b-vs-3b.md`](docs/results/qwen3-14b-vs-3b.md). *(Caveat:
these ran through a direct transformers+bnb-4bit generation path, not the
Inspect+vLLM harness — scoring is identical; only generation differs.)*

**Benchmark packs** (defined in `data/bench/packs/`, loaded by
`src/attacklm/bench/packs.py`):

| Pack | What it measures | Scoring | Data |
| :--- | :--- | :--- | :--- |
| `cybermetric-500` | Cybersecurity knowledge MCQ (CyberMetric-500) | harness-owned (`inspect_evals`) | via Inspect AI |
| `ctibench-mcq` | CTI knowledge MCQ | local `mcq_choice` | CC BY-NC-SA, fetched at run time |
| `ctibench-ate` | MITRE ATT&CK technique-ID extraction | local `attack_technique_set` | CC BY-NC-SA, fetched at run time |
| `secbench-en` | English security knowledge MCQ | local `mcq_choice` | MIT, fetched at run time |
| `seceval` | Multi-select security MCQ | local `mcq_choice` | CC BY-NC-SA, fetched at run time |
| `applied-attack` | Applied ATT&CK reasoning + posture judgement | local, optional LLM judge | in-repo |

Fetched packs are downloaded into `data/bench/cache/` on first use;
their data is **never redistributed** — cached copies and raw results stay
local (`bench_results/` is gitignored); only the human-readable write-ups
under `docs/results/` are committed.

Two pack modes: **harness_scored** (the harness owns the dataset and
scoring — used when `inspect_evals` already implements the benchmark,
so we inherit its validated prompt and answer extraction) and
**local_scored** (we own the items and scoring; the harness only
generates completions).

Scorer and judge knobs worth knowing:

- `mcq_choice`: per-item bounded option alphabet (a 4-option question
  extracts only A–D, a 6-option one A–F), invalid-completion policies
  (`count_wrong` / `exclude` / `fail_run`), multi-select modes
  (`exact` / `partial`).
- `attack_technique_set` (backs CTI-Bench's ATE task): sub-technique
  policy `strip` (default — compare parent techniques only) / `keep` /
  `either`, scored as per-item micro-F1.
- **Contamination check** (`src/attacklm/bench/contamination.py`):
  every local-scored run can be checked against your training records
  with 20-gram MinHash Jaccard (default gate 0.80), plus a sensitivity
  curve at 1.0 / 0.9 / 0.8 / 0.7 / 0.5 and a per-source overlap
  breakdown — a flattering score from a leaked eval set is caught, not
  celebrated.
- **Judge validation** (`scripts/bench_judge_validate.py`): re-grades
  completions under the real question (not a placeholder) so an LLM
  judge's verdicts are reproducible before you trust them.

Run a pack directly:

```bash
python scripts/bench_run.py --pack cybermetric-500 \
    --base-model models/merged/attacklm-3b-16g \
    --output evals/bench_cybermetric.json
```

…or through the queue: all six packs are registered gauntlet members
(`bench_*` task types), so a gauntlet can pair audits with capability
benchmarks, and `attacklm queue compare` reports the Δ per benchmark
with a 95% bootstrap CI — same as it does for audits.

### Queue & gauntlets

`attacklm queue` chains a training run to a gauntlet of audits so
you can walk away and come back to reports. The headline flow:

```bash
attacklm queue chain --single-model --then gauntlet core
attacklm queue start --follow
```

Prefer to run it in the background and check in later:

```bash
attacklm queue start --detach            # forks a runner, prints its PID
attacklm queue status                    # or: attacklm queue list
```

Audit an already-trained model directly (no training task needed)
with `add-audit`:

```bash
attacklm queue add-audit --attack 1 --base-model models/merged/attacklm-3b-16g
attacklm queue start --exit-when-idle
```

`--base-model` may point at a merged model **or** at a PEFT adapter
directory — in the latter case the base model is auto-resolved from
the adapter's `adapter_config.json`. Reports land in
`evals/queue/artifacts/<task-id>/` (gitignored, local-only).

A base-model **baseline** is queued automatically alongside `chain`/
`gauntlet` (opt out with `--no-baseline`) so there's always something
to compare against.

#### Did it get worse?

```bash
attacklm queue chain --single-model --then gauntlet core
attacklm queue start --exit-when-idle
attacklm queue compare
```

`queue compare` pairs your latest run against its baseline item-by-item
and reports Δ with a 95% bootstrap confidence interval and a verdict:

| Category | n | A | B | Δ | 95% CI | Verdict |
|---|---|---|---|---|---|---|
| overall | 49 | 0.388 | 0.582 | +0.194 | [+0.071, +0.316] | **WORSE** |
| translation | 3 | 0.667 | 0.500 | −0.167 | [−0.500, +0.000] | n<5 |

`WORSE`/`BETTER` means the CI excludes zero on that side; `SAME` means
it doesn't (or the interval is degenerate); `n<5` means too few paired
items for a verdict at all; `n/a` means zero items paired (nothing to
compare, e.g. disjoint ids). Address either side by adapter path, a
merged model's path, or leave both blank to compare the newest run
against its baseline: `attacklm queue compare <base-model> <adapter-or-merged-path>`.
`attacklm queue history [--subject X] [--jsonl PATH]` lists every
completed audit, newest first, with its headline metric.

### Run the TUI

```bash
attacklm gui
```

A Textual-based terminal UI that runs over SSH, WSL, or headless
servers (no X11, no browser, no GPU required). Features:

- **Tabbed training form** — 40+ parameters across Basic, LoRA,
  GaLore, Advanced, Hardware tabs
- **Live training monitor** — loss sparkline, VRAM gauge, token
  throughput, scrolling log
- **Built-in presets** — one-click configurations for 3B/7B
  (Q-GaLore Spectrum, QLoRA, etc.)
- **Audit screen** — 2 tabs (Extraction / MIA), each form
  constructs the `attacklm audit` CLI command with hover
  tooltips on every field
- **Queue screen** — enqueue audits/gauntlets, start/stop the
  background runner, retry, a "Baseline" checkbox (default on;
  unchecking appends `--no-baseline`), a "Compare latest" button
  that streams `queue compare` into the log
- **Pause/Resume** — SIGSTOP/SIGCONT the training process without
  losing progress
- **One-click commands** — Init, Balance, Infer, Build, Eval,
  Audit, Demo all from the main menu

---

## Training methods

Choose by available VRAM and target quality:

| Method | Description | VRAM | Best for |
| :--- | :--- | :--- | :--- |
| **QLoRA** | 4-bit quantized base + LoRA adapters. Trains small adapter matrices only. | Lowest (~8GB for 3B) | Quick experiments, limited VRAM |
| **GaLore** | Full-parameter training with gradient low-rank projection. | Medium (~16GB for 3B) | Best quality on consumer GPUs |
| **Q-GaLore** | GaLore with quantization. | Medium-Low | High quality on 16GB GPUs |
| **Spectrum** | SNR-based layer freezing. Trains high-SNR layers only. | Medium | Reduces VRAM, speeds training |
| **PiSSA** | Principal Singular Values initialization for LoRA. | Same as QLoRA | Better convergence than standard LoRA |
| **DeepSpeed ZeRO-3** | Shards model across GPU + CPU RAM. Offloads params/optimizer to system memory. | Lowest (model 3-5× VRAM) | Training 40B+ on 16GB GPU |
| **COAP** | Compressed Optimizer Adaptive Parameterization. | Ultra-Low | Massive models on modest GPUs |
| **FlashOptim** | Optimized FlashAttention kernels. | Low | High-throughput training |
| **FP8** | Native 8-bit floating point. | Medium-Low | H100/Blackwell hardware |
| **BitNet** | 1.58-bit quantization. | Lowest | Near-zero VRAM training |
| **torch.compile** | PyTorch 2.x JIT compilation. Fuses operations. | 10-20% less than baseline | Any model, free performance |
| **LOMO** | Fuses gradient computation + parameter update. Never materializes full gradient. | Lowest (7B full-param on 8GB) | Full-parameter quality on tiny GPUs |

### Hardware reference

| GPU VRAM | System RAM | Recommended | Max model |
| :--- | :--- | :--- | :--- |
| 8 GB | 32 GB | ZeRO-2 + CPU offload | ~13B |
| 16 GB | 64 GB | ZeRO-3 + CPU offload | ~40B |
| 24 GB | 64 GB | ZeRO-3 + CPU offload | ~70B |
| 24 GB | 128 GB | ZeRO-3 + CPU offload | ~70B+ |
| H100/B100 | 128 GB+ | FP8 / FlashOptim | 175B+ |
| Any | 32 GB+ | BitNet / COAP | 100B+ (extreme quant) |

### DeepSpeed configs

Pre-built configs live in `presets/deepspeed/`:

| Config | ZeRO stage | CPU offload | Best for |
| :--- | :--- | :--- | :--- |
| `zero3_cpu_offload.json` | 3 | Params + Optimizer | Single GPU, model > VRAM |
| `zero3_gpu_only.json` | 3 | None | Multi-GPU setups |
| `zero2_cpu_offload.json` | 2 | Optimizer only | Faster, model ~2× VRAM |

Auto-generate a config (defaults to ZeRO-3 + CPU offload):

```bash
attacklm train -- --use-deepspeed --dataset data/balanced.jsonl --train
```

---

## Dataset & provenance

The training data lives in the separate
[Veedubin/attacklm-dataset](https://github.com/Veedubin/attacklm-dataset)
package. It is **not** a Python wheel — it's a data bundle with a
thin Python wrapper, distributed via a GitHub Releases tarball
(downloaded by `attacklm init`).

**Composition (26,459 records, 19 sources):**

| Category | Source examples | Approx. pairs | License |
| :--- | :--- | :--- | :--- |
| **Offensive** | Metasploit, Atomic Red Team, MITRE Stockpile | 15,000+ | BSD-3 / MIT / Apache-2.0 |
| **Defensive** | Sigma, Elastic, Splunk, Mordor, ThreatHunter | 7,000+ | DRL-1.1 / Apache-2.0 |
| **AI Security** | Garak, Promptfoo, PromptMap | 100+ | MIT / Apache-2.0 |
| **AI Security (ATLAS)** | MITRE ATLAS, ATLAS Arsenal | 1,800+ | Apache-2.0 |
| **Meta/IR** | NIST IR, Orchestrator | 500+ | Public Domain / MIT |
| **Synthetic** | LLM-generated, AttackLM synthetic, Replay | 2,000+ | GPL-3.0 / MIT |

**Per-record provenance.** Every record carries:

```json
{
  "source": "atomic-red-team",
  "source_uri": "https://github.com/redcanaryco/atomic-red-team",
  "license": "MIT",
  "license_uri": "https://opensource.org/licenses/MIT",
  "rights_contact": "see data/REMOVAL.md"
}
```

Three high-risk sources (RTA, infection_monkey, BPL) are excluded
from the public dataset and live only at
`archive/restricted-sources/` (gitignored, never re-ingested).

For the full per-record attribution, see
[data/ATTRIBUTION.md](https://github.com/Veedubin/attacklm-dataset/blob/main/data/ATTRIBUTION.md).
For the legal rights statement (the "trend" DMCA-style notice),
see
[attacklm-dataset/RIGHTS.md](https://github.com/Veedubin/attacklm-dataset/blob/main/RIGHTS.md).

---

## Research toolkit (audit)

`attacklm audit` is the privacy/security side of the package.
It is the **owner-side** test for memorization — the question is
"if I ship this model, what can an attacker extract from it?",
which the model owner wants to know *before* shipping.

### Roadmap: RL post-training
> Inspired by MAI-Thinking-1 §3.1.1 (Adaptive entropy control) by The Microsoft AI Team, June 2026. Full recipe: [docs/RL_RECIPE.md](docs/RL_RECIPE.md)

AttackLM currently supports SFT. Future versions will incorporate the MAI-Thinking-1 adaptive GRPO recipe to enable stable, reasoning-focused RL climbs. See the full recipe for details on entropy control and length penalties.

```bash
# Prompt injection audit (Attack 1)
attacklm audit prompt-injection --base-model ... --adapter ...

# System prompt audit (Attack 2)
attacklm audit system-prompt --base-model ... --adapter ...

# Canary extraction audit (Attack 3)
attacklm audit canary-extraction --base-model ... --adapter ... --canaries data/canaries.jsonl

# Calibration audit (Attack 7)
attacklm audit calibration --base-model ... --adapter ... --in-distribution ...

# Full audit (all attack classes, all MIA methods)
attacklm audit --attack all --mia-method per_token \
  --model models/attacklm-single_TIMESTAMP

# Just prefix-completion extraction
attacklm audit --attack extraction --max-records 100

# Just LiRA MIA (requires pre-computed shadow loss files)
attacklm audit --attack mia --mia-method lira \
  --lira-params shadow_params.json

# Dry run (stats only, no real model load)
attacklm audit --attack all --dry-run
```

Output is `data/audit/<date>/` with `summary.json` (aggregate
metrics, safe to share) and `inversion_results.jsonl` (raw
reconstructions, chmod 0600, stay workspace-internal).

The audit harness is built on the
[attacklm-dataset `scripts/inversion/`](https://github.com/Veedubin/attacklm-dataset/tree/main/scripts/inversion)
package. The methodology docs are maintainer-local and not
distributed; the attack classes and canonical papers are listed in
the [attacklm-dataset README](https://github.com/Veedubin/attacklm-dataset#privacy-audit-research-toolkit).

**All attack code is for defensive, audit, and academic-research use
only** — see [RIGHTS.md](https://github.com/Veedubin/attacklm-dataset/blob/main/RIGHTS.md).

---

## Documentation

| Doc | What it covers |
|-----|---------------|
| [RL Recipe](docs/RL_RECIPE.md) | Adaptive GRPO reinforcement learning recipe (MAI-Thinking-1 §3.1-3.4) |
| [Evaluation](EVALUATION.md) | Model evaluation methodology and benchmarks |
| [Contributing](CONTRIBUTING.md) | How to contribute to AttackLM |
| [Attribution](ATTRIBUTION.md) | Upstream source attribution and licensing |
| [Changelog](CHANGELOG.md) | Full release history |

---

## Architecture

```
                     ┌─────────────────────────────────────────┐
                     │           attacklm (this repo)          │
                     ├─────────────────────────────────────────┤
                     │                                         │
                     │   ┌─────────────┐    ┌──────────────┐  │
                     │   │   Trainer   │    │    Audit     │  │
                     │   │  (train.py) │    │  (audit.py)  │  │
                     │   └──────┬──────┘    └──────┬───────┘  │
                     │          │                 │          │
                     │   ┌──────┴─────────────────┴───────┐  │
                     │   │  TUI (Textual) — attacklm gui  │  │
                     │   └────────────────────────────────┘  │
                     │                                         │
                     └────────────┬────────────────────────────┘
                                  │ downloads tarball
                                  ▼
                     ┌─────────────────────────────────────────┐
                     │  Veedubin/attacklm-dataset (separate)  │
                     ├─────────────────────────────────────────┤
                     │  data/datasets/buckets/sources/<s>/...  │
                     │  scripts/inversion/{probe,scoring,     │
                     │    lira,shadow_train,...}              │
                     │  scripts/extract_<source>_to_jsonl.py  │
                     └─────────────────────────────────────────┘
```

The trainer and audit live in the same repo because they share
infrastructure: the same dataset (via `attacklm-dataset`), the same
inference code (`scripts/infer.py`), the same model artifact format
(adapters in `models/attacklm-single_TIMESTAMP/`). The split from
v0.11.0 isolates the *data* and the *attack code* (which is
defensive research) from the *training* and *user-facing tools*
(general-purpose infrastructure).

---

## CLI reference

`attacklm` is a tiered dispatcher. Top-level flags are handled by
the CLI itself; flags after `--` are forwarded to the underlying
scripts.

### Top-level commands

| Command | Purpose |
| :--- | :--- |
| `attacklm init` | Initialize the dataset (download tarball or build from source) |
| `attacklm balance` | Build a balanced training subset (anti-source-bias) |
| `attacklm train` | Core training engine (QLoRA, GaLore, DeepSpeed, etc.) |
| `attacklm build` | Merge LoRA adapter, convert to GGUF, register with Ollama |
| `attacklm infer` | Inference & smoke testing against representative prompts |
| `attacklm eval` | Retention evaluation, reference collection, regression gates |
| `attacklm audit` | **Research toolkit** — inversion attack audit (extraction + MIA) |
| `attacklm steer` | Steering-vector inference (activation intervention) |
| `attacklm bench` | Domain + speed benchmarks (capability benchmarks live in [Benchmark model capability](#benchmark-model-capability)) |
| `attacklm pipeline` | Run the full pipeline (init → balance → train → build) |
| `attacklm gui` | Launch the TUI |
| `attacklm demo` | Run the multi-agent orchestrator demo |

### `attacklm train` flags

**Dispatcher flags** (handled by the CLI):
- `--all` — Train all buckets (multi-model or single-model combined)
- `--hpo` — Hyperparameter optimization sweep

**Forwarded flags** (after `--`):
- `--dataset <path>` — Path to JSONL dataset or `all`
- `--base-model <model>` — HuggingFace model ID (default `Qwen/Qwen2.5-Coder-3B-Instruct`)
- `--epochs <n>` — default 3
- `--lora-r <n>`, `--lora-alpha <n>`, `--lora-dropout <n>` — LoRA config
- `--use-galore`, `--use-qgalore`, `--spectrum`, `--use-pissa` — method toggles
- `--use-deepspeed`, `--deepspeed-stage {1,2,3}`, `--deepspeed-config <path>` — DeepSpeed
- `--use-lomo`, `--use-coap`, `--use-flashoptim`, `--use-fp8`, `--use-bitnet`, `--use-signround` — advanced optimizers
- `--compile`, `--compile-mode {default,reduce-overhead,max-autotune}` — torch.compile
- `--evolved-ratio <n>`, `--evolved-dir <path>` — evolved pairs
- `--replay-ratio <n>`, `--replay-source <path>` — anti-forgetting replay
- `--single-model` — combine all buckets into one
- `--multi-gpu`, `--moe-safe-target`, `--use-unsloth` — hardware
- `--train` — execute (omitting performs a dry-run with stats)
- `--dry-run`, `--force`, `--resume-from-checkpoint` — execution

Full flag list: `attacklm train --help`. See
[scripts/train_template.py](scripts/train_template.py) for the
canonical training command with QLoRA/GaLore/PiSSA/Spectrum/DeepSpeed
attribution comments.

### `attacklm audit` flags

- `--attack {extraction,mia,all}` — attack class (default `all`)
- `--mia-method {reference,zlib,per_token,lira,all}` — MIA scoring (default `per_token`)
- `--mia-threshold-mode {median,percentile,holdout_file,lrt}` — threshold derivation
- `--mia-percentile <n>` — percentile for threshold (default 5)
- `--model <path>` — model being audited
- `--source-filter <name>` — restrict to specific source(s)
- `--top-k <n>` — top-k candidates to evaluate
- `--max-records <n>` — cap on records to audit
- `--max-new-tokens <n>`, `--temperature <n>` — generation params
- `--lira-k <n>` — number of shadow models for LiRA (default 16)
- `--lira-params <path>` — path to `shadow_params.json`
- `--dry-run` — stats only, no real model load

Full flag list: `attacklm audit --help`.

### `attacklm queue` subcommands

| Subcommand | Purpose |
| :--- | :--- |
| `add-train` | Add a training task |
| `add-audit` | Add an audit task (`--attack 1-7`, `--base-model`, `--adapter`) |
| `add-holdouts` | Add a calibration-holdout generation task |
| `chain` | Chain a train task followed by a gauntlet or audit |
| `gauntlet` | Add a gauntlet of audit tasks (`core`/`full`/`quick`/`memorization`) |
| `baseline` | Queue a base-model baseline gauntlet (`--preset`, `--force`) |
| `compare` | Compare a subject vs. its baseline or another subject (`--preset`, `--json`, `--resamples`, `--seed`) |
| `history` | List completed audits, newest first (`--subject`, `--limit`, `--jsonl`) |
| `list` / `status` | List tasks / show queue or single-task status |
| `start` | Start the runner (`--follow`, `--detach`, `--exit-when-idle`) |
| `stop` | Stop the runner after the current task |
| `remove` | Remove a task (refuses tasks in `running` status) |
| `retry` | Retry a failed/interrupted task |
| `clean` | Delete completed/failed tasks (`--yes` to confirm) |
| `reset` | DANGER: drop and recreate the queue database |

Full flag list per subcommand: `attacklm queue <subcommand> --help`.

---

## Testing

AttackLM is **defensive-tested**, not just smoke-tested. As of
v0.21.0 there are 1,061+ tests across 50+ test files, all hermetic
(no network, no GPU required, fast enough to run in CI on every
PR):

```
tests/test_audit.py + test_attack_audit.py   (research toolkit + attack framework)
tests/test_bench_*.py              (20 files — capability benchmark)
tests/test_cli.py                  (CLI dispatcher)
tests/test_coap_flashoptim.py + test_memory_optimization.py   (memory optimizers)
tests/test_collect_reference.py + test_compare_scores.py
+ test_eval_*.py + test_score_candidates.py   (eval suite)
tests/test_domain_bench.py + test_speed_bench.py   (domain + speed benchmarks)
tests/test_fp8_bitnet.py + test_mixed_precision.py   (quantized training)
tests/test_golden_vectors.py       (regression gates)
tests/test_gui*.py                 (6 files — TUI smoke, presets, queue, tooltips)
tests/test_neuralgentics_*.py      (init flow + merge)
tests/test_project_root.py         (repo layout)
tests/test_queue*.py               (6 files — queue, baseline, compare, history, stats)
tests/test_steering.py             (steering vectors)
tests/test_training_integration.py (end-to-end on a tiny model)
```

Run them all:

```bash
pip install -e ".[all]"
pytest tests/ -v
```

The `test_training_integration.py::TestTrainingIntegration::test_tiny_model_one_step`
test loads a 1-layer LlamaForCausalLM, runs one forward+backward pass
through `attacklm.train` on a single 16-token example, and asserts
the loss is finite. This is the canary for "did someone break the
core training loop?".

The `test_gui.py` tests mount the TUI under a `run_test()` pilot
and assert every main-menu button, every tooltip key, and every
command-form widget is present. Catches "I added a new feature
but forgot to add a tooltip" regressions.

---

## License & contributing

- **Code**: [MIT](LICENSE)
- **Training data**: Mixed per-source — see
  [data/ATTRIBUTION.md](https://github.com/Veedubin/attacklm-dataset/blob/main/data/ATTRIBUTION.md)
  and [RIGHTS.md](https://github.com/Veedubin/attacklm-dataset/blob/main/RIGHTS.md)
- **Audit / research-tool code**: Defensive, audit, and academic-
  research use only. See
  [RIGHTS.md](https://github.com/Veedubin/attacklm-dataset/blob/main/RIGHTS.md)
  for the full rights statement and the canonical paper list.

**Contributing**: PRs welcome. For dataset changes, edit the
extractors in
[attacklm-dataset/scripts/](https://github.com/Veedubin/attacklm-dataset/tree/main/scripts)
and re-run `attacklm init --from-source`. For training-method
additions, edit `scripts/train_template.py`. For audit additions,
add a module to `attacklm-dataset/scripts/inversion/` and a CLI
flag in `attacklm audit`.

[CHANGELOG.md](CHANGELOG.md) — full version history.
For roadmap, bugs, and feature requests, open an issue on
[GitHub Issues](https://github.com/Veedubin/AttackLM/issues).

