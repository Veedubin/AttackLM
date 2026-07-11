# AttackLM

[![PyPI version](https://img.shields.io/pypi/v/attacklm.svg?label=version&color=blue)](https://pypi.org/project/attacklm/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://docs.python.org/3.10/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests: 504+](https://img.shields.io/badge/tests-504%2B-brightgreen.svg)](#testing)
[![GH release: v0.12.3](https://img.shields.io/badge/release-v0.12.3-blue.svg)](https://github.com/Veedubin/AttackLM/releases)

**A security-AI fine-tuning platform and research toolkit.**

AttackLM is two things in one package:

1. **A fine-tuning pipeline** for MITRE ATT&CK-grounded security LLMs.
   Trains Qwen2.5-Coder, DeepSeek, and other open models on a curated
   security corpus (24K+ training pairs, 18 sources) using parameter-
   efficient methods (QLoRA, GaLore, PiSSA, Spectrum) and full-parameter
   methods (DeepSpeed ZeRO-3 + CPU offload, LOMO, FP8, BitNet).

2. **A research toolkit** for owner-side model security testing.
   `attacklm audit` runs inversion attacks (Carlini 2021 prefix-
   completion extraction, Carlini 2022 reference attack, per-token
   loss, and LiRA shadow-model MIA) against your own model so you
   can quantify what it memorized before deployment.

Both are wired into a **terminal GUI** (`attacklm gui`) for
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
attacklm --version       # 0.12.3
attacklm --help
pytest tests/ -q         # 504+ passed
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
to pick a MIA variant. See
[attacklm-dataset/docs/ATTACK_TAXONOMY.md](https://github.com/Veedubin/attacklm-dataset/blob/main/docs/ATTACK_TAXONOMY.md)
for the full design.

The audit harness is **hermetic** — it does not call out to any
network, does not require GPU, runs on a CPU laptop in minutes.
Mocked model loaders mean you can test the audit pipeline in CI
without owning a real model.

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

**Composition (24,652 records, 18 sources, 11 active):**

| Category | Source examples | Approx. pairs | License |
| :--- | :--- | :--- | :--- |
| **Offensive** | Metasploit, Atomic Red Team, MITRE Stockpile | 15,000+ | BSD-3 / MIT / Apache-2.0 |
| **Defensive** | Sigma, Elastic, Splunk, Mordor, ThreatHunter | 7,000+ | DRL-1.1 / Apache-2.0 |
| **AI Security** | Garak, Promptfoo, PromptMap | 100+ | MIT / Apache-2.0 |
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

```bash
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
package and the design is documented in:

- [ATTACK_TAXONOMY.md](https://github.com/Veedubin/attacklm-dataset/blob/main/docs/ATTACK_TAXONOMY.md)
  — the 3-attack-class taxonomy, the LLM MI = TDE collapse argument,
  and the CLI flag mapping
- [LIRA.md](https://github.com/Veedubin/attacklm-dataset/blob/main/docs/LIRA.md)
  — LiRA design, K parameter guide, compute cost
- [MIA_THRESHOLD_CALIBRATION.md](https://github.com/Veedubin/attacklm-dataset/blob/main/docs/MIA_THRESHOLD_CALIBRATION.md)
  — threshold calibration design

**All attack code is for defensive, audit, and academic-research use
only** — see [RIGHTS.md](https://github.com/Veedubin/attacklm-dataset/blob/main/RIGHTS.md).

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
| `attacklm bench` | Domain-specific + speed benchmarks |
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

---

## Testing

AttackLM is **defensive-tested**, not just smoke-tested. As of
v0.12.3 there are 504+ tests across 24 test files, all hermetic
(no network, no GPU required, fast enough to run in CI on every
PR):

```
tests/test_audit.py                  (research toolkit)
tests/test_cli.py                    (CLI dispatcher)
tests/test_coap_flashoptim.py        (memory optimizers)
tests/test_collect_reference.py       (eval suite)
tests/test_compare_scores.py         (eval suite)
tests/test_domain_bench.py           (benchmarks)
tests/test_eval_loader.py            (eval suite)
tests/test_eval_retention.py         (eval suite)
tests/test_fp8_bitnet.py             (quantized training)
tests/test_golden_vectors.py         (regression gates)
tests/test_gui.py                    (TUI smoke + tooltip coverage)
tests/test_memory_optimization.py    (memory optimizers)
tests/test_mixed_precision.py        (FP8/BF16)
tests/test_neuralgentics_init.py     (init flow)
tests/test_score_candidates.py       (eval suite)
tests/test_speed_bench.py            (benchmarks)
tests/test_steering.py               (steering vectors)
tests/test_training_integration.py   (end-to-end on a tiny model)
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
[TASKS.md](../TASKS.md) — current work-in-progress.
[HANDOFF.md](../HANDOFF.md) — session continuity notes.
