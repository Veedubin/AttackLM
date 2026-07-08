# AttackLM

[![PyPI version](https://img.shields.io/pypi/v/attacklm.svg?label=version&color=blue)](https://pypi.org/project/attacklm/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://docs.python.org/3.10/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)

**A high-performance fine-tuning pipeline (QLoRA, GaLore, Q-GaLore, Spectrum, PiSSA, DeepSpeed, COAP, FlashOptim, FP8, BitNet) for creating MITRE ATT&CK-grounded security AI assistants.**

---

## Table of Contents
- [Quickstart](#quickstart)
- [Installation Guide](#installation-guide)
- [Features](#features)
- [Usage](#usage)
- [Dataset & Attribution](#dataset--attribution)
- [Architecture](#architecture)
- [CLI Reference](#cli-reference)
- [License & Contributing](#license--contributing)

---

## Quickstart

Get from zero to a trained security model in four commands:

```bash
# 1. Install the full training stack (includes dataset)
pip install "attacklm[all]"

> **Note**: The MITRE ATT&CK dataset is now a separate package. `attacklm init` will
> automatically use [attacklm-dataset](https://github.com/Veedubin/attacklm-dataset)
> if installed, or guide you to install it.

# 2. Initialize the MITRE-grounded dataset
attacklm init --yes

# 3. Balance the dataset to prevent source-bias (e.g., Metasploit overfitting)
attacklm balance

# 4. Launch training on Qwen2.5-Coder
attacklm train -- --dataset all --epochs 5 --train
```

---

## Installation Guide

### Prerequisites
- **OS**: Linux (Ubuntu recommended) or WSL2
- **Python**: 3.10+
- **Hardware**: NVIDIA GPU with 8GB+ VRAM (RTX 30-series/40-series) or AMD ROCm compatible GPU.

### Installation
Install based on your hardware acceleration preference:

**NVIDIA CUDA (Recommended)**
```bash
pip install "attacklm[all]"
# OR using uv for faster installation
uv pip install "attacklm[all]"
```

**AMD ROCm**
```bash
pip install "attacklm[all-rocm]"
```

**Verification**
```bash
attacklm --version
```

### Memory Optimization: Flash-Attention vs. SDP
AttackLM is designed for maximum accessibility without sacrificing the efficiency of modern attention mechanisms.

**The Problem with `flash-attn`**
Traditional `flash-attn` installations require a full CUDA toolkit, specific NVCC versions, and lengthy source compilation, which frequently fails in constrained environments or varying OS versions.

**The AttackLM Solution: Memory Efficient SDP**
By default, AttackLM leverages PyTorch's built-in `torch.backends.cuda.enable_mem_efficient_sdp()`. 

- **Technical Advantage**: It implements the same $\mathcal{O}(1)$ tiled algorithm as FlashAttention. 
- **VRAM Impact**: At a sequence length of 12,000, a standard $\mathcal{O}(n^2)$ attention matrix would consume ~8GB of VRAM just for the matrix. Memory Efficient SDP keeps this overhead constant.
- **Zero Friction**: No compilation required. It works natively across all supported PyTorch/CUDA environments.

*Note: If you have a perfectly configured environment and want the absolute maximum throughput, you can still install the standalone flash-attention: `pip install "attacklm[flash-attn]"`.*

---

## Features

- **Comprehensive Security Corpus**: 24,652 high-quality training pairs across 16 security sources (via the [attacklm-dataset](https://github.com/Veedubin/attacklm-dataset) package).
- **Advanced Training Methods**: Support for QLoRA, GaLore, Q-GaLore, Spectrum, and PiSSA to enable training of large models on consumer hardware.
- **Training Pair Evolution**: New capability to synthetically expand short, factual pairs into complex reasoning examples using three specialized strategies:

| Strategy | Approach | Impact |
| :--- | :--- | :--- |
| **Evol-Instruct** | Rewrites responses with deeper reasoning and edge cases | 3-5x increase in response length/depth |
| **Multi-turn** | Decomposes Q&A into interactive conversations | Improved conversational flow and context |
| **CoT Injection** | Adds explicit "Chain-of-Thought" reasoning steps | Higher logical consistency in complex tasks |

- **Memory Optimization**: Seven advanced techniques to train larger models on consumer hardware:
  - **COAP** — Compressed Optimizer Adaptive Parameterization. Drastic reduction in optimizer state VRAM.
  - **FlashOptim** — Optimized FlashAttention-2 kernels for specific sequence lengths.
  - **Unsloth GC** — Aggressive garbage collection and memory pinning for LoRA/QLoRA.
  - **Mixed-precision LoRA** — Strategic use of FP8/BF16 across adapter layers.
  - **FP8 Training** — Native 8-bit floating point training (H100/Blackwell).
  - **BitNet** — 1.58-bit quantization for near-zero VRAM training.
  - **SignRoundV2** — Advanced stochastic rounding for low-bit weights.
  - **DeepSpeed ZeRO-3 + CPU Offload** — Shards parameters, gradients, and optimizer states across GPU VRAM + system RAM. Train 40B+ parameter models on a 16GB GPU with 64GB system RAM.
  - **torch.compile** — PyTorch 2.x JIT compilation. 20-40% training speedup with 10-20% memory reduction. One flag: `--compile`.
  - **LOMO Optimizer** — Full-parameter fine-tuning (not just adapters) of 7B models on 8GB GPUs.

- **Zero-Config Setup**: One-shot `init` command that handles dataset retrieval, extraction, and bucket organization.
- **Anti-Bias Balancing**: Integrated balancing engine to ensure the model learns diverse tactics rather than just the most voluminous sources.
- **Provenance Tracking**: Strict per-source attribution and license tracking for every record in the dataset.
- **Terminal GUI**: A professional Textual-based TUI that eliminates the need to memorize 40+ CLI flags. Features include:
  - **Tabbed Training Form** — 40+ parameters organized across Basic, LoRA, GaLore, Advanced, and Hardware tabs
  - **Live Training Monitor** — Real-time loss sparkline, VRAM gauge, token throughput, and scrolling log output
  - **Built-in Presets** — One-click configurations for 3B/7B models (Q-GaLore Spectrum, QLoRA, etc.)
  - **One-Click Commands** — Init, Balance, Infer, Build, and Eval all accessible from the main menu
  - **Pause/Resume Controls** — SIGSTOP/SIGCONT the training process without losing progress
  - **Zero Dependencies** — No X11, no GPU, no browser required. Works over SSH, WSL, and headless servers.
- **Deployment Ready**: Built-in merge and conversion pipeline to export adapters to GGUF format for LM Studio or Ollama.
- **Rock-Solid Stability**: 26/26 core tests passing.

---

## Usage & Workflows

AttackLM provides a set of curated workflows to take you from raw data to a deployed security model.

### Common Workflows

**Workflow 1: Quick Start (5 minutes to training)**
```bash
# 1. Install the full training stack
pip install "attacklm[all]"

# 2. Initialize the MITRE-grounded dataset
attacklm init --yes

# 3. Balance the dataset for your hardware (e.g., 7B model on 16GB VRAM)
attacklm balance --profile 7b-16gb --preset red-team

# 4. Launch training on Qwen2.5-Coder
attacklm train -- --dataset data/datasets/balanced/balanced_7b-16gb.jsonl --epochs 10 --train
```

**Workflow 2: Maximum Quality (GaLore + Spectrum + Evolved Pairs)**
```bash
attacklm init --yes
attacklm balance --profile 7b-16gb
attacklm train --all -- --single-model --use-galore --spectrum --evolved-ratio 0.2 --epochs 20 --train
```

**Workflow 3: HPO $\rightarrow$ Train $\rightarrow$ Deploy**
```bash
# 1. Run Hyper-Parameter Optimization sweep to find best settings
attacklm train --hpo -- --dataset data/datasets/balanced/balanced_7b-16gb.jsonl

# 2. Train with optimized parameters
attacklm train -- --dataset data/datasets/balanced/balanced_7b-16gb.jsonl --lora-r 64 --lora-alpha 128 --epochs 15 --train

# 3. Merge adapter and convert to GGUF for local deployment
attacklm build -- --adapter models/attacklm_TIMESTAMP --name attacklm-v1
```

**Workflow 4: Evolve Pairs $\rightarrow$ Filter $\rightarrow$ Train**
```bash
# 1. Synthetically expand short pairs into complex reasoning examples
python scripts/evolve_pairs.py --strategy all --source metasploit-framework --count 500

# 2. Filter evolved pairs for quality
python scripts/filter_evolved.py --input data/datasets/evolved/ --all

# 3. Train with a high ratio of evolved pairs
attacklm train --all -- --single-model --evolved-ratio 0.3 --epochs 10 --train
```

**Workflow 5: Train 40B+ on 16GB GPU (DeepSpeed + CPU offload)**
```bash
# Workflow 5: Train 40B+ on 16GB GPU (DeepSpeed + CPU offload)
attacklm init --yes
attacklm balance --profile 7b-16gb
attacklm train -- --dataset data/datasets/balanced/balanced_7b-16gb.jsonl \
  --base-model Qwen/Qwen2.5-32B-Instruct \
  --use-deepspeed --deepspeed-stage 3 \
  --compile --epochs 5 --train
```

### Training Methods Explained

Choose your training method based on your available VRAM and quality requirements:

| Method | Description | VRAM | Best For |
| :--- | :--- | :--- | :--- |
| **QLoRA** | 4-bit quantized base + LoRA adapters. Only trains small adapter matrices. | Lowest (~8GB for 3B) | Quick experiments, limited VRAM |
| **GaLore** | Full-parameter training with gradient low-rank projection. | Medium (~16GB for 3B) | Best quality on consumer GPUs |
| **Q-GaLore** | GaLore with quantization. Balances quality and VRAM. | Medium-Low | High quality on 16GB GPUs |
| **Spectrum** | SNR-based layer freezing. Freezes low-SNR layers, trains high-SNR. | Medium | Reduces VRAM, speeds training |
| **PiSSA** | Principal Singular Values initialization for LoRA. | Same as QLoRA | Better convergence than standard LoRA |
| **DeepSpeed ZeRO-3** | Shards model across GPU + CPU RAM. Offloads params and optimizer to system memory. | Lowest (model 3-5x VRAM) | Training 40B+ on 16GB GPU |
| **COAP** | Compressed Optimizer Adaptive Parameterization. | Ultra-Low | Massive models on modest GPUs |
| **FlashOptim** | Optimized FlashAttention kernels. | Low | High-throughput training |
| **FP8** | Native 8-bit floating point. | Medium-Low | H100/Blackwell hardware |
| **BitNet** | 1.58-bit quantization. | Lowest | Near-zero VRAM training |
| **torch.compile** | PyTorch 2.x JIT compilation. Fuses operations for speed + memory. | 10-20% less than baseline | Any model, free performance |
| **LOMO** | Fuses gradient computation + parameter update. Never materializes full gradient. | Lowest (7B full-param on 8GB) | Full-parameter quality on tiny GPUs |

### Terminal GUI

For an interactive experience, use `attacklm gui`. This eliminates the need to memorize dozens of CLI flags and provides a real-time training dashboard with VRAM gauges and loss sparklines.

```bash
attacklm gui
```
---


---

## Dataset & Attribution

The dataset is meticulously partitioned into "buckets" to allow granular control over training composition.

### Core Composition
| Category | Source Examples | Approx. Pairs | License |
| :--- | :--- | :--- | :--- |
| **Offensive** | Metasploit, Atomic Red Team | 15,000 | BSD-3 / MIT |
| **Defensive** | Sigma, Elastic, Splunk | 7,000 | DRL-1.1 / Apache-2.0 |
| **AI Security** | Garak, Promptfoo | 1,652 | Mixed |
| **Meta/IR** | NIST IR, Orchestrator | 1,000 | Public Domain |

**Total Records**: 24,652  
**Base Models**: Qwen2.5-Coder (3B, 7B)

For a complete mapping of every record to its original source and license, see [ATTRIBUTION.md](ATTRIBUTION.md).

---

## Architecture

AttackLM employs a deterministic pipeline that separates raw data extraction from training logic.

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

This hierarchy ensures that the pipeline can be rebuilt from upstream sources without introducing hallucinations, while allowing the `balance` command to target specific tactics or sources for weighted sampling.

---

## CLI Reference

AttackLM uses a tiered command structure. Top-level flags are handled by the dispatcher, while flags following the `--` separator are forwarded directly to the specialized training or inference scripts.

### 1. `attacklm train` — Core Training Engine
The primary entry point for fine-tuning. Supports a variety of parameter-efficient and full-parameter methods.

**Dispatcher Flags**
- `--all` — Train all buckets (multi-model or single-model combined)
- `--hpo` — Run hyperparameter optimization sweep instead of standard training

**Forwarded Arguments (Post-`--`)**
- `--dataset <path>` — Path to JSONL dataset or `all` for all buckets
- `--base-model <model>` — HuggingFace model ID (default: `Qwen/Qwen2.5-Coder-3B-Instruct`)
- `--output <dir>` — Output directory for trained model
- `--epochs <n>` — Number of training epochs (default: 3)
- `--batch-size <n>` — Per-device batch size (default: 1)
- `--max-length <n>` — Maximum sequence length in tokens (default: 1024)
- `--lora-r <n>` — LoRA rank (default: 16)
- `--lora-alpha <n>` — LoRA alpha scaling (default: 32)
- `--lora-dropout <n>` — LoRA dropout rate (default: 0.05)
- `--use-galore` — Enable GaLore full-parameter training
- `--use-qgalore` — Enable Q-GaLore (quantized GaLore)
- `--spectrum` — Enable Spectrum layer freezing (SNR-based)
- `--use-pissa` — Enable PiSSA initialization
- `--packing` — Enable example packing for throughput
- `--train` — Execute training (omitting this performs a dry-run with stats)
- `--eval-split <n>` — Fraction held out for eval (default: 0.1)
- `--early-stop-steps <n>` — Early stopping patience in eval steps
- `--save-steps <n>` — Save checkpoint every N steps
- `--gradient-accumulation-steps <n>` — Gradient accumulation steps
- `--evolved-ratio <n>` — Fraction of training pairs from evolved datasets (0.0-1.0)
- `--evolved-dir <path>` — Directory containing evolved JSONL files
- `--replay-ratio <n>` — Fraction of replay (anti-forgetting) examples
- `--replay-source <path>` — Path to replay source directory
- `--single-model` — Combine all buckets into one training set
- `--include-orchestrator` — Include orchestrator bucket
- `--model-attacks` — Include AI model attack buckets
- `--include-tools` — Include tool buckets
- `--moe-safe-target` — Disable 4-bit quantization for MoE models
- `--multi-gpu` — Enable multi-GPU training
- `--use-unsloth` — Use Unsloth for faster training
- `--resume-from-checkpoint` — Resume from last checkpoint
- `--force` — Force re-tokenization (ignore cache)
- `--dry-run` — Print what would run without executing
- `--use-deepspeed` — Enable DeepSpeed ZeRO optimization
- `--deepspeed-stage {1,2,3}` — ZeRO stage (default: 3)
- `--deepspeed-config <path>` — Path to custom DeepSpeed JSON config
- `--no-deepspeed-offload` — Disable CPU offload (GPU-only ZeRO)
- `--compile` — Enable torch.compile (20-40% speedup)
- `--compile-mode {default,reduce-overhead,max-autotune}` — torch.compile mode (default: reduce-overhead)
- `--use-lomo` — Enable LOMO full-parameter optimizer
- `--use-coap` — Enable Compressed Optimizer Adaptive Parameterization
- `--use-flashoptim` — Enable optimized FlashAttention kernels
- `--use-fp8` — Enable native FP8 training (H100/Blackwell)
- `--use-bitnet` — Enable BitNet 1.58b quantization
- `--use-signround` — Enable SignRoundV2 stochastic rounding

**Examples**
```bash
# Single dataset, QLoRA
attacklm train -- --dataset data/balanced.jsonl --epochs 10 --train

# All buckets, GaLore + Spectrum, single model
attacklm train --all -- --single-model --use-galore --spectrum --epochs 20 --train

# With 20% evolved pairs
attacklm train --all -- --single-model --evolved-ratio 0.2 --epochs 10 --train

# HPO sweep
attacklm train --hpo -- --analyze-only

# Dry-run stats only
attacklm train -- --dataset data/balanced.jsonl
```

### DeepSpeed Configuration

AttackLM ships with pre-built DeepSpeed configs in `presets/deepspeed/`:

| Config | ZeRO Stage | CPU Offload | Best For |
|--------|-----------|-------------|----------|
| `zero3_cpu_offload.json` | 3 | Params + Optimizer | Single GPU, model > VRAM |
| `zero3_gpu_only.json` | 3 | None | Multi-GPU setups |
| `zero2_cpu_offload.json` | 2 | Optimizer only | Faster, model ~2x VRAM |

Auto-generate a config (defaults to ZeRO-3 + CPU offload):
```bash
attacklm train -- --use-deepspeed --dataset data/balanced.jsonl --train
```

Use a custom config:
```bash
attacklm train -- --use-deepspeed --deepspeed-config presets/deepspeed/zero2_cpu_offload.json --train
```

### Hardware Reference

| GPU VRAM | System RAM | Recommended Config | Max Model |
|----------|-----------|-------------------|-----------|
| 8 GB | 32 GB | ZeRO-2 + CPU offload | ~13B |
| 16 GB | 64 GB | ZeRO-3 + CPU offload | ~40B |
| 24 GB | 64 GB | ZeRO-3 + CPU offload | ~70B |
| 24 GB | 128 GB | ZeRO-3 + CPU offload | ~70B+ |
| H100/B100 | 128 GB+ | FP8 / FlashOptim | 175B+ |
| Any GPU | 32 GB+ | BitNet / COAP | 100B+ (Extreme Quant) |
```

---

### 2. `attacklm init` — Dataset Initialization
Handles the retrieval and organization of the security corpus.

**Flags**
- `--yes` — Skip confirmation prompts
- `--from-source` — Build from upstream git repos instead of downloading pre-built tarball
- `--dataset-url <url>` — Override download URL
- `--extract-only` — Run data extractors only
- `--buckets-only` — Organize data into buckets only
- `--attribute-only` — Add source/license attribution only
- `--clone-only` — Clone upstream repos only

**Examples**
```bash
# Default: download pre-built dataset
attacklm init --yes

# Build from source (clone repos, extract, attribute, bucket)
attacklm init --from-source

# Re-extract only (after updating extractors)
attacklm init --extract-only
```

---

### 3. `attacklm balance` — Subset Generation
Builds balanced training subsets to prevent overfitting to high-volume sources.

**Flags**
- `--profile <name>` — Hardware profile (`3b-16gb`, `7b-16gb`, `7b-24gb`, `7b-128gb`, `custom`)
- `--preset <name>` — Team preset (`red-team`, `blue-team`, `purple-team`)
- `--target-total <n>` — Target total pairs (for `custom` profile)
- `--strategy <name>` — Sampling strategy (`head` for highest quality first, `random`)
- `--dry-run` — Preview balancing without writing files

**Examples**
```bash
# Red-team preset for 7B on 16GB
attacklm balance --profile 7b-16gb --preset red-team

# Custom: 12,000 pairs, head strategy
attacklm balance --profile custom --target-total 12000 --strategy head

# Preview without writing
attacklm balance --profile 7b-16gb --dry-run
```

---

### 4. `attacklm build` — Model Deployment
Pipeline to merge adapters and export for local LLM runtimes.

**Flags**
- `--merge-only` — Merge LoRA adapter into base model only
- `--gguf-only` — Convert to GGUF format only
- `--register-ollama` — Register with local Ollama instance
- `--adapter <path>` — Path to trained adapter directory
- `--name <name>` — Output model name
- `--quant <type>` — Quantization: `q4_k_m`, `q5_k_m`, `q8_0`, `f16`

**Examples**
```bash
# Full build pipeline: Merge $\rightarrow$ GGUF $\rightarrow$ Ollama
attacklm build -- --adapter models/attacklm-single_TIMESTAMP --name attacklm-security

# Merge only
attacklm build --merge-only -- --adapter models/attacklm-single_TIMESTAMP

# GGUF only
attacklm build --gguf-only -- --adapter models/attacklm-single_TIMESTAMP --quant q4_k_m
```

---

### 5. `attacklm infer` — Inference & Smoke Testing
Test trained models against representative security prompts.

**Flags**
- `--adapter <path>` — Path to trained adapter
- `--base-model <model>` — Base model (default: `Qwen2.5-Coder-3B`)
- `--prompt <text>` — Single prompt to test
- `--prompts-file <path>` — JSONL file of prompts
- `--max-new-tokens <n>` — Max tokens to generate
- `--temperature <n>` — Sampling temperature

**Examples**
```bash
# Single prompt
attacklm infer -- --adapter models/attacklm-single_TIMESTAMP --prompt "How do I perform T1059.001?"

# Batch from file
attacklm infer -- --adapter models/attacklm-single_TIMESTAMP --prompts-file prompts.jsonl
```

---

### 6. `attacklm eval` — Evaluation Suite
Run retention evaluation and regression gates.

**Flags**
- `--collect-ref` — Generate reference continuations from base model
- `--score` — Score candidate models against references
- `--compare` — Compare two score TSV files
- `--golden` — Golden vector generation/validation
- `--adapter <path>` — Path to trained adapter
- `--base-model <model>` — Base model for reference collection

**Examples**
```bash
# Full retention evaluation
attacklm eval -- --adapter models/attacklm-single_TIMESTAMP

# Collect reference outputs
attacklm eval --collect-ref -- --base-model Qwen/Qwen2.5-Coder-3B-Instruct

# Score a candidate
attacklm eval --score -- --adapter models/attacklm-single_TIMESTAMP
```

---

### 7. `attacklm gui` — Terminal Interface
Launch the professional Textual-based TUI. Requires `pip install attacklm-gui`.

```bash
attacklm gui
```

### 8. `attacklm demo` — Orchestrator Demo
Run a demonstration of the AttackLM multi-agent orchestrator.

```bash
attacklm demo
```


---

## License & Contributing

**Code License**: This project is licensed under the [MIT License](LICENSE).

**Data License**: Training data consists of mixed licenses per source. Please refer to [ATTRIBUTION.md](ATTRIBUTION.md) for the full legal mapping.

**Contributing**: We welcome contributions to the extraction pipeline and training methods. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**History**: For a full list of changes and version milestones, see [CHANGELOG.md](CHANGELOG.md).
