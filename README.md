# AttackLM

[![PyPI version](https://img.shields.io/pypi/v/attacklm.svg?label=version&color=blue)](https://pypi.org/project/attacklm/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://docs.python.org/3.10/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)

**A high-performance QLoRA fine-tuning pipeline for creating MITRE ATT&CK-grounded security AI assistants.**

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
# 1. Install the full training stack
pip install "attacklm[all]"

# 2. Initialize the MITRE-grounded dataset (downloads pre-built tarball)
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

- **Comprehensive Security Corpus**: 24,652 high-quality training pairs across 18 distinct security sources.
- **Advanced Training Methods**: Support for QLoRA, GaLore, Q-GaLore, Spectrum, and PiSSA to enable training of large models on consumer hardware.
- **Zero-Config Setup**: One-shot `init` command that handles dataset retrieval, extraction, and bucket organization.
- **Anti-Bias Balancing**: Integrated balancing engine to ensure the model learns diverse tactics rather than just the most voluminous sources.
- **Provenance Tracking**: Strict per-source attribution and license tracking for every record in the dataset.
- **Terminal GUI**: A professional Textual-based TUI for managing training runs without memorizing 40+ CLI flags.
- **Deployment Ready**: Built-in merge and conversion pipeline to export adapters to GGUF format for LM Studio or Ollama.
- **Rock-Solid Stability**: 26/26 core tests passing.

---

## Usage

### Dataset Management
`attacklm init`
Initialize the environment. Downloads the pre-built dataset for instant use.
```bash
attacklm init --yes
```

`attacklm balance`
Create a balanced training subset to ensure tactical coverage.
```bash
attacklm balance --profile 7b-16gb --preset red-team
```

### Model Training
`attacklm train`
The core training engine. Supports Qwen2.5-Coder 3B and 7B base models.
```bash
# Train a single model on the entire balanced dataset
attacklm train -- --dataset all --epochs 10 --lora-r 16 --use-galore
```

### Deployment & Testing
`attacklm build`
Merge LoRA adapters and convert to GGUF for local deployment.
```bash
attacklm build -- --adapter models/attacklm-single_TIMESTAMP --name attacklm-security
```

`attacklm infer`
Perform a smoke-test of the trained model against representative security prompts.
```bash
attacklm infer -- --adapter models/attacklm-single_TIMESTAMP
```

### Specialized Tools
- `attacklm eval`: Run the retention evaluation suite and score candidate models.
- `attacklm gui`: Launch the Terminal GUI for an interactive training experience.
- `attacklm demo`: Run the multi-agent orchestrator demo.

---

## Dataset & Attribution

The dataset is meticulously partitioned into "buckets" to allow granular control over training composition.

### Core Composition
| Category | Source Examples | Approx. Pairs | License |
| :--- | :--- | :--- | :--- |
| **Offensive** | Metasploit, Atomic Red Team | 15,000+ | BSD-3 / MIT |
| **Defensive** | Sigma, Elastic, Splunk | 7,000+ | DRL-1.1 / Apache-2.0 |
| **AI Security** | Garak, Promptfoo | 100+ | Mixed |
| **Meta/IR** | NIST IR, Orchestrator | 500+ | Public Domain |

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

| Command | Description |
| :--- | :--- |
| `attacklm train` | Train a model (QLoRA, GaLore, Q-GaLore, Spectrum, PiSSA) |
| `attacklm train --dataset all` | Train all buckets combined |
| `attacklm train --hpo` | Run Hyper-Parameter Optimization sweep |
| `attacklm init` | Initialize dataset: download pre-built or clone $\rightarrow$ extract $\rightarrow$ attribute |
| `attacklm balance` | Build a balanced subset of buckets to prevent overfitting |
| `attacklm build` | Merge adapter $\rightarrow$ GGUF conversion $\rightarrow$ LM Studio/Ollama register |
| `attacklm infer` | Smoke-test inference on trained adapters |
| `attacklm eval` | Run retention evaluation and regression gates |
| `attacklm gui` | Launch Terminal GUI (TUI) for all operations |
| `attacklm demo` | Run multi-agent orchestrator demo |

---

## License & Contributing

**Code License**: This project is licensed under the [MIT License](LICENSE).

**Data License**: Training data consists of mixed licenses per source. Please refer to [ATTRIBUTION.md](ATTRIBUTION.md) for the full legal mapping.

**Contributing**: We welcome contributions to the extraction pipeline and training methods. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**History**: For a full list of changes and version milestones, see [CHANGELOG.md](CHANGELOG.md).
