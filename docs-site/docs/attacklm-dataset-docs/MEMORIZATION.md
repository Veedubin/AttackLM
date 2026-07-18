# Memorization-Aware Epoch Capping

> **Methodology credit**: This document implements techniques from
> "MAI-Thinking-1: Building a Hill-Climbing Machine" by The Microsoft AI Team
> (June 2026, 109 pages). The specific methodology we adopted is described in
> [section 2.5.4 of that paper](paper-citation). We thank the Microsoft AI Team
> for sharing their development methodology in detail.
> 
> **Section reference**: MAI-Thinking-1 §2.5.4 — Mid-training Data Mixture (memorization-aware epoch capping)
> **What we took**: The NLL<0.01 fraction proxy as a heuristic for memorization/structural repetition, used to recommend per-source epoch caps.
> **What we adapted**: Microsoft's heuristic is applied during mid-training for a 35B model; we apply it to per-source data prep for downstream training.
> **What we did NOT take**: The mid-training-vs-final-training decision rule (their context has 3 training phases; ours is data prep, not training).
>
> *If the paper later gets a public URL, replace `(paper-citation)` in this block with the real URL. The section number + title is the canonical link for now.*

## Overview

In large-scale language model training, "memorization" occurs when a model ceases to learn general patterns and instead begins to verbatim store specific training sequences. This is particularly problematic for security-centric datasets like AttackLM, where high structural repetition (e.g., similar boilerplate in different Metasploit modules) can lead to overfitting, reduced generalization, and potential privacy leakage.

To combat this, we need a quantitative proxy to identify which data sources are being "over-memorized" by a model. We adopt the NLL < 0.01 fraction proxy from MAI-Thinking-1. By measuring the proportion of tokens in a source that the model predicts with "near-certainty" (extremely low Negative Log-Likelihood), we can estimate the degree of memorization or structural repetition. This allows us to recommend a conservative epoch cap for high-memorization sources, ensuring the model is exposed to them only as much as necessary without inducing overfitting.

## The NLL < 0.01 fraction proxy

The memorization proxy is calculated by evaluating the per-token Negative Log-Likelihood (NLL) of a sampled subset of records from each data source.

### The Formula

For a given source, the memorization fraction $\text{frac}_{\text{mem}}$ is defined as:

$$\text{frac}_{\text{mem}} = \frac{\text{count}(\text{tokens where } \text{NLL} < 0.01)}{\text{total tokens scored}}$$

### Interpretation

- **NLL < 0.01**: A token with an NLL below 0.01 is predicted with very high confidence (near-certainty). 
- **High Fraction**: If a large percentage of tokens in a source consistently fall below this threshold, it suggests the model has already "solved" the structural patterns of that source or has memorized the content verbatim.
- **Low Fraction**: A low fraction indicates the source contains high-entropy information that the model has not yet fully internalized, suggesting more training exposure (epochs) is beneficial.

## CLI

The memorization report is generated using `scripts/memorization_report.py`.

### Usage
```bash
python scripts/memorization_report.py \
    --model /path/to/model \
    --dataset-root data/datasets/buckets/sources/ \
    --report-output report.json
```

### Flag Specification

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model` | `Path` | **Required** | Path to model directory (HF safetensors) or `.gguf` file. |
| `--model-format` | `choice` | `None` | Model format: `hf` or `gguf`. Auto-detected if omitted. |
| `--dataset-root` | `Path` | `data/datasets/buckets/sources/` | Path to the source-organized dataset root. |
| `--source-filter` | `list` | `None` | Restrict analysis to specific source names (e.g., `metasploit-framework sigma-hq`). |
| `--sample-size-per-source` | `int` | `200` | Number of records to sample per source for the proxy calculation. |
| `--nll-threshold` | `float` | `0.01` | The "near-certainty" threshold. Tokens with NLL below this are counted as memorized. |
| `--epoch-cap-table` | `Path` | `None` | Path to a JSON file defining custom epoch-cap thresholds. |
| `--report-output` | `Path` | `stdout` | Path to write the JSON report. |
| `--num-proc` | `int` | `1` | Number of processes. Note: This task is GPU-bound; parallelism is typically not beneficial. |
| `--dry-run` | `flag` | `False` | Print execution plan without loading the model or computing NLL. |

## Output schema

The tool produces a JSON report containing a per-source analysis.

### Example Report
```json
{
  "schema_version": "1.0",
  "methodology": "MAI-Thinking-1 §2.5.4 NLL<0.01 fraction proxy",
  "nll_threshold": 0.01,
  "model_path": "/models/attacklm-7b",
  "model_format": "hf",
  "dataset_root": "/home/user/attacklm-dataset/data/datasets/buckets/sources/",
  "source_filter": ["all"],
  "sample_size_per_source": 200,
  "epoch_cap_thresholds": [[0.05, 8], [0.15, 4], [0.3, 2], [1.01, 1]],
  "sources": [
    {
      "source": "metasploit-framework",
      "n_records_sampled": 200,
      "total_tokens": 184000,
      "nll_lt_threshold_tokens": 31200,
      "memorization_fraction": 0.1696,
      "recommended_epoch_cap": 2,
      "example_records": ["msf-001", "msf-042", "msf-109", "msf-211", "msf-500"]
    }
  ]
}
```

## Epoch cap table

The script maps the calculated `memorization_fraction` to a recommended training epoch cap based on the following 4-tier table:

| Memorization Fraction | Recommended Epoch Cap | Exposure Level |
| :--- | :--- | :--- |
| $< 0.05$ | **8 epochs** | High exposure (Low memorization) |
| $0.05 \le \text{frac} < 0.15$ | **4 epochs** | Moderate exposure |
| $0.15 \le \text{frac} < 0.30$ | **2 epochs** | Low exposure (Elevated memorization) |
| $\ge 0.30$ | **1 epoch** | Minimum exposure (High memorization) |

### Overriding Thresholds
You can provide a custom table via `--epoch-cap-table` using a JSON file with the following format:
```json
[[0.05, 8], [0.15, 4], [0.30, 2], [1.01, 1]]
```

## Interpretation guide

When reviewing the `memorization_fraction` for a source:

1. **Fraction $\approx 0$**: The model finds this source's content novel and non-repetitive. It can likely handle higher epoch counts without overfitting.
2. **Fraction $0.10 - 0.30$**: The source has a significant amount of "boilerplate" or structurally repetitive content that the model has already mastered. Reducing epochs prevents the model from over-weighting these patterns.
3. **Fraction $> 0.30$**: This source is highly memorized. Training beyond 1 epoch may cause the model to collapse into verbatim reproduction of this source, damaging general reasoning abilities.

## Known limitations

- **Heuristic Nature**: The NLL < 0.01 proxy is a heuristic. While effective in the MAI-Thinking-1 scale, it may vary slightly across different model architectures.
- **Structural vs. Semantic**: The proxy captures structural repetition and verbatim memorization, but it does not explicitly detect semantic duplication across different sources.
- **Cross-source Overlap**: The tool analyzes sources independently. It does not account for cases where Source A and Source B contain the same data.
- **Model Requirement**: Requires white-box access to model logits (HuggingFace format) to compute per-token NLL. GGUF models are not supported as they typically do not expose the required log-prob tensors for this specific fraction calculation.

## References

- MAI-Thinking-1 §2.5.4 — Mid-training Data Mixture (memorization-aware epoch capping)
