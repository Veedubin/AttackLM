# Held-Out NLL Evaluation

> **Methodology credit**: This document implements techniques from
> "MAI-Thinking-1: Building a Hill-Climbing Machine" by The Microsoft AI Team
> (June 2026, 109 pages). The specific methodology we adopted is described in
> [section 2.3 + 2.3.2 of that paper](paper-citation). We thank the Microsoft AI Team
> for sharing their development methodology in detail.
> 
> **Section reference**: MAI-Thinking-1 §2.3 (Evaluation Methodology) + §2.3.2 (Comparison of Accuracy and NLL Evaluations)
> **What we took**: The weighted Eq-3 aggregate across 5 buckets (Code / STEM / Math / General / Multilingual).
> **What we adapted**: Bucket assignments translated to our 11 security-corpus sources; added `equal` and `custom` escape hatches.
> **What we did NOT take**: The full accuracy-vs-NLL ablation (we use NLL only, per §2.3.2).
>
> *If the paper later gets a public URL, replace `(paper-citation)` in this block
> with the real URL. The section number + title is the canonical link for now.*

## Overview

Held-out Negative Log-Likelihood (NLL) evaluation provides a cheap, reproducible, and contamination-resistant signal to determine if model changes (hyperparameters, data mixture, or architecture) are making the model better or worse.

Unlike standard benchmark accuracy, which is a "hard" signal (right or wrong), NLL is a "soft" signal that measures how well the model predicts the actual tokens of a held-out set.

## How it Works

### 1. Data Splitting
At extraction time, 300 records per source are randomly held out using `scripts/split_held_out.py`. These records are never seen by the model during training, ensuring the evaluation is truly held-out and resistant to training-set leakage.

### 2. Measurement
The `scripts/held_out_nll.py` script computes the mean per-token NLL for each source. For every record in the held-out set, the model calculates the cross-entropy loss of the target tokens given the prompt.

### 3. Aggregation
To get a single "global" score, we use a weighted aggregate based on a 5-bucket mapping (inspired by Eq-3 in MAI-Thinking-1). 

**Bucket Mapping & Weights:**

| Bucket | Weights | Security Source Mapping (Example) |
| :--- | :--- | :--- |
| **Code** | 0.3 | Metasploit, Atomic Red Team |
| **STEM** | 0.2 | Sigma, Elastic |
| **Math** | 0.2 | (Mapped to high-logic sources) |
| **General** | 0.2 | Splunk, Mordor |
| **Multilingual** | 0.1 | (Mapped to diverse-source sets) |

**Redistribution Rule:** If a bucket is empty (no sources mapped to it), its weight is redistributed proportionally across the remaining active buckets.

## Why NLL instead of Accuracy?

Following the reasoning in MAI-Thinking-1 §2.3.2, we use NLL because:
- **Smoother Gradient**: NLL is more sensitive to small improvements that don't yet flip a hard accuracy bit.
- **No Gold Labels**: It doesn't require a strictly defined "correct" answer (which is difficult for complex security reasoning); it only requires the ground-truth token sequence.
- **Efficiency**: It is computationally cheaper to calculate than full generation-based accuracy.

## Usage

### Model Comparison
To determine if Checkpoint A is better than Checkpoint B:
1. Run `held_out_nll.py` on both.
2. Compare the **Aggregate NLL**.
3. **Lower is better.**

### Adding New Sources
To incorporate a new source into the evaluation:
1. Edit the bucket mapping dictionary in `scripts/held_out_nll.py`.
2. Alternatively, provide a JSON mapping file via `--aggregation-formula custom`.

## CLI Reference

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--model` | None (Required) | Path to the model weights or HF repo. |
| `--dataset-root` | `data/` | Path to the dataset containing held-out splits. |
| `--aggregation-formula` | `mimic_mai` | Use `mimic_mai` for Eq-3 weights, `equal` for simple average, or `custom`. |
| `--output` | `nll_results.json` | Path to save the evaluation results. |
| `--batch-size` | 1 | Inference batch size. |

## Output Schema

The results are exported as a JSON object:

```json
{
  "aggregate_nll": 1.42,
  "formula_used": "mimic_mai",
  "per_source": {
    "metasploit": 1.31,
    "sigma": 1.55,
    "atomic-red-team": 1.28
  },
  "per_bucket": {
    "code": 1.29,
    "stem": 1.55,
    "math": 1.40,
    "general": 1.62,
    "multilingual": 1.45
  }
}
```

## Known Limitations

- **Model Dependency**: NLL values are not comparable across different model families (e.g., do not compare a Llama-3 NLL to a Qwen-2 NLL).
- **Tokenization**: Results are dependent on the tokenizer; ensure the same tokenizer is used for both the target model and the evaluation script.
