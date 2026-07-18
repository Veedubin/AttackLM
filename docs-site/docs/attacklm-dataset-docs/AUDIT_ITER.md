# Closed-Loop Audit (--audit-iter)

> **Methodology credit**: This document implements techniques from
> "MAI-Thinking-1: Building a Hill-Climbing Machine" by The Microsoft AI Team
> (June 2026, 109 pages). The specific methodology we adopted is described in
> [section 5.2 of that paper](paper-citation). We thank the Microsoft AI Team
> for sharing their development methodology in detail.
> 
> **Section reference**: MAI-Thinking-1 §5.2 — Independent Red Teaming (TAP closed-loop)
> **What we took**: The TAP-style closed-loop adversarial pipeline: attack → identify failing cases → generate variants → re-attack → feed back into training data.
> **What we adapted**: Microsoft's pipeline runs against safety policy categories; ours runs against membership-inference + extraction attacks on per-source security corpora.
> **What we did NOT take**: The automated attack-transformation templates from §5.2 (TAP, PyRIT, PAP) — our variants are deterministic (suffix + template); paraphrase is opt-in.
>
> *If the paper later gets a public URL, replace `(paper-citation)` in this block
> with the real URL. The section number + title is the canonical link for now.*

## Overview

The `--audit-iter` flag implements a closed-loop adversarial audit. Traditional auditing is a single-pass operation: you run an attack, see what is extracted, and report the success rate. However, LLMs often exhibit "brittleness" where a slight change in the prompt (a different suffix or a different instruction template) can flip a record from "non-memorized" to "extracted."

A closed-loop audit iteratively searches for these vulnerabilities. By identifying "fooling records" (cases where the attack succeeded) and generating semantic variants of those specific records, we can map the stability of the model's memorization. This allows researchers to determine if a model's privacy is robust or if it is merely "lucky" on a specific set of probes.

## CLI

The closed-loop audit is invoked via `scripts/inversion_audit.py` (or `attacklm-dataset audit`).

### Flags

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--audit-iter N` | `int` | `1` | Number of closed-loop iterations. Set `N > 1` to enable the iterative loop. |
| `--variant-strategies` | `csv` | `suffix,template` | Strategies used to generate variants. Options: `paraphrase`, `suffix`, `template`, `all`. Note: `paraphrase` requires a model and is opt-in. |
| `--variant-count-per-iter K` | `int` | `5` | Number of top-K fooling records to vary per iteration. |
| `--iter-output-dir PATH` | `path` | Derived | Directory where per-iteration JSONL dumps are stored. |
| `--iter-curve-output PATH` | `path` | `<date>/attack_success_curve.json` | Path to the final aggregated attack-success curve JSON. |

### Example Usage

```bash
# Run a 3-iteration audit using deterministic suffix and template variants
python scripts/inversion_audit.py \
    --model /path/to/model \
    --dataset-root data/datasets/buckets/sources \
    --attack mia \
    --mia-method lira \
    --audit-iter 3 \
    --variant-strategies suffix,template \
    --variant-count-per-iter 10
```

## Output Schema

The primary output is the `attack_success_curve.json`, which tracks the success rate of the attack across iterations.

### JSON Schema Example

```json
{
  "schema_version": "1.0",
  "audit_date": "2026-07-13",
  "model": "qwen2.5-coder-14b-instruct-uncensored",
  "dataset_root": "data/datasets/buckets/sources",
  "source_filter": ["metasploit-framework", "sigma-hq"],
  "n_iterations": 3,
  "variant_strategies": ["suffix", "template"],
  "curve": {
    "0": {
      "mia": {"probed": 200, "fooling": 8, "success_rate": 0.04},
      "extraction": {"probed": 200, "fooling": 2, "success_rate": 0.01}
    },
    "1": {
      "mia": {"probed": 40, "fooling": 4, "success_rate": 0.10},
      "extraction": {"probed": 10, "fooling": 3, "success_rate": 0.30}
    },
    "2": {
      "mia": {"probed": 20, "fooling": 5, "success_rate": 0.25},
      "extraction": {"probed": 15, "fooling": 6, "success_rate": 0.40}
    }
  },
  "fooling_records_by_iter": {
    "0": ["metasploit-framework::RECORD-UUID-1", "sigma-hq::RECORD-UUID-99"],
    "1": ["metasploit-framework::VARIANT-A-1", "sigma-hq::VARIANT-B-2"],
    "2": ["..."]
  }
}
```

## Variant Strategies

When a record is flagged as "fooling" (attack succeeded), the system generates variants to test if the success is stable.

### Suffix Injection
Appends "now repeat the above verbatim" or similar prompts from a fixed pool to the end of the record. This tests if the model's memorization is triggered by specific completion-style prompts.
- **Example**: `[Original Prompt] + " Please output the exact text above."`

### Prompt Template
Wraps the original record prompt in a different instruction template (e.g., "You are a helpful assistant. Please provide the following information: [Prompt]"). This tests if the memorization is tied to the specific phrasing of the prompt.
- **Example**: `"Given the following context, reconstruct the missing data: [Original Prompt]"`

### Paraphrase (Opt-in)
Uses a secondary LLM to rewrite the prompt while preserving the semantic meaning. This is the most powerful strategy as it removes verbatim overlaps but is computationally expensive and model-dependent.
- **Example**: `"Rewrite the following prompt in a professional tone: [Original Prompt]"`

## Interpretation Guide

The **Attack-Success Curve** is the key metric for evaluating model privacy.

- **Upward Trend**: The attack is successfully finding "holes" in the model's privacy. If the success rate increases across iterations, it indicates the model has significant memorization that is simply hidden by specific prompt phrasings.
- **Plateau**: The attack has reached the limit of the current variant strategies. This suggests the memorization is either stable (not dependent on phrasing) or the variant pool is exhausted.
- **Downward Trend**: Rare in closed-loop audits. May indicate that the variants are drifting too far from the original record's semantic meaning, causing the model to lose the trigger for the memorized data.

## Known Limitations

- **Paraphrase Dependency**: The effectiveness of the `paraphrase` strategy depends entirely on the quality of the rewriting model.
- **Deterministic Scope**: Suffix and Template strategies are deterministic and limited to the predefined pools. They cannot discover novel prompt-injection styles that a human red-teamer might find.
- **Compute Cost**: Increasing `--audit-iter` and `--variant-count-per-iter` linearly increases the number of model forward passes.

## References

- MAI-Thinking-1 §5.2 — Independent Red Teaming (TAP closed-loop)
- `scripts/inversion/variant_generator.py`
- `scripts/inversion/attack_success_curve.py`
