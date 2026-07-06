# AttackLM — Evaluation Strategy

## Philosophy: Narrow-Bet Validation

AttackLM follows the **narrow-bet** evaluation philosophy (inspired by
[ds4/DwarfStar](https://github.com/antirez/ds4)): deeply validate 3-4 top
candidate models rather than shallowly testing all 14 possible base models.

### Why Narrow-Bet?

1. **Resource efficiency**: Each full evaluation (Patterns 1-3) takes ~30 minutes
   on an RTX 4080 SUPER. Testing 14 models would take 7 hours. Testing 4 takes 2
   hours.

2. **Signal quality**: Deep evaluation — 100 prompts with NLL scoring, golden
   vector comparison, and domain-specific benchmarking — provides more actionable
   signal than shallow evaluation with 4 smoke-test prompts.

3. **Regression detection**: Golden vectors catch tokenizer, template, attention,
   and logits regressions that shallow testing misses entirely. A model that
   passes 4 smoke tests can still have subtle quality degradation.

4. **Per-bucket insight**: Reference scoring breaks down quality by bucket
   (MITRE tactics, Metasploit, phishing, etc.), revealing *where* a model is
   stronger or weaker — not just *whether* it's better overall.

---

## Candidate Selection Criteria

The top 3-4 candidates are selected based on:

| Criterion | Priority | Description |
|-----------|----------|-------------|
| **Model size** | Gate | Must fit in 16GB VRAM with QLoRA (4-bit NF4 + LoRA adapter). Practical limit: 3B-7B parameters. |
| **Architecture** | High | Qwen2.5 family preferred — proven with AttackLM training, LoRA target modules well-known, chat template compatible. |
| **License** | High | Permissive (Apache 2.0, MIT) preferred for distribution. Llama Community License acceptable for research. |
| **Code specialization** | Medium | Code-specialized variants (Qwen2.5-Coder, DeepSeek-Coder) may perform better on Metasploit command generation. |
| **Novelty** | Low | At least one candidate from a different architecture family (Phi, Llama) for comparison diversity. |

---

## Current Candidates (v0.10.1)

| Rank | Model | Size | VRAM (QLoRA) | Rationale |
|------|-------|------|-------------|-----------|
| 1 | `Qwen/Qwen2.5-Coder-3B-Instruct` | 3B | ~6 GB | **Current default.** Proven with AttackLM training. Apache-2.0. |
| 2 | `Qwen/Qwen2.5-Coder-7B-Instruct` | 7B | ~12 GB | Larger capacity. Tight VRAM. |
| 3 | `Qwen/Qwen3-4B` | 4B | ~8 GB | Next-gen Qwen architecture with `assistant_only_loss` support (~2x training efficiency). |
| 4 | `microsoft/Phi-4-mini-instruct` | 3.8B | ~7 GB | Alternative architecture (Microsoft Phi). MIT license. Strong reasoning benchmark scores. |

### Candidate Rotation Policy

- Candidates are re-evaluated when new model families are released (e.g., Qwen4, Llama 4)
- A candidate is **promoted** to default if it beats the current default on ≥ 3 of 4 evaluation dimensions
- A candidate is **dropped** if it fails Golden Vectors (Pattern 2) or scores < 0.60 on Domain Benchmark (Pattern 3)
- At least one "diversity candidate" (different architecture family) is always included

---

## Evaluation Pipeline

For each candidate, run these 4 patterns in order:

### Step 1: Golden Vectors (Pattern 2) — Fast Regression Gate

**Command**: `attacklm eval --golden ...`

**Duration**: < 2 minutes

**What it measures**: Token-byte exact match rate and logprob rank correlation (Spearman rho) against reference golden vectors.

**Decision**:
- **PASS** (match_rate ≥ 0.95, rho ≥ 0.80) → Proceed to Step 2
- **WARN** (match_rate ≥ 0.90, rho ≥ 0.70) → Proceed with caution, flag in report
- **FAIL** (otherwise) → **Candidate rejected.** Do not proceed.

### Step 2: Reference Scoring (Pattern 1) — Quality Comparison

**Command**: `attacklm eval --collect-ref ...` then `attacklm eval --compare ...`

**Duration**: ~15 minutes

**What it measures**: Negative log-likelihood (NLL) and longest common prefix (LCP) against reference continuations from the current best model. **v0.10.1 now integrates Judge-and-Revise quality filtering to prune noise from the evaluation pipeline.**

**Output**: Per-bucket delta report showing where candidate improves or regresses.

### Step 3: Domain Benchmark (Pattern 3) — Capability Evaluation

**Command**: `attacklm eval --score ...`

**Duration**: ~10 minutes

**What it measures**: Accuracy on 100 curated questions across 5 categories:
- MITRE technique identification (25 questions)
- Metasploit command generation (25 questions)
- Prompt injection detection (25 questions)
- Phishing email generation (17 questions)
- Orchestrator routing (8 questions)

### Step 4: Speed Benchmark (Pattern 4) — Performance Comparison

**Command**: `attacklm eval --speed ...` (or via `scripts/speed_bench.py`)

**Duration**: ~5 minutes

**What it measures**: Tokens/sec at context frontiers (512, 1024, 2048, 4096) and VRAM usage.

---

## Decision Matrix

| Candidate | Golden Vectors | Ref NLL | Domain Score | Speed (2048 ctx) | Decision |
|-----------|---------------|---------|-------------|------------------|----------|
| 3B | PASS | 0.342 | 0.78 | 34.2 t/s | **DEFAULT** — keep as primary |
| 7B | PASS | 0.298 | 0.82 | 22.1 t/s | **UPGRADE** — if quality matters more than speed |
| Qwen3-4B | WARN | 0.401 | 0.71 | 28.5 t/s | **EXPERIMENTAL** — monitor |
| Phi-4-mini | FAIL | — | — | — | **REJECTED** — tokenizer/logits incompatible |

### Decision Rules

- **DEFAULT**: Best overall balance of quality, speed, and reliability. Used for all production training.
- **UPGRADE**: Better quality but slower or larger. Offered as an alternative for users with more VRAM.
- **EXPERIMENTAL**: Shows promise but has issues (WARN on golden vectors, low domain score). Tracked for future re-evaluation.
- **REJECTED**: Fails golden vectors or scores below 0.60 on domain benchmark. Not suitable for AttackLM.

---

## When to Re-Evaluate

| Trigger | Scope | Frequency |
|---------|-------|-----------|
| **New base model release** | All 4 candidates + new model | On new Qwen/Llama/Phi family release |
| **Training change** (new dataset, hyperparams) | Current default only | Per change |
| **Before release** | Current default (all 4 patterns) | Per release (see `QA_BEFORE_RELEASES.md`) |
| **Monthly baseline** | Current default (Patterns 2+3 only) | Monthly |
| **Architecture change** (new LoRA targets, quantization) | All candidates | Per change |

---

## Quick Evaluation (For Rapid Iteration)

When iterating quickly (e.g., testing hyperparameter changes), run only:

```bash
# Fast gate: 2 minutes
attacklm eval --golden \
  --base-model <model> --adapter <adapter> \
  --golden data/golden/vectors.json \
  --output /tmp/golden_check.json

# If PASS, run domain benchmark: 10 minutes
attacklm eval --score \
  --base-model <model> --adapter <adapter> \
  --questions data/bench/questions.jsonl \
  --output /tmp/bench_check.json
```

Full evaluation (all 4 patterns, ~30 minutes) is reserved for candidate selection and release gates.

---

## Reference Data

| Data File | Purpose | How to Generate |
|-----------|---------|----------------|
| `data/reference/prompts.jsonl` | 100 prompts for reference scoring | Curated once, updated when dataset changes |
| `data/reference/continuations/` | Reference continuations from best model | `python scripts/collect_reference.py` |
| `data/golden/prompts.jsonl` | 50 prompts for golden vectors | Subset of reference prompts |
| `data/golden/vectors.json` | Golden vectors from best model | `python scripts/golden_vectors.py generate` |
| `data/bench/questions.jsonl` | 100 domain benchmark questions | Curated once, updated when new domains added |
| `data/bench/speed_context.txt` | Context text for speed benchmarking | Generated from Metasploit documentation |

**Regenerate reference continuations and golden vectors** whenever the default model changes (new training run, new base model).

---

*Last updated: 2026-07-05 — AttackLM v0.10.1*
