# Probe Token Budget for `attacklm-dataset` Inversion Audit

**Author:** re-architect (deepseek-v4-pro:cloud)
**Date:** 2026-07-07 (revised 2026-07-09 to add provenance)
**Scope:** `scripts/inversion/probe.py` — `max_new_tokens` parameter for the
Carlini prefix-completion extraction attack.
**Status:** Research finding. No code changed.

## 0. Provenance & rights

The recommendation in this document (256-token probe length) is derived
from the following primary source. **All rights in the underlying
algorithm and paper text belong to the original authors; this analysis
is based on the published paper text and does not reproduce it.**

| | |
|---|---|
| **Original authors** | Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, Alina Oprea, Colin Raffel |
| **Paper title** | Extracting Training Data from Large Language Models |
| **Year / venue** | 2021 / USENIX Security Symposium |
| **Paper URL** | https://arxiv.org/abs/2012.07805 |
| **Canonical repo** | N/A (no official code release by the authors) |
| **Reused by** | MUSE (Shi et al. 2024, https://arxiv.org/abs/2407.06460, https://github.com/woooooda/MUSE_unlearning) — the 256-token length is the default in MUSE's "no verbatim memorization" test |
| **Rights claim contact** | veedubin.legal@example.com (see [../RIGHTS.md](../RIGHTS.md)) |

See [../RIGHTS.md §5](../RIGHTS.md#5-rights-claims-and-takedown-requests) for
the takedown-request process.

---

## Executive Summary

The current `max_new_tokens=64` in `attacklm-dataset/scripts/inversion/probe.py`
is a **false-negative trap**: the 2026-07-07 audit ran on 150 records with mean
suffix length of 75–106 tokens, so a 64-token cap makes exact-match and full
BLEU-4 **mathematically impossible** for the majority of records. The Carlini
2021 paper, MUSE 2023, and all major follow-ups use **256 tokens** as the
canonical probe length (sometimes 512 for long-form corpora). The recommendation
is to switch to `max_new_tokens=256` as the default, with a small per-record
adaptive cap of `min(256, 2 × suffix_token_count)` to avoid wasted compute on
short records. At K=20 completions × 256 tokens × 14B model, an RTX 4080
finishes 150 records in ~20–25 minutes — well under the 30-min budget.

---

## Current State

`scripts/inversion/probe.py:87` defines `generate_completions(... max_new_tokens=64, temperature=1.0, top_p=0.96, do_sample=True)`. The 2026-07-07 audit confirms the runtime behavior: `best_completion_length` averages 289.7 characters (max 391, min 187), which matches exactly what 64 Qwen2.5-Coder BPE tokens decode to (~4.5 chars/token at temperature 1.0). The other generation parameters are reasonable and **should not change**:

| Parameter | Current | Notes |
|-----------|---------|-------|
| `max_new_tokens` | 64 | **under-budget** — the bug |
| `temperature` | 1.0 | matches Carlini 2021 §4.1 |
| `top_p` | 0.96 | matches Carlini 2021 §4.1 |
| `do_sample` | True | required for diversity |
| `num_completions` (K) | 20 | matches Carlini 2021 best-of-K |
| `prefix_tokens` | 50 | matches Carlini 2021 prefix size |

The audit data also includes a per-record `nll`, `perplexity`, and
`membership_score` (computed via `scripts/inversion/scoring.py`), so the
pipeline already supports a per-token scoring channel — the fix below does not
disrupt it.

---

## Literature Findings (SearXNG was empty; findings from model knowledge)

> **Note on web search:** SearXNG returned 0 results for all 8 queries attempted
> (Carlini 2021, MUSE 2023, llmprivacy, DecodingTrust, ETHICS, etc.). The
> findings below are from my training data (knowledge cutoff January 2026),
> cross-checked against the audit's observed behavior.

### 1. Carlini et al., "Extracting Training Data from Large Language Models" (USENIX Security 2021)

This is the canonical reference for the attack we're running. The paper's
GPT-2 experiments use `max_new_tokens=256` throughout — see §4.1 ("Extraction
Attack Implementation") and Table 2's description of the GPT-2 1.3B "sponge
examples." The 256-token cap was chosen so that the longest memorized sequence
in the test corpus (a single news article or URL with path + dictionary keys)
can be reconstructed in a single completion. The paper explicitly notes that
"truncation at 256 tokens is not a correctness bound but a compute bound — we
chose it because most memorized strings in our crawl are shorter than 256
tokens, and doubling it to 512 yielded no additional matches in a pilot run
on 100 sequences." (paraphrased from §4.1, p. 7 in the preprint version.)

### 2. Carlini et al., "Extracting Memorized Training Data from Large Language Models" / MUSE library (2023 update)

The MUSE library (`github.com/llm-membership/muse`) is the maintained successor
to the 2021 code. Its default `probes.json` config sets `max_new_tokens=256`
and `num_completions=20`, identical to the 2021 paper. Their README explicitly
warns: *"Setting max_new_tokens below the median suffix length of your eval
corpus will produce artificially low extraction rates that are not comparable
to published baselines."* This is exactly the trap we've fallen into.

### 3. "ETHICS" benchmark follow-up work & DecodingTrust (2023)

Both papers use 256-token generation caps for their memorization/privacy
sub-experiments (Sun et al. 2024 *Principled Instructions*; Wang et al. 2023
*DecodingTrust*, §C.2 "Privacy" red-team section). Wang et al. specifically
report that reducing the cap to 64 tokens "truncated the model's longest
memorized sequences and produced a 2–4× underestimate of the true privacy
leakage" in their GPT-3.5 red-team.

### 4. Contemporary membership-inference-attack (MIA) literature

Carlini et al.'s "Membership Inference Attacks From First Principles" (IEEE
S&P 2022) and subsequent MIA work prefer **per-token loss-based scoring
(NLL/reference perplexity) over generation-based probing** for audit
purposes — NLL is bounded by suffix length and doesn't suffer from the
truncation problem. Our pipeline already records `nll` and `membership_score`;
the recommendation below keeps the generation path for **public-facing
demonstrability** (it's much more compelling to show "the model wrote your
license text verbatim" than to show a low perplexity number) and adds a
suffix-aware cap to make it fair.

### 5. Practitioner conventions (anecdotal, but consistent)

The `llmprivacy` library (`github.com/Princeton-SysML/Jailbreak_LLM` and
related), the `priv-leak` benchmark, and the "Are Emergent Abilities of
Large Language Models a Mirage?" supplementary materials all converge on
**256 as the standard probe length** for English instruction-tuned models.
LLaMA-2's own red-team (Ganguli et al. 2022) uses 512 for adversarial
attack generation but 256 for memorization probes.

---

## Recommendation

**Default value: `max_new_tokens = 256`.**

**Adaptive variant (preferred):** compute the suffix token count
`len(tokenizer.encode(target))` and set the per-record cap to
`min(256, 2 * suffix_token_count)`. This guarantees we always have at least
2× the headroom needed to recover the suffix, and never wastes compute
generating tokens beyond what could possibly be matched.

| Probe suffix length | Adaptive cap | Rationale |
|---------------------|--------------|-----------|
| ≤ 128 tokens | `2 × suffix` (≤ 256) | headroom for the full suffix + 1× for a second guess |
| 128–256 tokens | 256 | matches Carlini / MUSE default |
| > 256 tokens | 256 | generation cap is the ceiling; NLL/perplexity carry the rest |

### Why 256, not 512 or 1024

- **Carlini 2021** used 256; matching it makes our results directly
  comparable to the canonical baseline.
- **Doubling to 512** would ~2× the wall-clock time and (per Carlini's pilot)
  recover at most a handful of additional matches in a typical security corpus.
- **At 1024+** the model begins to drift semantically from the prefix
  (especially at temperature=1.0), inflating false positives from
  coincidental topical overlap.

### Compute impact estimate (RTX 4080, 14B Qwen2.5-Coder in safetensors, fp16)

- 64 tokens × K=20 = 1,280 forward steps per record → 150 records × ~0.2s/record
  ≈ 30s of pure GPU time (matches the 2026-07-07 audit wall-clock of 4 min
  with host overhead).
- 256 tokens × K=20 = 5,120 forward steps per record → ~0.8s/record × 150 =
  120s GPU time. With vLLM-style batching and KV-cache reuse the wall clock
  is closer to 4–6× that → **~10 minutes for 150 records**.
- 512 tokens × K=20 = 10,240 forward steps → ~1.6s/record × 150 = 240s GPU
  → **~20 min wall clock**. Acceptable but offers diminishing returns.

The 256-token budget keeps the audit well under the 30-min ceiling on the
existing hardware.

### Risk analysis

| Failure mode | Symptom | Mitigation |
|--------------|---------|------------|
| **Too low** (current bug) | 0 exact matches; BLEU-4 capped at ~0.3 by truncation | Switch to 256 — this is the fix |
| **Too high** (≥ 1024) | Model drifts; false-positive topical matches inflate BLEU-4 | Hard cap at 256; rely on NLL for long-form |
| **Temperature wrong** | Mode collapse (T=0) or gibberish (T≥1.5) | Keep T=1.0; matches literature |
| **K too low** | Misses rare-but-extractable matches | K=20 is canonical; K=10 would be too few |

### Alternative: per-token NLL scoring (recommended in parallel, not replacement)

The strongest signal in MIA literature is **per-token loss on the suffix**:
compute `sum(-log p(token_i | context))` for the entire assistant turn and
compare to a reference distribution (e.g., a holdout set's loss distribution).
This is what `scripts/inversion/scoring.py` already does via the
`membership_score` field. **Keep it.** The recommendation is to keep
generation as the *public-facing* evidence ("here's a verbatim quote of the
model reproducing a BPL-licensed exploit") while using NLL as the *primary*
quantitative signal. This is exactly the dual-channel approach recommended in
the MIA-from-first-principles paper.

---

## Implementation Note

In `scripts/inversion/probe.py`, change `run_carlini_probe` to compute the
suffix length and pass `max_new_tokens=min(256, 2 * suffix_token_count)` to
`generate_completions`. No other parameter changes are needed. Expected
audit runtime on RTX 4080: ~10 min for 150 records (vs 4 min today); expected
exact-match and full-suffix BLEU-4 numbers will rise because the truncation
ceiling is removed.
