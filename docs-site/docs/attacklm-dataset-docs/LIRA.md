# LiRA: Likelihood Ratio Attack — Design and Usage

> **Status**: v0.5.0 (shipped)
> **Last updated**: 2026-07-09
> **Module**: `scripts/inversion/lira.py`, `scripts/inversion/shadow_train.py`

## 0. Provenance & rights

This document and the code it describes are derived from the following primary
source. **All rights in the underlying algorithm and paper text belong to the
original authors; this implementation is a clean-room reimplementation
based on the published paper text.**

| | |
|---|---|
| **Original authors** | Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, Florian Tramer |
| **Paper title** | Membership Inference Attacks From First Principles |
| **Year / venue** | 2022 / IEEE Symposium on Security and Privacy (S&P 2022) |
| **Paper URL** | https://arxiv.org/abs/2112.03570 |
| **Canonical repo** | N/A (the original authors did not release official code) |
| **Implementation type** | Clean-room reimplementation of the §4 algorithm |
| **Foundational work** | Shokri et al. 2017 (https://arxiv.org/abs/1610.05820) — the original shadow-model MIA paper that LiRA refines |
| **Rights claim contact** | veedubin.legal@example.com (see [../RIGHTS.md](../RIGHTS.md)) |

If you believe any content in this document or the corresponding code violates
your copyright or license terms, see [../RIGHTS.md §5](../RIGHTS.md#5-rights-claims-and-takedown-requests)
for the takedown-request process.

---

## 1. What is LiRA?

LiRA (Likelihood Ratio Attack) is the **gold-standard membership inference attack**
from Carlini et al. 2022 §4. It trains K shadow models on disjoint subsets of the
same distribution as the target, computes each shadow's loss on each audit record,
fits per-record Gaussian distributions to the IN and OUT losses, and then uses the
log-likelihood ratio (LRT) at audit time to decide membership.

**Key property**: LiRA is **"10× more powerful at low FPR"** than the reference
attack (Carlini 2022 §4.4). The LRT statistic is naturally calibrated — no
held-out set is needed. The decision threshold is simply 0.0 (positive logit =
likely IN/member).

## 2. Why LiRA?

| Method | FPR Control | AUC Improvement | Cost | Threshold |
|--------|-------------|----------------|------|-----------|
| Reference (NLL only) | Poor | Baseline | 1 forward pass | Percentile-based |
| zlib calibration | Better | +5-10% | 1 forward pass | Percentile-based |
| Per-token (MUSE 2023) | Better | +10-15% | 1 forward pass | Percentile-based |
| **LiRA** | **Best** | **+100-1000%** at low FPR | K shadow retrains | **Natural 0.0** |

The percentile-based thresholds used by reference, zlib, and per-token MIA are
calibration artifacts. LiRA's LRT threshold is **0.0 by construction**, because
the log-likelihood ratio directly measures the evidence for membership.

## 3. How it works

### Step 1: Train K shadow models

Train K=16 (default) shadow models of the **same architecture and size** as the
target, each on a **disjoint subset** of the same source distribution.

```bash
for k in $(seq 0 15); do
    attacklm train \
        --dataset data/shadows/shadow_${k}/train.jsonl \
        --output models/shadow_${k} \
        ...
done
```

### Step 2: Score each shadow model on the audit set

Compute each shadow model's loss on each audit record. This step uses
`scripts/score_shadow.py` — the canonical helper that loads a HF model,
reads a JSONL of training records, and writes one `shadow_{K}.json` file
per shadow model.

```bash
# Step 2: Score each shadow model on the audit set
# (uses scripts/score_shadow.py — the canonical helper)
for k in $(seq 0 15); do
    python scripts/score_shadow.py \
        --model models/shadow_${k} \
        --records data/audit_set.jsonl \
        --output-dir losses/ \
        --shadow-index ${k}
done
```

Each `shadow_{K}.json` contains `{record_id: total_nll, ...}` — a flat dict
matching the format expected by `load_shadow_losses()` in
`scripts/inversion/shadow_train.py`.

### Step 3: Fit per-record Gaussians

For each record r, compute:
- **IN distribution**: losses from shadows where r was in the training set → `N(μ_in, σ_in)`
- **OUT distribution**: losses from shadows where r was out of the training set → `N(μ_out, σ_out)`

```bash
python -m inversion.shadow_train \
    --loss-dir losses/ \
    --in-records data/shadows/in_set.jsonl \
    --out-records data/shadows/out_set.jsonl \
    --output shadow_params.json
```

### Step 4: Audit with LiRA

At audit time, compute the target model's loss on each record, then compute the
log-likelihood ratio:

```
lira_logit = log N(loss_target(r); μ_in, σ_in) - log N(loss_target(r); μ_out, σ_out)
```

- `lira_logit > 0` → more likely IN (member)
- `lira_logit < 0` → more likely OUT (non-member)

## 4. AttackLM v0.5.0 workflow

```bash
python scripts/inversion_audit.py \
    --model /path/to/target/model \
    --dataset-root data/datasets/buckets/sources \
    --attack mia \
    --mia-method lira \
    --lira-params shadow_params.json \
    --mia-threshold-mode lrt
```

**Required flags**:
- `--mia-method lira` — select the LiRA scoring method
- `--lira-params <path>` — path to the `shadow_params.json` file produced by step 3

**Optional flags**:
- `--lira-k 16` — number of shadow models (informational only; the audit reads the K from the shadow params file at `args.lira_params`, written there by `shadow_train.py` from `len(shadow_losses)`)
- `--mia-threshold-mode lrt` — use the natural 0.0 threshold (default for LiRA)

### 4.5. Quick baseline: `--mia-method offline`

If you don't have time to train K=16 shadow models (~16h on a 3B model), the
**offline MIA** provides a no-shadow-model baseline:

```
z = (nll - μ_out) / σ_out
```

where `μ_out` and `σ_out` are the sample mean and standard deviation of the
audit set's own NLL values. Records with `z < threshold` (default -1.5) are
flagged as potential training-set members.

**Key properties:**

- **NO shadow models required** — uses only the target model's NLL on the audit set
- Equivalent to Carlini 2022 §3.2 (reference attack) with sample-std normalization
- Requires **N >= 30 records** for the sample standard deviation to be reliable
- The threshold is percentile-based, not theoretically grounded like LiRA's 0.0

**Example invocation:**

```bash
python scripts/inversion_audit.py \
    --model /path/to/target/model \
    --dataset-root data/datasets/buckets/sources \
    --attack mia \
    --mia-method offline \
    --offline-z-threshold -1.5
```

**Caveat:** This is a **baseline, not LiRA**. For gold-standard MIA at low FPR,
train K=16 shadow models and use `--mia-method lira`. The offline method is
useful for rapid sanity checks and as a lower bound on MIA power.

## 5. Compute cost

| K (shadows) | Model size | Time per shadow | Total time | GPU memory |
|-------------|-----------|-----------------|------------|------------|
| 1 | 3B | ~1h | ~1h | RTX 4080 16GB |
| 4 | 3B | ~1h | ~4h | RTX 4080 16GB |
| 16 | 3B | ~1h | ~16h | RTX 4080 16GB |
| 1 | 14B | ~8h | ~8h | RTX 4080 16GB (with QLoRA) |
| 16 | 14B | ~8h | ~5d | RTX 4080 16GB (with QLoRA) |

**Important**: The shadow-model training is the user's responsibility. AttackLM
ships the **scoring framework**, not the training pipeline. The `shadow_train.py`
CLI scaffold reads precomputed shadow losses and produces the `shadow_params.json`.

## 6. K parameter guide

| K | Method | Description |
|---|--------|-------------|
| 1 | Reference-model MIA | Degenerate case. Only the OUT Gaussian is fit. LRT reduces to `(loss - μ_out) / σ_out`. Equivalent to Sablayrolles 2020. |
| 4 | Cheap LiRA | Reasonable power at modest cost. Good for initial exploration. |
| 16 | Gold standard LiRA | Full Carlini 2022 §4. "10× more powerful at low FPR." |

**K=1** is supported and useful for validating the scoring pipeline without
the cost of training multiple shadow models.

## 7. Storage cost

Per record: 4 floats × 4 bytes = **16 bytes**.

For 24,652 records: 24,652 × 16 = **394 KB total**. Negligible.

The `shadow_params.json` file also stores `lira_k` (number of shadows) and
`created_at` (ISO timestamp), adding ~100 bytes of overhead.

## 8. Threshold

The natural threshold for LiRA is **0.0**:

- `lira_logit >= 0` → predicted **member** (IN)
- `lira_logit < 0` → predicted **non-member** (OUT)

Use `--mia-threshold-mode lrt` to select this threshold explicitly. No
`holdout_file` is needed — the LRT is self-calibrating.

If you have labeled calibration data (known members and non-members), you can
use `calibrate_lira_threshold()` to find a threshold that achieves a target FPR.
This is for research purposes; the 0.0 threshold is the theoretically optimal
choice under the Gaussian assumption.

## 9. References

1. **Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., & Tramer, F. (2022).**
   "Membership Inference Attacks From First Principles." IEEE S&P. arXiv:2112.03570 §4.
   The canonical LiRA paper. Introduces the Gaussian fit and LRT formulation.

2. **Watson, L., Guo, C., & Cormode, G. (2021).**
   "On the Importance of Difficulty Calibration in Membership Inference Attacks."
   Shows that difficulty calibration (what LiRA achieves via shadow models)
   is critical for MIA power at low FPR.

3. **Sablayrolles, A., Douze, M., Schmid, C., & Jégou, H. (2020).**
   "White-box vs Black-box: Bayes Optimal Strategies for Membership Inference." ICML.
   LiRA with K=1 reduces to the Sablayrolles 2020 optimal white-box MIA.

4. **Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017).**
   "Membership Inference Attacks against Machine Learning Models." IEEE S&P.
   The original shadow-model MIA. LiRA's Gaussian fit is provably better than
   Shokri's classifier-on-shadow-losses approach.