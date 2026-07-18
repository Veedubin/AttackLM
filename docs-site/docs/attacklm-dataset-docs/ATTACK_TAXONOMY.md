# Attack Taxonomy — Inversion Attacks on Generative Language Models

> **Status**: v0.5.0 (updated; LiRA shipped)
> **Audience**: anyone running, extending, or auditing the AttackLM inversion-audit program
> **Last updated**: 2026-07-09

## 0. Provenance & rights

The attack classes and MIA scoring methods enumerated in this document are
derived from the following primary sources. **All rights in the underlying
algorithms and paper texts belong to the original authors; the
implementations in `scripts/inversion/` are clean-room reimplementations
based on the published paper texts.**

| Attack class / MIA method                | Paper (click for arXiv)                                                  | Year | Authors                                                                                                                                |
| ---------------------------------------- | ------------------------------------------------------------------------ | ---- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Training-data extraction (Strategy 1)    | [Carlini et al. 2021](https://arxiv.org/abs/2012.07805)                  | 2021 | N. Carlini, F. Tramer, E. Wallace, M. Jagielski, A. Herbert-Voss, K. Lee, A. Roberts, T. Brown, D. Song, U. Erlingsson, A. Oprea, C. Raffel |
| Production extraction                    | [Nasr et al. 2023](https://arxiv.org/abs/2311.17035)                     | 2023 | M. Nasr, N. Carlini, J. Hayase, M. Jagielski, A. F. Cooper, D. Ippolito, C. A. Choquette-Choo, E. Wallace, F. Tramèr, K. Lee             |
| MIA reference attack (loss + zlib)       | [Carlini et al. 2022](https://arxiv.org/abs/2112.03570) (§3)             | 2022 | N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis, F. Tramer                                                                            |
| LiRA (likelihood-ratio MIA)              | [Carlini et al. 2022](https://arxiv.org/abs/2112.03570) (§4)             | 2022 | (same as above)                                                                                                                        |
| Per-token MIA (MUSE default)             | [Shi et al. 2024 / MUSE](https://arxiv.org/abs/2407.06460)               | 2024 | W. Shi, J. Lee, Y. Huang, S. Malladi, J. Zhao, A. Holtzman, D. Liu, L. Zettlemoyer, N. A. Smith, C. Zhang                              |
| Original MIA (shadow-model paradigm)     | [Shokri et al. 2017](https://arxiv.org/abs/1610.05820)                   | 2017 | R. Shokri, M. Stronati, C. Song, V. Shmatikov                                                                                            |
| Per-example loss-threshold MIA           | [Yeom et al. 2018](https://arxiv.org/abs/1709.01604)                    | 2018 | S. Yeom, I. Giacomelli, M. Fredrikson, S. Jha                                                                                            |
| BLEU-4 (extraction scoring metric)       | [Papineni et al. 2002](https://aclanthology.org/P02-1040/)               | 2002 | K. Papineni, S. Roukos, T. Ward, W.-J. Zhu                                                                                              |

**Rights claim contact:** veedubin.legal@example.com — see [../RIGHTS.md](../RIGHTS.md).

---

This document explains what "inversion attack" means in the AttackLM context,
why we have a two-attack CLI surface (not three), and which MIA methods are
shipped vs. planned.

---

## 1. The 3-attack family (per Yang 2020, arXiv:2005.03915)

Yang et al. 2020 ("Defending Model Inversion and Membership Inference Attacks
via Prediction Purification") is the canonical taxonomy paper. It identifies
three classes of "data inference attack" on ML models:

| Attack                            | What is leaked                                | Attacker goal                                       | Foundational paper                                       |
| --------------------------------- | --------------------------------------------- | --------------------------------------------------- | -------------------------------------------------------- |
| **Model Inversion**               | **Reconstruction** of input features          | Recover the input `x` given a model `f`             | Fredrikson et al. 2015, USENIX Security                |
| **Membership Inference (MIA)**    | **Membership** (yes/no) of a specific record  | Was record `r` in the training set?                 | Shokri et al. 2017, IEEE S&P; Carlini et al. 2022, IEEE S&P |
| **Training-Data Extraction (TDE)**| **Verbatim training sequences**               | Recover individual training examples                | Carlini et al. 2021, USENIX Security                    |

> *"Neural networks are susceptible to data inference attacks such as the
> model inversion attack and the membership inference attack, where the
> attacker could infer the **reconstruction** and the **membership** of a
> data sample from the confidence scores predicted by the target classifier."*
> — Yang et al. 2020, abstract

The three attacks share a parent class ("data inference attacks on ML") and
exploit the same vulnerability surface (the model's outputs leak information
about its training data), but they answer three different questions:

- **Model Inversion** → what input would produce these outputs?
- **MIA** → was this specific record ever used to train the model?
- **TDE** → give me back a verbatim piece of training data.

---

## 2. Why "Model Inversion" = "Training-Data Extraction" for generative LLMs

The original Fredrikson 2015 model-inversion attack was designed for
**classifiers**: given a target class `c` and a trained classifier `f`,
reconstruct a representative input `x̂` such that `f(x̂) = c`. This is the
"show me what class c looks like" attack.

For a **generative LM** like AttackLM, the "class" abstraction does not
exist. The only output surface is the next-token distribution (the LM head).
The LLM analog of "reconstruct x from f" is "**generate x given a prefix of
x**" — which is precisely Carlini et al. 2021's **prefix-completion
extraction attack**.

So on generative LMs, Yang 2020's first class (Model Inversion) and third
class (Training-Data Extraction) collapse into a single attack class. There
is no useful distinction between "show me what class c looks like" and
"regurgitate training record r" when the model is an instruction-tuned LM.

This is why AttackLM's CLI has a **two-attack surface**, not three:

| CLI flag value  | Attack class                                 | Maps to |
| --------------- | -------------------------------------------- | ------- |
| `extraction`    | TDE / LLM model inversion (Carlini 2021)     | `scripts/inversion/probe.py` — Strategy 1 |
| `mia`           | Membership inference                         | `scripts/inversion/scoring.py` — Strategy 2 |
| `all` (default) | Both                                         | both modules |

> **Naming note**: We keep the CLI term `extraction` rather than `model_inversion`
> because (a) it is the more common term in the LLM literature, (b) it
> preserves backward compatibility with the v0.3.1 `--probe-carlini` flag,
> and (c) it is more descriptive of what the attack actually does on a
> generative model. The literature mapping (TDE = LLM model inversion) is
> documented in §2 above.

---

## 3. What AttackLM ships

### 3.1 Extraction (= LLM model inversion = TDE) — Carlini 2021

**Implementation**: `scripts/inversion/probe.py` (`run_carlini_probe()`)

**How it works**:
1. Take the first N tokens (default 50) of a training record's user message as a prompt prefix.
2. Generate K completions (default K=20) with temperature sampling (T=1.0, top_p=0.96).
3. Score each completion against the original assistant turn with three metrics:
   - **Exact match** (after normalization)
   - **Longest common subsequence** length
   - **BLEU-4** score
4. Return the best-of-K result.

**Adaptive probe budget** (v0.3.1): `max_new_tokens = min(256, max(64, 2 * suffix_token_count))` per Carlini 2021 and MUSE 2023 defaults. The previous hard-coded 64 was smaller than the median suffix length of our audit data (75-106 tokens), suppressing real matches.

**References**:
- Carlini et al. 2021, "Extracting Training Data from Large Language Models," USENIX Security. arXiv:2012.07805.
- MUSE 2023 default probe budget. `github.com/llm-membership/muse`

---

### 3.2 MIA — reference attack (Carlini 2022 §3.2)

**Implementation**: `scripts/inversion/scoring.py` (`score_record()`, `compute_mia_score()`)

**How it works**:
1. Compute the total NLL of the full record text under the target model: `NLL = sum(-log P(token_i | context))`.
2. Compute the zlib-compressed length of the text: `zlib_length = len(zlib.compress(text.encode()))`.
3. Combine: `membership_score = NLL - alpha * zlib_length` (default `alpha = 1.0`).
4. **Lower scores = more likely memorized** (lower NLL = model finds the text easy = likely seen; smaller zlib_length = text is more compressible = more structured/repetitive = typical of training data).

**Calibration** (v0.3.1, Track 1):
- `--mia-threshold-mode {median, percentile, holdout_file}` (default `percentile`)
- `--mia-percentile` (default 5)
- Per-run `threshold.md` artifact documents the derivation
- See `docs/MIA_THRESHOLD_CALIBRATION.md` for the design rationale

**References**:
- Shokri et al. 2017, "Membership Inference Attacks against Machine Learning Models," IEEE S&P. arXiv:1610.05820.
- Carlini et al. 2022, "Membership Inference Attacks From First Principles," IEEE S&P. arXiv:2112.03570.

---

### 3.3 MIA — per-token loss on the suffix (MUSE 2023) — **v0.4.0 (NEW)**

**Implementation**: `scripts/inversion/scoring.py` (`score_per_token()`, NEW)

**How it works**:
1. Extract the **assistant turn only** (the final message in the record), not the full record. The system+user prompt is context; the assistant turn is the candidate for memorization.
2. Compute NLL of the assistant turn under the target model: `NLL_total = sum(-log P(token_i | context))`.
3. Compute per-token NLL: `nll_per_token = NLL_total / num_suffix_tokens`.
4. Compute per-token zlib ratio: `zlib_ratio_per_token = zlib_length / num_suffix_tokens` (more length-stable than `zlib_length`).
5. Combine: `membership_score = nll_per_token - alpha * zlib_ratio_per_token`. **Lower = more likely memorized**.

**Why per-token?**
- Full-record NLL is biased by record length: a 500-token record accumulates more NLL than a 50-token record just because it has more tokens. Per-token NLL normalizes this away.
- Per-token NLL is roughly normally distributed within a source (full-record NLL varies by an order of magnitude across sources). Normal distribution = cleaner threshold calibration.
- This is the MUSE 2023 default and was already flagged in `docs/PROBE_TOKEN_BUDGET.md` §84 as "the strongest signal."

**Compute cost**: zero additional forward passes. The full-record `compute_nll()` already gives us the per-token NLL; we just need to extract the suffix length.

**Threshold calibration**: percentile mode (default 5) works, but the percentile must be re-tuned per source because per-token NLL distributions are tighter within a source and wider across sources. The existing `--mia-threshold-mode percentile` is the recommended default; future work should derive separate calibration files for the per-token score column.

---

### 3.4 MIA — LiRA / shadow-model (Carlini 2022 §4) — **v0.5.0 (SHIPPED)**

**Implementation**: `scripts/inversion/{shadow_train, lira}.py` (NEW, v0.5.0)

**How it works**:
1. Train K=16 (default) **shadow models** of the same architecture and size as the target, each on a disjoint subset of the same source distribution.
2. For each query record `r`, compute the target model's loss `loss_target(r)`.
3. For each shadow model `k`, compute the shadow's loss `loss_k(r)`. This contributes to either the IN distribution (if `r` was in shadow k's training set) or the OUT distribution (if it was not).
4. Fit two Gaussians per record: `N(μ_in, σ_in)` over the K IN losses and `N(μ_out, σ_out)` over the K OUT losses.
5. The **LiRA score** is the log-likelihood ratio:
   ```
   lira_logit = log N(loss_target(r); μ_in, σ_in) - log N(loss_target(r); μ_out, σ_out)
   ```
6. The **natural threshold** is 0.0 (positive LRT = more likely in).

**Why LiRA is the gold standard**:
- "10× more powerful at low FPR" than the reference attack (Carlini 2022 §4.4)
- The Gaussian fit is provably better than the original Shokri 2017 classifier-on-shadow-losses approach.
- The LRT statistic is naturally calibrated — no held-out set needed.

**Storage cost**: 4 floats per record per shadow configuration = 16 bytes per record. For 24,652 records this is 394 KB total. Negligible.

**Audit-time compute**: 1 forward pass on the target model per record (to get `loss_target(r)`), plus 4 floats of Gaussian parameters from the calibration file, plus 1 log-likelihood computation. **Identical to the reference attack** at audit time. The cost is the K shadow retrains, which happen once and are reused across all future audits.

**Why we collapse shadow-model-MIA and reference-model-MIA into LiRA**:
- Reference-model MIA (Sablayrolles 2020, Watson 2021) = LiRA with K=1 and only the OUT Gaussian. The LRT reduces to `(loss_r - μ_out) / σ_out`.
- Shadow-model MIA (Shokri 2017) = LiRA with K=16, no Gaussian fit, threshold chosen on the held-out distribution. The original Shokri attack used a classifier on shadow model losses; LiRA's Gaussian fit is provably better.
- **A single `--lira-k` flag covers both degenerate cases.** K=1 (reference-model) and K=4 (cheap LiRA) are the practical options on a single RTX 4080.

**References**:
- Carlini et al. 2022 §4, arXiv:2112.03570
- Watson et al. 2021, "On the Importance of Difficulty Calibration in Membership Inference Attacks"
- Sablayrolles et al. 2020, "White-box vs Black-box: Bayes Optimal Strategies for Membership Inference," ICML

---

## 4. Why embedding-layer MIA is deferred to v1.0.0

The embedding-layer MIA asks "does the model have a memorized embedding for
token sequence X?" rather than "does the model predict token Y given
context?" It attacks a different threat surface (the input embedding table
vs. the LM head) and requires a different scoring pipeline:

- Instead of `model(**inputs, labels=input_ids).loss`, we need
  `model.get_input_embeddings()(input_ids)` and a similarity metric
  (e.g., nearest-neighbor in the embedding space).
- The existing `--mia-threshold-mode` flags don't apply; a new calibration
  approach is needed.

For a security audit of an **instruction-tuned model**, the LM-head attack
surface is more relevant (it tests "does the model regurgitate training
examples when prompted?") than the embedding-table surface. So embedding-layer
MIA is deferred to v1.0.0 as a research thread.

See `CONTEXT.md` §"Future Directions" for the deferred research write-up.

---

## 5. CLI mapping

| CLI flag                | Values                                              | Effect                                                       |
| ----------------------- | --------------------------------------------------- | ------------------------------------------------------------ |
| `--attack`              | `extraction` / `mia` / `all` (default `all`)        | Which attack class(es) to run                                |
| `--mia-method`          | `reference` / `zlib` / `per_token` / `lira` / `all` (default `reference`) | Which MIA scoring algorithm to use                           |
| `--probe-carlini`       | flag (boolean, default True)                        | **DEPRECATED v0.4.0**, removed v0.6.0. Alias for `--attack extraction` |
| `--probe-mia`           | flag (boolean, default True)                        | **DEPRECATED v0.4.0**, removed v0.6.0. Alias for `--attack mia` |
| `--mia-threshold-mode`  | `median` / `percentile` / `holdout_file` / `lrt`   | How to derive the membership threshold. `lrt` is the natural 0.0 for LiRA. |
| `--mia-percentile`      | int (default 5)                                     | Percentile for `percentile` mode                             |
| `--lira-k`              | int (default 16)                                    | Number of shadow models for LiRA. K=1 = reference-model, K=4 = cheap LiRA. **v0.5.0+ only** |
| `--lira-params`         | path (required for `--mia-method lira`)             | Path to shadow_params.json from `python -m inversion.shadow_train`. **v0.5.0+ only** |

**Resolution rules** (in `inversion_audit.py main()`):
- If `--probe-carlini` or `--probe-mia` is set, emit a `DeprecationWarning` and map to `--attack`.
- `--no-probe-carlini` → `--attack mia`
- `--no-probe-mia` → `--attack extraction`
- If both are explicitly disabled, `--attack none` (skip everything — useful for `--dry-run`).

---

## 6. References

### Canonical attack papers

1. **Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017).** "Membership Inference Attacks against Machine Learning Models." IEEE Symposium on Security and Privacy. arXiv:1610.05820.
2. **Carlini, N., et al. (2021).** "Extracting Training Data from Large Language Models." USENIX Security Symposium. arXiv:2012.07805.
3. **Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., & Tramer, F. (2022).** "Membership Inference Attacks From First Principles." IEEE Symposium on Security and Privacy. arXiv:2112.03570.
4. **Fredrikson, M., Jha, S., & Ristenpart, T. (2015).** "Model Inversion Attacks that Exploit Confidence Information and Countermeasures." USENIX Security Symposium.

### MIA reference models and improvements

5. **Yeom, S., Giacomelli, I., Fredrikson, M., & Jha, S. (2018).** "Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting." IEEE Computer Security Foundations Symposium (CSF). The original loss-threshold MIA.
6. **Sablayrolles, A., Douze, M., Schmid, C., & Jégou, H. (2020).** "White-box vs Black-box: Bayes Optimal Strategies for Membership Inference." ICML.
7. **Watson, L., Guo, C., & Cormode, G. (2021).** "On the Importance of Difficulty Calibration in Membership Inference Attacks."
8. **MUSE library (2023).** `github.com/llm-membership/muse`. The maintained successor to the original MIA code. Defaults to per-token loss on the suffix.

### Taxonomy

9. **Yang, Z., Shao, B., Xuan, B., Chang, E.-C., & Zhang, F. (2020).** "Defending Model Inversion and Membership Inference Attacks via Prediction Purification." arXiv:2005.03915. The explicit three-attack taxonomy paper. **The §2 collapse argument is derived from this paper's abstract.**

### Adjacent / context

10. **Carlini, N., Mironov, I., Nasr, M., & Song, S. (2022).** "The Privacy Onion Effect: Memorization is Relative." NeurIPS.
11. **Mireshghallah, F., Goyal, K., Uniyal, A., Berg-Kirkpatrick, T., & Shokri, R. (2022).** "Quantifying Privacy Risks of Masked Language Models Using Membership Inference Attacks." EMNLP.
12. **Duan, M., Suri, A., Mireshghallah, F., Min, S., Zhong, W., Zettlemoyer, L., Tsvetkov, Y., Choi, Y., Evans, D., & Hajishirzi, H. (2024).** "Do Membership Inference Attacks Work on Large Language Models?" arXiv:2402.07841. **Relevant counter-argument: MIA is much harder on modern LLMs than on the small models in Shokri 2017's original work. Our framework is designed to support both the pessimistic and optimistic cases.**

---

## 7. See also

- `docs/MIA_THRESHOLD_CALIBRATION.md` — detailed design of the MIA threshold modes
- `docs/PROBE_TOKEN_BUDGET.md` — detailed design of the extraction probe budget
- `docs/AUDIT_RUNNER.md` — how to run an overnight audit
- `docs/LIRA.md` — LiRA design and usage (v0.5.0, shipped)
- `CONTEXT.md` §"Future Directions" — embedding-layer MIA research thread (deferred to v1.0.0)
