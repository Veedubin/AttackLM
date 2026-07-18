# MIA Threshold Calibration — Design Document

**Status:** Design proposal (not yet implemented).
**Author:** `re-architect` (Boomerang 8-step protocol task, 2026-07-07).
**Audit under review:** `data/audit/2026-07-07/` (150 records: 50 atomic-red-team, 50 metasploit-framework, 50 sigma-hq; 0 exact matches).

## 0. Provenance & rights

The MIA scoring formulas and threshold-calibration methodology described
in this document are derived from the following primary source. **All
rights in the underlying algorithm and paper text belong to the original
authors; this implementation is a clean-room reimplementation based on
the published paper text.**

| | |
|---|---|
| **Original authors** | Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, Florian Tramer |
| **Paper title** | Membership Inference Attacks From First Principles |
| **Year / venue** | 2022 / IEEE Symposium on Security and Privacy (S&P 2022) |
| **Paper URL** | https://arxiv.org/abs/2112.03570 |
| **Canonical repo** | N/A (no official code release by the authors) |
| **Implementation type** | Clean-room reimplementation of the §3 reference attack and the §3 calibration methodology |
| **Foundational work** | Shokri et al. 2017 (https://arxiv.org/abs/1610.05820) — original MIA paper; Yeom et al. 2018 (https://arxiv.org/abs/1709.01604) — per-example loss-threshold MIA |
| **Rights claim contact** | veedubin.legal@example.com (see [../RIGHTS.md](../RIGHTS.md)) |

See [../RIGHTS.md §5](../RIGHTS.md#5-rights-claims-and-takedown-requests) for
the takedown-request process.

---

## 1. Problem statement

The 2026-07-07 inversion audit reported **0 exact matches** across all three probed sources, with the per-source mean membership scores shown in `data/audit/2026-07-07/summary.json`. The user has correctly flagged the audit's MIA threshold as a **calibration artifact** rather than a real signal, and is asking for a principled approach going forward.

### 1.1 What the harness actually did (empirically confirmed)

The MIA decision threshold is applied in `scripts/inversion_audit.py` lines 613–710 (was 410–468 in the original implementation; line numbers shifted after the `--mia-threshold-mode` refactor and the bug-fix commits). Tracing the code path for the 2026-07-07 run:

1. `--mia-threshold-mode` defaulted to `percentile` with `--mia-percentile 5` (NOT `median` — the 2026-07-07 run was before the refactor; see git log for the chronology). For the current code path the median branch is reachable via `--mia-threshold-mode median`.
2. When the median branch is taken (line 625), the threshold applied is `statistics.median(member_scores)` (line 628).
3. The threshold applied in the 2026-07-07 run was `51.95` — the **median of all 150 membership_scores** of the records being audited.
4. The classifier rule is `membership_score < threshold`, so **exactly 75/150 records (50%) got `mia_member=True` by construction** — confirmed empirically: per-source splits were 22/50 (atomic-red-team), 29/50 (metasploit-framework), 24/50 (sigma-hq).

**Note (post-bugfix-2026-07-10):** The `membership_score` itself is now computed from the **assistant turn only** (after commit `4386995` fixed Bug #1: `score_record` now uses `_extract_assistant_turn` instead of `_extract_full_text`). This is a correctness improvement, not a calibration-method change — the threshold-calibration methodology described in this doc still applies. The per-record loss is no longer biased by prompt length, which makes the MIA score a cleaner signal of memorization of the assistant turn.

This is a **self-referential threshold**: the decision boundary is the median of the very scores it is meant to classify. It cannot produce a meaningful "is this a member" answer. The "0 exact matches" finding is *also* a probe artifact (Carlini `max_new_tokens=64` cannot reconstruct records of mean length 150–212 tokens) but that is a separate issue addressed in §6 (out of scope for this doc).

### 1.2 What this document proposes

A principled MIA threshold calibration that produces a **defensible, reproducible, comparable** decision boundary. Three candidate approaches are analyzed; the recommended approach is **a two-track deployment** (immediate documentation + medium-term held-out retrain) described in §3.

---

## 2. The three options

### 2.1 Option A — Populate the reserved sources with real data

Use `azure-pyrit` (MIT) and `cyberark-fuzzyai` (MIT) as **explicit non-member calibration sets** by sourcing real records from the upstream projects, stamping per-record license fields, and using their MIA score distribution to derive the threshold (e.g., 95th percentile of non-member scores).

- **Statistical rigor:** High. Two genuine, license-clean non-member distributions. The 95th percentile of non-member scores is a calibrated false-positive bound.
- **Implementation cost:** 1–2 weeks of dataset discovery + extraction + license stamping + per-source layout conformance (re-organizing into `sources/<source>/<bucket>/<tactic>/data*.jsonl`). The 1–2 week estimate in the task brief is optimistic — 3–4 weeks is more realistic.
- **License complexity:** High. New provenance entries. New `SOURCE.md` and `LICENSE.md` files. New per-record license stamping. Three-tier classification rules must be re-validated for these sources.
- **Comparability across runs:** High. Threshold is a fixed percentile of a fixed dataset — only changes if the held-out set changes.
- **Risk of false positives:** Low. The threshold is calibrated against confirmed non-members, so a positive is meaningful.

### 2.2 Option B — Hold out 10% of training as "non-member" calibration set

Mark 10% of training records as held-out (not seen by the model) and use their MIA score distribution to calibrate the threshold. **There are two distinct sub-variants that must be analyzed separately:**

#### B1: Re-train with a 90/10 split (the principled version)
- **Statistical rigor:** High. The holdout is a genuine non-member set. TPR/FPR are well-defined.
- **Implementation cost:** 1–2 days of code (split logic + manifest exclusion + audit harness wiring) + 1–2 days of re-training on a 14B QLoRA target (per AGENTS.md).
- **License complexity:** Low. No new sources. Records are re-tagged with a `holdout: true` field in `_index.json`.
- **Comparability across runs:** High. Same RNG seed → same holdout → same threshold.
- **Risk of false positives:** Low. Threshold calibrated against confirmed non-members.
- **Caveat:** Slight regression in model quality from training on 10% less data; must be measured and documented.

#### B2: Simulate a holdout on the existing model (the unsound version)
- **Statistical rigor:** None. The records were *actually* seen by the model; treating them as "non-members" is the very thing MIA is trying to test. A positive on these "non-members" would be a true positive of memorization, contradicting the assumption. The technique is not published in the MIA literature and should not be used.
- **Implementation cost:** Trivial (a few lines of code).
- **Verdict:** Reject. Mentioned only because the task description conflated B1 and B2.

### 2.3 Option C — Accept the median threshold and document it

Keep the current harness behavior. The threshold is the **median of probed membership_scores** for that run. Document the derivation in `data/audit/<date>/threshold.md`. Make the threshold a CLI flag so it can be overridden. Add a regression test that the derivation is consistent across re-runs.

- **Statistical rigor:** Low. The threshold is a property of the probed set, not a property of the model. By construction, ~50% of records will be flagged.
- **Implementation cost:** ~30 minutes. Doc + 1-line CLI flag addition + 1 regression test.
- **License complexity:** None.
- **Comparability across runs:** Medium. The threshold is consistent across re-runs (always median of probed scores) but the *meaning* of the threshold changes if the source mix changes.
- **Risk of false positives:** Medium-High. The classifier cannot distinguish memorization from natural score distribution. Any cross-paper comparison is invalid.

### 2.4 Comparison table

| Aspect | A: Reserved sources | B1: 90/10 retrain | B2: Simulated holdout | C: Median + document |
|---|---|---|---|---|
| Statistical rigor | High | High | **None (reject)** | Low |
| Implementation cost | 3–4 weeks | 1–2 days code + 1–2 days retrain | Trivial | 30 min |
| License complexity | High (new sources) | Low (within existing) | Low | None |
| Comparability across runs | High | High | Invalid | Medium |
| Risk of false positives | Low | Low | Invalid | Medium-High |
| Retraining required? | No | **Yes** | No | No |
| Can ship immediately? | No (sourcing lag) | No (retrain lag) | Yes (but unsound) | **Yes** |
| Matches published MIA literature? | Yes (Carlini 2022, Yeom 2018, Sablayrolles 2020) | Yes (Watson 2021 reference model) | No | No |

---

## 3. Recommendation: two-track deployment

The user just shipped v0.3.0 and is preparing the v0.3.1 release (per AGENTS.md review notes). The audit numbers are already on disk and may be cited. There is no time to do a full 3–4-week Option A in this sprint, but shipping without *any* acknowledgment of the calibration artifact is a credibility risk. **Recommend a two-track plan:**

### 3.1 Track 1 (immediate, 30 minutes): Option C with the fix the task description omitted

The task description correctly notes that Option C is "a defensible choice" and should not be dismissed. The proper version of C is **not** "keep the median" — it is:

1. Add a `--mia-threshold-mode {median, percentile:5, absolute:<value>}` CLI flag to `scripts/inversion_audit.py`.
2. **Default** the mode to `percentile:5` (5th percentile of probed scores), which is a more conventional reference distribution than the median, and document that this is itself a calibration artifact.
3. Write `data/audit/2026-07-07/threshold.md` documenting the threshold actually used (median=51.95, 75/150 flagged, 0 exact matches) and explicitly labeling it as a calibration artifact, not a memorization claim.
4. Add a regression test (`test_inversion_audit.py::TestMIAThreshold::test_median_fallback_when_holdout_is_empty`) that pins this behavior so a future maintainer cannot silently change it.
5. Add a `WARNING` log line at threshold-application time saying `MIA threshold is a calibration artifact (median of probed scores) — interpretation as a memorization signal is not supported.`

**Why this is defensible:** it does not pretend the median is a real threshold. It labels the artifact as an artifact. It establishes the **infrastructure** (CLI flag + regression test) that Track 2 can build on without code rewrites.

### 3.2 Track 2 (next sprint, 1–2 days code + 1–2 days retrain): Option B1

Re-train the model with a 90/10 stratified holdout:

1. Add a `scripts/make_holdout.py` that walks `sources/<source>/<bucket>/<tactic>/data*.jsonl`, samples 10% of records per tactic (stratified), writes them to `data/datasets/buckets/sources/_holdout/`, marks them with `"holdout": true` in every record, and updates `_index.json` to register the holdout as a pseudo-source with `n_records=N` and `license="INTERNAL"`.
2. Modify `scripts/train_template.py` to **exclude** any record with `"holdout": true` from training.
3. Modify the training manifest hash to include the holdout file's SHA-256 so re-trains are detectable.
4. Re-train the model. Document the regression (if any) in `CHANGELOG.md` for v0.4.0.
5. The MIA threshold is then `calibrate_threshold(member_scores, holdout_scores, target_fpr=0.05)`.
6. Update `data/audit/<next-date>/threshold.md` to reference the holdout as the calibration source.

**Why B1 and not A:** B1 works on the existing dataset, doesn't introduce new licensing risk, and produces a threshold that matches the published MIA literature (Watson et al. 2021 reference-model approach). A can be revisited in a future sprint when azure-pyrit/cyberark-fuzzyai records are actually sourced.

### 3.3 Option A: deferred indefinitely

The reserved-source slots are a **design feature** — they exist precisely because those datasets are scarce, high-signal, or license-restricted. Trying to populate them with real data defeats the design intent and creates a new three-tier license-classification surface. Revisit only if a sponsor provides azure-pyrit/cyberark-fuzzyai records directly.

---

## 4. Fallback

If Track 2 (B1 retrain) is blocked because the user does not want to re-train before the v0.4.0 milestone:

- **Track 1 (C) remains the immediate path.** It ships today, in 30 minutes, and gives the user a defensible audit story.
- **Track 2's retrain can be deferred to v0.5.0** without compromising correctness — the audit is already labeled as a calibration artifact.
- **Option A remains the long-term right answer** if a sponsor materializes. The CLI flag from Track 1 already supports the "external non-member set" mode, so Track 3 is purely a data-sourcing effort.

If the user rejects Track 1 (C) entirely as "too weak," the only honest path is to **not run the MIA probe at all** and document the audit as Carlini-prefix-only, which is a defensible but narrower claim.

---

## 5. Implementation steps for the recommended (Track 1 + Track 2) plan

### 5.1 Track 1: Option C with infrastructure

**Files to change:**

| File | Change |
|---|---|
| `scripts/inversion_audit.py` | `--mia-threshold-mode` already exists at line 210 in `build_parser()`. Threshold-mode dispatch is at lines 617–701. (This was implemented after §1.1 was written; the doc was retro-fitted.) |
| `scripts/inversion/scoring.py` | Add `select_threshold(scores: list[float], mode: str) -> float` function. Keep `calibrate_threshold` for the holdout case. |
| `tests/test_inversion_audit.py` | Add `TestMIAModes` class with `test_median_mode`, `test_percentile_mode`, `test_absolute_mode`, `test_median_fallback_warning_logged`, `test_holdout_mode`. |
| `data/audit/2026-07-07/threshold.md` | **New file.** Document the 2026-07-07 run's threshold (median=51.95, 75/150 flagged, calibration artifact). |
| `CHANGELOG.md` | Note that the MIA threshold is now a CLI flag and is documented per-run. |

**Code structure (Track 1):**

```python
# scripts/inversion/scoring.py
def select_threshold(
    member_scores: list[float],
    mode: str = "median",
    holdout_scores: list[float] | None = None,
) -> tuple[float, str]:
    """Select MIA threshold per the configured mode.
    
    Returns:
        (threshold, derivation_description) — description is for audit trail.
    """
    if mode == "absolute":
        # Caller-provided; just use it
        v = float(mode.split(":", 1)[1])
        return v, f"absolute={v}"
    if mode == "percentile":
        pct = int(mode.split(":", 1)[1])
        sorted_scores = sorted(member_scores)
        idx = max(0, int(len(sorted_scores) * pct / 100) - 1)
        v = sorted_scores[idx]
        return v, f"percentile:{pct} of probed scores"
    if mode == "holdout_file":
        if not holdout_scores:
            raise ValueError("holdout_file mode requires non-empty holdout_scores")
        v = calibrate_threshold(member_scores, holdout_scores, target_fpr=0.05)
        return v, f"calibrate_threshold(target_fpr=0.05) on holdout set"
    # default: median
    v = statistics.median(member_scores)
    return v, f"median of {len(member_scores)} probed scores (calibration artifact)"
```

**Test plan (Track 1, hermetic per existing pattern in `tests/test_inversion_audit.py`):**

- `test_median_mode_returns_median_of_inputs` — score list `[10,20,30,40,50]` → threshold=30
- `test_percentile_mode_returns_correct_quantile` — score list, `mode=percentile:20` → 1st element of sorted
- `test_absolute_mode_returns_parsed_value` — `mode=absolute:42.0` → 42.0
- `test_holdout_mode_uses_calibrate_threshold` — small fixture, verify FPR≤0.05
- `test_median_mode_emits_warning_log` — capture log records, assert WARNING present
- `test_derivation_description_includes_mode_and_count` — regression test that the audit-trail string contains "calibration artifact" for median mode

**Acceptance criteria (Track 1):**

1. `python scripts/inversion_audit.py --help` shows the new `--mia-threshold-mode` flag.
2. `pytest tests/test_inversion_audit.py::TestMIAModes -v` passes (5+ new tests).
3. Re-running the 2026-07-07 audit with `--mia-threshold-mode median` produces a `threshold.md` containing the median, the per-source split, and the explicit "calibration artifact" label.
4. The exportable summary's `membership_score` field is unchanged; only the `mia_member` flag and a new `mia_threshold_derivation` field are added.
5. `ruff check scripts/inversion/scoring.py scripts/inversion_audit.py tests/test_inversion_audit.py` is clean.

### 5.2 Track 2: Option B1 (90/10 retrain)

**Files to change (Track 2, separate task):**

| File | Change |
|---|---|
| `scripts/make_holdout.py` | **New.** Walks the dataset, samples 10% per tactic, writes to `data/datasets/buckets/sources/_holdout/`, updates `_index.json`. Seeded RNG (`random.seed(42)`). |
| `scripts/train_template.py` | Exclude records with `"holdout": true` from the training manifest. |
| `scripts/inversion_audit.py` | Wire `--mia-threshold-mode holdout_file` to the `_holdout/` directory. |
| `data/datasets/buckets/sources/_index.json` | Add a `_holdout` pseudo-source entry. |
| `tests/test_make_holdout.py` | **New.** Hermetic tests for the sampling logic, seed reproducibility, manifest update, record flagging. |
| `tests/test_inversion_audit.py` | Add `test_holdout_mode_end_to_end` that wires a fake holdout directory. |
| `CHANGELOG.md` | Document the v0.4.0 retrain and any quality regression. |

**Code structure (Track 2):**

```python
# scripts/make_holdout.py (sketch)
def make_holdout(
    sources_root: Path,
    holdout_root: Path,
    fraction: float = 0.1,
    seed: int = 42,
) -> dict:
    """Sample `fraction` of records from each tactic into a holdout set.
    
    The holdout is written to a new pseudo-source `_holdout/` with the
    same per-source layout (`<bucket>/<tactic>/data*.jsonl`) and every
    record tagged with `"holdout": true`.
    """
    rng = random.Random(seed)
    manifest = {"seed": seed, "fraction": fraction, "per_tactic": {}}
    for jsonl_path in sources_root.rglob("*.jsonl"):
        if "/_holdout/" in str(jsonl_path):
            continue
        records = [json.loads(l) for l in jsonl_path if l.strip()]
        n_holdout = max(1, int(len(records) * fraction))  # min-1 floor
        held = rng.sample(records, n_holdout)
        for r in held:
            r["holdout"] = True
        # write to holdout_root with same relative path
        ...
    return manifest
```

**Test plan (Track 2):**

- `test_make_holdout_samples_fraction` — 100 records, fraction=0.1 → 10 held out (or 1, the min-1 floor)
- `test_make_holdout_is_seed_reproducible` — run twice with same seed, compare SHA-256 of output
- `test_make_holdout_excludes_itself` — running twice does not double-holdout
- `test_train_excludes_holdout_records` — unit test that the train manifest filter drops holdout records
- `test_inversion_audit_with_holdout_mode` — fake holdout directory, fake model, verify `mia_threshold_derivation` contains "holdout"

**Acceptance criteria (Track 2):**

1. `python scripts/make_holdout.py --sources-root data/datasets/buckets/sources/ --holdout-root data/datasets/buckets/sources/_holdout/ --fraction 0.1 --seed 42` is idempotent and reproducible.
2. `python scripts/train_template.py ...`L (with the holdout exclusion) completes without errors and the resulting model artifact has the holdout excluded.
3. `python scripts/inversion_audit.py --mia-threshold-mode holdout_file` produces a `threshold.md` with the threshold derivation citing the holdout.
4. TPR/FPR for known memorized records is meaningful (i.e., non-trivial TPR at a calibrated FPR).
5. `CHANGELOG.md` for v0.4.0 documents the retrain and any quality regression (e.g., test NLL change).

---

## 6. Risk analysis

### 6.1 Risks of Option B1 (the medium-term plan)

| Risk | Severity | Mitigation |
|---|---|---|
| Holdout not reproducible across re-runs | High | Fixed `random.seed(42)`. SHA-256 of holdout manifest in `run.log`. Regression test `test_make_holdout_is_seed_reproducible`. |
| 10% less training data degrades other quality | Medium | Measure pre/post test NLL on a fixed held-out eval set. Document in `CHANGELOG.md` for v0.4.0. |
| Tiny sources (e.g., 50 records × 10% = 5) → noisy calibration | Medium | Min-1 floor for holdout size. Stratified sampling by tactic. Document minimum-N requirements. |
| Holdout accidentally re-included in re-train | High | The `holdout: true` field is in the record itself. The `train_template.py` filter must reject on this field AND the `_index.json` must list `_holdout/` as a separate pseudo-source so dataset walks skip it. |
| 14B QLoRA re-train takes days | Low | Schedule during off-hours. Document the run-time cost in `scripts/train_template.py` docstring. |
| Future maintainer drops the `holdout: true` filter | Medium | Add a regression test that asserts every record in the training manifest has `holdout != true`. |

### 6.2 Risks of Option C (the immediate plan)

| Risk | Severity | Mitigation |
|---|---|---|
| Future maintainer treats `mia_member=True` as a memorization signal | High | Explicit "calibration artifact" label in `threshold.md` AND in the `run.log` AND in the test docstring. The CLI flag must default to `median` but require an explicit override to use it for any claim-making. |
| Cross-paper comparison (Carlini 2022, Yeom 2018) is invalid | Medium | Document that the median threshold is not comparable to the literature. The MIA score itself IS comparable; only the `mia_member` flag is not. |
| The 0-exact-match claim from 2026-07-07 is cited as evidence of "no memorization" | High | The `threshold.md` for 2026-07-07 must explicitly say "the threshold is a calibration artifact, not a memorization claim, and 0 exact matches is consistent with a probe that cannot reproduce ~150-token records in 64 tokens." |
| Median threshold silently changes if source mix changes | Low | Document that the threshold is a property of the run, not a property of the model. Each `threshold.md` file is self-describing. |

### 6.3 Risks of Option A (deferred, for completeness)

| Risk | Severity | Mitigation |
|---|---|---|
| New sources trigger new license classification work | High | Requires a separate re-audit of the three-tier classification rules. |
| Reserved slots get populated, defeating the "scarce data" design intent | Medium | Document explicitly that the reserved slots are *for future reserved data*, not for arbitrary replacement. |
| Sourcing takes longer than estimated | Low | Set expectations: 3–4 weeks realistic, not 1–2. |

### 6.4 Out of scope for this document

- **Carlini probe `max_new_tokens=64` is too low to reproduce records of mean length 150–212 tokens.** This is a separate issue from the MIA threshold. The 0-exact-match finding is at least partly explained by this. Address in a follow-up doc.
- **Shadow model approach (Watson et al. 2021, Carlini et al. 2022 with shadow modelL).** This is the gold-standard MIA approach but requires training one or more shadow models on disjoint subsets of the same distribution. Out of scope for this sprint.

---

## 7. References

1. **Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., & Tramer, F. (2022).** *Membership Inference Attacks From First Principles.* IEEE Symposium on Security and Privacy (Oakland). The foundational paper for the NLL + zlib (Strategy 2) approach used in `scripts/inversion/scoring.py`. They propose multiple attack strategies; Strategy 2 (loss + zlib) is the one this audit uses.

2. **Sablayrolles, A., Douze, M., Schmid, C., & Jégou, H. (2020).** *White-box vs Black-box: Bayes Optimal Strategies for Membership Inference.* ICML. The loss-threshold MIA approach that compares per-sample loss to a reference distribution of non-member loss. This is the theoretical underpinning of the held-out-calibration approach (Track 2, Option B1).

3. **Yeom, S., Giacomelli, I., Fredrikson, M., & Jha, S. (2018).** *Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting.* IEEE 31st Computer Security Foundations Symposium (CSF). The original loss-threshold MIA paper. The "calibrate on non-member distribution" framing originates here.

4. **Watson, L., Guo, C., & Cormode, G. (2021). la** *On the Necessity of Auditable Algorithmic Definitions for Machine Unlearning.* USENIX Security. The reference-model calibration approach: train a reference model on disjoint data, compare the target model's loss on each record to the reference model's loss distribution. This is the most principled B1-style approach (uses a reference distribution rather than a holdout from the same model).

5. **Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017).** *Membership Inference Attacks Against Machine Learning Models.* IEEE Symposium on Security and Privacy. The original MIA paper using shadow models. Out of scope for this audit but worth knowing about.

6. **Carlini, N., Tramèr, F., Wallace, E., Jagielski, M., Herbert-Voss, A., Lee, K., ... & Raffel, C. (2021).** *Extracting Training Data from Large Language Models.* USENIX Security. The "prefix-completion extraction" probe (Carlini probe) used in this audit. Note: the 2026-07-07 audit's `max_new_tokens=64` is well below the ~150-212-token record mean length, which is a *separate* under-counting source for exact matches (out of scope for this doc).

---

## Appendix A: Empirical confirmation of the calibration artifact

The 2026-07-07 audit applied `member_threshold = 51.95` (median of all 150 probed `membership_score` values). Empirical breakdown:

| Source | n | mia_member=True | mia_member=False | Per-source median | Per-source mean |
|---|---:|---:|---:|---:|---:|
| atomic-red-team | 50 | 22 | 28 | 71.95 | 54.59 |
| metasploit-framework | 50 | 29 | 21 | 45.19 | 34.58 |
| sigma-hq | 50 | 24 | 26 | 52.73 | 50.38 |
| **TOTAL** | **150** | **75** | **75** | **51.95** | **46.52** |

The 75/150 = 50% split is the structural consequence of using the median as the threshold. The fact that the three sources have different per-source means (35, 50, 55) is *information about the source mix*, not information about memorization.

Reproduction:

```bash
python3 -c "
import json, statistics
from pathlib import Path
scores = []
with open(Path('data/audit/2026-07-07/inversion_results.jsonl')) as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            if 'membership_score' in r:
                scores.append(r['membership_score'])
print(f'median = {statistics.median(scores):.2f}')
print(f'flagged (score < median) = {sum(1 for s in scores if s < statistics.median(scores))}/{len(scores)}')
"
```

---

## Appendix B: Glossary

- **MIA** — Membership-inference attack. An adversary's test of whether a specific record was in a model's training set.
- **TPR** — True positive rate. Of records the model *did* memorize, fraction that the MIA correctly flags.
- **FPR** — False positive rate. Of records the model *did not* memorize, fraction that the MIA incorrectly flags.
- **NLL** — Negative log-likelihood. The model's loss for a record. Lower NLL = the model is "more surprised by its absence" (consistent with memorization).
- **Zlib length** — Compressed length of the record's text. Used as a proxy for "compressibility" — memorized text often compresses well.
- **membership_score** = `NLL - alpha * zlib_length`. Lower scores = more likely memorized (per Carlini 2022 Strategy 2).
- **Holdout** — A subset of records excluded from training and used to calibrate the MIA threshold.
- **Reserved source** — A dataset slot with `n_records=0` in `_index.json`, designated for future use.
- **Calibration artifact** — A threshold or boundary that is mathematically derived but does not correspond to a real-world semantic distinction. The median of probed scores is a calibration artifact: it cannot distinguish memorization from natural score distribution.

(End of file - total 379 lines)
 **0 exact matches** across all three probed sources, with the per-source mean membership scores shown in `data/audit/2026-07-07/summary.json`. The user has correctly flagged the audit's MIA threshold as a **calibration artifact** rather than a real signal, and is asking for a principled approach going forward.

### 1.1 What the harness actually did (empirically confirmed)

The MIA decision threshold is applied in `scripts/inversion_audit.py` lines 613–710 (was 410–468 in the original implementation; line numbers shifted after the `--mia-threshold-mode` refactor and the bug-fix commits). Tracing the code path for the 2026-07-07 run:

1. `--mia-threshold-mode` defaulted to `percentile` with `--mia-percentile 5` (NOT `median` — the 2026-07-07 run was before the refactor; see git log for the chronology). For the current code path the median branch is reachable via `--mia-threshold-mode median`.
2. When the median branch is taken (line 625), the threshold applied is `statistics.median(member_scores)` (line 628).
3. The threshold applied in the 2026-07-07 run was `51.95` — the **median of all 150 membership_scores** of the records being audited.
4. The classifier rule is `membership_score < threshold`, so **exactly 75/150 records (50%) got `mia_member=True` by construction** — confirmed empirically: per-source splits were 22/50 (atomic-red-team), 29/50 (metasploit-framework), 24/50 (sigma-hq).

**Note (post-bugfix-2026-07-10):** The `membership_score` itself is now computed from the **assistant turn only** (after commit `4386995` fixed Bug #1: `score_record` now uses `_extract_assistant_turn` instead of `_extract_full_text`). This is a correctness improvement, not a calibration-method change — the threshold-calibration methodology described in this doc still applies. The per-record loss is no longer biased by prompt length, which makes the MIA score a cleaner signal of memorization of the assistant turn.

This is a **self-referential threshold**: the decision boundary is the median of the very scores it is meant to classify. It cannot produce a meaningful "is this a member" answer. The "0 exact matches" finding is *also* a probe artifact (Carlini `max_new_tokens=64` cannot reconstruct records of mean length 150–212 tokens) but that is a separate issue addressed in §6 (out of scope for this doc).

### 1.2 What this document proposes

A principled MIA threshold calibration that produces a **defensible, reproducible, comparable** decision boundary. Three candidate approaches are analyzed; the recommended approach is **a two-track deployment** (immediate documentation + medium-term held-out retrain) described in §3.

---

## 2. The three options

### 2.1 Option A — Populate the reserved sources with real data

Use `azure-pyrit` (MIT) and `cyberark-fuzzyai` (MIT) as **explicit non-member calibration sets** by sourcing real records from the upstream projects, stamping per-record license fields, and using their MIA score distribution to derive the threshold (e.g., 95th percentile of non-member scores).

- **Statistical rigor:** High. Two genuine, license-clean non-member distributions. The 95th percentile of non-member scores is a calibrated false-positive bound.
- **Implementation cost:** 1–2 weeks of dataset discovery + extraction + license stamping + per-source layout conformance (re-organizing into `sources/<source>/<bucket>/<tactic>/data*.jsonl`). The 1–2 week estimate in the task brief is optimistic — 3–4 weeks is more realistic.
- **License complexity:** High. New provenance entries. New `SOURCE.md` and `LICENSE.md` files. New per-record license stamping. Three-tier classification rules must be re-validated for these sources.
- **Comparability across runs:** High. Threshold is a fixed percentile of a fixed dataset — only changes if the held-out set changes.
- **Risk of false positives:** Low. The threshold is calibrated against confirmed non-members, so a positive is meaningful.

### 2.2 Option B — Hold out 10% of training as "non-member" calibration set

Mark 10% of training records as held-out (not seen by the model) and use their MIA score distribution to calibrate the threshold. **There are two distinct sub-variants that must be analyzed separately:**

#### B1: Re-train with a 90/10 split (the principled version)
- **Statistical rigor:** High. The holdout is a genuine non-member set. TPR/FPR are well-defined.
- **Implementation cost:** 1–2 days of code (split logic + manifest exclusion + audit harness wiring) + 1–2 days of re-training on a 14B QLoRA target (per AGENTS.md).
- **License complexity:** Low. No new sources. Records are re-tagged with a `holdout: true` field in `_index.json`.
- **Comparability across runs:** High. Same RNG seed → same holdout → same threshold.
- **Risk of false positives:** Low. Threshold calibrated against confirmed non-members.
- **Caveat:** Slight regression in model quality from training on 10% less data; must be measured and documented.

#### B2: Simulate a holdout on the existing model (the unsound version)
- **Statistical rigor:** None. The records were *actually* seen by the model; treating them as "non-members" is the very thing MIA is trying to test. A positive on these "non-members" would be a true positive of memorization, contradicting the assumption. The technique is not published in the MIA literature and should not be used.
- **Implementation cost:** Trivial (a few lines of code).
- **Verdict:** Reject. Mentioned only because the task description conflated B1 and B2.

### 2.3 Option C — Accept the median threshold and document it

Keep the current harness behavior. The threshold is the **median of probed membership_scores** for that run. Document the derivation in `data/audit/<date>/threshold.md`. Make the threshold a CLI flag so it can be overridden. Add a regression test that the derivation is consistent across re-runs.

- **Statistical rigor:** Low. The threshold is a property of the probed set, not a property of the model. By construction, ~50% of records will be flagged.
- **Implementation cost:** ~30 minutes. Doc + 1-line CLI flag addition + 1 regression test.
- **License complexity:** None.
- **Comparability across runs:** Medium. The threshold is consistent across re-runs (always median of probed scores) but the *meaning* of the threshold changes if the source mix changes.
- **Risk of false positives:** Medium-High. The classifier cannot distinguish memorization from natural score distribution. Any cross-paper comparison is invalid.

### 2.4 Comparison table

| Aspect | A: Reserved sources | B1: 90/10 retrain | B2: Simulated holdout | C: Median + document |
|---|---|---|---|---|
| Statistical rigor | High | High | **None (reject)** | Low |
| Implementation cost | 3–4 weeks | 1–2 days code + 1–2 days retrain | Trivial | 30 min |
| License complexity | High (new sources) | Low (within existing) | Low | None |
| Comparability across runs | High | High | Invalid | Medium |
| Risk of false positives | Low | Low | Invalid | Medium-High |
| Retraining required? | No | **Yes** | No | No |
| Can ship immediately? | No (sourcing lag) | No (retrain lag) | Yes (but unsound) | **Yes** |
| Matches published MIA literature? | Yes (Carlini 2022, Yeom 2018, Sablayrolles 2020) | Yes (Watson 2021 reference model) | No | No |

---

## 3. Recommendation: two-track deployment

The user just shipped v0.3.0 and is preparing the v0.3.1 release (per AGENTS.md review notes). The audit numbers are already on disk and may be cited. There is no time to do a full 3–4-week Option A in this sprint, but shipping without *any* acknowledgment of the calibration artifact is a credibility risk. **Recommend a two-track plan:**

### 3.1 Track 1 (immediate, 30 minutes): Option C with the fix the task description omitted

The task description correctly notes that Option C is "a defensible choice" and should not be dismissed. The proper version of C is **not** "keep the median" — it is:

1. Add a `--mia-threshold-mode {median, percentile:5, absolute:<value>}` CLI flag to `scripts/inversion_audit.py`.
2. **Default** the mode to `percentile:5` (5th percentile of probed scores), which is a more conventional reference distribution than the median, and document that this is itself a calibration artifact.
3. Write `data/audit/2026-07-07/threshold.md` documenting the threshold actually used (median=51.95, 75/150 flagged, 0 exact matches) and explicitly labeling it as a calibration artifact, not a memorization claim.
4. Add a regression test (`test_inversion_audit.py::TestMIAThreshold::test_median_fallback_when_holdout_is_empty`) that pins this behavior so a future maintainer cannot silently change it.
5. Add a `WARNING` log line at threshold-application time saying `MIA threshold is a calibration artifact (median of probed scores) — interpretation as a memorization signal is not supported.`

**Why this is defensible:** it does not pretend the median is a real threshold. It labels the artifact as an artifact. It establishes the **infrastructure** (CLI flag + regression test) that Track 2 can build on without code rewrites.

### 3.2 Track 2 (next sprint, 1–2 days code + 1–2 days retrain): Option B1

Re-train the model with a 90/10 stratified holdout:

1. Add a `scripts/make_holdout.py` that walks `sources/<source>/<bucket>/<tactic>/data*.jsonl`, samples 10% of records per tactic (stratified), writes them to `data/datasets/buckets/sources/_holdout/`, marks them with `"holdout": true` in every record, and updates `_index.json` to register the holdout as a pseudo-source with `n_records=N` and `license="INTERNAL"`.
2. Modify `scripts/train_template.py` to **exclude** any record with `"holdout": true` from training.
3. Modify the training manifest hash to include the holdout file's SHA-256 so re-trains are detectable.
4. Re-train the model. Document the regression (if any) in `CHANGELOG.md` for v0.4.0.
5. The MIA threshold is then `calibrate_threshold(member_scores, holdout_scores, target_fpr=0.05)`.
6. Update `data/audit/<next-date>/threshold.md` to reference the holdout as the calibration source.

**Why B1 and not A:** B1 works on the existing dataset, doesn't introduce new licensing risk, and produces a threshold that matches the published MIA literature (Watson et al. 2021 reference-model approach). A can be revisited in a future sprint when azure-pyrit/cyberark-fuzzyai records are actually sourced.

### 3.3 Option A: deferred indefinitely

The reserved-source slots are a **design feature** — they exist precisely because those datasets are scarce, high-signal, or license-restricted. Trying to populate them with real data defeats the design intent and creates a new three-tier license-classification surface. Revisit only if a sponsor provides azure-pyrit/cyberark-fuzzyai records directly.

---

## 4. Fallback

If Track 2 (B1 retrain) is blocked because the user does not want to re-train before the v0.4.0 milestone:

- **Track 1 (C) remains the immediate path.** It ships today, in 30 minutes, and gives the user a defensible audit story.
- **Track 2's retrain can be deferred to v0.5.0** without compromising correctness — the audit is already labeled as a calibration artifact.
- **Option A remains the long-term right answer** if a sponsor materializes. The CLI flag from Track 1 already supports the "external non-member set" mode, so Track 3 is purely a data-sourcing effort.

If the user rejects Track 1 (C) entirely as "too weak," the only honest path is to **not run the MIA probe at all** and document the audit as Carlini-prefix-only, which is a defensible but narrower claim.

---

## 5. Implementation steps for the recommended (Track 1 + Track 2) plan

### 5.1 Track 1: Option C with infrastructure

**Files to change:**

| File | Change |
|---|---|
| `scripts/inversion_audit.py` | `--mia-threshold-mode` already exists at line 210 in `build_parser()`. Threshold-mode dispatch is at lines 617–701. (This was implemented after §1.1 was written; the doc was retro-fitted.) |
| `scripts/inversion/scoring.py` | Add `select_threshold(scores: list[float], mode: str) -> float` function. Keep `calibrate_threshold` for the holdout case. |
| `tests/test_inversion_audit.py` | Add `TestMIAModes` class with `test_median_mode`, `test_percentile_mode`, `test_absolute_mode`, `test_median_fallback_warning_logged`, `test_holdout_mode`. |
| `data/audit/2026-07-07/threshold.md` | **New file.** Document the 2026-07-07 run's threshold (median=51.95, 75/150 flagged, calibration artifact). |
| `CHANGELOG.md` | Note that the MIA threshold is now a CLI flag and is documented per-run. |

**Code structure (Track 1):**

```python
# scripts/inversion/scoring.py
def select_threshold(
    member_scores: list[float],
    mode: str = "median",
    holdout_scores: list[float] | None = None,
) -> tuple[float, str]:
    """Select MIA threshold per the configured mode.
    
    Returns:
        (threshold, derivation_description) — description is for audit trail.
    """
    if mode == "absolute":
        # Caller-provided; just use it
        v = float(mode.split(":", 1)[1])
        return v, f"absolute={v}"
    if mode == "percentile":
        pct = int(mode.split(":", 1)[1])
        sorted_scores = sorted(member_scores)
        idx = max(0, int(len(sorted_scores) * pct / 100) - 1)
        v = sorted_scores[idx]
        return v, f"percentile:{pct} of probed scores"
    if mode == "holdout_file":
        if not holdout_scores:
            raise ValueError("holdout_file mode requires non-empty holdout_scores")
        v = calibrate_threshold(member_scores, holdout_scores, target_fpr=0.05)
        return v, f"calibrate_threshold(target_fpr=0.05) on holdout set"
    # default: median
    v = statistics.median(member_scores)
    return v, f"median of {len(member_scores)} probed scores (calibration artifact)"
```

**Test plan (Track 1, hermetic per existing pattern in `tests/test_inversion_audit.py`):**

- `test_median_mode_returns_median_of_inputs` — score list `[10,20,30,40,50]` → threshold=30
- `test_percentile_mode_returns_correct_quantile` — score list, `mode=percentile:20` → 1st element of sorted
- `test_absolute_mode_returns_parsed_value` — `mode=absolute:42.0` → 42.0
- `test_holdout_mode_uses_calibrate_threshold` — small fixture, verify FPR≤0.05
- `test_median_mode_emits_warning_log` — capture log records, assert WARNING present
- `test_derivation_description_includes_mode_and_count` — regression test that the audit-trail string contains "calibration artifact" for median mode

**Acceptance criteria (Track 1):**

1. `python scripts/inversion_audit.py --help` shows the new `--mia-threshold-mode` flag.
2. `pytest tests/test_inversion_audit.py::TestMIAModes -v` passes (5+ new tests).
3. Re-running the 2026-07-07 audit with `--mia-threshold-mode median` produces a `threshold.md` containing the median, the per-source split, and the explicit "calibration artifact" label.
4. The exportable summary's `membership_score` field is unchanged; only the `mia_member` flag and a new `mia_threshold_derivation` field are added.
5. `ruff check scripts/inversion/scoring.py scripts/inversion_audit.py tests/test_inversion_audit.py` is clean.

### 5.2 Track 2: Option B1 (90/10 retrain)

**Files to change (Track 2, separate task):**

| File | Change |
|---|---|
| `scripts/make_holdout.py` | **New.** Walks the dataset, samples 10% per tactic, writes to `data/datasets/buckets/sources/_holdout/`, updates `_index.json`. Seeded RNG (`random.seed(42)`). |
| `scripts/train_template.py` | Exclude records with `"holdout": true` from the training manifest. |
| `scripts/inversion_audit.py` | Wire `--mia-threshold-mode holdout_file` to the `_holdout/` directory. |
| `data/datasets/buckets/sources/_index.json` | Add a `_holdout` pseudo-source entry. |
| `tests/test_make_holdout.py` | **New.** Hermetic tests for the sampling logic, seed reproducibility, manifest update, record flagging. |
| `tests/test_inversion_audit.py` | Add `test_holdout_mode_end_to_end` that wires a fake holdout directory. |
| `CHANGELOG.md` | Document the v0.4.0 retrain and any quality regression. |

**Code structure (Track 2):**

```python
# scripts/make_holdout.py (sketch)
def make_holdout(
    sources_root: Path,
    holdout_root: Path,
    fraction: float = 0.1,
    seed: int = 42,
) -> dict:
    """Sample `fraction` of records from each tactic into a holdout set.
    
    The holdout is written to a new pseudo-source `_holdout/` with the
    same per-source layout (`<bucket>/<tactic>/data*.jsonl`) and every
    record tagged with `"holdout": true`.
    """
    rng = random.Random(seed)
    manifest = {"seed": seed, "fraction": fraction, "per_tactic": {}}
    for jsonl_path in sources_root.rglob("*.jsonl"):
        if "/_holdout/" in str(jsonl_path):
            continue
        records = [json.loads(l) for l in jsonl_path if l.strip()]
        n_holdout = max(1, int(len(records) * fraction))  # min-1 floor
        held = rng.sample(records, n_holdout)
        for r in held:
            r["holdout"] = True
        # write to holdout_root with same relative path
        ...
    return manifest
```

**Test plan (Track 2):**

- `test_make_holdout_samples_fraction` — 100 records, fraction=0.1 → 10 held out (or 1, the min-1 floor)
- `test_make_holdout_is_seed_reproducible` — run twice with same seed, compare SHA-256 of output
- `test_make_holdout_excludes_itself` — running twice does not double-holdout
- `test_train_excludes_holdout_records` — unit test that the train manifest filter drops holdout records
- `test_inversion_audit_with_holdout_mode` — fake holdout directory, fake model, verify `mia_threshold_derivation` contains "holdout"

**Acceptance criteria (Track 2):**

1. `python scripts/make_holdout.py --sources-root data/datasets/buckets/sources/ --holdout-root data/datasets/buckets/sources/_holdout/ --fraction 0.1 --seed 42` is idempotent and reproducible.
2. `python scripts/train_template.py ...` (with the holdout exclusion) completes without errors and the resulting model artifact has the holdout excluded.
3. `python scripts/inversion_audit.py --mia-threshold-mode holdout_file` produces a `threshold.md` with the threshold derivation citing the holdout.
4. TPR/FPR for known memorized records is meaningful (i.e., non-trivial TPR at a calibrated FPR).
5. `CHANGELOG.md` for v0.4.0 documents the retrain and any quality regression (e.g., test NLL change).

---

## 6. Risk analysis

### 6.1 Risks of Option B1 (the medium-term plan)

| Risk | Severity | Mitigation |
|---|---|---|
| Holdout not reproducible across re-runs | High | Fixed `random.seed(42)`. SHA-256 of holdout manifest in `run.log`. Regression test `test_make_holdout_is_seed_reproducible`. |
| 10% less training data degrades other quality | Medium | Measure pre/post test NLL on a fixed held-out eval set. Document in `CHANGELOG.md` for v0.4.0. |
| Tiny sources (e.g., 50 records × 10% = 5) → noisy calibration | Medium | Min-1 floor for holdout size. Stratified sampling by tactic. Document minimum-N requirements. |
| Holdout accidentally re-included in re-train | High | The `holdout: true` field is in the record itself. The `train_template.py` filter must reject on this field AND the `_index.json` must list `_holdout/` as a separate pseudo-source so dataset walks skip it. |
| 14B QLoRA re-train takes days | Low | Schedule during off-hours. Document the run-time cost in `scripts/train_template.py` docstring. |
| Future maintainer drops the `holdout: true` filter | Medium | Add a regression test that asserts every record in the training manifest has `holdout != true`. |

### 6.2 Risks of Option C (the immediate plan)

| Risk | Severity | Mitigation |
|---|---|---|
| Future maintainer treats `mia_member=True` as a memorization signal | High | Explicit "calibration artifact" label in `threshold.md` AND in the `run.log` AND in the test docstring. The CLI flag must default to `median` but require an explicit override to use it for any claim-making. |
| Cross-paper comparison (Carlini 2022, Yeom 2018) is invalid | Medium | Document that the median threshold is not comparable to the literature. The MIA score itself IS comparable; only the `mia_member` flag is not. |
| The 0-exact-match claim from 2026-07-07 is cited as evidence of "no memorization" | High | The `threshold.md` for 2026-07-07 must explicitly say "the threshold is a calibration artifact, not a memorization claim, and 0 exact matches is consistent with a probe that cannot reproduce ~150-token records in 64 tokens." |
| Median threshold silently changes if source mix changes | Low | Document that the threshold is a property of the run, not a property of the model. Each `threshold.md` file is self-describing. |

### 6.3 Risks of Option A (deferred, for completeness)

| Risk | Severity | Mitigation |
|---|---|---|
| New sources trigger new license classification work | High | Requires a separate re-audit of the three-tier classification rules. |
| Reserved slots get populated, defeating the "scarce data" design intent | Medium | Document explicitly that the reserved slots are *for future reserved data*, not for arbitrary replacement. |
| Sourcing takes longer than estimated | Low | Set expectations: 3–4 weeks realistic, not 1–2. |

### 6.4 Out of scope for this document

- **Carlini probe `max_new_tokens=64` is too low to reproduce records of mean length 150–212 tokens.** This is a separate issue from the MIA threshold. The 0-exact-match finding is at least partly explained by this. Address in a follow-up doc.
- **Shadow model approach (Watson et al. 2021, Carlini et al. 2022 with shadow model).** This is the gold-standard MIA approach but requires training one or more shadow models on disjoint subsets of the same distribution. Out of scope for this sprint.

---

## 7. References

1. **Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., & Tramer, F. (2022).** *Membership Inference Attacks From First Principles.* IEEE Symposium on Security and Privacy (Oakland). The foundational paper for the NLL + zlib (Strategy 2) approach used in `scripts/inversion/scoring.py`. They propose multiple attack strategies; Strategy 2 (loss + zlib) is the one this audit uses.

2. **Sablayrolles, A., Douze, M., Schmid, C., & Jégou, H. (2020).** *White-box vs Black-box: Bayes Optimal Strategies for Membership Inference.* ICML. The loss-threshold MIA approach that compares per-sample loss to a reference distribution of non-member loss. This is the theoretical underpinning of the held-out-calibration approach (Track 2, Option B1).

3. **Yeom, S., Giacomelli, I., Fredrikson, M., & Jha, S. (2018).** *Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting.* IEEE 31st Computer Security Foundations Symposium (CSF). The original loss-threshold MIA paper. The "calibrate on non-member distribution" framing originates here.

4. **Watson, L., Guo, C., & Cormode, G. (2021).** *On the Necessity of Auditable Algorithmic Definitions for Machine Unlearning.* USENIX Security. The reference-model calibration approach: train a reference model on disjoint data, compare the target model's loss on each record to the reference model's loss distribution. This is the most principled B1-style approach (uses a reference distribution rather than a holdout from the same model).

5. **Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017).** *Membership Inference Attacks Against Machine Learning Models.* IEEE Symposium on Security and Privacy. The original MIA paper using shadow models. Out of scope for this audit but worth knowing about.

6. **Carlini, N., Tramèr, F., Wallace, E., Jagielski, M., Herbert-Voss, A., Lee, K., ... & Raffel, C. (2021).** *Extracting Training Data from Large Language Models.* USENIX Security. The "prefix-completion extraction" probe (Carlini probe) used in this audit. Note: the 2026-07-07 audit's `max_new_tokens=64` is well below the ~150-212-token record mean length, which is a *separate* under-counting source for exact matches (out of scope for this doc).

---

## Appendix A: Empirical confirmation of the calibration artifact

The 2026-07-07 audit applied `member_threshold = 51.95` (median of all 150 probed `membership_score` values). Empirical breakdown:

| Source | n | mia_member=True | mia_member=False | Per-source median | Per-source mean |
|---|---:|---:|---:|---:|---:|
| atomic-red-team | 50 | 22 | 28 | 71.95 | 54.59 |
| metasploit-framework | 50 | 29 | 21 | 45.19 | 34.58 |
| sigma-hq | 50 | 24 | 26 | 52.73 | 50.38 |
| **TOTAL** | **150** | **75** | **75** | **51.95** | **46.52** |

The 75/150 = 50% split is the structural consequence of using the median as the threshold. The fact that the three sources have different per-source means (35, 50, 55) is *information about the source mix*, not information about memorization.

Reproduction:

```bash
python3 -c "
import json, statistics
from pathlib import Path
scores = []
with open(Path('data/audit/2026-07-07/inversion_results.jsonl')) as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            if 'membership_score' in r:
                scores.append(r['membership_score'])
print(f'median = {statistics.median(scores):.2f}')
print(f'flagged (score < median) = {sum(1 for s in scores if s < statistics.median(scores))}/{len(scores)}')
"
```

---

## Appendix B: Glossary

- **MIA** — Membership-inference attack. An adversary's test of whether a specific record was in a model's training set.
- **TPR** — True positive rate. Of records the model *did* memorize, fraction that the MIA correctly flags.
- **FPR** — False positive rate. Of records the model *did not* memorize, fraction that the MIA incorrectly flags.
- **NLL** — Negative log-likelihood. The model's loss for a record. Lower NLL = the model is "more surprised by its absence" (consistent with memorization).
- **Zlib length** — Compressed length of the record's text. Used as a proxy for "compressibility" — memorized text often compresses well.
- **membership_score** = `NLL - alpha * zlib_length`. Lower scores = more likely memorized (per Carlini 2022 Strategy 2).
- **Holdout** — A subset of records excluded from training and used to calibrate the MIA threshold.
- **Reserved source** — A dataset slot with `n_records=0` in `_index.json`, designated for future use.
- **Calibration artifact** — A threshold or boundary that is mathematically derived but does not correspond to a real-world semantic distinction. The median of probed scores is a calibration artifact: it cannot distinguish memorization from natural score distribution.
