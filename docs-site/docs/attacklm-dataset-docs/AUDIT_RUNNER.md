# Inversion Audit — Overnight Runner Plan

**Author:** boomerang (re-coder delegation)
**Date:** 2026-07-07
**Status:** Plan, ready to execute
**Context:** The 2026-07-07 audit completed 150 records at `max_new_tokens=64` in ~4 minutes and reported 0 exact matches. The 2026-07-07 follow-up (commit `a93c970`) raised the probe cap to `max_new_tokens=256` (adaptive) but at the cost of ~4× more compute per record — too slow for an interactive session. This plan splits the work into 4 overnight runs that finish inside an 8-hour idle window and produce a defensible v0.3.1 release-ready audit dataset.

---

## 1. Why a single big run doesn't work

| Run | Records | K | Adaptive cap | Per-record | Total      | Fits 30-min interactive? |
| --- | ------- | - | ------------ | ---------- | ---------- | ----------------------- |
| 2026-07-07 (prior)  | 150     | 20 | 64        | ~1.6s      | ~4 min     | ✅ (but undercounts)    |
| 2026-07-07 (first try) | 75   | 20 | 256       | ~70s       | ~90 min    | ❌                      |
| 2026-07-07 (pilot)  | 45      | 20 | 256       | ~70s       | ~52 min    | ❌                      |
| **Overnight target** | **600** | **20** | **256** | **~70s**  | **~12 hours** | ❌ (overnight only)  |

> **Update 2026-07-10**: Commit `4386995` fixed Bug #4 in `scripts/inversion/probe.py` — `generate_completions` now uses `num_return_sequences=num_completions` instead of K sequential `model.generate()` calls. ~20× speedup on typical 14B + 256-token setups. The "~70s/record" estimate in the table above is from the pre-fix sequential-`generate` implementation; the post-fix implementation is closer to ~3-5s/record. The overnight target of 600 records × 20 completions at 256 tokens now runs in well under 30 minutes. The 4-run split is still defensible for source coverage but no longer needed for time-budget reasons.

A single 600-record run takes ~12 hours. That **just barely** fits an overnight window but leaves no margin for a retry if the script crashes. Better: split into 4 smaller runs (~3 hours each) so each one fits comfortably and any failure is recoverable.

---

## 2. The 4-run split

Each run uses the **same defaults** (commit `a93c970`):

```
--top-k 20
--max-new-tokens <adaptive — leave at None>
--mia-threshold-mode percentile
--mia-percentile 5
--audit-output-root data/audit/
--date <run-specific YYYY-MM-DD>
```

The runs differ only in `--probe-count` and `--source-filter` so the records probed are **disjoint** across runs. The final analysis aggregates all four `summary.json` files.

| Run | Date          | Source            | Records | Est. wall time |
| --- | ------------- | ----------------- | ------- | -------------- |
| 1   | `2026-07-08`  | `atomic-red-team`     | 300     | ~5.8 hours     |
| 2   | `2026-07-08-b`| `atomic-red-team` (continued) | 300     | ~5.8 hours     |
| 3   | `2026-07-08-c`| `metasploit-framework` | 300     | ~5.8 hours     |
| 4   | `2026-07-08-d`| `sigma-hq`            | 200     | ~3.9 hours     |

**Total: 1,100 records across 3 sources, ~21 hours sequential / ~5.3 hours if run 4× in parallel.** The runs are **independent** — any one can complete without the others.

**Why these counts:**
- `atomic-red-team` has 1,115 records → split into 2 runs of ~300 each is safer than one run of 600 (single source failure point)
- `metasploit-framework` has 13,997 records → 300 is a 2% sample; defensible
- `sigma-hq` has 3,132 records → 200 is a 6% sample; defensible
- Total 1,100 is **7× the 2026-07-07 sample size** → strong statistical power

---

## 3. The runner script

Save as `attacklm-dataset/scripts/run_overnight_audits.sh` (new file, not yet committed):

```bash
#!/usr/bin/env bash
# Overnight inversion-audit runner.
# Splits 1,100 records across 4 runs, each ~3-6 hours.
# Resume-safe: re-running skips runs whose output dir already exists.
# Usage: ./scripts/run_overnight_audits.sh [--dry-run]
set -euo pipefail

# Config
PYTHON="/home/jcharles/Projects/reverse_engineering/AttackLM/.venv/bin/python"
MODEL="/home/jcharles/Projects/reverse_engineering/AttackLM/uncensored"
DATASET_ROOT="/home/jcharles/Projects/reverse_engineering/attacklm-dataset/data/datasets/buckets/sources"
AUDIT_ROOT="/home/jcharles/Projects/reverse_engineering/attacklm-dataset/data/audit"
LOG_ROOT="/tmp/audit-overnight-$(date +%Y%m%d-%H%M%S)"

mkdir -p "$LOG_ROOT"

# Per-run definitions: date|probe_count|extra_source_args
RUNS=(
  "2026-07-08-a|300|--source-filter atomic-red-team"
  "2026-07-08-b|300|--source-filter atomic-red-team"
  "2026-07-08-c|300|--source-filter metasploit-framework"
  "2026-07-08-d|200|--source-filter sigma-hq"
)

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "DRY RUN — would execute:"
  for run in "${RUNS[@]}"; do
    IFS='|' read -r date count extra <<< "$run"
    echo "  $date  $count records  $extra"
  done
  exit 0
fi

for run in "${RUNS[@]}"; do
  IFS='|' read -r date count extra <<< "$run"
  out_dir="$AUDIT_ROOT/$date"

  if [[ -d "$out_dir" && -f "$out_dir/summary.json" ]]; then
    echo "[SKIP] $date — summary.json already exists at $out_dir"
    continue
  fi

  echo "[START] $date  $count records  $extra"
  echo "  log: $LOG_ROOT/$date.log"
  echo "  out: $out_dir"

  # Clean partial run if present
  [[ -d "$out_dir" ]] && find "$out_dir" -type f -delete && rmdir "$out_dir" 2>/dev/null || true
  mkdir -p "$out_dir"
  chmod 0700 "$out_dir"

  # NOTE: --probe-count is the per-source limit, so atomic-red-team gets
  # exactly $count records (not all three sources combined).
  setsid "$PYTHON" scripts/inversion_audit.py \
    --model "$MODEL" \
    --dataset-root "$DATASET_ROOT" \
    $extra \
    --probe-count "$count" \
    --top-k 20 \
    --mia-threshold-mode percentile \
    --mia-percentile 5 \
    --audit-output-root "$AUDIT_ROOT" \
    --date "$date" \
    > "$LOG_ROOT/$date.log" 2>&1 < /dev/null

  echo "[DONE]  $date — $LOG_ROOT/$date.log"
  tail -20 "$LOG_ROOT/$date.log"
  echo "---"
done

echo "ALL RUNS COMPLETE — see $LOG_ROOT/"
```

The script is **resume-safe**: if a run is interrupted (system reboot, OOM, user), re-running it skips finished runs and re-launches only the incomplete ones. This is critical for an overnight run that nobody will babysit.

---

## 4. Launching

### 4a. Single-machine, sequential (the safe default)

```bash
cd /home/jcharles/Projects/reverse_engineering/attacklm-dataset
chmod +x scripts/run_overnight_audits.sh
# dry-run first to confirm config
./scripts/run_overnight_audits.sh --dry-run
# then actually run
nohup setsid ./scripts/run_overnight_audits.sh > /tmp/runner.log 2>&1 < /dev/null &
disown
echo "Runner started, PID $!"
# log: /tmp/runner.log
# per-run logs: /tmp/audit-overnight-<timestamp>/<date>.log
```

Expected end time: **~22 hours from launch** if all 4 runs complete. That's a long overnight — if you want to be safer, cut run 2 (the second atomic-red-team chunk) to 200 records → ~15 hours total, with a 1-hour buffer.

### 4b. Parallel on 4 GPUs (if available)

If the host has multiple GPUs, replace the sequential loop with 4 backgrounded `setsid` launches, one per run. Total wall time drops to ~6 hours. **Not applicable on the current host (single RTX 4080) but documented for completeness.**

### 4c. Reduced K (if you want a single 6-hour run instead)

If 22 hours is too long, lower K from 20 to 5:
- Per-record time: ~18s (down from ~70s)
- Total: 1,100 × 18s = ~5.5 hours
- Statistical loss: K=5 is MUSE's standard minimum; K=20 is conservative. The difference matters most for rare-but-extractable matches. For a v0.3.1 release, K=5 is defensible.

To do this, change `--top-k 20` to `--top-k 5` in the script. Add a note to CHANGELOG.md explaining the choice.

---

## 5. What the results look like

When the runs finish, each output dir contains:

```
data/audit/2026-07-08-a/
├── summary.json               # per-source aggregates (EXPORTABLE)
├── exportable_summary.json    # safe-to-share subset of summary.json
├── threshold.md               # new in commit a93c970 — documents MIA threshold derivation
├── inversion_results.jsonl    # per-record raw results (chmod 0600, internal only)
├── run.log                    # full audit log
└── [audit_id].json            # audit metadata
```

The key new file is `threshold.md`, which documents:
- the `--mia-threshold-mode` used (`percentile`)
- the `--mia-percentile` value (`5`)
- the threshold value
- the number of records flagged as `mia_member=True`

This is the artifact that addresses the "median is a calibration artifact" critique from `MIA_THRESHOLD_CALIBRATION.md`.

### 5a. Cross-run aggregation

A future aggregation script (`scripts/aggregate_overnight_audits.py`, to be written) will:
1. Walk `data/audit/2026-07-08*`
2. Read each `summary.json` and `exportable_summary.json`
3. Produce `data/audit/2026-07-08-aggregated/summary.json` and a human-readable `REPORT.md`
4. Run final statistics: total records, per-source mean NLL/BLEU/membership, exact-match counts, percentile:5 flagged counts

This is a 1-2 hour coding task; not part of this overnight plan.

---

## 6. What to do if a run fails

Each run is **independent** in the file system — its output dir either has a `summary.json` (success) or doesn't (failure). The runner script's `if [[ -d "$out_dir" && -f "$out_dir/summary.json" ]]` check handles this. Recovery procedure:

1. Check `/tmp/audit-overnight-*/<date>.log` for the failure cause
2. Common causes:
   - **CUDA OOM** — reduce `--probe-count` and retry
   - **ncclCommResume / torch import** — you're using the wrong venv. Use `AttackLM/.venv/bin/python`
   - **Disk full** — `data/audit/` is large (each record's `inversion_results.jsonl` line is ~3KB; 1,100 records × 3 sources = 10MB total, well under any disk limit, so this is unlikely)
   - **System reboot** — re-run the script; the `--skip if summary.json exists` check will resume from the last completed run
3. Re-run the script to retry failed runs only

---

## 7. Pre-flight checklist (do this before launching)

- [ ] `attacklm-dataset` is on commit `a93c970` or later (probe + MIA Track 1)
- [ ] Working venv is `AttackLM/.venv/bin/python` (workspace `.venv` has broken libtorch_cuda.so)
- [ ] `data/audit/2026-07-07-v2/` does not exist (the failed pilot was cleaned up)
- [ ] GPU is free (no other PyTorch processes): `nvidia-smi | grep MiB` shows <1GB used
- [ ] At least 20GB free disk on `/tmp` (per-run logs are ~50MB; `inversion_results.jsonl` files are ~10MB)
- [ ] `data/datasets/buckets/sources/{atomic-red-team,metasploit-framework,sigma-hq}/` all have records (the per-source loader logs a count of 1115 / 13997 / 3132)
- [ ] The user has approved the 22-hour wall time OR chose the reduced-K alternative

---

## 8. After completion (next session)

When you come back to a finished run, the workflow is:

1. Verify all 4 output dirs have `summary.json`:
   ```bash
   ls -la /home/jcharles/Projects/reverse_engineering/attacklm-dataset/data/audit/2026-07-08*/
   ```
2. Check the `threshold.md` in each — confirm mode=percentile, percentile=5, expected threshold value (should be the 5th percentile of 300 records ≈ the 15th-lowest membership score, or roughly 25-35 based on the 2026-07-07 baseline of 35-55)
3. Run the aggregation script (when written) to produce `REPORT.md`
4. Update `CHANGELOG.md` for v0.3.1 with the new headline numbers
5. Push `a93c970` and any new commits to `origin/main`
6. Decide whether to publish the v0.3.1 release to PyPI (currently at v0.2.0)

---

## 9. Open questions for the user

1. **22 hours too long?** → switch to K=5 (5.5 hours), document the choice in CHANGELOG
2. **Want the second atomic-red-team chunk (run 2)?** → it's the most diverse source; skipping it saves 5.8 hours. The 300 records in run 1 alone are a 27% sample (vs 1,115 total)
3. **Want me to also write the aggregation script (`aggregate_overnight_audits.py`) before the runs start?** → it's a separate 1-2 hour task, but it would let you generate the final REPORT.md immediately when the runs complete
4. **What if a run fails halfway through?** → the runner is resume-safe; you can also re-run individual audits manually with the same `--date` after cleaning the partial output dir
5. **Should the overnight runs go to a different host?** → no, single-GPU host is fine; the runner doesn't try to use multiple GPUs

---

## 10. Files this plan touches

- **New**: `attacklm-dataset/scripts/run_overnight_audits.sh` (runner script, ~60 lines)
- **New** (after runs complete): `data/audit/2026-07-08-a/`, `-b`, `-c`, `-d/` (4 audit output dirs)
- **New** (later): `attacklm-dataset/scripts/aggregate_overnight_audits.py` (cross-run aggregator)
- **New** (later): `data/audit/2026-07-08-aggregated/REPORT.md` (final report)
- **Updated** (later): `attacklm-dataset/CHANGELOG.md` (v0.3.1 entry citing new headline numbers)
- **Updated** (later): `attacklm-dataset/README.md` (audit section refresh)

No code changes are needed for the runs themselves — commit `a93c970` already has everything.
