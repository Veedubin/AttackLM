# Changelog

All notable changes to AttackLM are documented in this file. Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.5.0] — 2026-06-24 — Blue-team data sources, team presets, defensive extractors

### Added

- **6 new blue-team/defensive data sources** (~5,850 pairs):
  - SigmaHQ/sigma: 3,000 detection rules (DRL-1.1)
  - Elastic/detection-rules: 1,200 EQL/KQL rules (Elastic-2.0)
  - Splunk/security_content: 800 SPL detections (Apache-2.0)
  - OTRF/Security-Datasets (Mordor): 500 event log scenarios (Apache-2.0)
  - OTRF/ThreatHunter-Playbook: 150 hunting playbooks (Apache-2.0)
  - NIST SP 800-61r3: 200 IR procedure pairs (Public Domain, template-based)
- **6 new extractor scripts**: `extract_sigma_defensive.py`, `extract_elastic_rules.py`,
  `extract_splunk_content.py`, `extract_mordor.py`, `extract_threathunter_playbook.py`,
  `extract_nist_ir.py`
- **3 team presets** (`presets/red-team.json`, `purple-team.json`, `blue-team.json`)
  with pre-configured bucket weights for offensive/defensive mix control
- **`--preset` and `--system-prompt` flags** on `attacklm-balance`
- **3 new defensive buckets**: `defensive/detection_engineering` (5,000),
  `defensive/threat_hunting` (650), `defensive/incident_response` (200)
- **12 attribution files** (SOURCE.md + LICENSE.md per new source)
- **`notes/BLUE_TEAM_DESIGN_v0.5.0.md`** — full architectural plan (792 lines)

### Changed

- `scripts/init_pipeline.py` — 5 new local probes, 5 new remote repos, 6 new extractors
- `src/attacklm/cli.py` — 6 new extractors in extraction sequence
- `data/datasets/buckets/manifest.json` — 21,865 total pairs, 23 buckets, 17 sources
- `scripts/balance_buckets.py` — preset loading, weight resolution, `--preset` flag
- `src/attacklm/__version__.py` — bumped to `0.5.0`

### Fixed

- `balance_buckets.py` `BASE_DIR` undefined in preset helpers
- `_caps_for_target_total` fallback for buckets not in any category share
- Manifest `total_pairs` math consistency (source_totals sum = tier_totals sum = total_pairs)

---

## [0.4.1] — 2026-06-22 — Evaluation framework, steering vectors, dataset cleanup

### Added

- **7-pattern ds4 evaluation framework** (Patterns 1-7):
  - Pattern 1: Reference-continuation NLL scoring (`collect_reference.py`, `score_candidates.py`, `compare_scores.py`)
  - Pattern 2: Golden-vector regression gates (`golden_vectors.py`, 558 lines)
  - Pattern 3: 100-question domain benchmark (`domain_bench.py`, 563 lines)
  - Pattern 4: Speed profiling / context-frontier (`speed_bench.py`, 463 lines)
  - Pattern 5: QA checklist (documentation)
  - Pattern 6: Steering vectors (`steering.py`, 1,120 lines) — extract, apply, sweep, diagnose
  - Pattern 7: Narrow-bet philosophy doc
- **Shared model loader** `_eval_loader.py` (149 lines) — extracted from `eval_retention.py`
- **4 new CLI entry points**: `attacklm-collect-ref`, `attacklm-score`, `attacklm-compare`, `attacklm-golden`
- **198 hermetic pytest tests** across 8 test files (all pass, 0.62s)
- **`notes/EVAL_DESIGN_v0.4.0.md`** (1,098 lines), **`notes/STEERING_REVIEW_v0.4.0.md`** (793 lines)
- **`EVALUATION.md`**, **`QA_BEFORE_RELEASES.md`** documentation

### Changed

- `scripts/eval_retention.py` refactored to import from `_eval_loader.py`
- `scripts/domain_bench.py` — integrated steering vector flags
- `pyproject.toml` — 4 new eval entry points, `eval` optional-dependency group

### Removed

- **8,649 template-generated synthetic records** deleted — dataset reduced from 25,601 → 16,015 pairs
  (only 380 orchestrator records kept from synthetic sources)

---

## [0.4.0] — 2026-06-22 — MoE-safe training, retention eval, and experience replay

### Added

- **Experience-replay / mixed-corpus mixer** (`scripts/replay_mixer.py`).
  - Stratified mixing of one or more replay sources into any fine-tuning batch.
  - New CLI flags on `attacklm-train-all`:
    `--replay-source`, `--replay-ratio`, `--replay-max-examples`,
    `--replay-stratify` / `--no-replay-stratify`, `--replay-domain-ratios`.
  - Caches combined datasets under `data/datasets/combined/replay_<hash>.jsonl`.
  - Records replay source composition in `state.json`.
  - New `replay-general` source skeleton with starter samples for code,
    conversation, factual, and reasoning domains.
  - New acquisition script: `scripts/acquire_replay_general.py`.
- **`attacklm-eval` retention suite** (`scripts/eval_retention.py`).
  - Measures target-task gain vs. pretraining-domain retention.
  - CLI entry point and 30 hermetic tests.
- **Advanced training options** in `scripts/train_template.py`:
  - `--use-dora`, `--loftq-init`, `--bf16`, `--fp16`, `--fp32`.
  - `--use-rslora` / `--no-use-rslora`.
  - `--target-modules` for explicit LoRA target selection.
  - `--moe-safe-target` for Mixture-of-Experts models: bf16 only, no 4-bit
    quantization, router/lm_head excluded from LoRA targets.
- **Auto bf16 default** on Ampere/Ada/Hopper/Blackwell GPUs (compute capability >= 8.0),
  with `--fp16` override for backward compatibility.

### Changed

- `scripts/train_all.py` now wires all new LoRA/DoRA/LoftQ/bf16/MoE-safe flags
  through `build_train_cmd()` and integrates the replay mixer.
- `scripts/train_template.py` records replay provenance in `state.json`.
- `src/attacklm/cli.py` adds `attacklm-train-lora` and `attacklm-eval` entry points.
- `pyproject.toml` registers `attacklm-eval` and `attacklm-train-lora` console scripts.

### Fixed

- `tests/test_thinking_models.py` updated to match the deprecated/removed
  thinking-model helpers; now tests only the surviving `strip_thinking()` logic.
- `tests/test_balance_buckets.py` updated its data-dependent total assertion
  to match the current manifest.

---

## [0.3.3] — 2026-06-11

### Fixed

- `pyproject.toml` dynamic version resolution.
- Bad v0.3.2 publish artifact.

## [0.3.2] — 2026-06-11

### Added

- PyPI trusted publishing workflow (`.github/workflows/release.yml`).

## [0.3.1] — 2026-06-11 — `attacklm-init` one-shot setup

### Added

- **`attacklm-init`** — single-command replacement for the four-step
  manual init sequence (`attacklm-clone` → `attacklm-extract` →
  `attacklm-attribute` → `attacklm-buckets`).
  - **Probes local `data/` first.** Walks the six upstream source
    directories (atomic-red-team, stockpile, sigma,
    metasploit-framework, infection_monkey, RTA) and skips the clone
    step entirely if every source is present and has working-tree
    content above a per-source size threshold. A bare `.git` directory
    alone is **not** enough — the working tree must be checked out.
  - **Prompts for network access** when sources are missing. The user
    can decline (exit 2) or accept with `--yes` for non-interactive use.
  - **Idempotent re-runs.** Each stage is skipped if its output already
    exists; pass `--force-clone` / `--force-extract` to override.
  - **`--dry-run`** prints the plan and exits without modifying
    anything — useful for CI and for users to see what would happen.
  - Honors `--skip-clone`, `--skip-attribute`, `--skip-buckets` for
    partial runs.
- **`scripts/init_pipeline.py`** (≈370 lines) — the orchestrator module
  imported by `attacklm.cli:main_init`. Returns documented exit codes
  for downstream tooling: `0` success, `2` user-declined, `3` network
  failure, `4` `--skip-clone` with missing data.
- **`tests/test_init_pipeline.py`** — 18 hermetic tests covering local
  probe (all-present / missing / too-small), stage runners (skip-when-
  missing), bucket-built detection, and full CLI dispatch via
  `monkeypatch`. All pass.
- **CLI help text** updated to list `attacklm-init` alongside the
  existing entry points.

### Changed

- `src/attacklm/cli.py` — new `main_init()` dispatcher (sibling of
  `main_clone`, `main_extract`, etc.) that wraps the new orchestrator.
- `pyproject.toml` — new `attacklm-init = "attacklm.cli:main_init"`
  entry point, version bumped 0.3.0 → 0.3.1.
- `src/attacklm/__version__.py` — bumped to `0.3.1`.
- The four individual commands (`attacklm-clone`, `attacklm-extract`,
  `attacklm-attribute`, `attacklm-buckets`) are **unchanged** and
  remain available for users who want fine-grained control.

### Notes for PyPI users

After `pip install attacklm`, a single `attacklm-init` brings a fresh
install from zero to a fully populated `data/datasets/buckets/`
layout. If the user already cloned the repo (e.g. `git clone
https://github.com/Veedubin/AttackLM.git`) the orchestrator detects the
local data and skips the network fallback. The `--yes` flag enables
unattended first-run.

---

## [0.3.0] — 2026-06-11 — Dataset license audit & restructure

### ⚠️ BREAKING — Dataset license cleanup

Per a 2026-06-11 review of upstream source licenses (see
`data/ATTRIBUTION.md` and `data/LEGAL.md`), **three sources have been
removed from the public dataset** due to legal risk:

| Source | License | Records | Reason |
|---|---|---:|---|
| `endgameinc/RTA`                  | AGPL-3.0      | 76  | Viral copyleft. Distributing a derivative dataset would force the entire AttackLM dataset under AGPL-3.0. |
| `guardicore/infection_monkey`     | GPL-3.0       | 36  | Viral copyleft. Plugin manifests are derivative works of upstream code. |
| `TheBigPromptLibrary`             | mixed/unclear | 6   | Copyright laundering. The repo hosts leaked and reverse-engineered proprietary system prompts. |

The data is preserved locally at `archive/restricted-sources/` (gitignored)
for the author's private research, training, and experimentation. It is
**not** redistributed as part of AttackLM. See
`archive/restricted-sources/README.md` for the full rationale.

### Added

- **`data/LEGAL.md`** — research-only scope statement, full source license
  table, rights-holder contact pointer.
- **`data/REMOVAL.md`** — explicit removal-request process for rights
  holders. Acknowledgement within 48 hours, removal within 7 days of
  verification, git-history scrub at next release.
- **Per-source data layout** — `data/datasets/buckets/sources/<source>/<bucket>/<tactic>/data*.jsonl`
  replaces the previous flat `data/datasets/buckets/<bucket>/data.jsonl`
  layout. Each source directory contains `LICENSE.md` (license, license
  URI, per-bucket record counts) and `SOURCE.md` (narrative description,
  use case, risk note).
- **Provenance stamp on every record** — `source`, `source_uri`, `license`,
  `license_uri`, `rights_contact` fields added by
  `scripts/stamp_and_reorg.py`. These fields are written by the ETL
  pipeline and **must not be stripped** by downstream re-distributors.
- **License-specific attribution on every record** — for sources with
  attribution requirements beyond the base provenance, additional
  per-license fields are added by `scripts/add_attribution.py`:
  - **Metasploit Framework (BSD-3-Clause)** — 13,997 records. Each
    record carries `upstream_copyright`, `upstream_license_uri`,
    `attribution_required: true`, `bsd_3_clause_notice` (the full
    BSD-3-Clause notice text), `derived_from`, and (where extractable)
    `upstream_module_path` and `upstream_cve`. BSD-3 §1 requires the
    copyright notice and license text to be preserved in derivative
    works, so these fields are NOT optional.
  - **SigmaHQ (DRL 1.1)** — 0 records in the current dataset, but the
    field schema is ready for future use: `attribution_required: true`,
    `drl_11_attribution`, `sigma_rule_id`, `sigma_rule_author`,
    `sigma_rule_date`, `sigma_rule_title`.
- **`scripts/stamp_and_reorg.py`** — idempotent script that classifies
  every kept record by source, stamps provenance fields, and writes the
  per-source layout.
- **`scripts/add_attribution.py`** — idempotent script that stamps
  license-specific attribution fields. Currently handles Metasploit
  (BSD-3-Clause) and Sigma (DRL 1.1); trivially extensible for other
  attribution-bearing licenses.
- **`scripts/generate_source_layout.py`** — generates the
  `sources/<source>/LICENSE.md` and `SOURCE.md` files plus
  `sources/_index.json`.
- **`scripts/scrub_bpl_callback.py`** — blob callback for git-filter-repo
  that scrubs BigPromptLibrary records from any historical
  `ai/jailbreaking/data.jsonl` blob.

### Changed

- **`data/datasets/buckets/manifest.json`** — version bumped to **v5**.
  Reads from the per-source layout. Records per-source totals and
  license info at the manifest level. Lists excluded sources explicitly.
- **`scripts/rebuild_manifest.py`** — rewritten to walk the per-source
  layout. Also scans the legacy flat layout for back-compat reporting.
- **`scripts/bucket_loader.py`** — `build_combined()` now aggregates from
  the per-source layout (multiple sources may contribute to a single
  bucket). `_CATEGORY_RESOLVERS` and `_ALIAS_RESOLVERS` extended to
  include per-domain attack categories (web_app, cloud, ics, wireless,
  etc.) so `--dataset all` returns all 20 buckets.
- **`scripts/balance_buckets.py`** — `_load_bucket()` now reads from
  the per-source layout.
- **`scripts/train_all.py`** — `tactic_combined_path` and
  `dataset_path` now use `build_combined()` (per-source layout) instead
  of the old `BUCKETS_DIR / <bucket> / data.jsonl` paths.
- **`scripts/validate_per_category.py`** — `load_all_buckets()` now
  aggregates from the per-source layout.
- **`scripts/audit_dataset.py`** — reads from the per-source layout
  (`sources/.../data*.jsonl`) instead of the flat layout. Category
  classification extended to include per-domain attack categories.
- **`scripts/augment_attribution.py`** — now a no-op (prints a notice
  and exits 0) when the per-source layout is in use. The provenance
  fields are added by `scripts/stamp_and_reorg.py` + `scripts/add_attribution.py`
  instead.
- **`data/ATTRIBUTION.md`** — updated to reflect the new per-source
  layout. License table is sourced from the canonical LICENSE.md files
  in each `sources/<source>/` directory.
- **`archive/`** directory now contains `old-flat-layout/` (25,820
  records) in addition to `restricted-sources/` and `tui-source/`.
  All archive contents remain gitignored.

### Removed

- `data/datasets/buckets/tools/rta/` (76 records) — moved to
  `archive/restricted-sources/rta/bucket/`.
- `data/datasets/buckets/tools/infection_monkey/` (36 records) — moved
  to `archive/restricted-sources/infection_monkey/bucket/`.
- 6 BigPromptLibrary records from
  `data/datasets/buckets/ai/jailbreaking/data.jsonl` — moved to
  `archive/restricted-sources/bigpromptlibrary/bucket/`. The remaining 50
  garak (Apache-2.0) records stay in the public dataset.
- The entire **old flat layout** at `data/datasets/buckets/<bucket>/`
  (25,820 records) — moved to `archive/old-flat-layout/`. The flat
  layout is no longer the source of truth; everything reads from
  `data/datasets/buckets/sources/<source>/...`.

### Security / privacy

- **Git history scrub**: ran `git-filter-repo` to remove the 3
  high-risk paths and the 6 BPL records from ALL historical commits
  (not just HEAD). The `.git` directory shrunk from 20 MB to 4.8 MB
  after pruning. Old blobs are unrecoverable. **All commit hashes
  changed** (HEAD is now `119cd61`, was `1ea35c2`); a force-push to
  `origin/main` is required.

### Records totals

- **Before**: 23,981 records across 23 buckets, 64% "unknown" source
  attribution per audit.
- **After**: **25,601 records across 20 buckets, 11 sources, 100%
  per-record attribution**. 11 sources fully credited:
  atomic-red-team (MIT), mitre-stockpile (Apache-2.0),
  mitre-atlas-arsenal (Apache-2.0), metasploit-framework (BSD-3-Clause,
  attribution required), nvidia-garak (Apache-2.0), promptfoo (MIT),
  promptmap (MIT), llm-generated (GPL-3.0), attacklm-synthetic (MIT).
  Two reserved slots for future: azure-pyrit (MIT), cyberark-fuzzyai
  (Apache-2.0). Metasploit-Framework records additionally carry the
  full BSD-3-Clause notice and copyright text per BSD §1.

---

## [0.2.3] — 2026-06-11

### Added

- **`scripts/generate_synthetic_scarce.py` — live metrics + cleaner console output.**
  - `call_llm()` now returns `{content, usage, latency_ms}` — tracks prompt / completion / total tokens from both OpenAI-compatible (LMStudio) and Ollama responses.
  - Per-batch backend/model/temp spam eliminated. Backend info prints **once per category** instead of once per batch.
  - Live progress bar with real metrics: **tokens/sec**, **pairs/sec**, and **latency (ms)** per batch.
  - Optional `rich` library progress bar; plain-text ASCII fallback if `rich` is not installed.
  - Final summary line: `Wrote N pairs | X tok/s avg | Y pair/s avg | Z.s total | filename`.
  - Metrics persisted to `{category}_llm_meta.json` under `"metrics"`.

- **`scripts/llm_generate_wrapper.py` — complete rewrite.**
  - Named count overrides: `--web-app`, `--cloud`, `--social-engineering`, `--supply-chain`, `--ics-scada`, `--wireless`.
  - Single-category mode: `--only web_app`.
  - Backend / model / temperature as CLI flags (`--backend`, `--model`, `--temperature`) — **no env var syntax needed**.
  - `--sleep` flag (default OFF) for inter-batch pauses.
  - Wrapper is now the **sole entry point**; `generate_synthetic_scarce.py` accepts `--category`/`--count` only. Passing positional counts directly to it produces `unrecognized arguments` (by design).
  - Env vars passed explicitly via `subprocess.run(env=...)` instead of relying on shell inheritance.

- **`scripts/train_template.py` — tokens/sec progress replaces useless `it/s`.**
  - New `LiveProgressCallback` prints: step count, loss, **tok/s**, **pair/s**, and VRAM usage every 10 steps.
  - HF Trainer's default tqdm disabled via `disable_tqdm=True`.
  - `it/s` was meaningless because it conflates batch size, gradient accumulation, and packing into a single opaque number. `tok/s` and `pair/s` reflect actual data throughput.

### Fixed

- **Fish shell / bash multi-line env var syntax no longer required.** All backend configuration is via CLI flags in the wrapper. Eliminates `fish: Unknown command: ' '` errors when breaking lines inside `BACKEND=lmstudio \` assignments.

---

## [0.2.2] — 2026-06-10

### Fixed

- **Epoch counter now accurate.** `state.json[progress].current_epoch` is now read from the trainer's own `log_history` (the real fractional value at the last logged step) instead of HF Trainer's rounded `train_result.metrics["epoch"]` (which is an int). Previously the counter reported `epoch: 3.0` when the trainer actually ran `2.999` epochs. Also added `state.json[progress].target_epochs` (the user's --epochs value) and `state.json[progress].filtered_examples` (how many examples the long-example filter dropped) so the user can see exactly what happened.
- **`attacklm-train` no longer clobbers previous runs.** Default behavior: a `_YYYY-MM-DD_HH-MM` timestamp is appended to `--output` so each run is preserved (matching what `attacklm-train-all` has done since v0.1.6). Opt out with `--no-timestamp`. If `--no-timestamp` is set and the output is a **completed** run, the trainer refuses unless `--force` is also passed. If the path already ends in a timestamp, it's left alone (so re-runs of `train_all.py` keep working).
- **`attacklm-gguf` no longer silently skips when source is newer.** Previously, `attacklm-gguf` printed `⏭ attacklm-single — already exists` and skipped even when the source BF16 model was just re-merged. v0.2.2 compares mtime: if `models/gguf/{name}.Q4_K_M.gguf` is older than the source `model.safetensors`, it's treated as stale and re-converted (with a clear log line). `--force` bypasses the mtime check entirely.
- **`attacklm-gguf --name` now exists.** Previously `--name` was referenced in the help text and in a v0.2.0-era TODO but never implemented — passing it produced `unrecognized arguments`. v0.2.2 adds the flag; it overrides the auto-derived name from `--input`.

### Added

- **`attacklm-build`** — One-shot full pipeline: merge LoRA → BF16 → GGUF → install to LM Studio → (optional) register with Ollama → drop a build manifest at `models/built/{name}_{timestamp}/`. Replaces the 3-command shell pipeline (`attacklm-merge && rm && attacklm-gguf --install-lmstudio`) with a single command. Auto-detects the base model from the adapter's `state.json` / `adapter_config.json`. Defaults: `--install-lmstudio` ON, `--register-ollama` OFF. Wired in as a console script.

- **`attacklm-gguf --quant` and `--register-ollama`.** `--quant` lets you pick `Q8_0` / `Q5_K_M` / `Q6_K` instead of the hardcoded `Q4_K_M`. `--register-ollama` writes a `Modelfile` next to the GGUF and runs `ollama create {name}`, so the model shows up in `ollama list` and you can run it via `ollama run {name}`.

- **`scripts/balance_buckets.py`** — Balanced bucket sampler for SFT data. Auto-sizes per-bucket caps based on target model + VRAM profile (3b-16gb / 7b-16gb / 7b-128gb / 14b-128gb / 31b-128gb / full / custom). Three within-bucket sampling strategies (head / random / **stratified** — the default). Stratified sampler groups examples by their first MITRE technique ID, source, or first assistant-content line, then allocates with **minimum-1-per-group** so every technique / module gets representation. Solves the "metasploit 49% of training" problem for round-2 SFT.

  Category-balanced allocation (the default for `--target-total`): targets 50% base / 25% tools / 15% ai / 10% orchestrator, then redistributes proportionally when small categories hit their caps. Overridable via `--category-shares` JSON.

  Wired in as `attacklm-balance` console script.

- **`tests/test_balance_buckets.py`** — 19 unit tests covering stratification key selection, sampling strategies, cap resolution, integration with the real bucket manifest, and CLI output. All pass.

- **`.gitignore`** — `data/datasets/balanced/` excluded (regenerable output of the new sampler).

- **`README.md`** — new "Balanced sampling" section in the Training area; `attacklm-balance` added to the 11-script table (now 12).

### Usage

```bash
# Dry-run: see the per-bucket caps + total without writing
attacklm-balance --profile 7b-128gb --dry-run

# Write a balanced dataset to data/datasets/balanced/
attacklm-balance --profile 7b-128gb \
    --output data/datasets/balanced/balanced_7b-128gb.jsonl

# Pass to training
attacklm-train --dataset data/datasets/balanced/balanced_7b-128gb.jsonl \
               --output models/attacklm-7b-128gb \
               --base-model huihui-ai/Qwen2.5-Coder-7B-Instruct-abliterated

# Custom target total with custom category shares
attacklm-balance --profile custom --target-total 12000 \
    --category-shares '{"tactic": 0.3, "tools": 0.4, "ai_redteam": 0.2, "meta": 0.1}'
```

### Profiles

| Profile    | Per-bucket cap | Total pairs | Train time (3B) | Train time (7B) | Train time (14B) |
|------------|---------------:|------------:|----------------:|----------------:|-----------------:|
| 3b-16gb    |            800 |     ~7,500  |          2-3 hr |          3-4 hr |                 |
| 7b-16gb    |            800 |     ~7,500  |          2-3 hr |          3-4 hr |                 |
| 7b-128gb   |          1,500 |     ~9,800  |          1-2 hr |          4-6 hr |          5-7 hr  |
| 14b-128gb  |          1,500 |     ~9,800  |          1-2 hr |          4-6 hr |          5-7 hr  |
| 31b-128gb  |          2,000 |    ~10,600  |                |                |          6-8 hr  |
| full       |   unlimited    |    16,982   |                |                |       12-16 hr  |

---

## [0.2.1] — 2026-06-10

**Bucket layout reorganized into 4 parents.** The v0.2.0 layout was asymmetric (10 flat tactic dirs + 1 flat orchestrator + 2 nested parents) which made the on-disk filesystem not match the user-facing spec syntax. v0.2.1 normalizes to 4 parent directories that all work the same way.

### Changed

- **Bucket layout reorganized** to 4 parents, all of them real directories on disk:

  ```
  data/datasets/buckets/
    base/                      <- NEW: 10 tactic buckets move here
      collection/                (634 pairs)
      command_and_control/       (105 pairs)
      credential_access/         (589 pairs)
      defense_evasion/         (1,375 pairs)
      discovery/               (1,846 pairs)
      execution/                 (767 pairs)
      exfiltration/              (173 pairs)
      lateral_movement/          (252 pairs)
      persistence/             (1,120 pairs)
      privilege_escalation/      (537 pairs)
    tools/                     <- unchanged
      metasploit/              (8,349 pairs)
      infection_monkey/           (36 pairs)
      rta/                        (76 pairs)
    ai/                        <- RENAMED from ai-models/
      prompt-injection/         (687 pairs)
      jailbreaking/              (56 pairs)
    orchestrator/              <- unchanged
      data.jsonl                 (380 pairs)
    manifest.json              <- paths updated
    ATTRIBUTION.md             <- unchanged
  ```

  The `--dataset` spec syntax is unchanged: `--dataset base/`, `--dataset tools/`, `--dataset ai/`, `--dataset orchestrator`, `--dataset all`. The 12 bucket paths in `manifest.json` are updated to reflect the new on-disk locations. The `category` field (used to filter by `get_tactic_buckets()`) is preserved.

- **`--dataset ai-models/` is still accepted** as a backward-compat alias for `--dataset ai/`. The internal `_CATEGORY_RESOLVERS` map has both keys pointing to `get_ai_model_buckets()`. Old v0.2.0 user scripts keep working.

### Added

- **`scripts/migrate_buckets_to_v021.py`** (one-shot migration):
  - `--dry-run`: shows what would happen without making changes
  - Default: backs up the current layout to `data/.bucket_layout_backup/buckets_<ts>/`, moves 12 buckets, updates `manifest.json` paths atomically
  - `--rollback <snapshot_name>`: restore from a backup
  - `--list-backups`: show available rollback snapshots
  - Empty parent dirs (e.g. `ai-models/` after its children move into `ai/`) are removed as part of the move pass
  - The `data/.bucket_layout_backup/` directory is gitignored

### Migration from v0.2.0

If you already pulled v0.2.0, run the migration script once:

```bash
# Dry-run first to see what will happen
python scripts/migrate_buckets_to_v021.py --dry-run

# Then do it for real
python scripts/migrate_buckets_to_v021.py
```

The script backs up the current layout to `data/.bucket_layout_backup/buckets_<ts>/` before moving anything. To undo:

```bash
python scripts/migrate_buckets_to_v021.py --list-backups
python scripts/migrate_buckets_to_v021.py --rollback buckets_20260610_053407
```

### Verified

- 12 buckets moved, 0 errors
- `manifest.json` paths updated atomically (write to `.tmp`, rename)
- All 12 spec resolver cases pass: 4 parents (`base/`, `tools/`, `ai/`, `orchestrator`), 3 aliases (`all`, `tactics`, `tools-all`), 3 subpaths (`tools/metasploit/`, `ai/jailbreaking/`, `base/collection/`), 1 backward-compat (`ai-models/` → 2 buckets), 1 default set (tactics + orchestrator = 11 buckets)
- `build_combined` reads from new paths (`base/collection/data.jsonl`, `ai/jailbreaking/data.jsonl`, etc.) cleanly
- `--dataset all` → 16,982 pairs across 4 parents, cache key `3ecf6ee42505`

---

## [0.2.0] — 2026-06-10

**Multi-round SFT, run provenance, and dataset spec DSL.** This is a major version bump because the training loop architecture changed: runs are now self-describing (state.json), can be layered on top of each other (round-2 SFT), and the user-facing dataset selection was redesigned for clarity.

### Added

- **`state.json` sidecar at the root of every training run dir.** Records `base_model`, `hparams`, `dataset info` (including the spec list), `progress` (global_step, current/last loss, token accuracy, eval loss), and a `completed` flag. Written at training start with `completed: false`, updated on completion with `completed: true`. Enables:
  - **Auto-resume**: passing a started (incomplete) run dir as `--base-model` sets `resume_from_checkpoint=True` automatically. The trainer finds the latest `checkpoint-N/` and reloads model + optimizer + scheduler.
  - **Round-2 SFT auto-detection**: passing a completed run dir as `--base-model` loads the merged weights and trains a fresh LoRA on top. Surfaced with `↻ Round-2 SFT detected: previous completed run at {name}` and the previous run's hparams.
  - **Reproducibility**: `state.json[dataset.specs]` records the exact `--dataset` arguments, so re-running with the same specs reproduces the same combined dataset (same cache key).

- **Multi-positional `--dataset` flag (preferred over the legacy `--include-*` booleans).**
  ```bash
  --dataset base/                              # 10 tactics (7,398 pairs)
  --dataset base/ tools/                       # + 3 tools (15,859 pairs)
  --dataset base/ tools/metasploit/            # + just metasploit (15,747 pairs)
  --dataset tools/                             # 3 tools only (8,461 pairs)
  --dataset all                                # everything (16,982 pairs)
  ```
  Specs are dir-shaped (`base/`, `tools/`, `ai/`, `orchestrator`) and hierarchical (`tools/metasploit/` picks one bucket; `tools/` picks all). Aliases: `all`, `tactics`, `tools-all`. Bucket resolver: `bucket_loader.resolve_dataset_spec(s)`.

- **`--backup` / `--no-backup` round-2 SFT backup.** When round-2 SFT is detected, the previous run dir + merged model are tar.gz'd to `models/.backups/{name}_{timestamp}.tar.gz` with a progress bar. `--backup` is the default; `--no-backup` skips it. Tar size is ~5 GB for a 3B BF16 model (BF16 doesn't compress well; ~80% of uncompressed). The previous run dir is **never deleted** — it stays in `models/{name}_*/` for inspection.

- **Timestamped run dirs.** Each training run gets its own `models/{agent}_{YYYY-MM-DD}_{HH-MM}[_N]/` instead of clobbering `models/{agent}/`. Older runs remain on disk for rollback. Merged models still write to `models/merged/{agent}/` (single deployable artifact per agent).

- **`_find_latest_run_dir(agent_name)` helper.** Lexicographic sort on the timestamped dir names picks the most recent run for an agent. Used by the round-2 SFT auto-detect and the `merge_all` glob.

- **`_strip_timestamp_suffix(adapter_path)` helper.** Strips `_YYYY-MM-DD_HH-MM[_N]` from a directory name so merged-model output dir is `models/merged/{agent}/`, not `models/merged/{agent}_{timestamp}/`.

- **`dataset.specs` in `state.json`.** Persisted via `ATTACKLM_DATASET_SPECS` env var set by `train_all.py`. Future round-2 invocations can read the previous run's specs and reproduce the same combined dataset exactly.

### Changed

- **Default `--base-model` is now `None` instead of `Qwen/Qwen2.5-Coder-7B-Instruct`.** When `None`, `train_all.py` resolves the base in this order:
  1. Latest completed run for the same agent (round-2 SFT)
  2. `huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated` (the README-recommended abliterated base)
  3. Whatever `--base-model` the user passed explicitly (always wins)

  This makes round-2 SFT the default workflow instead of an opt-in.

- **`convert_to_gguf.py` rejects LoRA adapter directories with a helpful error.** Previously it let adapter dirs through and crashed at `convert_hf_to_gguf.py: Failed to detect model architecture`. Now: `ERROR: ... looks like a LoRA adapter, not a merged model.` with the actual base path auto-detected from `state.json` / `adapter_config.json`, plus the exact `attacklm-merge` command to fix it.

- **`merge_all` picks the most-recent adapter per agent.** Previously it tried to merge every adapter dir, which collided on output. Now groups by base name (timestamp stripped) and returns only the latest per group.

- **LM Studio install is gated on conversion success.** Previously a failed conversion was masked as a "successful install" (the glob picked up stale GGUFs from prior runs and re-copied them). Now tracks `converted_now` and only installs files produced this run.

- **Cache key for combined datasets is now based on resolved specs, not boolean flags.** `--include-tools` and `--dataset tools/` produce the same cache key (same content → same hash).

- **`.gitignore` excludes `data/datasets/combined/*.jsonl`** (regenerable, ~15-17 MB each; saves ~100 MB on the repo) and explicitly excludes `models/.backups/` (dotfile dir, also already excluded by `models/`).

### Fixed

- **`NameError: name 'os' is not defined` after training completed (v0.1.5 latent).** The trainer finished successfully but crashed on `os.makedirs(args.output, exist_ok=True)` in the save block, leaving stale adapters on disk. The actual trained adapter was in `checkpoint-13292/adapter_model.safetensors` and had to be recovered by hand. Added `import os` to `train_template.py` and a checkpoint-completeness pre-flight check.

- **`NameError: name 'os' is not defined` in `train_all.py` (v0.1.6).** New `os.environ["ATTACKLM_DATASET_SPECS"]` line was added in the `--dataset` feature without `import os`. Crashed the moment `build_train_cmd` was called. Added `import os`.

- **`AttributeError: 'Namespace' object has no attribute 'round_two_base'` in `train_all.py` (v0.1.6).** Vestigial reference to a flag that was never defined. Removed; the new auto-resolve logic makes it unnecessary.

- **`--base-model` literal default blocking round-2 detection.** The default was a literal string, so `if not args.base_model` was always False, and the round-2 detection block never ran in practice. Now the default is `None` and round-2 detection fires automatically.

- **`state.json` metric coercion.** Some HuggingFace versions serialize `eval_loss` and `mean_token_accuracy` as strings (`"4.647"` instead of `4.647`); the v0.1.6 state.json was recording them as `null`. Added `_coerce_float()` helper that tolerates both types.

- **`max_steps` in `state.json` was always 0.** Replaced the `initial_state["progress"]["max_steps"]` lookup (which was always 0 before training started) with reading `trainer_state.json` from the latest checkpoint, with a dataset-size fallback. Now records the actual step count.

- **`convert_to_gguf.py` had a `json` NameError too.** Same bug class. Added `import json`.

- **Stale GGUFs getting re-installed to LM Studio.** Fixed (see "Changed").

### Migration from v0.1.5

If you ran a training on v0.1.5, your output dir is at `models/{agent}/` (no timestamp). v0.2.0 expects timestamped dirs. Two options:

**Option A (recommended): manually rename + backfill state.json:**
```bash
# Rename the existing run dir
mv models/attacklm-single models/attacklm-single_2026-06-10_01-12

# Backfill state.json from the latest checkpoint-N/
python3 -c "
import json, shutil
from datetime import datetime, timezone
src = 'models/attacklm-single_2026-06-10_01-12'
ckpt_dirs = sorted([d for d in __import__('os').listdir(src)
                    if d.startswith('checkpoint-')])
ts = json.load(open(f'{src}/{ckpt_dirs[-1]}/trainer_state.json'))
# ... (see scripts/migrate_v015_to_v020.py for the full template)
"
```

**Option B (cleanest): just re-train.** v0.1.6 and later will write everything correctly from scratch.

The v0.1.5 → v0.2.0 migration script lives at `scripts/migrate_v015_to_v020.py` and handles the rename + backfill automatically.

---

## [0.1.5] — 2026-06-10

LM Studio path fix (`~/.lmstudio/models/local/`, third time's the charm), `kernels<0.13` pin with WHY comment in pyproject.toml, `_resolve_model_path()` helper for path-based `--base-model`, and `.gitignore` excludes for heretic output dirs (`n/`, `uncensored/`, `decensored/`).

## [0.1.4] — 2026-06-10

`merge_adapter.py`: auto-detect base from `adapter_config.json`, strip `-bnb-4bit` suffix, fix glob pattern, use `torch.bfloat16`. `convert_to_gguf.py`: add `import shutil`, `--input PATH` flag, `--install-lmstudio` opt-in, robust glob. End-to-end verified: 6.18 GB BF16 → 1.93 GB Q4_K_M GGUF.

## [0.1.3] — 2026-06-10

Real root cause of `Qwen3NextForCausalLM` error on ROCm: bitsandbytes 0.49.2 has no ROCm wheel. Exception chain walker in error handler reveals libbitsandbytes CUDA-only issue, HIP mismatch, or C++ extension failure. README points to rocm7.2 (torch 2.12.0 only on rocm7.1/7.2).

## [0.1.2] — 2026-06-10

ROCm support via `scripts/device_utils.py` (220 lines): `is_cuda`, `is_rocm`, `is_mps`, `backend`, `setup_allocator_env`, `enable_tf32`, `empty_cache_and_sync`, `gpu_mem_info`, `suggest_attn_implementation`, `print_hardware_banner`. Pyproject split: `[train-cuda]`, `[train-rocm]`, `[infer-cuda]`, `[infer-rocm]`, `[all-cuda]`, `[all-rocm]`. README rewritten with two clear stacks.

## [0.1.1] — 2026-06-10

Quant auto-detect via `detect_quantization_scheme()` in `train_template.py` (bitsandbytes_4bit/8bit, fp8, fbgemm_fp8, gptq, awq, compressed-tensors, torchao). HPO ceilings raised: lora_r 64→512 (7 steps), lora_dropout 0.3→0.5 (6 steps). Fixed `\j` Python 3.12+ SyntaxWarning in `demo.py`. Fixed `%` in `--packing` help.

## [0.1.0] — 2026-06-10

Initial public release. 11 console scripts (`attacklm-train`, `attacklm-train-all`, `attacklm-hpo`, `attacklm-infer`, `attacklm-merge`, `attacklm-gguf`, `attacklm-demo`, `attacklm-extract`, `attacklm-buckets`, `attacklm-attribute`, `attacklm-clone`). 16,982 training pairs across 16 buckets. 14 upstream sources credited in `ATTRIBUTION.md`. Apache-2.0 / MIT / AGPL-3.0 license mix documented.

---

[0.3.1]: https://github.com/Veedubin/AttackLM/releases/tag/v0.3.1
[0.3.0]: https://github.com/Veedubin/AttackLM/releases/tag/v0.3.0
[0.2.3]: https://github.com/Veedubin/AttackLM/releases/tag/v0.2.3
[0.2.2]: https://github.com/Veedubin/AttackLM/releases/tag/v0.2.2
[0.2.1]: https://github.com/Veedubin/AttackLM/releases/tag/v0.2.1
[0.2.0]: https://github.com/Veedubin/AttackLM/releases/tag/v0.2.0
[0.1.5]: https://github.com/Veedubin/AttackLM/releases/tag/v0.1.5
[0.1.4]: https://github.com/Veedubin/AttackLM/releases/tag/v0.1.4
[0.1.3]: https://github.com/Veedubin/AttackLM/releases/tag/v0.1.3
[0.1.2]: https://github.com/Veedubin/AttackLM/releases/tag/v0.1.2
[0.1.1]: https://github.com/Veedubin/AttackLM/releases/tag/v0.1.1
[0.1.0]: https://github.com/Veedubin/AttackLM/releases/tag/v0.1.0
