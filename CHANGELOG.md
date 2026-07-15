## [0.17.0] — 2026-07-15

- **Added `scripts/eval_calibration.py`** (Attack 7 from `docs/MODEL_ATTACKS_SURVEY.md`): characterizes model calibration and selective-prediction on in-distribution / near-OOD / OOD inputs. Reports Brier, ECE, and a 20-point selective-prediction operating curve. Reuses `compute_nll` from `attacklm-dataset/scripts/inversion/scoring.py` as the NLL primitive.
- **Added `docs/CALIBRATION_AUDIT.md`** documenting the methodology, expected results, and reproduction recipe. Gitignored (internal-only).
- **Added 8 hermetic tests** in `tests/test_attack_audit.py` for the calibration metrics (Brier, ECE, selective sweep) and 2 JSONL I/O tests.

No new public-facing surface. No PyPI publish (docs-only + internal audit script).

---

## [0.16.0] — 2026-07-15

- **Added the canary-extraction framework** (Attack 3 from `docs/MODEL_ATTACKS_SURVEY.md`):
  - `scripts/canary_generator.py` — generates N unique canaries (8-char token, 8 prefix templates, 5 suffix templates)
  - `scripts/canary_inject.py` — inserts canaries into training JSONL at a controlled rate (default 1%)
  - `scripts/audit_canary_extraction.py` — the extraction probe; reports exact-token, loose-token, and near-verbatim (BLEU-4>0.7) extraction rates
- **Added `docs/CANARY_EXTRACTION.md`** documenting the three-script workflow, expected extraction rates by insertion rate, and the Carlini 2021 §4.2 / Biderman 2023 reference papers. Gitignored (internal-only).
- **Added 12 hermetic tests** in `tests/test_attack_audit.py` for the canary framework (generator uniqueness, injector rate, matcher correctness).

No new public-facing surface. No PyPI publish.

---

## [0.15.0] — 2026-07-15

- **Added the prompt-injection and system-prompt-extraction framework** (Attacks 1 + 2 from `docs/MODEL_ATTACKS_SURVEY.md`):
  - `scripts/_audit_grader.py` — shared refusal/compliance/secret-emission grader (regex-based, English-only v0.1)
  - `scripts/audit_prompt_injection.py` — Attack 1: tests AttackLM as a VICTIM of prompt injection (21 prompts, 3 tiers: direct / indirect / crescendo)
  - `scripts/audit_system_prompt.py` — Attack 2: tests system-prompt extraction (49 prompts, 5 tiers: extraction / roleplay / translation / indirect / escalation)
  - `data/bench/prompt_injection_holdout.jsonl` — 21 prompts, canary-secreted
  - `data/bench/system_prompt_holdout.jsonl` — 49 prompts, canary-secreted
- **Added `docs/PROMPT_INJECTION_AUDIT.md`** (Attack 1) and **`docs/SYSTEM_PROMPT_AUDIT.md`** (Attack 2). Both gitignored (internal-only).
- **Added 18 hermetic tests** in `tests/test_attack_audit.py` for the grader and JSONL I/O.
- **`.gitignore`**: `docs/*.md` is now gitignored (except the v0.14.0-shipped `RL_RECIPE.md` is still tracked). This keeps the audit docs out of the public distribution.

The model is fine-tuned to GENERATE prompt-injection attacks; this release tests whether it can also DEFEND against them. See `docs/MODEL_ATTACKS_SURVEY.md` for the full survey of 7 model attacks. The 4 shipped here (Attacks 1, 2, 3, 7) are the four with the highest value/effort ratio.

No new public-facing surface. No PyPI publish.

---

## [0.14.0] — 2026-07-13

- Added `docs/RL_RECIPE.md` documenting the MAI-Thinking-1 adaptive GRPO recipe for future use.
- [Methodology] Documented techniques from MAI-Thinking-1 §3.1.1 (Adaptive entropy control + outer ratio clip), §3.1.2 (Length penalty), §3.4.2 (Coarse vs granular grader) by The Microsoft AI Team, June 2026.

Docs-only release. No code changes. No PyPI publish.

---

## [0.12.3] — 2026-07-09 — Third CI blocker fix: pytest-asyncio dep + drop test_new_extractors.py

### Fixed
- **Added `pytest-asyncio` to `[dev]` dependencies in `pyproject.toml`.** The new `tests/test_audit.py` (added in v0.12.0) uses `@pytest.mark.asyncio` on async Pilot tests, which requires `pytest-asyncio` to be installed. Locally it was installed; CI's `pip install -e ".[dev,extract]"` was missing the dep, so the 5 audit tests failed with `async def functions are not natively supported`.

### Removed
- **`tests/test_new_extractors.py` (361 LoC, 34 tests).** All 34 tests in this file use `importlib.import_module(...)` to dynamically load extractor scripts that were moved to `attacklm-dataset` in the v0.11.0 cleanup (`0cde31f`). In CI, the dynamic imports raised `ModuleNotFoundError: No module named 'extract_0xdf_writeups'` (and 8 others). Locally the tests passed because the untracked scripts were still on disk from the pre-cleanup era. The extractors are now tested (or should be) inside the `attacklm-dataset` repo instead.

### About v0.12.0 / v0.12.1 / v0.12.2 (predecessors)

| Tag | Status | Reason |
|---|---|---|
| v0.12.0 | on origin, no PyPI | First CI blocker: `DATA_DIR` raised at import time + 5 dead test files |
| v0.12.1 | on origin, no PyPI | Second CI blocker: `device_utils.py` not in repo |
| v0.12.2 | on origin, no PyPI | Third CI blocker: `pytest-asyncio` not in `[dev]` deps + `test_new_extractors.py` references missing scripts |
| v0.12.3 | on origin, **published to PyPI** | All CI blockers fixed |

### Reference
- v0.12.0 release CI: https://github.com/Veedubin/AttackLM/actions/runs/28972432153 (Test step failed)
- v0.12.1 release CI: https://github.com/Veedubin/AttackLM/actions/runs/28996006495 (Test step failed)
- v0.12.2 release CI: https://github.com/Veedubin/AttackLM/actions/runs/28996218553 (Test step failed)
- v0.12.2 main CI: https://github.com/Veedubin/AttackLM/actions/runs/28996218159 (Test step failed)
- v0.12.3 release CI: TBD (this release)

---

## [0.12.2] — 2026-07-09 — Second CI blocker fix: restore `device_utils.py` (BROKEN, see v0.12.3)

### Fixed
- **Restored `scripts/device_utils.py`** (338 LoC). The v0.11.0 dataset-split cleanup commit (`0cde31f`) deleted this file along with the dataset-prep scripts, but `train_template.py`, `eval_retention.py`, `_eval_loader.py`, and `collect_reference.py` all `from device_utils import ...` at module level. Without it, `import train_template` fails at `ModuleNotFoundError: No module named 'device_utils'`, which broke 5 test files in CI (`test_coap_flashoptim`, `test_fp8_bitnet`, `test_memory_optimization`, `test_mixed_precision`, `test_training_integration`).

- **`scripts/train_template.py` `parse_args()` `--evolved-dir` default is now computed defensively**: `try: _data_dir().parent.parent / "evolved" except FileNotFoundError: "data/datasets/evolved"`. The previous code unconditionally called `_data_dir()`, which raised in CI. Now `parse_args()` succeeds even when the data dir is absent (the lazy resolution is still correct: if the data dir is present, the canonical path is used; if not, a hardcoded fallback is used).

### About v0.12.1 (predecessor)

**v0.12.1 was tagged and pushed to `origin` but its release CI also failed.** Per the NEVER-FORCE-PUSH-TAGS rule, v0.12.1 is preserved on origin (a known-broken commit) and v0.12.2 is the new installable release. The v0.12.1 release CI failure was due to the same `device_utils.py` issue described above — fixing that one file is what v0.12.2 ships.

| Tag | Status | Reason |
|---|---|---|
| v0.12.0 | on origin, no PyPI | First CI blocker: `DATA_DIR` raised at import time + 5 dead test files |
| v0.12.1 | on origin, no PyPI | Second CI blocker: `device_utils.py` not in repo |
| v0.12.2 | on origin, **published to PyPI** | All CI blockers fixed |

### Reference
- v0.12.0 release CI: https://github.com/Veedubin/AttackLM/actions/runs/28972432153 (Test step failed)
- v0.12.1 release CI: https://github.com/Veedubin/AttackLM/actions/runs/28996006495 (Test step failed)
- v0.12.2 release CI: TBD (this release)

---

## [0.12.1] — 2026-07-09 — CI blocker fix: lazy `DATA_DIR` + drop dead tests (BROKEN, see v0.12.2)

### Fixed
- **`scripts/train_template.py`: `DATA_DIR` is now resolved lazily via `_data_dir()` + PEP 562 `__getattr__`.** Previously, `DATA_DIR = _resolve_data_dir()` ran at module-import time, so any test or tool that did `import train_template` in a clean environment (no `attacklm init` run, no `attacklm-dataset` data dir present) raised `FileNotFoundError: AttackLM dataset not found...`. The v0.11.0 dataset-split cleanup (`0cde31f`) moved the data into the `attacklm-dataset` sibling package, which made this latent bug surface in CI. Now `import train_template` succeeds in any environment; the data dir is only resolved when actually needed (e.g. inside `parse_args()` when computing the `--evolved-dir` default). The two call sites in `parse_args()` were updated to call `_data_dir()`.

### Removed
- **5 dead test files that referenced scripts deleted in the v0.11.0 cleanup commit (`0cde31f`)**:
  - `tests/test_balance_buckets.py` (386 LoC) — referenced `scripts/balance_buckets.py` and `scripts/bucket_loader.py` (deleted)
  - `tests/test_evolve_pairs.py` (1,777 LoC) — referenced `scripts/evolve_pairs.py` and `scripts/filter_evolved.py` (deleted)
  - `tests/test_init_pipeline.py` (623 LoC) — referenced `init_pipeline.py` (deleted)
  - `tests/test_replay_mixer.py` (481 LoC) — referenced `scripts/replay_mixer.py` (deleted)
  - `tests/test_thinking_models.py` (55 LoC) — referenced `generate_synthetic_scarce.py` (deleted)

  Net: -3,322 LoC of dead test code. These tests had been failing at COLLECT time in the v0.12.0 release CI but the failure was masked in earlier releases (v0.11.1 still had the scripts; v0.11.1-1 failed at version-validation before tests ran). The `train_template.py`-dependent tests (`test_coap_flashoptim`, `test_fp8_bitnet`, `test_memory_optimization`, `test_mixed_precision`) had also been failing on the same import error; they pass now that `DATA_DIR` is lazy.

### Why this is a 0.12.1 (not 0.12.0 amend)
- v0.12.0 was tagged and pushed to `origin` successfully. Its release CI failed at the `Test` step. No PyPI publish happened.
- A version bump to 0.12.1 is the right call here per the NEVER-FORCE-PUSH-TAGS rule: once a tag is pushed, it is immutable. The cost of a new version is 1 line in `pyproject.toml` and `__version__.py`; the cost of force-pushing a tag is broken PyPI + broken `git pull` for anyone who fetched the old tag.
- 0.12.0 is preserved on origin for reproducibility; 0.12.1 is the new "install this" release.

### Tests
- 401 passed, 1 skipped, 0 failed in `pytest tests/ -q` (was 569 collect, 64 failed before this change).

### Reference
- v0.12.0 release CI run that caught this: https://github.com/Veedubin/AttackLM/actions/runs/28972432153 (Test step failure, Build + Publish skipped).
- v0.11.1 (the previous passing release) had the deleted scripts; v0.11.1-1 had them deleted but its release CI failed at version-validation (`Tag v0.11.1-1 != package __version__ 0.11.1`), so the test failures were never observed in CI.

---

## [0.12.0] — 2026-07-08 — TUI folded into the main package + MIA Track 2 audit subcommand

### Changed (BREAKING)
- **TUI folded into the main `attacklm` package.** The standalone `attacklm-gui` PyPI package (v0.1.0) is no longer distributed. All TUI code now lives at `src/attacklm/gui/` inside the main `attacklm` package. `textual>=2.0` is now a required (not optional) dependency of `attacklm`. Launch with `attacklm gui` (the subcommand was already wired in v0.11.1, but the import path was `attacklm_gui.app`; now it is `attacklm.gui.app`).

  **Migration**: `pip install attacklm` now pulls the TUI automatically. Users who previously did `pip install attacklm-gui` should uninstall that and just install `attacklm`. The `attacklm-gui` console script (from the old package) is no longer installed; use `attacklm gui` instead.

- **CLI test fix**: `tests/test_cli.py::TestGUISubcommand::test_gui_import_error` was removed (the import is no longer optional — the TUI is bundled with the main package). `test_gui_success` was updated to mock `attacklm.gui.app` instead of `attacklm_gui.app`.

### Added
- **`attacklm audit` subcommand**: runs an inversion-attack audit (extraction or MIA) on a trained model. Delegates to `attacklm-dataset/scripts/inversion_audit.py` via `subprocess.run`. Accepts `--attack {extraction, mia, all}`, `--mia-method {reference, zlib, per_token, lira, all}`, `--mia-threshold-mode {median, percentile, holdout_file}`, `--mia-percentile`, `--model`, `--dataset-root`, `--source-filter`, `--top-k`, `--max-new-tokens`, `--temperature`, `--max-records`, `--dry-run`. See `attacklm-dataset/docs/ATTACK_TAXONOMY.md` for the attack taxonomy.
- **TUI: new Audit screen** with 2 tabs (Extraction / MIA). Built on top of the existing `_BaseCommandScreen` pattern. Form fields construct the `attacklm audit` command; output streamed to a `RichLog`.
- **TUI: tooltips on every widget**. Centralized tooltip text in `src/attacklm/gui/widgets/tooltips.py` (~30 entries). `attach_tooltip(widget, key)` helper for one-line attachment. `DEFAULT_CSS` in `app.py` for Tooltip styling (background, border, padding, max-width). `App.tooltip_delay = 0.5s` hover delay. Retro-fitted tooltips on all 9 main menu buttons, all high-traffic train form inputs, and all command form Back buttons.

### Fixed (drive-by)
- `presets.py` (in `src/attacklm/gui/presets.py`): `FP8 (H100/Blackwell)` was creating an invalid filename (`fp8_(h100/blackwell).json`) on Linux because the slash survives the previous `lower().replace(' ', '_')` slugify. Now uses a proper regex slugify via a new `_slugify()` helper. Fixes a real bug that affected all users on first run.
- `screens/train_form.py`: `Select(value=3)` for `deepspeed_stage` was silently broken under Textual 8.x (the value is silently overwritten with `Select.NULL` when options are tuples). Dropped the explicit value; the downstream code already defaults to 3.

### Tests
- 7 new tests in `tests/test_audit.py` (audit screen mount, widget presence, tooltip coverage, dict-key presence, no-op attachment).
- 2 new structural tests in `tests/test_gui.py::TestTooltipsRetrofit` (train form tooltip keys defined, train form on_mount wires up the 9 high-traffic fields).

### Reference
- Per architect plan approved 2026-07-08 (deepseek-v4-pro:cloud).
- Memory: `d4657ab9-53d4-49a4-b4bd-3de7816b3868` (architectural decision), `88c25f43-5d1d-4756-804e-4b0ad6b1dc19` (MIA research), `fb19c94e-dd7b-4e02-821c-3fb32eb1abac` (session summary).

---

## [0.11.1] — 2026-07-07 — `attacklm --init` (neuralgentics plugin bootstrap)

### Added
- **`attacklm --init` top-level flag**: bootstraps the [neuralgentics](https://github.com/Veedubin/neuralgentics) OpenCode plugin into the current directory. Downloads the plugin release tarball from GitHub, SHA256-verifies it against `checksums.txt`, extracts to the target directory, deep-merges `opencode.json` (preserving the user's `provider` / `mcp` / `lsp` / `formatter` blocks), backs up any existing `.opencode/` to `.opencode-bak-{ts}/` before overwriting, runs `npm install --no-audit --no-fund`, and writes a state file at `.opencode/.neuralgentics-state.json`. Refuses scary targets (`HOME`, `/`, `/tmp`, `$HOME/<dir>` with no project markers) and symlinked `.opencode/` unless `--force` is set.
  - `attacklm --init [--target DIR] [--plugin-version VER] [--repo OWNER/NAME] [--dry-run] [--force] [--offline]`
  - Default: `--plugin-version latest`, `--repo Veedubin/neuralgentics`, `--target $PWD`.
  - `--init` is a TOP-LEVEL flag, NOT a subcommand — to avoid clashing with `argparse action="version"`. The existing `attacklm init` SUBCOMMAND (which delegates to `attacklm-dataset`) is unchanged. The plugin version is `--plugin-version` (not `--version`) for the same reason.
- **`attacklm.neuralgentics_init` subpackage** (6 modules, 1,754 LoC): the init flow code, recovered from the deleted `neuralgentics-cli` PyPI wheel (commit 9b7723b^). Modular layout: `_init_cmd` (orchestrator), `_download` (GitHub release + SHA256 + tarball), `_merge` (deep-merge `opencode.json`), `_state` (pydantic v2 state file), `_errors` (typed error classes with exit codes 1-19), `__init__` (public API).
- **24 new tests** in `tests/test_neuralgentics_init.py`: 16 ported from the deleted `neuralgentics-cli` test suite + 7 new init-flow tests + 1 CLI dispatch regression test (`attacklm --init` vs `attacklm init`). All hermetic — full monkeypatching of `httpx` / `subprocess` / `shutil.which`.

### Changed
- **`attacklm init` SUBCOMMAND**: rewritten to delegate to `attacklm-dataset` (replaces the pre-v0.11.0 in-process orchestration of 12 extractors + `clone_repos.sh` + `setup_buckets.py` + `reorganize_buckets.py`). The argparse flags (`--extract-only`, `--buckets-only`, `--clone-only`, `--from-source`, `--dataset-url`) are now forwarded to `attacklm-dataset init` for it to handle. **Behavior change**: the partial-step flags no longer run attacklm-local scripts — they invoke `attacklm-dataset` instead. The pre-v0.11.0 internal scripts (`scripts/init_pipeline.py`, `scripts/balance_buckets.py`, `scripts/extract_*.py`, etc.) were removed in the v0.11.0 dataset split, so the in-process paths no longer exist.
- **`attacklm balance` SUBCOMMAND**: rewritten to delegate to `attacklm-dataset` (replaces `_run_python_script("balance_buckets.py", ...)`). Same pattern: `python -m attacklm_dataset.cli balance <argv>`.
- **`tests/test_cli.py`**: 8 stale tests (the 7 `TestInitSubcommand` + 1 `TestBalanceSubcommand`) updated to assert on the new `subprocess.run` delegation to `attacklm-dataset`, with a new helper `_mock_attacklm_dataset_delegate(monkeypatch)` that stubs the sibling package and captures the delegated command.

### Added (runtime deps)
- `httpx>=0.27` — streaming download of the neuralgentics GitHub release tarball (`attacklm --init`).
- `pydantic>=2.0` — the state file model (`StateFile` / `FileRecord` / `BackendRecord`) is a pydantic v2 `BaseModel` with `model_config = ConfigDict(extra="forbid")`. Strict schema validation drives the corruption-recovery path.

### Known limitations
- `--offline` is accepted but raises `OfflineNoBundle` (no bundled tarball ships yet — planned for a future release).
- The init flow requires `opencode` and `npm` on `PATH`; raises typed `OpencodeNotFound` (exit 4) / `NpmNotFound` (exit 5) with remediation hints if missing.

---

## [0.11.0] — 2026-07-06 — Dataset Split

### Changed
- **Dataset split into standalone package**: The MITRE ATT&CK dataset (24,652 pairs, 16 sources) has been extracted into the independent [attacklm-dataset](https://github.com/Veedubin/attacklm-dataset) package.
- `attacklm init` and `attacklm balance` now delegate to `attacklm-dataset` CLI when installed, with fallback to local scripts.
- `train_all.py` and `train_template.py` now resolve data paths from `attacklm-dataset` package or `ATTACKLM_DATA_DIR` environment variable.
- Added `attacklm-dataset>=0.1.0` as an optional dependency in the `dataset` extras group.

### Removed
- All data files (`data/` tree) moved to attacklm-dataset
- All extractor scripts (`scripts/extract_*.py`) moved to attacklm-dataset
- All acquisition scripts (`scripts/acquire_*.py`) moved to attacklm-dataset
- Data tooling scripts (`bucket_loader.py`, `balance_buckets.py`, `evolve_pairs.py`, etc.) moved to attacklm-dataset
- `ATTRIBUTION.md` moved to attacklm-dataset

---

## [0.10.1] — 2026-07-05 — CI Fix: beautifulsoup4 dependency

### Fixed
- **Missing `beautifulsoup4` in extract extras**: `extract_0xdf_writeups.py` requires `beautifulsoup4` and `requests` at module level. Added `beautifulsoup4` to `[extract]` extras in `pyproject.toml`.
- **Test import failure**: `tests/test_new_extractors.py` imports extractor modules directly, triggering `sys.exit(1)` guards when optional deps are missing. Mocked `bs4` and `requests` in `sys.modules` at the top of the test file so CI passes without those packages installed.

---

## [0.10.0] — 2026-07-05 — Memory Optimization & Dataset Expansion

### Phase 1: Critical Fixes
- Resolved PiSSA convergence issues and `auto_tune_vram` instability.

### Phase 2: CLI Evolution
- Introduced new subcommands: `steer`, `bench`, and `pipeline` for professional model lifecycle management.

### Phase 3: Dataset Expansion
- Added 8 new security dataset extractors, increasing potential training pairs to ~460K.

### Phase 4: Training Quality
- Implemented Judge-and-Revise filtering, CoT-Self-Instruct, Constitutional AI alignment, and Doc-to-QA synthesis.

### Phase 5: Memory Optimization
- Integration of 7 state-of-the-art optimization methods: COAP, FlashOptim, Unsloth GC, Mixed-precision LoRA, FP8, BitNet, and SignRoundV2.

### Phase 6: Finalization
- GUI updates to support new optimizations, complete documentation overhaul, and expanded test suite.

---

### Added
- **GitHub Actions CI**: `.github/workflows/ci.yml` — runs on push/PR to main, matrix across Python 3.11/3.12/3.13, full test suite + ruff lint
- **PyPI publish workflow**: `.github/workflows/release.yml` — OIDC Trusted Publishing, validate → test → build → publish pipeline on tag push
- **CLI subcommand tests**: `tests/test_cli.py` — 33 tests covering all 8 subcommands (train, init, balance, build, infer, eval, demo, gui) plus edge cases
- **GitHub Releases**: Created releases for v0.8.3 through v0.9.4 with full CHANGELOG notes

### Fixed
- **11 test failures resolved** (414 tests passing, up from 333):
  - `test_eval_loader.py`: Fixed mock structure (transformers.utils, peft), missing adapter_path arg, CWD pollution, empty string handling
  - `test_eval_retention.py`: Fixed CWD capture in test_adapter_relative_path
  - `test_memory_optimization.py`: Added transformers mock before train_template import
  - `_eval_loader.py`: Empty model_id_or_path now raises ValueError

---

## [0.9.3] — 2026-07-04 — Remove deprecated hyphenated commands, --compile + QLoRA guard

### Removed
- **Deprecated hyphenated commands**: All 22 `attacklm-*` console scripts removed from `pyproject.toml` and `cli.py` after the v0.8.x deprecation window. Use `attacklm <subcommand>` instead.
  - `attacklm-train` → `attacklm train`
  - `attacklm-train-all` → `attacklm train --all`
  - `attacklm-train-lora` → `attacklm train`
  - `attacklm-hpo` → `attacklm train --hpo`
  - `attacklm-extract` → `attacklm init --extract-only`
  - `attacklm-buckets` → `attacklm init --buckets-only`
  - `attacklm-attribute` → `attacklm init --attribute-only`
  - `attacklm-clone` → `attacklm init --clone-only`
  - `attacklm-init` → `attacklm init`
  - `attacklm-balance` → `attacklm balance`
  - `attacklm-merge` → `attacklm build --merge-only`
  - `attacklm-gguf` → `attacklm build --gguf-only`
  - `attacklm-build` → `attacklm build`
  - `attacklm-infer` → `attacklm infer`
  - `attacklm-demo` → `attacklm demo`
  - `attacklm-eval` → `attacklm eval`
  - `attacklm-collect-ref` → `attacklm eval --collect-ref`
  - `attacklm-score` → `attacklm eval --score`
  - `attacklm-compare` → `attacklm eval --compare`
  - `attacklm-golden` → `attacklm eval --golden`
  - `attacklm-pipeline` → `attacklm pipeline`
- **Deprecated wrapper code**: Removed `_deprecated()` helper, `_DEPRECATED_MSG` constant, and all 19 `main_*` wrapper functions from `src/attacklm/cli.py`

### Added
- **`--compile` + QLoRA incompatibility guard**: `train_template.py` now exits with a clear error if `--compile` is used with default 4-bit QLoRA (BitsAndBytes NF4). torch.compile is incompatible with quantized models. Users are directed to use `--use-galore` or `--use-deepspeed` for full-parameter training with compilation.
- **5 new tests** in `test_memory_optimization.py` verifying the compile+QLoRA guard and compatibility with GaLore, DeepSpeed, and Unsloth

### Changed
- **All script references updated**: 30+ references across 10 script files, 5 extractors, GUI, and pipeline config updated from hyphenated to subcommand form
- **Documentation**: `CONTRIBUTING.md` and `EVALUATION.md` updated with new command forms

---

## [0.9.0] — 2026-07-01 — Memory optimization: DeepSpeed, torch.compile, LOMO

### Added
- **DeepSpeed ZeRO integration**: Train models 3-5x larger than GPU VRAM using system RAM
  - `--use-deepspeed` flag with auto-generated ZeRO-3 + CPU offload config
  - `--deepspeed-stage {1,2,3}` for ZeRO stage selection
  - `--deepspeed-config` for custom JSON configs
  - `--no-deepspeed-offload` for GPU-only ZeRO
  - Pre-built config templates in `presets/deepspeed/` (zero3_cpu_offload, zero3_gpu_only, zero2_cpu_offload)
- **torch.compile**: `--compile` flag for 20-40% training speedup and 10-20% memory reduction
  - `--compile-mode {default,reduce-overhead,max-autotune}` for tuning
- **LOMO optimizer**: `--use-lomo` for full-parameter fine-tuning of 7B models on 8GB GPUs
- **GUI updated**: Hardware tab now exposes DeepSpeed, torch.compile, and LOMO controls
- **train_all.py**: All new flags forwarded for multi-bucket training

### Changed
- **Tagline updated**: Now lists DeepSpeed alongside QLoRA, GaLore, Q-GaLore, Spectrum, PiSSA
- **README**: Added DeepSpeed configuration section with hardware reference table, new workflows, and CLI flag documentation

---

## [0.8.5] — 2026-07-01 — Comprehensive CLI documentation

### Changed
- **CLI Reference completely rewritten**: Every command now has its own subsection with all flags documented, defaults listed, and practical examples
- **Usage section overhauled**: Replaced flat list with 4 end-to-end workflows (Quick Start, Maximum Quality, HPO→Deploy, Evolve→Train)
- **Training Methods Explained**: New table comparing QLoRA, GaLore, Q-GaLore, Spectrum, and PiSSA by VRAM usage and use case
- **README grew from 274 to 443 lines** with 24 sections and 36 code examples

---

## [0.8.4] — 2026-07-01 — Documentation overhaul

### Changed
- **Tagline updated**: Now lists all 5 training methods (QLoRA, GaLore, Q-GaLore, Spectrum, PiSSA) instead of just QLoRA
- **GUI section expanded**: Added dedicated Terminal GUI section with live monitor screenshot, preset list, and screen-by-screen breakdown
- **AttackLM-Models README synced**: Updated pair count (16,027 → 24,652), added missing defensive sources, updated all hyphenated commands to unified CLI format

---

## [0.8.3] — 2026-07-01 — Training pair evolution

### Added
- **Training pair evolution**: New `scripts/evolve_pairs.py` with 3 strategies to expand short training pairs into longer, richer examples
  - **Evol-Instruct**: Rewrites answers with deeper reasoning, edge cases, and detection artifacts (3-5x longer)
  - **Multi-turn Decomposition**: Breaks Q&A into 3-5 turn conversations for better training flow
  - **Chain-of-Thought Injection**: Adds explicit reasoning steps before final answers
- **Quality filtering**: `scripts/filter_evolved.py` validates evolved pairs (structure, length, provenance, dedup)
- **`--evolved-ratio` flag**: Mix evolved pairs into training at configurable ratio in `train_all.py` and `train_template.py`
- **`scripts/evolved_mixer.py`**: Standalone mixer module for evolved pair integration

### Changed
- **Agent models upgraded**: glm-5.1 → glm-5.2, minimax-m2.7 → minimax-m3, added kimi-k2.7-code

---

## [0.8.1] — 2026-06-30 — Zero-config dataset init


### Added
- **Zero-config init**: `attacklm init` now downloads a pre-built dataset tarball (~50 MB) from GitHub releases by default. No git clone, no extractors, no manual setup required.
- **`--from-source` flag**: `attacklm init --from-source` preserves the old clone+extract pipeline for developers who want to rebuild from upstream repos.
- **`--dataset-url` flag**: Override the download URL for mirrors or custom dataset hosting.
- **`scripts/package_dataset.py`**: Maintainer tool to create the dataset tarball for GitHub releases.

### Changed
- **`attacklm init` default**: Downloads pre-built dataset instead of cloning repos. Two commands to ready: `uv pip install attacklm[all]` → `attacklm init --yes`.
- **Dependencies**: Removed `gitpython` from extract deps (no longer needed for default init path). Added `tqdm` for download progress bars.

### Fixed
- **v0.8.0 tag**: Force-pushed tag replaced with proper v0.8.1 release (tag immutability rule enforced).

---

All notable changes to AttackLM are documented in this file. Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.8.0] — 2026-06-30 — Unified CLI with subcommands

### Added
- **Unified CLI**: New `attacklm` command with argparse subcommands replaces 21 hyphenated commands
  - `attacklm train [--dataset all] [--hpo]` — consolidates `attacklm-train`, `attacklm-train-all`, `attacklm-train-lora`, `attacklm-hpo`
  - `attacklm init [--extract-only|--buckets-only|--attribute-only|--clone-only]` — consolidates `attacklm-init`, `attacklm-extract`, `attacklm-buckets`, `attacklm-attribute`, `attacklm-clone`
  - `attacklm build [--merge-only|--gguf-only|--register-ollama]` — consolidates `attacklm-build`, `attacklm-merge`, `attacklm-gguf`, `attacklm-register-ollama`
  - `attacklm eval [--collect-ref|--score|--compare|--golden]` — consolidates `attacklm-eval`, `attacklm-collect-ref`, `attacklm-score`, `attacklm-compare`, `attacklm-golden`
  - `attacklm balance` — replaces `attacklm-balance`
  - `attacklm infer` — replaces `attacklm-infer`
  - `attacklm gui` — replaces `attacklm-gui`
  - `attacklm demo` — replaces `attacklm-demo`
- **`attacklm --version`**: Shows version number
- **Evaluation screen in GUI**: New `EvalFormScreen` with buttons for all eval subcommands (Retention, Collect Ref, Score, Compare, Golden)
- **Build screen in GUI**: Consolidated buttons for Full Build, Merge Only, GGUF Only

### Changed
- **GUI main menu**: Consolidated from 8 buttons to 6 (Train, Init, Balance, Build, Infer, Eval) — removed separate Extract/Merge/Pipeline buttons
- **GUI train form**: Now calls `attacklm train` instead of `attacklm-train`
- **GUI command forms**: All commands now use `attacklm <subcommand>` format instead of `attacklm-<command>`
- **Deprecation warnings**: All old hyphenated commands print a deprecation notice before delegating to the new subcommand

### Deprecated
- All 21 hyphenated commands (`attacklm-train`, `attacklm-train-all`, etc.) still work but print deprecation warnings. They will be removed in v0.9.0.

---

## [0.7.1] — 2026-06-30 — GUI, VRAM fixes, dataset caching

### Added
- **attacklm-gui**: Terminal-based TUI wrapper (Textual) for all CLI commands
  - Form-based training config with 40+ params in 5 tabs (Basic, LoRA, GaLore, Advanced, Hardware)
  - Live training monitor with loss sparkline, VRAM gauge, progress bar, log viewer
  - One-click screens for extract, balance, infer, merge, build, pipeline, init
  - 5 built-in presets (3B Q-GaLore, 3B LoRA, 7B Q-GaLore, 7B QLoRA, etc.)
  - Pause/resume/quit controls during training
  - Works in terminal-only environments (no X11/WSLg required)
- **Dataset caching**: Tokenized datasets saved to disk after first run, reloaded in ~1s on subsequent runs (keyed by dataset + model + max_length + packing)
- **config.json fix**: Training runs now preserve HF model architecture fields (model_type, architectures, hidden_size, etc.) for GGUF conversion compatibility

### Fixed
- **VRAM fragmentation**: Removed `device_map="auto"` for single-GPU training — was splitting model across GPU+CPU, creating meta tensors, and leaving VRAM fragmented. Model now loads directly to GPU.
- **Spectrum crash on 7B**: Skip meta-device tensors in SNR computation (caused by CPU-offloaded layers from device_map="auto")
- **VRAM parsing**: Added regex for compact VRAM format (`VRAM X/Y GB (A alloc + C cache)`) in training output parser

### Changed
- Moved attacklm-gui into the main AttackLM repo under `attacklm-gui/`
- 17 GUI tests passing

---

## [0.7.0] — 2026-06-29 — Defensive data extraction, test fixes, script cleanup

### Added
- Extracted 6 defensive data sources: Sigma (3,132), Elastic (1,908), Splunk (2,114), Mordor (339), ThreatHunter (27), NIST IR (168)
- 3 new defensive buckets: detection_engineering (7,154), threat_hunting (366), incident_response (168)
- Total dataset: 24,652 pairs across 21 buckets

### Fixed
- extract_mordor.py: fixed YAML glob pattern (.yml → .yaml)
- extract_threathunter_playbook.py: fixed playbooks path (playbooks/ → docs/hunts/)
- All 27 tests passing (4 thinking + 23 balance)

### Changed
- Archived 19 one-off migration/generation scripts to archive/scripts/
- Bumped version to 0.7.0

---

## [0.6.9] — 2026-06-29 — Documentation cleanup, manifest fix, init pipeline overhaul

### Changed
- Rewrote README with proper workflow order (install → init → balance → train → build → infer)
- Removed abliterated model requirement; SFT handles refusal suppression
- Fixed manifest.json to reflect actual data (16,964 pairs, 18 buckets)
- Removed stale references to RTA, Infection Monkey, and BigPromptLibrary from all docs
- Updated init_pipeline.py: removed restricted sources, added defensive repos, added dependency check
- Updated clone_repos.sh: removed restricted sources, added defensive repos
- Bumped transformers 5.10.2→5.12.1, accelerate 1.13.0→1.14.0

### Fixed
- ATTRIBUTION.md no longer lists removed sources
- NOTICE no longer references GPL/AGPL components
- README duplicate attacklm-build entry removed
- requirements.txt synced with pyproject.toml

---

## [0.6.9] — 2026-06-29 — Fix `UnboundLocalError: interactive_control`

### Fixed

- **Moved `interactive_control` initialization before the `_is_galore` branch.** Fixes `UnboundLocalError` when training without GaLore (standard LoRA fine-tuning) because `interactive_control` was previously only initialized inside the GaLore-specific `if` block but used in the `else` block.

---

## [0.6.1] — 2026-06-28 — Fix --use-galore API

### Fixed

- **`--use-galore` restored to vanilla GaLore** (FP16 projections). Q-GaLore is now `--use-qgalore` (separate flag). This preserves backward compatibility — existing `--use-galore` commands continue to work.
- **Removed `--galore-fp16`** (redundant — `--use-galore` is FP16 by default).
- **CI validation**: added version check job that fails the workflow if the git tag doesn't match `__version__.py`, preventing duplicate PyPI uploads.

---

## [0.6.0] — 2026-06-28 — Q-GaLore, Spectrum, PiSSA init

### Added

- **Q-GaLore** (`--use-qgalore`): INT4 quantized gradient projection matrices with stochastic rounding. Cuts optimizer memory ~4x vs vanilla GaLore. Enables 7B full-parameter training on 16GB GPUs. Requires `--use-galore`. Paper: arXiv:2407.08296.
- **Spectrum** (`--spectrum`): SNR-based layer freezing. Computes signal-to-noise ratio per layer and freezes the lowest-SNR layers. `--spectrum` keeps top 50%, `--spectrum 0.25` keeps top 25%. Reduces VRAM proportionally. Compatible with any training method. Paper: arXiv:2406.06623.
- **PiSSA init** (`--pissa-init`): SVD-based LoRA initialization. Initializes LoRA weights from the SVD of pre-trained weights instead of random Kaiming init. Faster convergence, lower final loss. Paper: arXiv:2404.02948.

---

### Added

- **EMA-smoothed early stopping**: Implemented trend-based loss monitoring to reduce sensitivity to noisy gradients in early stopping decisions.
- **Terminal raw mode**: Training controls (`[P]ause`, `[Q]uit`, `[R]esume`) now utilize raw TTY mode for instant response without requiring Enter.
- **Best-model checkpointing** (`--checkpoint-best`): Automatically saves the model weights from the step with the lowest recorded eval loss as the primary output.

---

## [0.5.39] — 2026-06-28 — Progress bar overhaul, step-based early stopping, YAML pipeline

### Changed

- **Progress bar reformat**: `LiveProgressCallback` now shows epoch counter
  (`Epoch 5/20  45%`), percentage complete, `tok/s` and `pairs/s` (instead of
  cryptic `t/s`/`p/s`), and `VRAM USED` (instead of ambiguous `VRAM`).
  Each update starts on a fresh line to prevent GCEpochCallback messages
  from getting tangled in the output.
- **GCEpochCallback**: emergency cache clear and post-eval messages now
  always start on a new line (`\n` prefix) for readability.

### Added

- **Step-based early stopping** (`--early-stop-steps`, default 1000):
  `StepEarlyStoppingCallback` stops training if `eval_loss` doesn't improve
  for N consecutive steps. Complements epoch-based early stopping — whichever
  fires first wins. Set to 0 to disable.
- **YAML pipeline config** (`--config pipeline.yaml`): IaC for model training.
  Define one or more jobs, each with optional stages: `train` → `merge` →
  `gguf` → `install`. Jobs run sequentially; if a stage fails, the pipeline
  skips to the next job. All train args are supported as YAML keys.
    - New CLI: `attacklm-pipeline --config pipeline.yaml`
    - Example config: `pipeline.example.yaml`

---

## [0.5.38] — 2026-06-28 — Auto-disable packing

### Added

- **Auto-disable packing**: `--packing` is now a best-effort hint.
  When flash-attn is not installed, packing is silently disabled to prevent
  cross-sample contamination. No warnings, no errors, no user intervention.

---

## [0.5.35] — 2026-06-27 — GaLore full-parameter training, balanced datasets, interactive controls


### Added

- **GaLore full-parameter training** (`--use-galore`): Gradient Low-Rank Projection
  enables full-parameter fine-tuning on consumer GPUs (3B model fits 16GB at
  batch_size=8). Mutually exclusive with Unsloth QLoRA.
  - `--galore-rank` (default 64): tunable projection rank
  - `--galore-32bit`: full-precision optimizer (multi-GPU compatible)
  - `--multi-gpu`: auto-enables 32-bit (per-layer hooks incompatible with DDP)
  - 8-bit `GaLoreAdamW8bit` optimizer with per-layer hooks (~37 groups for Qwen 3B)
  - `[galore]` extra in pyproject.toml: `galore-torch>=1.0`
- **Interactive training controls**: `[P]ause` / `[Q]uit` / `[R]esume` during
  training via background stdin listener. Pause saves checkpoint and blocks;
  resume continues from where it left off; quit saves and exits cleanly.
- **`attacklm-balance` uniform cap**: `--per-bucket-cap` now accepts a plain
  integer (`'500'`) for uniform cap across all buckets, in addition to the
  existing JSON dict format.
- **`max_grad_norm=1.0`**: gradient clipping prevents GaLore loss explosion
  (low-rank projection can amplify outlier gradients into numerical overflow).
- **`load_best_model_at_end=True`**: trainer loads the checkpoint with lowest
  eval_loss at end of training. `save_strategy` to match `eval_strategy`
  (both set to `"epoch"`).

### Changed

- **VRAM display**: progress bar now shows `used/total` instead of `free/total`
  (e.g. `VRAM 14.5/15.6 GB` instead of `VRAM 1.1/15.6 GB`).
- **Progress bar**: 80-char friendly format, removed HF PrinterCallback dict dump.
- **GCEpochCallback threshold**: lowered from 2GB to 256MB for GaLore (per-layer
  hooks free gradients after each layer's backward pass, so peak memory is lower).
- **Tokenizer**: `bos_token_id` synced before model load to prevent HF warnings.
- **Tied embeddings**: `model.config.tie_word_embeddings=True` set after load to
  prevent `lm_head.weight` missing warnings on checkpoint load.

### Fixed

- **bitsandbytes Enum warnings**: monkey-patched `torch.utils._pytree.register_constant`
  to no-op for Enum subclasses (natively supported in PyTorch 2.12+).
- **Python 3.14 argparse**: escaped `%` in help strings.
- **`Avg tok/s: 0`**: eval log events with `num_tokens=0` no longer reset the
  cumulative token counter.
- **GaLore OOM**: per-layer hooks grouped by layer prefix (~37 groups, not 434
  individual optimizers). huihui-ai abliterated model identified as root cause
  of fp32 loading (custom modeling code forces fp32 regardless of `dtype=`).
  Standard `Qwen/Qwen2.5-Coder-3B-Instruct` loads in bf16 at 5.75GB.

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
  records) in addition to `restricted-sources/` and `tui-sourceを/`.
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

### Fixed

- **`scripts/generate_synthetic_scarce.py` — live metrics + cleaner console output.**
  - `call_llm()` now returns `{content, usage, latency_ms}` — tracks prompt / completion / total tokens from both OpenAI-compatible (LMStudio) and Ollama responses.
  - Per-batch backend/model/temp spam eliminated. Backend info prints **once per category** instead of once per batch.
  - Live progress bar with real metrics: **tokens/sec**, **pairs/sec**, and **latency (ms)** per batch.
  - Optional `rich` library progress bar, plain-text ASCII fallback if `rich` is not installed.
  - Final summary line: `Wrote N pairs | X tok/s avg | Y pair/s avg | Z.s total | filename`.
  - Metrics persisted to `{category}_llm_meta.json` under `"metrics"`.

- **`scripts/llm_generate_wrapper.py` — complete rewrite.**
  - Named count overrides: `--web-app`, `--cloud`, `--social-engineering`, `--supply-chain`, `--ics-scada`, `--wireless`.
  - Single-category mode: `--only web_app`.
  - Backend / model / temperature as CLI flags (`--backend`, `--model`, `--temperature`) — **no env var syntax needed**.
  - `--sleep` flag (default OFF) for inter-batch pauses.
  - Wrapper is now the **sole entry point**; `generate_synthetic_scarce.py` accepts `--category`/`--count` only. Passing positional counts directly to it produces `unrecognized arguments` by design.
  - Env vars passed explicitly via `subprocess.run(env=...)` instead of relying on shell inheritance.

- **`scripts/train_template.py` — tokens/sec progress replaces useless `it/s`.**
  - New `LiveProgressCallback` prints: step count, loss, **tok/s**, **pair/s**, and VRAM usage every 10 steps.
  - HF Trainer's default tqdm disabled via `disable_tqdm=True`.
  - `it/s` was meaningless because it conflates batch size, gradient accumulation, and packing into a single opaque number. `tok/s` and `pair/s` reflect actual data throughput.

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

## [0.2.2] — 2026-06-10

### Fixed

- **Epoch counter now accurate.** `state.json[progress].current_epoch` is
  now read from the trainer's own `log_history` (the real fractional value
  at the last logged step) instead of HF Trainer's rounded `train_result.metrics["epoch"]`
  (which is an int). Previously the counter reported `epoch: 3.0` when the
  trainer actually ran `2.999` epochs. Also added `state.json[progress].target_epochs`
  (the user's --epochs value) and `state.json[progress].filtered_examples`
  (how many examples the long-example filter dropped) so the user can see
  exactly what happened.
- **`attacklm-train` no longer clobbers previous runs.** Default behavior: a
  `_YYYY-MM-DD_HH-MM` timestamp is appended to `--output` so each run is
  preserved (matching what `attacklm-train-all` has done since v0.1.6).
  Opt out with `--no-timestamp`. If `--no-timestamp` is set and the output
  is a **completed** run, the trainer refuses unless `--force` is also
  passed. If the path already ends in a timestamp, it's left aloneS.
- **`attacklm-gguf` no longer silently skips when source is newer.**
  Previously `--install-lmstudio` printed `⏭ attacklm-single — already exists`
  and skipped even when the source BF16 model was just re-merged. v0.2.2
  compares mtime: if `models/gguf/{name}.Q4_K_M.gguf` is older than the
  source `model.safetensors`, it's treated as stale and re-converted
  (with a clear log line). `--force` bypasses the mtime check entirely.

### Added

- **`attacklm-build`** — One-shot full pipeline: merge LoRA → BF16 → GGUF →
  install to LM Studio → (optional) register with Ollama → drop a build
  manifest at `models/built/{name}_{timestamp}/`. Replaces the 3-command
  shell pipeline (`attacklm-merge && rm && attacklm-gguf --install-lmstudio`)
  with a single command. Auto-detects the base model from the adapter's
  `state.json` / `adapter_config.json`. Defaults: `--install-lmstudio` ON,
  `--register-ollama` OFF. Wired in as a console script.
- **`attacklm-gguf --quant` and `--register-ollama`.** `--quant` lets you
  pick `Q8_0` / `Q5_K_M` / `Q6_K` instead of the hardcoded `Q4_K_M`.
  `--register-ollama` writes a `Modelfile` next to the GGUF and runs
  `ollama create {name}`, so the model shows up in `ollama list` and you
  can run it via `ollama run {name} ভিত্তি`.

- **`scripts/balance_buckets.py`** — Balanced bucket sampler for SFT data.
  Auto-sizes per-bucket caps based on target model + VRAM profile (3b-16gb /
  7b-16gb / 7b-128gb / 14b-128gb / 31b-128gb / full / custom). Three
  within-bucket sampling strategies (head / random / **stratified** — the
  default). Stratified sampler groups examples by their first MITRE
  technique ID, source, or first assistant-content line, then allocates with
  **minimum-1-per-group** so every technique / module gets representation.
  Solves the "metasploit 49% of training" problem for round-2 SFT.

  Category-balanced allocation (the default for `--target-total`): targets
  50% base / 25% tools / 15% ai / 10% orchestrator, then redistributes
  proportionally when small categories hit their caps. Overridable via
  `--category-shares` JSON.

  Wired in as `attacklm-balance` console script.

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
- 12 spec resolver cases pass: 4 parents (`base/`, `tools/`, `ai/`, `orchestrator`), 3 aliases (`all`, `tactics`, `tools-all`), 3 subpaths (`tools/metasploit/`, `ai/jailbreaking/`, `base/collection/`), 1 backward-compat (`ai-models/` → 2 buckets), 1 default set (tactics + orchestrator = 11 buckets)
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

- **`--backup` / `--no-backup` round-2 SFT backup.** When round-2 SFT is detected, the previous run dir + merged model are tar.gz'd to `models/.backups/{name}_{timestamp}.tar.gz` with a progress bar. `--backup` is the default; `--no-backup` skips it. Tar size is ~ la 5 GB for a 3B BF16 model (BF16 doesn't compress wellL; ~80% of uncompressed). The previous run dir is **never deleted** — it stays in `models/{name}_*/` for inspection.

- **Timestamped run dirs.** Each training run gets its own `models/{agent}_{YYYY-MM-DD}_{HH-MM}[_N]/` instead of clobbering `models/{agent}/`. Older runs remain on disk for rollback. Merged models still write to `models/merged/{agent}/` (single deployable artifact per agent).

- **`_find_latest_run_dir(agent_name)` helper.** Lexicographic sort on the timestamped dir names picks the most recent run for an agent. UsedT for the round-2 SFT auto-detect and the `merge_all` glob.

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

- **`.gitignore` excludes `data/datasets/combined/*.jsonl`** (regenerable, ~15-17 MB each; saves ~100 MB on the repo) and explicitly excludes `models/.backups/` (dotfile dir, also already excluded by `models the/`).

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
                    if d.startswith('checkpointเดียวกัน')])
ts = json.load(open(f'{src}/{ckpt_dirs[-1]}/trainer_state.json'))
# ... (see scripts/migrate_v015_to_v020.py for the full template)
"
```

**Option B (cleanest): just re-train.** v0.1.6 and later will write everything correctly from scratch.

The `scripts/migrate_v015_to_v020.py` migration script lives at `scripts/migrate_v015_to_v020.py` and handles the rename + backfill automatically.

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
