# Changelog — attacklm-gui

All notable changes to the AttackLM GUI (`attacklm-gui`) package.

## [Unreleased] — 2026-07-08 — Audit screen + tooltips

This is NOT yet a tagged release. The version is `0.1.0`; it will be
bumped to `0.2.0` when this branch is merged and tagged. The TUI
version is decoupled from the main `attacklm` package version
(currently `0.11.1-1`).

### Added
- **`screens/audit.py`** — new Audit screen with 2 tabs:
  - **Extraction tab** — Carlini 2021 prefix-completion extraction. Fields: model path, dataset root, source filter, top-K, max new tokens, temperature, max records.
  - **MIA tab** — Membership inference attack. Fields: model path, dataset root, source filter, MIA method (Select: reference / zlib / per_token / lira), threshold mode (Select: percentile / median / holdout_file), percentile, max records.
  - Output panel: live `RichLog` showing the `attacklm audit` subprocess output. Status bar at the bottom.
- **`widgets/tooltips.py`** — centralized tooltip text for all TUI widgets. ~30 entries covering the main menu, audit screen, and train form high-traffic fields. `attach_tooltip(widget, key)` helper for one-line attachment.
- **`attacklm audit` subcommand** in the main `attacklm` CLI (`src/attacklm/cli.py`) — bridges the TUI/CLI to `attacklm-dataset/scripts/inversion_audit.py`. Forwards all audit flags: `--attack`, `--mia-method`, `--mia-threshold-mode`, `--mia-percentile`, `--model`, `--dataset-root`, `--source-filter`, `--top-k`, `--max-new-tokens`, `--temperature`, `--max-records`, `--dry-run`.
- **Tooltips retrofit** on existing TUI screens:
  - All 9 main menu buttons (`btn-train`, `btn-init`, `btn-balance`, `btn-build`, `btn-infer`, `btn-eval`, `btn-steer`, `btn-bench`, `btn-audit`).
  - High-traffic train form inputs: `epochs`, `batch_size`, `lora_r`, `lora_alpha`, `galore_rank`, `max_length`, `spectrum`, `use_qgalore`, `use_dora`.
  - All command form Back buttons.
- **`btn-audit`** in the main menu (between "Evaluate" and "Steer") — opens the new Audit screen.
- **`DEFAULT_CSS`** in `app.py` for Tooltip styling (background, border, padding, max-width=60 chars).
- **`App.tooltip_delay = 0.5s`** — hover delay before tooltips show.

### Drive-by bug fixes (not in the MIA Track 2 spec)
- **`presets.py`** — `FP8 (H100/Blackwell)` was creating an invalid filename (`fp8_(h100/blackwell).json`) on Linux because the slash survives the previous `lower().replace(' ', '_')` slugify. Now uses a proper regex slugify via a new `_slugify()` helper. Fixes a real bug that affected all users on first run.
- **`screens/train_form.py`** — `Select(value=3)` for `deepspeed_stage` was silently broken under Textual 8.x (the value is silently overwritten with `Select.NULL` when options are tuples). Dropped the explicit value; the downstream code (`values.get("deepspeed_stage", 3)`) already defaults to 3.

### Tests
- 7 new tests in `tests/test_audit.py`:
  - `TestAuditScreen`: `test_audit_screen_mounts`, `test_audit_extraction_has_model_input`, `test_audit_mia_method_select_exists`.
  - `TestTooltips`: `test_audit_screen_inputs_have_tooltips`, `test_main_menu_buttons_have_tooltips`, `test_tooltips_dict_has_all_required_keys`, `test_attach_tooltip_no_op_for_missing_key`.
- 2 new structural tests in `tests/test_gui.py::TestTooltipsRetrofit`:
  - `test_train_form_tooltip_keys_are_defined` (the train form has pre-existing Textual 8.x mount bugs in its `Select(value=N)` widgets; we test the constants and the structural retrofit instead of mounting).
  - `test_train_form_calls_attach_tooltip_in_on_mount` (verifies the train form's on_mount has tooltip wiring for all 9 high-traffic fields).

### Test state
- 26/26 tests pass (7 audit + 19 gui).
- 1 pre-existing ruff issue in `train_live.py` (F841 unused `trend_class`) — NOT in M2 scope, left alone.

### Reference
- Per architect plan approved 2026-07-08 (deepseek-v4-pro:cloud).
- Memory: `d4657ab9-53d4-49a4-b4bd-3de7816b3868` (architectural decision), `88c25f43-5d1d-4756-804e-4b0ad6b1dc19` (MIA research), `fb19c94e-dd7b-4e02-821c-3fb32eb1abac` (session summary).
- Branch: `wip/inversion-audit-track2-2026-07-08` in both repos. Nothing pushed. Rollback is a single `git reset --hard origin/main && git branch -D wip/...` per repo.

---

## [0.1.0] — initial release (prior to this session)

Terminal GUI wrapper for the `attacklm` CLI. Built with Textual. 6 main menu buttons (Train / Init / Balance / Build / Infer / Eval), live training monitor, presets, and command forms for the simpler subcommands.
