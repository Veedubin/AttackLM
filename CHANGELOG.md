# Changelog

All notable changes to AttackLM are documented in this file. Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.0]: https://github.com/Veedubin/AttackLM/releases/tag/v0.2.0
[0.1.5]: https://github.com/Veedubin/AttackLM/releases/tag/v0.1.5
[0.1.4]: https://github.com/Veedubin/AttackLM/releases/tag/v0.1.4
[0.1.3]: https://github.com/Veedubin/AttackLM/releases/tag/v0.1.3
[0.1.2]: https://github.com/Veedubin/AttackLM/releases/tag/v0.1.2
[0.1.1]: https://github.com/Veedubin/AttackLM/releases/tag/v0.1.1
[0.1.0]: https://github.com/Veedubin/AttackLM/releases/tag/v0.1.0
