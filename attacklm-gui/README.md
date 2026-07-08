# AttackLM GUI

Terminal-based GUI wrapper for the [AttackLM](https://github.com/Veedubin/AttackLM) CLI training tools. Built with [Textual](https://textual.textualize.io/).

## Why?

AttackLM has a unified CLI (`attacklm train`, `attacklm init`, etc.) with 8 subcommands. `attacklm train` alone has 40+ parameters. Remembering all the flags is painful. The GUI provides:

- **Form-based parameter input** with tabs, presets, and validation
- **Live training monitoring** with loss charts, VRAM gauges, and log streaming
- **One-click commands** for extract, balance, infer, merge, build, and more
- **Works everywhere** — terminal-only, no X11 required (WSL, SSH, headless)

The unified CLI (`attacklm <subcommand>`) still works as before. The GUI is a thin wrapper that constructs and runs CLI commands.

## Installation

```bash
pip install attacklm-gui
```

Or from source:

```bash
git clone https://github.com/Veedubin/AttackLM.git
cd AttackLM-Models/attacklm-gui
pip install -e .
```

Requires `attacklm` (the CLI package) and `textual>=2.0`.

## Usage

```bash
attacklm-gui
```

### Main Menu

```
┌──────────────────────────────────────────┐
│         AttackLM GUI v0.1.0              │
│                                          │
│  🏋️  Train Model                         │
│  🚀  Init Dataset                        │
│  ⚖️  Balance Dataset                     │
│  📦  Build & Install                     │
│  🧠  Run Inference                       │
│  📊  Evaluate                            │
│  🧭  Steer Model                         │
│  📏  Benchmark                           │
│  🔍  Audit (MIA / Extraction)            │
└──────────────────────────────────────────┘
```

Every menu button has a **tooltip** — hover with the mouse (or focus with the keyboard and read the footer) to see a one-sentence description of what the screen does.

### Training Form

The training form organizes 40+ parameters into tabs:

| Tab | Parameters |
|-----|-----------|
| **Basic** | dataset, output, base model, epochs, batch size, max length |
| **LoRA** | rank, alpha, dropout, DoRA, RSLoRA, target modules, PiSSA |
| **GaLore** | Q-GaLore, rank, 32-bit, Spectrum layer freezing |
| **Advanced** | eval split, early stopping, save steps, gradient accum, optimizer, packing, live LR |
| **Hardware** | precision (BF16/FP16/FP32), multi-GPU, MoE safe, auto-tune |

**Presets** save and load complete configurations. Built-in presets:
- `3B Q-GaLore Spectrum` — full-parameter training on 16GB GPU
- `3B Q-GaLore Rank 128` — higher quality variant
- `3B LoRA Default` — standard QLoRA
- `7B Q-GaLore` — for 24GB GPUs
- `7B QLoRA Default` — standard QLoRA for 7B

### Live Training Monitor

```
┌─────────────────────────────────────────────────────┐
│  Training: attacklm-3b-qgalore-spectrum             │
│  Epoch 12/30  |  Step 1896  |  Elapsed: 1h 23m     │
├──────────────────────┬──────────────────────────────┤
│  Loss: 1.1143 ▂▃▅▆▇  │  VRAM: 5.9/15.6 GB (62%)   │
│  Eval Loss: 1.176    │  alloc: 5.9  cache: 4.8     │
│  Trend: ↓ -0.0116    │  ████████░░░░               │
├──────────────────────┴──────────────────────────────┤
│  Log Output (scrolling)                             │
│  Epoch 12/20  58% | loss 1.1143 | 2,844 tok/s ...  │
├─────────────────────────────────────────────────────┤
│  [P]ause  [S]top at checkpoint  [Q]uit              │
└─────────────────────────────────────────────────────┘
```

Controls:
- **Pause/Resume** — SIGSTOP/SIGCONT the training process
- **Stop at Checkpoint** — graceful stop after current eval
- **Quit** — kill training immediately

### Other Commands

| Screen | What it does |
|--------|-------------|
| **Init** | Clone repos, extract data, organize buckets |
| **Balance** | Balance a dataset with cap size |
| **Inference** | Run a trained model with a prompt |
| **Build** | Merge → GGUF → install to LM Studio |
| **Eval** | Retention eval, collect-ref, score, compare, golden |
| **Pipeline** | Run a YAML training pipeline |
| **Audit (🔍)** | Run an inversion-attack audit on a trained model. Two tabs: **Extraction** (Carlini 2021 prefix-completion probing) and **MIA** (membership inference attack). Delegates to `attacklm audit`, which calls `attacklm-dataset/scripts/inversion_audit.py` with the new `--attack {extraction, mia, all}` and `--mia-method {reference, zlib, per_token, lira, all}` flags (v0.4.0+). See `attacklm-dataset/docs/ATTACK_TAXONOMY.md` for the attack taxonomy. |

### Tooltips

Every input field, select, and button across the TUI has a **tooltip** — hover with the mouse to see a one-sentence explanation of what it does. Tooltip text is centralized in `src/attacklm_gui/widgets/tooltips.py` so it is reviewable in one place. The `DEFAULT_CSS` rule styles tooltips with a surface background, accent border, and 60-character max width. Hover delay is 0.5 seconds (`App.tooltip_delay`).

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `q` | Quit |
| `m` | Return to main menu |
| `Tab` | Next field |
| `Enter` | Activate button |
| `Esc` | Go back |

## Architecture

```
attacklm-gui/
├── src/attacklm_gui/
│   ├── app.py              # Textual App entry point (DEFAULT_CSS for Tooltip styling)
│   ├── cli.py              # CLI entry point
│   ├── runner.py           # Subprocess manager + output parser
│   ├── presets.py          # Save/load training presets (slugify-safe filenames)
│   ├── screens/
│   │   ├── main_menu.py    # Command launcher (9 buttons including Audit)
│   │   ├── train_form.py   # Training parameter form (with tooltips)
│   │   ├── train_live.py   # Live training monitor
│   │   ├── command_forms.py # Init, Balance, Infer, Build, Eval, Pipeline screens
│   │   └── audit.py        # Inversion-attack audit (Extraction / MIA tabs)
│   └── widgets/
│       ├── __init__.py
│       └── tooltips.py     # Centralized tooltip text (~30 entries)
└── tests/
    ├── test_gui.py         # Existing tests + TestTooltipsRetrofit
    └── test_audit.py       # New: 7 tests for the audit screen and tooltips
```

The GUI never touches AttackLM's internal logic. It constructs CLI command strings and runs them via `asyncio.create_subprocess_exec`. All training, extraction, and inference logic stays in `scripts/*.py`. The audit screen delegates to the `attacklm audit` subcommand, which is a thin wrapper around `attacklm-dataset/scripts/inversion_audit.py` — the TUI does not depend on `attacklm-dataset` directly.

## Requirements

- Python 3.10+
- Linux (any flavor) or WSL
- Terminal with 256-color support
- AttackLM installed (`pip install attacklm[all]`)

No X11, no GPU, no browser required. Works over SSH.
