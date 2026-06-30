# AttackLM GUI

Terminal-based GUI wrapper for the [AttackLM](https://github.com/Veedubin/AttackLM) CLI training tools. Built with [Textual](https://textual.textualize.io/).

## Why?

AttackLM has 21 CLI commands and `attacklm-train` alone has 40+ parameters. Remembering all the flags is painful. The GUI provides:

- **Form-based parameter input** with tabs, presets, and validation
- **Live training monitoring** with loss charts, VRAM gauges, and log streaming
- **One-click commands** for extract, balance, infer, merge, build, and more
- **Works everywhere** — terminal-only, no X11 required (WSL, SSH, headless)

The CLI still works exactly as before. The GUI is a thin wrapper that constructs and runs CLI commands.

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
┌──────────────────────────────────────┐
│         AttackLM GUI v0.1.0          │
│                                      │
│  🏋️  Train Model                     │
│  📊  Extract Data                    │
│  ⚖️  Balance Dataset                 │
│  🧠  Run Inference                   │
│  🔗  Merge Adapter                   │
│  📦  Build & Install                 │
│  🔧  Pipeline                        │
│  🚀  Init Dataset                    │
└──────────────────────────────────────┘
```

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
| **Extract** | Run data extractors on upstream sources |
| **Balance** | Balance a dataset with cap size |
| **Inference** | Run a trained model with a prompt |
| **Merge** | Merge LoRA adapter into base model |
| **Build** | Merge → GGUF → install to LM Studio |
| **Pipeline** | Run a YAML training pipeline |
| **Init** | One-click dataset initialization |

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
│   ├── app.py              # Textual App entry point
│   ├── cli.py              # CLI entry point
│   ├── runner.py           # Subprocess manager + output parser
│   ├── presets.py          # Save/load training presets
│   ├── screens/
│   │   ├── main_menu.py    # Command launcher
│   │   ├── train_form.py   # Training parameter form
│   │   ├── train_live.py   # Live training monitor
│   │   └── command_forms.py # Other command screens
│   └── widgets/
│       └── __init__.py
└── tests/
    └── test_gui.py
```

The GUI never touches AttackLM's internal logic. It constructs CLI command strings and runs them via `asyncio.create_subprocess_exec`. All training, extraction, and inference logic stays in `scripts/*.py`.

## Requirements

- Python 3.10+
- Linux (any flavor) or WSL
- Terminal with 256-color support
- AttackLM installed (`pip install attacklm[all]`)

No X11, no GPU, no browser required. Works over SSH.
