"""Training parameter form screen."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Label,
    Select,
    Static,
    TabbedContent,
    TabPane,
)


class TrainFormScreen(Screen):
    """Form for configuring and launching training runs."""

    CSS = """
    TrainFormScreen {
        align: center middle;
    }

    #form-container {
        width: 70;
        height: 90%;
        border: solid $accent;
        background: $surface;
    }

    #form-title {
        text-align: center;
        text-style: bold;
        padding: 1 0;
        background: $accent;
        color: $text;
    }

    TabbedContent {
        height: 1fr;
    }

    TabPane {
        padding: 1 2;
    }

    .form-row {
        height: 3;
        margin: 0 0;
    }

    .form-label {
        width: 22;
        padding: 0 1;
        text-align: right;
    }

    .form-input {
        width: 40;
    }

    .form-help {
        color: $text-muted;
        text-style: italic;
    }

    .section-header {
        text-style: bold;
        color: $accent;
        padding: 1 0 0 0;
    }

    #button-row {
        dock: bottom;
        height: 3;
        align: center middle;
    }

    #button-row Button {
        margin: 0 1;
    }

    #preset-row {
        dock: top;
        height: 3;
        align: center middle;
        background: $surface-darken-1;
    }

    #preset-row Button {
        margin: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Container(id="form-container"):
            yield Label("Training Configuration", id="form-title")

            with Horizontal(id="preset-row"):
                yield Label("Preset:")
                yield Select(
                    [
                        (p, p)
                        for p in [
                            "3B Q-GaLore Spectrum",
                            "3B Q-GaLore Rank 128",
                            "3B LoRA Default",
                            "7B Q-GaLore",
                            "7B QLoRA Default",
                            "DeepSpeed 40B+",
                        ]
                    ],
                    id="preset-select",
                    value="3B Q-GaLore Spectrum",
                )
                yield Button("Load", id="btn-load-preset", variant="primary")
                yield Button("Save", id="btn-save-preset")

            with TabbedContent():
                with TabPane("Basic", id="tab-basic"):
                    yield self._row(
                        "Dataset Path",
                        "dataset",
                        placeholder="data/datasets/balanced/balanced_1000cap.jsonl",
                    )
                    yield self._row(
                        "Output Dir", "output", placeholder="models/my-model"
                    )
                    yield self._row(
                        "Base Model",
                        "base_model",
                        placeholder="Qwen/Qwen2.5-Coder-3B-Instruct",
                    )
                    yield self._row("Epochs", "epochs", placeholder="20", value="20")
                    yield self._row(
                        "Batch Size", "batch_size", placeholder="1", value="1"
                    )
                    yield self._row(
                        "Max Length", "max_length", placeholder="12000", value="12000"
                    )
                    yield self._checkbox_row("Dry Run", "dry_run")
                    yield self._checkbox_row("Enable Training", "train", default=True)

                with TabPane("LoRA", id="tab-lora"):
                    yield self._row(
                        "LoRA Rank (r)", "lora_r", placeholder="16", value="16"
                    )
                    yield self._row(
                        "LoRA Alpha", "lora_alpha", placeholder="32", value="32"
                    )
                    yield self._row(
                        "LoRA Dropout", "lora_dropout", placeholder="0.05", value="0.05"
                    )
                    yield self._checkbox_row("Use DoRA", "use_dora")
                    yield self._checkbox_row("Use RSLoRA", "use_rslora", default=True)
                    yield self._row(
                        "Target Modules",
                        "target_modules",
                        placeholder="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                    )
                    yield self._checkbox_row("LoftQ Init", "loftq_init")
                    yield self._checkbox_row("PiSSA Init", "pissa_init")

                with TabPane("GaLore", id="tab-galore"):
                    yield self._checkbox_row("Use GaLore", "use_galore")
                    yield self._checkbox_row(
                        "Use Q-GaLore", "use_qgalore", default=True
                    )
                    yield self._row(
                        "GaLore Rank", "galore_rank", placeholder="64", value="64"
                    )
                    yield self._checkbox_row("GaLore 32-bit", "galore_32bit")
                    yield self._row(
                        "Spectrum (0-1)", "spectrum", placeholder="0.5", value="0.5"
                    )

                with TabPane("Advanced", id="tab-advanced"):
                    yield self._row(
                        "Eval Split", "eval_split", placeholder="0.1", value="0.1"
                    )
                    yield self._row(
                        "Early Stop Steps",
                        "early_stop_steps",
                        placeholder="5",
                        value="5",
                    )
                    yield self._row(
                        "Early Stop Patience",
                        "early_stopping_patience",
                        placeholder="3",
                        value="3",
                    )
                    yield self._row(
                        "Save Steps", "save_steps", placeholder="200", value="200"
                    )
                    yield self._row(
                        "Gradient Accum Steps",
                        "gradient_accumulation_steps",
                        placeholder="1",
                        value="1",
                    )
                    yield self._row(
                        "Optimizer",
                        "optim",
                        placeholder="paged_adamw_8bit",
                        value="paged_adamw_8bit",
                    )
                    yield self._checkbox_row("Packing", "packing", default=True)
                    yield self._checkbox_row("Live LR", "live_lr", default=True)
                    yield self._row(
                        "Max Steps (-1=epochs)",
                        "max_steps",
                        placeholder="-1",
                        value="-1",
                    )

                with TabPane("Hardware", id="tab-hardware"):
                    yield Label("Precision:", classes="form-label")
                    yield Select(
                        [("BF16 (auto)", "bf16"), ("FP16", "fp16"), ("FP32", "fp32")],
                        id="precision",
                        value="bf16",
                    )
                    yield self._checkbox_row("Multi-GPU (DDP)", "multi_gpu")
                    yield self._checkbox_row("MoE Safe Target", "moe_safe_target")
                    yield self._checkbox_row("Auto-Tune", "auto_tune")
                    yield self._checkbox_row("No Timestamp", "no_timestamp")

                    yield Static(
                        "DeepSpeed ZeRO Optimization", classes="section-header"
                    )
                    yield self._checkbox_row("Enable DeepSpeed ZeRO", "use_deepspeed")
                    yield Horizontal(
                        Label("ZeRO Stage:", classes="form-label"),
                        Select(
                            [
                                (1, "ZeRO-1 (optimizer states)"),
                                (2, "ZeRO-2 (optimizer + gradients)"),
                                (3, "ZeRO-3 (params + grads + optimizer)"),
                            ],
                            id="deepspeed_stage",
                            value=3,
                        ),
                        classes="form-row",
                    )
                    yield self._checkbox_row(
                        "CPU Offload (use system RAM)",
                        "deepspeed_offload",
                        default=True,
                    )

                    yield Static("PyTorch Compilation", classes="section-header")
                    yield self._checkbox_row("Enable torch.compile", "compile")
                    yield Horizontal(
                        Label("Compile Mode:", classes="form-label"),
                        Select(
                            [
                                ("default", "default"),
                                ("reduce-overhead", "reduce-overhead"),
                                ("max-autotune", "max-autotune"),
                            ],
                            id="compile_mode",
                            value="reduce-overhead",
                        ),
                        classes="form-row",
                    )

                    yield Static("LOMO Optimizer", classes="section-header")
                    yield self._checkbox_row(
                        "Enable LOMO (full-param, low VRAM)", "use_lomo"
                    )

            with Horizontal(id="button-row"):
                yield Button("Dry Run", id="btn-dry-run", variant="default")
                yield Button("Start Training", id="btn-start", variant="primary")
                yield Button("Back", id="btn-back", variant="default")

    def _row(
        self, label: str, input_id: str, placeholder: str = "", value: str = ""
    ) -> Horizontal:
        """Create a labeled input row."""
        return Horizontal(
            Label(label, classes="form-label"),
            Input(
                placeholder=placeholder, value=value, id=input_id, classes="form-input"
            ),
            classes="form-row",
        )

    def _checkbox_row(
        self, label: str, checkbox_id: str, default: bool = False
    ) -> Horizontal:
        """Create a labeled checkbox row."""
        return Horizontal(
            Label(label, classes="form-label"),
            Checkbox(label="", value=default, id=checkbox_id),
            classes="form-row",
        )

    def _get_form_values(self) -> dict:
        """Collect all form values into a dict."""
        values = {}
        for widget in self.query("Input"):
            if widget.id:
                values[widget.id] = widget.value
        for widget in self.query("Checkbox"):
            if widget.id:
                values[widget.id] = widget.value
        for select_id in ("precision", "deepspeed_stage", "compile_mode"):
            select = self.query_one(f"#{select_id}", Select)
            if select.value is not None:
                values[select_id] = select.value
        return values

    def _build_command(self, values: dict) -> list[str]:
        """Build the attacklm-train command from form values."""
        cmd = ["attacklm", "train"]

        # Required
        if values.get("dataset"):
            cmd.extend(["--dataset", values["dataset"]])
        if values.get("output"):
            cmd.extend(["--output", values["output"]])

        # Basic
        if values.get("base_model"):
            cmd.extend(["--base-model", values["base_model"]])
        if values.get("epochs"):
            cmd.extend(["--epochs", values["epochs"]])
        if values.get("batch_size"):
            cmd.extend(["--batch-size", values["batch_size"]])
        if values.get("max_length"):
            cmd.extend(["--max-length", values["max_length"]])

        # LoRA
        if values.get("lora_r"):
            cmd.extend(["--lora-r", values["lora_r"]])
        if values.get("lora_alpha"):
            cmd.extend(["--lora-alpha", values["lora_alpha"]])
        if values.get("lora_dropout"):
            cmd.extend(["--lora-dropout", values["lora_dropout"]])
        if values.get("use_dora"):
            cmd.append("--use-dora")
        if not values.get("use_rslora", True):
            cmd.append("--no-use-rslora")
        if values.get("target_modules"):
            cmd.extend(["--target-modules", values["target_modules"]])
        if values.get("loftq_init"):
            cmd.append("--loftq-init")
        if values.get("pissa_init"):
            cmd.append("--pissa-init")

        # GaLore
        if values.get("use_galore"):
            cmd.append("--use-galore")
        if values.get("use_qgalore"):
            cmd.append("--use-qgalore")
        if values.get("galore_rank"):
            cmd.extend(["--galore-rank", values["galore_rank"]])
        if values.get("galore_32bit"):
            cmd.append("--galore-32bit")
        if values.get("spectrum"):
            cmd.extend(["--spectrum", values["spectrum"]])

        # Advanced
        if values.get("eval_split"):
            cmd.extend(["--eval-split", values["eval_split"]])
        if values.get("early_stop_steps"):
            cmd.extend(["--early-stop-steps", values["early_stop_steps"]])
        if values.get("early_stopping_patience"):
            cmd.extend(["--early-stopping-patience", values["early_stopping_patience"]])
        if values.get("save_steps"):
            cmd.extend(["--save-steps", values["save_steps"]])
        if values.get("gradient_accumulation_steps"):
            cmd.extend(
                ["--gradient-accumulation-steps", values["gradient_accumulation_steps"]]
            )
        if values.get("optim"):
            cmd.extend(["--optim", values["optim"]])
        if values.get("packing"):
            cmd.append("--packing")
        if values.get("live_lr"):
            cmd.append("--live-lr")
        if values.get("max_steps") and values["max_steps"] != "-1":
            cmd.extend(["--max-steps", values["max_steps"]])

        # Hardware
        precision = values.get("precision", "bf16")
        if precision == "bf16":
            cmd.append("--bf16")
        elif precision == "fp16":
            cmd.append("--fp16")
        elif precision == "fp32":
            cmd.append("--fp32")
        if values.get("multi_gpu"):
            cmd.append("--multi-gpu")
        if values.get("moe_safe_target"):
            cmd.append("--moe-safe-target")
        if values.get("auto_tune"):
            cmd.append("--auto-tune")
        if values.get("no_timestamp"):
            cmd.append("--no-timestamp")

        # DeepSpeed
        if values.get("use_deepspeed"):
            cmd.append("--use-deepspeed")
            cmd.extend(["--deepspeed-stage", str(values.get("deepspeed_stage", 3))])
            if not values.get("deepspeed_offload", True):
                cmd.append("--no-deepspeed-offload")

        # torch.compile
        if values.get("compile"):
            cmd.append("--compile")
            cmd.extend(
                ["--compile-mode", str(values.get("compile_mode", "reduce-overhead"))]
            )

        # LOMO
        if values.get("use_lomo"):
            cmd.append("--use-lomo")

        # Training mode
        if values.get("train"):
            cmd.append("--train")
        elif values.get("dry_run"):
            pass  # dry run is default

        return cmd

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        btn_id = event.button.id

        if btn_id == "btn-back":
            self.app.pop_screen()

        elif btn_id == "btn-load-preset":
            self._load_preset()

        elif btn_id == "btn-save-preset":
            self._save_preset()

        elif btn_id in ("btn-dry-run", "btn-start"):
            values = self._get_form_values()
            if btn_id == "btn-dry-run":
                values["train"] = False
            else:
                values["train"] = True
            cmd = self._build_command(values)

            from attacklm_gui.screens.train_live import TrainLiveScreen

            self.app.push_screen(TrainLiveScreen(cmd))

    def _load_preset(self) -> None:
        """Load a preset into the form."""
        from attacklm_gui.presets import Preset

        select = self.query_one("#preset-select", Select)
        name = str(select.value) if select.value else ""
        preset = Preset.load(name)
        if preset is None:
            self.notify(f"Preset '{name}' not found", severity="error")
            return

        params = preset.params
        field_map = {
            "epochs": "epochs",
            "batch_size": "batch_size",
            "max_length": "max_length",
            "lora_r": "lora_r",
            "lora_alpha": "lora_alpha",
            "lora_dropout": "lora_dropout",
            "galore_rank": "galore_rank",
            "early_stop_steps": "early_stop_steps",
            "optim": "optim",
        }
        bool_map = {
            "use_qgalore": "use_qgalore",
            "use_galore": "use_galore",
            "use_dora": "use_dora",
            "use_rslora": "use_rslora",
            "packing": "packing",
            "live_lr": "live_lr",
            "use_deepspeed": "use_deepspeed",
            "deepspeed_offload": "deepspeed_offload",
            "compile": "compile",
            "use_lomo": "use_lomo",
        }
        select_map = {
            "deepspeed_stage": "deepspeed_stage",
            "compile_mode": "compile_mode",
        }

        for param_key, widget_id in field_map.items():
            if param_key in params:
                widget = self.query_one(f"#{widget_id}", Input)
                widget.value = str(params[param_key])

        for param_key, widget_id in bool_map.items():
            if param_key in params:
                widget = self.query_one(f"#{widget_id}", Checkbox)
                widget.value = bool(params[param_key])

        for param_key, widget_id in select_map.items():
            if param_key in params:
                widget = self.query_one(f"#{widget_id}", Select)
                widget.value = params[param_key]

        # Spectrum is special - it's a float but stored as string in input
        if "spectrum" in params:
            widget = self.query_one("#spectrum", Input)
            widget.value = str(params["spectrum"])

        self.notify(f"Loaded preset: {name}")

    def _save_preset(self) -> None:
        """Save current form values as a preset."""
        from attacklm_gui.presets import Preset

        select = self.query_one("#preset-select", Select)
        name = str(select.value) if select.value else "custom"
        values = self._get_form_values()

        preset = Preset(name=name, params=values, description=f"Saved from GUI")
        preset.save()
        self.notify(f"Saved preset: {name}")
