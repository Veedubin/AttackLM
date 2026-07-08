"""Configuration preset management for AttackLM training parameters."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PRESETS_DIR = Path.home() / ".config" / "attacklm" / "presets"


def _slugify(name: str) -> str:
    """Slugify a preset name: lowercase, replace any non-alphanumeric run
    with a single underscore. Used as the on-disk filename.

    Avoids filesystem errors on names with "/" or other special chars
    (e.g. "FP8 (H100/Blackwell)" → "fp8_h100_blackwell.json").
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


@dataclass
class Preset:
    """A saved training configuration."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    @property
    def filename(self) -> str:
        return f"{_slugify(self.name)}.json"

    @property
    def path(self) -> Path:
        return PRESETS_DIR / self.filename

    def save(self) -> None:
        """Save preset to disk."""
        PRESETS_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "description": self.description,
            "params": self.params,
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, name: str) -> Preset | None:
        """Load a preset by name."""
        filename = f"{_slugify(name)}.json"
        path = PRESETS_DIR / filename
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return cls(
            name=data["name"],
            params=data.get("params", {}),
            description=data.get("description", ""),
        )

    @classmethod
    def list_all(cls) -> list[str]:
        """List all saved preset names."""
        PRESETS_DIR.mkdir(parents=True, exist_ok=True)
        presets = []
        for p in sorted(PRESETS_DIR.glob("*.json")):
            try:
                with open(p) as f:
                    data = json.load(f)
                presets.append(data.get("name", p.stem))
            except (json.JSONDecodeError, OSError):
                pass
        return presets

    @classmethod
    def delete(cls, name: str) -> bool:
        """Delete a preset by name. Returns True if deleted."""
        filename = f"{_slugify(name)}.json"
        path = PRESETS_DIR / filename
        if path.exists():
            path.unlink()
            return True
        return False


# --- Built-in presets ---

BUILTIN_PRESETS: list[Preset] = [
    Preset(
        name="3B Q-GaLore Spectrum",
        description="Q-GaLore full-parameter training for 3B models on 16GB GPU",
        params={
            "use_qgalore": True,
            "spectrum": 0.5,
            "galore_rank": 64,
            "epochs": 20,
            "batch_size": 1,
            "max_length": 12000,
            "packing": True,
            "early_stop_steps": 5,
            "live_lr": True,
            "optim": "paged_adamw_8bit",
        },
    ),
    Preset(
        name="3B Q-GaLore Rank 128",
        description="Higher-rank Q-GaLore for better quality on 3B models",
        params={
            "use_qgalore": True,
            "spectrum": 0.5,
            "galore_rank": 128,
            "epochs": 30,
            "batch_size": 1,
            "max_length": 12000,
            "packing": True,
            "early_stop_steps": 5,
            "live_lr": True,
            "optim": "paged_adamw_8bit",
        },
    ),
    Preset(
        name="3B LoRA Default",
        description="Standard QLoRA training for 3B models",
        params={
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "epochs": 10,
            "batch_size": 2,
            "max_length": 2048,
            "packing": False,
            "early_stop_steps": 5,
            "optim": "paged_adamw_8bit",
        },
    ),
    Preset(
        name="7B Q-GaLore",
        description="Q-GaLore for 7B models on 24GB GPU",
        params={
            "use_qgalore": True,
            "spectrum": 0.5,
            "galore_rank": 128,
            "epochs": 20,
            "batch_size": 1,
            "max_length": 8192,
            "packing": True,
            "early_stop_steps": 5,
            "live_lr": True,
            "optim": "paged_adamw_8bit",
        },
    ),
    Preset(
        name="7B QLoRA Default",
        description="Standard QLoRA for 7B models",
        params={
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "epochs": 10,
            "batch_size": 2,
            "max_length": 2048,
            "packing": False,
            "early_stop_steps": 5,
            "optim": "paged_adamw_8bit",
        },
    ),
    Preset(
        name="DeepSpeed 40B+",
        description="DeepSpeed ZeRO-3 with CPU offload and torch.compile for 40B+ models",
        params={
            "use_deepspeed": True,
            "deepspeed_stage": 3,
            "deepspeed_offload": True,
            "compile": True,
            "compile_mode": "reduce-overhead",
            "epochs": 10,
            "batch_size": 1,
            "max_length": 4096,
            "packing": True,
            "early_stop_steps": 5,
            "optim": "adamw_torch",
        },
    ),
    Preset(
        name="COAP 8-bit",
        description="COAP optimizer with 8-bit quantization for memory-efficient training",
        params={
            "use_coap": True,
            "coap_8bit": True,
            "coap_rank": 128,
            "epochs": 20,
            "batch_size": 1,
            "max_length": 8192,
            "packing": True,
            "early_stop_steps": 5,
            "optim": "paged_adamw_8bit",
        },
    ),
    Preset(
        name="FlashOptim",
        description="FlashOptim accelerated training for modern GPUs",
        params={
            "use_flashoptim": True,
            "epochs": 20,
            "batch_size": 1,
            "max_length": 12000,
            "packing": True,
            "early_stop_steps": 5,
            "live_lr": True,
            "optim": "paged_adamw_8bit",
        },
    ),
    Preset(
        name="FP8 (H100/Blackwell)",
        description="FP8 precision training — requires H100 or Blackwell GPU",
        params={
            "fp8": True,
            "epochs": 20,
            "batch_size": 2,
            "max_length": 8192,
            "packing": True,
            "early_stop_steps": 5,
            "optim": "paged_adamw_8bit",
        },
    ),
    Preset(
        name="BitNet 2B",
        description="BitNet 1.58-bit training with microsoft/bitnet-b1.58-2B4T base",
        params={
            "bitnet": True,
            "base_model": "microsoft/bitnet-b1.58-2B4T",
            "epochs": 20,
            "batch_size": 1,
            "max_length": 8192,
            "packing": True,
            "early_stop_steps": 5,
            "optim": "paged_adamw_8bit",
        },
    ),
]


def ensure_builtin_presets() -> None:
    """Create built-in presets if they don't exist."""
    existing = set(Preset.list_all())
    for preset in BUILTIN_PRESETS:
        if preset.name not in existing:
            preset.save()
