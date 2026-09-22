"""Task type registry — single source of truth for queue task types."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Resolve scripts directory (same logic as cli.py). Overridable via
# ATTACKLM_SCRIPTS_DIR so a detached runner subprocess (spawned with a
# different cwd/package layout in tests) can find the same stub scripts.
_SCRIPTS_DIR = Path(
    os.environ.get("ATTACKLM_SCRIPTS_DIR")
    or Path(__file__).resolve().parent.parent.parent.parent / "scripts"
)


@dataclass(frozen=True)
class TaskSpec:
    """Specification for a queue task type."""

    type: str  # e.g. "train", "audit_prompt_injection"
    label: str  # "Train (single-model)"
    script: str  # "train_all.py" or "audit_prompt_injection.py"
    runner_mode: str  # "subprocess" | "pipeline" | "noop"
    produces_artifact: str | None  # "adapter" | "report" | "data" | None
    consumes_artifact: str | None  # "adapter" (for audits) | None
    arg_schema: dict  # JSON-schema-ish; used for add-time validation
    default_timeout_s: int | None  # None = no timeout
    gauntlet_member: bool  # appears in gauntlet full / core
    implemented: bool  # False for attacks 4/5/6 placeholders
    description: str  # one-liner for queue list

    @property
    def script_path(self) -> Path:
        """Absolute path to the task's script."""
        return _SCRIPTS_DIR / self.script


# ---------------------------------------------------------------------------
# The 9 registered task types (v0.18.0 — Phase 1)
# ---------------------------------------------------------------------------

REGISTRY: dict[str, TaskSpec] = {
    "train": TaskSpec(
        type="train",
        label="Train",
        script="train_all.py",
        runner_mode="subprocess",
        produces_artifact="adapter",
        consumes_artifact=None,
        arg_schema={
            "base_model": {"type": "string", "required": False},
            "single_model": {"type": "boolean", "required": False},
            "single_model_name": {"type": "string", "required": False},
            "include_orchestrator": {"type": "boolean", "required": False},
            "model_attacks": {"type": "boolean", "required": False},
            "include_tools": {"type": "boolean", "required": False},
            "epochs": {"type": "integer", "required": False},
            "batch_size": {"type": "integer", "required": False},
        },
        default_timeout_s=None,  # no timeout for training
        gauntlet_member=False,
        implemented=True,
        description="Train a model (single or multi-bucket)",
    ),
    "audit_prompt_injection": TaskSpec(
        type="audit_prompt_injection",
        label="Audit: Prompt Injection (1)",
        script="audit_prompt_injection.py",
        runner_mode="subprocess",
        produces_artifact="report",
        consumes_artifact="adapter",
        arg_schema={
            "questions": {"type": "string", "required": False},
            "output": {"type": "string", "required": False},
            "base_model": {"type": "string", "required": False},
            "adapter": {"type": "string", "required": False},
            "max_new_tokens": {"type": "integer", "required": False},
        },
        default_timeout_s=3600,
        gauntlet_member=True,
        implemented=True,
        description="Attack 1 — prompt-injection resistance",
    ),
    "audit_system_prompt": TaskSpec(
        type="audit_system_prompt",
        label="Audit: System-Prompt Extraction (2)",
        script="audit_system_prompt.py",
        runner_mode="subprocess",
        produces_artifact="report",
        consumes_artifact="adapter",
        arg_schema={
            "questions": {"type": "string", "required": False},
            "output": {"type": "string", "required": False},
            "base_model": {"type": "string", "required": False},
            "adapter": {"type": "string", "required": False},
            "max_new_tokens": {"type": "integer", "required": False},
        },
        default_timeout_s=3600,
        gauntlet_member=True,
        implemented=True,
        description="Attack 2 — system-prompt extraction",
    ),
    "audit_canary_pipeline": TaskSpec(
        type="audit_canary_pipeline",
        label="Audit: Canary Extraction (3)",
        script="audit_canary_extraction.py",
        runner_mode="pipeline",
        produces_artifact="report",
        consumes_artifact="adapter",
        arg_schema={
            "canaries": {"type": "string", "required": False},      # existing canaries.jsonl; generated if absent
            "num_canaries": {"type": "integer", "required": False}, # count for canary_generator.py (default 50)
            "base_model": {"type": "string", "required": False},
            "adapter": {"type": "string", "required": False},
            "output": {"type": "string", "required": False},
            "max_new_tokens": {"type": "integer", "required": False},
        },
        default_timeout_s=7200,
        gauntlet_member=True,
        implemented=True,
        description="Attack 3 — canary extraction (generate canaries → probe; inject at train time via canary_inject.py)",
    ),
    "audit_calibration": TaskSpec(
        type="audit_calibration",
        label="Audit: Calibration (7)",
        script="eval_calibration.py",
        runner_mode="subprocess",
        produces_artifact="report",
        consumes_artifact="adapter",
        arg_schema={
            "in_distribution": {"type": "string", "required": False},
            "near_ood": {"type": "string", "required": False},
            "ood": {"type": "string", "required": False},
            "base_model": {"type": "string", "required": False},
            "adapter": {"type": "string", "required": False},
            "output": {"type": "string", "required": False},
        },
        default_timeout_s=3600,
        gauntlet_member=True,
        implemented=True,
        description="Attack 7 — calibration & selective prediction",
    ),
    "audit_gcg": TaskSpec(
        type="audit_gcg",
        label="Audit: GCG Triggers (4)",
        script="audit_gcg.py",
        runner_mode="subprocess",
        produces_artifact="report",
        consumes_artifact="adapter",
        arg_schema={
            "base_model": {"type": "string", "required": False},
            "adapter": {"type": "string", "required": False},
            "output": {"type": "string", "required": False},
        },
        default_timeout_s=7200,
        gauntlet_member=True,
        implemented=False,
        description="Attack 4 — GCG adversarial triggers (NOT YET IMPLEMENTED)",
    ),
    "audit_backdoor": TaskSpec(
        type="audit_backdoor",
        label="Audit: Backdoor (5)",
        script="audit_backdoor.py",
        runner_mode="subprocess",
        produces_artifact="report",
        consumes_artifact="adapter",
        arg_schema={
            "base_model": {"type": "string", "required": False},
            "adapter": {"type": "string", "required": False},
            "output": {"type": "string", "required": False},
        },
        default_timeout_s=7200,
        gauntlet_member=True,
        implemented=False,
        description="Attack 5 — backdoor detection (NOT YET IMPLEMENTED)",
    ),
    "audit_repeated_sampling": TaskSpec(
        type="audit_repeated_sampling",
        label="Audit: Repeated Sampling (6)",
        script="audit_repeated_sampling.py",
        runner_mode="subprocess",
        produces_artifact="report",
        consumes_artifact="adapter",
        arg_schema={
            "base_model": {"type": "string", "required": False},
            "adapter": {"type": "string", "required": False},
            "output": {"type": "string", "required": False},
        },
        default_timeout_s=3600,
        gauntlet_member=True,
        implemented=False,
        description="Attack 6 — repeated-sampling extraction (NOT YET IMPLEMENTED)",
    ),
    "gen_calibration_holdouts": TaskSpec(
        type="gen_calibration_holdouts",
        label="Gen Calibration Holdouts",
        script="gen_calibration_holdouts.py",
        runner_mode="subprocess",
        produces_artifact="data",
        consumes_artifact=None,
        arg_schema={
            "output_dir": {"type": "string", "required": False},
        },
        default_timeout_s=600,
        gauntlet_member=False,
        implemented=True,
        description="Generate calibration holdout files (in/near-ood/ood)",
    ),
}


# Mapping from short attack ids/aliases to registry keys.
ATTACK_ALIASES: dict[str, str] = {
    "1": "audit_prompt_injection",
    "2": "audit_system_prompt",
    "3": "audit_canary_pipeline",
    "7": "audit_calibration",
    "4": "audit_gcg",
    "5": "audit_backdoor",
    "6": "audit_repeated_sampling",
    "prompt_injection": "audit_prompt_injection",
    "system_prompt": "audit_system_prompt",
    "canary": "audit_canary_pipeline",
    "calibration": "audit_calibration",
    "gcg": "audit_gcg",
    "backdoor": "audit_backdoor",
    "repeated": "audit_repeated_sampling",
}


def resolve_attack(attack_id: str, include_unshipped: bool = False) -> list[str]:
    """Resolve an attack identifier to a list of registry keys.

    Args:
        attack_id: One of 'all', 'core', a number (1-7), or a name alias.
        include_unshipped: If True, include unimplemented attacks (4/5/6).

    Returns:
        List of registry keys (e.g. ['audit_prompt_injection', ...]).
    """
    if attack_id == "all":
        keys = [
            "audit_prompt_injection",
            "audit_system_prompt",
            "audit_canary_pipeline",
            "audit_calibration",
        ]
        if include_unshipped:
            keys.extend(["audit_gcg", "audit_backdoor", "audit_repeated_sampling"])
        return keys
    if attack_id == "core":
        return [
            "audit_prompt_injection",
            "audit_system_prompt",
            "audit_canary_pipeline",
            "audit_calibration",
        ]
    # Single attack — resolve alias.
    key = ATTACK_ALIASES.get(attack_id, attack_id)
    if key not in REGISTRY:
        raise ValueError(f"Unknown attack: {attack_id!r}")
    spec = REGISTRY[key]
    if not spec.implemented and not include_unshipped:
        raise ValueError(
            f"Attack {attack_id!r} is not yet implemented. "
            f"Use --include-unshipped to add it as pending_unimplemented."
        )
    return [key]
