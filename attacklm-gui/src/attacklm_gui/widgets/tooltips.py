"""Centralized tooltip text for AttackLM TUI widgets.

Why centralized:
1. All tooltip text reviewable in one place (a code review can verify every
   tooltip is accurate and concise).
2. Translatable later if the project ever needs i18n.
3. Easy to grep for "what does this widget do?" questions.
"""

from __future__ import annotations


TOOLTIPS: dict[str, str] = {
    # === Main menu buttons (8 existing) ===
    "btn-train": "Open the training form. 40+ parameters organized into "
    "tabs (Basic, LoRA, GaLore, Advanced, Hardware).",
    "btn-init": "Initialize the AttackLM dataset: clone upstream sources, "
    "run extractors, organize into buckets. Takes 10-30 min.",
    "btn-balance": "Balance a dataset by cap size. Useful when one source "
    "dominates (e.g. Metasploit LOW records).",
    "btn-build": "Build a trained model: merge LoRA → export GGUF → "
    "install to LM Studio. Run after a successful training.",
    "btn-infer": "Run inference on a trained model with a prompt. Returns "
    "the model's completion. Useful for spot-checking outputs.",
    "btn-eval": "Run the retention eval suite (30+ tests). Compares a "
    "candidate model against a golden set. Use after training.",
    "btn-steer": "Steer a trained model: adjust generation parameters "
    "without retraining (temperature, repetition penalty, etc.).",
    "btn-bench": "Benchmark a trained model: tokens/sec, time-to-first-"
    "token, peak VRAM. Useful for hardware planning.",
    "btn-audit": "Run an inversion-attack audit (extraction or MIA). Tests "
    "whether the model memorizes training data (extraction) or "
    "whether specific records were in the training set (MIA).",
    # === New: Audit screen (M2) ===
    "audit_model": "Path to the trained AttackLM model directory "
    "(e.g. /home/user/AttackLM/uncensored).",
    "audit_dataset_root": "Path to the per-source dataset root "
    "(default: data/datasets/buckets/sources).",
    "audit_source_filter": "Optional comma-separated list of sources to probe "
    "(e.g. metasploit-framework,sigma-hq). Empty = all.",
    "audit_attack": "Which attack class to run. 'extraction' tests if the "
    "model can regurgitate training data given a prefix. "
    "'mia' tests if the model has seen a specific record.",
    "audit_mia_method": "MIA scoring algorithm. 'reference' = NLL only. "
    "'zlib' = NLL - zlib_length (best for short text). "
    "'per_token' = NLL / suffix tokens (best for long "
    "text, MUSE 2023 default). 'lira' requires v0.5.0.",
    "audit_mia_threshold": "How to derive the membership threshold. "
    "'percentile' (default, recommended) = use the "
    "5th percentile of probed scores. 'median' is a "
    "calibration artifact (see MIA_THRESHOLD_CALIBRATION.md).",
    "audit_mia_percentile": "Percentile for the threshold (default 5 = top "
    "5% of most-memorized records flagged).",
    "audit_top_k": "Number of completions to generate per prefix. K=20 is "
    "the Carlini 2021 standard; K=10 is faster but less thorough.",
    "audit_max_new_tokens": "Token budget per completion. 256 is the Carlini "
    "2021 standard; lower values cause under-counting. "
    "See attacklm-dataset/docs/PROBE_TOKEN_BUDGET.md.",
    "audit_temperature": "Sampling temperature for the extraction probe. "
    "T=1.0 is the Carlini 2021 default; T=0.0 is greedy.",
    "audit_max_records": "Maximum number of records to probe (per source). "
    "Use a small number (e.g. 50) for quick checks, "
    "large (e.g. 1000+) for production audits.",
    # === Existing: train_form.py high-traffic params (10) ===
    "train_epochs": "Number of training passes over the dataset. 3-5 for "
    "QLoRA, 1-2 for Q-GaLore (the optimizer compensates).",
    "train_batch_size": "Per-device batch size. Memory-bound: lower if you "
    "OOM, raise to speed up if you have VRAM headroom.",
    "train_lora_r": "LoRA rank. Higher = more parameters = more capacity, "
    "but slower training. 32 is a good default; 64 for "
    "large datasets.",
    "train_lora_alpha": "LoRA scaling. alpha/r = effective scaling. Common: "
    "alpha=2r (so effective scaling = 2).",
    "train_galore_rank": "GaLore projection rank. 128 is the standard; lower "
    "for memory-constrained runs.",
    "train_use_qgalore": "Quantized GaLore: memory-efficient full-parameter "
    "training. Mutually exclusive with LoRA.",
    "train_use_dora": "DoRA (Decomposed Low-Rank Adaptation). Better than "
    "LoRA at preserving quality; ~5% slower. v0.10.0+.",
    "train_spectrum": "Spectrum layer freezing. Trains only the SNIP top-k% "
    "layers, freezing the rest. Memory saver.",
    "train_max_length": "Maximum sequence length in tokens. 1024 is common; "
    "raise to 2048 for long-form content, lower to 512 "
    "for memory savings.",
    "train_learning_rate": "Optimizer learning rate. 2e-4 for QLoRA, 1e-5 "
    "for Q-GaLore, 5e-6 for full-parameter.",
}


def attach_tooltip(widget, key: str) -> None:
    """Attach a tooltip to a widget by TOOLTIPS key. No-op if key is missing.

    Usage:
        yield Button("Run", id="btn-run")
        # In on_mount or after the widget is mounted:
        attach_tooltip(self.query_one("#btn-run", Button), "btn-run")
    """
    text = TOOLTIPS.get(key)
    if text is not None:
        widget.tooltip = text
