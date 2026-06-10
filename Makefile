# AttackLM Makefile
# ==================
# Convenience targets for the most common workflows. Each target
# documents itself when you run `make help` or `make <target>`.

.PHONY: help install clone extract buckets train hpo merge demo clean \
        audit test all

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Base model — override on the command line, e.g. `make train MODEL=...`
MODEL ?= unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit
EPOCHS ?= 5
MAX_LENGTH ?= 2048
LORA_R ?= 16
LORA_ALPHA ?= 32
LORA_DROPOUT ?= 0.05
BATCH_SIZE ?= 1
SAVE_STEPS ?= 200

OUTPUT_DIR ?= models/attacklm-single
HPO_DIR ?= hpo_runs
LOG_DIR ?= logs

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

help:  ## Show this help message
	@echo "AttackLM Makefile"
	@echo ""
	@echo "Common targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Configuration (override on command line):"
	@echo "  MODEL=$(MODEL)"
	@echo "  EPOCHS=$(EPOCHS)"
	@echo "  MAX_LENGTH=$(MAX_LENGTH)"
	@echo "  LORA_R=$(LORA_R) LORA_ALPHA=$(LORA_ALPHA) LORA_DROPOUT=$(LORA_DROPOUT)"
	@echo "  OUTPUT_DIR=$(OUTPUT_DIR)"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install:  ## Install Python dependencies (uv sync)
	uv sync

clone:  ## Clone upstream data source repos (~1.5GB)
	uv run python scripts/clone_repos.sh

# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

extract:  ## Run all data extractors
	uv run python scripts/extract_atomic_red_team_to_jsonl.py
	uv run python scripts/extract_caldera_plugins_to_jsonl.py
	uv run python scripts/parse_metasploit_to_jsonl.py
	uv run python scripts/extract_rta_to_jsonl.py
	uv run python scripts/extract_infection_monkey_to_jsonl.py
	uv run python scripts/extract_ai_tools_to_jsonl.py
	@echo ""
	@echo "  All extractors complete. Next: make buckets"

attribution:  ## Add per-pair source/license attribution to JSONLs
	uv run python scripts/augment_attribution.py

buckets:  ## Organize extracted data into 16 MITRE/AI/tools buckets
	uv run python scripts/setup_buckets.py
	uv run python scripts/reorganize_buckets.py

data: clone extract attribution buckets  ## Full data pipeline: clone → extract → attribute → bucket

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

train:  ## Train single model on combined dataset
	uv run python scripts/train_all.py --single-model \
	  --base-model $(MODEL) \
	  --epochs $(EPOCHS) --max-length $(MAX_LENGTH) \
	  --batch-size $(BATCH_SIZE) \
	  --lora-r $(LORA_R) --lora-alpha $(LORA_ALPHA) --lora-dropout $(LORA_DROPOUT) \
	  --save-steps $(SAVE_STEPS) \
	  --output-name $$(basename $(OUTPUT_DIR))

train-multi:  ## Train one model per bucket (multi-model MoE mode)
	uv run python scripts/train_all.py \
	  --base-model $(MODEL) \
	  --epochs $(EPOCHS) --max-length $(MAX_LENGTH) \
	  --batch-size $(BATCH_SIZE) \
	  --lora-r $(LORA_R) --lora-alpha $(LORA_ALPHA) --lora-dropout $(LORA_DROPOUT)

hpo:  ## Run coordinate-descent HPO + final training
	uv run python scripts/train_all.py --hpo --single-model \
	  --base-model $(MODEL) \
	  --epochs $(EPOCHS) --max-length $(MAX_LENGTH) \
	  --hpo-output-dir $(HPO_DIR)

hpo-analyze:  ## Show results from a prior HPO run
	uv run python scripts/hpo_runner.py --hpo-dir $(HPO_DIR) --analyze-only

merge:  ## Merge LoRA adapter into base model for deployment
	uv run python scripts/merge_adapter.py \
	  --base-model $(MODEL) \
	  --adapter $(OUTPUT_DIR) \
	  --output $$(echo $(OUTPUT_DIR) | sed 's/-single/-merged/')

demo:  ## Run inference demo with the trained adapter
	uv run python scripts/demo.py --adapter $(OUTPUT_DIR)

# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

audit:  ## Audit which scripts are actively used
	@uv run python -c "import os, ast; [print(f, '->', len([n for n in ast.parse(open(f).read()).body if isinstance(n, ast.FunctionDef)]), 'funcs') for f in sorted(__import__('glob').glob('scripts/*.py'))]"

clean:  ## Remove temporary files (does NOT delete trained models or data)
	rm -rf hpo_runs/*_trial*/
	rm -f hpo_runs/*.csv hpo_runs/*.dataset_stats.json
	rm -f logs/*.log logs/*.tmp
	@echo "  Cleaned HPO trial outputs and logs"
	@echo "  (trained models in $(OUTPUT_DIR) untouched)"

# ---------------------------------------------------------------------------
# Quickstart alias
# ---------------------------------------------------------------------------

all: data train  ## Full pipeline: data → train
