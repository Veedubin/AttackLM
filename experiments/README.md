# Experiments

This directory contains experimental scripts that are not part of the
canonical training pipeline. They are kept here for reference but should
not be used in production runs.

## synthetic_dataset_mutator_evol_instruct.py

A prototype LLM-based data mutator that uses a local LLM (LMStudio /
Ollama) to generate diverse phrasings of red-team scenarios. Useful for
augmenting the synthetic orchestrator dataset with realistic operator
noise (Slack messages, C2 dashboard reports, etc.).

**Status:** Experimental. Requires a local LLM server running on
`localhost:1234`. The orchestrator bucket in the canonical dataset uses
a deterministic generator (`scripts/generate_orchestrator.py`) instead.

**Why not in main pipeline:** LLM-generated data can hallucinate
technically incorrect tradecraft. Deterministic generation is preferred
for training data quality.
