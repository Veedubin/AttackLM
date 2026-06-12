# PyRIT (Azure)

Python Risk Identification Tool for generative AI. Orchestrates automated red-teaming of LLM endpoints.

## What it's used for

Jailbreak template definitions (no records currently in dataset; slot reserved for future).

## Provenance

| Field | Value |
|---|---|
| **Source name** | `azure-pyrit` |
| **Display name** | PyRIT (Azure) |
| **Upstream** | <https://github.com/Azure/PyRIT> |
| **License** |  |
| **Risk level** | low |

## How it's ingested

The records in this directory are produced by
`scripts/stamp_and_reorg.py`, which reads from
`data/datasets/buckets/<bucket>/` and writes here.

To re-run: `uv run python scripts/stamp_and_reorg.py`

## Rights-holder contact

If you are a rights holder for `PyRIT (Azure)` and would like any of
these records removed, see **`data/REMOVAL.md`** at the repository root.
Removal is fast and unconditional.
