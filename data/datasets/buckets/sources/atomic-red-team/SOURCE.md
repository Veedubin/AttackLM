# Atomic Red Team (Red Canary)

Community-driven library of tests mapped to MITRE ATT&CK. Each test is a small, safe-to-run adversary emulation: command, input arguments, expected detection artifact, cleanup.

## What it's used for

Exact commands, expected artifacts, and cleanup for ~700 ATT&CK techniques.

## Provenance

| Field | Value |
|---|---|
| **Source name** | `atomic-red-team` |
| **Display name** | Atomic Red Team (Red Canary) |
| **Upstream** | <https://github.com/redcanaryco/atomic-red-team> |
| **License** | MIT |
| **Risk level** | low |

## How it's ingested

The records in this directory are produced by
`scripts/stamp_and_reorg.py`, which reads from
`data/datasets/buckets/<bucket>/` and writes here.

To re-run: `uv run python scripts/stamp_and_reorg.py`

## Rights-holder contact

If you are a rights holder for `Atomic Red Team (Red Canary)` and would like any of
these records removed, see **`data/REMOVAL.md`** at the repository root.
Removal is fast and unconditional.
