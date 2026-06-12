# AttackLM Training Data — Per-Source Layout

This directory is the **canonical** layout for the AttackLM training data.
Records are organized by upstream **source** first, then by bucket.

## Layout

```
sources/
├── _index.json                  # machine-readable manifest
├── atomic-red-team/             # MIT — Red Canary
├── mitre-stockpile/             # Apache-2.0 — MITRE Caldera
├── mitre-atlas-arsenal/         # Apache-2.0 — MITRE ATLAS
├── metasploit-framework/        # BSD-3-Clause — Rapid7 (medium risk: attribution required)
├── nvidia-garak/                # Apache-2.0 — NVIDIA
├── azure-pyrit/                 # MIT — Azure (slot reserved, no records yet)
├── cyberark-fuzzyai/            # Apache-2.0 — CyberArk (slot reserved, no records yet)
├── promptfoo/                   # MIT
├── promptmap/                   # MIT — utkusen
├── llm-generated/               # GPL-3.0 — qwen2.5-coder-14b output
└── attacklm-synthetic/          # MIT — in-repo deterministic templates
```

Each source directory contains:
- `LICENSE.md` — the license, license URI, per-bucket record counts
- `SOURCE.md` — narrative description, use case, risk note
- `<bucket>/<tactic>/data.jsonl` (and `data_synth.jsonl`, `data_llm.jsonl`
  where applicable)

## Why this layout

The previous flat layout (`data/datasets/buckets/<bucket>/data.jsonl`) made
attribution ambiguous: a single bucket could mix MIT, Apache-2.0, BSD-3, and
in-repo synthetic records, and the per-record `source` field was the only
way to disambiguate.

The per-source layout:
1. **Makes attribution unambiguous** — every record in `metasploit-framework/`
   is BSD-3-Clause, every record in `atomic-red-team/` is MIT, etc.
2. **Simplifies license compliance** — re-distributors can drop an entire
   source by deleting one directory.
3. **Simplifies rights-holder removal** — `data/REMOVAL.md` requests are
   answered by deleting one directory.
4. **Preserves the per-tactic bucket structure** — each source still has
   `<bucket>/<tactic>/data.jsonl` so training scripts that consume by
   tactic continue to work.

## High-risk sources (NOT in this directory)

The following sources have been moved to `archive/restricted-sources/`
(per `data/ATTRIBUTION.md` license review 2026-06-11) and are **excluded**
from the public dataset:

| Source | License | Reason |
|---|---|---|
| endgameinc/RTA | AGPL-3.0 | Viral copyleft |
| guardicore/infection_monkey | GPL-3.0 | Viral copyleft |
| TheBigPromptLibrary | mixed/unclear | Copyright laundering of leaked prompts |

The data is preserved locally for the user's private research. See
`archive/restricted-sources/README.md` for details.

## Migration

The flat layout at `data/datasets/buckets/<bucket>/` is no longer the
canonical source. Training scripts and the audit tool should be updated to
read from `data/datasets/buckets/sources/<source>/...`.

The manifest at `data/datasets/buckets/manifest.json` is regenerated from
this directory by `scripts/rebuild_manifest.py`.

## Provenance stamp

Every record in this directory carries these fields (added by
`scripts/stamp_and_reorg.py`):

```json
{
  "source": "atomic-red-team",
  "source_uri": "https://github.com/redcanaryco/atomic-red-team",
  "license": "MIT",
  "license_uri": "https://opensource.org/licenses/MIT",
  "rights_contact": "see data/REMOVAL.md"
}
```

These fields are written by the ETL pipeline and **must not be stripped**
by downstream re-distributors.

## Total record count

25,601 records across 11 sources.
