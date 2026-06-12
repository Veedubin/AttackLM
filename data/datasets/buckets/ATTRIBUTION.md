# Bucket Data — Per-Source Attribution

**Last updated:** 2026-06-11 (post license audit & restructure)

The training data is now organized **per-source** rather than per-bucket.
The canonical layout is:

```
data/datasets/buckets/sources/<source>/<bucket>/<tactic>/data*.jsonl
```

Each source directory contains a `LICENSE.md` and a `SOURCE.md`. The
flat `data/datasets/buckets/<bucket>/data.jsonl` layout has been moved
to `archive/old-flat-layout/` and is no longer the source of truth.

## Bucket manifest

The full machine-readable manifest is `manifest.json` in this directory
(version 5). It records each bucket's name, MITRE tactic ID (TA0001-
TA0011, TA0040 for AI), record count, per-source breakdown, and
dominant source license.

## Per-source totals

| Source | License | Records | Risk |
|---|---|---:|---|
| metasploit-framework  | BSD-3-Clause       | 13,997 | medium |
| attacklm-synthetic    | MIT                | 9,029  | low    |
| atomic-red-team       | MIT                | 1,115  | low    |
| llm-generated         | GPL-3.0            | 937    | low    |
| mitre-stockpile       | Apache-2.0         | 390    | low    |
| nvidia-garak          | Apache-2.0         | 50     | low    |
| promptfoo             | MIT                | 33     | low    |
| promptmap             | MIT                | 30     | low    |
| mitre-atlas-arsenal   | Apache-2.0         | 20     | low    |
| **TOTAL**             |                    | **25,601** |  |

Reserved slots (no records currently): `azure-pyrit` (MIT),
`cyberark-fuzzyai` (Apache-2.0).

## Excluded (high-risk) sources

| Source | License | Records | Reason |
|---|---|---:|---|
| endgameinc/RTA                  | AGPL-3.0        | 76  | Viral copyleft |
| guardicore/infection_monkey     | GPL-3.0         | 36  | Viral copyleft |
| TheBigPromptLibrary             | mixed/unclear   | 6   | Copyright laundering |

The excluded data lives at `archive/restricted-sources/<source>/bucket/`
(gitignored). See `archive/restricted-sources/README.md` for the full
rationale and `data/REMOVAL.md` for the rights-holder contact process.

## Adding new buckets / sources

1. **New source** — create `data/datasets/buckets/sources/<source>/`
   with a `LICENSE.md` and `SOURCE.md` following the existing
   convention. Add a `PROVENANCE` entry to
   `scripts/stamp_and_reorg.py` and re-run it. Update
   `data/ATTRIBUTION.md` and this file.
2. **New records to an existing source** — drop them into the
   appropriate `sources/<source>/<bucket>/<tactic>/data.jsonl` file.
   Re-run `scripts/stamp_and_reorg.py` to add provenance fields.
3. **Re-distribution review** — every record carries
   `source` / `source_uri` / `license` / `license_uri` /
   `rights_contact`. Do not strip these fields when redistributing.
