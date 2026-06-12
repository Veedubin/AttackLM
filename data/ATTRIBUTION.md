# Data Source Attribution

**Last updated:** 2026-06-11 (post license audit & restructure)

This document credits every upstream project whose content is used in
AttackLM's training data. The training dataset is a *transformation*
(typically a reformatting of upstream data into
`messages:[{role, content}]` chat triples) of the sources below. The
resulting model weights are a *new statistical artifact* learned from
these sources; they are not a verbatim copy.

## Per-source provenance

Every record in the public dataset carries these fields (added by
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

The canonical per-source directory layout is
`data/datasets/buckets/sources/<source>/<bucket>/<tactic>/data*.jsonl`.
Each source directory contains a `LICENSE.md` (license, license URI,
per-bucket record counts) and `SOURCE.md` (narrative description, use
case, risk note).

## Quick summary

| Source | License | Records | Risk | Directory |
|---|---|---:|---|---|
| Atomic Red Team (Red Canary)         | MIT                | 1,115  | low    | `sources/atomic-red-team/` |
| MITRE Caldera — Stockpile            | Apache-2.0         | 390    | low    | `sources/mitre-stockpile/` |
| MITRE ATLAS Arsenal                  | Apache-2.0         | 20     | low    | `sources/mitre-atlas-arsenal/` |
| Metasploit Framework (Rapid7)        | BSD-3-Clause       | 13,997 | medium | `sources/metasploit-framework/` |
| garak (NVIDIA)                       | Apache-2.0         | 50     | low    | `sources/nvidia-garak/` |
| promptfoo                            | MIT                | 33     | low    | `sources/promptfoo/` |
| promptmap (utkusen)                  | MIT                | 30     | low    | `sources/promptmap/` |
| LLM-generated (qwen2.5-coder-14b)    | GPL-3.0            | 937    | low    | `sources/llm-generated/` |
| AttackLM Synthetic (in-repo)         | MIT                | 9,029  | low    | `sources/attacklm-synthetic/` |
| **Total**                            |                    | **25,601** |    | |

Reserved slots (no records currently): `azure-pyrit` (MIT), `cyberark-fuzzyai` (Apache-2.0).

## Excluded from public dataset (high-risk)

These sources were reviewed on 2026-06-11 and **excluded** from the
public dataset. The data is preserved locally at
`archive/restricted-sources/` (gitignored) for the author's private
research only. See `data/LEGAL.md` for the rationale and
`data/REMOVAL.md` for the rights-holder contact process.

| Source | License | Records | Reason for exclusion |
|---|---|---:|---|
| endgameinc/RTA                  | AGPL-3.0        | 76  | Viral copyleft. Distributing a derivative dataset would force the entire AttackLM dataset under AGPL-3.0. |
| guardicore/infection_monkey     | GPL-3.0         | 36  | Viral copyleft. Plugin manifests are derivative works of upstream code. |
| TheBigPromptLibrary             | mixed/unclear   | 6   | Copyright laundering of leaked/reverse-engineered proprietary system prompts. |

## Sigma HQ (DRL-1.1) — no current direct use

Sigma HQ's `sigma` rules are licensed under DRL 1.1, which requires
attribution propagation when redistributing modified forms. As of this
audit, no DRL-licensed content is included in the public dataset — the
Sigma rule *structure* (the DRL 1.1 spec) is referenced when designing
triples, but no actual Sigma rules are redistributed.

If Sigma rules are added in the future, each record must carry the
upstream attribution: `sigma.rule.title`, `sigma.rule.id`,
`sigma.rule.author`, `sigma.rule.date`. See
`data/LEGAL.md` for details.

## How to use this document

- If you distribute AttackLM or its training data, **preserve the
  per-record attribution fields** (`source`, `source_uri`, `license`,
  `license_uri`, `rights_contact`) and the per-source `LICENSE.md`
  files. Do not strip them.
- If you modify the training data, **document what you changed** in
  `CHANGELOG.md`.
- If you re-distribute the trained model, review
  `data/LEGAL.md` §5 (BSD-3-Clause attribution for Metasploit-derived
  records, which are 54.7% of the dataset).

## License of this Repository

The **code, scripts, and orchestration** in this repository are
released under the MIT License. See `LICENSE` at the root.

The **training data** at `data/datasets/buckets/sources/<source>/` is a
derivative work of the upstream sources above and inherits each source's
respective license. The **trained model weights** are a new statistical
artifact learned from openly licensed material; the model is released
under the MIT License with the understanding that it is a new work, not
a verbatim copy.

---

For the full legal notice, see
[**data/LEGAL.md**](LEGAL.md).

For the rights-holder removal process, see
[**data/REMOVAL.md**](REMOVAL.md).
