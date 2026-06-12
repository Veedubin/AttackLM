# Legal Notice — AttackLM Training Data

**Last updated:** 2026-06-11

## Purpose

AttackLM is a research dataset for training language models on offensive
security techniques, mapped to MITRE ATT&CK. The dataset is used by the
author to experiment with training a model that can reason about the
"art of hacking" — the belief being that understanding offense is a
prerequisite to building good defense.

**It takes a hacker to catch a hacker.** That's the research premise.

This document covers:
1. Scope of use (research only)
2. Sources and their licenses
3. Sources that have been excluded due to license risk
4. Removal requests

For the rights-holder removal process, see
[**REMOVAL.md**](REMOVAL.md) in this directory.

---

## 1. Scope of use — RESEARCH ONLY

The AttackLM dataset and any model trained on it is for **non-commercial
research purposes only**. Specifically:

- **Permitted uses**: personal research, private experimentation, academic
  study, security-tool evaluation against your own systems, red-team
  training for defensive teams.
- **NOT permitted uses**: commercial deployment, redistribution of the
  raw dataset for commercial purposes, use of trained models to attack
  systems you do not own or have explicit permission to test, generating
  attack payloads for use against third parties.

The author does not provide the trained model weights, the raw dataset,
or any derivative dataset as a hosted service. Distribution is via
the GitHub repository at <https://github.com/Veedubin/AttackLM>, which
is the user's choice of platform.

---

## 2. Sources and their licenses

The current public dataset is **25,601 records** across 11 upstream
sources. Every record carries a `source`, `source_uri`, `license`, and
`license_uri` field. See
**`data/datasets/buckets/sources/_index.json`** for the full breakdown.

| Source | License | Records | Risk |
|---|---|---:|---|
| Atomic Red Team (Red Canary)         | MIT                | 1,115 | low    |
| MITRE Caldera — Stockpile            | Apache-2.0         | 390   | low    |
| MITRE ATLAS Arsenal                  | Apache-2.0         | 20    | low    |
| Metasploit Framework (Rapid7)        | BSD-3-Clause       | 13,997 | medium |
| garak (NVIDIA)                       | Apache-2.0         | 50    | low    |
| promptfoo                            | MIT                | 33    | low    |
| promptmap (utkusen)                  | MIT                | 30    | low    |
| LLM-generated (qwen2.5-coder-14b)    | GPL-3.0            | 937   | low    |
| AttackLM Synthetic (in-repo)         | MIT                | 9,029 | low    |
| **Total**                            |                    | **25,601** | |

The full per-source breakdown with bucket-level counts is in
`data/datasets/buckets/sources/<source>/LICENSE.md` for each source.

---

## 3. Sources EXCLUDED from the public dataset

The following sources were reviewed and **excluded** from the public
dataset due to license / copyright risk. The data is preserved locally
in `archive/restricted-sources/` for the author's private research only
and is **not** redistributed as part of AttackLM.

| Source | License | Records | Reason for exclusion |
|---|---|---:|---|
| endgameinc/RTA                  | AGPL-3.0        | 76  | Viral copyleft. Distributing a derivative dataset would force the entire AttackLM dataset under AGPL-3.0. |
| guardicore/infection_monkey     | GPL-3.0         | 36  | Viral copyleft. Plugin manifests are derivative works of upstream code. |
| TheBigPromptLibrary             | mixed/unclear   | 6   | IP / copyright laundering. The repo hosts leaked and reverse-engineered proprietary system prompts under an "MIT" badge that does not actually grant copyright. |

For the full audit trail of this decision, see the 2026-06-11 license
review notes in `CHANGELOG.md` and the `archive/restricted-sources/README.md`.

---

## 4. Sigma HQ (DRL-1.1) and Metasploit (BSD-3-Clause) — attribution

Two sources in the dataset have attribution requirements:

- **SigmaHQ/sigma** (Detection Rule License 1.1): not currently used as a
  direct training source; only the rule *structure* (DRL 1.1 spec) is
  referenced when designing triples. No DRL-licensed content is
  distributed in the dataset.
- **Metasploit Framework** (BSD-3-Clause): the largest source by record
  count (13,997 records). BSD-3 requires that the original copyright
  notice and license text be retained in any redistribution. The
  upstream copyright is preserved in the per-record `source` /
  `source_uri` fields and in
  `data/datasets/buckets/sources/metasploit-framework/LICENSE.md`.

**If you redistribute this dataset or a model trained on it, preserve
the per-record `source` / `source_uri` / `license` / `license_uri`
fields and the per-source `LICENSE.md` files. Do not strip them.**

---

## 5. Rights-holder contact

If you are a rights holder for any of the sources listed above (or any
other source in the dataset) and would like any records removed:

- See [**REMOVAL.md**](REMOVAL.md) in this directory.
- Removal is fast and unconditional.
- The author will not dispute removal requests; this is a research
  project, not a commercial product.

---

## 6. License of this repository

The **code, scripts, and orchestration** in the AttackLM repository are
released under the MIT License (see `LICENSE` at the repository root).

The **training data** at `data/datasets/buckets/sources/<source>/` is a
derivative work of the upstream sources above and inherits each source's
respective license. The **trained model weights** are a new statistical
artifact learned from openly licensed material; the model is released
under the MIT License with the understanding that it is a new work, not
a verbatim copy.
