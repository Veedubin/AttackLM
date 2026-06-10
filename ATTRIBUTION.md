# Data Source Attribution

This document credits every upstream project whose content is used in
AttackLM's training data. **All training data is derived from openly
licensed open-source projects.** We do not claim authorship of any
technique, command, module, or rule — the original authors do.

The training dataset is a *transformation* (typically a reformatting of
upstream data into `messages:[{role, content}]` chat triples) of the
sources below. The resulting model weights are a *new statistical
artifact* learned from these sources; they are not a verbatim copy.

## How to use this document

- If you distribute AttackLM or its training data, **preserve the
  attribution in this file** per each source's license.
- If you modify the training data, **document what you changed**.
- If you re-distribute the trained model commercially, **review the
  AGPLv3 RTA note below** — it has network-distribution implications.

---

## Training Data Sources (16,982 pairs across 16 buckets)

### 1. Atomic Red Team (redcanaryco)

| Field | Value |
|---|---|
| **Pairs** | 2,506 |
| **Buckets** | 9 MITRE tactics + tools/atomic |
| **Repository** | <https://github.com/redcanaryco/atomic-red-team> |
| **License** | MIT License |
| **Source file** | `data/atomic-red-team/LICENSE.txt` |
| **Used for** | Atomic test triples — exact commands, expected artifacts, cleanup for ~700 ATT&CK techniques |

Copyright (c) Red Canary, LLC. All rights reserved.

### 2. MITRE Caldera — Stockpile (mitre)

| Field | Value |
|---|---|
| **Pairs** | 608 |
| **Buckets** | All 10 MITRE tactics (re-routed via T-IDs) |
| **Repository** | <https://github.com/mitre/stockpile> |
| **License** | Apache License 2.0 |
| **Source file** | `data/stockpile/LICENSE` |
| **Used for** | Adversary emulation ability triples (YAML-defined TTP descriptors) |

© MITRE Corporation. Approved for public release.

### 3. MITRE Caldera — Arsenal / Manx / Access plugins

| Field | Value |
|---|---|
| **Pairs** | 56 (Arsenal 42 + Manx 6 + Access 8) |
| **Buckets** | Mixed |
| **Repositories** | <https://github.com/mitre/caldera> (subdirs: `plugins/arsenal`, `plugins/manx`, `plugins/access`) |
| **License** | Apache License 2.0 |
| **Used for** | Specialized Caldera plugin descriptors |

### 4. MITRE ATT&CK Framework (attack.mitre.org)

| Field | Value |
|---|---|
| **Pairs** | (used as labels, not training content) |
| **Website** | <https://attack.mitre.org> |
| **License** | Apache License 2.0 (per MITRE Terms of Use) |
| **Used for** | Tactic IDs (TA0001-TA0011, TA0040 for AI), technique IDs (T-numbers), tactic descriptions |

**Note:** MITRE ATT&CK itself is a taxonomy. The descriptions and IDs are
the "vocabulary" the training data is tagged with. We do not train on
the prose of the ATT&CK website directly.

### 5. Sigma Rules (SigmaHQ)

| Field | Value |
|---|---|
| **Pairs** | (used as auxiliary context, not direct training) |
| **Repository** | <https://github.com/SigmaHQ/sigma> |
| **License** | Detection Rule License (DRL) 1.1 for rules; public domain for spec |
| **Source file** | `data/sigma/LICENSE` |
| **Used for** | Sigma's 30+ detection rule fields (title, description, logsource, detection) inform the structure of triples |

DRL 1.1 permits use, modification, and distribution with attribution.

### 6. Metasploit Framework (rapid7)

| Field | Value |
|---|---|
| **Pairs** | 8,349 |
| **Buckets** | `tools/metasploit` (15 module categories consolidated) + re-routed to 5 MITRE tactics |
| **Repository** | <https://github.com/rapid7/metasploit-framework> |
| **License** | BSD 3-Clause License |
| **Source files** | `data/metasploit-framework/LICENSE`, `data/metasploit-framework/COPYING` |
| **Used for** | Module descriptions, options, references, payloads — converted to "what this module does" triples |

Copyright (c) 2006-2026, Rapid7, Inc. All rights reserved.
Redistribution permitted under BSD-3-Clause terms.

### 7. Infection Monkey (guardicore)

| Field | Value |
|---|---|
| **Pairs** | 36 |
| **Buckets** | `tools/infection_monkey` |
| **Repository** | <https://github.com/guardicore/monkey> |
| **License** | GNU General Public License v3.0 |
| **Source file** | `data/infection_monkey/LICENSE` |
| **Used for** | Plugin manifests + MITRE mapping |

Copyright (C) 2007 Free Software Foundation, Inc.

### 8. RTA — Red Team Automation (endgameinc)

| Field | Value |
|---|---|
| **Pairs** | 76 |
| **Buckets** | `tools/rta` |
| **Repository** | <https://github.com/endgameinc/RTA> |
| **License** | GNU Affero General Public License v3.0 |
| **Source file** | `data/RTA/LICENSE.txt` |
| **Used for** | Python TTP scripts (`# ATT&CK: TXXXX` headers) |

**⚠️ AGPLv3 IMPLICATION FOR RTA DATA**

RTA is licensed under the **GNU Affero General Public License v3.0**,
which is significantly more restrictive than the other sources:

> *"If you modify the Program, your modified version must prominently
> offer all users interacting with it remotely an opportunity to
> receive the Corresponding Source of your version by providing access
> to the Corresponding Source from a network server."*
> — AGPLv3 §13 (Remote Network Interaction; Use with the GNU General
> Public License v3)

The RTA-derived triples in our dataset are transformations of the
upstream RTA scripts. If you distribute a model trained on this data
over a network (e.g., as a hosted API), the AGPLv3 may require you to
also provide the modified RTA source. **We are providing the modified
training script and intermediate JSONL files in this repository, which
satisfies the source-availability requirement.**

The trained **model weights are a new statistical artifact** and are not
a verbatim copy of the RTA scripts. Whether the model is a "derivative
work" of RTA in the copyright sense is an unsettled legal question; we
make no representation either way. **If you need an AGPL-clean
deployment, retrain the model after removing the `tools/rta` bucket.**

Copyright (C) 2018 info@endgame.com. Used with permission under
AGPLv3 terms.

### 9. promptfoo (promptfoo)

| Field | Value |
|---|---|
| **Pairs** | (varies; AI security category) |
| **Buckets** | `ai/prompt-injection` |
| **Repository** | <https://github.com/promptfoo/promptfoo> |
| **License** | MIT License |
| **Used for** | Red-team TypeScript plugin definitions |

### 10. garak (NVIDIA)

| Field | Value |
|---|---|
| **Pairs** | (varies; AI security category) |
| **Buckets** | `ai/jailbreaking` |
| **Repository** | <https://github.com/NVIDIA/garak> |
| **License** | Apache License 2.0 |
| **Used for** | DAN/probe JSON & TXT resources |

### 11. TheBigPromptLibrary (Resident-Falker)

| Field | Value |
|---|---|
| **Pairs** | (varies; AI security category) |
| **Buckets** | `ai/jailbreaking` |
| **Repository** | <https://github.com/Resident-Falker/TheBigPromptLibrary> |
| **License** | Mixed (per-file; mostly MIT or MPL-2.0) |
| **Used for** | Jailbreak prompts, system prompts, security-focused markdown |

### 12. promptmap (utkusen)

| Field | Value |
|---|---|
| **Pairs** | (varies; AI security category) |
| **Buckets** | `ai/prompt-injection` |
| **Repository** | <https://github.com/utkusen/promptmap> |
| **License** | MIT License |
| **Used for** | Prompt injection YAML rule files |

### 13. PyRIT (Azure)

| Field | Value |
|---|---|
| **Pairs** | (varies; AI security category) |
| **Buckets** | `ai/jailbreaking` |
| **Repository** | <https://github.com/Azure/PyRIT> |
| **License** | MIT License |
| **Used for** | Jailbreak template definitions |

### 14. FuzzyAI (CyberArk)

| Field | Value |
|---|---|
| **Pairs** | (varies; AI security category) |
| **Buckets** | `ai/jailbreaking` |
| **Repository** | <https://github.com/cyberark/FuzzyAI> |
| **License** | Apache License 2.0 |
| **Used for** | Adversarial prompt resources, suffixes, harmful-behaviors CSV |

---

## Synthetic Data

### AttackLM Orchestrator (synthesized)

| Field | Value |
|---|---|
| **Pairs** | 380 |
| **Buckets** | `orchestrator` |
| **Source** | Generated procedurally by `scripts/generate_orchestrator.py` |
| **License** | Same as this repository (MIT) |
| **Used for** | Agent routing decisions across 6 sub-agents |

### Prompt Injection (synthesized)

| Field | Value |
|---|---|
| **Pairs** | 687 |
| **Buckets** | `ai/prompt-injection` |
| **Source** | Generated procedurally by `scripts/generate_prompt_injection.py` |
| **License** | Same as this repository (MIT) |
| **Used for** | Augmenting upstream promptfoo/promptmap data |

---

## Bucket Manifest

The full per-bucket manifest is at `data/datasets/buckets/manifest.json`.
It records which pairs come from which upstream source (via the
`source_file` field). Total: **16,982 pairs across 16 buckets**.

| Bucket | Pairs | Upstream sources |
|---|---:|---|
| collection | 634 | atomic, caldera, metasploit |
| command_and_control | 105 | atomic, caldera, metasploit |
| credential_access | 589 | atomic, caldera, infection_monkey, metasploit |
| defense_evasion | 1,375 | atomic, caldera, metasploit, RTA |
| discovery | 1,846 | atomic, caldera, infection_monkey, metasploit, RTA |
| execution | 767 | atomic, caldera, infection_monkey, metasploit, RTA |
| exfiltration | 173 | atomic, caldera, metasploit |
| lateral_movement | 252 | atomic, caldera, infection_monkey, metasploit |
| persistence | 1,120 | atomic, caldera, infection_monkey, metasploit, RTA |
| privilege_escalation | 537 | atomic, caldera, metasploit |
| orchestrator | 380 | synthetic |
| ai/jailbreaking | 56 | garak, PyRIT, FuzzyAI, TheBigPromptLibrary |
| ai/prompt-injection | 687 | promptfoo, promptmap, synthetic |
| tools/infection_monkey | 36 | guardicore/monkey |
| tools/metasploit | 8,349 | rapid7/metasploit-framework |
| tools/rta | 76 | endgameinc/RTA (AGPLv3) |

---

## Re-Distribution Guidance

If you fork AttackLM and re-distribute:

1. **Keep this `ATTRIBUTION.md` file intact** in any derivative repo.
2. **Keep all upstream LICENSE files** in `data/<source>/LICENSE*`.
3. **Do not remove any source's attribution** when removing its data —
   if you remove the RTA bucket, also remove the RTA entry above.
4. **For Apache 2.0 sources (Caldera, MITRE ATT&CK, garak, FuzzyAI):**
   include a `NOTICE` file in your distribution (template below).
5. **For AGPLv3 (RTA):** if you host a model trained on RTA data over
   a network, provide access to the corresponding source — this
   public GitHub repo satisfies that requirement.
6. **For GPLv3 (Infection Monkey):** the dataset is a transformation
   of upstream code; the GPL may apply to the JSONL files but not
   to model weights learned from them.

### NOTICE template (required by Apache 2.0)

This product includes software developed at
[MITRE Corporation](https://www.mitre.org/) (Caldera, Stockpile) and
[NVIDIA](https://www.nvidia.com/) (garak), and by
[CyberArk](https://www.cyberark.com/) (FuzzyAI).

---

## License of this Repository

The **code, scripts, and orchestration** in this repository are
released under the MIT License. See `LICENSE` at the root.

The **training data** is a derivative work of the upstream sources
above and inherits their respective licenses. The **trained model
weights** are released under the same MIT License as the code, with
the understanding that they are a new statistical artifact learned
from openly licensed material.
