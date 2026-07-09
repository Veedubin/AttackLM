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
  DRL-1.1 and BSD-3-Clause requirements below** — they have attribution
  obligations.

---

## Training Data Sources (24,652 pairs across 21 buckets)

### 1. Atomic Red Team (redcanaryco)

| Field | Value |
|---|---|
| **Pairs** | 1,115 |
| **Buckets** | 9 MITRE tactics + tools/atomic |
| **Repository** | <https://github.com/redcanaryco/atomic-red-team> |
| **License** | MIT License |
| **Source file** | `data/atomic-red-team/LICENSE.txt` |
| **Used for** | Atomic test triples — exact commands, expected artifacts, cleanup for ~700 ATT&CK techniques |

Copyright (c) Red Canary, LLC. All rights reserved.

### 2. MITRE Caldera — Stockpile (mitre)

| Field | Value |
|---|---|
| **Pairs** | 390 |
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
| **Pairs** | 3,132 |
| **Repository** | <https://github.com/SigmaHQ/sigma> |
| **License** | Detection Rule License (DRL) 1.1 for rules; public domain for spec |
| **Source file** | `data/sigma/LICENSE` |
| **Used for** | Detection rule-based triples |

DRL 1.1 permits use, modification, and distribution with attribution.

### 6. Metasploit Framework (rapid7)

| Field | Value |
|---|---|
| **Pairs** | 13,997 |
| **Buckets** | `tools/metasploit` (15 module categories consolidated) + re-routed to 5 MITRE tactics |
| **Repository** | <https://github.com/rapid7/metasploit-framework> |
| **License** | BSD 3-Clause License |
| **Source files** | `data/metasploit-framework/LICENSE`, `data/metasploit-framework/COPYING` |
| **Used for** | Module descriptions, options, references, payloads — converted to "what this module does" triples |

Copyright (c) 2006-2026, Rapid7, Inc. All rights reserved.
Redistribution permitted under BSD-3-Clause terms.

### 7. Elastic Security Rules (Elastic)

| Field | Value |
|---|---|
| **Pairs** | 1,908 |
| **Repository** | <https://github.com/elastic/detection-rules> |
| **License** | Elastic License 2.0 |
| **Used for** | EQL/KQL rule-based triples |

### 8. Splunk Security Content (Splunk)

| Field | Value |
|---|---|
| **Pairs** | 2,114 |
| **Repository** | <https://github.com/splunk/security_content> |
| **License** | Apache License 2.0 |
| **Used for** | SPL search-based triples |

### 9. Mordor (OTRF)

| Field | Value |
|---|---|
| **Pairs** | 339 |
| **Repository** | <https://github.com/OTRF/Security-Datasets> |
| **License** | Apache License 2.0 |
| **Used for** | Event log scenario triples |

### 10. ThreatHunter Playbook (OTRF)

| Field | Value |
|---|---|
| **Pairs** | 27 |
| **Repository** | <https://github.com/OTRF/ThreatHunter-Playbook> |
| **License** | Apache License 2.0 |
| **Used for** | Hunting playbook triples |

### 11. NIST IR Guidelines (NIST)

| Field | Value |
|---|---|
| **Pairs** | 168 |
| **Source** | NIST SP 800-61r3 |
| **License** | Public Domain |
| **Used for** | Incident response procedure triples |

### 12. promptfoo (promptfoo)

| Field | Value |
|---|---|
| **Pairs** | 33 |
| **Buckets** | `ai/prompt-injection` |
| **Repository** | <https://github.com/promptfoo/promptfoo> |
| **License** | MIT License |
| **Used for** | Red-team TypeScript plugin definitions |

### 13. garak (NVIDIA)

| Field | Value |
|---|---|
| **Pairs** | 50 |
| **Buckets** | `ai/jailbreaking` |
| **Repository** | <https://github.com/NVIDIA/garak> |
| **License** | Apache License 2.0 |
| **Used for** | DAN/probe JSON & TXT resources |

### 14. promptmap (utkusen)

| Field | Value |
|---|---|
| **Pairs** | 30 |
| **Buckets** | `ai/prompt-injection` |
| **Repository** | <https://github.com/utkusen/promptmap> |
| **License** | MIT License |
| **Used for** | Prompt injection YAML rule files |

### 15. PyRIT (Azure)

| Field | Value |
|---|---|
| **Pairs** | 0 (reserved) |
| **Buckets** | `ai/jailbreaking` |
| **Repository** | <https://github.com/Azure/PyRIT> |
| **License** | MIT License |
| **Used for** | Jailbreak template definitions |

### 16. FuzzyAI (CyberArk)

| Field | Value |
|---|---|
| **Pairs** | 0 (reserved) |
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

### LLM-generated (synthesized)

| Field | Value |
|---|---|
| **Pairs** | 937 |
| **Buckets** | cloud, ics, social_engineering, wireless |
| **Source** | Generated via `scripts/llm_generate_wrapper.py` |
| **License** | Same as this repository (MIT) |
| **Used for** | High-quality synthetic triples for scarce domains |

---

## Bucket Manifest

The full per-bucket manifest is at `data/datasets/buckets/manifest.json`.
It records which pairs come from which upstream source (via the
`source_file` field). Total: **24,652 pairs across 21 buckets**.

| Bucket | Pairs | Upstream sources |
|---|---|---:|
| collection | 634 | atomic, metasploit, stockpile |
| command_and_control | 105 | atomic, metasploit, stockpile |
| credential_access | 589 | metasploit |
| defense_evasion | 1,375 | metasploit |
| discovery | 1,846 | atomic, metasploit, stockpile, atlas-arsenal |
| execution | 767 | atomic, metasploit, stockpile |
| exfiltration | 53 | atomic, stockpile |
| lateral_movement | 252 | metasploit |
| persistence | 1,120 | atomic, metasploit, stockpile |
| privilege_escalation | 537 | metasploit |
| orchestrator | 380 | synthetic |
| ai/jailbreaking | 50 | garak |
| ai/prompt-injection | 687 | promptfoo, promptmap, synthetic |
| tools/metasploit | 13,997 | rapid7/metasploit-framework |
| defensive/detection_engineering | 7,154 | sigma, elastic, splunk |
| defensive/threat_hunting | 366 | mordor, threathunter |
| defensive/incident_response | 168 | nist-ir |
| synthetic/domain_specific | 937 | llm-generated |

---

## Re-Distribution Guidance

If you fork AttackLM and re-distribute:

1. **Keep this `ATTRIBUTION.md` file intact** in any derivative repo.
2. **Keep all upstream LICENSE files** in `data/<source>/LICENSE*`.
3. **Do not remove any source's attribution** when removing its data.
4. **For Apache 2.0 sources (Caldera, garak, Mordor, ThreatHunter-Playbook, Splunk):**
   include a `NOTICE` file in your distribution (template below).
5. **For BSD-3-Clause (Metasploit):** preserve the copyright notice and
   license text in derivative works per BSD §1.
6. **For DRL-1.1 (Sigma):** preserve attribution to Sigma rule authors
   per the Detection Rule License.

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
