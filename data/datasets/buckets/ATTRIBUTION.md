# Bucket Data — Per-Source Attribution

Each bucket in this directory contains training pairs derived from
upstream open-source security projects. Each row carries a `source`
and `license` field (added by `scripts/augment_attribution.py`).

For the full per-source attribution, license analysis, and
re-distribution guidance, see
[**/ATTRIBUTION.md**](../../../ATTRIBUTION.md) at the repository root.

## Bucket manifest

The full machine-readable manifest is `manifest.json` in this directory.
It records each bucket's:
- name and display name
- MITRE tactic ID (TA0001-TA0011, TA0040 for AI)
- pair count
- source data file

## Per-bucket source mix

| Bucket | Pairs | Source mix (dominant) |
|---|---:|---|
| collection | 634 | atomic-red-team + caldera + metasploit |
| command_and_control | 105 | atomic-red-team + caldera + metasploit |
| credential_access | 589 | atomic-red-team + caldera + metasploit |
| defense_evasion | 1,375 | atomic-red-team + caldera + metasploit |
| discovery | 1,846 | atomic-red-team + caldera + metasploit |
| execution | 767 | atomic-red-team + caldera + metasploit |
| exfiltration | 173 | atomic-red-team + caldera + metasploit |
| lateral_movement | 252 | atomic-red-team + caldera + metasploit |
| persistence | 1,120 | atomic-red-team + caldera + metasploit |
| privilege_escalation | 537 | atomic-red-team + caldera + metasploit |
| orchestrator | 380 | synthetic (MIT) |
| ai-models/jailbreaking | 56 | garak + pyrit + fuzzyai + bigpromptlib |
| ai-models/prompt-injection | 687 | promptfoo + promptmap + synthetic |
| tools/infection_monkey | 36 | guardicore/monkey (GPL-3.0) |
| tools/metasploit | 8,349 | rapid7/metasploit-framework (BSD-3-Clause) |
| tools/rta | 76 | endgameinc/RTA (AGPL-3.0) |

## Adding new buckets

See [CONTRIBUTING.md](../../../CONTRIBUTING.md) §"Adding a new bucket".
