# Gemini Research: Human Sources for Under-Represented MITRE Tactics

**Date generated:** 2026-06-11
**Source:** Gemini (Deep Research prompt, sent 2026-06-11)
**Purpose:** Identify 15-30 high-quality human-authored sources to fill the
Lateral Movement, Collection, Exfiltration, C2, and Impact gaps in AttackLM.
**Status:** Research only — **DO NOT INGEST** without legal review. Multiple
sources have copyright/licensing caveats that need resolution.

---

## Tactic Context (current AttackLM coverage)

| Tactic                                | Records | % of dataset | Status              |
| ------------------------------------- | ------- | ------------ | ------------------- |
| TA0001 Initial Access                 | 2,557   | 10.7%        | balanced            |
| TA0002 Execution                      | 1,555   | 6.5%         | balanced            |
| TA0003 Persistence                    | 1,495   | 6.2%         | balanced            |
| TA0004 Privilege Escalation           | 1,241   | 5.2%         | balanced            |
| TA0005 Defense Evasion                | 1,799   | 7.5%         | balanced            |
| TA0006 Credential Access              | 1,692   | 7.1%         | balanced            |
| TA0007 Discovery                      | 2,139   | 8.9%         | balanced            |
| **TA0008 Lateral Movement**           | **664** | **2.8%**     | **under-represented** |
| **TA0009 Collection**                 | **623** | **2.6%**     | **under-represented** |
| **TA0010 Exfiltration**               | **435** | **1.8%**     | **under-represented** |
| **TA0011 Command and Control**        | **623** | **2.6%**     | **under-represented** |
| **TA0012 Impact**                     | **209** | **0.9%**     | **MOST CRITICAL GAP** |
| TA0040 Prompt Injection               | 8       | 0.0%         | near-zero (bonus)   |
| TA0042 Resource Development           | 10      | 0.0%         | near-zero (bonus)   |
| TA0043 Reconnaissance                 | 2,160   | 9.0%         | balanced            |

---

## 17 Sources (Gemini Output)

### Source #1: Stratus Red Team (Datadog)
- **URL**: https://github.com/DataDog/stratus-red-team
- **Organization / Author**: Datadog
- **License**: Apache License 2.0
- **Type**: repo
- **Coverage**: Exfiltration (TA0010), Collection (TA0009), Impact (TA0012)
- **Volume estimate**: 50-100 records extractable
- **MITRE status**: mapped
- **Format**: multi (Go code, markdown documentation, YAML definitions)
- **Ingestion effort**: scriptable (clean YAML files map directly to ATT&CK)
- **Quality assessment**: gold — Extremely granular, cloud-native execution
  paths (AWS/GCP/Azure) with exact CLI and API calls used for exfil and
  collection.
- **Why this helps the gap**: Directly provides deterministic execution
  chains for cloud storage exfiltration (T1530) and cloud collection.
- **Risk or caveat**: Mostly cloud-focused; you will still need on-prem AD
  data for a balanced dataset.

### Source #2: LOLBAS (Living Off The Land Binaries and Scripts)
- **URL**: https://github.com/LOLBAS-Project/LOLBAS
- **Organization / Author**: LOLBAS Project Community
- **License**: MIT
- **Type**: dataset / repo
- **Coverage**: Command and Control (TA0011), Lateral Movement (TA0008),
  Collection (TA0009)
- **Volume estimate**: 150-300 records extractable
- **MITRE status**: mapped
- **Format**: yaml (highly structured)
- **Ingestion effort**: trivial (YAML directly maps to ATT&CK with execution
  commands)
- **Quality assessment**: gold — Canonical source for living-off-the-land
  techniques, providing direct command-line execution examples for C2
  tunneling and execution.
- **Why this helps the gap**: Massively boosts T1105 (Ingress Tool Transfer)
  and proxy/C2 execution primitives.
- **Risk or caveat**: Provides the commands and descriptions, but lacks
  narrative context; you will need to stitch these into "chains" if your
  model requires longer context.

### Source #3: Splunk Security Content (ASC)
- **URL**: https://github.com/splunk/security_content
- **Organization / Author**: Splunk Threat Research Team
- **License**: Apache License 2.0
- **Type**: repo / dataset
- **Coverage**: ALL (Heavy on Impact, Exfiltration, Lateral Movement)
- **Volume estimate**: 1,000-1,500 records extractable
- **MITRE status**: mapped
- **Format**: yaml
- **Ingestion effort**: scriptable
- **Quality assessment**: gold — While these are detection rules, the
  `known_false_positives` and `how_to_implement` fields often contain rich,
  human-authored descriptions of the offensive techniques.
- **Why this helps the gap**: Provides deep context on ransomware behavior
  (Impact) and lateral movement tool execution (WMI, PtH) from a
  detection-engineering perspective.
- **Risk or caveat**: Requires careful parsing to invert the perspective from
  "how to detect" to "how it is executed," though the descriptions are
  largely technique-focused.

### Source #4: The DFIR Report
- **URL**: https://thedfirreport.com/
- **Organization / Author**: The DFIR Report Team
- **License**: Copyright (All Rights Reserved) - *See Caveat*
- **Type**: blog / sandbox-reports
- **Coverage**: Impact (Ransomware), Lateral Movement, Command and Control,
  Exfiltration
- **Volume estimate**: 100-200 complex chains extractable
- **MITRE status**: mapped (they explicitly map chains at the end of posts)
- **Format**: html
- **Ingestion effort**: needs-scraping
- **Quality assessment**: gold — The absolute best public source for
  end-to-end, human-authored attack chains involving ransomware and AD
  lateral movement.
- **Why this helps the gap**: Provides the exact timing, tooling, and
  sequence for Cobalt Strike C2 -> BloodHound -> RDP lateral movement ->
  Exfil -> Ransomware Impact.
- **Risk or caveat**: **LICENSING.** Content is not explicitly open-source.
  Scraping for non-commercial ML training may fall under fair use, but
  consult legal before redistributing the raw text.

### Source #5: MITRE Center for Threat-Informed Defense (CTID) Adversary Emulation Library
- **URL**: https://github.com/center-for-threat-informed-defense/adversary_emulation_library
- **Organization / Author**: MITRE Engenuity
- **License**: Apache License 2.0
- **Type**: emulation-plan
- **Coverage**: Lateral Movement, Exfiltration, C2, Impact
- **Volume estimate**: 200-400 records extractable
- **MITRE status**: mapped
- **Format**: yaml / markdown / stix
- **Ingestion effort**: scriptable
- **Quality assessment**: gold — Canonical, peer-reviewed emulation plans
  for FIN6, menuPass, APT29, etc., containing specific CLI commands and
  payloads.
- **Why this helps the gap**: Deeply covers multi-stage C2 channels, data
  staging, and lateral movement exactly as APTs perform them.
- **Risk or caveat**: The repository structure varies slightly between older
  and newer emulation plans, requiring custom parsers per threat actor.

### Source #6: HackTricks (Carlos Polop)
- **URL**: https://github.com/carlospolop/hacktricks
- **Organization / Author**: Carlos Polop
- **License**: GPL-3.0
- **Type**: cheat-sheet / repo
- **Coverage**: Lateral Movement, Collection, C2, Resource Development
- **Volume estimate**: 500-1,000 records extractable
- **MITRE status**: unmapped (some partial references)
- **Format**: markdown
- **Ingestion effort**: needs-MITRE-labeling / needs-parsing
- **Quality assessment**: silver — Massive volume of raw offensive commands
  and techniques, but requires significant NLP or LLM pre-processing to map
  to MITRE.
- **Why this helps the gap**: Unmatched depth on protocol-specific lateral
  movement (WinRM, DCOM, SSH hijacking) and credential collection from
  files/browsers.
- **Risk or caveat**: Unstructured markdown. You will need to build a
  pipeline to chunk the markdown by headers and use a classifier to map
  them to MITRE techniques.

### Source #7: Sliver C2 Framework Wiki
- **URL**: https://github.com/BishopFox/sliver/wiki
- **Organization / Author**: Bishop Fox
- **License**: GPL-3.0
- **Type**: docs
- **Coverage**: Command and Control (TA0011), Lateral Movement (TA0008)
- **Volume estimate**: 50-100 records extractable
- **MITRE status**: unmapped
- **Format**: markdown
- **Ingestion effort**: scriptable / needs-MITRE-labeling
- **Quality assessment**: silver — Excellent technical documentation on how
  modern C2 frameworks implement DNS, mTLS, and WireGuard tunneling.
- **Why this helps the gap**: Direct documentation of T1572 (Protocol
  Tunneling) and T1090 (Proxy) from the tool developers.
- **Risk or caveat**: Will require manual or LLM-assisted mapping of the
  wiki pages to specific ATT&CK C2 sub-techniques.

### Source #8: PurpleSharp
- **URL**: https://github.com/mvelazc0/PurpleSharp
- **Organization / Author**: Mauricio Velazco
- **License**: GPL-3.0
- **Type**: repo / dataset
- **Coverage**: Lateral Movement (TA0008), Collection (TA0009)
- **Volume estimate**: 100-150 records extractable
- **MITRE status**: mapped
- **Format**: json / c#
- **Ingestion effort**: trivial (JSON simulation playbooks are pre-mapped)
- **Quality assessment**: gold — Purpose-built adversary simulation tool with
  highly structured JSON playbooks that map directly to ATT&CK.
- **Why this helps the gap**: High-fidelity Active Directory lateral
  movement techniques (Kerberoasting, DCSync, WMI execution) defined as
  structured data.
- **Risk or caveat**: Volume is relatively low, but data quality and
  structure are exceptional.

### Source #9: GhostPack Documentation & Source (harmj0y)
- **URL**: https://github.com/GhostPack
- **Organization / Author**: Will Schroeder / SpecterOps
- **License**: BSD 3-Clause
- **Type**: repo / docs
- **Coverage**: Collection (TA0009), Lateral Movement (TA0008)
- **Volume estimate**: 50-150 records extractable (from readmes and code
  comments)
- **MITRE status**: unmapped
- **Format**: markdown / text
- **Ingestion effort**: needs-parsing / needs-MITRE-labeling
- **Quality assessment**: silver — The gold standard for AD offensive
  tooling (Rubeus, Seatbelt, SharpUp), but requires extraction from
  READMEs.
- **Why this helps the gap**: Perfect coverage for T1550 (Use Alternate
  Auth Material) and T1005 (Data from Local System - Seatbelt).
- **Risk or caveat**: Requires parsing markdown. You may want to ingest
  just the READMEs and wiki pages rather than the source code.

### Source #10: Leonidas (WithSecure / F-Secure)
- **URL**: https://github.com/WithSecureLabs/leonidas
- **Organization / Author**: WithSecure Labs
- **License**: Apache License 2.0
- **Type**: repo
- **Coverage**: Exfiltration (TA0010), Collection (TA0009), Impact (TA0012)
- **Volume estimate**: 60-100 records extractable
- **MITRE status**: mapped
- **Format**: yaml / json
- **Ingestion effort**: trivial
- **Quality assessment**: gold — Structured cloud attack cases (AWS) mapped
  to MITRE, providing the exact API calls used by attackers.
- **Why this helps the gap**: Excellent for Cloud Account (T1078.004)
  manipulation and Cloud Storage Object (T1530) exfiltration.
- **Risk or caveat**: Narrowly focused on AWS; less coverage for Azure or
  GCP.

### Source #11: SigmaHQ Rules (Offensive Context Extraction)
- **URL**: https://github.com/SigmaHQ/sigma
- **Organization / Author**: Florian Roth / SigmaHQ Community
- **License**: Detection Rule License (DRL) 1.1 / MIT
- **Type**: repo
- **Coverage**: ALL (Excellent for Lateral Movement and Impact)
- **Volume estimate**: 2,000-3,000 records extractable
- **MITRE status**: mapped
- **Format**: yaml
- **Ingestion effort**: scriptable
- **Quality assessment**: gold — By extracting the `description`, `logsource`,
  and `selection` (strings/commands), you can automatically generate
  offensive technique descriptions.
- **Why this helps the gap**: Contains hundreds of rules specifically
  detailing wiper malware (Impact) and lateral movement commands.
- **Risk or caveat**: Requires a transformation script to translate
  "detects command X" into "adversary executes command X."

### Source #12: WADComs (Windows Active Directory Commands)
- **URL**: https://github.com/WADComs/WADComs.github.io
- **Organization / Author**: WADComs community
- **License**: MIT
- **Type**: repo / cheat-sheet
- **Coverage**: Lateral Movement (TA0008), Collection (TA0009)
- **Volume estimate**: 150-250 records extractable
- **MITRE status**: mapped (partially, mostly mapped by tactic/tool)
- **Format**: markdown (Jekyll frontmatter)
- **Ingestion effort**: scriptable
- **Quality assessment**: silver — Highly structured cheatsheet specifically
  for AD attacks.
- **Why this helps the gap**: Fills the specific AD lateral movement gap
  (Pass the Hash, Overpass the Hash, DCOM).
- **Risk or caveat**: Some commands are highly specific to tools
  (e.g., Impacket) rather than the underlying API.

### Source #13: CISA Cybersecurity Advisories (AA series STIX/Markdown)
- **URL**: https://github.com/cisagov/CSAF (and CISA's main site for AA
  alerts)
- **Organization / Author**: CISA (Cybersecurity and Infrastructure Security
  Agency)
- **License**: Public Domain
- **Type**: reports / stix
- **Coverage**: Impact (TA0012), Exfiltration (TA0010), C2 (TA0011)
- **Volume estimate**: 500-1,000 records extractable
- **MITRE status**: mapped
- **Format**: stix / html / pdf
- **Ingestion effort**: scriptable (if using STIX) / needs-scraping (if HTML)
- **Quality assessment**: gold — Definitive, human-analyzed reports on
  ransomware operators and APTs, explicitly mapped to ATT&CK.
- **Why this helps the gap**: The single best source for Impact (TA0012)
  data, detailing exactly how ransomware encrypts files, inhibits recovery,
  and exfiltrates data.
- **Risk or caveat**: The STIX feeds are sometimes less descriptive than the
  human-readable PDF/HTML reports, so scraping the HTML might yield better
  NLP training text.

### Source #14: Greshake LLM Security Repo (Prompt Injection)
- **URL**: https://github.com/greshake/llm-security
- **Organization / Author**: Kai Greshake et al.
- **License**: MIT
- **Type**: repo / dataset
- **Coverage**: Prompt Injection (TA0040)
- **Volume estimate**: 50-100 records extractable
- **MITRE status**: unmapped (MITRE ATLAS mappings needed)
- **Format**: markdown / text
- **Ingestion effort**: scriptable / needs-MITRE-labeling
- **Quality assessment**: gold — The foundational research repository for
  Indirect Prompt Injection.
- **Why this helps the gap**: Directly addresses the Prompt Injection
  (TA0040) / MITRE ATLAS gap with real-world exploit payloads.
- **Risk or caveat**: You will need to manually map these to OWASP LLM Top
  10 or MITRE ATLAS framework identifiers.

### Source #15: OWASP Top 10 for LLM Applications
- **URL**: https://github.com/OWASP/www-project-top-10-for-large-language-model-applications
- **Organization / Author**: OWASP Foundation
- **License**: CC-BY-4.0
- **Type**: docs
- **Coverage**: Prompt Injection (TA0040)
- **Volume estimate**: 20-50 records extractable (dense concepts)
- **MITRE status**: unmapped (maps to OWASP LLM)
- **Format**: markdown
- **Ingestion effort**: trivial
- **Quality assessment**: gold — Canonical definitions for LLM01 through
  LLM10.
- **Why this helps the gap**: Provides the baseline human-authored
  descriptions, impact statements, and examples of Prompt Injection,
  Overreliance, and Agency issues.
- **Risk or caveat**: Volume is low, but the semantic quality is extremely
  high.

### Source #16: Kusto (KQL) Hunt Queries (Azure Sentinel)
- **URL**: https://github.com/Azure/Azure-Sentinel/tree/master/Hunting%20Queries
- **Organization / Author**: Microsoft Threat Intelligence
- **License**: MIT
- **Type**: repo
- **Coverage**: Exfiltration, Lateral Movement, Collection
- **Volume estimate**: 400-600 records extractable
- **MITRE status**: mapped
- **Format**: yaml / kql
- **Ingestion effort**: scriptable
- **Quality assessment**: silver — Similar to Sigma, the metadata in these
  hunting queries provides rich descriptions of cloud and endpoint attacker
  behavior.
- **Why this helps the gap**: Exceptional for Cloud Collection and
  Exfiltration techniques, authored by Microsoft defenders.
- **Risk or caveat**: Requires stripping the KQL logic to extract the
  English descriptions and mappings.

### Source #17: Red Canary Threat Detection Report (TDR) Data
- **URL**: https://redcanary.com/threat-detection-report/ (and associated
  GitHub repos/blogs)
- **Organization / Author**: Red Canary
- **License**: Standard Copyright (Scrape with caution) / MIT for some repos
- **Type**: reports
- **Coverage**: C2, Lateral Movement, Collection
- **Volume estimate**: 50-100 highly detailed records
- **MITRE status**: mapped
- **Format**: html / pdf
- **Ingestion effort**: needs-scraping
- **Quality assessment**: gold — Some of the most accurate, deeply technical
  descriptions of specific techniques (like WMI lateral movement) available
  publicly.
- **Why this helps the gap**: Provides deep-dive narrative explanations of
  *how* a technique works, not just a command to run it.
- **Risk or caveat**: Licensing. Best to use as an unstructured data source
  for embedding/training, rather than 1:1 redistribution.

---

## Ranked Top 5 Recommendations (Gemini's pick)

1. **MITRE CTID Adversary Emulation Library** — Highest fidelity,
   pre-mapped to your exact schema needs, permissive license, covers all
   priority gaps (C2, Exfil, Lateral).
2. **Stratus Red Team (Datadog)** — Best-in-class for cloud-specific
   Collection and Exfiltration. Trivial to parse (YAML) and pre-mapped.
3. **LOLBAS** — Critical for expanding the Command and Control / Ingress
   tool transfer data. Extremely high signal-to-noise ratio and scriptable
   YAML.
4. **PurpleSharp** — Highly structured JSON playbooks specifically covering
   the Active Directory Lateral Movement gap with zero manual labeling
   required.
5. **SigmaHQ / Splunk ASC** — Massive volume. Translating detection metadata
   into offensive descriptions provides thousands of structurally sound,
   MITRE-mapped records.

---

## Combined Coverage Matrix

| Source                          | TA0008 (Lateral) | TA0009 (Collection) | TA0010 (Exfil) | TA0011 (C2) | TA0012 (Impact) | TA0040 (LLM) |
| ------------------------------- | :--------------: | :-----------------: | :------------: | :---------: | :-------------: | :----------: |
| Stratus Red Team                |                  | X                   | X              |             | X               |              |
| LOLBAS                          | X                | X                   |                | X           |                 |              |
| Splunk ASC                      | X                | X                   | X              | X           | X               |              |
| The DFIR Report                 | X                | X                   | X              | X           | X               |              |
| MITRE CTID                      | X                | X                   | X              | X           | X               |              |
| HackTricks                      | X                | X                   |                | X           |                 |              |
| Sliver Wiki                     | X                |                     |                | X           |                 |              |
| PurpleSharp                     | X                | X                   |                |             |                 |              |
| GhostPack                       | X                | X                   |                |             |                 |              |
| Leonidas                        | X                | X                   | X              |             | X               |              |
| SigmaHQ                         | X                | X                   | X              | X           | X               |              |
| WADComs                         | X                | X                   |                |             |                 |              |
| CISA AA Alerts                  |                  |                     | X              | X           | X               |              |
| Greshake Repo                   |                  |                     |                |             |                 | X            |
| OWASP LLM                       |                  |                     |                |             |                 | X            |
| Azure Sentinel                  | X                | X                   | X              |             |                 |              |
| Red Canary TDR                  | X                | X                   |                | X           |                 |              |

---

## Licensing Summary

- **MITRE CTID Library**: Apache 2.0 (Redistribution safe)
- **Stratus Red Team**: Apache 2.0 (Redistribution safe)
- **LOLBAS**: MIT (Redistribution safe)
- **PurpleSharp**: GPL-3.0 (Redistribution safe, provided your dataset
  adheres to GPL requirements if packaged as code, though generally safe
  for ML training data).
- **SigmaHQ**: Detection Rule License (DRL) 1.1 / MIT (Redistribution safe
  with attribution).

**Conclusion:** The Top 5 are inherently open-source and structurally safe
for automated ingestion and dataset redistribution.

---

## Scraping / Dataset Risk Assessment

1. **The DFIR Report & Red Canary TDR**: These are traditional copyrighted
   blogs. While scraping them for internal ML training usually falls under
   fair use in the US, redistributing the exact scraped text in an open
   dataset (like HuggingFace) is a copyright violation risk.
   *Mitigation*: Use an LLM to summarize or extract the TTP chains into an
   abstract JSON structure rather than hosting the raw article text.

2. **CISA AA Alerts**: Public domain. You can scrape and redistribute the
   HTML/PDFs freely, but the format varies wildly. Writing a universal
   parser will be brittle. Rely on their STIX JSON feeds where possible.

3. **HackTricks**: It is a Git repository, so you do not need to web scrape,
   but parsing the unstructured markdown headers into clean, isolated
   technique descriptions is technically challenging and prone to noise
   (e.g., capturing raw base64 payloads as text).

---

## Next-Step Ingestion Plan (per Gemini — DO NOT START UNTIL USER APPROVES)

1. **Phase 1: The YAML/JSON Easy Wins (1-2 Days)**
   - Clone `Datadog/stratus-red-team`, `WithSecureLabs/leonidas`, `LOLBAS`,
     and `PurpleSharp`.
   - Write a Python script using `pyyaml` and `json` to extract the
     technique ID, name, description, and execution commands.
   - Output directly to your `AttackLM` `.jsonl` format.

2. **Phase 2: The MITRE / Emulation Library (2-3 Days)**
   - Clone
     `center-for-threat-informed-defense/adversary_emulation_library`.
   - Parse the structured YAML/STIX files in each threat actor directory.
     Extract the `description` and `command` fields.

3. **Phase 3: Detection Reversal (Sigma/Splunk) (3-5 Days)**
   - Clone `SigmaHQ` and `Splunk/security_content`.
   - Write a script that filters for files containing `tags` mapped to
     your 5 missing tactics (e.g., `attack.t1021.001`).
   - Extract the `description` and `falsepositives` fields, prepending
     them with context (e.g., "Attackers utilize this technique by...").

4. **Phase 4: Addressing AI/LLM Bonus Tactics (1 Day)**
   - Clone `OWASP/www-project-top-10-for-large-language-model-applications`
     and `greshake/llm-security`.
   - Manually map the OWASP markdown files into 10 high-quality JSONL
     records.

5. **Phase 5: The Hard Scraping (As needed)**
   - Only if volume is still low: Implement a BeautifulSoup scraper for
     CISA AA alerts or The DFIR report, passing the HTML through a
     lightweight LLM pipeline (like local LLaMA or Gemini) strictly to
     format the output into your JSON schema. **We have massive copyright
     issues to work out. Do not start working on that Gemini list.**

---

## Important Notes

- **All 17 sources are research output only.** No cloning, no scraping, no
  ingestion has been performed.
- **The "next-step ingestion plan" is on hold** until copyright/licensing
  concerns with sources #4 (DFIR Report), #6 (HackTricks), #11 (SigmaHQ
  DRL), #13 (CISA), and #17 (Red Canary TDR) are resolved by the user.
- The architect triage report for the existing metasploit bucket
  (`data/datasets/buckets/tools/metasploit/TRIAGE.md`) is a separate task
  and is ready for the user's review.

---

## References

- Gemini research prompt: `/tmp/gemini-research-prompt.md` (also at
  `data/datasets/buckets/GEMINI_RESEARCH_PROMPT.md` if promoted to project
  docs)
- Tactic coverage: `data/audit_report.json` and `scripts/audit_dataset.py`
  output
- Metasploit triage: `data/datasets/buckets/tools/metasploit/TRIAGE.md`
