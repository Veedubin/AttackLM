# AttackLM v0.5.0 — Blue Team Data Sources & Team Presets — Design

> **Author**: boomerang-orchestrator (kimi-k2.6)
> **Date**: 2026-06-24
> **Status**: Architectural plan — ready for implementation
> **Based on**: Existing extractor patterns (extract_atomic_red_team_to_jsonl.py, parse_metasploit_to_jsonl.py), balance_buckets.py, init_pipeline.py

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [New Data Sources](#2-new-data-sources)
3. [Bucket Structure](#3-bucket-structure)
4. [Extractor Designs](#4-extractor-designs)
5. [Team Preset System](#5-team-preset-system)
6. [Balance Script Update](#6-balance-script-update)
7. [Init Pipeline Update](#7-init-pipeline-update)
8. [CLI Updates](#8-cli-updates)
9. [Attribution & License Design](#9-attribution--license-design)
10. [Manifest Update](#10-manifest-update)
11. [Testing Strategy](#11-testing-strategy)
12. [Implementation Order](#12-implementation-order)

---

## 1. Executive Summary

AttackLM v0.4.1 has 16,015 training pairs — all offensive (red team). This design adds 6 defensive/blue-team data sources (~5,550 new records) and a team preset system that lets users train red, purple, or blue team specialists at different offensive/defensive ratios.

**New total**: ~21,565 records (16,015 existing + ~5,550 new)

**Team presets**:
| Preset | Offensive | Defensive | Orchestrator | Total |
|--------|-----------|-----------|-------------|-------|
| Red Team | 90% | 10% | 380 | ~16,000 |
| Purple Team | 50% | 50% | 380 | ~16,000 |
| Blue Team | 10% | 90% | 380 | ~16,000 |

**All licenses are permissive** — no AGPL/GPL issues. DRL-1.1 already handled by existing attribution system.

---

## 2. New Data Sources

| # | Source | Repo | Records (est.) | License | Bucket |
|---|--------|------|----------------|---------|--------|
| 1 | SigmaHQ/sigma | https://github.com/SigmaHQ/sigma | ~3,000 | DRL-1.1 | defensive/detection_engineering |
| 2 | OTRF/Security-Datasets | https://github.com/OTRF/Security-Datasets | ~200 | Apache 2.0 | defensive/threat_hunting |
| 3 | OTRF/ThreatHunter-Playbook | https://github.com/OTRF/ThreatHunter-Playbook | ~150 | Apache 2.0 | defensive/threat_hunting |
| 4 | Elastic/detection-rules | https://github.com/elastic/detection-rules | ~1,200 | Elastic License 2.0 | defensive/detection_engineering |
| 5 | Splunk/security_content | https://github.com/splunk/security_content | ~800 | Apache 2.0 | defensive/detection_engineering |
| 6 | NIST SP 800-61r3 | https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r3.pdf | ~200 | Public Domain | defensive/incident_response |

**License compatibility check**:
- DRL-1.1: Already handled by AttackLM's attribution system (used for Sigma rules in existing dataset)
- Apache 2.0: Compatible with MIT. Requires NOTICE preservation.
- Elastic License 2.0: Non-copyleft source-available. Permits use, modification, distribution. Compatible for training data.
- Public Domain: Fully compatible, no restrictions.

---

## 3. Bucket Structure

New `defensive/` hierarchy under the existing per-source layout:

```
data/datasets/buckets/sources/
├── sigma-hq/
│   └── defensive/detection_engineering/data.jsonl
├── mordor/
│   └── defensive/threat_hunting/data.jsonl
├── threathunter-playbook/
│   └── defensive/threat_hunting/data.jsonl
├── elastic-rules/
│   └── defensive/detection_engineering/data.jsonl
├── splunk-content/
│   └── defensive/detection_engineering/data.jsonl
└── nist-ir/
    └── defensive/incident_response/data.jsonl
```

Each source also gets `SOURCE.md` and `LICENSE.md` at its root.

**Bucket categories for balance_buckets.py** — add 3 new categories:
```python
"defensive_detection",   # Sigma + Elastic + Splunk rules
"defensive_hunting",     # Mordor + ThreatHunter-Playbook
"defensive_ir",          # NIST SP 800-61
```

---

## 4. Extractor Designs

All extractors follow the existing pattern: argparse CLI, YAML/JSON parsing, system message, user/assistant pairs, JSONL output, per-record attribution fields.

### 4.1 SigmaHQ/sigma Extractor

**File**: `scripts/extract_sigma_defensive.py`

**Input**: `data/sigma/rules/` — YAML files with `title`, `description`, `detection`, `falsepositives`, `tags`

**System message**:
```
You are a Detection Engineering specialist. Write precise Sigma detection rules
mapped to MITRE ATT&CK with detection logic, false positive analysis, and
deployment guidance.
```

**Pair generation** (1 pair per rule):
- **User**: "Write a Sigma detection rule for {title}. Map it to MITRE ATT&CK {technique_id} ({technique_name}). Include detection logic, false positive scenarios, and SIEM deployment notes."
- **Assistant**: Structured explanation of the rule: title, description, detection logic (selection + condition), false positives, MITRE mapping, severity, references

**MITRE extraction**: Parse `tags` field for `attack.t*` patterns

**Output**: `data/datasets/buckets/sources/sigma-hq/defensive/detection_engineering/data.jsonl`

**Attribution fields**:
```json
{
  "source": "sigma-hq",
  "source_uri": "https://github.com/SigmaHQ/sigma",
  "license": "DRL-1.1",
  "license_uri": "https://github.com/SigmaHQ/sigma/blob/master/LICENSE",
  "rights_contact": "SigmaHQ",
  "attribution_text": "Detection Rule License (DRL) 1.1 — ..."
}
```

**Estimated**: ~3,000 records, ~400 lines Python

### 4.2 OTRF/Security-Datasets (Mordor) Extractor

**File**: `scripts/extract_mordor.py`

**Input**: `data/mordor/` — JSON event logs + metadata YAML files organized by platform/group/technique

**System message**:
```
You are a Threat Hunting specialist. Analyze security event logs, identify
adversary techniques, extract indicators of compromise, and provide detection
queries for SIEM platforms.
```

**Pair generation** (2-3 pairs per scenario):
- **Pair 1 — Technique identification**: "Analyze these {platform} event logs. What MITRE ATT&CK technique is being executed? What specific indicators confirm your assessment?"
- **Pair 2 — Detection query**: "Write a {platform} detection query for the technique you identified. Include data sources, field mappings, and expected false positive scenarios."
- **Pair 3 — Hunting methodology** (if metadata has hunting steps): "Describe a threat hunting methodology for detecting this technique at scale in a {platform} environment."

**MITRE extraction**: From metadata YAML `attack_technique` field

**Output**: `data/datasets/buckets/sources/mordor/defensive/threat_hunting/data.jsonl`

**Attribution fields**:
```json
{
  "source": "mordor",
  "source_uri": "https://github.com/OTRF/Security-Datasets",
  "license": "Apache-2.0",
  "license_uri": "https://github.com/OTRF/Security-Datasets/blob/master/LICENSE",
  "rights_contact": "Open Threat Research (OTRF)",
  "attribution_text": "Copyright (c) Open Threat Research. Licensed under Apache 2.0."
}
```

**Estimated**: ~200 scenarios × 2-3 pairs = ~500 records, ~350 lines Python

### 4.3 OTRF/ThreatHunter-Playbook Extractor

**File**: `scripts/extract_threathunter_playbook.py`

**Input**: `data/threathunter-playbook/` — Markdown playbooks with KQL queries, data sources, hunting steps

**System message**:
```
You are a Threat Hunting methodology specialist. Design detection playbooks
with KQL queries, data source requirements, and step-by-step hunting procedures
mapped to MITRE ATT&CK.
```

**Pair generation** (1 pair per playbook):
- **User**: "How would you hunt for {technique_name} ({technique_id}) in a {platform} environment? Provide KQL queries, required data sources, and expected artifacts."
- **Assistant**: The playbook content: methodology overview, data sources, KQL queries, expected artifacts, false positive handling, MITRE mapping

**MITRE extraction**: From playbook metadata/frontmatter

**Output**: `data/datasets/buckets/sources/threathunter-playbook/defensive/threat_hunting/data.jsonl`

**Attribution fields**:
```json
{
  "source": "threathunter-playbook",
  "source_uri": "https://github.com/OTRF/ThreatHunter-Playbook",
  "license": "Apache-2.0",
  "license_uri": "https://github.com/OTRF/ThreatHunter-Playbook/blob/master/LICENSE",
  "rights_contact": "Open Threat Research (OTRF)",
  "attribution_text": "Copyright (c) Open Threat Research. Licensed under Apache 2.0."
}
```

**Estimated**: ~150 records, ~300 lines Python

### 4.4 Elastic/detection-rules Extractor

**File**: `scripts/extract_elastic_rules.py`

**Input**: `data/elastic-detection-rules/rules/` — TOML files with `[rule]`, `[rule.threat]` sections

**System message**:
```
You are an Elastic Security detection engineer. Write production-grade detection
rules in EQL/KQL mapped to MITRE ATT&CK with severity scoring, risk assessment,
and false positive analysis.
```

**Pair generation** (1 pair per rule):
- **User**: "Write an Elastic detection rule for {rule.name}. Map to MITRE ATT&CK {technique_id}. Include the EQL/KQL query, severity, risk score, and false positive scenarios."
- **Assistant**: Structured explanation: rule name, description, query (EQL/KQL), severity, risk score, MITRE mapping, false positives, investigation guide

**MITRE extraction**: From `[rule.threat]` TOML sections

**Output**: `data/datasets/buckets/sources/elastic-rules/defensive/detection_engineering/data.jsonl`

**Attribution fields**:
```json
{
  "source": "elastic-rules",
  "source_uri": "https://github.com/elastic/detection-rules",
  "license": "Elastic-2.0",
  "license_uri": "https://github.com/elastic/detection-rules/blob/main/LICENSE.txt",
  "rights_contact": "Elasticsearch B.V.",
  "attribution_text": "Copyright Elasticsearch B.V. Licensed under Elastic License 2.0."
}
```

**Estimated**: ~1,200 records, ~350 lines Python

### 4.5 Splunk/security_content Extractor

**File**: `scripts/extract_splunk_content.py`

**Input**: `data/splunk-security-content/detections/` + `stories/` — YAML detection files + Markdown analytic stories

**System message**:
```
You are a Splunk detection engineer. Write production-grade SPL detection queries
mapped to MITRE ATT&CK with data source configuration, notable event setup, and
false positive analysis.
```

**Pair generation** (1 pair per detection):
- **User**: "Write a Splunk detection for {detection_name}. Map to MITRE ATT&CK {technique_id}. Include the SPL query, data sources, notable event configuration, and false positive analysis."
- **Assistant**: Structured explanation: detection name, SPL query, data sources, notable event fields, MITRE mapping, false positives, how to implement

**MITRE extraction**: From YAML `tags.mitre_attack_id` or `mitre_attack` fields

**Output**: `data/datasets/buckets/sources/splunk-content/defensive/detection_engineering/data.jsonl`

**Attribution fields**:
```json
{
  "source": "splunk-content",
  "source_uri": "https://github.com/splunk/security_content",
  "license": "Apache-2.0",
  "license_uri": "https://github.com/splunk/security_content/blob/main/LICENSE",
  "rights_contact": "Splunk Inc.",
  "attribution_text": "Copyright (c) Splunk Inc. Licensed under Apache 2.0."
}
```

**Estimated**: ~800 records, ~350 lines Python

### 4.6 NIST SP 800-61r3 Extractor

**File**: `scripts/extract_nist_ir.py`

**Input**: NIST SP 800-61r3 PDF (manually downloaded to `data/nist-sp800-61r3.pdf`) — parsed into structured text

**System message**:
```
You are an Incident Response specialist following the NIST SP 800-61 framework.
Provide phase-specific IR procedures including containment, eradication, recovery,
evidence collection, and stakeholder communication.
```

**Pair generation** (~200 pairs from document structure):
- **Phase-based pairs**: "A {incident_type} has been detected on {asset_type}. Walk through the NIST SP 800-61 {phase} phase. Include containment steps, evidence collection procedures, and stakeholder communication requirements."
- **Decision tree pairs**: "You are responding to a {scenario}. Based on NIST SP 800-61, what is the appropriate escalation path? What factors determine whether to contain, eradicate, or recover first?"
- **Cross-phase pairs**: "Compare the NIST SP 800-61 {phase_a} and {phase_b} phases. What handoff artifacts should be produced? What decisions made in {phase_a} constrain options in {phase_b}?"

**MITRE mapping**: Not directly MITRE-mapped — use `mitre_ids: []` and tag as `defensive/incident_response`

**Output**: `data/datasets/buckets/sources/nist-ir/defensive/incident_response/data.jsonl`

**Attribution fields**:
```json
{
  "source": "nist-sp800-61r3",
  "source_uri": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r3.pdf",
  "license": "Public-Domain",
  "license_uri": "https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications",
  "rights_contact": "NIST",
  "attribution_text": "Public Domain — United States Government work. NIST SP 800-61 Revision 3."
}
```

**Estimated**: ~200 records, ~300 lines Python

---

## 5. Team Preset System

Three JSON preset files in `presets/`:

### `presets/red-team.json`
```json
{
  "name": "red-team",
  "version": "1.0",
  "description": "Offensive security specialist — 90% red team, 10% blue team. Includes orchestrator routing data.",
  "bucket_weights": {
    "base/*": 0.50,
    "tools/*": 0.30,
    "ai/*": 0.05,
    "orchestrator": 0.05,
    "defensive/*": 0.10
  },
  "total_pairs": 16000,
  "system_prompt": "You are an authorized Red Team specialist conducting security assessments. Provide accurate, detailed technical information about offensive security techniques, adversary emulation, and MITRE ATT&CK tradecraft."
}
```

### `presets/purple-team.json`
```json
{
  "name": "purple-team",
  "version": "1.0",
  "description": "Adversary emulation + detection engineering — 50% red team, 50% blue team. Includes orchestrator routing data.",
  "bucket_weights": {
    "base/*": 0.25,
    "tools/*": 0.15,
    "ai/*": 0.05,
    "orchestrator": 0.05,
    "defensive/*": 0.50
  },
  "total_pairs": 16000,
  "system_prompt": "You are a Purple Team specialist bridging offensive and defensive security. Provide adversary emulation techniques alongside detection engineering guidance, mapped to MITRE ATT&CK."
}
```

### `presets/blue-team.json`
```json
{
  "name": "blue-team",
  "version": "1.0",
  "description": "Defensive security specialist — 10% red team, 90% blue team. Includes orchestrator routing data.",
  "bucket_weights": {
    "base/*": 0.05,
    "tools/*": 0.03,
    "ai/*": 0.01,
    "orchestrator": 0.01,
    "defensive/*": 0.90
  },
  "total_pairs": 16000,
  "system_prompt": "You are a Blue Team defensive security specialist. Provide detection engineering, threat hunting, and incident response guidance mapped to MITRE ATT&CK. Include SIEM queries, log analysis, and defensive methodology."
}
```

**Key design decisions**:
- Orchestrator data always included (5% red/purple, 1% blue) — preserves task-following capability
- Wildcard patterns (`base/*`, `defensive/*`) match all sub-buckets
- `total_pairs` is a target — actual count may vary slightly due to bucket availability
- System prompt changes per preset — injected during training via `--system-prompt` flag

---

## 6. Balance Script Update

### Changes to `scripts/balance_buckets.py`

**Add `--preset` flag**:
```python
parser.add_argument(
    "--preset",
    type=str,
    default=None,
    help="Team preset: 'red-team', 'purple-team', 'blue-team', or path to custom preset JSON",
)
```

**Add `--system-prompt` flag**:
```python
parser.add_argument(
    "--system-prompt",
    type=str,
    default=None,
    help="Override system prompt in output pairs (used with --preset)",
)
```

**Preset loading function** (~50 lines):
```python
def load_preset(preset_name: str) -> dict:
    """Load a team preset by name or path."""
    preset_paths = {
        "red-team": BASE_DIR / "presets" / "red-team.json",
        "purple-team": BASE_DIR / "presets" / "purple-team.json",
        "blue-team": BASE_DIR / "presets" / "blue-team.json",
    }
    if preset_name in preset_paths:
        path = preset_paths[preset_name]
    else:
        path = Path(preset_name)
    
    if not path.exists():
        raise FileNotFoundError(f"Preset not found: {path}")
    
    with open(path) as f:
        preset = json.load(f)
    
    # Validate required fields
    for field in ["name", "bucket_weights", "total_pairs"]:
        if field not in preset:
            raise ValueError(f"Preset missing required field: {field}")
    
    return preset
```

**Bucket weight resolution** (~40 lines):
```python
def resolve_bucket_weights(preset: dict, buckets: list[str]) -> dict[str, float]:
    """Resolve wildcard bucket weights to per-bucket weights."""
    weights = preset["bucket_weights"]
    resolved = {}
    
    for bucket in buckets:
        # Check exact match first
        if bucket in weights:
            resolved[bucket] = weights[bucket]
            continue
        
        # Check wildcard patterns
        matched = False
        for pattern, weight in weights.items():
            if pattern.endswith("/*"):
                prefix = pattern[:-2]
                if bucket.startswith(prefix + "/") or bucket == prefix:
                    resolved[bucket] = weight
                    matched = True
                    break
        
        if not matched:
            resolved[bucket] = 0.0  # bucket not in preset — skip
    
    return resolved
```

**Integration with existing balance logic**:
- When `--preset` is used, override the profile-based caps with weight-based allocation
- `total_pairs` from preset becomes the `--target-total` value
- Per-bucket allocation: `bucket_pairs = total_pairs * weight / sum(weights)`
- If `--system-prompt` is provided, replace the system message in every output pair
- Default behavior (no `--preset`) remains completely unchanged

**Estimated**: ~150 lines added to balance_buckets.py

---

## 7. Init Pipeline Update

### Changes to `scripts/init_pipeline.py`

**Add 5 new local probes** (Sigma already exists):
```python
_LOCAL_PROBES: list[tuple[str, Path, Path, int]] = [
    # ... existing probes ...
    (
        "mordor",
        DATA_DIR / "mordor",
        DATA_DIR / "mordor" / "datasets",
        1024,
    ),
    (
        "threathunter-playbook",
        DATA_DIR / "threathunter-playbook",
        DATA_DIR / "threathunter-playbook" / "playbooks",
        1024,
    ),
    (
        "elastic-detection-rules",
        DATA_DIR / "elastic-detection-rules",
        DATA_DIR / "elastic-detection-rules" / "rules",
        1024,
    ),
    (
        "splunk-security-content",
        DATA_DIR / "splunk-security-content",
        DATA_DIR / "splunk-security-content" / "detections",
        1024,
    ),
    (
        "nist-sp800-61r3",
        DATA_DIR / "nist-sp800-61r3",
        DATA_DIR / "nist-sp800-61r3" / "NIST.SP.800-61r3.pdf",
        1024,
    ),
]
```

**Add 5 new remote repos**:
```python
_REMOTE_REPOS: list[tuple[str, str, Path]] = [
    # ... existing repos ...
    (
        "mordor",
        "https://github.com/OTRF/Security-Datasets.git",
        DATA_DIR / "mordor",
    ),
    (
        "threathunter-playbook",
        "https://github.com/OTRF/ThreatHunter-Playbook.git",
        DATA_DIR / "threathunter-playbook",
    ),
    (
        "elastic-detection-rules",
        "https://github.com/elastic/detection-rules.git",
        DATA_DIR / "elastic-detection-rules",
    ),
    (
        "splunk-security-content",
        "https://github.com/splunk/security_content.git",
        DATA_DIR / "splunk-security-content",
    ),
    # NIST SP 800-61r3: manual download (no git repo)
    # User must place PDF at data/nist-sp800-61r3/NIST.SP.800-61r3.pdf
]
```

**Add 6 new extractors to the extraction sequence**:
```python
def main_extract() -> int:
    extractors = [
        # ... existing extractors ...
        "extract_sigma_defensive.py",
        "extract_mordor.py",
        "extract_threathunter_playbook.py",
        "extract_elastic_rules.py",
        "extract_splunk_content.py",
        "extract_nist_ir.py",
    ]
    # ... rest of extraction logic ...
```

**NIST SP 800-61r3 special handling**:
- Not a git repo — user must manually download the PDF
- Probe checks for PDF existence and size
- If missing, print instructions: "Download NIST SP 800-61r3 from https://nvlpubs.nist.gov/... and place at data/nist-sp800-61r3/NIST.SP.800-61r3.pdf"
- Extraction uses PyPDF2 or pdfplumber to parse the PDF

**Estimated**: ~100 lines added to init_pipeline.py

---

## 8. CLI Updates

### Changes to `src/attacklm/cli.py`

**Add `--preset` to `main_balance`**:
```python
def main_balance(argv: Sequence[str] | None = None) -> int:
    """Build a balanced subset: attacklm-balance [--preset red|purple|blue]"""
    return _run_python_script(
        "balance_buckets.py", argv if argv is not None else sys.argv[1:]
    )
```
No code change needed — `balance_buckets.py` handles the `--preset` flag via argparse.

**Update `main_extract` to include new extractors**:
```python
def main_extract(argv: Sequence[str] | None = None) -> int:
    """Run all data extractors in sequence: attacklm-extract"""
    _ = argv
    extractors = [
        # ... existing extractors ...
        "extract_sigma_defensive.py",
        "extract_mordor.py",
        "extract_threathunter_playbook.py",
        "extract_elastic_rules.py",
        "extract_splunk_content.py",
        "extract_nist_ir.py",
    ]
    # ... rest of extraction logic ...
```

**Update help message**:
```python
print("  Presets:    attacklm-balance --preset red-team|purple-team|blue-team")
```

**Estimated**: ~20 lines changed in cli.py

---

## 9. Attribution & License Design

### Per-Source Files

Each new source gets two files at its root:

**`data/datasets/buckets/sources/<source>/SOURCE.md`**:
```markdown
# Source: <Display Name>

- **Repository**: <URL>
- **License**: <License Name>
- **License URI**: <URL>
- **Rights Contact**: <Organization>
- **Extraction Date**: <YYYY-MM-DD>
- **Extractor Script**: scripts/extract_<source>.py
- **Records Extracted**: <N>
- **Description**: <1-2 sentence description of what the source contains>
```

**`data/datasets/buckets/sources/<source>/LICENSE.md`**:
Full license text verbatim from upstream.

### Per-Record Attribution Fields

Every JSONL record includes:
```json
{
  "source": "<source-name>",
  "source_uri": "<upstream-url>",
  "license": "<spdx-identifier>",
  "license_uri": "<license-url>",
  "rights_contact": "<organization>",
  "attribution_text": "<copyright-notice>"
}
```

### License-Specific Attribution Text

| License | attribution_text Template |
|---------|--------------------------|
| DRL-1.1 | "Detection Rule License (DRL) 1.1 — Copyright (c) {year} {author}. See {license_uri} for full terms." |
| Apache-2.0 | "Copyright (c) {year} {author}. Licensed under Apache License 2.0. See {license_uri}." |
| Elastic-2.0 | "Copyright {author}. Licensed under Elastic License 2.0. See {license_uri}." |
| Public-Domain | "Public Domain — United States Government work. {document_title}. See {license_uri}." |

---

## 10. Manifest Update

### Changes to `data/datasets/buckets/manifest.json`

**Add 6 new sources** to `source_totals`:
```json
"source_totals": {
    "... existing ...": 0,
    "sigma-hq": 3000,
    "mordor": 500,
    "threathunter-playbook": 150,
    "elastic-rules": 1200,
    "splunk-content": 800,
    "nist-ir": 200
}
```

**Update totals**:
```json
"total_pairs": 21565,
"tier_totals": {
    "human": 21565,
    "llm": 0,
    "synth": 380
}
```

**Add new source entries** to `sources` object (same structure as existing sources).

**Add new buckets** to the bucket list:
```json
"buckets": [
    "... existing buckets ...",
    "defensive/detection_engineering",
    "defensive/threat_hunting",
    "defensive/incident_response"
]
```

---

## 11. Testing Strategy

### Unit Tests

| Test File | Tests For | Est. Lines |
|-----------|-----------|------------|
| `tests/test_extract_sigma_defensive.py` | Sigma YAML parsing, pair generation, MITRE extraction | ~200 |
| `tests/test_extract_mordor.py` | Mordor JSON parsing, multi-pair generation | ~200 |
| `tests/test_extract_threathunter_playbook.py` | Markdown playbook parsing | ~150 |
| `tests/test_extract_elastic_rules.py` | TOML parsing, rule extraction | ~200 |
| `tests/test_extract_splunk_content.py` | YAML detection parsing | ~200 |
| `tests/test_extract_nist_ir.py` | PDF parsing, pair generation | ~150 |
| `tests/test_presets.py` | Preset loading, validation, weight resolution | ~150 |
| `tests/test_balance_presets.py` | Balance with --preset flag, weight allocation | ~150 |

### Integration Tests

- `attacklm-init --dry-run` with new sources
- `attacklm-balance --preset red-team --dry-run`
- `attacklm-balance --preset purple-team --dry-run`
- `attacklm-balance --preset blue-team --dry-run`
- Attribution completeness check (all records have all 6 fields)

### Hermetic Pattern

All tests follow the existing hermetic pattern:
- Mock upstream data (sample YAML/JSON/TOML files in test fixtures)
- No network access
- No GPU required
- Fast execution (< 5 seconds total)

**Estimated**: ~1,400 lines of tests

---

## 12. Implementation Order

| Phase | Component | Est. Lines | Dependencies |
|-------|-----------|------------|--------------|
| 1 | `extract_sigma_defensive.py` | ~400 | None |
| 1 | `extract_elastic_rules.py` | ~350 | None |
| 1 | `extract_splunk_content.py` | ~350 | None |
| 2 | `extract_mordor.py` | ~350 | None |
| 2 | `extract_threathunter_playbook.py` | ~300 | None |
| 2 | `extract_nist_ir.py` | ~300 | PyPDF2/pdfplumber |
| 3 | `presets/red-team.json` | ~15 | None |
| 3 | `presets/purple-team.json` | ~15 | None |
| 3 | `presets/blue-team.json` | ~15 | None |
| 4 | `balance_buckets.py` update | ~150 | Presets exist |
| 5 | `init_pipeline.py` update | ~100 | Extractors exist |
| 6 | `cli.py` update | ~20 | All above |
| 7 | `manifest.json` update | ~100 | Extractors run |
| 8 | Attribution files (SOURCE.md, LICENSE.md) | ~200 | Extractors run |
| 9 | Tests (8 test files) | ~1,400 | All above |

**Total**: ~4,150 lines (2,050 Python + 45 JSON + 200 docs + 1,400 tests + 455 existing pattern reuse)

**Parallelizable**: Phases 1+2 (all 6 extractors) can be built simultaneously. Phases 3-6 are sequential.

---

## Appendix A: pyproject.toml Changes

Add new dependency for NIST PDF parsing:
```toml
[project.optional-dependencies]
extract = [
    "pyyaml",
    "requests",
    "gitpython",
    "pdfplumber>=0.11",  # for NIST SP 800-61r3 PDF parsing
]
```

---

## Appendix B: New CLI Commands Summary

After implementation, these commands will work:

```bash
# Full init with new sources
attacklm-init --yes

# Balance with team presets
attacklm-balance --preset red-team --output data/datasets/balanced/red-team.jsonl
attacklm-balance --preset purple-team --output data/datasets/balanced/purple-team.jsonl
attacklm-balance --preset blue-team --output data/datasets/balanced/blue-team.jsonl

# Train with preset
attacklm-train --dataset data/datasets/balanced/red-team.jsonl --output models/attacklm-red-team

# Custom preset
attacklm-balance --preset my-custom-preset.json --output data/datasets/balanced/custom.jsonl
```

---

*Last updated: 2026-06-24 — AttackLM v0.5.0 design*
