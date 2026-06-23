# replay-general — License Notice

This is a **mixed-license** planned source for general-domain replay data used
to mitigate catastrophic forgetting during AttackLM fine-tuning.

## Status

**Planned / not yet populated.** No data files exist under this directory yet.

## Expected Data Sources and Licenses

Each domain file (`data_<domain>.jsonl`) will carry per-record provenance
fields (`source`, `source_uri`, `license`, `license_uri`, `rights_contact`)
as required by the AttackLM attribution system.

Planned upstream sources and their expected licenses:

| Domain | Planned Source | Expected License | Risk Level |
| --- | --- | --- | --- |
| `code` | The Stack v2 (permissive split) | Apache-2.0 / MIT / BSD mix | Low |
| `conversation` | OpenAssistant OASST1 | Apache-2.0 | Low |
| `conversation` | Anthropic HH-RLHF (harmless) | MIT | Low |
| `factual` | SlimPajama (sampled) | Apache-2.0 | Low |
| `reasoning` | FLAN / Natural Instructions (public pool) | Apache-2.0 | Low |

**Important:** Each license must be **verified before ingestion**. Do not
assume the licenses listed above are current or accurate — check the upstream
repository's LICENSE file and any dataset card on HuggingFace.

## Excluded Licenses

Consistent with AttackLM's high-risk source exclusions:

- **GPL / AGPL**: Viral copyleft; conflicts with MIT-licensed project
- **CC BY-NC / non-commercial**: Redistribution terms unclear
- **OpenAI-generated data** (Alpaca, ShareGPT): ToS ambiguity
- **Copyright-unclear scrapes**: Moved to `archive/restricted-sources/`

## Redistribute Notice

If you redistribute this source directory, you **must** include this
`LICENSE.md` file and honor the per-record license fields. Each record's
`license` and `license_uri` fields take precedence over this summary.