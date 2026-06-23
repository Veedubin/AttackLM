# replay-general — Source Description

## Identity

- **Source name:** replay-general
- **Source type:** General-domain replay corpus (experience replay)
- **Risk level:** Low
- **Upstream URL:** See domain-specific sources below
- **Local path:** `data/datasets/buckets/sources/replay-general/`

## Purpose

This source provides general-domain training examples mixed into AttackLM
fine-tuning batches to mitigate **catastrophic forgetting**. When fine-tuning
on narrow offensive-security data, the model rapidly loses coding, reasoning,
conversational, and factual capabilities. A small fraction of replay data
(5–10%) acts as a regularizer on shared layers and MoE routers.

## Planned Domains

| File | Domain | Planned Upstream | Upstream URL |
| --- | --- | --- | --- |
| `data_code.jsonl` | General coding | The Stack v2 (permissive split) | https://huggingface.co/datasets/bigcode/the-stack-v2 |
| `data_conversation.jsonl` | Assistant dialogues | OpenAssistant OASST1 + Anthropic HH-RLHF (harmless) | https://huggingface.co/datasets/OpenAssistant/oasst1 |
| `data_factual.jsonl` | Encyclopedic / web text | SlimPajama (sampled) | https://huggingface.co/datasets/cerebras/SlimPajama-627B |
| `data_reasoning.jsonl` | Instruction / QA / reasoning | FLAN / Natural Instructions (public pool) | https://huggingface.co/datasets/google/flan_t5_xxl |

## Data Format

Each JSONL record follows the AttackLM per-source provenance convention:

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "source": "replay-general/code",
  "source_uri": "https://huggingface.co/datasets/bigcode/the-stack-v2",
  "license": "Apache-2.0",
  "license_uri": "https://www.apache.org/licenses/LICENSE-2.0",
  "rights_contact": "legal@example.com",
  "domain": "code"
}
```

## Ingestion Notes

- **No data files exist yet.** This is a skeleton for planned ingestion.
- Each domain file must be independently downloadable and license-verified
  before population.
- Target: ~1,000 total examples (300 code, 250 conversation, 250 factual,
  200 reasoning) for a 7% replay ratio on a 15K security dataset.
- The `scripts/replay_mixer.py` tool handles discovery, stratified sampling,
  and mixing into training batches.

## Integration

Use with `attacklm-train-all`:

```bash
attacklm-train-all --single-model \
  --dataset base/ \
  --replay-source replay-general/ \
  --replay-ratio 0.07
```

Or with custom domain weights:

```bash
attacklm-train-all --single-model \
  --dataset base/ \
  --replay-source replay-general/ \
  --replay-ratio 0.07 \
  --replay-domain-ratios '{"code":0.3,"conversation":0.25,"factual":0.25,"reasoning":0.2}'
```