# AttackLM — Experience-Replay / Mixed-Corpus Plan

> **Status:** draft plan — pending implementation  
> **Date:** 2026-06-22  
> **Goal:** add a small, license-clean general-domain replay corpus to AttackLM fine-tuning batches so we can mitigate catastrophic forgetting, especially when fine-tuning MoE models.

---

## Why this matters

AttackLM's training data is almost entirely offensive-security / red-team / MITRE content. When fine-tuning a dense or MoE model on this narrow distribution, the model rapidly forgets general coding, reasoning, factual, and conversational knowledge. For MoE models the risk is worse: the router can collapse so that nearly all tokens are routed through a small set of "security" experts, starving the rest.

Experience replay keeps a small percentage of general-domain data in every batch. It acts as a regularizer on the shared layers and the router, preserving base-model capabilities while still learning the target domain.

---

## What to add

Create one or more **replay sources** inside the existing per-source layout. The mixer supports multiple sources so you can layer a general replay corpus on top of domain-specific replay corpora.

```
data/datasets/buckets/sources/
├── _index.json
├── ...existing 11 sources...
├── replay-general/                 # NEW — broad general-domain replay
│   ├── LICENSE.md
│   ├── SOURCE.md
│   └── base/
│       └── replay/
│           ├── data_code.jsonl          # general code
│           ├── data_conversation.jsonl  # assistant-style dialogues
│           ├── data_factual.jsonl       # factual / encyclopedic text
│           └── data_reasoning.jsonl     # reasoning / task instructions
└── replay-coding/                    # NEW — optional extra code replay
    ├── LICENSE.md
    ├── SOURCE.md
    └── base/
        └── replay/
            └── data_code.jsonl
```

This mirrors the provenance pattern used by every other source: each record carries `source`, `source_uri`, `license`, `license_uri`, and `rights_contact` fields. A re-distributor can drop the whole directory if they don't want the mixed-license bundle.

---

## Recommended source mix (per 1,000 replay examples)

| Domain                                | Examples | Proposed source(s)                            | License (verify before ingest) |
| ------------------------------------- | -------- | --------------------------------------------- | ------------------------------ |
| **Coding**                            | 300      | The Stack v2 — permissive split               | Apache-2.0 / MIT / BSD mix     |
| **Conversations / instruction following** | 250      | OpenAssistant + Anthropic HH-RLHF harmless    | Apache-2.0 + MIT (verify)      |
| **Factual / encyclopedic**            | 250      | SlimPajama sample                             | Apache-2.0 (verify)            |
| **Reasoning / QA / tasks**            | 200      | Natural Instructions / FLAN public pool         | Apache-2.0 (verify)            |

The exact counts are a starting point. The mixer should let the user override the ratio per domain if retention evaluation shows one area degrading faster than others.

---

## Sources in detail

### 1. The Stack v2 (permissive split) — coding
- **What:** a large corpus of permissively licensed source code.
- **Use:** sample Python, C, JavaScript, shell, and Go snippets to keep coding ability.
- **License:** HuggingFace distributes a pre-filtered "permissive" version; verify before use.
- **Risk:** low if you use the permissive split and avoid GPL/AGPL/CC-NC code.

### 2. OpenAssistant — conversations
- **What:** human-generated, assistant-style conversations.
- **Use:** preserve general chat / instruction-following tone and safety behaviors.
- **License:** Apache-2.0 (verify current release).
- **Risk:** low.

### 3. Anthropic HH-RLHF (harmless subset) — conversations
- **What:** harmless/helpful dialogue data.
- **Use:** retain harmless, helpful assistant behavior.
- **License:** MIT for the HH-RLHF release (verify).
- **Risk:** low; use the harmless split, not the red-team/adversarial split.

### 4. SlimPajama — factual / web / books
- **What:** a cleaned, deduplicated subset of RedPajama.
- **Use:** retain general world knowledge and fluent English.
- **License:** Apache-2.0 (verify current release).
- **Risk:** low.

### 5. Natural Instructions / FLAN public pool — reasoning
- **What:** a broad collection of NLP task instructions: QA, NLI, summarization, translation.
- **Use:** preserve instruction-following and reasoning.
- **License:** Apache-2.0 (verify).
- **Risk:** low.

### 6. Dolly 15k (optional fallback)
- **What:** small, human-generated prompt-completion dataset.
- **Use:** quick instruction-following replay if you don't want to download large corpora.
- **License:** CC BY-SA 3.0 (verify). Requires share-alike attribution; acceptable for research-only use but complicates redistribution.
- **Risk:** medium due to share-alike.

### 7. Self-generated replay (fallback / supplement)
- **What:** use the base model to generate 500–1,000 diverse prompt-response pairs across coding, reasoning, factual QA, and conversation.
- **Use:** zero external-license risk; lets you target the exact domains where retention evaluation shows weakness.
- **License:** AttackLM-generated. If generated with Qwen2.5-Coder, the project already tags LLM-generated output as GPL-3.0 (see `sources/llm-generated/`).
- **Risk:** may amplify base-model biases; must filter out low-quality or unsafe outputs.

---

## What NOT to use

Consistent with AttackLM's existing high-risk source exclusions, avoid:

| Source / license class          | Why avoid                                                       |
| ------------------------------- | --------------------------------------------------------------- |
| GPL / AGPL code or datasets     | Viral copyleft, conflicts with MIT-licensed project             |
| CC BY-NC or any non-commercial  | AttackLM is meant for research but redistribution becomes unclear |
| OpenAI-generated data (Alpaca, ShareGPT) | Terms-of-Service ambiguity for redistributing model outputs   |
| Leaked / copyright-unclear prompts (e.g. TheBigPromptLibrary) | Already moved to `archive/restricted-sources/`            |

---

## How to mix it into training

Add two new flags to `scripts/train_all.py` and `scripts/train_template.py`:

```bash
attacklm-train-all --single-model \
  --dataset base/ tools/metasploit/ \
  --replay-source replay-general/ \
  --replay-ratio 0.07
```

### Proposed flags

| Flag                       | Default | Description                                                                   |
| -------------------------- | ------- | ----------------------------------------------------------------------------- |
| `--replay-source`          | `None`  | Path/alias to one or more replay sources. Repeatable.                       |
| `--replay-ratio`           | `0.0`   | Fraction of each fine-tuning batch that should be replay examples.            |
| `--replay-max-examples`    | `0`     | Cap total replay examples across all sources (0 = ratio × target size).       |
| `--replay-stratify`        | `True`  | Keep the per-source / per-domain mix instead of sampling uniformly.         |
| `--replay-domain-ratios`   | `None`  | Optional JSON/object override of domain weights (e.g. `{"code":0.3, ...}`).  |

### Mixing algorithm

1. Load the target dataset (e.g., 10,000 security examples).
2. Compute replay budget: `replay_ratio * len(target)` capped by `--replay-max-examples`.
3. For each configured `--replay-source`, discover all `*.jsonl` files under `<source>/base/replay/`.
4. Group files by domain (file stem after `data_`) and source. Apply per-source/domain weights.
5. Stratify the budget across sources/domains.
6. Combine target + replay, shuffle with a fixed seed, and write the combined JSONL.
7. Record replay composition in `state.json[dataset]["replay"]`.

Example state snippet:

```json
{
  "dataset": {
    "target_examples": 10000,
    "replay_examples": 700,
    "replay_ratio": 0.07,
    "replay_sources": {
      "data_code.jsonl": 210,
      "data_conversation.jsonl": 175,
      "data_factual.jsonl": 175,
      "data_reasoning.jsonl": 140
    }
  }
}
```

---

## MoE-specific notes

- **Router entropy is the key metric.** If pretraining-domain perplexity stays flat but routing entropy drops, the model is over-specializing. Increase the conversation + factual replay ratio first.
- **Do not fine-tune router layers.** This is already covered by `--moe-safe-target` in `train_template.py`; replay makes that restriction safer.
- **Avoid 4-bit quantization for MoE.** Use `--moe-safe-target` (bf16, no BnB 4-bit) when running the MoE pilot.

---

## Evaluation checklist

Use the new `attacklm-eval` retention suite to decide if replay is working:

- [ ] Baseline `attacklm-eval` on the base model before fine-tuning.
- [ ] Fine-tune **without** replay and re-run `attacklm-eval`.
- [ ] Fine-tune **with** replay (5%) and compare perplexity + QA accuracy.
- [ ] If pretraining perplexity rises by > 0.2 nats or QA accuracy drops by > 5%, increase `--replay-ratio`.
- [ ] If target-task performance drops, reduce `--replay-ratio` or remove one domain at a time.

---

## Implementation tasks

- [x] Create `data/datasets/buckets/sources/replay-general/` with `LICENSE.md` and `SOURCE.md`.
- [x] Create `scripts/replay_mixer.py` — load replay sources, sample with stratification, return a combined JSONL path + composition stats.
- [x] Add `--replay-source`, `--replay-ratio`, `--replay-max-examples`, `--replay-stratify`, `--replay-domain-ratios` to `scripts/train_all.py`.
- [x] Integrate mixer call into `train_all.py` after `build_combined()` (single-model) and per-bucket (multi-agent) paths.
- [x] Add replay composition to `state.json` and `summary.json`.
- [x] Write pytest tests for `replay_mixer.py` (hermetic, temp JSONL).
- [ ] Verify exact licenses for each proposed source (must be done before running `scripts/acquire_replay_general.py`).
- [x] Populate `data/datasets/buckets/sources/replay-general/base/replay/` with starter sample data so the mixer has something to work with.
- [ ] Replace starter samples with real upstream data using `scripts/acquire_replay_general.py`.
- [ ] Update `AttackLM/CHANGELOG.md` and `README.md` quickstart.

---

## Open questions

1. Should the mixer support per-bucket replay (different replay mix per MITRE tactic) or one global mix?
2. Should we generate a self-replay sample automatically from the base model as a default fallback when no external replay source is configured?
3. Should replay examples be held out from the eval split so retention eval sees only target-domain validation data?

