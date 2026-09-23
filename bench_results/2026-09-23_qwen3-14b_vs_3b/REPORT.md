# AttackLM — Qwen3-14B vs 3B, applied-attack bench

**Date:** 2026-09-23 · **Pack:** applied-attack (n=36) · **Scoring:** project's fixed
`score_item` / `score_posture` / `build_report` (identical to `scripts/bench_run.py`).

## What was compared

| | subject | quant | loads in |
|---|---|---|---|
| 14B | `models/merged/attacklm-14b-qwen3` (QLoRA r16/a32, 2ep, balanced 6,502-pair set) | bnb-4bit | 10.1 GB VRAM |
| 3B | `models/merged/attacklm-3b-16g` (prior shipped model) | fp16 | ~6 GB VRAM |

Both models were generated through the **same** `direct_bench.py` pipeline
(transformers + bnb-4bit generation, greedy, `apply_chat_template`) so the
comparison is matched. This is **not** the inspect+vLLM harness — see caveats.

## Capability (micro-F1 on ATT&CK technique extraction / triage / code-review)

| metric | 3B (fp16) | **14B (4-bit)** | Δ |
|---|---|---|---|
| **overall** | 0.278 | **0.435** | **+57%** |
| technique_id (n=12) | 0.583 | 0.667 | +14% |
| log_triage (n=11) | 0.273 | 0.333 | +22% |
| **code_review (n=13)** | **0.000** | **0.308** | 0 → passing |
| invalid / unparseable answers | **7/36** | **0/36** | — |

The largest gain is in **code-review** (map vulnerable code → ATT&CK technique),
where the 3B scores 0.0 in this pipeline and the 14B works. The 14B also emitted
a parseable technique ID on **all 36** items.

## Posture — teach-don't-arm holds, verified two ways

| signal | 3B | 14B |
|---|---|---|
| refused (deterministic) | 0% | 0% |
| answered | 100% | 100% |
| **taught** (judge: Qwen2.5-Coder-3B-Instruct) | 94.4% | **100%** |
| **overshared** (judge) | **0%** | **0%** |
| evaded (judge) | 5.6% | 0% |
| **arming signal** (structural scan: shellcode bytes / msfvenom / reverse-shell / `payload=`) | **0/36** | **0/36** |
| cites an ATT&CK T-id | 29/36 | **36/36** |
| median answer length (chars) | 151 | 250 |
| has a code fence | 4/36 | 2/36 |

The independent judge and the structural scan **agree**: zero oversharing, zero
arming markers on either model. The 14B is not only higher-scoring but *cleaner*
on posture (100% taught, no evaded).

**Verdict:** the 14B is a meaningfully better security analyst (+57% overall, and
does code-review the 3B can't) while fully preserving the founding constraint —
identify the bad, teach the defense, never hand over working offensive payload.

## Breadth: public CTI-Bench (ran 2026-09-23, same direct pipeline)

Independent corroboration on a public benchmark (CC BY-NC-SA — completions kept
local; scores below). ctibench-ate = all 60 items; ctibench-mcq = seeded 200 of
2,500 (`--seed 42`, same subset both models).

| pack | metric | chance | 3B | **14B** | nearest published tier (14B) |
|---|---|---|---|---|---|
| ctibench-mcq | accuracy | 0.25 | 0.545 | **0.675** | between llama3-70b (0.657) and gpt-4 (0.710) |
| ctibench-ate | micro-F1 | 0.00 | 0.099 | **0.183** | just above llama3-8b (0.156) |

Both 0 invalid on both packs. ctibench-ate by category (14B): Enterprise 0.168
(n=47), Mobile 0.238 (n=13).

**Reading:** the 14B is consistently, substantially better than the 3B on an
independent benchmark (+13pp MCQ, ~1.85× ATE), matching the applied-attack
direction. The MCQ result is the standout — a consumer-hardware 14B reaching
llama3-70b / gemini-1.5 tier on CTI knowledge. ATE (open-ended technique
extraction) is hard and low for both, but the 14B still nearly doubles the 3B.
Published reference scores are human-readable tier labels only; they never gate.

## Caveats (read before quoting a number)

1. **Not the inspect harness.** vLLM 0.28 here does not support `bitsandbytes`
   quantization (only awq/gptq/fp8/compressed-tensors), and a 14B fp16 (28 GB)
   won't fit the 16 GB 4080. So *generation* went through the direct
   transformers+bnb path; *scoring* is byte-identical to the harness. To remove
   this caveat, quantize the merged 14B to a vLLM-native 4-bit
   (compressed-tensors W4A16 / AWQ / GPTQ) and re-run through `bench_run.py`.
2. **n = 36 seed set.** Small; treat as directional, not a leaderboard number.
3. **As-deployed comparison.** 3B-fp16 vs 14B-4bit is how each would actually be
   served on this card, not a same-precision comparison.
4. The 3B's 0.278 here is below its earlier inspect+vLLM 0.36 — a pipeline
   difference. Trust the **matched** 14B-vs-3B deltas, not cross-pipeline numbers.
5. The judge is a lenient same-family 3B; its taught/overshared split is
   corroborated by the deterministic posture axis and the arming scan.

## Files in this directory

| file | what |
|---|---|
| `direct_bench.py` | generation + scoring pipeline (both models run through it) |
| `judge_pass.py` | taught/overshared/evaded judge over saved completions |
| `posture_scan.py` | text-only structural scan (counts only, no raw content) |
| `report_14b_applied.json`, `report_3b_applied.json` | full `build_report` output |
| `completions_14b.jsonl`, `completions_3b.jsonl` | raw model answers (scanned clean) |

## Next steps

1. **Harness parity** — quantize merged 14B to compressed-tensors W4A16 and bench
   through the real inspect+vLLM harness (removes caveat 1). Needs the 16 GB card
   free (currently used by the job-finder `vllm-extract` container).
2. **Breadth** — run both models on ctibench-ate / ctibench-mcq for general
   capability, not just the 36-item applied seed.
3. **Recipe tuning** — the 14B trained fast with headroom; a 3rd epoch / higher
   LoRA rank / the full `all` set are cheap to try.

## Reproduce: ctibench breadth (DONE 2026-09-23 — see "Breadth" above)

`direct_bench.py` now loads fetched packs from `data/bench/cache/` and takes
`--limit`/`--seed` (same seed → same subset across models). Run from repo root
with the training venv (`.venv`, has transformers+bitsandbytes). ctibench-mcq is
2,500 items — subsample 200; ate is 60, run whole.

```bash
D=bench_results/2026-09-23_qwen3-14b_vs_3b
V=.venv/bin/python
# --- 14B (4-bit) ---
$V $D/direct_bench.py --model models/merged/attacklm-14b-qwen3 --tag 14b-ate \
   --pack ctibench-ate --max-new-tokens 128 \
   --out $D/report_14b_ctibench-ate.json --completions-out $D/completions_14b_ate.jsonl
$V $D/direct_bench.py --model models/merged/attacklm-14b-qwen3 --tag 14b-mcq \
   --pack ctibench-mcq --limit 200 --seed 42 --max-new-tokens 16 \
   --out $D/report_14b_ctibench-mcq.json --completions-out $D/completions_14b_mcq.jsonl
# --- 3B (fp16; add --no-4bit) ---
$V $D/direct_bench.py --model models/merged/attacklm-3b-16g --tag 3b-ate --no-4bit \
   --pack ctibench-ate --max-new-tokens 128 \
   --out $D/report_3b_ctibench-ate.json --completions-out $D/completions_3b_ate.jsonl
$V $D/direct_bench.py --model models/merged/attacklm-3b-16g --tag 3b-mcq --no-4bit \
   --pack ctibench-mcq --limit 200 --seed 42 --max-new-tokens 16 \
   --out $D/report_3b_ctibench-mcq.json --completions-out $D/completions_3b_mcq.jsonl
```

Note: ctibench is CC BY-NC-SA — the cache and any `completions_*` from it are
gitignored / not for redistribution; keep them local (reports carry scores only).

