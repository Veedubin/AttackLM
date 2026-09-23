#!/usr/bin/env python3
"""Direct capability+posture bench: transformers + bnb-4bit generation, then the
project's own (fixed) scorers/report. Bypasses the inspect+vLLM harness only
for GENERATION (that harness's vLLM can't 4-bit a 14B on 16GB); scoring is
identical to bench_run. Run the SAME pipeline for every model so cross-model
comparison is clean, and note this is not the inspect harness.

    python direct_bench.py --model models/merged/attacklm-14b-qwen3 --tag qwen3-14b
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO = Path("/home/Veedubin/Projects/reverse_engineering/AttackLM")
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402
from attacklm.bench.items import load_items  # noqa: E402
from attacklm.bench.packs import PACKS_DIR, get_pack  # noqa: E402
from attacklm.bench.scorers import ScoreConfig, score_item  # noqa: E402
from attacklm.bench.posture import PostureConfig, score_posture  # noqa: E402
from attacklm.bench.report import build_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pack", default="applied-attack")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--load-4bit", action="store_true", default=True)
    ap.add_argument("--no-4bit", dest="load_4bit", action="store_false")
    ap.add_argument("--out", required=True)
    ap.add_argument("--completions-out", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="seeded subsample to at most N items (for large packs like ctibench-mcq)")
    ap.add_argument("--seed", type=int, default=42,
                    help="subsample seed — MUST match across models for a comparable subset")
    args = ap.parse_args()

    pack = get_pack(args.pack, PACKS_DIR)
    # local packs (applied-attack) live in-repo; fetched packs (ctibench-*, etc.)
    # are materialized to the gitignored cache by the harness fetch layer.
    if pack.source.kind == "local":
        items_path = REPO / pack.source.repo
    else:
        items_path = REPO / "data" / "bench" / "cache" / f"{pack.name}.jsonl"
    if not items_path.exists():
        raise SystemExit(
            f"[{args.tag}] items not found at {items_path} — for a fetched pack, "
            f"run the harness once (or the fetch layer) to populate data/bench/cache/.")
    items = load_items(items_path)
    if args.limit and len(items) > args.limit:
        import random
        items = random.Random(args.seed).sample(items, args.limit)
        # deterministic order so the report reads stably run-to-run
        items.sort(key=lambda it: it.question_id)
    print(f"[{args.tag}] {len(items)} items from {items_path}"
          f"{f' (subsampled seed={args.seed})' if args.limit else ''}", flush=True)

    kw = dict(torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    if args.load_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    model.eval()
    print(f"[{args.tag}] loaded in {time.time()-t0:.0f}s "
          f"(4bit={args.load_4bit}) VRAM={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)

    completions: dict[str, str] = {}
    for i, it in enumerate(items):
        prompt = tok.apply_chat_template(it.messages, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                  do_sample=False, temperature=None, top_p=None,
                                  pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        completions[it.question_id] = text
        if (i + 1) % 6 == 0:
            print(f"[{args.tag}]  {i+1}/{len(items)}", flush=True)

    cfg = ScoreConfig()
    pcfg = PostureConfig()
    scores = [score_item(it, completions[it.question_id], cfg) for it in items]
    posture = [score_posture(it, completions[it.question_id], pcfg) for it in items]
    report = build_report(pack, scores, {"model": args.model, "tag": args.tag,
                                         "generation": "transformers+bnb4bit-direct"},
                          posture=posture, judged=False)

    Path(args.out).write_text(json.dumps(report, indent=2))
    if args.completions_out:
        Path(args.completions_out).write_text(
            "\n".join(json.dumps({"question_id": q, "category": next(it.category for it in items if it.question_id==q),
                                  "completion": c}) for q, c in completions.items()))

    s = report["summary"]
    print(f"\n===== {args.tag} =====")
    print(f"capability micro-F1: {s['score_raw']['score']}  (n={s['score_raw']['n']}, invalid={s['score_raw']['invalid']})")
    for cat, v in sorted(s.get("by_category", {}).items()):
        print(f"  {cat:14s} {v['score']}  (n={v['n']})")
    ps = s.get("posture", {})
    print(f"posture: refused={ps.get('refusal_rate')} answered={ps.get('answered_rate')} evaded={ps.get('evaded_rate')} (n={ps.get('n')})")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
