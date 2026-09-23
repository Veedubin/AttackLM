#!/usr/bin/env python3
"""Judge the taught/overshared/evaded split of saved completions, reusing the
project's judge prompt + verdict parser. Loads a judge model (4-bit) and scores
each answered completion. Reports counts only (no raw content)."""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent  # repo root (scripts/..)
sys.path.insert(0, str(REPO / "src"))
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402
from attacklm.bench.items import load_items  # noqa: E402
from attacklm.bench.packs import PACKS_DIR, get_pack  # noqa: E402
from attacklm.bench.posture import (  # noqa: E402
    JudgeError, build_judge_prompt, parse_judge_verdict,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--completions", required=True, help="jsonl {question_id, completion}")
    ap.add_argument("--pack", default="applied-attack")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()

    pack = get_pack(args.pack, PACKS_DIR)
    items = {it.question_id: it for it in load_items(REPO / pack.source.repo)}
    comps = {json.loads(ln)["question_id"]: json.loads(ln)["completion"]
             for ln in open(args.completions) if ln.strip()}

    tok = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16),
    )
    model.eval()

    verdicts = Counter()
    unparseable = 0
    for qid, comp in comps.items():
        item = items[qid]
        jprompt = build_judge_prompt(item, comp)
        msgs = [{"role": "user", "content": jprompt}]
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        vtext = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            verdicts[parse_judge_verdict(vtext)] += 1
        except JudgeError:
            unparseable += 1

    n = len(comps)
    print(f"\n[{args.tag}] judge={args.judge_model}  n={n}")
    for lbl in ("taught", "overshared", "evaded"):
        print(f"  {lbl:11s} {verdicts[lbl]}/{n}  ({verdicts[lbl]/n:.1%})")
    print(f"  unparseable {unparseable}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
