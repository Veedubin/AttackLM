#!/usr/bin/env python3
"""Validate a posture-judge model against labelled data.

The taught/overshared judge is only trustworthy if it agrees with human labels,
so this runs a judge model over a labelled fixture (expected posture verdicts)
and reports its agreement + a confusion table. Run it whenever you swap the
judge model. It reuses the SAME Inspect subprocess harness as bench_run, so the
judge may be any local (vllm/hf) or API (openai-api) model.

    ATTACKLM_BENCH_PYTHON=~/.venvs/vllm/bin/python \
    python scripts/bench_judge_validate.py \
        --judge-model Qwen/Qwen2.5-Coder-3B-Instruct --judge-backend vllm \
        --model-arg gpu_memory_utilization=0.85 --model-arg max_model_len=4096

The fixture is JSONL with {"label": <taught|overshared|evaded>, "completion": ...}.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import bench_run  # noqa: E402
from attacklm.bench.adapters.inspect_adapter import (  # noqa: E402
    InspectAdapter,
    RunConfig,
)
from attacklm.bench.items import BenchItem  # noqa: E402
from attacklm.bench.packs import PACKS_DIR, get_pack  # noqa: E402
from attacklm.bench.posture import (  # noqa: E402
    JUDGE_LABELS,
    JudgeError,
    build_judge_prompt,
    judge_agreement,
    parse_judge_verdict,
)

DEFAULT_FIXTURE = REPO / "tests" / "fixtures" / "judge_labeled.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--judge-model", required=True)
    p.add_argument("--judge-backend", default="vllm")
    p.add_argument("--fixture", default=str(DEFAULT_FIXTURE),
                   help="labelled JSONL: {label, completion} per line")
    p.add_argument("--pack", default="applied-attack",
                   help="any local_scored pack; only its jsonl_task plumbing is used")
    p.add_argument("--packs-dir", default=str(PACKS_DIR))
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-arg", action="append", default=[], dest="model_args")
    p.add_argument("--output", help="write the agreement report JSON here")
    return p.parse_args(argv)


def _load_fixture(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("label") not in JUDGE_LABELS:
            raise SystemExit(f"fixture row has label not in {JUDGE_LABELS}: {rec.get('label')!r}")
        rows.append(rec)
    if not rows:
        raise SystemExit(f"no rows in fixture {path}")
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = _load_fixture(Path(args.fixture))
    pack = get_pack(args.pack, Path(args.packs_dir))
    adapter = InspectAdapter()

    # One judge item per fixture row: its prompt is the judge prompt built from
    # a neutral question and the row's completion (the text being graded).
    neutral = BenchItem("x", "validation", "",
                        [{"role": "user", "content": "Classify the assistant's answer."}], {}, {})
    judge_items = [
        BenchItem(
            question_id=f"fx{i}",
            category=row["label"],
            tier="judge-val",
            messages=[{"role": "user", "content": build_judge_prompt(neutral, row["completion"])}],
            ground_truth={},
            metadata={},
        )
        for i, row in enumerate(rows)
    ]

    with tempfile.TemporaryDirectory(prefix="attacklm-judgeval-") as tmp:
        log_dir = Path(tmp)
        cfg = RunConfig(
            model=args.judge_model, backend=args.judge_backend,
            max_tokens=args.max_tokens, temperature=0.0, top_p=1.0, seed=args.seed,
            log_dir=log_dir, model_args=list(args.model_args),
            items_path=bench_run._write_items(judge_items, log_dir),
        )
        samples = bench_run._run_harness(pack, cfg, adapter)

    verdict_text = {s.sample_id: s.completion for s in samples}
    pairs: list[tuple[str, str | None]] = []
    for i, row in enumerate(rows):
        try:
            got = parse_judge_verdict(verdict_text.get(f"fx{i}", ""))
        except JudgeError:
            got = None
        pairs.append((row["label"], got))

    report = {
        "judge_model": args.judge_model,
        "judge_backend": args.judge_backend,
        "fixture": str(args.fixture),
        **judge_agreement(pairs),
        "per_row": [{"expected": e, "judged": g} for e, g in pairs],
    }

    acc = report["accuracy"]
    print(f"judge: {args.judge_model}")
    print(f"agreement: {report['agree']}/{report['n']}"
          + (f" ({acc:.1%})" if acc is not None else ""))
    print("confusion (expected -> judged):")
    for exp in sorted(report["confusion"]):
        print(f"  {exp:11s} -> {report['confusion'][exp]}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
