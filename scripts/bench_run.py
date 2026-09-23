#!/usr/bin/env python3
"""AttackLM — capability benchmark runner.

Shells out to an eval harness (Inspect AI), normalises its log into the report
schema ``queue compare`` consumes, and writes it to --output.

Two pack modes (see src/attacklm/bench/packs.py):

  harness_scored  The harness owns the dataset AND the scoring. We normalise
                  its per-sample scores. Used for benchmarks inspect_evals
                  already implements, so we inherit their validated prompt and
                  answer extraction rather than second-guessing it.

  local_scored    We own the items and the scoring; the harness only
                  generates completions.

Examples:

  # harness_scored
  python scripts/bench_run.py --pack cybermetric-500 \\
      --base-model models/merged/attacklm-3b-16g \\
      --output evals/bench_cybermetric.json

  # local_scored, with our own questions
  python scripts/bench_run.py --pack my-pack --questions data/bench/mine.jsonl \\
      --base-model Qwen/Qwen2.5-Coder-3B-Instruct --rung 200 \\
      --output evals/bench_mine.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from attacklm.bench.adapters.inspect_adapter import (  # noqa: E402
    InspectAdapter,
    ParsedSample,
    RunConfig,
)
from attacklm.bench.contamination import (  # noqa: E402
    DEFAULT_THRESHOLD,
    check_contamination,
)
from attacklm.bench.items import BenchItem, load_items, sample_items  # noqa: E402
from attacklm.bench.packs import PACKS_DIR, Pack, get_pack  # noqa: E402
from attacklm.bench.posture import (  # noqa: E402
    PostureConfig,
    build_judge_prompt,
    parse_judge_verdict,
    score_posture,
)
from attacklm.bench.report import build_report  # noqa: E402
from attacklm.bench.scorers import (  # noqa: E402
    INVALID_POLICIES,
    InvalidResponseError,
    ItemScore,
    ScoreConfig,
    score_item,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AttackLM capability benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pack", required=True, help="pack name under data/bench/packs/")
    p.add_argument("--packs-dir", default=str(PACKS_DIR), help="where pack manifests live")
    p.add_argument("--questions", help="item JSONL (local_scored packs only)")
    p.add_argument("--output", default="evals/bench.json")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", help="LoRA adapter path")

    gen = p.add_argument_group("inference")
    gen.add_argument("--backend", default="vllm",
                 choices=["vllm", "hf", "openai-api", "mock"],
                 help="mock = Inspect's stub provider; no GPU, for smoke-testing the pipeline")
    gen.add_argument("--base-url", help="for --backend openai-api")
    gen.add_argument("--max-tokens", type=int, default=512)
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--top-p", type=float, default=1.0)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--stop", action="append", default=[],
                     help="stop sequence; repeatable. None by default on purpose")
    gen.add_argument("--max-connections", type=int)
    gen.add_argument("--model-arg", action="append", default=[], dest="model_args",
                     help="native harness model arg, e.g. gpu_memory_utilization=0.85")

    samp = p.add_argument_group("sampling")
    samp.add_argument("--rung", type=int, help="item count; omit for the full set")
    samp.add_argument("--sample-seed", type=int, default=42)
    samp.add_argument("--num-shots", type=int, default=0)

    sc = p.add_argument_group("scoring")
    sc.add_argument("--answer-regex", help="override answer extraction (local_scored)")
    sc.add_argument("--invalid-policy", default="count_wrong", choices=list(INVALID_POLICIES))

    jd = p.add_argument_group("posture judge (taught vs overshared)")
    jd.add_argument("--judge-model",
                    help="model that grades ANSWERED completions taught/overshared/evaded "
                         "via a second harness pass; omit to keep posture deterministic "
                         "(refused/answered/evaded only)")
    jd.add_argument("--judge-backend",
                    help="backend for the judge model (default: same as --backend)")
    jd.add_argument("--judge-max-tokens", type=int, default=256,
                    help="generation cap for the judge's verdict")

    dc = p.add_argument_group("contamination (the \"SAE correction\")")
    dc.add_argument("--no-decontam", action="store_true",
                    help="skip the overlap check; the report then carries a null score_clean")
    dc.add_argument("--decontam-threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="Jaccard at or above which an item counts as contaminated")
    dc.add_argument("--training-records",
                    help="JSONL of training records to check against; "
                         "defaults to the attacklm-dataset corpus when present")

    p.add_argument("--keep-logs", help="copy the harness log here instead of discarding it")
    return p.parse_args(argv)


def _run_harness(pack: Pack, cfg: RunConfig, adapter: InspectAdapter) -> list[ParsedSample]:
    """Invoke the harness and parse its log. Patched out in tests."""
    from attacklm.bench.adapters.inspect_adapter import bench_env

    argv = adapter.build_argv(pack, cfg)
    print("harness:", " ".join(argv), flush=True)
    subprocess.run(argv, check=True, env=bench_env())
    return adapter.parse_log(cfg.log_dir)


def _apply_invalid_policy(
    question_id: str, category: str, policy: str
) -> ItemScore:
    """Decide an unparseable sample's fate, identically for both pack modes."""
    if policy == "fail_run":
        raise InvalidResponseError(f"{question_id}: harness could not parse an answer")
    return ItemScore(question_id, category, None if policy == "exclude" else 0.0, None, False)


def _score_harness_mode(samples: list[ParsedSample], policy: str) -> list[ItemScore]:
    scores: list[ItemScore] = []
    for s in samples:
        category = s.category or "uncategorised"
        if s.valid and s.score is not None:
            scores.append(ItemScore(s.sample_id, category, s.score, s.answer, True))
        else:
            scores.append(_apply_invalid_policy(s.sample_id, category, policy))
    return scores


def _score_local_mode(
    items, samples: list[ParsedSample], cfg: ScoreConfig
) -> list[ItemScore]:
    # Pair by id, never by position.
    completions = {s.sample_id: s.completion for s in samples}
    return [score_item(item, completions.get(item.question_id, ""), cfg) for item in items]


def _score_posture_local(items, samples: list[ParsedSample], pcfg: PostureConfig):
    """Posture-score the SAME completions the capability scorer saw.

    Posture is orthogonal to capability, so it runs over the identical
    id-keyed completions rather than a second generation pass.
    """
    completions = {s.sample_id: s.completion for s in samples}
    return [score_posture(item, completions.get(item.question_id, ""), pcfg) for item in items]


def _run_judge_pass(pack, items, samples, args, adapter, pcfg: PostureConfig):
    """Refine ANSWERED posture labels to taught/overshared/evaded via a judge.

    The judge runs through the SAME Inspect subprocess as a second pass over
    judge-prompt items, so no vLLM enters this venv and any model (local or
    openai-api) can judge. Only items the deterministic tier called 'answered'
    are judged -- refusals and empty answers are already decided.
    """
    import dataclasses

    completions = {s.sample_id: s.completion for s in samples}

    # Which items actually reached the judge tier?
    answered = [
        item for item in items
        if score_posture(item, completions.get(item.question_id, ""), pcfg).label == "answered"
    ]
    if not answered:
        return _score_posture_local(items, samples, pcfg)

    # One judge item per answered completion: its prompt IS the judge prompt.
    judge_items = [
        BenchItem(
            question_id=item.question_id,
            category=item.category,
            tier="judge",
            messages=[{"role": "user",
                       "content": build_judge_prompt(item, completions.get(item.question_id, ""))}],
            ground_truth={},
            metadata={},
        )
        for item in answered
    ]

    with tempfile.TemporaryDirectory(prefix="attacklm-judge-") as tmp:
        jlog = Path(tmp)
        judge_cfg = dataclasses.replace(
            RunConfig(model=args.judge_model),
            model=args.judge_model,
            backend=args.judge_backend or args.backend,
            max_tokens=args.judge_max_tokens,
            temperature=0.0, top_p=1.0, seed=args.seed,
            base_url=args.base_url, model_args=list(args.model_args),
            log_dir=jlog,
            items_path=_write_items(judge_items, jlog),
        )
        judge_samples = _run_harness(pack, judge_cfg, adapter)
        if args.keep_logs:
            dest = Path(args.keep_logs)
            dest.mkdir(parents=True, exist_ok=True)
            for log in jlog.glob("*.json"):
                dest.joinpath("judge_" + log.name).write_text(log.read_text())

    verdict_text = {s.sample_id: s.completion for s in judge_samples}

    def _judge_fn(item, _completion, _qid_lookup=verdict_text):
        # parse_judge_verdict raises JudgeError on unparseable output, which
        # score_posture catches -> the item stays 'answered', unlabelled.
        return parse_judge_verdict(_qid_lookup.get(item.question_id, ""))

    return [
        score_posture(item, completions.get(item.question_id, ""), pcfg, judge_fn=_judge_fn)
        for item in items
    ]


def _write_items(items: list[BenchItem], out_dir: Path) -> Path:
    """Serialise the sampled items for the harness to read.

    Written per-run beside the harness log, so the file the harness actually
    saw matches the rung that was sampled.
    """
    path = out_dir / "items.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({
                "question_id": i.question_id, "category": i.category, "tier": i.tier,
                "messages": i.messages, "ground_truth": i.ground_truth,
                "metadata": i.metadata,
            })
            for i in items
        )
    )
    return path


def _load_training_records(path: str | None) -> list[dict] | None:
    """Records to check benchmark items against, or None when unavailable.

    None means "no check was possible", which the report records as an
    explicitly null score_clean. It must never silently become an empty list,
    which would read as "checked, nothing overlapped".
    """
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    records = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pack = get_pack(args.pack, Path(args.packs_dir))
    adapter = InspectAdapter()

    items = []
    if pack.mode == "local_scored":
        # --questions wins; otherwise an authored (source.kind == local) pack
        # names its own in-repo item file via source.repo, so it is
        # self-contained without the operator repeating the path.
        questions = args.questions
        if not questions and pack.source.kind == "local" and pack.source.repo:
            questions = pack.source.repo
        if not questions:
            raise SystemExit(
                f"pack {pack.name!r} is local_scored; --questions is required"
            )
        items = sample_items(load_items(Path(questions)), args.rung, seed=args.sample_seed)

    with tempfile.TemporaryDirectory(prefix="attacklm-bench-") as tmp:
        log_dir = Path(tmp)
        cfg = RunConfig(
            model=args.base_model, adapter=args.adapter, backend=args.backend,
            max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p,
            seed=args.seed, stop=list(args.stop), base_url=args.base_url,
            num_shots=args.num_shots, max_connections=args.max_connections,
            log_dir=log_dir, limit=args.rung, model_args=list(args.model_args),
            items_path=_write_items(items, log_dir) if items else None,
        )
        samples = _run_harness(pack, cfg, adapter)

        if args.keep_logs:
            dest = Path(args.keep_logs)
            dest.mkdir(parents=True, exist_ok=True)
            for log in log_dir.glob("*.json"):
                dest.joinpath(log.name).write_text(log.read_text())

    if pack.mode == "harness_scored":
        scores = _score_harness_mode(samples, args.invalid_policy)
    else:
        scores = _score_local_mode(
            items,
            samples,
            ScoreConfig(answer_regex=args.answer_regex, invalid_policy=args.invalid_policy),
        )

    # Posture is scored over the SAME completions (Phase 3). Only local_scored
    # packs carry their own items, which posture scoring needs for identity.
    posture_cfg = PostureConfig()
    posture_scores = None
    if pack.posture and pack.mode == "local_scored" and items:
        if args.judge_model:
            posture_scores = _run_judge_pass(pack, items, samples, args, adapter, posture_cfg)
        else:
            posture_scores = _score_posture_local(items, samples, posture_cfg)

    # Contamination is post-processing over one MinHash pass, so both the raw
    # and corrected numbers come from this single run at no extra GPU cost.
    contamination = None
    if not args.no_decontam:
        checked_items = items or [
            BenchItem(s.question_id, s.category, "", [], {}, {}) for s in scores
        ]
        contamination = check_contamination(
            checked_items,
            _load_training_records(args.training_records),
            threshold=args.decontam_threshold,
        )

    info = adapter.probe()
    report = build_report(pack, scores, {
        "model": args.base_model,
        "adapter": args.adapter,
        "backend": args.backend,
        "harness": {"name": info.name, "version": info.version},
        # FULLY RESOLVED, not merely what the user typed. The QCRI audit found
        # 61% of pipeline config unreproducible from documentation; a report
        # that cannot be replayed is not evidence.
        "resolved_config": {
            "max_tokens": args.max_tokens, "temperature": args.temperature,
            "top_p": args.top_p, "seed": args.seed, "stop": list(args.stop),
            "sample_seed": args.sample_seed, "rung": args.rung,
            "num_shots": args.num_shots, "answer_regex": args.answer_regex,
            "invalid_policy": args.invalid_policy, "backend": args.backend,
            "base_url": args.base_url, "model_args": list(args.model_args),
            "pack_revision": pack.source.revision,
            "decontam": (
                None if args.no_decontam else {"threshold": args.decontam_threshold}
            ),
            "posture": (
                {
                    "enabled": True,
                    "refusal_scan_chars": posture_cfg.refusal_scan_chars,
                    "refusal_patterns": (
                        "builtin" if posture_cfg.refusal_patterns is None else "custom"
                    ),
                    "judge_model": args.judge_model,
                    "judge_backend": (
                        (args.judge_backend or args.backend) if args.judge_model else None
                    ),
                    "judge_max_tokens": args.judge_max_tokens if args.judge_model else None,
                }
                if posture_scores is not None
                else {"enabled": False}
            ),
        },
    }, contamination, posture=posture_scores)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    raw = report["summary"]["score_raw"]
    clean = report["summary"].get("score_clean")
    print(f"{pack.name}: raw={raw['score']} n={raw['n']} invalid={raw['invalid']}")
    if clean:
        print(
            f"{' ' * len(pack.name)}  clean={clean['score']} n={clean['n']} "
            f"(contamination {report['summary']['contamination_rate']:.1%})"
        )
    else:
        reason = report["summary"].get("score_clean_reason")
        print(f"{' ' * len(pack.name)}  clean=n/a ({reason})")
    posture_summary = report["summary"].get("posture")
    if posture_summary:
        pad = " " * len(pack.name)
        print(
            f"{pad}  posture: refused={posture_summary['refusal_rate']:.1%} "
            f"answered={posture_summary['answered_rate']:.1%} "
            f"evaded={posture_summary['evaded_rate']:.1%} (n={posture_summary['n']})"
        )
        if "taught_rate" in posture_summary:
            print(
                f"{pad}  judged:  taught={posture_summary['taught_rate']:.1%} "
                f"overshared={posture_summary['overshared_rate']:.1%}"
            )
    print(f"Report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
