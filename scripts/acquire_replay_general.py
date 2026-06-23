#!/usr/bin/env python3
"""Acquire and sample license-clean general-domain replay corpora for AttackLM.

This script downloads small, stratified samples from permissive public datasets
and writes them into the per-source layout:

    data/datasets/buckets/sources/replay-general/
        base/replay/
            data_code.jsonl
            data_conversation.jsonl
            data_factual.jsonl
            data_reasoning.jsonl

You must verify the exact license of each upstream dataset before running.
The default targets are believed to be permissive as of 2026-06-22, but licenses
change. See `data/datasets/buckets/sources/replay-general/SOURCE.md` for links.

Usage:
    uv run python scripts/acquire_replay_general.py --check    # print plan, no downloads
    uv run python scripts/acquire_replay_general.py            # download + sample
    uv run python scripts/acquire_replay_general.py --seed 123 --samples 500
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPLAY_GENERAL_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "datasets"
    / "buckets"
    / "sources"
    / "replay-general"
    / "base"
    / "replay"
)

# Upstream dataset configs. license is the SPDX identifier we believe applies;
# verify before ingest.
DATASETS: dict[str, dict[str, Any]] = {
    "code": {
        "source_name": "the-stack-v2-permissive",
        "source_uri": "https://huggingface.co/datasets/bigcode/the-stack-v2--smol-permissible",
        "license": "Apache-2.0 AND MIT AND BSD-3-Clause",
        "license_uri": "https://huggingface.co/datasets/bigcode/the-stack-v2-0.3",
        "hf_name": "bigcode/the-stack-v2-0.3",
        "config": "Python",
        "split": "train",
        "column": "content",
        "filter": None,
    },
    "conversation": {
        "source_name": "OpenAssistant",
        "source_uri": "https://huggingface.co/datasets/OpenAssistant/oasst1",
        "license": "Apache-2.0",
        "license_uri": "https://www.apache.org/licenses/LICENSE-2.0",
        "hf_name": "OpenAssistant/oasst1",
        "config": None,
        "split": "train",
        "column": "text",
        "filter": lambda ex: ex.get("role") == "assistant",
    },
    "factual": {
        "source_name": "SlimPajama",
        "source_uri": "https://huggingface.co/datasets/cerebras/SlimPajama-627B",
        "license": "Apache-2.0",
        "license_uri": "https://www.apache.org/licenses/LICENSE-2.0",
        "hf_name": "cerebras/SlimPajama-627B",
        "config": None,
        "split": "train",
        "column": "text",
        "filter": None,
    },
    "reasoning": {
        "source_name": "natural_instructions",
        "source_uri": "https://huggingface.co/datasets/Muennighoff/natural-instructions",
        "license": "Apache-2.0",
        "license_uri": "https://www.apache.org/licenses/LICENSE-2.0",
        "hf_name": "Muennighoff/natural-instructions",
        "config": None,
        "split": "train",
        "column": "definition",
        "filter": None,
    },
}

# Default domain mix for the starter replay-general corpus.
DEFAULT_DOMAIN_MIX: dict[str, float] = {
    "code": 0.30,
    "conversation": 0.25,
    "factual": 0.25,
    "reasoning": 0.20,
}


def _text_to_messages(text: str, system: str = "") -> list[dict[str, str]]:
    """Turn a plain-text replay snippet into an AttackLM-style messages triple."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": "Continue or summarize the following."})
    messages.append({"role": "assistant", "content": text})
    return messages


def _build_record(text: str, domain: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Stamp provenance onto a single replay record."""
    return {
        "messages": _text_to_messages(
            text, system=f"You are a helpful assistant. Domain: {domain}."
        ),
        "source": cfg["source_name"],
        "source_uri": cfg["source_uri"],
        "license": cfg["license"],
        "license_uri": cfg["license_uri"],
        "rights_contact": "see data/REMOVAL.md",
        "attribution_required": True,
        "derived_from": cfg["source_name"],
        "domain": domain,
    }


def _sample_hf_dataset(
    cfg: dict[str, Any],
    n_samples: int,
    seed: int,
    max_text_len: int = 4096,
) -> list[dict[str, Any]]:
    """Download and sample n_examples from a HuggingFace dataset config."""
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": cfg["split"], "streaming": True}
    if cfg.get("config"):
        kwargs["name"] = cfg["config"]

    ds = load_dataset(cfg["hf_name"], **kwargs)
    rng = random.Random(seed)

    samples: list[str] = []
    for i, ex in enumerate(ds):
        if cfg.get("filter") and not cfg["filter"](ex):
            continue
        text = ex.get(cfg["column"], "")
        if not isinstance(text, str) or not text.strip():
            continue
        text = text.strip()[:max_text_len]
        # Reservoir sample
        if len(samples) < n_samples:
            samples.append(text)
        else:
            j = rng.randint(0, i)
            if j < n_samples:
                samples[j] = text
        if i >= 5_000_000:
            # Hard cap streaming reads to keep acquisition bounded.
            break

    return samples


def acquire_domain(
    domain: str,
    n_samples: int,
    seed: int,
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Acquire n_samples records for a given domain."""
    cfg = cfg or DATASETS[domain]
    texts = _sample_hf_dataset(cfg, n_samples, seed)
    records = [_build_record(t, domain, cfg) for t in texts]
    print(f"  {domain}: collected {len(records)} records from {cfg['source_name']}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Acquire general-domain replay corpus for AttackLM"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Total number of replay examples to download (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print the planned download targets and license notes without downloading.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPLAY_GENERAL_DIR),
        help="Directory to write replay JSONL files (default: replay-general/base/replay/)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.check:
        print("Planned replay-general corpus:")
        print(f"  Total samples: {args.samples}")
        print(f"  Output dir:    {output_dir}")
        print("  Domain mix:")
        for domain, ratio in DEFAULT_DOMAIN_MIX.items():
            n = int(args.samples * ratio)
            cfg = DATASETS[domain]
            print(
                f"    {domain:14s} {n:4d}  source={cfg['source_name']:25s} "
                f"license={cfg['license']}"
            )
        print("\nVerify each license before running without --check.")
        return 0

    print("Acquiring replay-general corpus...")
    print(f"  Output dir: {output_dir}")
    for domain, ratio in DEFAULT_DOMAIN_MIX.items():
        n = int(args.samples * ratio)
        records = acquire_domain(domain, n, args.seed)
        out_path = output_dir / f"data_{domain}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"    → wrote {out_path} ({len(records)} records)")

    print("\nDone. Next: run a dry-run with replay to validate mixing:")
    print(
        "  uv run python scripts/train_all.py --single-model --dataset base/ "
        "--replay-source replay-general/ --replay-ratio 0.07 --dry-run"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
