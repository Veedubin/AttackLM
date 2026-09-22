"""Inspect task definitions AttackLM ships for ``local_scored`` packs.

A ``harness_scored`` pack names a task the harness already implements
(``inspect_evals/cybermetric_500``), which owns its dataset and its scorer. A
``local_scored`` pack instead owns its own items, so the harness is used only
to generate completions and this module is the bridge: Inspect loads
``jsonl_task`` and is told where our normalised JSONL lives.

    inspect eval <this file>@jsonl_task -T items=/path/to/items.jsonl

Deliberately NO scorer is attached. Grading happens in
``attacklm.bench.scorers`` against the pack's answer key, so the extraction
and invalid-response policies stay under our control and get recorded in the
report's ``resolved_config``. Inspect here is a generation engine, nothing
more.

This is the only module besides ``inspect_adapter`` permitted to import
``inspect_ai``; it runs inside the harness's own interpreter
(``ATTACKLM_BENCH_PYTHON``), never inside AttackLM's.
"""

from __future__ import annotations

import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate


def _load_samples(items_path: str) -> list[Sample]:
    """Read our normalised item JSONL into Inspect Samples.

    ``id`` is carried through verbatim as the ``question_id``: every downstream
    comparison pairs by it, so losing or renaming it here would silently
    scramble a run. ``category`` rides along in metadata because the harness
    log is where it is read back from.
    """
    samples: list[Sample] = []
    for line in Path(items_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)

        messages = rec.get("messages") or []
        system = next(
            (m["content"] for m in messages if m.get("role") == "system"), None
        )
        user = "\n\n".join(
            m["content"] for m in messages if m.get("role") == "user"
        )

        samples.append(
            Sample(
                id=rec["question_id"],
                input=user,
                metadata={
                    "category": rec.get("category", "uncategorised"),
                    "tier": rec.get("tier", ""),
                    **({"system": system} if system else {}),
                },
            )
        )
    return samples


@task
def jsonl_task(items: str):
    """Generate completions for a normalised AttackLM item file."""
    return Task(dataset=_load_samples(items), solver=generate())
