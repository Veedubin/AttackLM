"""Convert upstream benchmark files into AttackLM's normalised item schema.

Only for ``local_scored`` packs — the ones the harness does not already
implement. A ``harness_scored`` pack (CyberMetric) never comes through here;
its harness fetches and grades its own data on purpose, so that we inherit its
validated prompt and answer extraction.

Data is fetched at eval time into a gitignored cache and never committed.
CTI-Bench is CC BY-NC-SA 4.0, so vendoring it into this MIT repo is not an
option; the manifest's ``redistributable: false`` records that and a test
enforces it.

Every loader is pinned to the manifest's ``source.revision``. The upstream
column names below were read from the real files at revision 9237e163, not
from documentation.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable

# data/bench/cache/ — gitignored; see .gitignore
CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "bench" / "cache"

_MCQ_SYSTEM = (
    "You are a cyber threat intelligence analyst. Answer the multiple-choice "
    "question with the single letter of the correct option."
)
_ATE_SYSTEM = (
    "You are a cyber threat intelligence analyst. Extract every MITRE ATT&CK "
    "technique described in the text and list the technique IDs."
)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def load_ctibench_mcq(path: Path) -> list[dict[str, Any]]:
    """cti-mcq.tsv -> items. Columns: URL, Question, Option A-D, Prompt, GT.

    ``GT`` is a bare letter. The upstream ``Prompt`` column carries CTI-Bench's
    own instruction text; we use it verbatim as the system message rather than
    inventing our own, so the measurement stays comparable to published
    numbers.
    """
    items: list[dict[str, Any]] = []
    for idx, row in enumerate(_read_tsv(path)):
        options = "\n".join(
            f"{letter}. {row[f'Option {letter}']}" for letter in ("A", "B", "C", "D")
        )
        items.append(
            {
                "question_id": f"ctibench_mcq_{idx:05d}",
                "category": "cti_knowledge",
                "tier": "mcq",
                "messages": [
                    {"role": "system", "content": row.get("Prompt") or _MCQ_SYSTEM},
                    {"role": "user", "content": f"{row['Question']}\n\n{options}"},
                ],
                "ground_truth": {"type": "mcq_choice", "answer": row["GT"].strip()},
                "metadata": {
                    "pack": "ctibench-mcq",
                    "source_index": idx,
                    "url": row.get("URL"),
                    "license": "CC-BY-NC-SA-4.0",
                },
            }
        )
    return items


def load_ctibench_ate(path: Path) -> list[dict[str, Any]]:
    """cti-ate.tsv -> items. Columns: URL, Platform, Description, Prompt, GT.

    ``GT`` is a comma-separated technique list, e.g. "T1071, T1573, T1083".
    Confirmed against the real file: these are PARENT techniques only, with no
    sub-technique ids — which is why the scorer's default
    ``attack_id_subtechniques="strip"`` is the right reading of this task.
    """
    items: list[dict[str, Any]] = []
    for idx, row in enumerate(_read_tsv(path)):
        gold = [tid.strip() for tid in row["GT"].split(",") if tid.strip()]
        items.append(
            {
                "question_id": f"ctibench_ate_{idx:05d}",
                "category": row.get("Platform") or "enterprise",
                "tier": "ate",
                "messages": [
                    {"role": "system", "content": row.get("Prompt") or _ATE_SYSTEM},
                    {"role": "user", "content": row["Description"]},
                ],
                "ground_truth": {"type": "attack_technique_set", "answer": gold},
                "metadata": {
                    "pack": "ctibench-ate",
                    "source_index": idx,
                    "url": row.get("URL"),
                    "license": "CC-BY-NC-SA-4.0",
                },
            }
        )
    return items


# pack name -> (upstream filename, loader)
SOURCE_LOADERS: dict[str, tuple[str, Callable[[Path], list[dict[str, Any]]]]] = {
    "ctibench-mcq": ("cti-mcq.tsv", load_ctibench_mcq),
    "ctibench-ate": ("cti-ate.tsv", load_ctibench_ate),
}


def fetch_items(pack_name: str, repo: str, revision: str, cache_dir: Path = CACHE_DIR) -> Path:
    """Download an upstream file at a pinned revision and normalise it.

    Returns the path to the cached JSONL. Raises RuntimeError rather than
    silently proceeding when huggingface_hub is unavailable — a benchmark that
    quietly ran on nothing is worse than one that refused to start.
    """
    if pack_name not in SOURCE_LOADERS:
        raise KeyError(f"no loader for pack {pack_name!r}; known: {sorted(SOURCE_LOADERS)}")
    filename, loader = SOURCE_LOADERS[pack_name]

    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{pack_name}.jsonl"
    if out.is_file():
        return out

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to fetch benchmark data "
            f"for {pack_name!r}; install it or supply --questions directly"
        ) from exc

    local = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
    write_items(loader(Path(local)), out)
    return out


def write_items(items: list[dict[str, Any]], path: Path) -> Path:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(item) for item in items) + "\n")
    return path
