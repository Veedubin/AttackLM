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
import json
from pathlib import Path
from typing import Any, Callable

# data/bench/cache/ — gitignored; see .gitignore
CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "bench" / "cache"

_MCQ_SYSTEM = (
    "You are a security expert answering a multiple-choice question. "
    "Reply with the letter(s) of every correct option."
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


_LETTERS = ("A", "B", "C", "D", "E", "F")


def _gt_for_letters(label: str) -> dict[str, Any]:
    """Single letter -> mcq_choice; several -> mcq_multi.

    Choosing per item rather than per pack is not a nicety: SecEval is ~43%
    multiple-SELECT ("AC", "BD"), and grading those with a single-letter
    extractor would be flatly wrong rather than merely imprecise.
    """
    letters = [c for c in label.strip().upper() if c in _LETTERS]
    if len(letters) == 1:
        return {"type": "mcq_choice", "answer": letters[0]}
    return {"type": "mcq_multi", "answer": letters}


def load_secbench_mcq(path: Path, language: str = "English") -> list[dict[str, Any]]:
    """SecBench MCQs_2730.jsonl -> items.

    Columns: question, answers (list), label, language, ability, domain.

    Filtered to English by default: 2,069 of the 2,730 public items are
    Chinese, and mixing languages would confound a capability delta with a
    multilingual one. Pass language=None to keep everything.
    """
    items: list[dict[str, Any]] = []
    for idx, row in enumerate(
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ):
        if language and row.get("language") != language:
            continue
        options = "\n".join(
            f"{letter}. {text}" for letter, text in zip(_LETTERS, row["answers"])
        )
        items.append(
            {
                "question_id": f"secbench_mcq_{idx:05d}",
                "category": row.get("domain") or "uncategorised",
                "tier": row.get("ability") or "mcq",
                "messages": [
                    {"role": "system", "content": _MCQ_SYSTEM},
                    {"role": "user", "content": f"{row['question']}\n\n{options}"},
                ],
                "ground_truth": _gt_for_letters(row["label"]),
                "metadata": {
                    "pack": "secbench-en",
                    "source_index": idx,
                    "language": row.get("language"),
                    "license": "MIT",
                },
            }
        )
    return items


def load_seceval(path: Path) -> list[dict[str, Any]]:
    """SecEval questions.json -> items.

    Keys: id, source, question, choices (pre-lettered "A: ..."), answer,
    keyword, topics. Roughly 43% of answers are multi-letter, so the item type
    is chosen per item.
    """
    data = json.loads(Path(path).read_text())
    items: list[dict[str, Any]] = []
    for idx, row in enumerate(data):
        topics = row.get("topics") or []
        items.append(
            {
                "question_id": f"seceval_{idx:05d}",
                "category": (topics[0] if topics else row.get("source")) or "uncategorised",
                "tier": "mcq",
                "messages": [
                    {"role": "system", "content": _MCQ_SYSTEM},
                    {
                        "role": "user",
                        "content": row["question"] + "\n\n" + "\n".join(row["choices"]),
                    },
                ],
                "ground_truth": _gt_for_letters(row["answer"]),
                "metadata": {
                    "pack": "seceval",
                    "source_index": idx,
                    "upstream_id": row.get("id"),
                    "license": "CC-BY-NC-SA-4.0",
                },
            }
        )
    return items


# pack name -> (upstream filename, loader)
SOURCE_LOADERS: dict[str, tuple[str, Callable[[Path], list[dict[str, Any]]]]] = {
    "ctibench-mcq": ("cti-mcq.tsv", load_ctibench_mcq),
    "ctibench-ate": ("cti-ate.tsv", load_ctibench_ate),
    "secbench-en": ("data/MCQs_2730.jsonl", load_secbench_mcq),
    "seceval": ("questions.json", load_seceval),
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
