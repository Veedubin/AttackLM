#!/usr/bin/env python3
"""Upstream -> normalised item conversion, and the licensing guard.

Hermetic: the upstream TSVs are written inline from the REAL column layout
read at revision 9237e163 (cti-mcq.tsv: URL/Question/Option A-D/Prompt/GT;
cti-ate.tsv: URL/Platform/Description/Prompt/GT). No network.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from attacklm.bench.items import load_items
from attacklm.bench.packs import PACKS_DIR, list_packs
from attacklm.bench.scorers import ScoreConfig, score_item
from attacklm.bench.sources import (
    SOURCE_LOADERS,
    load_ctibench_ate,
    load_ctibench_mcq,
    write_items,
)

REPO = Path(__file__).resolve().parent.parent

MCQ_TSV = (
    "URL\tQuestion\tOption A\tOption B\tOption C\tOption D\tPrompt\tGT\n"
    "https://attack.mitre.org/techniques/T1548/\tWhich mitigation?\t"
    "Audit\tExecution Prevention\tOS Configuration\tUAC\tAnswer with a letter.\tB\n"
)

ATE_TSV = (
    "URL\tPlatform\tDescription\tPrompt\tGT\n"
    "https://attack.mitre.org/software/S0066/\tEnterprise\t"
    "3PARA RAT communicates over HTTP and encrypts its traffic.\t"
    "Extract ATT&CK techniques.\tT1071, T1573, T1083\n"
)


def _tsv(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body)
    return p


# --------------------------------------------------------------------------
# MCQ
# --------------------------------------------------------------------------


def test_mcq_conversion_shape(tmp_path):
    items = load_ctibench_mcq(_tsv(tmp_path, "cti-mcq.tsv", MCQ_TSV))
    assert len(items) == 1
    item = items[0]
    assert item["question_id"] == "ctibench_mcq_00000"
    assert item["ground_truth"] == {"type": "mcq_choice", "answer": "B"}
    assert item["metadata"]["license"] == "CC-BY-NC-SA-4.0"


def test_mcq_options_are_rendered_into_the_prompt(tmp_path):
    items = load_ctibench_mcq(_tsv(tmp_path, "cti-mcq.tsv", MCQ_TSV))
    user = items[0]["messages"][1]["content"]
    for letter in ("A.", "B.", "C.", "D."):
        assert letter in user
    assert "Execution Prevention" in user


def test_mcq_uses_upstream_prompt_verbatim(tmp_path):
    """Keeps the measurement comparable to CTI-Bench's published numbers."""
    items = load_ctibench_mcq(_tsv(tmp_path, "cti-mcq.tsv", MCQ_TSV))
    assert items[0]["messages"][0]["content"] == "Answer with a letter."


# --------------------------------------------------------------------------
# ATE
# --------------------------------------------------------------------------


def test_ate_conversion_shape(tmp_path):
    items = load_ctibench_ate(_tsv(tmp_path, "cti-ate.tsv", ATE_TSV))
    item = items[0]
    assert item["question_id"] == "ctibench_ate_00000"
    assert item["ground_truth"]["type"] == "attack_technique_set"
    assert item["ground_truth"]["answer"] == ["T1071", "T1573", "T1083"]
    assert item["category"] == "Enterprise"


def test_ate_gold_is_parent_techniques_only(tmp_path):
    """Why the scorer defaults to attack_id_subtechniques='strip'."""
    items = load_ctibench_ate(_tsv(tmp_path, "cti-ate.tsv", ATE_TSV))
    assert all("." not in tid for tid in items[0]["ground_truth"]["answer"])


# --------------------------------------------------------------------------
# round trip through the real loader and scorer
# --------------------------------------------------------------------------


def test_converted_items_load_and_grade(tmp_path):
    out = write_items(
        load_ctibench_ate(_tsv(tmp_path, "cti-ate.tsv", ATE_TSV)), tmp_path / "items.jsonl"
    )
    items = load_items(out)
    perfect = score_item(items[0], "Techniques: T1071, T1573, T1083", ScoreConfig())
    assert perfect.score == 1.0 and perfect.valid


def test_refusal_is_invalid_not_a_genuine_zero(tmp_path):
    """A refusal must stay distinguishable from a wrong answer."""
    out = write_items(
        load_ctibench_mcq(_tsv(tmp_path, "cti-mcq.tsv", MCQ_TSV)), tmp_path / "items.jsonl"
    )
    refusal = score_item(load_items(out)[0], "I cannot help with that.", ScoreConfig())
    assert refusal.valid is False


def test_every_shipped_local_pack_has_a_loader():
    for pack in list_packs(PACKS_DIR):
        # A fetch loader exists to pull EXTERNAL data. An authored pack
        # (source.kind == local) ships its items in-repo and has nothing to
        # fetch, so it is exempt by design.
        if pack.mode == "local_scored" and pack.source.kind != "local":
            assert pack.name in SOURCE_LOADERS, f"{pack.name} has no source loader"


def test_unknown_pack_raises():
    from attacklm.bench.sources import fetch_items

    with pytest.raises(KeyError, match="nope"):
        fetch_items("nope", "repo", "rev")


# --------------------------------------------------------------------------
# licensing guard
# --------------------------------------------------------------------------


def test_non_redistributable_data_is_gitignored():
    """CC BY-NC-SA and unlicensed data must never be committable here."""
    probe = "data/bench/cache/ctibench-mcq.jsonl"
    result = subprocess.run(
        ["git", "check-ignore", "-q", probe], cwd=REPO, check=False
    )
    assert result.returncode == 0, f"{probe} is NOT gitignored; fetched data could be committed"


def test_no_fetched_data_is_tracked_by_git():
    tracked = subprocess.run(
        ["git", "ls-files", "data/bench/cache"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert tracked.stdout.strip() == "", f"fetched data is tracked: {tracked.stdout}"


# --- code-review fix #5: single-choice prompt must not go plural ---

MCQ_TSV_NO_PROMPT = (
    "URL\tQuestion\tOption A\tOption B\tOption C\tOption D\tPrompt\tGT\n"
    "https://attack.mitre.org/techniques/T1548/\tWhich mitigation?\t"
    "Audit\tExecution Prevention\tOS Configuration\tUAC\t\tB\n"
)


def test_ctibench_mcq_fallback_prompt_is_single_choice(tmp_path):
    """ctibench-mcq is single-choice; its fallback prompt must not ask for
    'the letter(s) of every correct option' -- that plural phrasing changed a
    single-choice pack's prompt and broke comparability with recorded runs."""
    from attacklm.bench.sources import _MCQ_SYSTEM_SINGLE

    items = load_ctibench_mcq(_tsv(tmp_path, "cti-mcq.tsv", MCQ_TSV_NO_PROMPT))
    sys_msg = items[0]["messages"][0]["content"]
    assert sys_msg == _MCQ_SYSTEM_SINGLE
    assert "letter(s)" not in sys_msg
    assert "single letter" in sys_msg


def test_multi_select_packs_keep_the_plural_prompt():
    """SecBench/SecEval are genuinely multi-select and must ask for every
    correct option."""
    from attacklm.bench.sources import _MCQ_SYSTEM

    assert "letter(s) of every correct option" in _MCQ_SYSTEM
