#!/usr/bin/env python3
"""The domain benchmark's question file must ship with the repo.

Its absence is what has kept `scripts/domain_bench.py` and attack 7
(`audit_calibration`) unrunnable since v0.20.0.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
QUESTIONS = REPO / "data" / "bench" / "questions.jsonl"


def test_questions_file_exists():
    assert QUESTIONS.is_file(), f"{QUESTIONS} missing — domain_bench and attack 7 cannot run"


def test_questions_have_required_keys():
    records = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    assert len(records) >= 99
    for rec in records:
        assert "question_id" in rec
        assert "category" in rec
        assert "messages" in rec
        assert "ground_truth" in rec
