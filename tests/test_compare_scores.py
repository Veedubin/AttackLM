#!/usr/bin/env python3
"""Tests for scripts/compare_scores.py — Score Comparison Tool.

These tests are hermetic: they use temporary TSV files and no GPU.

Run with:
    python -m pytest tests/test_compare_scores.py -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Make the scripts/ dir importable
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import compare_scores as cs  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tsv(path: Path, rows: list[dict]) -> None:
    """Write a scores TSV file from a list of dicts."""
    if not rows:
        path.write_text("")
        return
    header = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(h, "")) for h in header) + "\n")


# ---------------------------------------------------------------------------
# Tests: compute_verdict
# ---------------------------------------------------------------------------


class TestComputeVerdict(unittest.TestCase):
    """Verify verdict logic."""

    def test_improved_nll_better_lcp_better(self):
        """delta_nll < -0.01 AND delta_lcp > 0.02 → improved."""
        self.assertEqual(cs.compute_verdict(-0.05, 0.05), "improved")

    def test_improved_boundary(self):
        """Just past boundary should be improved."""
        self.assertEqual(cs.compute_verdict(-0.0101, 0.0201), "improved")

    def test_improved_exact_boundary_neutral(self):
        """Exactly at boundary (not past) should be neutral."""
        self.assertEqual(cs.compute_verdict(-0.01, 0.02), "neutral")

    def test_regressed_nll_worse(self):
        """delta_nll > 0.02 → regressed."""
        self.assertEqual(cs.compute_verdict(0.03, 0.0), "regressed")

    def test_regressed_lcp_worse(self):
        """delta_lcp < -0.05 → regressed."""
        self.assertEqual(cs.compute_verdict(0.0, -0.06), "regressed")

    def test_regressed_both_worse(self):
        """Both worse → regressed."""
        self.assertEqual(cs.compute_verdict(0.05, -0.1), "regressed")

    def test_neutral_small_changes(self):
        """Small changes in both directions → neutral."""
        self.assertEqual(cs.compute_verdict(0.0, 0.0), "neutral")

    def test_neutral_nll_improved_but_lcp_not_enough(self):
        """NLL improved but LCP not enough → neutral."""
        self.assertEqual(cs.compute_verdict(-0.05, 0.01), "neutral")

    def test_neutral_lcp_improved_but_nll_not_enough(self):
        """LCP improved but NLL not enough → neutral."""
        self.assertEqual(cs.compute_verdict(-0.005, 0.05), "neutral")

    def test_neutral_nll_worse_but_lcp_good(self):
        """NLL slightly worse but LCP good → regressed (nll > 0.02)."""
        self.assertEqual(cs.compute_verdict(0.025, 0.05), "regressed")

    def test_neutral_lcp_worse_but_nll_good(self):
        """LCP slightly worse but NLL good → regressed (lcp < -0.05)."""
        self.assertEqual(cs.compute_verdict(-0.05, -0.051), "regressed")


# ---------------------------------------------------------------------------
# Tests: load_scores_tsv
# ---------------------------------------------------------------------------


class TestLoadScoresTSV(unittest.TestCase):
    """Verify TSV loading."""

    def test_load_valid_tsv(self):
        """Valid TSV should return dict keyed by prompt_id."""
        rows = [
            {
                "prompt_id": "p1",
                "bucket": "mitre",
                "category": "T1569.002",
                "avg_nll": "1.5",
                "first_token_matches": "1",
                "avg_greedy_lcp": "0.8",
                "tokens_generated": "10",
                "ref_tokens": "5",
            },
            {
                "prompt_id": "p2",
                "bucket": "metasploit",
                "category": "exploit",
                "avg_nll": "2.0",
                "first_token_matches": "0",
                "avg_greedy_lcp": "0.5",
                "tokens_generated": "8",
                "ref_tokens": "5",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.tsv"
            _write_tsv(path, rows)
            scores = cs.load_scores_tsv(str(path))
            self.assertEqual(len(scores), 2)
            self.assertIn("p1", scores)
            self.assertIn("p2", scores)
            self.assertEqual(scores["p1"]["avg_nll"], 1.5)
            self.assertEqual(scores["p1"]["first_token_matches"], 1)
            self.assertEqual(scores["p1"]["avg_greedy_lcp"], 0.8)

    def test_load_empty_tsv(self):
        """Empty TSV should return empty dict."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.tsv"
            path.write_text("")
            scores = cs.load_scores_tsv(str(path))
            self.assertEqual(scores, {})

    def test_load_tsv_with_only_header(self):
        """TSV with only header should return empty dict."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "header_only.tsv"
            path.write_text("prompt_id\tbucket\tcategory\tavg_nll\n")
            scores = cs.load_scores_tsv(str(path))
            self.assertEqual(scores, {})

    def test_numeric_conversion_handles_bad_values(self):
        """Bad numeric values should default to 0."""
        rows = [
            {
                "prompt_id": "p1",
                "bucket": "mitre",
                "category": "T1569.002",
                "avg_nll": "not_a_number",
                "first_token_matches": "bad",
                "avg_greedy_lcp": "0.8",
                "tokens_generated": "10",
                "ref_tokens": "5",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.tsv"
            _write_tsv(path, rows)
            scores = cs.load_scores_tsv(str(path))
            self.assertEqual(scores["p1"]["avg_nll"], 0.0)
            self.assertEqual(scores["p1"]["first_token_matches"], 0)


# ---------------------------------------------------------------------------
# Tests: Integration (real TSV files)
# ---------------------------------------------------------------------------


class TestIntegration(unittest.TestCase):
    """End-to-end test with real temporary TSV files."""

    def test_main_creates_delta_report(self):
        """main() should create a valid delta TSV."""
        baseline_rows = [
            {
                "prompt_id": "p1",
                "bucket": "mitre",
                "category": "T1569.002",
                "avg_nll": "1.5",
                "first_token_matches": "1",
                "avg_greedy_lcp": "0.8",
                "tokens_generated": "10",
                "ref_tokens": "5",
            },
            {
                "prompt_id": "p2",
                "bucket": "metasploit",
                "category": "exploit",
                "avg_nll": "2.0",
                "first_token_matches": "0",
                "avg_greedy_lcp": "0.5",
                "tokens_generated": "8",
                "ref_tokens": "5",
            },
        ]
        candidate_rows = [
            {
                "prompt_id": "p1",
                "bucket": "mitre",
                "category": "T1569.002",
                "avg_nll": "1.2",
                "first_token_matches": "1",
                "avg_greedy_lcp": "0.9",
                "tokens_generated": "12",
                "ref_tokens": "5",
            },
            {
                "prompt_id": "p2",
                "bucket": "metasploit",
                "category": "exploit",
                "avg_nll": "2.5",
                "first_token_matches": "0",
                "avg_greedy_lcp": "0.3",
                "tokens_generated": "6",
                "ref_tokens": "5",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            baseline_path = tmpdir / "baseline.tsv"
            candidate_path = tmpdir / "candidate.tsv"
            output_path = tmpdir / "delta.tsv"

            _write_tsv(baseline_path, baseline_rows)
            _write_tsv(candidate_path, candidate_rows)

            rc = cs.main(
                [
                    "--baseline",
                    str(baseline_path),
                    "--candidate",
                    str(candidate_path),
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(rc, 0)
            self.assertTrue(output_path.exists())

            with open(output_path) as f:
                lines = f.read().strip().split("\n")
            self.assertGreaterEqual(len(lines), 2)
            header = lines[0].split("\t")
            self.assertIn("verdict", header)
            self.assertIn("delta_nll", header)
            self.assertIn("delta_lcp", header)

            # p1: delta_nll = 1.2 - 1.5 = -0.3, delta_lcp = 0.9 - 0.8 = 0.1 → improved
            # p2: delta_nll = 2.5 - 2.0 = 0.5, delta_lcp = 0.3 - 0.5 = -0.2 → regressed
            data_lines = [l for l in lines[1:] if l.strip()]
            self.assertEqual(len(data_lines), 2)

    def test_main_no_common_prompts_returns_error(self):
        """No common prompt_ids should return exit code 1."""
        baseline_rows = [
            {
                "prompt_id": "p1",
                "bucket": "mitre",
                "category": "T1569.002",
                "avg_nll": "1.5",
                "first_token_matches": "1",
                "avg_greedy_lcp": "0.8",
                "tokens_generated": "10",
                "ref_tokens": "5",
            },
        ]
        candidate_rows = [
            {
                "prompt_id": "p2",
                "bucket": "metasploit",
                "category": "exploit",
                "avg_nll": "2.0",
                "first_token_matches": "0",
                "avg_greedy_lcp": "0.5",
                "tokens_generated": "8",
                "ref_tokens": "5",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            baseline_path = tmpdir / "baseline.tsv"
            candidate_path = tmpdir / "candidate.tsv"
            output_path = tmpdir / "delta.tsv"

            _write_tsv(baseline_path, baseline_rows)
            _write_tsv(candidate_path, candidate_rows)

            rc = cs.main(
                [
                    "--baseline",
                    str(baseline_path),
                    "--candidate",
                    str(candidate_path),
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(rc, 1)

    def test_main_missing_baseline_returns_error(self):
        """Missing baseline file should return exit code 1."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            candidate_path = tmpdir / "candidate.tsv"
            output_path = tmpdir / "delta.tsv"

            _write_tsv(
                candidate_path,
                [
                    {
                        "prompt_id": "p1",
                        "bucket": "b",
                        "category": "c",
                        "avg_nll": "1.0",
                        "first_token_matches": "1",
                        "avg_greedy_lcp": "0.5",
                        "tokens_generated": "5",
                        "ref_tokens": "5",
                    },
                ],
            )

            rc = cs.main(
                [
                    "--baseline",
                    str(tmpdir / "nonexistent.tsv"),
                    "--candidate",
                    str(candidate_path),
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(rc, 1)

    def test_main_missing_candidate_returns_error(self):
        """Missing candidate file should return exit code 1."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            baseline_path = tmpdir / "baseline.tsv"
            output_path = tmpdir / "delta.tsv"

            _write_tsv(
                baseline_path,
                [
                    {
                        "prompt_id": "p1",
                        "bucket": "b",
                        "category": "c",
                        "avg_nll": "1.0",
                        "first_token_matches": "1",
                        "avg_greedy_lcp": "0.5",
                        "tokens_generated": "5",
                        "ref_tokens": "5",
                    },
                ],
            )

            rc = cs.main(
                [
                    "--baseline",
                    str(baseline_path),
                    "--candidate",
                    str(tmpdir / "nonexistent.tsv"),
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(rc, 1)

    def test_verdicts_correct_in_output(self):
        """Verify verdict assignments in output TSV."""
        baseline_rows = [
            {
                "prompt_id": "p1",
                "bucket": "mitre",
                "category": "T1569.002",
                "avg_nll": "1.5",
                "first_token_matches": "1",
                "avg_greedy_lcp": "0.8",
                "tokens_generated": "10",
                "ref_tokens": "5",
            },
        ]
        candidate_rows = [
            {
                "prompt_id": "p1",
                "bucket": "mitre",
                "category": "T1569.002",
                "avg_nll": "1.2",
                "first_token_matches": "1",
                "avg_greedy_lcp": "0.9",
                "tokens_generated": "12",
                "ref_tokens": "5",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            baseline_path = tmpdir / "baseline.tsv"
            candidate_path = tmpdir / "candidate.tsv"
            output_path = tmpdir / "delta.tsv"

            _write_tsv(baseline_path, baseline_rows)
            _write_tsv(candidate_path, candidate_rows)

            rc = cs.main(
                [
                    "--baseline",
                    str(baseline_path),
                    "--candidate",
                    str(candidate_path),
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(rc, 0)

            with open(output_path) as f:
                lines = f.read().strip().split("\n")
            data_line = [l for l in lines[1:] if l.strip()][0]
            self.assertIn("improved", data_line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
