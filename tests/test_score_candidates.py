#!/usr/bin/env python3
"""Tests for scripts/score_candidates.py — Candidate Scorer.

These tests are hermetic: they mock HuggingFace models, tokenizers, and file I/O
to avoid GPU requirements.

Run with:
    python -m pytest tests/test_score_candidates.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

# Make the scripts/ dir importable
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# Mock device_utils BEFORE importing score_candidates
_device_utils_mock = MagicMock()
_device_utils_mock.is_cuda.return_value = False
_device_utils_mock.print_hardware_banner.return_value = "cpu"
sys.modules["device_utils"] = _device_utils_mock

import score_candidates as sc  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _write_reference_json(
    path: Path,
    prompt_id: str,
    bucket: str = "mitre",
    category: str = "T1569.002",
    token_ids: list[int] | None = None,
    text: str = "ref text",
) -> None:
    data = {
        "prompt_id": prompt_id,
        "bucket": bucket,
        "category": category,
        "model": "reference-model",
        "adapter": "none",
        "timestamp": "2026-06-22T00:00:00Z",
        "generation_config": {
            "temperature": 0.0,
            "max_new_tokens": 64,
            "do_sample": False,
            "seed": 42,
        },
        "continuation": {
            "text": text,
            "token_ids": token_ids or [1, 2, 3, 4, 5],
            "tokens": ["tok1", "tok2", "tok3", "tok4", "tok5"],
            "num_tokens": 5,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _make_mock_model_and_tokenizer():
    """Create mock model and tokenizer for GPU-free testing.

    Returns real tensors so torch.cat works.
    """

    class _MockTokenizerOutput(dict):
        def to(self, device):
            return self

    class _MockTokenizer:
        pad_token_id = 0
        eos_token_id = 0
        pad_token = "<|endoftext|>"

        def apply_chat_template(self, messages, **kwargs):
            return "mock chat template output"

        def __call__(self, text, **kwargs):
            result = _MockTokenizerOutput()
            result["input_ids"] = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            result["attention_mask"] = torch.ones(1, 10)
            return result

    tokenizer = _MockTokenizer()

    model = MagicMock()
    model.device = "cpu"

    # Forward pass: return object with real .loss.item()
    class _MockModelOutput:
        class _MockLoss:
            def item(self):
                return 1.5

        loss = _MockLoss()

    def _forward(*args, **kwargs):
        return _MockModelOutput()

    model.side_effect = _forward

    # generate: return real tensor
    def _generate(**kwargs):
        input_ids = kwargs.get("input_ids", torch.tensor([[1, 2, 3]]))
        input_len = input_ids.shape[1]
        return torch.tensor([[1] * (input_len + 5)])

    model.generate = _generate

    return model, tokenizer


# ---------------------------------------------------------------------------
# Tests: CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLIParsing(unittest.TestCase):
    """Verify CLI argument parsing and defaults."""

    def test_required_args(self):
        """All required args must be present."""
        with self.assertRaises(SystemExit):
            sc.parse_args([])

    def test_minimal_args(self):
        """Minimal valid args should parse without error."""
        args = sc.parse_args(
            [
                "--base-model",
                "Qwen/Qwen2.5-7B-Instruct",
                "--reference-dir",
                "refs/",
                "--output",
                "scores.tsv",
            ]
        )
        self.assertEqual(args.base_model, "Qwen/Qwen2.5-7B-Instruct")
        self.assertIsNone(args.adapter)
        self.assertEqual(args.max_new_tokens, 512)
        self.assertEqual(args.seed, 42)

    def test_all_args(self):
        """All args should parse correctly."""
        args = sc.parse_args(
            [
                "--base-model",
                "huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated",
                "--adapter",
                "models/attacklm-single",
                "--reference-dir",
                "data/reference/continuations",
                "--output",
                "evals/candidate_scores.tsv",
                "--max-new-tokens",
                "256",
                "--seed",
                "123",
                "--compute-dtype",
                "bf16",
            ]
        )
        self.assertEqual(args.adapter, "models/attacklm-single")
        self.assertEqual(args.max_new_tokens, 256)
        self.assertEqual(args.seed, 123)
        self.assertEqual(args.compute_dtype, "bf16")


# ---------------------------------------------------------------------------
# Tests: load_reference_continuations
# ---------------------------------------------------------------------------


class TestLoadReferenceContinuations(unittest.TestCase):
    """Verify reference continuation loading."""

    def test_load_valid_references(self):
        """Valid JSON files should be loaded."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            _write_reference_json(tmpdir / "p1.json", "p1")
            _write_reference_json(tmpdir / "p2.json", "p2")

            refs = sc.load_reference_continuations(str(tmpdir))
            self.assertEqual(len(refs), 2)
            self.assertEqual(refs[0]["prompt_id"], "p1")
            self.assertEqual(refs[1]["prompt_id"], "p2")

    def test_load_empty_directory(self):
        """Empty directory should return empty list."""
        with tempfile.TemporaryDirectory() as tmp:
            refs = sc.load_reference_continuations(tmp)
            self.assertEqual(refs, [])

    def test_nonexistent_directory_raises(self):
        """Non-existent directory should raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            sc.load_reference_continuations("/nonexistent/dir")

    def test_skip_invalid_json_files(self):
        """Invalid JSON files should be skipped with warning."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            _write_reference_json(tmpdir / "p1.json", "p1")
            (tmpdir / "bad.json").write_text("not valid json")

            refs = sc.load_reference_continuations(str(tmpdir))
            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0]["prompt_id"], "p1")


# ---------------------------------------------------------------------------
# Tests: load_prompts_for_metadata
# ---------------------------------------------------------------------------


class TestLoadPromptsForMetadata(unittest.TestCase):
    """Verify prompts metadata loading."""

    def test_load_valid_prompts(self):
        """Valid prompts JSONL should return dict keyed by prompt_id."""
        records = [
            {
                "prompt_id": "p1",
                "bucket": "mitre",
                "category": "T1569.002",
                "messages": [],
            },
            {
                "prompt_id": "p2",
                "bucket": "metasploit",
                "category": "exploit",
                "messages": [],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            _write_jsonl(path, records)
            result = sc.load_prompts_for_metadata(str(path))
            self.assertEqual(len(result), 2)
            self.assertEqual(result["p1"]["bucket"], "mitre")
            self.assertEqual(result["p2"]["category"], "exploit")

    def test_nonexistent_file_returns_empty(self):
        """Non-existent file should return empty dict."""
        result = sc.load_prompts_for_metadata("/nonexistent.jsonl")
        self.assertEqual(result, {})

    def test_empty_path_returns_empty(self):
        """Empty path should return empty dict."""
        result = sc.load_prompts_for_metadata("")
        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# Tests: compute_nll
# ---------------------------------------------------------------------------


class TestComputeNLL(unittest.TestCase):
    """Verify NLL computation."""

    def test_compute_nll_returns_float(self):
        """compute_nll should return a finite float."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        nll = sc.compute_nll(model, tokenizer, "prompt text", [1, 2, 3, 4, 5])
        self.assertIsInstance(nll, float)
        self.assertEqual(nll, 1.5)

    def test_compute_nll_empty_ref_tokens(self):
        """Empty reference tokens should still work."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        nll = sc.compute_nll(model, tokenizer, "prompt text", [])
        self.assertIsInstance(nll, float)


# ---------------------------------------------------------------------------
# Tests: compute_greedy_lcp
# ---------------------------------------------------------------------------


class TestComputeGreedyLCP(unittest.TestCase):
    """Verify LCP computation."""

    def test_lcp_returns_tuple(self):
        """compute_greedy_lcp should return (lcp_ratio, first_match, tokens_gen)."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        lcp_ratio, first_match, tokens_gen = sc.compute_greedy_lcp(
            model, tokenizer, "prompt text", [1, 2, 3, 4, 5], max_new_tokens=64
        )
        self.assertIsInstance(lcp_ratio, float)
        self.assertIsInstance(first_match, bool)
        self.assertIsInstance(tokens_gen, int)

    def test_lcp_empty_ref_tokens(self):
        """Empty reference tokens should return 0.0 LCP."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        lcp_ratio, first_match, tokens_gen = sc.compute_greedy_lcp(
            model, tokenizer, "prompt text", [], max_new_tokens=64
        )
        self.assertEqual(lcp_ratio, 0.0)
        self.assertFalse(first_match)


# ---------------------------------------------------------------------------
# Tests: Integration (mocked model, real files)
# ---------------------------------------------------------------------------


class TestIntegration(unittest.TestCase):
    """End-to-end test with mocked model and real temporary files."""

    @patch("score_candidates.load_model_and_tokenizer")
    def test_main_creates_tsv(self, mock_load):
        """main() should create a valid TSV file."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        mock_load.return_value = (model, tokenizer)

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)

            # Create reference files
            ref_dir = tmpdir / "references"
            ref_dir.mkdir()
            _write_reference_json(ref_dir / "p1.json", "p1", token_ids=[1, 2, 3])
            _write_reference_json(ref_dir / "p2.json", "p2", token_ids=[4, 5, 6])

            output_path = tmpdir / "scores.tsv"

            rc = sc.main(
                [
                    "--base-model",
                    "Qwen/Qwen2.5-7B-Instruct",
                    "--reference-dir",
                    str(ref_dir),
                    "--output",
                    str(output_path),
                    "--max-new-tokens",
                    "32",
                ]
            )

            self.assertEqual(rc, 0)
            self.assertTrue(output_path.exists())

            with open(output_path) as f:
                lines = f.read().strip().split("\n")
            self.assertGreaterEqual(len(lines), 2)  # header + at least 1 row
            header = lines[0].split("\t")
            self.assertIn("prompt_id", header)
            self.assertIn("avg_nll", header)
            self.assertIn("avg_greedy_lcp", header)

    @patch("score_candidates.load_model_and_tokenizer")
    def test_main_empty_references_returns_error(self, mock_load):
        """Empty reference directory should return exit code 1."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        mock_load.return_value = (model, tokenizer)

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            ref_dir = tmpdir / "empty_refs"
            ref_dir.mkdir()
            output_path = tmpdir / "scores.tsv"

            rc = sc.main(
                [
                    "--base-model",
                    "Qwen/Qwen2.5-7B-Instruct",
                    "--reference-dir",
                    str(ref_dir),
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(rc, 1)

    @patch("score_candidates.load_model_and_tokenizer")
    def test_main_nonexistent_reference_dir(self, mock_load):
        """Non-existent reference dir should return exit code 1."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        mock_load.return_value = (model, tokenizer)

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "scores.tsv"

            rc = sc.main(
                [
                    "--base-model",
                    "Qwen/Qwen2.5-7B-Instruct",
                    "--reference-dir",
                    "/nonexistent/refs",
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(rc, 1)

    @patch("score_candidates.load_model_and_tokenizer")
    def test_main_model_load_failure(self, mock_load):
        """Model load failure should return exit code 1."""
        mock_load.side_effect = FileNotFoundError("Model not found")

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            ref_dir = tmpdir / "refs"
            ref_dir.mkdir()
            _write_reference_json(ref_dir / "p1.json", "p1")
            output_path = tmpdir / "scores.tsv"

            rc = sc.main(
                [
                    "--base-model",
                    "nonexistent/model",
                    "--reference-dir",
                    str(ref_dir),
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
