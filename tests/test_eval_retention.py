#!/usr/bin/env python3
"""Tests for scripts/eval_retention.py — Retention Evaluation Suite.

These tests are hermetic: they use temporary JSONL files and mock the
HuggingFace model/tokenizer to avoid GPU requirements.

Run with:
    python -m pytest tests/test_eval_retention.py -v

Or directly:
    python tests/test_eval_retention.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

# ---------------------------------------------------------------------------
# Inject a mock `datasets` module BEFORE importing eval_retention.
# The real `datasets` package is not installed in this environment, but
# eval_retention.py does `from datasets import load_dataset` inside its
# function bodies.  We pre-seed sys.modules so that import succeeds and
# returns our mock.
# ---------------------------------------------------------------------------

_mock_datasets = MagicMock()
_mock_load_dataset = MagicMock()
_mock_datasets.load_dataset = _mock_load_dataset
sys.modules["datasets"] = _mock_datasets

# Make the scripts/ dir importable
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import eval_retention as er  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write a list of dicts as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _make_mock_model_and_tokenizer():
    """Create a mock model and tokenizer for GPU-free testing.

    The mock model returns a fixed loss of 2.0 for perplexity and
    generates a fixed answer for QA.
    """
    tokenizer = MagicMock()
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 0

    # __call__: return a dict with input_ids and attention_mask
    def _tokenize(texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        batch_size = len(texts)
        length = 8
        input_ids = MagicMock()
        input_ids.shape = (batch_size, length)
        input_ids.__getitem__ = lambda s, i: MagicMock()
        attn_mask = MagicMock()
        attn_mask.shape = (batch_size, length)
        attn_mask.sum.return_value = batch_size * length
        return {"input_ids": input_ids, "attention_mask": attn_mask}

    tokenizer.__call__ = _tokenize

    # decode: return a fixed answer
    def _decode(token_ids, **kwargs):
        return "mock generated answer"

    tokenizer.decode = _decode

    # Model
    model = MagicMock()
    model.device = "cpu"

    # Forward pass: return a mock with .loss
    def _forward(**kwargs):
        mock_output = MagicMock()
        mock_output.loss = MagicMock()
        mock_output.loss.item.return_value = 2.0  # fixed loss
        return mock_output

    model.__call__ = _forward

    # generate: return fixed output
    def _generate(**kwargs):
        input_ids = kwargs.get("input_ids", MagicMock())
        input_len = input_ids.shape[1] if hasattr(input_ids, "shape") else 8
        mock_ids = MagicMock()
        mock_ids.shape = (1, input_len + 5)
        return mock_ids

    model.generate = _generate

    return model, tokenizer


def _make_mock_dataset(records: list[dict]):
    """Create a mock datasets.Dataset from a list of dicts."""
    mock_ds = MagicMock()
    mock_ds.__len__.return_value = len(records)
    mock_ds.__getitem__ = lambda s, i: records[i] if i < len(records) else None
    return mock_ds


# ---------------------------------------------------------------------------
# Tests: CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLIParsing(unittest.TestCase):
    """Verify CLI argument parsing and defaults."""

    def test_required_args(self):
        """All required args must be present."""
        with self.assertRaises(SystemExit):
            er.parse_args([])

    def test_minimal_args(self):
        """Minimal valid args should parse without error."""
        args = er.parse_args(
            [
                "--base-model",
                "Qwen/Qwen2.5-7B-Instruct",
                "--pretraining-corpus",
                "pretrain.jsonl",
                "--target-corpus",
                "target.jsonl",
                "--downstream-qa",
                "qa.jsonl",
                "--output",
                "report.json",
            ]
        )
        self.assertEqual(args.base_model, "Qwen/Qwen2.5-7B-Instruct")
        self.assertIsNone(args.adapter)
        self.assertEqual(args.max_samples, 500)
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(args.max_length, 2048)
        self.assertIsNone(args.compute_dtype)

    def test_all_args(self):
        """All args should parse correctly."""
        args = er.parse_args(
            [
                "--base-model",
                "huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated",
                "--adapter",
                "models/attacklm-single_2026-06-22_12-00",
                "--pretraining-corpus",
                "data/pretraining_sample.jsonl",
                "--target-corpus",
                "data/target.jsonl",
                "--downstream-qa",
                "data/qa.jsonl",
                "--output",
                "evals/report.json",
                "--max-samples",
                "100",
                "--batch-size",
                "4",
                "--max-length",
                "1024",
                "--compute-dtype",
                "bf16",
            ]
        )
        self.assertEqual(
            args.base_model, "huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated"
        )
        self.assertEqual(args.adapter, "models/attacklm-single_2026-06-22_12-00")
        self.assertEqual(args.max_samples, 100)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.max_length, 1024)
        self.assertEqual(args.compute_dtype, "bf16")

    def test_max_samples_zero(self):
        """max_samples=0 means evaluate all samples."""
        args = er.parse_args(
            [
                "--base-model",
                "Qwen/Qwen2.5-7B-Instruct",
                "--pretraining-corpus",
                "p.jsonl",
                "--target-corpus",
                "t.jsonl",
                "--downstream-qa",
                "q.jsonl",
                "--output",
                "r.json",
                "--max-samples",
                "0",
            ]
        )
        self.assertEqual(args.max_samples, 0)

    def test_adapter_optional(self):
        """--adapter is optional (baseline mode)."""
        args = er.parse_args(
            [
                "--base-model",
                "Qwen/Qwen2.5-7B-Instruct",
                "--pretraining-corpus",
                "p.jsonl",
                "--target-corpus",
                "t.jsonl",
                "--downstream-qa",
                "q.jsonl",
                "--output",
                "r.json",
            ]
        )
        self.assertIsNone(args.adapter)


# ---------------------------------------------------------------------------
# Tests: Perplexity computation
# ---------------------------------------------------------------------------


class TestPerplexity(unittest.TestCase):
    """Perplexity computation with mocked model."""

    def test_compute_perplexity_returns_float(self):
        """compute_perplexity should return a finite float."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        _mock_load_dataset.return_value = _make_mock_dataset(
            [
                {"text": "The quick brown fox jumps over the lazy dog."},
                {"text": "Hello world this is a test."},
            ]
        )

        ppl = er.compute_perplexity(
            model,
            tokenizer,
            "dummy.jsonl",
            max_samples=10,
            batch_size=2,
            max_length=2048,
            device="cpu",
        )

        self.assertIsInstance(ppl, float)
        self.assertTrue(ppl > 0)

    def test_compute_perplexity_respects_max_samples(self):
        """max_samples should cap the number of samples evaluated."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        _mock_load_dataset.return_value = _make_mock_dataset(
            [{"text": f"Sample {i}"} for i in range(20)]
        )

        ppl = er.compute_perplexity(
            model,
            tokenizer,
            "dummy.jsonl",
            max_samples=5,
            batch_size=2,
            max_length=2048,
            device="cpu",
        )

        self.assertIsInstance(ppl, float)
        self.assertTrue(ppl > 0)

    def test_compute_perplexity_empty_corpus(self):
        """Empty corpus should return NaN."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        _mock_load_dataset.return_value = _make_mock_dataset([])

        ppl = er.compute_perplexity(
            model,
            tokenizer,
            "dummy.jsonl",
            max_samples=10,
            batch_size=2,
            max_length=2048,
            device="cpu",
        )

        self.assertTrue(ppl != ppl)  # NaN check


# ---------------------------------------------------------------------------
# Tests: Downstream QA
# ---------------------------------------------------------------------------


class TestDownstreamQA(unittest.TestCase):
    """Downstream QA accuracy evaluation."""

    def test_qa_returns_dict(self):
        """evaluate_downstream_qa should return a dict with expected keys."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        _mock_load_dataset.return_value = _make_mock_dataset(
            [
                {
                    "question": "What is SQL injection?",
                    "answer": "mock generated answer",
                },
            ]
        )

        result = er.evaluate_downstream_qa(
            model,
            tokenizer,
            "dummy.jsonl",
            max_samples=10,
            max_length=2048,
            device="cpu",
        )

        self.assertIn("accuracy", result)
        self.assertIn("n_total", result)
        self.assertIn("n_correct", result)
        self.assertEqual(result["n_total"], 1)

    def test_qa_empty_dataset(self):
        """Empty QA dataset should return zero accuracy."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        _mock_load_dataset.return_value = _make_mock_dataset([])

        result = er.evaluate_downstream_qa(
            model,
            tokenizer,
            "dummy.jsonl",
            max_samples=10,
            max_length=2048,
            device="cpu",
        )

        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["n_total"], 0)


# ---------------------------------------------------------------------------
# Tests: Text normalization and matching
# ---------------------------------------------------------------------------


class TestTextMatching(unittest.TestCase):
    """Verify _normalize_text and _is_correct."""

    def test_normalize_lowercase(self):
        self.assertEqual(er._normalize_text("Hello World"), "hello world")

    def test_normalize_strip_punctuation(self):
        self.assertEqual(er._normalize_text("Hello, World!"), "hello world")

    def test_normalize_whitespace(self):
        self.assertEqual(er._normalize_text("  Hello   World  "), "hello world")

    def test_exact_match(self):
        self.assertTrue(er._is_correct("SQL injection", "SQL injection"))

    def test_case_insensitive_match(self):
        self.assertTrue(er._is_correct("SQL INJECTION", "sql injection"))

    def test_substring_match(self):
        self.assertTrue(
            er._is_correct(
                "SQL injection is a code injection technique",
                "SQL injection",
            )
        )

    def test_no_match(self):
        self.assertFalse(er._is_correct("Buffer overflow", "SQL injection"))

    def test_punctuation_insensitive_match(self):
        self.assertTrue(er._is_correct("SQL injection!", "SQL injection"))


# ---------------------------------------------------------------------------
# Tests: Report schema
# ---------------------------------------------------------------------------


class TestReportSchema(unittest.TestCase):
    """Verify the JSON report has the correct schema."""

    def test_build_report_has_all_keys(self):
        report = er.build_report(
            ppl_pretraining=12.34,
            ppl_target=8.56,
            qa_results={"accuracy": 0.85, "n_total": 100, "n_correct": 85},
            base_model="Qwen/Qwen2.5-7B-Instruct",
            adapter_path=None,
            compute_dtype="bf16",
        )

        # Top-level keys
        self.assertIn("perplexity", report)
        self.assertIn("downstream_qa", report)
        self.assertIn("metadata", report)

        # Perplexity sub-keys
        ppl = report["perplexity"]
        self.assertIn("pretraining", ppl)
        self.assertIn("target", ppl)
        self.assertIn("delta", ppl)
        self.assertEqual(ppl["pretraining"], 12.34)
        self.assertEqual(ppl["target"], 8.56)
        self.assertAlmostEqual(ppl["delta"], 3.78)

        # QA sub-keys
        qa = report["downstream_qa"]
        self.assertIn("accuracy", qa)
        self.assertIn("n_total", qa)
        self.assertIn("n_correct", qa)
        self.assertEqual(qa["accuracy"], 0.85)
        self.assertEqual(qa["n_total"], 100)
        self.assertEqual(qa["n_correct"], 85)

        # Metadata
        meta = report["metadata"]
        self.assertIn("base_model", meta)
        self.assertIn("adapter_path", meta)
        self.assertIn("compute_dtype", meta)
        self.assertIn("timestamp", meta)
        self.assertEqual(meta["base_model"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertIsNone(meta["adapter_path"])
        self.assertEqual(meta["compute_dtype"], "bf16")

    def test_report_serializes_to_json(self):
        """The report should be JSON-serializable."""
        report = er.build_report(
            ppl_pretraining=12.34,
            ppl_target=8.56,
            qa_results={"accuracy": 0.85, "n_total": 100, "n_correct": 85},
            base_model="Qwen/Qwen2.5-7B-Instruct",
            adapter_path="models/adapter",
            compute_dtype="fp16",
        )
        json_str = json.dumps(report)
        self.assertIsInstance(json_str, str)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["metadata"]["adapter_path"], "models/adapter")


# ---------------------------------------------------------------------------
# Tests: Adapter path resolution
# ---------------------------------------------------------------------------


class TestAdapterPathResolution(unittest.TestCase):
    """Verify adapter path resolution logic."""

    def test_adapter_absolute_path(self):
        """Absolute adapter path should be resolved correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            adapter_dir = Path(tmp) / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text("{}")

            resolved = er.resolve_model_path(str(adapter_dir))
            self.assertEqual(resolved, str(adapter_dir.resolve()))

    def test_adapter_nonexistent_raises(self):
        """Non-existent adapter path should raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            er.resolve_model_path("/nonexistent/path/to/adapter")

    def test_adapter_relative_path(self):
        """Relative adapter path should be resolved to absolute."""
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                adapter_dir = Path(tmp) / "my_adapter"
                adapter_dir.mkdir()
                (adapter_dir / "adapter_config.json").write_text("{}")

                resolved = er.resolve_model_path("my_adapter")
                self.assertTrue(Path(resolved).is_absolute())
                self.assertTrue(Path(resolved).exists())
            finally:
                os.chdir(cwd)


# ---------------------------------------------------------------------------
# Tests: Compute dtype detection
# ---------------------------------------------------------------------------


class TestComputeDtype(unittest.TestCase):
    """Verify compute dtype auto-detection."""

    def test_user_specified_bf16(self):
        dtype = er.detect_compute_dtype("bf16")
        self.assertEqual(dtype, torch.bfloat16)

    def test_user_specified_fp16(self):
        dtype = er.detect_compute_dtype("fp16")
        self.assertEqual(dtype, torch.float16)

    def test_user_specified_fp32(self):
        dtype = er.detect_compute_dtype("fp32")
        self.assertEqual(dtype, torch.float32)

    def test_user_specified_unknown_falls_back(self):
        """Unknown dtype string should fall back to auto-detect."""
        dtype = er.detect_compute_dtype("unknown")
        self.assertIn(dtype, (torch.bfloat16, torch.float32))


# ---------------------------------------------------------------------------
# Tests: Integration (mocked model, real files)
# ---------------------------------------------------------------------------


class TestIntegration(unittest.TestCase):
    """End-to-end test with mocked model and real temporary files."""

    @patch("eval_retention.load_model_and_tokenizer")
    def test_main_creates_report(self, mock_load):
        """main() should create a valid JSON report file."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        mock_load.return_value = (model, tokenizer)
        _mock_load_dataset.return_value = _make_mock_dataset(
            [
                {"text": "General text."},
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            output = tmpdir / "report.json"

            rc = er.main(
                [
                    "--base-model",
                    "Qwen/Qwen2.5-7B-Instruct",
                    "--pretraining-corpus",
                    str(tmpdir / "pretrain.jsonl"),
                    "--target-corpus",
                    str(tmpdir / "target.jsonl"),
                    "--downstream-qa",
                    str(tmpdir / "qa.jsonl"),
                    "--output",
                    str(output),
                    "--max-samples",
                    "5",
                ]
            )

            self.assertEqual(rc, 0)
            self.assertTrue(output.exists())

            with open(output) as f:
                report = json.load(f)

            self.assertIn("perplexity", report)
            self.assertIn("downstream_qa", report)
            self.assertIn("metadata", report)
            self.assertEqual(
                report["metadata"]["base_model"], "Qwen/Qwen2.5-7B-Instruct"
            )
            self.assertIsNone(report["metadata"]["adapter_path"])

    @patch("eval_retention.load_model_and_tokenizer")
    def test_main_with_adapter(self, mock_load):
        """main() should work with an adapter path."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        mock_load.return_value = (model, tokenizer)
        _mock_load_dataset.return_value = _make_mock_dataset(
            [
                {"text": "General text."},
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            output = tmpdir / "report.json"

            rc = er.main(
                [
                    "--base-model",
                    "Qwen/Qwen2.5-7B-Instruct",
                    "--adapter",
                    "models/some-adapter",
                    "--pretraining-corpus",
                    str(tmpdir / "pretrain.jsonl"),
                    "--target-corpus",
                    str(tmpdir / "target.jsonl"),
                    "--downstream-qa",
                    str(tmpdir / "qa.jsonl"),
                    "--output",
                    str(output),
                    "--max-samples",
                    "5",
                ]
            )

            self.assertEqual(rc, 0)
            with open(output) as f:
                report = json.load(f)
            self.assertEqual(report["metadata"]["adapter_path"], "models/some-adapter")


# ---------------------------------------------------------------------------
# Tests: Max-samples truncation
# ---------------------------------------------------------------------------


class TestMaxSamples(unittest.TestCase):
    """Verify max-samples truncation behavior."""

    @patch("eval_retention.load_model_and_tokenizer")
    def test_max_samples_caps_evaluation(self, mock_load):
        """With max_samples=1, only 1 sample should be evaluated."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        mock_load.return_value = (model, tokenizer)
        _mock_load_dataset.return_value = _make_mock_dataset(
            [{"text": f"Sample {i}"} for i in range(10)]
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            output = tmpdir / "report.json"

            rc = er.main(
                [
                    "--base-model",
                    "Qwen/Qwen2.5-7B-Instruct",
                    "--pretraining-corpus",
                    str(tmpdir / "pretrain.jsonl"),
                    "--target-corpus",
                    str(tmpdir / "target.jsonl"),
                    "--downstream-qa",
                    str(tmpdir / "qa.jsonl"),
                    "--output",
                    str(output),
                    "--max-samples",
                    "1",
                ]
            )

            self.assertEqual(rc, 0)
            with open(output) as f:
                report = json.load(f)

            # QA should have n_total=1 (capped by max_samples)
            self.assertEqual(report["downstream_qa"]["n_total"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
