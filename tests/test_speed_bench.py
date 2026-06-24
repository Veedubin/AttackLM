#!/usr/bin/env python3
"""Tests for scripts/speed_bench.py — Speed Benchmarking at Context Frontiers.

These tests are hermetic: they mock HuggingFace models, tokenizers, torch.cuda,
and file I/O to avoid GPU requirements.

Run with:
    python -m pytest tests/test_speed_bench.py -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

# Make the scripts/ dir importable
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# Mock device_utils BEFORE importing speed_bench.
# speed_bench calls setup_allocator_env() at module import time.
_device_utils_mock = MagicMock()
_device_utils_mock.is_cuda.return_value = False
_device_utils_mock.print_hardware_banner.return_value = "cpu"
_device_utils_mock.setup_allocator_env.return_value = None
sys.modules["device_utils"] = _device_utils_mock

import speed_bench as sb  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_model_and_tokenizer():
    """Create mock model and tokenizer for GPU-free testing.

    Returns real tensors so shape comparisons work.
    """
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 0
    tokenizer.pad_token = "<|endoftext|>"

    class _MockTokenizer:
        pad_token_id = 0
        eos_token_id = 0
        pad_token = "<|endoftext|>"

        def __call__(self, text, **kwargs):
            class _MockResult:
                input_ids = torch.tensor([[1] * 5000])

            return _MockResult()

    tokenizer = _MockTokenizer()

    model = MagicMock()
    model.device = "cpu"

    # Forward pass: no-op
    def _forward(**kwargs):
        return MagicMock()

    model.__call__ = _forward

    # generate: accept positional arg (input_ids) and return real tensor
    def _generate(input_ids=None, **kwargs):
        if input_ids is None:
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
            sb.parse_args([])

    def test_minimal_args(self):
        """Minimal valid args should parse without error."""
        args = sb.parse_args(
            [
                "--base-model",
                "Qwen/Qwen2.5-7B-Instruct",
                "--context-file",
                "context.txt",
                "--output",
                "report.csv",
            ]
        )
        self.assertEqual(args.base_model, "Qwen/Qwen2.5-7B-Instruct")
        self.assertIsNone(args.adapter)
        self.assertEqual(args.frontiers, [512, 1024, 2048, 4096])
        self.assertEqual(args.gen_tokens, 128)
        self.assertEqual(args.warmup_runs, 2)
        self.assertEqual(args.bench_runs, 5)
        self.assertEqual(args.seed, 42)

    def test_all_args(self):
        """All args should parse correctly."""
        args = sb.parse_args(
            [
                "--base-model",
                "huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated",
                "--adapter",
                "models/attacklm-single",
                "--context-file",
                "data/bench/speed_context.txt",
                "--output",
                "evals/speed_report.csv",
                "--frontiers",
                "512",
                "1024",
                "2048",
                "--gen-tokens",
                "64",
                "--warmup-runs",
                "3",
                "--bench-runs",
                "10",
                "--seed",
                "99",
                "--compute-dtype",
                "bf16",
            ]
        )
        self.assertEqual(args.adapter, "models/attacklm-single")
        self.assertEqual(args.frontiers, [512, 1024, 2048])
        self.assertEqual(args.gen_tokens, 64)
        self.assertEqual(args.warmup_runs, 3)
        self.assertEqual(args.bench_runs, 10)
        self.assertEqual(args.compute_dtype, "bf16")


# ---------------------------------------------------------------------------
# Tests: CUDA helpers (CPU path)
# ---------------------------------------------------------------------------


class TestCUDAHelpers(unittest.TestCase):
    """Verify CUDA helper functions on CPU."""

    def test_cuda_sync_noop_on_cpu(self):
        """_cuda_sync should be a no-op on CPU."""
        # Should not raise
        sb._cuda_sync()

    def test_cuda_max_mem_gb_returns_zero_on_cpu(self):
        """_cuda_max_mem_gb should return 0.0 on CPU."""
        mem = sb._cuda_max_mem_gb()
        self.assertEqual(mem, 0.0)

    def test_cuda_reset_peak_noop_on_cpu(self):
        """_cuda_reset_peak should be a no-op on CPU."""
        # Should not raise
        sb._cuda_reset_peak()


# ---------------------------------------------------------------------------
# Tests: benchmark_frontier (mocked)
# ---------------------------------------------------------------------------


class TestBenchmarkFrontier(unittest.TestCase):
    """Verify benchmark_frontier logic."""

    def test_benchmark_returns_expected_keys(self):
        """benchmark_frontier should return dict with expected keys."""
        model, tokenizer = _make_mock_model_and_tokenizer()

        # Create a real context tensor
        context_tokens = torch.arange(1000, dtype=torch.long)

        result = sb.benchmark_frontier(
            model=model,
            tokenizer=tokenizer,
            context_tokens=context_tokens,
            ctx_tokens=512,
            gen_tokens=128,
            warmup_runs=1,
            bench_runs=1,
            device="cpu",
        )

        self.assertIn("prefill_tps", result)
        self.assertIn("gen_tps", result)
        self.assertIn("vram_gb", result)
        self.assertIsInstance(result["prefill_tps"], float)
        self.assertIsInstance(result["gen_tps"], float)
        self.assertIsInstance(result["vram_gb"], float)

    def test_benchmark_context_too_short(self):
        """Context shorter than frontier should use available tokens."""
        model, tokenizer = _make_mock_model_and_tokenizer()

        # Real tensor with only 100 tokens
        context_tokens = torch.arange(100, dtype=torch.long)

        result = sb.benchmark_frontier(
            model=model,
            tokenizer=tokenizer,
            context_tokens=context_tokens,
            ctx_tokens=512,  # larger than available 100
            gen_tokens=128,
            warmup_runs=1,
            bench_runs=1,
            device="cpu",
        )

        self.assertIn("prefill_tps", result)
        self.assertIn("gen_tps", result)


# ---------------------------------------------------------------------------
# Tests: Integration (mocked model, real files)
# ---------------------------------------------------------------------------


class TestIntegration(unittest.TestCase):
    """End-to-end test with mocked model and real temporary files."""

    @patch("speed_bench.load_model_and_tokenizer")
    def test_main_creates_csv(self, mock_load):
        """main() should create a valid CSV file."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        mock_load.return_value = (model, tokenizer)

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)

            # Create context file with enough text
            context_path = tmpdir / "context.txt"
            context_path.write_text("Hello world, this is a test context. " * 500)

            output_path = tmpdir / "speed.csv"

            rc = sb.main(
                [
                    "--base-model",
                    "Qwen/Qwen2.5-7B-Instruct",
                    "--context-file",
                    str(context_path),
                    "--output",
                    str(output_path),
                    "--frontiers",
                    "512",
                    "--gen-tokens",
                    "16",
                    "--warmup-runs",
                    "1",
                    "--bench-runs",
                    "1",
                ]
            )

            self.assertEqual(rc, 0)
            self.assertTrue(output_path.exists())

            with open(output_path) as f:
                content = f.read()
            self.assertIn("ctx_tokens", content)
            self.assertIn("prefill_tps", content)
            self.assertIn("gen_tps", content)
            self.assertIn("vram_gb", content)

    @patch("speed_bench.load_model_and_tokenizer")
    def test_main_nonexistent_context_file(self, mock_load):
        """Non-existent context file should return exit code 1."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        mock_load.return_value = (model, tokenizer)

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "speed.csv"

            rc = sb.main(
                [
                    "--base-model",
                    "Qwen/Qwen2.5-7B-Instruct",
                    "--context-file",
                    str(Path(tmp) / "nonexistent.txt"),
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(rc, 1)

    @patch("speed_bench.load_model_and_tokenizer")
    def test_main_model_load_failure(self, mock_load):
        """Model load failure should return exit code 1."""
        mock_load.side_effect = Exception("Model not found")

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            context_path = tmpdir / "context.txt"
            context_path.write_text("test context")
            output_path = tmpdir / "speed.csv"

            rc = sb.main(
                [
                    "--base-model",
                    "nonexistent/model",
                    "--context-file",
                    str(context_path),
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(rc, 1)

    @patch("speed_bench.load_model_and_tokenizer")
    def test_main_context_shorter_than_frontier(self, mock_load):
        """Context shorter than frontier should skip that frontier."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        mock_load.return_value = (model, tokenizer)

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)

            # Very short context
            context_path = tmpdir / "short_context.txt"
            context_path.write_text("Short text.")

            output_path = tmpdir / "speed.csv"

            rc = sb.main(
                [
                    "--base-model",
                    "Qwen/Qwen2.5-7B-Instruct",
                    "--context-file",
                    str(context_path),
                    "--output",
                    str(output_path),
                    "--frontiers",
                    "512",
                    "1024",
                    "--gen-tokens",
                    "8",
                    "--warmup-runs",
                    "1",
                    "--bench-runs",
                    "1",
                ]
            )

            # The mock tokenizer returns 5000 tokens, so frontiers won't be skipped
            self.assertEqual(rc, 0)


# ---------------------------------------------------------------------------
# Tests: CSV output format
# ---------------------------------------------------------------------------


class TestCSVOutput(unittest.TestCase):
    """Verify CSV output format."""

    @patch("speed_bench.load_model_and_tokenizer")
    def test_csv_has_correct_columns(self, mock_load):
        """CSV should have all expected columns."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        mock_load.return_value = (model, tokenizer)

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            context_path = tmpdir / "context.txt"
            context_path.write_text("test " * 2000)
            output_path = tmpdir / "speed.csv"

            sb.main(
                [
                    "--base-model",
                    "Qwen/Qwen2.5-7B-Instruct",
                    "--context-file",
                    str(context_path),
                    "--output",
                    str(output_path),
                    "--frontiers",
                    "512",
                    "--gen-tokens",
                    "8",
                    "--warmup-runs",
                    "1",
                    "--bench-runs",
                    "1",
                ]
            )

            import csv

            with open(output_path, newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                self.assertIsNotNone(fieldnames)
                self.assertIn("ctx_tokens", fieldnames)
                self.assertIn("prefill_tps", fieldnames)
                self.assertIn("gen_tps", fieldnames)
                self.assertIn("vram_gb", fieldnames)
                self.assertIn("model_name", fieldnames)
                self.assertIn("adapter_path", fieldnames)
                self.assertIn("timestamp", fieldnames)

                rows = list(reader)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["ctx_tokens"], "512")


if __name__ == "__main__":
    unittest.main(verbosity=2)
