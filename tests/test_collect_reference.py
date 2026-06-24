#!/usr/bin/env python3
"""Tests for scripts/collect_reference.py — Reference Continuation Collector.

These tests are hermetic: they mock HuggingFace models, tokenizers, and file I/O
to avoid GPU requirements.

Run with:
    python -m pytest tests/test_collect_reference.py -v
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

# Mock device_utils BEFORE importing collect_reference
_device_utils_mock = MagicMock()
_device_utils_mock.is_cuda.return_value = False
_device_utils_mock.print_hardware_banner.return_value = "cpu"
sys.modules["device_utils"] = _device_utils_mock

import collect_reference as cr  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _make_mock_model_and_tokenizer():
    """Create mock model and tokenizer for GPU-free testing.

    Returns real tensors and lists so JSON serialization works.
    """
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 0
    tokenizer.pad_token = "<|endoftext|>"

    def _apply_chat_template(messages, **kwargs):
        return "mock chat template output"

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
            return result

        def convert_ids_to_tokens(self, ids):
            return [f"token_{i}" for i in ids]

        def decode(self, ids, **kwargs):
            return "mock generated continuation text"

    tokenizer = _MockTokenizer()

    model = MagicMock()
    model.device = "cpu"

    def _generate(**kwargs):
        input_ids = kwargs.get("input_ids", torch.tensor([[1, 2, 3]]))
        input_len = input_ids.shape[1]
        # Return real tensor with generated tokens appended
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
            cr.parse_args([])

    def test_minimal_args(self):
        """Minimal valid args should parse without error."""
        args = cr.parse_args(
            [
                "--base-model",
                "Qwen/Qwen2.5-7B-Instruct",
                "--prompts",
                "prompts.jsonl",
                "--output-dir",
                "output/",
            ]
        )
        self.assertEqual(args.base_model, "Qwen/Qwen2.5-7B-Instruct")
        self.assertIsNone(args.adapter)
        self.assertEqual(args.max_new_tokens, 512)
        self.assertEqual(args.seed, 42)
        self.assertIsNone(args.compute_dtype)

    def test_all_args(self):
        """All args should parse correctly."""
        args = cr.parse_args(
            [
                "--base-model",
                "huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated",
                "--adapter",
                "models/attacklm-single",
                "--prompts",
                "data/reference/prompts.jsonl",
                "--output-dir",
                "data/reference/continuations",
                "--max-new-tokens",
                "256",
                "--seed",
                "123",
                "--compute-dtype",
                "bf16",
            ]
        )
        self.assertEqual(
            args.base_model, "huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated"
        )
        self.assertEqual(args.adapter, "models/attacklm-single")
        self.assertEqual(args.max_new_tokens, 256)
        self.assertEqual(args.seed, 123)
        self.assertEqual(args.compute_dtype, "bf16")


# ---------------------------------------------------------------------------
# Tests: load_prompts
# ---------------------------------------------------------------------------


class TestLoadPrompts(unittest.TestCase):
    """Verify prompt loading from JSONL."""

    def test_load_valid_prompts(self):
        """Valid JSONL should return list of dicts."""
        records = [
            {
                "prompt_id": "p1",
                "bucket": "mitre",
                "category": "T1569.002",
                "messages": [{"role": "user", "content": "hello"}],
            },
            {
                "prompt_id": "p2",
                "bucket": "metasploit",
                "category": "exploit",
                "messages": [{"role": "user", "content": "world"}],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            _write_jsonl(path, records)
            prompts = cr.load_prompts(str(path))
            self.assertEqual(len(prompts), 2)
            self.assertEqual(prompts[0]["prompt_id"], "p1")
            self.assertEqual(prompts[1]["prompt_id"], "p2")

    def test_load_empty_file(self):
        """Empty file should return empty list."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.jsonl"
            path.write_text("")
            prompts = cr.load_prompts(str(path))
            self.assertEqual(prompts, [])

    def test_skip_invalid_json(self):
        """Lines with invalid JSON should be skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            with open(path, "w") as f:
                f.write(
                    '{"prompt_id": "p1", "bucket": "b", "category": "c", "messages": []}\n'
                )
                f.write("not valid json\n")
            prompts = cr.load_prompts(str(path))
            self.assertEqual(len(prompts), 1)
            self.assertEqual(prompts[0]["prompt_id"], "p1")

    def test_skip_missing_fields(self):
        """Records with missing required fields should be skipped."""
        records = [
            {
                "prompt_id": "p1",
                "bucket": "mitre",
                "category": "T1569.002",
                "messages": [],
            },
            {"prompt_id": "p2"},  # missing bucket, category, messages
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            _write_jsonl(path, records)
            prompts = cr.load_prompts(str(path))
            self.assertEqual(len(prompts), 1)
            self.assertEqual(prompts[0]["prompt_id"], "p1")

    def test_skip_blank_lines(self):
        """Blank lines should be skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            with open(path, "w") as f:
                f.write(
                    '{"prompt_id": "p1", "bucket": "b", "category": "c", "messages": []}\n'
                )
                f.write("\n")
                f.write(
                    '{"prompt_id": "p2", "bucket": "b", "category": "c", "messages": []}\n'
                )
            prompts = cr.load_prompts(str(path))
            self.assertEqual(len(prompts), 2)


# ---------------------------------------------------------------------------
# Tests: generate_continuation
# ---------------------------------------------------------------------------


class TestGenerateContinuation(unittest.TestCase):
    """Verify continuation generation logic."""

    def test_generate_returns_expected_schema(self):
        """generate_continuation should return dict with expected keys."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        prompt = {
            "prompt_id": "test_prompt_001",
            "bucket": "mitre",
            "category": "T1569.002",
            "messages": [{"role": "user", "content": "Test question?"}],
        }

        result = cr.generate_continuation(
            model, tokenizer, prompt, max_new_tokens=64, seed=42
        )

        self.assertIn("prompt_id", result)
        self.assertIn("model", result)
        self.assertIn("timestamp", result)
        self.assertIn("generation_config", result)
        self.assertIn("continuation", result)
        self.assertEqual(result["prompt_id"], "test_prompt_001")
        self.assertEqual(result["generation_config"]["temperature"], 0.0)
        self.assertEqual(result["generation_config"]["max_new_tokens"], 64)
        self.assertEqual(result["generation_config"]["seed"], 42)
        self.assertIn("text", result["continuation"])
        self.assertIn("token_ids", result["continuation"])
        self.assertIn("tokens", result["continuation"])
        self.assertIn("num_tokens", result["continuation"])


# ---------------------------------------------------------------------------
# Tests: Integration (mocked model, real files)
# ---------------------------------------------------------------------------


class TestIntegration(unittest.TestCase):
    """End-to-end test with mocked model and real temporary files."""

    @patch("collect_reference.load_model_and_tokenizer")
    def test_main_creates_continuation_files(self, mock_load):
        """main() should create continuation JSON files."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        mock_load.return_value = (model, tokenizer)

        prompts = [
            {
                "prompt_id": "p1",
                "bucket": "mitre",
                "category": "T1569.002",
                "messages": [{"role": "user", "content": "Q1"}],
            },
            {
                "prompt_id": "p2",
                "bucket": "metasploit",
                "category": "exploit",
                "messages": [{"role": "user", "content": "Q2"}],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            prompts_path = tmpdir / "prompts.jsonl"
            _write_jsonl(prompts_path, prompts)
            output_dir = tmpdir / "continuations"

            rc = cr.main(
                [
                    "--base-model",
                    "Qwen/Qwen2.5-7B-Instruct",
                    "--prompts",
                    str(prompts_path),
                    "--output-dir",
                    str(output_dir),
                    "--max-new-tokens",
                    "32",
                    "--seed",
                    "42",
                ]
            )

            self.assertEqual(rc, 0)
            self.assertTrue(output_dir.exists())
            self.assertTrue((output_dir / "p1.json").exists())
            self.assertTrue((output_dir / "p2.json").exists())

            with open(output_dir / "p1.json") as f:
                data = json.load(f)
            self.assertEqual(data["prompt_id"], "p1")
            self.assertIn("continuation", data)

    @patch("collect_reference.load_model_and_tokenizer")
    def test_main_empty_prompts_returns_error(self, mock_load):
        """Empty prompts file should return exit code 1."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        mock_load.return_value = (model, tokenizer)

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            prompts_path = tmpdir / "empty.jsonl"
            prompts_path.write_text("")
            output_dir = tmpdir / "continuations"

            rc = cr.main(
                [
                    "--base-model",
                    "Qwen/Qwen2.5-7B-Instruct",
                    "--prompts",
                    str(prompts_path),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(rc, 1)

    @patch("collect_reference.load_model_and_tokenizer")
    def test_main_model_load_failure(self, mock_load):
        """Model load failure should return exit code 1."""
        mock_load.side_effect = FileNotFoundError("Model not found")

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            prompts_path = tmpdir / "prompts.jsonl"
            _write_jsonl(
                prompts_path,
                [
                    {"prompt_id": "p1", "bucket": "b", "category": "c", "messages": []},
                ],
            )
            output_dir = tmpdir / "continuations"

            rc = cr.main(
                [
                    "--base-model",
                    "nonexistent/model",
                    "--prompts",
                    str(prompts_path),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
