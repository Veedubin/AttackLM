#!/usr/bin/env python3
"""Tests for scripts/golden_vectors.py — Golden Vector Generation & Validation.

These tests are hermetic: they mock HuggingFace models, tokenizers, and file I/O
to avoid GPU requirements.

Run with:
    python -m pytest tests/test_golden_vectors.py -v
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

# Mock device_utils BEFORE importing golden_vectors
_device_utils_mock = MagicMock()
_device_utils_mock.is_cuda.return_value = False
_device_utils_mock.print_hardware_banner.return_value = "cpu"
sys.modules["device_utils"] = _device_utils_mock

import golden_vectors as gv  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _make_mock_model_and_tokenizer():
    """Create mock model and tokenizer for GPU-free testing.

    Returns real tensors so JSON serialization and torch operations work.
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
            return result

        def decode(self, ids, **kwargs):
            return "mock_token_bytes"

    tokenizer = _MockTokenizer()

    model = MagicMock()
    model.device = "cpu"

    # generate: return mock with sequences and scores
    def _generate(**kwargs):
        input_ids = kwargs.get("input_ids", torch.tensor([[1, 2, 3]]))
        input_len = input_ids.shape[1]

        # Mock scores: list of tensors for each generated position
        mock_scores = [
            torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]]),
            torch.tensor([[0.5, 0.4, 0.3, 0.2, 0.1]]),
            torch.tensor([[0.3, 0.3, 0.3, 0.3, 0.3]]),
        ]

        mock_output = MagicMock()
        # sequences[0] returns a real tensor
        mock_output.sequences = torch.tensor([[1] * (input_len + 3)])
        mock_output.scores = mock_scores
        return mock_output

    model.generate = _generate

    return model, tokenizer


# ---------------------------------------------------------------------------
# Tests: CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLIParsing(unittest.TestCase):
    """Verify CLI argument parsing and defaults."""

    def test_required_args_generate(self):
        """Generate subcommand requires its args."""
        with self.assertRaises(SystemExit):
            gv.parse_args(["generate"])

    def test_generate_minimal_args(self):
        """Minimal generate args should parse."""
        args = gv.parse_args(
            [
                "generate",
                "--base-model",
                "Qwen/Qwen2.5-7B-Instruct",
                "--prompts",
                "prompts.jsonl",
                "--output",
                "vectors.json",
            ]
        )
        self.assertEqual(args.command, "generate")
        self.assertEqual(args.max_new_tokens, 64)
        self.assertEqual(args.top_k, 20)
        self.assertEqual(args.seed, 42)

    def test_validate_minimal_args(self):
        """Minimal validate args should parse."""
        args = gv.parse_args(
            [
                "validate",
                "--base-model",
                "Qwen/Qwen2.5-7B-Instruct",
                "--golden",
                "vectors.json",
                "--output",
                "report.json",
            ]
        )
        self.assertEqual(args.command, "validate")
        self.assertEqual(args.seed, 42)
        self.assertIsNone(args.compute_dtype)

    def test_generate_all_args(self):
        """All generate args should parse."""
        args = gv.parse_args(
            [
                "generate",
                "--base-model",
                "huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated",
                "--adapter",
                "models/attacklm-single",
                "--prompts",
                "data/golden/prompts.jsonl",
                "--output",
                "data/golden/vectors.json",
                "--max-new-tokens",
                "128",
                "--top-k",
                "10",
                "--seed",
                "99",
            ]
        )
        self.assertEqual(args.max_new_tokens, 128)
        self.assertEqual(args.top_k, 10)
        self.assertEqual(args.seed, 99)

    def test_validate_all_args(self):
        """All validate args should parse."""
        args = gv.parse_args(
            [
                "validate",
                "--base-model",
                "Qwen/Qwen2.5-7B-Instruct",
                "--adapter",
                "models/attacklm-7b",
                "--golden",
                "data/golden/vectors.json",
                "--output",
                "evals/validation_report.json",
                "--seed",
                "99",
                "--compute-dtype",
                "bf16",
            ]
        )
        self.assertEqual(args.adapter, "models/attacklm-7b")
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
            prompts = gv.load_prompts(str(path))
            self.assertEqual(len(prompts), 2)

    def test_skip_invalid_json(self):
        """Invalid JSON lines should be skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            with open(path, "w") as f:
                f.write(
                    '{"prompt_id": "p1", "bucket": "b", "category": "c", "messages": []}\n'
                )
                f.write("not valid json\n")
            prompts = gv.load_prompts(str(path))
            self.assertEqual(len(prompts), 1)

    def test_skip_missing_fields(self):
        """Records with missing required fields should be skipped."""
        records = [
            {"prompt_id": "p1", "bucket": "b", "category": "c", "messages": []},
            {"prompt_id": "p2"},  # missing bucket, category, messages
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            _write_jsonl(path, records)
            prompts = gv.load_prompts(str(path))
            self.assertEqual(len(prompts), 1)


# ---------------------------------------------------------------------------
# Tests: cmd_generate (mocked)
# ---------------------------------------------------------------------------


class TestCmdGenerate(unittest.TestCase):
    """Verify generate subcommand."""

    @patch("golden_vectors.load_model_and_tokenizer")
    def test_generate_creates_vector_file(self, mock_load):
        """cmd_generate should create a vectors JSON file."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        mock_load.return_value = (model, tokenizer)

        prompts = [
            {
                "prompt_id": "p1",
                "bucket": "mitre",
                "category": "T1569.002",
                "messages": [{"role": "user", "content": "Q1"}],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            prompts_path = tmpdir / "prompts.jsonl"
            _write_jsonl(prompts_path, prompts)
            output_path = tmpdir / "vectors.json"

            args = MagicMock()
            args.base_model = "Qwen/Qwen2.5-7B-Instruct"
            args.adapter = None
            args.prompts = str(prompts_path)
            args.output = str(output_path)
            args.max_new_tokens = 16
            args.top_k = 5
            args.seed = 42

            rc = gv.cmd_generate(args)
            self.assertEqual(rc, 0)
            self.assertTrue(output_path.exists())

            with open(output_path) as f:
                data = json.load(f)
            self.assertIn("metadata", data)
            self.assertIn("vectors", data)
            self.assertIn("p1", data["vectors"])
            self.assertIn("positions", data["vectors"]["p1"])

    @patch("golden_vectors.load_model_and_tokenizer")
    def test_generate_empty_prompts_returns_error(self, mock_load):
        """Empty prompts should return exit code 1."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        mock_load.return_value = (model, tokenizer)

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            prompts_path = tmpdir / "empty.jsonl"
            prompts_path.write_text("")
            output_path = tmpdir / "vectors.json"

            args = MagicMock()
            args.base_model = "Qwen/Qwen2.5-7B-Instruct"
            args.adapter = None
            args.prompts = str(prompts_path)
            args.output = str(output_path)
            args.max_new_tokens = 16
            args.top_k = 5
            args.seed = 42

            rc = gv.cmd_generate(args)
            self.assertEqual(rc, 1)

    @patch("golden_vectors.load_model_and_tokenizer")
    def test_generate_model_load_failure(self, mock_load):
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
            output_path = tmpdir / "vectors.json"

            args = MagicMock()
            args.base_model = "nonexistent/model"
            args.adapter = None
            args.prompts = str(prompts_path)
            args.output = str(output_path)
            args.max_new_tokens = 16
            args.top_k = 5
            args.seed = 42

            rc = gv.cmd_generate(args)
            self.assertEqual(rc, 1)


# ---------------------------------------------------------------------------
# Tests: cmd_validate (mocked)
# ---------------------------------------------------------------------------


class TestCmdValidate(unittest.TestCase):
    """Verify validate subcommand."""

    def _make_golden_file(self, path: Path) -> None:
        """Create a golden vectors JSON file."""
        data = {
            "metadata": {
                "model": "reference-model",
                "adapter": "none",
                "timestamp": "2026-06-22T00:00:00Z",
                "num_prompts": 1,
                "generation_config": {
                    "temperature": 0.0,
                    "max_new_tokens": 16,
                    "do_sample": False,
                    "seed": 42,
                },
            },
            "vectors": {
                "p1": {
                    "prompt_token_count": 10,
                    "positions": [
                        {
                            "pos": 0,
                            "token_id": 1,
                            "token_bytes": "mock_token_bytes",
                            "top20_logprobs": {"1": -0.5, "2": -1.0, "3": -1.5},
                        },
                        {
                            "pos": 1,
                            "token_id": 2,
                            "token_bytes": "mock_token_bytes",
                            "top20_logprobs": {"1": -0.3, "2": -0.6, "3": -0.9},
                        },
                    ],
                },
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @patch("golden_vectors.load_model_and_tokenizer")
    def test_validate_creates_report(self, mock_load):
        """cmd_validate should create a validation report JSON file."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        mock_load.return_value = (model, tokenizer)

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            golden_path = tmpdir / "golden.json"
            self._make_golden_file(golden_path)
            output_path = tmpdir / "report.json"

            # Create prompts.jsonl next to golden for messages lookup
            _write_jsonl(
                tmpdir / "prompts.jsonl",
                [
                    {
                        "prompt_id": "p1",
                        "bucket": "b",
                        "category": "c",
                        "messages": [{"role": "user", "content": "Q1"}],
                    },
                ],
            )

            args = MagicMock()
            args.base_model = "Qwen/Qwen2.5-7B-Instruct"
            args.adapter = None
            args.golden = str(golden_path)
            args.output = str(output_path)
            args.seed = 42
            args.compute_dtype = None

            rc = gv.cmd_validate(args)
            # cmd_validate returns 0 for PASS/WARN, 1 for FAIL
            # With mocked model, byte_match_rate=0.0 and mean_spearman_rho=0.0 → FAIL → returns 1
            # We just check the report was created
            self.assertTrue(output_path.exists())

            with open(output_path) as f:
                report = json.load(f)
            self.assertIn("summary", report)
            self.assertIn("verdict", report["summary"])
            self.assertIn("per_prompt", report)
            self.assertIn("p1", report["per_prompt"])

    @patch("golden_vectors.load_model_and_tokenizer")
    def test_validate_nonexistent_golden(self, mock_load):
        """Non-existent golden file should return exit code 1."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        mock_load.return_value = (model, tokenizer)

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "report.json"

            args = MagicMock()
            args.base_model = "Qwen/Qwen2.5-7B-Instruct"
            args.adapter = None
            args.golden = str(Path(tmp) / "nonexistent.json")
            args.output = str(output_path)
            args.seed = 42
            args.compute_dtype = None

            rc = gv.cmd_validate(args)
            self.assertEqual(rc, 1)

    @patch("golden_vectors.load_model_and_tokenizer")
    def test_validate_model_load_failure(self, mock_load):
        """Model load failure should return exit code 1."""
        mock_load.side_effect = FileNotFoundError("Model not found")

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            golden_path = tmpdir / "golden.json"
            self._make_golden_file(golden_path)
            output_path = tmpdir / "report.json"

            args = MagicMock()
            args.base_model = "nonexistent/model"
            args.adapter = None
            args.golden = str(golden_path)
            args.output = str(output_path)
            args.seed = 42
            args.compute_dtype = None

            rc = gv.cmd_validate(args)
            self.assertEqual(rc, 1)


# ---------------------------------------------------------------------------
# Tests: main dispatch
# ---------------------------------------------------------------------------


class TestMain(unittest.TestCase):
    """Verify main() dispatches to correct subcommand."""

    @patch("golden_vectors.cmd_generate")
    def test_main_generate(self, mock_cmd):
        """main() with generate should call cmd_generate."""
        mock_cmd.return_value = 0
        rc = gv.main(
            ["generate", "--base-model", "m", "--prompts", "p", "--output", "o"]
        )
        self.assertEqual(rc, 0)
        mock_cmd.assert_called_once()

    @patch("golden_vectors.cmd_validate")
    def test_main_validate(self, mock_cmd):
        """main() with validate should call cmd_validate."""
        mock_cmd.return_value = 0
        rc = gv.main(
            ["validate", "--base-model", "m", "--golden", "g", "--output", "o"]
        )
        self.assertEqual(rc, 0)
        mock_cmd.assert_called_once()

    def test_main_unknown_command(self):
        """Unknown command should raise SystemExit (from argparse)."""
        with self.assertRaises(SystemExit):
            gv.main(["unknown"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
