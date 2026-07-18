#!/usr/bin/env python3
"""Tests for scripts/_eval_loader.py — Shared model/tokenizer loader.

These tests are hermetic: they mock HuggingFace transformers and torch.cuda
to avoid GPU requirements.

Run with:
    python -m pytest tests/test_eval_loader.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

# Make the scripts/ dir importable
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# Save the real transformers module so we can restore it after tests.
# Mocking at module level pollutes sys.modules for all subsequent tests.
_real_transformers = sys.modules.get("transformers")

# Mock device_utils and transformers BEFORE importing _eval_loader
_device_utils_mock = MagicMock()
_device_utils_mock.is_cuda.return_value = False
sys.modules["device_utils"] = _device_utils_mock

_mock_transformers = MagicMock()
_mock_auto_model = MagicMock()
_mock_auto_tokenizer = MagicMock()
_mock_transformers.AutoModelForCausalLM = _mock_auto_model
_mock_transformers.AutoTokenizer = _mock_auto_tokenizer
_mock_transformers.utils = MagicMock()
_mock_transformers.utils.PushToHubMixin = MagicMock()
sys.modules["transformers"] = _mock_transformers
sys.modules["transformers.utils"] = _mock_transformers.utils

_mock_peft = MagicMock()
_mock_peft.PeftModel = MagicMock()
sys.modules["peft"] = _mock_peft

import _eval_loader  # noqa: E402


def tearDownModule():
    """Restore the real transformers module after this test module finishes."""
    if _real_transformers is not None:
        sys.modules["transformers"] = _real_transformers
    else:
        sys.modules.pop("transformers", None)
    sys.modules.pop("transformers.utils", None)
    sys.modules.pop("peft", None)
    sys.modules.pop("device_utils", None)


# ---------------------------------------------------------------------------
# Tests: resolve_model_path
# ---------------------------------------------------------------------------


class TestResolveModelPath(unittest.TestCase):
    """Verify path resolution logic."""

    def test_hf_model_id_passed_through(self):
        """HuggingFace model IDs should be returned as-is."""
        result = _eval_loader.resolve_model_path("Qwen/Qwen2.5-Coder-3B-Instruct")
        self.assertEqual(result, "Qwen/Qwen2.5-Coder-3B-Instruct")

    def test_hf_model_id_with_trailing_slash(self):
        """Trailing slashes should be stripped from HF model IDs."""
        result = _eval_loader.resolve_model_path("org/model/")
        self.assertEqual(result, "org/model")

    def test_local_absolute_path_exists(self):
        """Absolute paths that exist should be returned as-is."""
        with tempfile.TemporaryDirectory() as tmp:
            result = _eval_loader.resolve_model_path(tmp)
            self.assertEqual(result, tmp)

    def test_local_absolute_path_nonexistent_raises(self):
        """Nonexistent absolute paths should raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            _eval_loader.resolve_model_path("/nonexistent/path/12345")

    def test_local_path_starting_with_slash_raises_if_not_exists(self):
        """Paths starting with / that don't exist should raise."""
        with self.assertRaises(FileNotFoundError):
            _eval_loader.resolve_model_path("/tmp/definitely_not_a_real_model_xyz")

    def test_local_relative_path(self):
        """Relative paths that exist should be resolved."""
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                os.makedirs("my_model", exist_ok=True)
                result = _eval_loader.resolve_model_path("my_model")
                self.assertTrue(os.path.isabs(result))
            finally:
                os.chdir(old_cwd)

    def test_nonexistent_relative_path_raises(self):
        """Bare names without path prefixes are treated as HF model IDs."""
        result = _eval_loader.resolve_model_path("nonexistent_dir_xyz")
        self.assertEqual(result, "nonexistent_dir_xyz")  # passed through as HF ID

    def test_empty_string(self):
        """Empty string should raise ValueError."""
        with self.assertRaises(ValueError):
            _eval_loader.resolve_model_path("")


# ---------------------------------------------------------------------------
# Tests: detect_compute_dtype
# ---------------------------------------------------------------------------


class TestDetectComputeDtype(unittest.TestCase):
    """Verify compute dtype detection."""

    def test_user_specified_bf16(self):
        """User-specified bf16 should be respected."""
        self.assertEqual(_eval_loader.detect_compute_dtype("bf16"), torch.bfloat16)

    def test_user_specified_fp16(self):
        """User-specified fp16 should be respected."""
        self.assertEqual(_eval_loader.detect_compute_dtype("fp16"), torch.float16)

    def test_user_specified_fp32(self):
        """User-specified fp32 should be respected."""
        self.assertEqual(_eval_loader.detect_compute_dtype("fp32"), torch.float32)

    def test_user_specified_case_insensitive(self):
        """Dtype strings should be case-insensitive."""
        self.assertEqual(_eval_loader.detect_compute_dtype("BF16"), torch.bfloat16)

    @patch("_eval_loader.is_cuda", return_value=False)
    def test_user_specified_unknown_falls_back(self, _mock_is_cuda):
        """Unknown dtype strings should fall back to fp32 on CPU."""
        self.assertEqual(_eval_loader.detect_compute_dtype("fp8"), torch.float32)

    @patch("_eval_loader.is_cuda", return_value=False)
    def test_none_auto_detect_cpu(self, _mock_is_cuda):
        """None should auto-detect (fp32 on CPU)."""
        self.assertEqual(_eval_loader.detect_compute_dtype(None), torch.float32)


# ---------------------------------------------------------------------------
# Tests: load_model_and_tokenizer
# ---------------------------------------------------------------------------


class TestLoadModelAndTokenizer(unittest.TestCase):
    """Verify model loading with mocked transformers."""

    def test_load_base_model(self):
        """Loading a base model should return model and tokenizer."""
        model, tokenizer = _eval_loader.load_model_and_tokenizer(
            "Qwen/Qwen2.5-Coder-3B-Instruct",
            adapter_path=None,
            compute_dtype=torch.float32,
        )
        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)

    def test_load_with_adapter(self):
        """Loading with an adapter should apply PEFT."""
        with tempfile.TemporaryDirectory() as tmp:
            adapter_dir = Path(tmp) / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text("{}")
            model, tokenizer = _eval_loader.load_model_and_tokenizer(
                "Qwen/Qwen2.5-Coder-3B-Instruct",
                adapter_path=str(adapter_dir),
                compute_dtype=torch.float32,
            )
            self.assertIsNotNone(model)
            self.assertIsNotNone(tokenizer)

    def test_load_with_nonexistent_adapter_raises(self):
        """Nonexistent adapter path should raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            _eval_loader.load_model_and_tokenizer(
                "Qwen/Qwen2.5-Coder-3B-Instruct",
                adapter_path="/nonexistent/adapter",
                compute_dtype=torch.float32,
            )

    def test_tokenizer_already_has_pad_token(self):
        """Tokenizer with existing pad_token should not be modified."""
        model, tokenizer = _eval_loader.load_model_and_tokenizer(
            "Qwen/Qwen2.5-Coder-3B-Instruct",
            adapter_path=None,
            compute_dtype=torch.float32,
        )
        self.assertIsNotNone(tokenizer.pad_token)


if __name__ == "__main__":
    unittest.main(verbosity=2)
