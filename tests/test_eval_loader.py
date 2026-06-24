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

# ---------------------------------------------------------------------------
# Mock device_utils and transformers BEFORE importing _eval_loader
# ---------------------------------------------------------------------------
_device_utils_mock = MagicMock()
_device_utils_mock.is_cuda.return_value = False
sys.modules["device_utils"] = _device_utils_mock

# Mock transformers module so from_pretrained calls don't hit HuggingFace
_mock_transformers = MagicMock()
_mock_auto_model = MagicMock()
_mock_auto_tokenizer = MagicMock()
_mock_transformers.AutoModelForCausalLM = _mock_auto_model
_mock_transformers.AutoTokenizer = _mock_auto_tokenizer
sys.modules["transformers"] = _mock_transformers

import _eval_loader  # noqa: E402


# ---------------------------------------------------------------------------
# Tests: resolve_model_path
# ---------------------------------------------------------------------------


class TestResolveModelPath(unittest.TestCase):
    """Verify path resolution logic."""

    def test_hf_model_id_passed_through(self):
        """HF Hub model IDs should be returned as-is."""
        result = _eval_loader.resolve_model_path("Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(result, "Qwen/Qwen2.5-7B-Instruct")

    def test_hf_model_id_with_trailing_slash(self):
        """Trailing slashes should be stripped."""
        result = _eval_loader.resolve_model_path("Qwen/Qwen2.5-7B-Instruct/")
        self.assertEqual(result, "Qwen/Qwen2.5-7B-Instruct")

    def test_local_absolute_path_exists(self):
        """Existing absolute path should be resolved."""
        with tempfile.TemporaryDirectory() as tmp:
            result = _eval_loader.resolve_model_path(tmp)
            self.assertEqual(result, str(Path(tmp).resolve()))

    def test_local_absolute_path_nonexistent_raises(self):
        """Non-existent absolute path should raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            _eval_loader.resolve_model_path("/nonexistent/path/to/model")

    def test_local_relative_path(self):
        """Relative path starting with ./ should be resolved."""
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                model_dir = Path(tmp) / "my_model"
                model_dir.mkdir()
                result = _eval_loader.resolve_model_path("./my_model")
                self.assertTrue(Path(result).is_absolute())
                self.assertTrue(Path(result).exists())
            finally:
                os.chdir(cwd)

    def test_local_path_starting_with_slash_raises_if_not_exists(self):
        """Path starting with / that doesn't exist should raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            _eval_loader.resolve_model_path("/nonexistent_local_path")

    def test_empty_string(self):
        """Empty string should be returned as-is."""
        result = _eval_loader.resolve_model_path("")
        self.assertEqual(result, "")

    def test_nonexistent_relative_path_raises(self):
        """Non-existent relative path should raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            _eval_loader.resolve_model_path("./nonexistent_relative_dir")


# ---------------------------------------------------------------------------
# Tests: detect_compute_dtype
# ---------------------------------------------------------------------------


class TestDetectComputeDtype(unittest.TestCase):
    """Verify compute dtype auto-detection."""

    def test_user_specified_bf16(self):
        dtype = _eval_loader.detect_compute_dtype("bf16")
        self.assertEqual(dtype, torch.bfloat16)

    def test_user_specified_fp16(self):
        dtype = _eval_loader.detect_compute_dtype("fp16")
        self.assertEqual(dtype, torch.float16)

    def test_user_specified_fp32(self):
        dtype = _eval_loader.detect_compute_dtype("fp32")
        self.assertEqual(dtype, torch.float32)

    def test_user_specified_case_insensitive(self):
        dtype = _eval_loader.detect_compute_dtype("BF16")
        self.assertEqual(dtype, torch.bfloat16)

    def test_user_specified_unknown_falls_back(self):
        """Unknown dtype string should fall back to auto-detect (fp32 on CPU)."""
        dtype = _eval_loader.detect_compute_dtype("unknown")
        self.assertEqual(dtype, torch.float32)

    def test_none_auto_detect_cpu(self):
        """None should auto-detect to fp32 on CPU."""
        dtype = _eval_loader.detect_compute_dtype(None)
        self.assertEqual(dtype, torch.float32)


# ---------------------------------------------------------------------------
# Tests: load_model_and_tokenizer (mocked)
# ---------------------------------------------------------------------------


class TestLoadModelAndTokenizer(unittest.TestCase):
    """Verify model loading with mocked transformers."""

    def setUp(self):
        # Reset mock call counts
        _mock_auto_model.reset_mock()
        _mock_auto_tokenizer.reset_mock()
        # Remove peft from sys.modules if present to test without adapter
        self._peft_backup = sys.modules.pop("peft", None)

    def tearDown(self):
        if self._peft_backup is not None:
            sys.modules["peft"] = self._peft_backup

    def test_load_base_model(self):
        """Loading base model should return model and tokenizer."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "[EOS]"
        _mock_auto_model.from_pretrained.return_value = mock_model
        _mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        model, tokenizer = _eval_loader.load_model_and_tokenizer(
            "Qwen/Qwen2.5-7B-Instruct", None, torch.float32
        )

        self.assertIs(model, mock_model)
        self.assertIs(tokenizer, mock_tokenizer)
        mock_model.eval.assert_called_once()
        # pad_token should be set to eos_token since it was None
        self.assertEqual(tokenizer.pad_token, "[EOS]")

    def test_load_with_adapter(self):
        """Loading with adapter should apply PeftModel."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<|endoftext|>"
        mock_tokenizer.eos_token = "<|endoftext|>"
        _mock_auto_model.from_pretrained.return_value = mock_model
        _mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        with tempfile.TemporaryDirectory() as tmp:
            adapter_dir = Path(tmp) / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text("{}")

            # Mock peft module
            mock_peft = MagicMock()
            mock_peft_model = MagicMock()
            mock_peft.PeftModel = MagicMock()
            mock_peft.PeftModel.from_pretrained.return_value = mock_peft_model
            sys.modules["peft"] = mock_peft

            model, tokenizer = _eval_loader.load_model_and_tokenizer(
                "Qwen/Qwen2.5-7B-Instruct", str(adapter_dir), torch.float32
            )

            self.assertIs(model, mock_peft_model)
            mock_peft.PeftModel.from_pretrained.assert_called_once()
            mock_peft_model.eval.assert_called_once()

    def test_load_with_nonexistent_adapter_raises(self):
        """Non-existent adapter path should raise FileNotFoundError."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<|endoftext|>"
        _mock_auto_model.from_pretrained.return_value = mock_model
        _mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Mock peft module so the import doesn't fail
        mock_peft = MagicMock()
        sys.modules["peft"] = mock_peft

        with self.assertRaises(FileNotFoundError):
            _eval_loader.load_model_and_tokenizer(
                "Qwen/Qwen2.5-7B-Instruct",
                "/nonexistent/adapter/path",
                torch.float32,
            )

    def test_tokenizer_already_has_pad_token(self):
        """If tokenizer already has pad_token, it should not be overwritten."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<|pad|>"
        mock_tokenizer.eos_token = "[EOS]"
        _mock_auto_model.from_pretrained.return_value = mock_model
        _mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        model, tokenizer = _eval_loader.load_model_and_tokenizer(
            "Qwen/Qwen2.5-7B-Instruct", None, torch.float32
        )

        # pad_token should remain as-is since it was already set
        self.assertEqual(tokenizer.pad_token, "<|pad|>")


if __name__ == "__main__":
    unittest.main(verbosity=2)
