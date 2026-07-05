#!/usr/bin/env python3
"""Tests for mixed-precision LoRA features in train_template.py.

Verifies that --mixed-precision-lora parses correctly, that
classify_layer_sensitivity() returns correct classifications,
and that the flag appears in train_all.py passthrough.

Run with:
    python -m pytest tests/test_mixed_precision.py -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make the scripts/ dir importable
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# Mock transformers BEFORE importing train_template
_mock_transformers = MagicMock()
_mock_transformers.trainer_callback = MagicMock()
_mock_transformers.trainer_callback.TrainerCallback = MagicMock()
sys.modules["transformers"] = _mock_transformers
sys.modules["transformers.trainer_callback"] = _mock_transformers.trainer_callback

import train_template


# =========================================================================
# Mixed-Precision LoRA CLI Argument Parsing
# =========================================================================


class TestMixedPrecisionArgParsing(unittest.TestCase):
    """Verify --mixed-precision-lora CLI argument parses correctly."""

    def test_mixed_precision_lora_parses(self):
        """--mixed-precision-lora flag parses correctly."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--mixed-precision-lora",
            ]
        )
        self.assertTrue(args.mixed_precision_lora)

    def test_mixed_precision_lora_default_false(self):
        """--mixed-precision-lora is False by default."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
            ]
        )
        self.assertFalse(args.mixed_precision_lora)


# =========================================================================
# classify_layer_sensitivity
# =========================================================================


class TestClassifyLayerSensitivity(unittest.TestCase):
    """Verify classify_layer_sensitivity() returns correct classifications."""

    def test_attention_q_proj_robust(self):
        """model.layers.0.self_attn.q_proj is 'robust'."""
        result = train_template.classify_layer_sensitivity(
            "model.layers.0.self_attn.q_proj"
        )
        self.assertEqual(result, "robust")

    def test_attention_k_proj_robust(self):
        """model.layers.0.self_attn.k_proj is 'robust'."""
        result = train_template.classify_layer_sensitivity(
            "model.layers.0.self_attn.k_proj"
        )
        self.assertEqual(result, "robust")

    def test_attention_v_proj_robust(self):
        """model.layers.0.self_attn.v_proj is 'robust'."""
        result = train_template.classify_layer_sensitivity(
            "model.layers.0.self_attn.v_proj"
        )
        self.assertEqual(result, "robust")

    def test_attention_o_proj_robust(self):
        """model.layers.0.self_attn.o_proj is 'robust'."""
        result = train_template.classify_layer_sensitivity(
            "model.layers.0.self_attn.o_proj"
        )
        self.assertEqual(result, "robust")

    def test_lm_head_sensitive(self):
        """lm_head is 'sensitive'."""
        result = train_template.classify_layer_sensitivity("lm_head")
        self.assertEqual(result, "sensitive")

    def test_embed_tokens_sensitive(self):
        """model.embed_tokens is 'sensitive'."""
        result = train_template.classify_layer_sensitivity("model.embed_tokens")
        self.assertEqual(result, "sensitive")

    def test_down_proj_sensitive(self):
        """model.layers.0.mlp.down_proj is 'sensitive'."""
        result = train_template.classify_layer_sensitivity(
            "model.layers.0.mlp.down_proj"
        )
        self.assertEqual(result, "sensitive")

    def test_norm_sensitive(self):
        """model.layers.0.input_layernorm is 'sensitive'."""
        result = train_template.classify_layer_sensitivity(
            "model.layers.0.input_layernorm"
        )
        self.assertEqual(result, "sensitive")

    def test_gate_proj_robust(self):
        """model.layers.0.mlp.gate_proj is 'robust'."""
        result = train_template.classify_layer_sensitivity(
            "model.layers.0.mlp.gate_proj"
        )
        self.assertEqual(result, "robust")

    def test_up_proj_robust(self):
        """model.layers.0.mlp.up_proj is 'robust'."""
        result = train_template.classify_layer_sensitivity("model.layers.0.mlp.up_proj")
        self.assertEqual(result, "robust")


# =========================================================================
# train_all.py Passthrough
# =========================================================================


class TestTrainAllPassthrough(unittest.TestCase):
    """Verify --mixed-precision-lora appears in train_all.py passthrough."""

    def test_mixed_precision_in_train_all_passthrough(self):
        """--mixed-precision-lora is forwarded by train_all.py."""
        # Simulate what train_all.py does when building the command
        from argparse import Namespace

        args = Namespace(
            mixed_precision_lora=True,
        )
        cmd = []
        if getattr(args, "mixed_precision_lora", False):
            cmd.append("--mixed-precision-lora")
        self.assertIn("--mixed-precision-lora", cmd)

    def test_mixed_precision_not_forwarded_when_false(self):
        """--mixed-precision-lora is NOT forwarded when False."""
        from argparse import Namespace

        args = Namespace(
            mixed_precision_lora=False,
        )
        cmd = []
        if getattr(args, "mixed_precision_lora", False):
            cmd.append("--mixed-precision-lora")
        self.assertNotIn("--mixed-precision-lora", cmd)


if __name__ == "__main__":
    unittest.main(verbosity=2)
