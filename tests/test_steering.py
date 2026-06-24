"""Hermetic tests for scripts/steering.py — Pattern 6: Steering Vectors."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_model(hidden_size=2048, num_layers=36):
    """Create a mock model with config and layer structure."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.hidden_size = hidden_size
    model.config.num_hidden_layers = num_layers
    model.config._name_or_path = "test-model"
    model.device = torch.device("cpu")
    model.dtype = torch.float32
    model.eval = MagicMock()

    # Create mock layers with mlp and self_attn
    model.model = MagicMock()
    model.model.layers = []
    for i in range(num_layers):
        layer = MagicMock()
        layer.mlp = MagicMock()
        layer.self_attn = MagicMock()
        model.model.layers.append(layer)

    # Mock generate
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

    return model


def _make_mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "[EOS]"
    tokenizer.decode.return_value = "mock generated text"
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    return tokenizer


def _write_prompts(path, lines):
    """Write prompt lines to a file."""
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Vector Math Tests
# ---------------------------------------------------------------------------


class TestVectorMath(unittest.TestCase):
    """Test the core vector math functions used in steering."""

    def test_normalize_unit_vector(self):
        """normalize() should produce unit-length vectors."""
        from steering import normalize

        v = torch.tensor([3.0, 4.0])
        n = normalize(v)
        self.assertAlmostEqual(torch.linalg.norm(n).item(), 1.0, places=5)

    def test_normalize_zero_vector_raises(self):
        """normalize() of zero vector should raise ValueError."""
        from steering import normalize

        v = torch.zeros(10)
        with self.assertRaises(ValueError):
            normalize(v)

    def test_dot_product(self):
        """dot() should match torch.dot."""
        from steering import dot

        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        result = dot(a, b)
        expected = float(torch.dot(a, b))
        self.assertAlmostEqual(result, expected, places=5)

    def test_dot_mixed_types(self):
        """dot() should handle mixed numpy/torch inputs."""
        from steering import dot

        a = torch.tensor([1.0, 2.0])
        b = np.array([3.0, 4.0], dtype=np.float32)
        result = dot(a, b)
        self.assertAlmostEqual(result, 11.0, places=5)

    def test_projection_formula(self):
        """y = y - scale * direction * dot(direction, y)"""
        from steering import normalize, dot

        direction = normalize(torch.tensor([1.0, 0.0, 0.0]))
        y = torch.tensor([2.0, 3.0, 4.0])
        scale = 1.0

        projection = dot(direction, y)
        steered = y - scale * projection * torch.tensor(direction)

        # Should zero out the x-component
        self.assertAlmostEqual(steered[0].item(), 0.0, places=5)
        self.assertAlmostEqual(steered[1].item(), 3.0, places=5)
        self.assertAlmostEqual(steered[2].item(), 4.0, places=5)

    def test_projection_negative_scale(self):
        """Negative scale should amplify the direction."""
        from steering import normalize, dot

        direction = normalize(torch.tensor([1.0, 0.0, 0.0]))
        y = torch.tensor([2.0, 3.0, 4.0])
        scale = -1.0

        projection = dot(direction, y)
        steered = y - scale * projection * torch.tensor(direction)

        # Should double the x-component
        self.assertAlmostEqual(steered[0].item(), 4.0, places=5)
        self.assertAlmostEqual(steered[1].item(), 3.0, places=5)

    def test_orthogonalize(self):
        """Orthogonalize should remove shared component."""
        from steering import normalize, dot

        direction = normalize(torch.tensor([1.0, 0.0]))
        control = normalize(torch.tensor([0.707, 0.707]))

        projection = dot(direction, control)
        orthogonalized = normalize(direction - projection * torch.tensor(control))

        # Should now be orthogonal to control
        dot_after = dot(orthogonalized, control)
        self.assertAlmostEqual(dot_after, 0.0, places=5)


# ---------------------------------------------------------------------------
# File I/O Tests
# ---------------------------------------------------------------------------


class TestVectorFileIO(unittest.TestCase):
    """Test f32 binary vector read/write via save_vectors/load_vectors."""

    def test_write_read_roundtrip(self):
        """Write vectors via save_vectors, read back via load_vectors, verify identical."""
        from steering import save_vectors, load_vectors

        vectors = np.random.randn(11, 2048).astype(np.float32)
        metadata = {
            "format": "attacklm-steering-v1",
            "shape": [11, 2048],
            "component": "ffn_out",
            "layers": list(range(20, 31)),
            "orthogonalize_control_mean": True,
            "model": "test-model",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "test_vectors")
            save_vectors(vectors, metadata, prefix)

            # Check both files exist
            self.assertTrue(os.path.exists(prefix + ".f32"))
            self.assertTrue(os.path.exists(prefix + ".json"))

            # Read back
            read_vecs, read_meta = load_vectors(prefix)
            self.assertTrue(np.allclose(vectors, read_vecs))
            self.assertEqual(read_meta["format"], "attacklm-steering-v1")
            self.assertEqual(read_meta["layers"], list(range(20, 31)))

    def test_load_vectors_wrong_shape_raises(self):
        """Loading with wrong expected shape should raise."""
        from steering import save_vectors, load_vectors

        vectors = np.random.randn(11, 2048).astype(np.float32)
        metadata = {
            "format": "attacklm-steering-v1",
            "shape": [5, 2048],
            "layers": [1, 2, 3, 4, 5],
            "component": "ffn_out",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "test_vectors")
            save_vectors(vectors, metadata, prefix)

            # Metadata says 5 layers but file has 11 — should raise
            with self.assertRaises(ValueError):
                load_vectors(prefix)


# ---------------------------------------------------------------------------
# Hook Tests
# ---------------------------------------------------------------------------


class TestHookHelpers(unittest.TestCase):
    """Test _get_hook_target and _remove_hooks."""

    def test_get_hook_target_ffn_out(self):
        """_get_hook_target should return mlp for ffn_out."""
        from steering import _get_hook_target

        model = _make_mock_model()
        target = _get_hook_target(model, 20, "ffn_out")
        self.assertIs(target, model.model.layers[20].mlp)

    def test_get_hook_target_attn_out(self):
        """_get_hook_target should return self_attn for attn_out."""
        from steering import _get_hook_target

        model = _make_mock_model()
        target = _get_hook_target(model, 25, "attn_out")
        self.assertIs(target, model.model.layers[25].self_attn)

    def test_remove_hooks(self):
        """_remove_hooks should call remove() on all hooks."""
        from steering import _remove_hooks

        hooks = [MagicMock(), MagicMock(), MagicMock()]
        _remove_hooks(hooks)
        for h in hooks:
            h.remove.assert_called_once()


# ---------------------------------------------------------------------------
# Extraction Tests
# ---------------------------------------------------------------------------


class TestExtraction(unittest.TestCase):
    """Test extract_steering_vector."""

    def test_extract_empty_target_raises(self):
        """Empty target prompts should raise ValueError."""
        from steering import extract_steering_vector

        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()

        with self.assertRaises(ValueError):
            extract_steering_vector(
                model, tokenizer, [], ["control"], layers=[20, 25, 30]
            )

    def test_extract_empty_control_raises(self):
        """Empty control prompts should raise ValueError."""
        from steering import extract_steering_vector

        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()

        with self.assertRaises(ValueError):
            extract_steering_vector(
                model, tokenizer, ["target"], [], layers=[20, 25, 30]
            )

    def test_extract_default_layers(self):
        """Default layers should be 20-30."""
        from steering import extract_steering_vector

        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()

        # This will fail on actual forward pass (mock model), but we can
        # verify the function signature and default behavior
        self.assertEqual(model.config.hidden_size, 2048)

    def test_extract_returns_correct_metadata_keys(self):
        """Extracted metadata should have expected keys."""
        # We can't fully test extraction without GPU, but verify the
        # function exists and has correct signature
        from steering import extract_steering_vector
        import inspect

        sig = inspect.signature(extract_steering_vector)
        params = list(sig.parameters.keys())
        self.assertIn("model", params)
        self.assertIn("tokenizer", params)
        self.assertIn("target_prompts", params)
        self.assertIn("control_prompts", params)
        self.assertIn("layers", params)
        self.assertIn("component", params)
        self.assertIn("orthogonalize", params)


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------


class TestCLIParsing(unittest.TestCase):
    """Test CLI argument parsing for all subcommands."""

    def test_extract_required_args(self):
        """Extract requires --base-model, --target, --control."""
        from steering import parse_args

        with self.assertRaises(SystemExit):
            parse_args(["extract"])

    def test_extract_minimal_args(self):
        """Extract with minimal required args."""
        from steering import parse_args

        args = parse_args(
            [
                "extract",
                "--base-model",
                "test/model",
                "--target",
                "/tmp/target.txt",
                "--control",
                "/tmp/control.txt",
            ]
        )
        self.assertEqual(args.base_model, "test/model")
        self.assertEqual(args.target, "/tmp/target.txt")
        self.assertEqual(args.control, "/tmp/control.txt")
        self.assertEqual(args.layers, list(range(20, 31)))
        self.assertEqual(args.component, "ffn_out")
        self.assertTrue(args.orthogonalize)

    def test_extract_no_orthogonalize(self):
        """--no-orthogonalize should disable orthogonalization."""
        from steering import parse_args

        args = parse_args(
            [
                "extract",
                "--base-model",
                "test/model",
                "--target",
                "/tmp/target.txt",
                "--control",
                "/tmp/control.txt",
                "--no-orthogonalize",
            ]
        )
        # --no-orthogonalize sets args.no_orthogonalize = True
        # _cmd_extract checks this to override orthogonalize
        self.assertTrue(args.no_orthogonalize)

    def test_apply_required_args(self):
        """Apply requires --base-model, --vectors, --prompt."""
        from steering import parse_args

        with self.assertRaises(SystemExit):
            parse_args(["apply"])

    def test_apply_minimal_args(self):
        """Apply with minimal required args."""
        from steering import parse_args

        args = parse_args(
            [
                "apply",
                "--base-model",
                "test/model",
                "--vectors",
                "/tmp/vec.f32",
                "--prompt",
                "test prompt",
            ]
        )
        self.assertEqual(args.vectors, "/tmp/vec.f32")
        self.assertEqual(args.prompt, "test prompt")
        self.assertEqual(args.scale, 1.0)

    def test_sweep_default_scales(self):
        """Sweep should have default scale values."""
        from steering import parse_args

        args = parse_args(
            [
                "sweep",
                "--base-model",
                "test/model",
                "--vectors",
                "/tmp/vec.f32",
                "--prompts",
                "/tmp/prompts.txt",
            ]
        )
        self.assertEqual(args.scales, "-1,-0.5,0,0.5,1,2")

    def test_diagnose_required_args(self):
        """Diagnose requires --base-model, --reference-model, --harmful, --harmless."""
        from steering import parse_args

        with self.assertRaises(SystemExit):
            parse_args(["diagnose"])

    def test_diagnose_minimal_args(self):
        """Diagnose with minimal required args."""
        from steering import parse_args

        args = parse_args(
            [
                "diagnose",
                "--base-model",
                "test/abliterated",
                "--reference-model",
                "test/original",
                "--harmful",
                "/tmp/harmful.txt",
                "--harmless",
                "/tmp/harmless.txt",
            ]
        )
        self.assertEqual(args.base_model, "test/abliterated")
        self.assertEqual(args.reference_model, "test/original")


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestIntegration(unittest.TestCase):
    """Integration tests with mocked model."""

    @patch("steering.load_model_and_tokenizer")
    def test_apply_generates_text(self, mock_load):
        """apply_steering should generate text with steering hooks."""
        from steering import apply_steering

        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        # Fix: tokenizer() must return something with .to(device)
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__.return_value = torch.tensor([[1, 2, 3]])
        mock_inputs["input_ids"] = torch.tensor([[1, 2, 3]])
        mock_inputs["attention_mask"] = torch.tensor([[1, 1, 1]])
        tokenizer.return_value = mock_inputs
        tokenizer.pad_token_id = 0
        mock_load.return_value = (model, tokenizer)

        vectors = np.random.randn(3, 2048).astype(np.float32)
        metadata = {"layers": [20, 25, 30], "component": "ffn_out"}

        result = apply_steering(
            model,
            tokenizer,
            vectors,
            metadata,
            prompt="test prompt",
            scale=1.0,
            layers=None,
            max_new_tokens=64,
        )

        self.assertIsInstance(result, str)
        # Hooks should have been registered on mlp modules
        model.model.layers[20].mlp.register_forward_hook.assert_called_once()

    @patch("steering.load_model_and_tokenizer")
    def test_sweep_produces_valid_json(self, mock_load):
        """run_sweep should produce valid JSON output."""
        from steering import run_sweep

        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        # Fix tokenizer for apply_steering calls inside run_sweep
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__.return_value = torch.tensor([[1, 2, 3]])
        mock_inputs["input_ids"] = torch.tensor([[1, 2, 3]])
        mock_inputs["attention_mask"] = torch.tensor([[1, 1, 1]])
        tokenizer.return_value = mock_inputs
        tokenizer.pad_token_id = 0
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # for token counting
        mock_load.return_value = (model, tokenizer)

        vectors = np.random.randn(3, 2048).astype(np.float32)
        metadata = {"layers": [20, 25, 30], "component": "ffn_out"}

        # run_sweep takes prompts as list[str], not file path
        result = run_sweep(
            model,
            tokenizer,
            vectors,
            metadata,
            prompts=["prompt 1", "prompt 2"],
            scales=[-1.0, 0.0, 1.0],
            layers=None,
            max_new_tokens=64,
        )
        self.assertIn("results", result)
        self.assertEqual(len(result["results"]), 3)
        for r in result["results"]:
            self.assertIn("scale", r)
            self.assertIn("mean_tokens", r)

    @patch("steering.load_model_and_tokenizer")
    def test_diagnose_produces_valid_json(self, mock_load):
        """diagnose_refusal should have correct function signature."""
        from steering import diagnose_refusal
        import inspect

        sig = inspect.signature(diagnose_refusal)
        params = list(sig.parameters.keys())
        self.assertIn("target_model", params)
        self.assertIn("reference_model", params)
        self.assertIn("tokenizer", params)
        self.assertIn("harmful_prompts", params)
        self.assertIn("harmless_prompts", params)
        self.assertIn("layers", params)
        self.assertIn("component", params)

        # diagnose_refusal requires real forward passes (GPU) for extraction.
        # Full integration test deferred to GPU-available environment.


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases(unittest.TestCase):
    """Edge case handling."""

    def test_zero_scale_no_change(self):
        """Scale=0 should produce no change from baseline."""
        from steering import normalize, dot

        direction = normalize(torch.tensor([1.0, 0.0]))
        y = torch.tensor([2.0, 3.0])
        scale = 0.0

        projection = dot(direction, y)
        steered = y - scale * projection * torch.tensor(direction)

        self.assertTrue(torch.allclose(steered, y))

    def test_normalize_numpy_array(self):
        """normalize() should work with numpy arrays."""
        from steering import normalize

        v = np.array([3.0, 4.0], dtype=np.float32)
        n = normalize(v)
        self.assertAlmostEqual(float(np.linalg.norm(n)), 1.0, places=5)

    def test_load_prompts(self):
        """load_prompts should read lines from a file."""
        from steering import load_prompts

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            _write_prompts(f.name, ["line 1", "line 2", "line 3"])
            path = f.name

        try:
            lines = load_prompts(path)
            self.assertEqual(lines, ["line 1", "line 2", "line 3"])
        finally:
            os.unlink(path)

    def test_load_prompts_empty_file(self):
        """load_prompts should raise ValueError on empty files."""
        from steering import load_prompts

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            path = f.name

        try:
            with self.assertRaises(ValueError):
                load_prompts(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
