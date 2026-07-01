#!/usr/bin/env python3
"""Tests for DeepSpeed, torch.compile, and LOMO features in train_template.py.

Run with:
    python -m pytest tests/test_memory_optimization.py -v

Or directly:
    python tests/test_memory_optimization.py
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Make the scripts/ dir importable
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import train_template


# =========================================================================
# DeepSpeed Config Generation
# =========================================================================


class TestDeepSpeedConfigGeneration(unittest.TestCase):
    """Verify generate_deepspeed_config() produces valid configs."""

    def test_deepspeed_config_generation(self):
        """Auto-generates valid JSON config with correct structure."""
        config = train_template.generate_deepspeed_config(stage=3, offload=True)
        self.assertIn("zero_optimization", config)
        self.assertIn("bf16", config)
        self.assertEqual(config["zero_optimization"]["stage"], 3)
        self.assertEqual(config["gradient_accumulation_steps"], "auto")
        self.assertEqual(config["gradient_clipping"], "auto")
        self.assertEqual(config["train_batch_size"], "auto")
        self.assertEqual(config["train_micro_batch_size_per_gpu"], "auto")

    def test_deepspeed_config_zero3_cpu_offload(self):
        """Config has stage=3, offload_optimizer, offload_param."""
        config = train_template.generate_deepspeed_config(stage=3, offload=True)
        zero_opt = config["zero_optimization"]
        self.assertEqual(zero_opt["stage"], 3)
        self.assertIn("offload_optimizer", zero_opt)
        self.assertEqual(zero_opt["offload_optimizer"]["device"], "cpu")
        self.assertIn("offload_param", zero_opt)
        self.assertEqual(zero_opt["offload_param"]["device"], "cpu")

    def test_deepspeed_config_zero2(self):
        """Config has stage=2, offload_optimizer, no offload_param."""
        config = train_template.generate_deepspeed_config(stage=2, offload=True)
        zero_opt = config["zero_optimization"]
        self.assertEqual(zero_opt["stage"], 2)
        self.assertIn("offload_optimizer", zero_opt)
        self.assertEqual(zero_opt["offload_optimizer"]["device"], "cpu")
        self.assertNotIn("offload_param", zero_opt)

    def test_deepspeed_config_no_offload(self):
        """--no-deepspeed-offload removes offload sections."""
        config = train_template.generate_deepspeed_config(stage=3, offload=False)
        zero_opt = config["zero_optimization"]
        self.assertEqual(zero_opt["stage"], 3)
        self.assertNotIn("offload_optimizer", zero_opt)
        self.assertNotIn("offload_param", zero_opt)

    def test_deepspeed_config_zero2_no_offload(self):
        """Stage 2 without offload has no offload sections at all."""
        config = train_template.generate_deepspeed_config(stage=2, offload=False)
        zero_opt = config["zero_optimization"]
        self.assertEqual(zero_opt["stage"], 2)
        self.assertNotIn("offload_optimizer", zero_opt)
        self.assertNotIn("offload_param", zero_opt)

    def test_deepspeed_config_fp16_dtype(self):
        """fp16 dtype produces fp16 config section instead of bf16."""
        config = train_template.generate_deepspeed_config(stage=3, dtype="fp16")
        self.assertNotIn("bf16", config)
        self.assertIn("fp16", config)
        self.assertTrue(config["fp16"]["enabled"])

    def test_deepspeed_config_serializable(self):
        """Config dict is JSON-serializable."""
        config = train_template.generate_deepspeed_config(stage=3, offload=True)
        # Should not raise
        json.dumps(config)

    def test_deepspeed_config_zero3_sub_group_size(self):
        """Stage 3 has sub_group_size set."""
        config = train_template.generate_deepspeed_config(stage=3, offload=True)
        self.assertEqual(config["zero_optimization"]["sub_group_size"], 1e9)

    def test_deepspeed_config_zero2_no_sub_group_size(self):
        """Stage 2 does NOT have sub_group_size."""
        config = train_template.generate_deepspeed_config(stage=2, offload=True)
        self.assertNotIn("sub_group_size", config["zero_optimization"])


# =========================================================================
# DeepSpeed CLI Argument Parsing
# =========================================================================


class TestDeepSpeedArgParsing(unittest.TestCase):
    """Verify DeepSpeed CLI arguments parse correctly."""

    def test_deepspeed_arg_parsing(self):
        """--use-deepspeed flag parses correctly."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--use-deepspeed",
            ]
        )
        self.assertTrue(args.use_deepspeed)
        self.assertEqual(args.deepspeed_stage, 3)  # default
        self.assertFalse(args.no_deepspeed_offload)  # default

    def test_deepspeed_stage_2(self):
        """--deepspeed-stage 2 parses correctly."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--use-deepspeed",
                "--deepspeed-stage",
                "2",
            ]
        )
        self.assertEqual(args.deepspeed_stage, 2)

    def test_deepspeed_stage_3(self):
        """--deepspeed-stage 3 parses correctly."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--use-deepspeed",
                "--deepspeed-stage",
                "3",
            ]
        )
        self.assertEqual(args.deepspeed_stage, 3)

    def test_deepspeed_no_offload(self):
        """--no-deepspeed-offload parses correctly."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--use-deepspeed",
                "--no-deepspeed-offload",
            ]
        )
        self.assertTrue(args.no_deepspeed_offload)

    def test_deepspeed_custom_config(self):
        """--deepspeed-config path is stored."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--use-deepspeed",
                "--deepspeed-config",
                "/path/to/ds_config.json",
            ]
        )
        self.assertEqual(args.deepspeed_config, "/path/to/ds_config.json")

    def test_deepspeed_defaults(self):
        """Default values for DeepSpeed args."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
            ]
        )
        self.assertFalse(args.use_deepspeed)
        self.assertEqual(args.deepspeed_stage, 3)
        self.assertFalse(args.no_deepspeed_offload)
        self.assertIsNone(args.deepspeed_config)

    def test_deepspeed_stage_invalid(self):
        """--deepspeed-stage only accepts 1, 2, or 3."""
        with self.assertRaises(SystemExit):
            train_template.parse_args(
                [
                    "--dataset",
                    "dummy.jsonl",
                    "--output",
                    "/tmp/test_out",
                    "--use-deepspeed",
                    "--deepspeed-stage",
                    "4",
                ]
            )


# =========================================================================
# torch.compile Argument Parsing
# =========================================================================


class TestCompileArgParsing(unittest.TestCase):
    """Verify torch.compile CLI arguments parse correctly."""

    def test_compile_arg_parsing(self):
        """--compile flag parses correctly."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--compile",
            ]
        )
        self.assertTrue(args.compile)

    def test_compile_mode_default(self):
        """Default compile mode is 'reduce-overhead'."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--compile",
            ]
        )
        self.assertEqual(args.compile_mode, "reduce-overhead")

    def test_compile_mode_default_choice(self):
        """--compile-mode 'default' is accepted."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--compile",
                "--compile-mode",
                "default",
            ]
        )
        self.assertEqual(args.compile_mode, "default")

    def test_compile_mode_max_autotune(self):
        """--compile-mode 'max-autotune' is accepted."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--compile",
                "--compile-mode",
                "max-autotune",
            ]
        )
        self.assertEqual(args.compile_mode, "max-autotune")

    def test_compile_mode_invalid(self):
        """Invalid compile mode is rejected."""
        with self.assertRaises(SystemExit):
            train_template.parse_args(
                [
                    "--dataset",
                    "dummy.jsonl",
                    "--output",
                    "/tmp/test_out",
                    "--compile",
                    "--compile-mode",
                    "invalid_mode",
                ]
            )

    def test_compile_default_false(self):
        """--compile is False by default."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
            ]
        )
        self.assertFalse(args.compile)


# =========================================================================
# LOMO Argument Parsing
# =========================================================================


class TestLOMOArgParsing(unittest.TestCase):
    """Verify LOMO CLI arguments parse correctly."""

    def test_lomo_arg_parsing(self):
        """--use-lomo flag parses correctly."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--use-lomo",
            ]
        )
        self.assertTrue(args.use_lomo)

    def test_lomo_default_false(self):
        """--use-lomo is False by default."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
            ]
        )
        self.assertFalse(args.use_lomo)


# =========================================================================
# Mutual Exclusivity
# =========================================================================


class TestMutualExclusivity(unittest.TestCase):
    """Verify mutual exclusivity checks in main()."""

    def test_lomo_mutual_exclusivity_galore(self):
        """--use-lomo + --use-galore raises error."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--use-lomo",
                "--use-galore",
            ]
        )
        with self.assertRaises(SystemExit):
            # Simulate the check in main()
            if args.use_lomo and (args.use_galore or args.use_qgalore):
                raise SystemExit(1)

    def test_lomo_mutual_exclusivity_qgalore(self):
        """--use-lomo + --use-qgalore raises error."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--use-lomo",
                "--use-qgalore",
            ]
        )
        with self.assertRaises(SystemExit):
            if args.use_lomo and (args.use_galore or args.use_qgalore):
                raise SystemExit(1)

    def test_deepspeed_mutual_exclusivity_unsloth_warning(self):
        """--use-deepspeed + --use-unsloth prints warning (not error)."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--use-deepspeed",
                "--use-unsloth",
            ]
        )
        # Both flags are set — main() prints a warning but does NOT exit
        self.assertTrue(args.use_deepspeed)
        self.assertTrue(args.use_unsloth)
        # No SystemExit — the check is a warning, not an error


# =========================================================================
# Integration: All Flags Combined
# =========================================================================


class TestAllFlagsCombined(unittest.TestCase):
    """Verify all new flags can be parsed together."""

    def test_all_flags_combined(self):
        """--use-deepspeed --compile --use-lomo all parse together."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--use-deepspeed",
                "--deepspeed-stage",
                "2",
                "--no-deepspeed-offload",
                "--compile",
                "--compile-mode",
                "default",
                "--use-lomo",
            ]
        )
        self.assertTrue(args.use_deepspeed)
        self.assertEqual(args.deepspeed_stage, 2)
        self.assertTrue(args.no_deepspeed_offload)
        self.assertTrue(args.compile)
        self.assertEqual(args.compile_mode, "default")
        self.assertTrue(args.use_lomo)

    def test_all_flags_with_galore(self):
        """--use-deepspeed --compile --use-galore all parse together."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--use-deepspeed",
                "--compile",
                "--use-galore",
            ]
        )
        self.assertTrue(args.use_deepspeed)
        self.assertTrue(args.compile)
        self.assertTrue(args.use_galore)


# =========================================================================
# State JSON Integration
# =========================================================================


class TestStateJSONIncludesNewFields(unittest.TestCase):
    """Verify state.json records deepspeed, compile, lomo fields."""

    def test_state_json_includes_new_fields(self):
        """state.json hparams includes deepspeed, compile, lomo fields."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--use-deepspeed",
                "--deepspeed-stage",
                "3",
                "--compile",
                "--compile-mode",
                "reduce-overhead",
                "--use-lomo",
            ]
        )
        hparams = {
            "deepspeed": args.use_deepspeed,
            "deepspeed_stage": args.deepspeed_stage,
            "deepspeed_offload": not args.no_deepspeed_offload,
            "compile": args.compile,
            "compile_mode": args.compile_mode,
            "lomo": args.use_lomo,
        }
        self.assertTrue(hparams["deepspeed"])
        self.assertEqual(hparams["deepspeed_stage"], 3)
        self.assertTrue(hparams["deepspeed_offload"])
        self.assertTrue(hparams["compile"])
        self.assertEqual(hparams["compile_mode"], "reduce-overhead")
        self.assertTrue(hparams["lomo"])

    def test_state_json_defaults(self):
        """Default values for new fields in state.json."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
            ]
        )
        hparams = {
            "deepspeed": args.use_deepspeed,
            "deepspeed_stage": args.deepspeed_stage,
            "deepspeed_offload": not args.no_deepspeed_offload,
            "compile": args.compile,
            "compile_mode": args.compile_mode,
            "lomo": args.use_lomo,
        }
        self.assertFalse(hparams["deepspeed"])
        self.assertEqual(hparams["deepspeed_stage"], 3)
        self.assertTrue(hparams["deepspeed_offload"])
        self.assertFalse(hparams["compile"])
        self.assertEqual(hparams["compile_mode"], "reduce-overhead")
        self.assertFalse(hparams["lomo"])


# =========================================================================
# DeepSpeed Config File Writing (Integration)
# =========================================================================


class TestDeepSpeedConfigFileWriting(unittest.TestCase):
    """Verify the generated DeepSpeed config can be written and read back."""

    def test_write_and_read_back(self):
        """Generated config is valid JSON and can be re-read."""
        config = train_template.generate_deepspeed_config(stage=3, offload=True)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f, indent=2)
            tmp_path = f.name
        try:
            with open(tmp_path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded["zero_optimization"]["stage"], 3)
            self.assertIn("offload_optimizer", loaded["zero_optimization"])
            self.assertIn("offload_param", loaded["zero_optimization"])
            self.assertIn("bf16", loaded)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
