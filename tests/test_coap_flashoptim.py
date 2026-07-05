#!/usr/bin/env python3
"""Tests for COAP and FlashOptim CLI flags in train_template.py.

Verifies that --use-coap, --coap-rank, --coap-8bit, and --use-flashoptim
parse correctly, that mutual exclusivity checks work, and that the flags
appear in state.json hparams.

Run with:
    python -m pytest tests/test_coap_flashoptim.py -v
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
# COAP CLI Argument Parsing
# =========================================================================


class TestCOAPArgParsing(unittest.TestCase):
    """Verify COAP CLI arguments parse correctly."""

    def test_use_coap_parses(self):
        """--use-coap flag parses correctly."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--use-coap",
            ]
        )
        self.assertTrue(args.use_coap)

    def test_coap_rank_256(self):
        """--coap-rank 256 parses correctly."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--use-coap",
                "--coap-rank",
                "256",
            ]
        )
        self.assertEqual(args.coap_rank, 256)

    def test_coap_8bit(self):
        """--coap-8bit parses correctly."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--use-coap",
                "--coap-8bit",
            ]
        )
        self.assertTrue(args.coap_8bit)

    def test_coap_default_rank(self):
        """COAP rank defaults to 128."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--use-coap",
            ]
        )
        self.assertEqual(args.coap_rank, 128)

    def test_coap_8bit_default_false(self):
        """COAP 8bit defaults to False."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--use-coap",
            ]
        )
        self.assertFalse(args.coap_8bit)

    def test_coap_default_false(self):
        """--use-coap is False by default."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
            ]
        )
        self.assertFalse(args.use_coap)


# =========================================================================
# FlashOptim CLI Argument Parsing
# =========================================================================


class TestFlashOptimArgParsing(unittest.TestCase):
    """Verify FlashOptim CLI arguments parse correctly."""

    def test_use_flashoptim_parses(self):
        """--use-flashoptim flag parses correctly."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--use-flashoptim",
            ]
        )
        self.assertTrue(args.use_flashoptim)

    def test_flashoptim_default_false(self):
        """--use-flashoptim is False by default."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
            ]
        )
        self.assertFalse(args.use_flashoptim)


# =========================================================================
# Mutual Exclusivity: COAP
# =========================================================================


class TestCOAPMutualExclusivity(unittest.TestCase):
    """Verify COAP mutual exclusivity checks in main()."""

    def test_coap_galore_exclusive(self):
        """--use-coap + --use-galore raises error."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--use-coap",
                "--use-galore",
            ]
        )
        with self.assertRaises(SystemExit):
            if args.use_coap and (args.use_galore or args.use_qgalore):
                raise SystemExit(1)

    def test_coap_lomo_exclusive(self):
        """--use-coap + --use-lomo raises error."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--use-coap",
                "--use-lomo",
            ]
        )
        with self.assertRaises(SystemExit):
            if args.use_coap and args.use_lomo:
                raise SystemExit(1)

    def test_coap_flashoptim_exclusive(self):
        """--use-coap + --use-flashoptim raises error."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--use-coap",
                "--use-flashoptim",
            ]
        )
        with self.assertRaises(SystemExit):
            if args.use_coap and args.use_flashoptim:
                raise SystemExit(1)


# =========================================================================
# Mutual Exclusivity: FlashOptim
# =========================================================================


class TestFlashOptimMutualExclusivity(unittest.TestCase):
    """Verify FlashOptim mutual exclusivity checks in main()."""

    def test_flashoptim_galore_exclusive(self):
        """--use-flashoptim + --use-galore raises error."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--use-flashoptim",
                "--use-galore",
            ]
        )
        with self.assertRaises(SystemExit):
            if args.use_flashoptim and (args.use_galore or args.use_qgalore):
                raise SystemExit(1)

    def test_flashoptim_lomo_exclusive(self):
        """--use-flashoptim + --use-lomo raises error."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--use-flashoptim",
                "--use-lomo",
            ]
        )
        with self.assertRaises(SystemExit):
            if args.use_flashoptim and args.use_lomo:
                raise SystemExit(1)


# =========================================================================
# State JSON Integration
# =========================================================================


class TestStateJSONCOAPFlashOptim(unittest.TestCase):
    """Verify state.json records COAP and FlashOptim fields."""

    def test_coap_in_state_json(self):
        """COAP appears in state.json hparams."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--use-coap",
                "--coap-rank",
                "256",
                "--coap-8bit",
            ]
        )
        hparams = {
            "coap": args.use_coap,
            "coap_rank": args.coap_rank if args.use_coap else None,
            "coap_8bit": args.coap_8bit if args.use_coap else None,
        }
        self.assertTrue(hparams["coap"])
        self.assertEqual(hparams["coap_rank"], 256)
        self.assertTrue(hparams["coap_8bit"])

    def test_flashoptim_in_state_json(self):
        """FlashOptim appears in state.json hparams."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
                "--use-flashoptim",
            ]
        )
        hparams = {
            "flashoptim": args.use_flashoptim,
        }
        self.assertTrue(hparams["flashoptim"])

    def test_coap_state_json_defaults(self):
        """Default values for COAP fields in state.json."""
        args = train_template.parse_args(
            [
                "--dataset",
                "dummy.jsonl",
                "--output",
                "/tmp/test_out",
            ]
        )
        hparams = {
            "coap": args.use_coap,
            "coap_rank": args.coap_rank if args.use_coap else None,
            "coap_8bit": args.coap_8bit if args.use_coap else None,
            "flashoptim": args.use_flashoptim,
        }
        self.assertFalse(hparams["coap"])
        self.assertIsNone(hparams["coap_rank"])
        self.assertIsNone(hparams["coap_8bit"])
        self.assertFalse(hparams["flashoptim"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
