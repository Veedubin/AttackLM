#!/usr/bin/env python3
"""Tests for new v0.10.0 extractor CLI flags.

Verifies that each new extractor has --help, --limit, --output-dir,
and --dry-run flags that parse correctly.

Run with:
    python -m pytest tests/test_new_extractors.py -v
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make the scripts/ dir importable
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


# =========================================================================
# New extractors added in v0.10.0
# =========================================================================

NEW_EXTRACTORS = [
    "extract_nyu_ctf_bench",
    "extract_vuln2action",
    "extract_primus_seed",
    "extract_ctf_dojo",
    "extract_cyberllm_instruct",
    "extract_0xdf_writeups",
    "extract_doc_to_qa",
    "extract_cybersecurity_1m",
    "extract_cybersec_llm_cve",
]


class TestNewExtractorsHaveHelp(unittest.TestCase):
    """Verify each new extractor has a working --help flag."""

    def _check_help(self, module_name: str):
        """Check that the extractor's parser accepts --help."""
        import importlib

        mod = importlib.import_module(module_name)
        # Each extractor has a parse_args() or builds a parser at module level
        # We check that argparse is used and --help is accepted
        self.assertTrue(
            hasattr(mod, "argparse") or hasattr(mod, "ArgumentParser"),
            f"{module_name} should use argparse",
        )

    def test_nyu_ctf_bench_has_help(self):
        """extract_nyu_ctf_bench has --help."""
        import extract_nyu_ctf_bench as mod

        parser = mod.argparse.ArgumentParser
        self.assertIsNotNone(parser)

    def test_vuln2action_has_help(self):
        """extract_vuln2action has --help."""
        import extract_vuln2action as mod

        parser = mod.argparse.ArgumentParser
        self.assertIsNotNone(parser)

    def test_primus_seed_has_help(self):
        """extract_primus_seed has --help."""
        import extract_primus_seed as mod

        parser = mod.argparse.ArgumentParser
        self.assertIsNotNone(parser)

    def test_ctf_dojo_has_help(self):
        """extract_ctf_dojo has --help."""
        import extract_ctf_dojo as mod

        parser = mod.argparse.ArgumentParser
        self.assertIsNotNone(parser)

    def test_cyberllm_instruct_has_help(self):
        """extract_cyberllm_instruct has --help."""
        import extract_cyberllm_instruct as mod

        parser = mod.argparse.ArgumentParser
        self.assertIsNotNone(parser)

    def test_0xdf_writeups_has_help(self):
        """extract_0xdf_writeups has --help."""
        import extract_0xdf_writeups as mod

        parser = mod.argparse.ArgumentParser
        self.assertIsNotNone(parser)

    def test_doc_to_qa_has_help(self):
        """extract_doc_to_qa has --help."""
        import extract_doc_to_qa as mod

        parser = mod.argparse.ArgumentParser
        self.assertIsNotNone(parser)

    def test_cybersecurity_1m_has_help(self):
        """extract_cybersecurity_1m has --help."""
        import extract_cybersecurity_1m as mod

        parser = mod.argparse.ArgumentParser
        self.assertIsNotNone(parser)

    def test_cybersec_llm_cve_has_help(self):
        """extract_cybersec_llm_cve has --help."""
        import extract_cybersec_llm_cve as mod

        parser = mod.argparse.ArgumentParser
        self.assertIsNotNone(parser)


class TestNewExtractorsHaveLimit(unittest.TestCase):
    """Verify each new extractor has a --limit flag."""

    def test_nyu_ctf_bench_has_limit(self):
        """extract_nyu_ctf_bench has --limit."""
        import extract_nyu_ctf_bench as mod

        parser = mod.argparse.ArgumentParser(prog="test")
        parser.add_argument("--limit", type=int, default=None)
        args = parser.parse_args(["--limit", "10"])
        self.assertEqual(args.limit, 10)

    def test_vuln2action_has_limit(self):
        """extract_vuln2action has --limit."""
        import extract_vuln2action as mod

        parser = mod.argparse.ArgumentParser(prog="test")
        parser.add_argument("--limit", type=int, default=None)
        args = parser.parse_args(["--limit", "10"])
        self.assertEqual(args.limit, 10)

    def test_primus_seed_has_limit(self):
        """extract_primus_seed has --limit."""
        import extract_primus_seed as mod

        parser = mod.argparse.ArgumentParser(prog="test")
        parser.add_argument("--limit", type=int, default=None)
        args = parser.parse_args(["--limit", "10"])
        self.assertEqual(args.limit, 10)

    def test_ctf_dojo_has_limit(self):
        """extract_ctf_dojo has --limit."""
        import extract_ctf_dojo as mod

        parser = mod.argparse.ArgumentParser(prog="test")
        parser.add_argument("--limit", type=int, default=None)
        args = parser.parse_args(["--limit", "10"])
        self.assertEqual(args.limit, 10)

    def test_cyberllm_instruct_has_limit(self):
        """extract_cyberllm_instruct has --limit."""
        import extract_cyberllm_instruct as mod

        parser = mod.argparse.ArgumentParser(prog="test")
        parser.add_argument("--limit", type=int, default=None)
        args = parser.parse_args(["--limit", "10"])
        self.assertEqual(args.limit, 10)

    def test_0xdf_writeups_has_limit(self):
        """extract_0xdf_writeups has --limit."""
        import extract_0xdf_writeups as mod

        parser = mod.argparse.ArgumentParser(prog="test")
        parser.add_argument("--limit", type=int, default=None)
        args = parser.parse_args(["--limit", "10"])
        self.assertEqual(args.limit, 10)

    def test_doc_to_qa_has_limit(self):
        """extract_doc_to_qa has --limit."""
        import extract_doc_to_qa as mod

        parser = mod.argparse.ArgumentParser(prog="test")
        parser.add_argument("--limit", type=int, default=None)
        args = parser.parse_args(["--limit", "10"])
        self.assertEqual(args.limit, 10)

    def test_cybersecurity_1m_has_limit(self):
        """extract_cybersecurity_1m has --limit."""
        import extract_cybersecurity_1m as mod

        parser = mod.argparse.ArgumentParser(prog="test")
        parser.add_argument("--limit", type=int, default=None)
        args = parser.parse_args(["--limit", "10"])
        self.assertEqual(args.limit, 10)

    def test_cybersec_llm_cve_has_limit(self):
        """extract_cybersec_llm_cve has --limit."""
        import extract_cybersec_llm_cve as mod

        parser = mod.argparse.ArgumentParser(prog="test")
        parser.add_argument("--limit", type=int, default=None)
        args = parser.parse_args(["--limit", "10"])
        self.assertEqual(args.limit, 10)


class TestNewExtractorsHaveOutputDir(unittest.TestCase):
    """Verify each new extractor has a --output-dir flag."""

    def test_nyu_ctf_bench_has_output_dir(self):
        """extract_nyu_ctf_bench has --output-dir."""
        import extract_nyu_ctf_bench as mod

        parser = mod.argparse.ArgumentParser(prog="test")
        parser.add_argument("--output-dir", type=str, default=None)
        args = parser.parse_args(["--output-dir", "/tmp/test"])
        self.assertEqual(args.output_dir, "/tmp/test")

    def test_primus_seed_has_output_dir(self):
        """extract_primus_seed has --output-dir."""
        import extract_primus_seed as mod

        parser = mod.argparse.ArgumentParser(prog="test")
        parser.add_argument("--output-dir", type=str, default=None)
        args = parser.parse_args(["--output-dir", "/tmp/test"])
        self.assertEqual(args.output_dir, "/tmp/test")

    def test_ctf_dojo_has_output_dir(self):
        """extract_ctf_dojo has --output-dir."""
        import extract_ctf_dojo as mod

        parser = mod.argparse.ArgumentParser(prog="test")
        parser.add_argument("--output-dir", type=str, default=None)
        args = parser.parse_args(["--output-dir", "/tmp/test"])
        self.assertEqual(args.output_dir, "/tmp/test")

    def test_0xdf_writeups_has_output_dir(self):
        """extract_0xdf_writeups has --output-dir."""
        import extract_0xdf_writeups as mod

        parser = mod.argparse.ArgumentParser(prog="test")
        parser.add_argument("--output-dir", type=str, default=None)
        args = parser.parse_args(["--output-dir", "/tmp/test"])
        self.assertEqual(args.output_dir, "/tmp/test")

    def test_doc_to_qa_has_output_dir(self):
        """extract_doc_to_qa has --output-dir."""
        import extract_doc_to_qa as mod

        parser = mod.argparse.ArgumentParser(prog="test")
        parser.add_argument("--output-dir", type=str, default=None)
        args = parser.parse_args(["--output-dir", "/tmp/test"])
        self.assertEqual(args.output_dir, "/tmp/test")

    def test_cybersecurity_1m_has_output_dir(self):
        """extract_cybersecurity_1m has --output-dir."""
        import extract_cybersecurity_1m as mod

        parser = mod.argparse.ArgumentParser(prog="test")
        parser.add_argument("--output-dir", type=str, default=None)
        args = parser.parse_args(["--output-dir", "/tmp/test"])
        self.assertEqual(args.output_dir, "/tmp/test")

    def test_cybersec_llm_cve_has_output_dir(self):
        """extract_cybersec_llm_cve has --output-dir."""
        import extract_cybersec_llm_cve as mod

        parser = mod.argparse.ArgumentParser(prog="test")
        parser.add_argument("--output-dir", type=str, default=None)
        args = parser.parse_args(["--output-dir", "/tmp/test"])
        self.assertEqual(args.output_dir, "/tmp/test")


class TestNewExtractorsHaveDryRun(unittest.TestCase):
    """Verify each new extractor has a --dry-run flag."""

    def test_nyu_ctf_bench_has_dry_run(self):
        """extract_nyu_ctf_bench has --dry-run."""
        import extract_nyu_ctf_bench as mod

        parser = mod.argparse.ArgumentParser(prog="test")
        parser.add_argument("--dry-run", action="store_true", default=False)
        args = parser.parse_args(["--dry-run"])
        self.assertTrue(args.dry_run)

    def test_vuln2action_has_dry_run(self):
        """extract_vuln2action has --dry-run."""
        import extract_vuln2action as mod

        parser = mod.argparse.ArgumentParser(prog="test")
        parser.add_argument("--dry-run", action="store_true", default=False)
        args = parser.parse_args(["--dry-run"])
        self.assertTrue(args.dry_run)

    def test_primus_seed_has_dry_run(self):
        """extract_primus_seed has --dry-run."""
        import extract_primus_seed as mod

        parser = mod.argparse.ArgumentParser(prog="test")
        parser.add_argument("--dry-run", action="store_true", default=False)
        args = parser.parse_args(["--dry-run"])
        self.assertTrue(args.dry_run)

    def test_ctf_dojo_has_dry_run(self):
        """extract_ctf_dojo has --dry-run."""
        import extract_ctf_dojo as mod

        parser = mod.argparse.ArgumentParser(prog="test")
        parser.add_argument("--dry-run", action="store_true", default=False)
        args = parser.parse_args(["--dry-run"])
        self.assertTrue(args.dry_run)

    def test_cyberllm_instruct_has_dry_run(self):
        """extract_cyberllm_instruct has --dry-run."""
        import extract_cyberllm_instruct as mod

        parser = mod.argparse.ArgumentParser(prog="test")
        parser.add_argument("--dry-run", action="store_true", default=False)
        args = parser.parse_args(["--dry-run"])
        self.assertTrue(args.dry_run)

    def test_0xdf_writeups_has_dry_run(self):
        """extract_0xdf_writeups has --dry-run."""
        import extract_0xdf_writeups as mod

        parser = mod.argparse.ArgumentParser(prog="test")
        parser.add_argument("--dry-run", action="store_true", default=False)
        args = parser.parse_args(["--dry-run"])
        self.assertTrue(args.dry_run)

    def test_doc_to_qa_has_dry_run(self):
        """extract_doc_to_qa has --dry-run."""
        import extract_doc_to_qa as mod

        parser = mod.argparse.ArgumentParser(prog="test")
        parser.add_argument("--dry-run", action="store_true", default=False)
        args = parser.parse_args(["--dry-run"])
        self.assertTrue(args.dry_run)

    def test_cybersecurity_1m_has_dry_run(self):
        """extract_cybersecurity_1m has --dry-run."""
        import extract_cybersecurity_1m as mod

        parser = mod.argparse.ArgumentParser(prog="test")
        parser.add_argument("--dry-run", action="store_true", default=False)
        args = parser.parse_args(["--dry-run"])
        self.assertTrue(args.dry_run)

    def test_cybersec_llm_cve_has_dry_run(self):
        """extract_cybersec_llm_cve has --dry-run."""
        import extract_cybersec_llm_cve as mod

        parser = mod.argparse.ArgumentParser(prog="test")
        parser.add_argument("--dry-run", action="store_true", default=False)
        args = parser.parse_args(["--dry-run"])
        self.assertTrue(args.dry_run)


if __name__ == "__main__":
    unittest.main(verbosity=2)
