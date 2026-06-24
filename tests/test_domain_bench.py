#!/usr/bin/env python3
"""Tests for scripts/domain_bench.py — Domain-Specific Capability Benchmark.

These tests are hermetic: they mock HuggingFace models, tokenizers, and file I/O
to avoid GPU requirements.

Run with:
    python -m pytest tests/test_domain_bench.py -v
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

# Mock device_utils BEFORE importing domain_bench
_device_utils_mock = MagicMock()
_device_utils_mock.is_cuda.return_value = False
_device_utils_mock.print_hardware_banner.return_value = "cpu"
sys.modules["device_utils"] = _device_utils_mock

import domain_bench as db  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _make_mock_model_and_tokenizer():
    """Create mock model and tokenizer for GPU-free testing."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 0
    tokenizer.pad_token = "<|endoftext|>"

    def _apply_chat_template(messages, **kwargs):
        return "mock chat template output"

    tokenizer.apply_chat_template = _apply_chat_template

    def _tokenize(text, **kwargs):
        mock_inputs = MagicMock()
        mock_inputs.shape = (1, 10)
        mock_inputs.to.return_value = mock_inputs
        return {"input_ids": mock_inputs}

    tokenizer.__call__ = _tokenize

    def _decode(ids, **kwargs):
        return "mock generated answer"

    tokenizer.decode = _decode

    model = MagicMock()
    model.device = "cpu"

    def _generate(**kwargs):
        input_ids = kwargs.get("input_ids", MagicMock())
        input_len = input_ids.shape[1] if hasattr(input_ids, "shape") else 10
        mock_ids = MagicMock()
        mock_ids.shape = (1, input_len + 5)
        mock_ids.__getitem__ = lambda s, i: MagicMock()
        return mock_ids

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
            db.parse_args([])

    def test_minimal_args(self):
        """Minimal valid args should parse without error."""
        args = db.parse_args(
            [
                "--base-model",
                "Qwen/Qwen2.5-7B-Instruct",
                "--questions",
                "questions.jsonl",
                "--output",
                "report.json",
            ]
        )
        self.assertEqual(args.base_model, "Qwen/Qwen2.5-7B-Instruct")
        self.assertIsNone(args.adapter)
        self.assertEqual(args.max_new_tokens, 256)
        self.assertEqual(args.seed, 42)
        self.assertIsNone(args.categories)

    def test_all_args(self):
        """All args should parse correctly."""
        args = db.parse_args(
            [
                "--base-model",
                "huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated",
                "--adapter",
                "models/attacklm-single",
                "--questions",
                "data/bench/questions.jsonl",
                "--output",
                "evals/domain_bench.json",
                "--max-new-tokens",
                "128",
                "--seed",
                "99",
                "--compute-dtype",
                "bf16",
                "--categories",
                "mitre_technique",
                "metasploit_command",
            ]
        )
        self.assertEqual(args.adapter, "models/attacklm-single")
        self.assertEqual(args.max_new_tokens, 128)
        self.assertEqual(args.seed, 99)
        self.assertEqual(args.compute_dtype, "bf16")
        self.assertEqual(args.categories, ["mitre_technique", "metasploit_command"])


# ---------------------------------------------------------------------------
# Tests: Grading functions
# ---------------------------------------------------------------------------


class TestGradeMitre(unittest.TestCase):
    """Verify MITRE technique grading."""

    def test_exact_match(self):
        """Exact MITRE ID match should pass."""
        self.assertTrue(db.grade_mitre("T1569.002", {"answer": "T1569.002"}))

    def test_match_with_surrounding_text(self):
        """MITRE ID in surrounding text should pass."""
        self.assertTrue(
            db.grade_mitre(
                "The technique used is T1569.002 which is a service execution technique.",
                {"answer": "T1569.002"},
            )
        )

    def test_no_match(self):
        """No MITRE ID should fail."""
        self.assertFalse(
            db.grade_mitre("This is a generic answer.", {"answer": "T1569.002"})
        )

    def test_acceptable_list(self):
        """Any ID in acceptable list should pass."""
        self.assertTrue(
            db.grade_mitre(
                "T1059",
                {"answer": "T1059.001", "acceptable": ["T1059", "T1059.001"]},
            )
        )

    def test_wrong_id(self):
        """Wrong MITRE ID should fail."""
        self.assertFalse(db.grade_mitre("T1569.001", {"answer": "T1569.002"}))

    def test_invalid_format(self):
        """Invalid MITRE ID format should fail."""
        self.assertFalse(db.grade_mitre("Not a technique", {"answer": "T1569.002"}))


class TestGradeMetasploit(unittest.TestCase):
    """Verify Metasploit command grading."""

    def test_module_and_options_present(self):
        """Module name and all required options should pass."""
        self.assertTrue(
            db.grade_metasploit(
                "use exploit/multi/handler\nset PAYLOAD windows/meterpreter/reverse_tcp\nset LHOST 10.0.0.1",
                {
                    "module": "exploit/multi/handler",
                    "required_options": ["PAYLOAD", "LHOST"],
                },
            )
        )

    def test_module_missing(self):
        """Missing module name should fail."""
        self.assertFalse(
            db.grade_metasploit(
                "set PAYLOAD windows/meterpreter/reverse_tcp",
                {
                    "module": "exploit/multi/handler",
                    "required_options": ["PAYLOAD"],
                },
            )
        )

    def test_option_missing(self):
        """Missing required option should fail."""
        self.assertFalse(
            db.grade_metasploit(
                "use exploit/multi/handler",
                {
                    "module": "exploit/multi/handler",
                    "required_options": ["PAYLOAD", "LHOST"],
                },
            )
        )

    def test_case_insensitive(self):
        """Module name should be case-insensitive."""
        self.assertTrue(
            db.grade_metasploit(
                "USE EXPLOIT/MULTI/HANDLER",
                {"module": "exploit/multi/handler", "required_options": []},
            )
        )

    def test_no_required_options(self):
        """No required options should pass if module present."""
        self.assertTrue(
            db.grade_metasploit(
                "use exploit/multi/handler",
                {"module": "exploit/multi/handler", "required_options": []},
            )
        )


class TestGradePromptInjection(unittest.TestCase):
    """Verify prompt injection grading."""

    def test_malicious_detected(self):
        """Malicious prompt should be detected."""
        self.assertTrue(
            db.grade_prompt_injection(
                "This is a malicious injection attack",
                {"answer": "malicious"},
            )
        )

    def test_malicious_not_detected(self):
        """Malicious prompt without keywords should fail."""
        self.assertFalse(
            db.grade_prompt_injection(
                "This is a normal question",
                {"answer": "malicious"},
            )
        )

    def test_benign_detected(self):
        """Benign prompt should be detected."""
        self.assertTrue(
            db.grade_prompt_injection(
                "This is a benign and safe request",
                {"answer": "benign"},
            )
        )

    def test_benign_not_detected(self):
        """Benign prompt without keywords should fail."""
        self.assertFalse(
            db.grade_prompt_injection(
                "This is a malicious attack",
                {"answer": "benign"},
            )
        )

    def test_jailbreak_keyword(self):
        """Jailbreak keyword should count as malicious."""
        self.assertTrue(
            db.grade_prompt_injection(
                "This is a jailbreak attempt",
                {"answer": "malicious"},
            )
        )


class TestGradePhishing(unittest.TestCase):
    """Verify phishing email grading."""

    def test_required_keywords_present(self):
        """All required keywords present should pass."""
        self.assertTrue(
            db.grade_phishing(
                "Dear user, please click the urgent link to verify your account.",
                {
                    "required_keywords": ["urgent", "click", "verify"],
                    "forbidden_keywords": [],
                },
            )
        )

    def test_required_keyword_missing(self):
        """Missing required keyword should fail."""
        self.assertFalse(
            db.grade_phishing(
                "Hello user.",
                {
                    "required_keywords": ["urgent", "click"],
                    "forbidden_keywords": [],
                },
            )
        )

    def test_forbidden_keyword_present(self):
        """Forbidden keyword present should fail."""
        self.assertFalse(
            db.grade_phishing(
                "Dear user, please click here for a legitimate offer.",
                {
                    "required_keywords": ["click"],
                    "forbidden_keywords": ["legitimate"],
                },
            )
        )

    def test_case_insensitive(self):
        """Keywords should be case-insensitive."""
        self.assertTrue(
            db.grade_phishing(
                "URGENT: CLICK HERE",
                {
                    "required_keywords": ["urgent", "click"],
                    "forbidden_keywords": [],
                },
            )
        )

    def test_no_required_or_forbidden(self):
        """No required or forbidden keywords should pass."""
        self.assertTrue(
            db.grade_phishing(
                "Any text at all",
                {"required_keywords": [], "forbidden_keywords": []},
            )
        )


class TestGradeOrchestrator(unittest.TestCase):
    """Verify orchestrator routing grading."""

    def test_exact_agent_name(self):
        """Exact agent name should pass."""
        self.assertTrue(
            db.grade_orchestrator(
                "re-coder",
                {"answer": "re-coder"},
            )
        )

    def test_agent_name_in_sentence(self):
        """Agent name in a sentence should pass."""
        self.assertTrue(
            db.grade_orchestrator(
                "I would route this to re-architect for analysis",
                {"answer": "re-architect"},
            )
        )

    def test_wrong_agent(self):
        """Wrong agent name should fail."""
        self.assertFalse(
            db.grade_orchestrator(
                "re-coder",
                {"answer": "re-architect"},
            )
        )

    def test_case_insensitive(self):
        """Agent name should be case-insensitive."""
        self.assertTrue(
            db.grade_orchestrator(
                "RE-CODER",
                {"answer": "re-coder"},
            )
        )


# ---------------------------------------------------------------------------
# Tests: load_questions
# ---------------------------------------------------------------------------


class TestLoadQuestions(unittest.TestCase):
    """Verify question loading from JSONL."""

    def test_load_all_questions(self):
        """All questions should be loaded when no category filter."""
        records = [
            {
                "question_id": "q1",
                "category": "mitre_technique",
                "messages": [],
                "ground_truth": {},
            },
            {
                "question_id": "q2",
                "category": "metasploit_command",
                "messages": [],
                "ground_truth": {},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "questions.jsonl"
            _write_jsonl(path, records)
            questions = db.load_questions(str(path), None)
            self.assertEqual(len(questions), 2)

    def test_filter_by_category(self):
        """Category filter should keep only matching questions."""
        records = [
            {
                "question_id": "q1",
                "category": "mitre_technique",
                "messages": [],
                "ground_truth": {},
            },
            {
                "question_id": "q2",
                "category": "metasploit_command",
                "messages": [],
                "ground_truth": {},
            },
            {
                "question_id": "q3",
                "category": "phishing",
                "messages": [],
                "ground_truth": {},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "questions.jsonl"
            _write_jsonl(path, records)
            questions = db.load_questions(str(path), ["mitre_technique", "phishing"])
            self.assertEqual(len(questions), 2)
            self.assertEqual(questions[0]["question_id"], "q1")
            self.assertEqual(questions[1]["question_id"], "q3")

    def test_nonexistent_file_exits(self):
        """Non-existent file should call sys.exit(1)."""
        with self.assertRaises(SystemExit):
            db.load_questions("/nonexistent.jsonl", None)

    def test_no_questions_after_filter_exits(self):
        """No questions after filtering should call sys.exit(1)."""
        records = [
            {
                "question_id": "q1",
                "category": "mitre_technique",
                "messages": [],
                "ground_truth": {},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "questions.jsonl"
            _write_jsonl(path, records)
            with self.assertRaises(SystemExit):
                db.load_questions(str(path), ["nonexistent_category"])


# ---------------------------------------------------------------------------
# Tests: build_report
# ---------------------------------------------------------------------------


class TestBuildReport(unittest.TestCase):
    """Verify report building."""

    def test_report_has_all_keys(self):
        """Report should have expected top-level keys."""
        results = [
            {
                "question_id": "q1",
                "category": "mitre_technique",
                "pass": True,
                "generated": "T1569.002",
                "ground_truth": "T1569.002",
                "tokens_generated": 10,
            },
            {
                "question_id": "q2",
                "category": "metasploit_command",
                "pass": False,
                "generated": "wrong",
                "ground_truth": "exploit",
                "tokens_generated": 5,
            },
        ]
        report = db.build_report(
            results, "Qwen/Qwen2.5-7B-Instruct", None, 256, 42, "fp32"
        )

        self.assertIn("metadata", report)
        self.assertIn("summary", report)
        self.assertIn("results", report)
        self.assertEqual(report["metadata"]["model"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertIsNone(report["metadata"]["adapter"])

        summary = report["summary"]
        self.assertIn("overall_score", summary)
        self.assertIn("overall_correct", summary)
        self.assertIn("overall_total", summary)
        self.assertIn("by_category", summary)
        self.assertEqual(summary["overall_correct"], 1)
        self.assertEqual(summary["overall_total"], 2)
        self.assertEqual(summary["overall_score"], 0.5)

    def test_report_empty_results(self):
        """Empty results should produce zero scores."""
        report = db.build_report([], "model", None, 256, 42, "fp32")
        self.assertEqual(report["summary"]["overall_score"], 0.0)
        self.assertEqual(report["summary"]["overall_total"], 0)

    def test_report_category_breakdown(self):
        """Category breakdown should be correct."""
        results = [
            {
                "question_id": "q1",
                "category": "mitre_technique",
                "pass": True,
                "generated": "a",
                "ground_truth": "a",
                "tokens_generated": 10,
            },
            {
                "question_id": "q2",
                "category": "mitre_technique",
                "pass": False,
                "generated": "b",
                "ground_truth": "b",
                "tokens_generated": 5,
            },
            {
                "question_id": "q3",
                "category": "phishing",
                "pass": True,
                "generated": "c",
                "ground_truth": "c",
                "tokens_generated": 8,
            },
        ]
        report = db.build_report(results, "model", None, 256, 42, "fp32")
        by_cat = report["summary"]["by_category"]
        self.assertIn("mitre_technique", by_cat)
        self.assertIn("phishing", by_cat)
        self.assertEqual(by_cat["mitre_technique"]["correct"], 1)
        self.assertEqual(by_cat["mitre_technique"]["total"], 2)
        self.assertEqual(by_cat["phishing"]["correct"], 1)
        self.assertEqual(by_cat["phishing"]["total"], 1)

    def test_report_serializes_to_json(self):
        """Report should be JSON-serializable."""
        results = [
            {
                "question_id": "q1",
                "category": "mitre_technique",
                "pass": True,
                "generated": "a",
                "ground_truth": "a",
                "tokens_generated": 10,
            },
        ]
        report = db.build_report(results, "model", "adapter/path", 256, 42, "bf16")
        json_str = json.dumps(report)
        self.assertIsInstance(json_str, str)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["metadata"]["adapter"], "adapter/path")


# ---------------------------------------------------------------------------
# Tests: Integration (mocked model, real files)
# ---------------------------------------------------------------------------


class TestIntegration(unittest.TestCase):
    """End-to-end test with mocked model and real temporary files."""

    @patch("domain_bench.load_model_and_tokenizer")
    def test_main_creates_report(self, mock_load):
        """main() should create a valid JSON report."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        mock_load.return_value = (model, tokenizer)

        questions = [
            {
                "question_id": "q1",
                "category": "mitre_technique",
                "messages": [{"role": "user", "content": "What technique is this?"}],
                "ground_truth": {"answer": "T1569.002"},
            },
            {
                "question_id": "q2",
                "category": "metasploit_command",
                "messages": [{"role": "user", "content": "Generate a command"}],
                "ground_truth": {
                    "module": "exploit/multi/handler",
                    "required_options": [],
                },
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            questions_path = tmpdir / "questions.jsonl"
            _write_jsonl(questions_path, questions)
            output_path = tmpdir / "report.json"

            rc = db.main(
                [
                    "--base-model",
                    "Qwen/Qwen2.5-7B-Instruct",
                    "--questions",
                    str(questions_path),
                    "--output",
                    str(output_path),
                    "--max-new-tokens",
                    "32",
                ]
            )

            self.assertEqual(rc, 0)
            self.assertTrue(output_path.exists())

            with open(output_path) as f:
                report = json.load(f)
            self.assertIn("summary", report)
            self.assertIn("results", report)
            self.assertEqual(len(report["results"]), 2)

    @patch("domain_bench.load_model_and_tokenizer")
    def test_main_with_category_filter(self, mock_load):
        """Category filter should only evaluate matching questions."""
        model, tokenizer = _make_mock_model_and_tokenizer()
        mock_load.return_value = (model, tokenizer)

        questions = [
            {
                "question_id": "q1",
                "category": "mitre_technique",
                "messages": [{"role": "user", "content": "Q1"}],
                "ground_truth": {"answer": "T1569.002"},
            },
            {
                "question_id": "q2",
                "category": "phishing",
                "messages": [{"role": "user", "content": "Q2"}],
                "ground_truth": {
                    "required_keywords": ["urgent"],
                    "forbidden_keywords": [],
                },
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            questions_path = tmpdir / "questions.jsonl"
            _write_jsonl(questions_path, questions)
            output_path = tmpdir / "report.json"

            rc = db.main(
                [
                    "--base-model",
                    "Qwen/Qwen2.5-7B-Instruct",
                    "--questions",
                    str(questions_path),
                    "--output",
                    str(output_path),
                    "--categories",
                    "mitre_technique",
                ]
            )

            self.assertEqual(rc, 0)
            with open(output_path) as f:
                report = json.load(f)
            self.assertEqual(len(report["results"]), 1)
            self.assertEqual(report["results"][0]["question_id"], "q1")

    @patch("domain_bench.load_model_and_tokenizer")
    def test_main_model_load_failure(self, mock_load):
        """Model load failure should return exit code 1."""
        mock_load.side_effect = Exception("Model not found")

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            questions_path = tmpdir / "questions.jsonl"
            _write_jsonl(
                questions_path,
                [
                    {
                        "question_id": "q1",
                        "category": "mitre_technique",
                        "messages": [],
                        "ground_truth": {"answer": "T1569.002"},
                    },
                ],
            )
            output_path = tmpdir / "report.json"

            rc = db.main(
                [
                    "--base-model",
                    "nonexistent/model",
                    "--questions",
                    str(questions_path),
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
