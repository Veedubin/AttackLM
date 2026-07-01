#!/usr/bin/env python3
"""Tests for scripts/evolve_pairs.py and scripts/filter_evolved.py.

Run with:
    python -m pytest tests/test_evolve_pairs.py -v

Or directly:
    python tests/test_evolve_pairs.py
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

import evolve_pairs
import filter_evolved


# =========================================================================
# evolve_pairs tests
# =========================================================================


class TestLoadRecords(unittest.TestCase):
    """load_records() loads JSONL files and returns valid records."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil

        shutil.rmtree(str(self.tmpdir), ignore_errors=True)

    def _write_jsonl(self, path: Path, records: list[dict]):
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    def test_loads_correct_count(self):
        """load_records returns the correct number of valid records."""
        records = [
            {
                "messages": [
                    {"role": "system", "content": "You are a red team specialist."},
                    {"role": "user", "content": "How do I use mimikatz?"},
                    {
                        "role": "assistant",
                        "content": "**T1003 — Credential Dumping**\n\nUse `sekurlsa::logonpasswords` to dump credentials from LSASS.",
                    },
                ],
                "source": "test-source",
            },
            {
                "messages": [
                    {"role": "system", "content": "You are a red team specialist."},
                    {"role": "user", "content": "How do I scan ports?"},
                    {
                        "role": "assistant",
                        "content": "**T1046 — Network Service Scanning**\n\nUse `nmap -sV -p- target` to scan all ports.",
                    },
                ],
                "source": "test-source",
            },
        ]
        jsonl_path = self.tmpdir / "data_test.jsonl"
        self._write_jsonl(jsonl_path, records)

        result = evolve_pairs.load_records([jsonl_path])
        self.assertEqual(len(result), 2)

    def test_skips_malformed_json(self):
        """Malformed JSON lines are skipped with a warning."""
        jsonl_path = self.tmpdir / "data_test.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write('{"messages": [{"role": "user", "content": "ok"}]}\n')
            f.write("not valid json\n")
            f.write('{"messages": [{"role": "user", "content": "ok2"}]}\n')

        result = evolve_pairs.load_records([jsonl_path])
        # The first record has < 2 messages, so it's skipped.
        # The second line is malformed JSON.
        # The third record also has < 2 messages.
        self.assertEqual(len(result), 0)

    def test_skips_records_with_fewer_than_two_messages(self):
        """Records with < 2 messages are skipped."""
        jsonl_path = self.tmpdir / "data_test.jsonl"
        self._write_jsonl(
            jsonl_path,
            [
                {
                    "messages": [{"role": "user", "content": "short"}],
                    "source": "test",
                },
            ],
        )
        result = evolve_pairs.load_records([jsonl_path])
        self.assertEqual(len(result), 0)

    def test_deduplicates_by_assistant_content_hash(self):
        """Duplicate assistant content is deduplicated."""
        jsonl_path = self.tmpdir / "data_test.jsonl"
        self._write_jsonl(
            jsonl_path,
            [
                {
                    "messages": [
                        {"role": "system", "content": "You are a red team specialist."},
                        {"role": "user", "content": "How do I use mimikatz?"},
                        {
                            "role": "assistant",
                            "content": "**T1003**\n\nUse `sekurlsa::logonpasswords`.",
                        },
                    ],
                    "source": "test",
                },
                {
                    "messages": [
                        {"role": "system", "content": "You are a red team specialist."},
                        {"role": "user", "content": "How do I use mimikatz?"},
                        {
                            "role": "assistant",
                            "content": "**T1003**\n\nUse `sekurlsa::logonpasswords`.",
                        },
                    ],
                    "source": "test",
                },
            ],
        )
        result = evolve_pairs.load_records([jsonl_path])
        self.assertEqual(len(result), 1)

    def test_respects_max_records(self):
        """max_records parameter limits the number of records returned."""
        jsonl_path = self.tmpdir / "data_test.jsonl"
        self._write_jsonl(
            jsonl_path,
            [
                {
                    "messages": [
                        {"role": "system", "content": "You are a red team specialist."},
                        {
                            "role": "user",
                            "content": f"Question number {i} about hacking?",
                        },
                        {
                            "role": "assistant",
                            "content": f"**T100{i}**\n\nHere is the detailed answer with content.",
                        },
                    ],
                    "source": "test",
                }
                for i in range(10)
            ],
        )
        result = evolve_pairs.load_records([jsonl_path], max_records=3)
        self.assertEqual(len(result), 3)


class TestDiscoverSourceFiles(unittest.TestCase):
    """discover_source_files() finds JSONL files under source directories."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        # Patch DATA_DIR to point to our temp dir
        self.patcher = patch.object(evolve_pairs, "DATA_DIR", self.tmpdir)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        import shutil

        shutil.rmtree(str(self.tmpdir), ignore_errors=True)

    def test_finds_jsonl_files_for_source(self):
        """discover_source_files finds data*.jsonl files under a source dir."""
        source_dir = self.tmpdir / "test-source" / "tools" / "metasploit"
        source_dir.mkdir(parents=True)
        (source_dir / "data_001.jsonl").touch()
        (source_dir / "data_002.jsonl").touch()
        # Non-matching file should be ignored
        (source_dir / "meta.json").touch()

        files = evolve_pairs.discover_source_files("test-source")
        self.assertEqual(len(files), 2)
        self.assertTrue(all(f.name.startswith("data") for f in files))
        self.assertTrue(all(f.suffix == ".jsonl" for f in files))

    def test_returns_all_sources_when_none_specified(self):
        """When source is None, discover all sources."""
        src1 = self.tmpdir / "source-a" / "tools"
        src1.mkdir(parents=True)
        (src1 / "data_001.jsonl").touch()

        src2 = self.tmpdir / "source-b" / "tactic"
        src2.mkdir(parents=True)
        (src2 / "data_001.jsonl").touch()

        files = evolve_pairs.discover_source_files()
        self.assertEqual(len(files), 2)

    def test_skips_underscore_dirs(self):
        """Directories starting with underscore are skipped."""
        src = self.tmpdir / "_private" / "tools"
        src.mkdir(parents=True)
        (src / "data_001.jsonl").touch()

        files = evolve_pairs.discover_source_files()
        self.assertEqual(len(files), 0)


class TestBuildPrompts(unittest.TestCase):
    """Prompt builders produce messages containing original content."""

    def _make_record(self, user_msg: str, assistant_msg: str) -> dict:
        return {
            "messages": [
                {"role": "system", "content": "You are a red team specialist."},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
            "source": "test-source",
        }

    def test_evol_instruct_prompt_contains_original(self):
        """build_evol_instruct_prompt includes original Q&A in the prompt."""
        record = self._make_record(
            "How do I use SQL injection?",
            "**T1190**\n\nUse `' OR 1=1--` to bypass auth.",
        )
        prompt = evolve_pairs.build_evol_instruct_prompt(record)
        self.assertEqual(len(prompt), 2)
        self.assertEqual(prompt[0]["role"], "system")
        self.assertEqual(prompt[1]["role"], "user")
        self.assertIn("How do I use SQL injection?", prompt[1]["content"])
        self.assertIn("T1190", prompt[1]["content"])

    def test_multi_turn_prompt_contains_original(self):
        """build_multi_turn_prompt includes original Q&A in the prompt."""
        record = self._make_record(
            "How do I exploit IMDSv1?",
            "**T1552.005**\n\nUse SSRF to access 169.254.169.254.",
        )
        prompt = evolve_pairs.build_multi_turn_prompt(record)
        self.assertEqual(len(prompt), 2)
        self.assertIn("How do I exploit IMDSv1?", prompt[1]["content"])
        self.assertIn("T1552.005", prompt[1]["content"])

    def test_cot_prompt_contains_original(self):
        """build_cot_prompt includes original Q&A in the prompt."""
        record = self._make_record(
            "Demonstrate Docker socket escape.",
            "**T1611**\n\nUse `docker run -v /:/host alpine chroot /host`.",
        )
        prompt = evolve_pairs.build_cot_prompt(record)
        self.assertEqual(len(prompt), 2)
        self.assertIn("Demonstrate Docker socket escape.", prompt[1]["content"])
        self.assertIn("T1611", prompt[1]["content"])

    def test_evol_instruct_returns_empty_for_invalid_record(self):
        """build_evol_instruct_prompt returns [] for records missing user/assistant."""
        record = {"messages": [{"role": "system", "content": "sys"}]}
        self.assertEqual(evolve_pairs.build_evol_instruct_prompt(record), [])

    def test_multi_turn_returns_empty_for_invalid_record(self):
        """build_multi_turn_prompt returns [] for records missing user/assistant."""
        record = {"messages": [{"role": "system", "content": "sys"}]}
        self.assertEqual(evolve_pairs.build_multi_turn_prompt(record), [])

    def test_cot_returns_empty_for_invalid_record(self):
        """build_cot_prompt returns [] for records missing user/assistant."""
        record = {"messages": [{"role": "system", "content": "sys"}]}
        self.assertEqual(evolve_pairs.build_cot_prompt(record), [])


class TestParseResponses(unittest.TestCase):
    """Response parsers handle valid and invalid LLM output."""

    def _make_record(
        self,
        user_msg: str = "How do I use SQL injection?",
        assistant_msg: str = "**T1190**\n\nUse `' OR 1=1--`.",
        source: str = "test-source",
    ) -> dict:
        return {
            "messages": [
                {"role": "system", "content": "You are a red team specialist."},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
            "source": source,
            "mitre_ids": ["T1190"],
            "license": "MIT",
            "source_uri": "https://example.com",
        }

    def test_parse_evol_instruct_valid(self):
        """parse_evol_instruct_response parses a valid long response."""
        record = self._make_record()
        raw = (
            "**Technique: SQL Injection — T1190**\n\n"
            "**Overview:**\n"
            "SQL injection is a code injection technique...\n\n"
            "**Prerequisites:**\n"
            "- Web application with user input\n\n"
            "**Step-by-Step Execution:**\n"
            "```sql\n"
            "' OR 1=1--\n"
            "```\n\n"
            "**Edge Cases & Variations:**\n"
            "1. Time-based blind injection\n\n"
            "**Detection Artifacts:**\n"
            "- WAF logs: SQL keywords in query params\n\n"
            "**Cleanup:**\n"
            "```bash\n"
            "rm -rf ~/.sqlmap/output/\n"
            "```"
        )
        result = evolve_pairs.parse_evol_instruct_response(raw, record)
        self.assertIsNotNone(result)
        self.assertEqual(result["strategy"], "evol_instruct")
        self.assertEqual(result["source"], "test-source")
        self.assertIn("messages", result)
        self.assertEqual(len(result["messages"]), 3)
        self.assertEqual(result["messages"][2]["role"], "assistant")
        self.assertIn("SQL Injection", result["messages"][2]["content"])

    def test_parse_evol_instruct_too_short(self):
        """parse_evol_instruct_response returns None for very short responses."""
        record = self._make_record()
        result = evolve_pairs.parse_evol_instruct_response("Short.", record)
        self.assertIsNone(result)

    def test_parse_evol_instruct_strips_code_fences(self):
        """parse_evol_instruct_response strips markdown code fences wrapping the response."""
        record = self._make_record()
        raw = (
            "```markdown\n"
            "**Technique: SQL Injection — T1190**\n\n"
            "**Overview:**\n"
            "SQL injection is a code injection technique...\n\n"
            "**Prerequisites:**\n"
            "- Web application with user input\n\n"
            "**Step-by-Step Execution:**\n"
            "```sql\n"
            "' OR 1=1--\n"
            "```\n\n"
            "**Edge Cases & Variations:**\n"
            "1. Time-based blind injection\n\n"
            "**Detection Artifacts:**\n"
            "- WAF logs\n\n"
            "**Cleanup:**\n"
            "```bash\n"
            "rm -rf ~/.sqlmap/output/\n"
            "```\n"
            "```"
        )
        result = evolve_pairs.parse_evol_instruct_response(raw, record)
        self.assertIsNotNone(result)
        # The content should not start with ``` after stripping
        self.assertFalse(result["messages"][2]["content"].startswith("```"))

    def test_parse_multi_turn_valid(self):
        """parse_multi_turn_response parses a valid JSON conversation array."""
        record = self._make_record()
        raw = json.dumps(
            [
                {
                    "role": "user",
                    "content": "What is SQL injection and when would you use it?",
                },
                {
                    "role": "assistant",
                    "content": "**T1190 — SQL Injection**\n\nSQL injection is a code injection technique that exploits unsanitized input in SQL queries. It is used when a web application directly concatenates user input into SQL statements without proper parameterization.",
                },
                {
                    "role": "user",
                    "content": "Walk me through the step-by-step execution.",
                },
                {
                    "role": "assistant",
                    "content": "**Step 1 — Identify injectable parameters:**\n\n```sql\n' OR 1=1--\n```\n\n**Step 2 — Extract data:**\n\n```sql\n' UNION SELECT 1,username,password FROM users--\n```",
                },
                {"role": "user", "content": "What artifacts does this leave?"},
                {
                    "role": "assistant",
                    "content": "**Detection Artifacts:**\n\n- Web server logs: 200/500 responses with SQL keywords\n- WAF logs: UNION/SELECT patterns blocked\n- Database audit: unusual SELECT queries",
                },
            ]
        )
        result = evolve_pairs.parse_multi_turn_response(raw, record)
        self.assertIsNotNone(result)
        self.assertEqual(result["strategy"], "multi_turn")
        self.assertEqual(result["source"], "test-source")
        # system + 6 messages = 7 total
        self.assertGreaterEqual(len(result["messages"]), 7)
        self.assertIn("turns", result)
        self.assertGreaterEqual(result["turns"], 3)

    def test_parse_multi_turn_invalid_json(self):
        """parse_multi_turn_response returns None for invalid JSON."""
        record = self._make_record()
        result = evolve_pairs.parse_multi_turn_response("not json at all", record)
        self.assertIsNone(result)

    def test_parse_multi_turn_too_few_turns(self):
        """parse_multi_turn_response returns None for < 6 messages."""
        record = self._make_record()
        raw = json.dumps(
            [
                {"role": "user", "content": "What is SQL injection?"},
                {"role": "assistant", "content": "It is a code injection technique."},
            ]
        )
        result = evolve_pairs.parse_multi_turn_response(raw, record)
        self.assertIsNone(result)

    def test_parse_cot_valid(self):
        """parse_cot_response parses a response with <thinking> block."""
        record = self._make_record()
        raw = (
            "<thinking>\n"
            "The user is asking about SQL injection. This is MITRE T1190. "
            "The attacker's goal is to extract data from the database. "
            "I should cover: identifying injectable parameters, extracting data, "
            "and bypassing WAFs.\n"
            "</thinking>\n\n"
            "**Technique: SQL Injection — T1190**\n\n"
            "**Step 1 — Identify injectable parameters:**\n"
            "```sql\n"
            "' OR 1=1--\n"
            "```\n\n"
            "**Step 2 — Extract data:**\n"
            "```sql\n"
            "' UNION SELECT 1,username,password FROM users--\n"
            "```\n\n"
            "**Detection Artifacts:**\n"
            "- WAF logs: SQL keywords in query params\n"
            "- Database audit: unusual SELECT queries"
        )
        result = evolve_pairs.parse_cot_response(raw, record)
        self.assertIsNotNone(result)
        self.assertEqual(result["strategy"], "cot")
        self.assertEqual(result["source"], "test-source")
        self.assertIn("messages", result)
        self.assertEqual(len(result["messages"]), 3)
        self.assertIn("<thinking>", result["messages"][2]["content"])

    def test_parse_cot_too_short(self):
        """parse_cot_response returns None for very short responses."""
        record = self._make_record()
        result = evolve_pairs.parse_cot_response("Short.", record)
        self.assertIsNone(result)

    def test_parse_cot_without_thinking_block(self):
        """parse_cot_response still works if the model omits the <thinking> block."""
        record = self._make_record()
        raw = (
            "**Technique: SQL Injection — T1190**\n\n"
            "**Step 1 — Identify injectable parameters:**\n"
            "```sql\n"
            "' OR 1=1--\n"
            "```\n\n"
            "**Detection Artifacts:**\n"
            "- WAF logs: SQL keywords in query params"
        )
        result = evolve_pairs.parse_cot_response(raw, record)
        self.assertIsNotNone(result)
        self.assertEqual(result["strategy"], "cot")

    def test_parse_evol_instruct_preserves_provenance(self):
        """parse_evol_instruct_response preserves source, license, mitre_ids."""
        record = self._make_record()
        raw = (
            "**Technique: SQL Injection — T1190**\n\n"
            "**Overview:**\n"
            "SQL injection is a code injection technique...\n\n"
            "**Prerequisites:**\n"
            "- Web application with user input\n\n"
            "**Step-by-Step Execution:**\n"
            "```sql\n"
            "' OR 1=1--\n"
            "```\n\n"
            "**Edge Cases & Variations:**\n"
            "1. Time-based blind injection\n\n"
            "**Detection Artifacts:**\n"
            "- WAF logs\n\n"
            "**Cleanup:**\n"
            "```bash\n"
            "rm -rf ~/.sqlmap/output/\n"
            "```"
        )
        result = evolve_pairs.parse_evol_instruct_response(raw, record)
        self.assertIsNotNone(result)
        self.assertEqual(result["source"], "test-source")
        self.assertEqual(result["mitre_ids"], ["T1190"])
        self.assertEqual(result["license"], "MIT")
        self.assertEqual(result["source_uri"], "https://example.com")
        self.assertIn("evolved_from", result)
        self.assertIn("evolved_timestamp", result)

    def test_parse_multi_turn_preserves_provenance(self):
        """parse_multi_turn_response preserves source, license, mitre_ids."""
        record = self._make_record()
        raw = json.dumps(
            [
                {
                    "role": "user",
                    "content": "What is SQL injection and when would you use it?",
                },
                {
                    "role": "assistant",
                    "content": "**T1190 — SQL Injection**\n\nSQL injection is a code injection technique that exploits unsanitized input in SQL queries.",
                },
                {
                    "role": "user",
                    "content": "Walk me through the step-by-step execution.",
                },
                {
                    "role": "assistant",
                    "content": "**Step 1 — Identify injectable parameters:**\n\n```sql\n' OR 1=1--\n```",
                },
                {"role": "user", "content": "What artifacts does this leave?"},
                {
                    "role": "assistant",
                    "content": "**Detection Artifacts:**\n\n- Web server logs: 200/500 responses with SQL keywords\n- WAF logs: UNION/SELECT patterns blocked",
                },
            ]
        )
        result = evolve_pairs.parse_multi_turn_response(raw, record)
        self.assertIsNotNone(result)
        self.assertEqual(result["source"], "test-source")
        self.assertEqual(result["mitre_ids"], ["T1190"])
        self.assertEqual(result["license"], "MIT")

    def test_parse_cot_preserves_provenance(self):
        """parse_cot_response preserves source, license, mitre_ids."""
        record = self._make_record()
        raw = (
            "<thinking>\n"
            "The user is asking about SQL injection. This is MITRE T1190.\n"
            "</thinking>\n\n"
            "**Technique: SQL Injection — T1190**\n\n"
            "**Step 1 — Identify injectable parameters:**\n"
            "```sql\n"
            "' OR 1=1--\n"
            "```"
        )
        result = evolve_pairs.parse_cot_response(raw, record)
        self.assertIsNotNone(result)
        self.assertEqual(result["source"], "test-source")
        self.assertEqual(result["mitre_ids"], ["T1190"])
        self.assertEqual(result["license"], "MIT")


# =========================================================================
# filter_evolved tests
# =========================================================================


class TestValidateJsonlStructure(unittest.TestCase):
    """validate_jsonl_structure checks messages array and roles."""

    def test_valid_structure_passes(self):
        """A record with valid messages array passes."""
        record = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "How do I use SQL injection?"},
                {"role": "assistant", "content": "**T1190**\n\nUse `' OR 1=1--`."},
            ],
        }
        reasons = filter_evolved.validate_jsonl_structure(record)
        self.assertEqual(reasons, [])

    def test_missing_messages_fails(self):
        """A record without messages field fails."""
        record = {"source": "test"}
        reasons = filter_evolved.validate_jsonl_structure(record)
        self.assertIn("missing_messages_field", reasons)

    def test_messages_not_array_fails(self):
        """messages that is not a list fails."""
        record = {"messages": "not an array"}
        reasons = filter_evolved.validate_jsonl_structure(record)
        self.assertIn("messages_not_array", reasons)

    def test_empty_messages_fails(self):
        """Empty messages array fails."""
        record = {"messages": []}
        reasons = filter_evolved.validate_jsonl_structure(record)
        self.assertIn("messages_empty", reasons)

    def test_invalid_role_fails(self):
        """A message with an invalid role fails."""
        record = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "How do I use SQL injection?"},
                {"role": "assistant", "content": "**T1190**\n\nUse `' OR 1=1--`."},
                {"role": "invalid_role", "content": "extra"},
            ],
        }
        reasons = filter_evolved.validate_jsonl_structure(record)
        self.assertTrue(any("invalid_role" in r for r in reasons))

    def test_missing_role_fails(self):
        """A message without a role field fails."""
        record = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"content": "no role here"},
            ],
        }
        reasons = filter_evolved.validate_jsonl_structure(record)
        self.assertTrue(any("missing_role" in r for r in reasons))

    def test_missing_content_fails(self):
        """A message without a content field fails."""
        record = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "How do I use SQL injection?"},
                {"role": "assistant"},
            ],
        }
        reasons = filter_evolved.validate_jsonl_structure(record)
        self.assertTrue(any("missing_content" in r for r in reasons))


class TestCheckLengthIncrease(unittest.TestCase):
    """check_length_increase validates word count ratio."""

    def _make_record(
        self, assistant_text: str, user_text: str = "How do I use SQL injection?"
    ) -> dict:
        return {
            "messages": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ],
        }

    def test_three_x_passes(self):
        """3x longer evolved passes with default min_factor=2.0."""
        original = self._make_record("Short answer.")
        evolved = self._make_record(
            "Long answer with many words here to make it three times longer "
            "than the original short answer that only had a few words. "
            "This should easily pass the length check because it is much "
            "longer than the original response."
        )
        reasons = filter_evolved.check_length_increase(evolved, original)
        self.assertEqual(reasons, [])

    def test_one_point_five_x_fails(self):
        """1.5x longer evolved fails with default min_factor=2.0."""
        original = self._make_record(
            "This is a longer original answer with several words in it."
        )
        evolved = self._make_record(
            "This evolved answer is a bit longer but not enough to pass the threshold."
        )
        reasons = filter_evolved.check_length_increase(evolved, original)
        self.assertTrue(any("length_increase_insufficient" in r for r in reasons))

    def test_both_empty_passes(self):
        """Both original and evolved empty — returns specific failure reason."""
        # Records with no messages have 0 total words
        original = {"messages": []}
        evolved = {"messages": []}
        reasons = filter_evolved.check_length_increase(evolved, original)
        # The function returns a specific failure reason for both-empty
        self.assertEqual(reasons, ["both_original_and_evolved_empty"])

    def test_original_empty_evolved_not_empty_passes(self):
        """Original empty but evolved has content passes."""
        original = {"messages": []}
        evolved = self._make_record("Some content here.")
        reasons = filter_evolved.check_length_increase(evolved, original)
        self.assertEqual(reasons, [])

    def test_custom_min_factor(self):
        """Custom min_factor is respected."""
        original = self._make_record("Short answer.")
        evolved = self._make_record("This evolved answer is a bit longer but not 3x.")
        # min_factor=3.0 should fail
        reasons = filter_evolved.check_length_increase(
            evolved, original, min_factor=3.0
        )
        self.assertTrue(any("length_increase_insufficient" in r for r in reasons))


class TestCheckMitrePreservation(unittest.TestCase):
    """check_mitre_id_preservation validates MITRE ID consistency."""

    def _make_record(self, mitre_ids: list[str] | None = None) -> dict:
        rec = {
            "messages": [
                {"role": "user", "content": "How do I use SQL injection?"},
                {"role": "assistant", "content": "**T1190**\n\nUse `' OR 1=1--`."},
            ],
        }
        if mitre_ids is not None:
            rec["mitre_ids"] = mitre_ids
        return rec

    def test_same_ids_pass(self):
        """Same mitre_ids in both records passes."""
        original = self._make_record(["T1190"])
        evolved = self._make_record(["T1190"])
        reasons = filter_evolved.check_mitre_id_preservation(evolved, original)
        self.assertEqual(reasons, [])

    def test_different_ids_fail(self):
        """Different mitre_ids fails."""
        original = self._make_record(["T1190"])
        evolved = self._make_record(["T1190", "T1059"])
        reasons = filter_evolved.check_mitre_id_preservation(evolved, original)
        self.assertTrue(any("mitre_ids_added" in r for r in reasons))

    def test_missing_ids_fail(self):
        """Missing mitre_ids in evolved fails."""
        original = self._make_record(["T1190"])
        evolved = self._make_record([])
        reasons = filter_evolved.check_mitre_id_preservation(evolved, original)
        self.assertTrue(any("mitre_ids_removed_all" in r for r in reasons))

    def test_both_none_passes(self):
        """Both records without mitre_ids passes."""
        original = self._make_record(None)
        evolved = self._make_record(None)
        reasons = filter_evolved.check_mitre_id_preservation(evolved, original)
        self.assertEqual(reasons, [])

    def test_case_insensitive_comparison(self):
        """MITRE ID comparison is case-insensitive."""
        original = self._make_record(["T1190"])
        evolved = self._make_record(["t1190"])
        reasons = filter_evolved.check_mitre_id_preservation(evolved, original)
        self.assertEqual(reasons, [])


class TestCheckProvenance(unittest.TestCase):
    """check_provenance_preservation validates metadata fields."""

    def _make_record(self, **overrides) -> dict:
        base = {
            "messages": [
                {"role": "user", "content": "How do I use SQL injection?"},
                {"role": "assistant", "content": "**T1190**\n\nUse `' OR 1=1--`."},
            ],
            "source": "test-source",
            "source_uri": "https://example.com",
            "license": "MIT",
            "license_uri": "https://opensource.org/licenses/MIT",
            "rights_contact": "test@example.com",
        }
        base.update(overrides)
        return base

    def test_all_fields_present_passes(self):
        """All provenance fields present and matching passes."""
        original = self._make_record()
        evolved = self._make_record()
        reasons = filter_evolved.check_provenance_preservation(evolved, original)
        self.assertEqual(reasons, [])

    def test_missing_field_fails(self):
        """Missing provenance field in evolved fails."""
        original = self._make_record()
        evolved = self._make_record()
        del evolved["license"]
        reasons = filter_evolved.check_provenance_preservation(evolved, original)
        self.assertTrue(any("provenance_missing_license" in r for r in reasons))

    def test_mismatched_field_fails(self):
        """Mismatched provenance field value fails."""
        original = self._make_record()
        evolved = self._make_record(license="GPL-3.0")
        reasons = filter_evolved.check_provenance_preservation(evolved, original)
        self.assertTrue(any("provenance_mismatch_license" in r for r in reasons))

    def test_original_missing_field_skips(self):
        """If original doesn't have a field, it's not checked."""
        original = self._make_record()
        del original["license_uri"]
        evolved = self._make_record()
        del evolved["license_uri"]
        reasons = filter_evolved.check_provenance_preservation(evolved, original)
        self.assertEqual(reasons, [])


class TestCheckHallucinatedContent(unittest.TestCase):
    """check_no_hallucinated_content detects fake MITRE IDs."""

    def _make_record(
        self, assistant_text: str, mitre_ids: list[str] | None = None
    ) -> dict:
        rec = {
            "messages": [
                {"role": "user", "content": "How do I use SQL injection?"},
                {"role": "assistant", "content": assistant_text},
            ],
        }
        if mitre_ids is not None:
            rec["mitre_ids"] = mitre_ids
        return rec

    def test_no_hallucination_passes(self):
        """Evolved with only known MITRE IDs passes."""
        original = self._make_record(
            "**T1190**\n\nUse `' OR 1=1--`.", mitre_ids=["T1190"]
        )
        evolved = self._make_record(
            "**T1190**\n\nUse `' OR 1=1--`.\n\n**Detection:** WAF logs."
        )
        reasons = filter_evolved.check_no_hallucinated_content(evolved, original)
        self.assertEqual(reasons, [])

    def test_hallucinated_id_fails(self):
        """Evolved with a MITRE ID not in original fails."""
        original = self._make_record(
            "**T1190**\n\nUse `' OR 1=1--`.", mitre_ids=["T1190"]
        )
        evolved = self._make_record(
            "**T1190**\n\nUse `' OR 1=1--`.\n\n**Also see T1059** for command execution."
        )
        reasons = filter_evolved.check_no_hallucinated_content(evolved, original)
        self.assertTrue(any("hallucinated_mitre_ids" in r for r in reasons))

    def test_id_in_original_content_not_hallucinated(self):
        """MITRE ID found in original content (not just mitre_ids field) is OK."""
        original = self._make_record(
            "**T1190** — SQL Injection\n\n**Also see T1059** for command execution.",
            mitre_ids=["T1190"],
        )
        evolved = self._make_record(
            "**T1190**\n\nUse `' OR 1=1--`.\n\n**T1059** is related for post-exploitation."
        )
        reasons = filter_evolved.check_no_hallucinated_content(evolved, original)
        self.assertEqual(reasons, [])


class TestDeduplication(unittest.TestCase):
    """check_deduplication detects near-duplicate records."""

    def _make_record(self, text: str) -> dict:
        return {
            "messages": [
                {"role": "user", "content": "How do I use SQL injection?"},
                {"role": "assistant", "content": text},
            ],
        }

    def test_duplicate_removed(self):
        """Near-duplicate records are detected."""
        text = (
            "**T1190 — SQL Injection**\n\n"
            "SQL injection is a code injection technique that exploits unsanitized "
            "input in SQL queries. It is used when a web application directly "
            "concatenates user input into SQL statements without proper parameterization."
        )
        records = [
            self._make_record(text),
            self._make_record(text),
            self._make_record(
                "**T1059 — Command Execution**\n\n"
                "Command execution is a technique where an attacker runs arbitrary "
                "commands on a target system."
            ),
        ]
        dup_map = filter_evolved.check_deduplication(records, threshold=0.9)
        # The first two are near-duplicates (identical), so one should be flagged
        self.assertEqual(len(dup_map), 1)
        # The duplicate should be index 1 (the second record)
        self.assertIn(1, dup_map)

    def test_unique_records_not_flagged(self):
        """Completely different records are not flagged as duplicates."""
        records = [
            self._make_record(
                "**T1190 — SQL Injection**\n\n"
                "SQL injection exploits unsanitized input in SQL queries."
            ),
            self._make_record(
                "**T1059 — Command Execution**\n\n"
                "Command execution runs arbitrary commands on a target system."
            ),
        ]
        dup_map = filter_evolved.check_deduplication(records, threshold=0.9)
        self.assertEqual(len(dup_map), 0)

    def test_empty_records_skipped(self):
        """Records with empty content are skipped in dedup check."""
        # Records with no messages at all have empty word sets
        records = [
            {"messages": []},
            {"messages": []},
        ]
        dup_map = filter_evolved.check_deduplication(records, threshold=0.9)
        self.assertEqual(len(dup_map), 0)


class TestFilterPipeline(unittest.TestCase):
    """End-to-end filter pipeline: input → filter → output."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil

        shutil.rmtree(str(self.tmpdir), ignore_errors=True)

    def _make_record(
        self,
        user_msg: str,
        assistant_msg: str,
        mitre_ids: list[str] | None = None,
        source: str = "test-source",
        source_uri: str = "https://example.com",
        license: str = "MIT",
        license_uri: str = "https://opensource.org/licenses/MIT",
        rights_contact: str = "test@example.com",
    ) -> dict:
        rec = {
            "messages": [
                {"role": "system", "content": "You are a red team specialist."},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
            "source": source,
            "source_uri": source_uri,
            "license": license,
            "license_uri": license_uri,
            "rights_contact": rights_contact,
        }
        if mitre_ids is not None:
            rec["mitre_ids"] = mitre_ids
        return rec

    def test_filter_pipeline_passes_good_records(self):
        """Good records pass through the filter pipeline."""
        original = self._make_record(
            user_msg="How do I use SQL injection?",
            assistant_msg="**T1190**\n\nUse `' OR 1=1--`.",
            mitre_ids=["T1190"],
        )
        evolved = self._make_record(
            user_msg="How do I use SQL injection?",
            assistant_msg=(
                "**T1190 — SQL Injection**\n\n"
                "**Overview:**\n"
                "SQL injection is a code injection technique that exploits unsanitized "
                "input in SQL queries.\n\n"
                "**Step-by-Step Execution:**\n"
                "```sql\n"
                "' OR 1=1--\n"
                "```\n\n"
                "**Detection Artifacts:**\n"
                "- WAF logs: SQL keywords in query params\n"
                "- Database audit: unusual SELECT queries\n\n"
                "**Cleanup:**\n"
                "```bash\n"
                "rm -rf ~/.sqlmap/output/\n"
                "```"
            ),
            mitre_ids=["T1190"],
        )

        # Write evolved file
        evolved_path = self.tmpdir / "test-source_evol_instruct_abc123.jsonl"
        with open(evolved_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(evolved) + "\n")

        # Write original file
        original_dir = self.tmpdir / "originals"
        original_dir.mkdir()
        original_path = original_dir / "data_001.jsonl"
        with open(original_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(original) + "\n")

        report, passed = filter_evolved.filter_evolved_file(
            evolved_path,
            original_path=original_dir,
        )
        self.assertEqual(report["total"], 1)
        self.assertEqual(report["passed"], 1)
        self.assertEqual(report["failed"], 0)
        self.assertEqual(len(passed), 1)

    def test_filter_pipeline_rejects_bad_records(self):
        """Records with missing provenance fail the filter."""
        original = self._make_record(
            user_msg="How do I use SQL injection?",
            assistant_msg="**T1190**\n\nUse `' OR 1=1--`.",
            mitre_ids=["T1190"],
        )
        # Evolved record missing the license field
        evolved = self._make_record(
            user_msg="How do I use SQL injection?",
            assistant_msg=(
                "**T1190 — SQL Injection**\n\n"
                "**Overview:**\n"
                "SQL injection is a code injection technique...\n\n"
                "**Step-by-Step Execution:**\n"
                "```sql\n"
                "' OR 1=1--\n"
                "```\n\n"
                "**Detection Artifacts:**\n"
                "- WAF logs: SQL keywords in query params\n\n"
                "**Cleanup:**\n"
                "```bash\n"
                "rm -rf ~/.sqlmap/output/\n"
                "```"
            ),
            mitre_ids=["T1190"],
        )
        del evolved["license"]

        evolved_path = self.tmpdir / "test-source_evol_instruct_abc123.jsonl"
        with open(evolved_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(evolved) + "\n")

        original_dir = self.tmpdir / "originals"
        original_dir.mkdir()
        original_path = original_dir / "data_001.jsonl"
        with open(original_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(original) + "\n")

        report, passed = filter_evolved.filter_evolved_file(
            evolved_path,
            original_path=original_dir,
        )
        self.assertEqual(report["total"], 1)
        self.assertEqual(report["passed"], 0)
        self.assertEqual(report["failed"], 1)
        # Check that the failure reason mentions provenance_missing
        failure_keys = list(report["failure_reasons"].keys())
        self.assertTrue(
            any("provenance_missing" in k for k in failure_keys),
            f"Expected provenance_missing in failure reasons, got {failure_keys}",
        )

    def test_filter_pipeline_deduplicates(self):
        """Duplicate evolved records are removed by the filter."""
        original = self._make_record(
            user_msg="How do I use SQL injection?",
            assistant_msg="**T1190**\n\nUse `' OR 1=1--`.",
            mitre_ids=["T1190"],
        )
        evolved_text = (
            "**T1190 — SQL Injection**\n\n"
            "**Overview:**\n"
            "SQL injection is a code injection technique that exploits unsanitized "
            "input in SQL queries.\n\n"
            "**Step-by-Step Execution:**\n"
            "```sql\n"
            "' OR 1=1--\n"
            "```\n\n"
            "**Detection Artifacts:**\n"
            "- WAF logs: SQL keywords in query params\n\n"
            "**Cleanup:**\n"
            "```bash\n"
            "rm -rf ~/.sqlmap/output/\n"
            "```"
        )
        evolved = self._make_record(
            user_msg="How do I use SQL injection?",
            assistant_msg=evolved_text,
            mitre_ids=["T1190"],
        )

        evolved_path = self.tmpdir / "test-source_evol_instruct_abc123.jsonl"
        with open(evolved_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(evolved) + "\n")
            f.write(json.dumps(evolved) + "\n")  # duplicate

        original_dir = self.tmpdir / "originals"
        original_dir.mkdir()
        original_path = original_dir / "data_001.jsonl"
        with open(original_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(original) + "\n")

        report, passed = filter_evolved.filter_evolved_file(
            evolved_path,
            original_path=original_dir,
        )
        self.assertEqual(report["total"], 2)
        self.assertEqual(report["passed"], 1)  # one passed, one deduped
        self.assertEqual(report["duplicates_removed"], 1)


class TestValidateSingleRecord(unittest.TestCase):
    """validate_single_record runs all per-record checks."""

    def _make_record(self, **overrides) -> dict:
        base = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "How do I use SQL injection?"},
                {"role": "assistant", "content": "**T1190**\n\nUse `' OR 1=1--`."},
            ],
            "source": "test-source",
            "source_uri": "https://example.com",
            "license": "MIT",
            "license_uri": "https://opensource.org/licenses/MIT",
            "rights_contact": "test@example.com",
            "mitre_ids": ["T1190"],
        }
        base.update(overrides)
        return base

    def test_valid_record_passes(self):
        """A fully valid record passes all checks."""
        original = self._make_record()
        # Evolved record with a much longer assistant message
        evolved = {
            "messages": [
                {"role": "system", "content": "You are a red team specialist."},
                {"role": "user", "content": "How do I use SQL injection?"},
                {
                    "role": "assistant",
                    "content": (
                        "**T1190 — SQL Injection**\n\n"
                        "**Overview:**\n"
                        "SQL injection is a code injection technique that exploits unsanitized "
                        "input in SQL queries. It is one of the most common web application "
                        "vulnerabilities and can lead to data exfiltration, authentication bypass, "
                        "and remote code execution in severe cases.\n\n"
                        "**Step-by-Step Execution:**\n"
                        "```sql\n"
                        "' OR 1=1--\n"
                        "```\n\n"
                        "**Detection Artifacts:**\n"
                        "- WAF logs: SQL keywords in query params\n"
                        "- Database audit: unusual SELECT queries\n\n"
                        "**Cleanup:**\n"
                        "```bash\n"
                        "rm -rf ~/.sqlmap/output/\n"
                        "```"
                    ),
                },
            ],
            "source": "test-source",
            "source_uri": "https://example.com",
            "license": "MIT",
            "license_uri": "https://opensource.org/licenses/MIT",
            "rights_contact": "test@example.com",
            "mitre_ids": ["T1190"],
        }
        reasons = filter_evolved.validate_single_record(evolved, original)
        self.assertEqual(reasons, [])

    def test_invalid_structure_fails(self):
        """A record with invalid structure fails."""
        evolved = {"no_messages": True}
        reasons = filter_evolved.validate_single_record(evolved, None)
        self.assertTrue(any("missing_messages_field" in r for r in reasons))

    def test_no_original_skips_comparison_checks(self):
        """When original is None, comparison checks are skipped."""
        evolved = self._make_record()
        reasons = filter_evolved.validate_single_record(evolved, None)
        # Only structure check runs — should pass
        self.assertEqual(reasons, [])


class TestHelperFunctions(unittest.TestCase):
    """Helper functions work correctly."""

    def test_total_word_count(self):
        """_total_word_count counts words across all messages."""
        record = {
            "messages": [
                {"role": "user", "content": "How do I use SQL injection?"},
                {"role": "assistant", "content": "**T1190**\n\nUse `' OR 1=1--`."},
            ],
        }
        count = filter_evolved._total_word_count(record)
        # "How do I use SQL injection?" = 6 words
        # "**T1190**\n\nUse `' OR 1=1--`." = 5 words
        self.assertEqual(count, 11)

    def test_record_match_key(self):
        """_record_match_key extracts first 200 chars of first user message."""
        record = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "How do I use SQL injection?"},
                {"role": "assistant", "content": "**T1190**\n\nUse `' OR 1=1--`."},
            ],
        }
        key = filter_evolved._record_match_key(record)
        self.assertEqual(key, "how do i use sql injection?")

    def test_record_match_key_empty_when_no_user(self):
        """_record_match_key returns '' when there's no user message."""
        record = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "assistant", "content": "**T1190**\n\nUse `' OR 1=1--`."},
            ],
        }
        key = filter_evolved._record_match_key(record)
        self.assertEqual(key, "")

    def test_parse_evolved_filename(self):
        """parse_evolved_filename extracts source name from filename."""
        name = filter_evolved.parse_evolved_filename(
            "metasploit-framework_multi_turn_abc123.jsonl"
        )
        self.assertEqual(name, "metasploit-framework")

    def test_parse_evolved_filename_filtered(self):
        """parse_evolved_filename handles _filtered suffix."""
        name = filter_evolved.parse_evolved_filename(
            "metasploit-framework_multi_turn_abc123_filtered.jsonl"
        )
        self.assertEqual(name, "metasploit-framework")

    def test_parse_evolved_filename_unknown(self):
        """parse_evolved_filename returns None for unrecognized pattern."""
        name = filter_evolved.parse_evolved_filename("random_file.jsonl")
        self.assertIsNone(name)

    def test_load_jsonl(self):
        """load_jsonl loads records from a JSONL file."""
        path = Path(tempfile.mktemp(suffix=".jsonl"))
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"messages": [{"role": "user", "content": "Q1"}]}\n')
                f.write('{"messages": [{"role": "user", "content": "Q2"}]}\n')
            records = filter_evolved.load_jsonl(path)
            self.assertEqual(len(records), 2)
        finally:
            path.unlink(missing_ok=True)

    def test_load_jsonl_skips_empty_lines(self):
        """load_jsonl skips empty lines."""
        path = Path(tempfile.mktemp(suffix=".jsonl"))
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"messages": [{"role": "user", "content": "Q1"}]}\n')
                f.write("\n")
                f.write('{"messages": [{"role": "user", "content": "Q2"}]}\n')
            records = filter_evolved.load_jsonl(path)
            self.assertEqual(len(records), 2)
        finally:
            path.unlink(missing_ok=True)

    def test_build_original_index(self):
        """build_original_index creates a lookup by user message content."""
        records = [
            {
                "messages": [
                    {"role": "user", "content": "How do I use SQL injection?"},
                    {"role": "assistant", "content": "**T1190**\n\nUse `' OR 1=1--`."},
                ],
            },
            {
                "messages": [
                    {"role": "user", "content": "How do I scan ports?"},
                    {"role": "assistant", "content": "**T1046**\n\nUse `nmap -sV`."},
                ],
            },
        ]
        index = filter_evolved.build_original_index(records)
        self.assertEqual(len(index), 2)
        self.assertIn("how do i use sql injection?", index)
        self.assertIn("how do i scan ports?", index)


if __name__ == "__main__":
    unittest.main(verbosity=2)
