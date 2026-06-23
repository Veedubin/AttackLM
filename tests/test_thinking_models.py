#!/usr/bin/env python3
"""Tests for thinking content stripping in generate_synthetic_scarce.py.

Thinking models (qwen3.6-12b-thinking, o1, deepseek-r1, etc.) were deprecated
on 2026-06-11 and the dedicated thinking-model detection helpers were removed.
The only thinking-related logic that remains is `strip_thinking()`, which is
still used to clean up legacy reasoning blocks if they appear in raw LLM
output.

Run with:
    python -m pytest tests/test_thinking_models.py -v
"""

import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import generate_synthetic_scarce as g  # noqa: E402


class TestStripThinking(unittest.TestCase):
    """strip_thinking() should remove legacy reasoning blocks."""

    def test_strip_thinking_preserves_real_content(self):
        """strip_thinking should not strip non-thinking content."""
        text = "===PAIR===\nQ: test?\nA: response\n===END==="
        result = g.strip_thinking(text)
        self.assertIn("===PAIR===", result)
        self.assertIn("Q: test?", result)
        self.assertIn("A: response", result)

    def test_strip_thinking_removes_legacy_patterns(self):
        """strip_thinking should remove <think>...</think> blocks."""
        text = "<think>\nreasoning here\n</think>\n===PAIR===\nQ: test?\nA: response"
        result = g.strip_thinking(text)
        self.assertNotIn("reasoning here", result)
        self.assertIn("===PAIR===", result)
        self.assertIn("Q: test?", result)

    def test_strip_thinking_empty(self):
        """Empty input should return empty."""
        self.assertEqual(g.strip_thinking(""), "")

    def test_strip_thinking_only_thinking(self):
        """If only thinking is present, result should be empty."""
        text = "<think>\njust thinking\n</think>"
        result = g.strip_thinking(text)
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
