#!/usr/bin/env python3
"""Tests for thinking-model handling in generate_synthetic_scarce.py.

Thinking models (qwen3.6-12b-thinking, o1, deepseek-r1, etc.) put
chain-of-thought in a separate `reasoning_content` field. Without
special handling, the script:
1. Sees empty `content` and retries 4 times, all failing
2. Returns 0 pairs and reports "0 pairs | 0 tok/s | 0 pair/s"
3. Even though LMStudio logs show thousands of completion tokens

These tests verify the auto-detection, max_tokens resolution, and
response parsing all work for thinking models.

Run with:
    python -m pytest tests/test_thinking_models.py -v
"""

import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import generate_synthetic_scarce as g  # noqa: E402


class TestIsThinkingModel(unittest.TestCase):
    """is_thinking_model() should detect thinking models by name."""

    def test_qwen3_6_thinking_detected(self):
        self.assertTrue(
            g.is_thinking_model(
                "qwen3.6-12b-iq-ultra-heretic-uncensored-thinking-v2-hightop"
            )
        )

    def test_qwen2_5_coder_not_thinking(self):
        """qwen2.5-coder is a non-thinking model."""
        self.assertFalse(g.is_thinking_model("qwen2.5-coder-14b-instruct-uncensored"))

    def test_openai_o1_detected(self):
        self.assertTrue(g.is_thinking_model("o1-mini"))
        self.assertTrue(g.is_thinking_model("o1-preview"))

    def test_qwen_qwq_detected(self):
        self.assertTrue(g.is_thinking_model("qwq-32b-preview"))

    def test_deepseek_r1_detected(self):
        self.assertTrue(g.is_thinking_model("deepseek-r1-distill-llama-70b"))

    def test_qwen3_r1_detected(self):
        """qwen3-1.7b-r1 should match the qwen3-r1 pattern."""
        self.assertTrue(g.is_thinking_model("qwen3-1.7b-r1"))
        self.assertTrue(g.is_thinking_model("qwen3.5-r1"))

    def test_heretic_model_detected(self):
        """Heretic/abliterated models often keep thinking behavior."""
        self.assertTrue(g.is_thinking_model("heretic-uncensored"))

    def test_llama3_not_thinking(self):
        self.assertFalse(g.is_thinking_model("llama3.1-70b"))
        self.assertFalse(g.is_thinking_model("llama3.2-3b"))

    def test_mistral_not_thinking(self):
        self.assertFalse(g.is_thinking_model("mistral-7b-instruct"))

    def test_empty_name(self):
        self.assertFalse(g.is_thinking_model(""))

    def test_case_insensitive(self):
        """Detection should be case-insensitive."""
        self.assertTrue(g.is_thinking_model("QWEN3.6-12B-THINKING"))
        self.assertTrue(g.is_thinking_model("DeepSeek-R1"))


class TestGetMaxTokensForModel(unittest.TestCase):
    """get_max_tokens_for_model() should auto-pick based on model type."""

    def test_thinking_model_default_9000(self):
        """Thinking model default: 9000 tokens."""
        self.assertEqual(
            g.get_max_tokens_for_model("qwen3.6-12b-thinking-v2", None), 9000
        )

    def test_non_thinking_model_default_3000(self):
        """Non-thinking model default: 3000 tokens."""
        self.assertEqual(g.get_max_tokens_for_model("qwen2.5-coder-14b", None), 3000)

    def test_explicit_override_wins(self):
        """Explicit override should beat auto-detection."""
        self.assertEqual(g.get_max_tokens_for_model("qwen3.6-12b-thinking", 5000), 5000)
        self.assertEqual(g.get_max_tokens_for_model("qwen2.5-coder-14b", 7000), 7000)

    def test_zero_override_falls_through(self):
        """Override of 0 (or None) should not be treated as 'use 0'."""
        self.assertEqual(g.get_max_tokens_for_model("qwen3.6-12b-thinking", 0), 9000)
        self.assertEqual(g.get_max_tokens_for_model("qwen2.5-coder-14b", None), 3000)


class TestResponseParsingForThinking(unittest.TestCase):
    """call_llm() should handle reasoning_content correctly.

    The actual call_llm() is hard to test without a real LLM, so we
    verify the response parsing logic by examining the data extraction
    patterns used in the code.
    """

    def test_strip_thinking_preserves_real_content(self):
        """strip_thinking should not strip non-thinking content."""
        text = "===PAIR===\nQ: test?\nA: response\n===END==="
        result = g.strip_thinking(text)
        self.assertIn("===PAIR===", result)
        self.assertIn("Q: test?", result)
        self.assertIn("A: response", result)

    def test_strip_thinking_removes_legacy_patterns(self):
        """strip_thinking should remove <think>...</think> blocks."""
        text = "<think>reasoning here</think>\n===PAIR===\nQ: test?\nA: response"
        result = g.strip_thinking(text)
        self.assertNotIn("reasoning here", result)
        self.assertIn("===PAIR===", result)
        self.assertIn("Q: test?", result)

    def test_strip_thinking_empty(self):
        """Empty input should return empty."""
        self.assertEqual(g.strip_thinking(""), "")

    def test_strip_thinking_only_thinking(self):
        """If only thinking is present, result should be empty."""
        text = "<think>just thinking</think>"
        result = g.strip_thinking(text)
        self.assertEqual(result, "")


class TestThinkingModelDefaults(unittest.TestCase):
    """Verify the constants used for thinking model handling."""

    def test_thinking_default_is_larger(self):
        """Thinking model default must be > non-thinking default."""
        self.assertGreater(
            g.DEFAULT_MAX_TOKENS_THINKING,
            g.DEFAULT_MAX_TOKENS_NON_THINKING,
        )

    def test_thinking_default_at_least_5000(self):
        """Thinking model needs at least 5000 to leave room for content
        after reasoning. We use 9000 to be safe."""
        self.assertGreaterEqual(g.DEFAULT_MAX_TOKENS_THINKING, 5000)

    def test_non_thinking_default_reasonable(self):
        """Non-thinking model default should be enough for ~5 pairs."""
        self.assertGreaterEqual(g.DEFAULT_MAX_TOKENS_NON_THINKING, 2000)


class TestCallLlmHandlesReasoningField(unittest.TestCase):
    """Verify call_llm reads reasoning_content as fallback.

    Uses a mock to test the response-parsing logic without a real LLM.
    """

    def test_falls_back_to_reasoning_when_content_empty(self):
        """When content is empty but reasoning_content has the response,
        call_llm should return the reasoning content."""
        from unittest.mock import patch, MagicMock

        # Mock response with empty content but rich reasoning
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",  # empty
                        "reasoning_content": "===PAIR===\nQ: test?\nA: response\n===END===",
                    },
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300,
                "completion_tokens_details": {"reasoning_tokens": 195},
            },
        }
        mock_response.raise_for_status = lambda: None

        with patch.object(g, "_get_session") as mock_session:
            mock_session.return_value.post.return_value = mock_response
            with patch.dict(
                "os.environ",
                {
                    "BACKEND": "lmstudio",
                    "LMSTUDIO_MODEL": "qwen3.6-12b-thinking",
                },
            ):
                # Force re-read of constants
                g.BACKEND = "lmstudio"
                g.LMSTUDIO_MODEL = "qwen3.6-12b-thinking"
                result = g.call_llm(
                    [{"role": "user", "content": "test"}],
                    max_retries=0,
                )
                # Should have fallen back to reasoning content
                self.assertIn("===PAIR===", result["content"])
                self.assertIn("Q: test?", result["content"])
                self.assertEqual(result["usage"]["reasoning_tokens"], 195)
                self.assertEqual(result["usage"]["finish_reason"], "length")

    def test_uses_content_when_present(self):
        """When content is present, use it (don't fall back to reasoning)."""
        from unittest.mock import patch, MagicMock

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "===PAIR===\nQ: real?\nA: real response",
                        "reasoning_content": "thinking notes that should be ignored",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150,
                "completion_tokens_details": {"reasoning_tokens": 30},
            },
        }
        mock_response.raise_for_status = lambda: None

        with patch.object(g, "_get_session") as mock_session:
            mock_session.return_value.post.return_value = mock_response
            with patch.dict(
                "os.environ",
                {"BACKEND": "lmstudio", "LMSTUDIO_MODEL": "qwen2.5-coder-14b"},
            ):
                g.BACKEND = "lmstudio"
                g.LMSTUDIO_MODEL = "qwen2.5-coder-14b"
                result = g.call_llm(
                    [{"role": "user", "content": "test"}],
                    max_retries=0,
                )
                self.assertIn("real response", result["content"])
                self.assertNotIn("thinking notes", result["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
