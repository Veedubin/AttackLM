#!/usr/bin/env python3
"""Tests for scripts/balance_buckets.py.

Run with:
    python -m pytest tests/test_balance_buckets.py -v

Or directly:
    python tests/test_balance_buckets.py
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Make the scripts/ dir importable
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import balance_buckets
from bucket_loader import BUCKETS_DIR


class TestStratifyKey(unittest.TestCase):
    """The _stratify_key() function picks a stable, content-based key
    for each example. Multiple fallback tiers are exercised here."""

    def test_mitre_id_preferred(self):
        ex = {
            "mitre_ids": ["T1001.002", "T1001"],
            "source": "redcanaryco/atomic-red-team",
            "messages": [{"role": "assistant", "content": "ignored"}],
        }
        self.assertEqual(balance_buckets._stratify_key(ex), "T1001.002")

    def test_source_used_when_no_mitre(self):
        ex = {
            "mitre_ids": None,
            "source": "rapid7/metasploit-framework",
            "messages": [{"role": "assistant", "content": "ignored"}],
        }
        self.assertEqual(
            balance_buckets._stratify_key(ex), "src:rapid7/metasploit-framework"
        )

    def test_first_assistant_line_used_for_metasploit(self):
        # The metasploit bucket has no mitre_ids and no source. We
        # fall back to the first line of the assistant response.
        ex = {
            "mitre_ids": None,
            "source": None,
            "messages": [
                {"role": "system", "content": "ignored"},
                {
                    "role": "assistant",
                    "content": (
                        "**Module: `exploits/windows/smb/psexec`** — "
                        "Microsoft Windows Authenticated User Code Execution\n"
                        "\nSome other content"
                    ),
                },
            ],
        }
        key = balance_buckets._stratify_key(ex)
        self.assertTrue(key.startswith("**Module:"))
        self.assertIn("psexec", key)
        # Truncated to 80 chars max
        self.assertLessEqual(len(key), 80)

    def test_unknown_fallback(self):
        ex = {
            "mitre_ids": None,
            "source": None,
            "messages": [{"role": "user", "content": "no assistant here"}],
        }
        self.assertEqual(balance_buckets._stratify_key(ex), "unknown")

    def test_empty_mitre_list_falls_through(self):
        ex = {
            "mitre_ids": [],
            "source": None,
            "messages": [
                {"role": "assistant", "content": "**Technique: T1059** — Cmd"}
            ],
        }
        # Empty list should not count as a valid mitre_id
        self.assertTrue(balance_buckets._stratify_key(ex).startswith("**Technique"))


class TestSamplingStrategies(unittest.TestCase):
    """Verify head / random / stratified produce the right number of
    examples, respect the seed for reproducibility, and stratified
    gives minimum 1 per group when possible."""

    def setUp(self):
        # Build a synthetic bucket with known structure
        # 5 groups: A=10, B=5, C=3, D=2, E=1 (21 total, 5 groups)
        self.examples = []
        for label, count in [("A", 10), ("B", 5), ("C", 3), ("D", 2), ("E", 1)]:
            for i in range(count):
                self.examples.append(
                    {
                        "messages": [
                            {
                                "role": "assistant",
                                "content": f"**Group {label} item {i}**",
                            }
                        ]
                    }
                )

    def test_head_returns_first_n(self):
        result = balance_buckets._sample_head(self.examples, 5, seed=42)
        self.assertEqual(len(result), 5)
        # All from group A (first 10 in the list)
        for ex in result:
            self.assertIn("Group A", ex["messages"][0]["content"])

    def test_random_respects_seed(self):
        r1 = balance_buckets._sample_random(self.examples, 10, seed=42)
        r2 = balance_buckets._sample_random(self.examples, 10, seed=42)
        self.assertEqual(r1, r2)

    def test_random_changes_with_seed(self):
        r1 = balance_buckets._sample_random(self.examples, 10, seed=42)
        r2 = balance_buckets._sample_random(self.examples, 10, seed=43)
        self.assertNotEqual(r1, r2)

    def test_stratified_gives_min_one_per_group(self):
        """If n >= num_groups, every group should be represented."""
        result = balance_buckets._sample_stratified(self.examples, 5, seed=42)
        self.assertEqual(len(result), 5)
        groups = {balance_buckets._stratify_key(ex) for ex in result}
        # 5 groups, target 5, so every group should appear exactly once
        self.assertEqual(len(groups), 5)

    def test_stratified_caps_at_n(self):
        """Output length must not exceed n even with rounding overshoot."""
        result = balance_buckets._sample_stratified(self.examples, 5, seed=42)
        self.assertLessEqual(len(result), 5)

    def test_stratified_falls_back_to_random_for_too_few_groups(self):
        """If a bucket has < 3 distinct groups, stratified is meaningless."""
        two_group_examples = [
            {"messages": [{"role": "assistant", "content": "**A**"}]},
            {"messages": [{"role": "assistant", "content": "**B**"}]},
            {"messages": [{"role": "assistant", "content": "**A**"}]},
            {"messages": [{"role": "assistant", "content": "**B**"}]},
        ]
        # Should not raise; should return n items
        result = balance_buckets._sample_stratified(two_group_examples, 2, seed=42)
        self.assertEqual(len(result), 2)

    def test_stratified_falls_back_to_random_when_too_many_groups(self):
        """If num_groups > n, can't give everyone 1, fall back to random."""
        # 10 groups, 5 examples, target 3 — can't do min-1-per-group
        many_groups = [
            {"messages": [{"role": "assistant", "content": f"**G{i}**"}]}
            for i in range(5)
        ]
        result = balance_buckets._sample_stratified(many_groups, 3, seed=42)
        self.assertEqual(len(result), 3)


class TestOutputPathResolution(unittest.TestCase):
    """Verify resolve_output_path() in train_template.py (v0.2.2+).

    The new behavior: --output gets a timestamp suffix by default so
    re-runs don't clobber. --no-timestamp is opt-out. --force is
    required to overwrite a completed run.
    """

    def setUp(self):
        # train_template is a big module with heavy imports (torch,
        # transformers, peft). We import it lazily here to keep the
        # other tests fast.
        import importlib
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        if "train_template" in sys.modules:
            self.train_template = importlib.reload(sys.modules["train_template"])
        else:
            self.train_template = importlib.import_module("train_template")

    def test_plain_name_gets_timestamp(self):
        """models/foo → models/foo_YYYY-MM-DD_HH-MM (when foo doesn't exist)"""
        out = self.train_template.resolve_output_path(
            "/tmp/never_existed_test_xyz", no_timestamp=False, force=False
        )
        # Should end in _YYYY-MM-DD_HH-MM
        import re

        self.assertRegex(Path(out).name, r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}$")

    def test_existing_timestamp_preserved(self):
        """models/foo_2026-06-10_12-00 stays as-is."""
        out = self.train_template.resolve_output_path(
            "/tmp/foo_2026-06-10_12-00", no_timestamp=False, force=False
        )
        self.assertEqual(out, str(Path("/tmp/foo_2026-06-10_12-00").resolve()))

    def test_no_timestamp_with_completed_dir_refuses(self):
        """--no-timestamp refuses to clobber a completed run."""
        # models/attacklm-3b_16g is a real completed run in the repo
        with self.assertRaises(FileExistsError) as ctx:
            self.train_template.resolve_output_path(
                "models/attacklm-3b_16g",
                no_timestamp=True,
                force=False,
            )
        self.assertIn("Refusing to clobber", str(ctx.exception))

    def test_no_timestamp_with_force_allows(self):
        """--no-timestamp + --force on a completed run is allowed."""
        out = self.train_template.resolve_output_path(
            "models/attacklm-3b_16g",
            no_timestamp=True,
            force=True,
        )
        self.assertTrue(
            out.endswith("models/attacklm-3b_16g") or "attacklm-3b_16g" in out
        )


class TestCapResolution(unittest.TestCase):
    """Verify per-bucket cap resolution for each profile type."""

    def setUp(self):
        # Use a synthetic bucket list (don't depend on real manifest)
        self.buckets = [
            {"path": "tiny", "category": "meta", "count": 50},
            {"path": "small", "category": "tactic", "count": 200},
            {"path": "medium", "category": "tactic", "count": 1000},
            {"path": "huge", "category": "tools", "count": 8000},
        ]

    def test_named_profile_uniform_cap(self):
        args = type("A", (), {"per_bucket_cap": None, "target_total": None})()
        caps = balance_buckets._resolve_caps(
            "3b-16gb", balance_buckets.PROFILES["3b-16gb"], args, self.buckets
        )
        # Cap is 800, capped at each bucket's size
        self.assertEqual(caps["tiny"], 50)
        self.assertEqual(caps["small"], 200)
        self.assertEqual(caps["medium"], 800)  # capped
        self.assertEqual(caps["huge"], 800)  # capped

    def test_full_profile_uncaps(self):
        args = type("A", (), {"per_bucket_cap": None, "target_total": None})()
        caps = balance_buckets._resolve_caps(
            "full", balance_buckets.PROFILES["full"], args, self.buckets
        )
        self.assertEqual(caps["huge"], 8000)  # not capped

    def test_target_total_respects_category_shares(self):
        # The default shares are 50/25/15/10 but this synthetic bucket
        # set has no ai_redteam buckets. That means ai_redteam's 15%
        # share immediately overshoots (its avail is 0) and gets
        # redistributed to the categories that DO have room. So we
        # don't expect strict 50/25/15/10 in the output.
        balance_buckets._CATEGORY_SHARES_OVERRIDE = None
        caps = balance_buckets._caps_for_target_total(5000, self.buckets)
        # tiny (meta) is kept whole
        self.assertEqual(caps["tiny"], 50)
        # tactic (small + medium) is capped at its total size (1200)
        self.assertEqual(caps["small"] + caps["medium"], 1200)
        # tools (huge) absorbs the redistribution from capped categories
        # — it must be the bulk of the remaining budget
        self.assertGreater(caps["huge"], 1500)
        # Total cap should not exceed target_total by much (rounding slack)
        self.assertLessEqual(sum(caps.values()), 5000 + 50)

    def test_target_total_custom_shares(self):
        balance_buckets._CATEGORY_SHARES_OVERRIDE = {
            "tactic": 0.3,
            "tools": 0.5,
            "ai_redteam": 0.0,
            "meta": 0.2,
        }
        # Synthetic buckets: no ai_redteam. So shares 0.3/0.5/0.0/0.2:
        # - tiny (meta, 50) is whole, rest of meta's 0.2 share (790-50=740)
        #   redistributes elsewhere
        # - tactic (small+medium, 1200) gets 0.3*3950=1185, capped at 1200
        # - tools (huge, 8000) gets 0.5*3950=1975, plus redistribution
        caps = balance_buckets._caps_for_target_total(4000, self.buckets)
        self.assertEqual(caps["tiny"], 50)
        self.assertLessEqual(caps["small"] + caps["medium"], 1200)
        # tools should be the dominant category with custom shares favoring it
        self.assertGreater(caps["huge"], 1975)
        balance_buckets._CATEGORY_SHARES_OVERRIDE = None


class TestIntegration(unittest.TestCase):
    """End-to-end test using the real bucket manifest."""

    def test_balance_3b_16gb_produces_correct_total(self):
        """The 3b-16gb profile should give ~7-12K pairs and respect all caps."""
        args = type(
            "A",
            (),
            {
                "per_bucket_cap": None,
                "target_total": None,
                "category_shares": None,
                "strategy": "stratified",
                "seed": 42,
            },
        )()
        selected, stats = balance_buckets.balance("3b-16gb", args)
        # Total is data-dependent; assert a sane window and that caps hold.
        total = stats["totals"]["selected"]
        self.assertGreaterEqual(total, 7000)
        self.assertLessEqual(total, 12000)
        # No bucket should exceed its cap
        for b in stats["per_bucket"]:
            self.assertLessEqual(b["selected"], b["cap"])
            self.assertLessEqual(b["selected"], b["available"])

    def test_balance_with_specific_buckets(self):
        """--buckets filter should restrict to just those buckets."""
        from bucket_loader import resolve_dataset_specs

        args = type(
            "A",
            (),
            {
                "per_bucket_cap": None,
                "target_total": None,
                "category_shares": None,
                "strategy": "head",
                "seed": 42,
            },
        )()
        resolved = resolve_dataset_specs(["tools/metasploit"])
        selected, stats = balance_buckets.balance("3b-16gb", args, buckets=resolved)
        # Only one bucket in the result
        self.assertEqual(len(stats["per_bucket"]), 1)
        self.assertEqual(stats["per_bucket"][0]["path"], "tools/metasploit")
        # Cap of 800 should give us 800 examples
        self.assertEqual(stats["per_bucket"][0]["selected"], 800)


class TestCLIOutput(unittest.TestCase):
    """Verify the CLI writes valid JSONL that can be re-read."""

    def test_write_and_read_back(self):
        from bucket_loader import list_buckets

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "balanced.jsonl"
            args = type(
                "A",
                (),
                {
                    "per_bucket_cap": None,
                    "target_total": None,
                    "category_shares": None,
                    "strategy": "stratified",
                    "seed": 42,
                    "buckets": None,
                },
            )()
            selected, _ = balance_buckets.balance("3b-16gb", args)
            balance_buckets._write_jsonl(selected, output_path)

            # Read it back
            with open(output_path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), len(selected))
            # Each line is valid JSON
            import json

            for line in lines:
                obj = json.loads(line)
                self.assertIn("messages", obj)


if __name__ == "__main__":
    unittest.main(verbosity=2)
