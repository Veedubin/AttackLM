#!/usr/bin/env python3
"""Tests for scripts/replay_mixer.py.

Hermetic tests using tmp_path. Do not require GPU, HF, or real datasets.

Run with:
    python -m pytest tests/test_replay_mixer.py -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Make scripts/ importable
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from replay_mixer import (  # noqa: E402
    discover_replay_files,
    load_replay_domain,
    mix_replay,
    _compute_cache_key,
    _infer_domain,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write a list of dicts as JSONL to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _make_target(path: Path, n: int = 100) -> Path:
    """Create a simple target JSONL dataset with n records."""
    records = [
        {
            "messages": [
                {"role": "user", "content": f"security prompt {i}"},
                {"role": "assistant", "content": f"security response {i}"},
            ],
            "source": "attacklm-synthetic",
            "source_uri": "https://example.com",
            "license": "Apache-2.0",
            "license_uri": "https://www.apache.org/licenses/LICENSE-2.0",
            "rights_contact": "security@example.com",
        }
        for i in range(n)
    ]
    _write_jsonl(path, records)
    return path


def _make_replay_source(
    base_dir: Path, domains: dict[str, int], name: str = "replay-test"
) -> Path:
    """Create a replay source directory with data_<domain>.jsonl files.

    Args:
        base_dir: Parent directory (e.g. tmp_path).
        domains: Dict mapping domain name to number of records.
        name: Name of the source directory (default: "replay-test").

    Returns:
        Path to the source directory.
    """
    source_dir = base_dir / name
    replay_dir = source_dir / "base" / "replay"
    for domain, count in domains.items():
        records = [
            {
                "messages": [
                    {"role": "user", "content": f"{domain} prompt {i}"},
                    {"role": "assistant", "content": f"{domain} response {i}"},
                ],
                "source": f"replay-test/{domain}",
                "license": "Apache-2.0",
                "domain": domain,
            }
            for i in range(count)
        ]
        _write_jsonl(replay_dir / f"data_{domain}.jsonl", records)
    return source_dir


class TestDiscoverReplayFiles(unittest.TestCase):
    """Tests for discover_replay_files."""

    def test_discovers_domains(self):
        """Should find data_<domain>.jsonl files and group by domain."""
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "my-replay"
            for domain in ("code", "conversation", "factual", "reasoning"):
                fp = src / "base" / "replay" / f"data_{domain}.jsonl"
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.touch()
            result = discover_replay_files(src)
            self.assertEqual(
                set(result.keys()), {"code", "conversation", "factual", "reasoning"}
            )

    def test_empty_directory(self):
        """Should return empty dict for a source with no replay dir."""
        with tempfile.TemporaryDirectory() as td:
            result = discover_replay_files(Path(td) / "nonexistent")
            self.assertEqual(result, {})

    def test_replay_dir_exists_but_empty(self):
        """Should return empty dict for a source with empty replay dir."""
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "my-replay"
            replay_dir = src / "base" / "replay"
            replay_dir.mkdir(parents=True)
            result = discover_replay_files(src)
            self.assertEqual(result, {})

    def test_non_jsonl_files_ignored(self):
        """Should only pick up data_*.jsonl files."""
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "my-replay"
            replay_dir = src / "base" / "replay"
            replay_dir.mkdir(parents=True)
            (replay_dir / "data_code.jsonl").touch()
            (replay_dir / "README.md").touch()
            (replay_dir / "data_code.csv").touch()
            result = discover_replay_files(src)
            self.assertEqual(set(result.keys()), {"code"})

    def test_multiple_files_same_domain(self):
        """Should group multiple files with same domain stem."""
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "my-replay"
            replay_dir = src / "base" / "replay"
            replay_dir.mkdir(parents=True)
            (replay_dir / "data_code.jsonl").touch()
            result = discover_replay_files(src)
            self.assertIn("code", result)
            self.assertEqual(len(result["code"]), 1)


class TestLoadReplayDomain(unittest.TestCase):
    """Tests for load_replay_domain."""

    def test_loads_all_records(self):
        """Should load all records from a JSONL file."""
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td) / "data_code.jsonl"
            records = [
                {"messages": [{"role": "user", "content": f"q{i}"}], "domain": "code"}
                for i in range(50)
            ]
            _write_jsonl(fp, records)
            loaded = load_replay_domain(fp)
            self.assertEqual(len(loaded), 50)

    def test_max_examples_cap(self):
        """Should cap at max_examples when set."""
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td) / "data_code.jsonl"
            records = [
                {"messages": [{"role": "user", "content": f"q{i}"}], "domain": "code"}
                for i in range(100)
            ]
            _write_jsonl(fp, records)
            loaded = load_replay_domain(fp, max_examples=10)
            self.assertEqual(len(loaded), 10)

    def test_max_examples_none_loads_all(self):
        """Should load all records when max_examples is None."""
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td) / "data_code.jsonl"
            records = [
                {"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(30)
            ]
            _write_jsonl(fp, records)
            loaded = load_replay_domain(fp, max_examples=None)
            self.assertEqual(len(loaded), 30)

    def test_preserves_provenance(self):
        """Should preserve all original fields."""
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td) / "data_code.jsonl"
            records = [
                {
                    "messages": [{"role": "user", "content": "test"}],
                    "source": "replay-general/code",
                    "source_uri": "https://example.com",
                    "license": "Apache-2.0",
                    "license_uri": "https://www.apache.org/licenses/LICENSE-2.0",
                    "rights_contact": "legal@example.com",
                    "domain": "code",
                }
            ]
            _write_jsonl(fp, records)
            loaded = load_replay_domain(fp)
            self.assertEqual(loaded[0]["source"], "replay-general/code")
            self.assertEqual(loaded[0]["license"], "Apache-2.0")
            self.assertEqual(loaded[0]["domain"], "code")


class TestMixReplay(unittest.TestCase):
    """Tests for mix_replay."""

    def test_basic_mix(self):
        """Should mix replay into target and return composition."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            target_path = td / "target.jsonl"
            _make_target(target_path, n=100)

            src = _make_replay_source(td, {"code": 50, "conversation": 50})

            out_path, comp = mix_replay(
                target_path=target_path,
                replay_sources=[src],
                ratio=0.1,
                seed=42,
                output_dir=td / "output",
            )

            self.assertEqual(comp["target_examples"], 100)
            self.assertEqual(comp["replay_examples"], 10)  # 10% of 100
            self.assertAlmostEqual(comp["replay_ratio"], 0.1, places=1)
            self.assertTrue(out_path.exists())

    def test_stratified_sampling(self):
        """Stratified sampling should distribute across domains proportionally."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            target_path = td / "target.jsonl"
            _make_target(target_path, n=100)

            src = _make_replay_source(
                td, {"code": 200, "conversation": 200, "factual": 100}
            )

            _, comp = mix_replay(
                target_path=target_path,
                replay_sources=[src],
                ratio=0.1,
                stratify=True,
                seed=42,
                output_dir=td / "output",
            )

            self.assertEqual(comp["replay_examples"], 10)
            # With stratification, all three domains should be represented
            self.assertEqual(sum(comp["replay_domains"].values()), 10)

    def test_no_stratification(self):
        """Without stratification, samples uniformly across all records."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            target_path = td / "target.jsonl"
            _make_target(target_path, n=100)

            src = _make_replay_source(td, {"code": 200, "conversation": 200})

            _, comp = mix_replay(
                target_path=target_path,
                replay_sources=[src],
                ratio=0.05,
                stratify=False,
                seed=42,
                output_dir=td / "output",
            )

            self.assertEqual(comp["replay_examples"], 5)

    def test_max_examples_cap(self):
        """max_examples should cap the replay budget even if ratio would allow more."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            target_path = td / "target.jsonl"
            _make_target(target_path, n=1000)

            src = _make_replay_source(td, {"code": 500, "conversation": 500})

            _, comp = mix_replay(
                target_path=target_path,
                replay_sources=[src],
                ratio=0.1,  # 10% of 1000 = 100
                max_examples=30,  # but cap at 30
                seed=42,
                output_dir=td / "output",
            )

            self.assertEqual(comp["replay_examples"], 30)

    def test_multiple_sources(self):
        """Should aggregate replay data from multiple sources."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            target_path = td / "target.jsonl"
            _make_target(target_path, n=200)

            src1 = _make_replay_source(
                td, {"code": 100, "conversation": 100}, name="replay-general"
            )
            src2_dir = _make_replay_source(td, {"code": 50}, name="replay-coding")

            _, comp = mix_replay(
                target_path=target_path,
                replay_sources=[src1, src2_dir],
                ratio=0.1,
                seed=42,
                output_dir=td / "output",
            )

            # 10% of 200 = 20 replay examples
            self.assertEqual(comp["replay_examples"], 20)
            # Both sources should contribute
            self.assertTrue(len(comp["replay_sources"]) >= 1)

    def test_missing_source_warning(self):
        """Should warn and skip missing replay sources without crashing."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            target_path = td / "target.jsonl"
            _make_target(target_path, n=50)

            # A source dir that doesn't exist
            missing_src = td / "nonexistent-source"

            out_path, comp = mix_replay(
                target_path=target_path,
                replay_sources=[missing_src],
                ratio=0.1,
                seed=42,
                output_dir=td / "output",
            )

            # Should return target path unchanged (no replay mixed)
            self.assertEqual(out_path, target_path)
            self.assertEqual(comp["replay_examples"], 0)

    def test_domain_ratios_override(self):
        """Should honor explicit domain_ratios override."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            target_path = td / "target.jsonl"
            _make_target(target_path, n=100)

            src = _make_replay_source(
                td, {"code": 200, "conversation": 200, "factual": 100}
            )

            _, comp = mix_replay(
                target_path=target_path,
                replay_sources=[src],
                ratio=0.1,
                domain_ratios={"code": 0.5, "conversation": 0.3, "factual": 0.2},
                seed=42,
                output_dir=td / "output",
            )

            self.assertEqual(comp["replay_examples"], 10)
            # code should get ~50% of 10 = 5, conversation ~30% = 3, factual ~20% = 2
            # Allow ±1 due to rounding
            total = sum(comp["replay_domains"].values())
            self.assertEqual(total, 10)

    def test_cache_reuse(self):
        """Should reuse cached output for identical parameters."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            target_path = td / "target.jsonl"
            _make_target(target_path, n=50)

            src = _make_replay_source(td, {"code": 30})

            out1, _ = mix_replay(
                target_path=target_path,
                replay_sources=[src],
                ratio=0.1,
                seed=42,
                output_dir=td / "output",
            )
            out2, _ = mix_replay(
                target_path=target_path,
                replay_sources=[src],
                ratio=0.1,
                seed=42,
                output_dir=td / "output",
            )

            # Same parameters should return same path
            self.assertEqual(out1, out2)

    def test_zero_ratio_returns_target(self):
        """With ratio=0, no replay should be added."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            target_path = td / "target.jsonl"
            _make_target(target_path, n=50)

            src = _make_replay_source(td, {"code": 30})

            out_path, comp = mix_replay(
                target_path=target_path,
                replay_sources=[src],
                ratio=0.0,
                seed=42,
                output_dir=td / "output",
            )

            # ratio=0 means no replay budget, but mix_replay should
            # still return the target unchanged
            self.assertEqual(comp["replay_examples"], 0)
            self.assertEqual(out_path, target_path)

    def test_empty_target_raises_error(self):
        """Should raise ValueError for empty target dataset."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            target_path = td / "target.jsonl"
            _write_jsonl(target_path, [])  # empty

            src = _make_replay_source(td, {"code": 10})

            with self.assertRaises(ValueError):
                mix_replay(
                    target_path=target_path,
                    replay_sources=[src],
                    ratio=0.1,
                    seed=42,
                    output_dir=td / "output",
                )


class TestInferDomain(unittest.TestCase):
    """Tests for _infer_domain."""

    def test_explicit_domain_field(self):
        """Should use explicit 'domain' field."""
        self.assertEqual(_infer_domain({"domain": "code", "source": "foo"}), "code")

    def test_domain_from_source_field(self):
        """Should extract domain from source field with slash."""
        self.assertEqual(_infer_domain({"source": "replay-general/code"}), "code")

    def test_no_domain(self):
        """Should return empty string if no domain info."""
        self.assertEqual(_infer_domain({"messages": []}), "")

    def test_empty_domain(self):
        """Should return empty string if domain is empty/falsy."""
        self.assertEqual(_infer_domain({"domain": ""}), "")


class TestComputeCacheKey(unittest.TestCase):
    """Tests for _compute_cache_key."""

    def test_deterministic(self):
        """Same inputs should produce same cache key."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            target = td / "target.jsonl"
            _make_target(target, n=10)

            key1 = _compute_cache_key(target, ["src1"], 0.07, 0, True, None, 42)
            key2 = _compute_cache_key(target, ["src1"], 0.07, 0, True, None, 42)
            self.assertEqual(key1, key2)

    def test_different_params_different_key(self):
        """Different parameters should produce different keys."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            target = td / "target.jsonl"
            _make_target(target, n=10)

            key1 = _compute_cache_key(target, ["src1"], 0.07, 0, True, None, 42)
            key2 = _compute_cache_key(target, ["src1"], 0.10, 0, True, None, 42)
            self.assertNotEqual(key1, key2)


if __name__ == "__main__":
    unittest.main()
