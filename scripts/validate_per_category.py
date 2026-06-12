#!/usr/bin/env python3
"""
validate_per_category.py — Validate dataset quality per category and
compare against target composition.

Loads all bucket data plus synthetic data, groups by category, computes
quality metrics, and flags categories that are significantly under or
over their target composition.

Usage:
    python scripts/validate_per_category.py              # Full report
    python scripts/validate_per_category.py --json-only    # JSON only (no table)
    python scripts/validate_per_category.py --threshold 0.05  # Flag if >5% deviation
"""

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
BUCKETS_DIR = PROJECT_DIR / "data" / "datasets" / "buckets"
SYNTHETIC_DIR = PROJECT_DIR / "data" / "datasets" / "synthetic"
OUTPUT_REPORT = PROJECT_DIR / "data" / "validation_report.json"

# ---------------------------------------------------------------------------
# Known MITRE technique IDs per domain
# ---------------------------------------------------------------------------
# This is a curated list of the most common MITRE ATT&CK technique IDs
# for each category. Used to compute MITRE ID coverage.
KNOWN_TECHNIQUES: dict[str, set[str]] = {
    "web_app": {
        "T1190",
        "T1189",
        "T1059.001",
        "T1059.003",
        "T1059.005",
        "T1071.001",
        "T1078",
        "T1110",
        "T1552.001",
        "T1070.004",
    },
    "cloud": {
        "T1078",
        "T1078.004",
        "T1611",
        "T1610",
        "T1552.005",
        "T1537",
        "T1530",
        "T1087.004",
        "T1098.001",
        "T1098.003",
    },
    "supply_chain": {
        "T1195",
        "T1195.001",
        "T1195.002",
        "T1059",
        "T1078",
        "T1195.003",
    },
    "social_engineering": {
        "T1566",
        "T1566.001",
        "T1566.002",
        "T1566.003",
        "T1598",
        "T1598.001",
        "T1598.002",
        "T1598.003",
    },
    "ics_scada": {
        "T0831",
        "T0832",
        "T0857",
        "T0859",
        "T0867",
        "T0842",
        "T0855",
        "T0883",
        "T0882",
        "T0866",
    },
    "wireless": {
        "T1595.001",
        "T1590.001",
        "T1546",
        "T1025",
        "T1046",
    },
    "persistence": {
        "T1543",
        "T1547",
        "T1136",
        "T1098",
        "T1197",
        "T1053",
        "T1543.003",
        "T1547.001",
        "T1547.004",
        "T1136.001",
        "T1053.005",
    },
    "execution": {
        "T1059",
        "T1059.001",
        "T1059.003",
        "T1059.005",
        "T1106",
        "T1204",
        "T1047",
        "T1059.004",
    },
    "privilege_escalation": {
        "T1068",
        "T1548",
        "T1134",
        "T1543",
        "T1548.001",
        "T1548.002",
        "T1134.001",
        "T1134.002",
    },
    "defense_evasion": {
        "T1027",
        "T1055",
        "T1562",
        "T1070",
        "T1027.001",
        "T1055.001",
        "T1055.003",
        "T1562.001",
    },
    "credential_access": {
        "T1003",
        "T1110",
        "T1555",
        "T1552",
        "T1003.001",
        "T1003.002",
        "T1558.001",
    },
    "discovery": {
        "T1057",
        "T1082",
        "T1087",
        "T1018",
        "T1049",
        "T1046",
        "T1087.001",
        "T1087.002",
    },
    "lateral_movement": {
        "T1021",
        "T1570",
        "T1080",
        "T1559",
        "T1550.002",
        "T1021.001",
        "T1021.002",
        "T1021.004",
    },
    "command_and_control": {
        "T1071",
        "T1071.001",
        "T1071.004",
        "T1090",
        "T1095",
        "T1572",
        "T1573",
    },
    "collection": {
        "T1005",
        "T1213",
        "T1114",
        "T1119",
    },
    "exfiltration": {
        "T1041",
        "T1567",
        "T1052",
        "T1011",
    },
    "ai_redteam": {
        "TA0040",
    },
    "tools": {
        "S0001",
        "S0002",
        "S0029",
        "S0050",
        "S0060",
    },
    "meta": set(),
}

# ---------------------------------------------------------------------------
# Target composition (percentages)
# ---------------------------------------------------------------------------
TARGET_COMPOSITION: dict[str, float] = {
    "web_app": 0.15,
    "identity": 0.10,
    "cloud": 0.10,
    "social_engineering": 0.08,
    "supply_chain": 0.06,
    "wireless": 0.04,
    "physical": 0.04,
    "ics": 0.05,
    "tactic": 0.15,
    "tools": 0.10,
    "ai_specific": 0.08,
    "meta": 0.05,
}

# Mapping from bucket paths to our category keys
BUCKET_TO_CATEGORY: dict[str, str] = {
    "base/collection": "tactic",
    "base/command_and_control": "tactic",
    "base/credential_access": "tactic",
    "base/defense_evasion": "tactic",
    "base/discovery": "tactic",
    "base/execution": "tactic",
    "base/exfiltration": "tactic",
    "base/lateral_movement": "tactic",
    "base/persistence": "tactic",
    "base/privilege_escalation": "tactic",
    "ai/jailbreaking": "ai_specific",
    "ai/prompt-injection": "ai_specific",
    "tools/infection_monkey": "tools",
    "tools/metasploit": "tools",
    "tools/rta": "tools",
    "orchestrator": "meta",
    "attack_tactics/red_team_tactics": "attack_tactics",
    "web_app/attacks": "web_app",
    "social_engineering/phishing": "social_engineering",
    "cloud/attacks": "cloud",
    "supply_chain/attacks": "supply_chain",
    "ics/attacks": "ics",
    "wireless/attacks": "wireless",
}

# Mapping from synthetic directory names to category keys
SYNTHETIC_TO_CATEGORY: dict[str, str] = {
    "web_app": "web_app",
    "cloud": "cloud",
    "supply_chain": "supply_chain",
    "social_engineering": "social_engineering",
    "ics_scada": "ics",
    "wireless": "wireless",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer with basic normalization."""
    return re.findall(r"\b\w+\b", text.lower())


def count_tokens_approx(text: str) -> int:
    """Approximate token count (words * 1.3 heuristic for subwords)."""
    words = len(tokenize(text))
    return int(words * 1.3)


def extract_mitre_ids(text: str) -> list[str]:
    """Extract all MITRE technique IDs from text."""
    return re.findall(r"T\d{4}(?:\.\d{3})?", text)


def has_code_block(text: str) -> bool:
    """Check if text contains a fenced code block."""
    return bool(re.search(r"```\w*\n", text) or re.search(r"```\n", text))


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, returning a list of parsed dicts."""
    entries: list[dict] = []
    if not path.exists():
        return entries
    with open(path, encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  WARNING: Invalid JSON at {path}:{line_num}: {exc}")
    return entries


def load_all_buckets() -> dict[str, list[dict]]:
    """Load all bucket data, grouped by category key."""
    categories: dict[str, list[dict]] = defaultdict(list)
    manifest_path = BUCKETS_DIR / "manifest.json"

    if not manifest_path.exists():
        print(f"  WARNING: No manifest at {manifest_path}", file=sys.stderr)
        return categories

    with open(manifest_path) as f:
        manifest = json.load(f)

    for bucket in manifest.get("buckets", []):
        bucket_path = bucket["path"]
        category_key = BUCKET_TO_CATEGORY.get(bucket_path, "tactic")

        # In the per-source layout (v0.3.0+), aggregate from all sources
        from pathlib import Path
        from bucket_loader import SOURCES_DIR

        candidates: list[Path] = []
        if SOURCES_DIR.exists():
            for src_dir in SOURCES_DIR.iterdir():
                if not src_dir.is_dir() or src_dir.name.startswith("_"):
                    continue
                p2 = src_dir / bucket_path
                if p2.is_dir():
                    candidates.extend(p2.glob("*.jsonl"))
                p1 = src_dir / bucket_path.split("/")[-1]
                if p1.is_dir() and p1 != p2:
                    candidates.extend(p1.glob("*.jsonl"))
        entries: list[dict] = []
        for jsonl in sorted(set(candidates)):
            entries.extend(load_jsonl(jsonl))

        # Tag each entry with the bucket category for downstream grouping
        for entry in entries:
            entry["_bucket_category"] = category_key
            entry["_bucket_path"] = bucket_path

        categories[category_key].extend(entries)

    return categories


def load_all_synthetic() -> dict[str, list[dict]]:
    """Load all synthetic data, grouped by category key."""
    categories: dict[str, list[dict]] = defaultdict(list)

    if not SYNTHETIC_DIR.exists():
        return categories

    for jsonl_file in SYNTHETIC_DIR.glob("*.jsonl"):
        # Extract category from filename: e.g., "web_app_synthetic.jsonl" → "web_app"
        stem = jsonl_file.stem.replace("_synthetic", "")
        category_key = SYNTHETIC_TO_CATEGORY.get(stem, stem)

        entries = load_jsonl(jsonl_file)
        for entry in entries:
            entry["_bucket_category"] = category_key
            entry["_source"] = entry.get("source", f"synthetic_{stem}")

        categories[category_key].extend(entries)

    return categories


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------
def compute_category_metrics(
    category_key: str,
    entries: list[dict],
) -> dict:
    """Compute quality metrics for a single category."""
    total_pairs = len(entries)
    if total_pairs == 0:
        return {
            "total_pairs": 0,
            "unique_mitre_ids": [],
            "mitre_id_count": 0,
            "mitre_coverage_pct": 0.0,
            "avg_tokens_per_message": 0.0,
            "language_diversity_score": 0.0,
            "code_block_density_pct": 0.0,
        }

    # MITRE IDs
    all_mitre_ids: set[str] = set()
    for entry in entries:
        # Check mitre_ids field first (synthetic data)
        if "mitre_ids" in entry:
            ids = entry["mitre_ids"]
            if isinstance(ids, list):
                all_mitre_ids.update(ids)
        # Also extract from content
        messages = entry.get("messages", [])
        for msg in messages:
            content = msg.get("content", "")
            all_mitre_ids.update(extract_mitre_ids(content))

    # Known techniques for this category
    known = KNOWN_TECHNIQUES.get(category_key, set())
    mitre_coverage_pct = 0.0
    if known:
        covered = all_mitre_ids & known
        mitre_coverage_pct = round(len(covered) / len(known) * 100, 1)

    # Average tokens per message
    total_tokens = 0
    message_count = 0
    for entry in entries:
        messages = entry.get("messages", [])
        for msg in messages:
            content = msg.get("content", "")
            total_tokens += count_tokens_approx(content)
            message_count += 1
    avg_tokens = round(total_tokens / max(message_count, 1), 1)

    # Language diversity (unique words / total words)
    all_words: list[str] = []
    for entry in entries:
        messages = entry.get("messages", [])
        for msg in messages:
            content = msg.get("content", "")
            all_words.extend(tokenize(content))

    total_words = len(all_words)
    unique_words = len(set(all_words)) if all_words else 0
    language_diversity = round(unique_words / max(total_words, 1), 4)

    # Code block density (% of assistant messages containing code blocks)
    assistant_with_code = 0
    total_assistant = 0
    for entry in entries:
        messages = entry.get("messages", [])
        for msg in messages:
            if msg.get("role") == "assistant":
                total_assistant += 1
                if has_code_block(msg.get("content", "")):
                    assistant_with_code += 1

    code_block_density = round(assistant_with_code / max(total_assistant, 1) * 100, 1)

    return {
        "total_pairs": total_pairs,
        "unique_mitre_ids": sorted(all_mitre_ids),
        "mitre_id_count": len(all_mitre_ids),
        "mitre_coverage_pct": mitre_coverage_pct,
        "avg_tokens_per_message": avg_tokens,
        "language_diversity_score": language_diversity,
        "code_block_density_pct": code_block_density,
    }


def compute_composition_flags(
    categories: dict[str, list[dict]],
    threshold: float = 0.05,
) -> list[dict]:
    """Compare actual composition against targets and flag deviations.

    threshold: float in 0-1 range (e.g., 0.05 = 5%).
    Flags categories where actual % differs from target by more than threshold.
    """
    total_pairs = sum(len(entries) for entries in categories.values())
    if total_pairs == 0:
        return []

    flags: list[dict] = []
    for cat_key, target_pct in TARGET_COMPOSITION.items():
        actual_count = len(categories.get(cat_key, []))
        actual_pct = actual_count / total_pairs
        deviation = actual_pct - target_pct

        flag = None
        if deviation > threshold:
            flag = "OVER_REPRESENTED"
        elif deviation < -threshold:
            flag = "UNDER_REPRESENTED"

        flags.append(
            {
                "category": cat_key,
                "target_pct": round(target_pct * 100, 1),
                "actual_pct": round(actual_pct * 100, 1),
                "actual_count": actual_count,
                "deviation_pct": round(deviation * 100, 1),
                "flag": flag,
            }
        )

    return flags


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(
    categories: dict[str, list[dict]],
    threshold: float = 0.05,
) -> dict:
    """Generate the full validation report."""
    report: dict = {
        "generated_at": datetime.now().isoformat(),
        "total_pairs": 0,
        "categories": {},
        "composition_flags": [],
        "summary": {},
    }

    # Per-category metrics
    total_pairs = 0
    for cat_key, entries in categories.items():
        metrics = compute_category_metrics(cat_key, entries)
        report["categories"][cat_key] = metrics
        total_pairs += metrics["total_pairs"]

    report["total_pairs"] = total_pairs

    # Composition flags
    report["composition_flags"] = compute_composition_flags(categories, threshold)

    # Summary stats
    all_mitre_ids: set[str] = set()
    total_tokens = 0
    total_messages = 0
    for cat_key, metrics in report["categories"].items():
        all_mitre_ids.update(metrics["unique_mitre_ids"])
        total_tokens += metrics["avg_tokens_per_message"] * metrics["total_pairs"] * 3
        total_messages += metrics["total_pairs"] * 3

    report["summary"] = {
        "total_pairs": total_pairs,
        "total_categories": len(categories),
        "unique_mitre_ids_total": len(all_mitre_ids),
        "avg_tokens_per_pair": round(total_tokens / max(total_messages, 1) * 3, 1),
        "under_represented_categories": [
            f["category"]
            for f in report["composition_flags"]
            if f["flag"] == "UNDER_REPRESENTED"
        ],
        "over_represented_categories": [
            f["category"]
            for f in report["composition_flags"]
            if f["flag"] == "OVER_REPRESENTED"
        ],
    }

    return report


def print_human_readable(report: dict) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 80)
    print("AttackLM Dataset Validation Report")
    print("=" * 80)
    print(f"Generated: {report['generated_at']}")
    print(f"Total pairs: {report['total_pairs']:,}")
    print(f"Categories: {report['summary']['total_categories']}")
    print(f"Unique MITRE IDs: {report['summary']['unique_mitre_ids_total']}")
    print()

    # Per-category table
    print(
        f"{'Category':<20} {'Pairs':>7} {'MITRE IDs':>10} {'Coverage':>9} "
        f"{'Avg Tok':>8} {'Diversity':>9} {'Code %':>7}"
    )
    print("-" * 80)
    for cat_key, metrics in report["categories"].items():
        print(
            f"{cat_key:<20} "
            f"{metrics['total_pairs']:>7,} "
            f"{metrics['mitre_id_count']:>10} "
            f"{metrics['mitre_coverage_pct']:>8.1f}% "
            f"{metrics['avg_tokens_per_message']:>8.1f} "
            f"{metrics['language_diversity_score']:>9.4f} "
            f"{metrics['code_block_density_pct']:>6.1f}%"
        )

    # Composition flags
    print()
    print("=" * 80)
    print("Composition vs Target")
    print("=" * 80)
    print(
        f"{'Category':<20} {'Target':>8} {'Actual':>8} {'Deviation':>10} {'Flag':>20}"
    )
    print("-" * 80)

    flagged = []
    for flag in report["composition_flags"]:
        flag_str = flag["flag"] or "OK"
        print(
            f"{flag['category']:<20} "
            f"{flag['target_pct']:>7.1f}% "
            f"{flag['actual_pct']:>7.1f}% "
            f"{flag['deviation_pct']:>+9.1f}% "
            f"{flag_str:>20}"
        )
        if flag["flag"]:
            flagged.append(flag)

    if flagged:
        print()
        print("⚠️  FLAGGED CATEGORIES:")
        for flag in flagged:
            direction = (
                "under-represented"
                if flag["flag"] == "UNDER_REPRESENTED"
                else "over-represented"
            )
            print(
                f"  • {flag['category']}: {direction} by "
                f"{abs(flag['deviation_pct']):.1f}% "
                f"(target: {flag['target_pct']:.1f}%, actual: {flag['actual_pct']:.1f}%)"
            )
    else:
        print("\n✅ All categories within target composition thresholds.")

    print()
    print(f"Report saved to: {OUTPUT_REPORT}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate AttackLM dataset quality per category."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Flag categories that deviate from target by more than this %% (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        default=False,
        help="Only output JSON report (no human-readable table)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for JSON report (default: data/validation_report.json)",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else OUTPUT_REPORT

    print("AttackLM Dataset Validator")
    print(f"Buckets dir:   {BUCKETS_DIR}")
    print(f"Synthetic dir: {SYNTHETIC_DIR}")
    print(f"Threshold:     {args.threshold * 100:.1f}%")
    print()

    # Load data
    print("Loading bucket data...")
    categories = load_all_buckets()

    print("Loading synthetic data...")
    synthetic = load_all_synthetic()

    # Merge synthetic into categories
    for cat_key, entries in synthetic.items():
        if cat_key in categories:
            categories[cat_key].extend(entries)
        else:
            categories[cat_key] = entries

    # Print loaded counts
    for cat_key, entries in sorted(categories.items()):
        bucket_count = sum(1 for e in entries if "_bucket_path" in e)
        synthetic_count = sum(
            1 for e in entries if "_source" in e and "synthetic" in e.get("_source", "")
        )
        print(
            f"  {cat_key}: {len(entries)} total ({bucket_count} bucket, {synthetic_count} synthetic)"
        )

    print()

    # Generate report
    report = generate_report(categories, threshold=args.threshold)

    # Save JSON report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # Print human-readable
    if not args.json_only:
        print_human_readable(report)
    else:
        print(f"JSON report saved to: {output_path}")
        print(f"Total pairs: {report['total_pairs']:,}")
        print(f"Categories: {report['summary']['total_categories']}")


if __name__ == "__main__":
    main()
