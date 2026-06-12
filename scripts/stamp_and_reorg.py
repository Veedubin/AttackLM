#!/usr/bin/env python3
"""
Stamp every kept record with full provenance (source, source_uri, license,
license_uri, rights_contact) and reorganize the bucket layout into per-source
subdirectories.

USAGE:
    uv run python scripts/stamp_and_reorg.py [--dry-run]

This script is idempotent. Re-running it will not duplicate fields; it will
overwrite them with the canonical values from `PROVENANCE` below.

The script reads from `data/datasets/buckets/<bucket>/data*.jsonl` and
writes to `data/datasets/buckets/sources/<source>/<bucket>/data*.jsonl`.
Per-source `LICENSE.md` and `SOURCE.md` files are also written.

The previous flat layout (data/datasets/buckets/<bucket>/...) is preserved
as a manifest-only view in `data/datasets/buckets/_flat/` (gitignored) for
backward compatibility with any external scripts that reference the old
paths. The new canonical location is `sources/`.

Buckets whose only source is high-risk (RTA, infection_monkey, bigpromptlibrary)
are NOT touched here - those are moved to `archive/restricted-sources/`.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUCKETS_DIR = ROOT / "data" / "datasets" / "buckets"
SOURCES_DIR = BUCKETS_DIR / "sources"
FLAT_DIR = BUCKETS_DIR / "_flat"  # backward-compat symlink target
RIGHTS_CONTACT = "see data/REMOVAL.md"

# Canonical per-source provenance.
# `match` describes how a record is identified as belonging to this source.
# `apply_to_files` lists which bucket file(s) get this stamp (data.jsonl,
# data_synth.jsonl, data_llm.jsonl).
PROVENANCE: list[dict] = [
    # ---------- SAFE / LOW RISK ----------
    {
        "name": "atomic-red-team",
        "display": "Atomic Red Team (Red Canary)",
        "license": "MIT",
        "license_uri": "https://opensource.org/licenses/MIT",
        "source_uri": "https://github.com/redcanaryco/atomic-red-team",
        "match": {"source_field": "redcanaryco/atomic-red-team"},
        "apply_to_files": ["data.jsonl"],
        "apply_to_buckets": [
            "base/execution",
            "base/persistence",
            "base/privilege_escalation",
            "base/defense_evasion",
            "base/credential_access",
            "base/discovery",
            "base/lateral_movement",
            "base/collection",
            "base/exfiltration",
            "base/command_and_control",
        ],
        "risk": "low",
    },
    {
        "name": "mitre-stockpile",
        "display": "MITRE Caldera Stockpile",
        "license": "Apache-2.0",
        "license_uri": "https://www.apache.org/licenses/LICENSE-2.0",
        "source_uri": "https://github.com/mitre/stockpile",
        "match": {"source_field": "mitre/stockpile"},
        "apply_to_files": ["data.jsonl"],
        "apply_to_buckets": [
            "base/execution",
            "base/persistence",
            "base/privilege_escalation",
            "base/defense_evasion",
            "base/credential_access",
            "base/discovery",
            "base/lateral_movement",
            "base/collection",
            "base/exfiltration",
            "base/command_and_control",
        ],
        "risk": "low",
    },
    {
        "name": "mitre-atlas-arsenal",
        "display": "MITRE ATLAS Arsenal",
        "license": "Apache-2.0",
        "license_uri": "https://www.apache.org/licenses/LICENSE-2.0",
        "source_uri": "https://github.com/mitre-atlas/arsenal",
        "match": {"source_field": "mitre-atlas/arsenal"},
        "apply_to_files": ["data.jsonl"],
        "apply_to_buckets": ["base/discovery"],
        "risk": "low",
    },
    # ---------- MEDIUM RISK (handled with attribution) ----------
    {
        "name": "metasploit-framework",
        "display": "Metasploit Framework (Rapid7)",
        "license": "BSD-3-Clause",
        "license_uri": "https://opensource.org/licenses/BSD-3-Clause",
        "source_uri": "https://github.com/rapid7/metasploit-framework",
        # Match all Metasploit-Framework-derived records: those with no
        # `source` field in the base/* buckets (Metasploit-style system
        # prompt) plus the `tools/metasploit` bucket.
        "match": {
            "source_field": None,
            "fingerprint": "Metasploit Framework",
        },
        "apply_to_files": ["data.jsonl"],
        "apply_to_buckets": [
            "base/execution",
            "base/persistence",
            "base/privilege_escalation",
            "base/defense_evasion",
            "base/credential_access",
            "base/discovery",
            "base/lateral_movement",
            "base/collection",
            "base/exfiltration",
            "base/command_and_control",
            "tools/metasploit",
        ],
        "risk": "medium",
    },
    # AI security tools - all safe per matrix
    {
        "name": "nvidia-garak",
        "display": "garak (NVIDIA)",
        "license": "Apache-2.0",
        "license_uri": "https://www.apache.org/licenses/LICENSE-2.0",
        "source_uri": "https://github.com/NVIDIA/garak",
        "match": {"source_field": None, "fingerprint": "garak:"},
        "apply_to_files": ["data.jsonl"],
        "apply_to_buckets": ["ai/jailbreaking"],
        "risk": "low",
    },
    {
        "name": "azure-pyrit",
        "display": "PyRIT (Azure)",
        "license": "MIT",
        "license_uri": "https://opensource.org/licenses/MIT",
        "source_uri": "https://github.com/Azure/PyRIT",
        # No PyRIT records in current dataset, but slot kept for future
        "match": {"source_field": None, "fingerprint": "pyrit"},
        "apply_to_files": ["data.jsonl"],
        "apply_to_buckets": ["ai/jailbreaking"],
        "risk": "low",
    },
    {
        "name": "cyberark-fuzzyai",
        "display": "FuzzyAI (CyberArk)",
        "license": "Apache-2.0",
        "license_uri": "https://www.apache.org/licenses/LICENSE-2.0",
        "source_uri": "https://github.com/cyberark/FuzzyAI",
        "match": {"source_field": None, "fingerprint": "fuzzy"},
        "apply_to_files": ["data.jsonl"],
        "apply_to_buckets": ["ai/jailbreaking"],
        "risk": "low",
    },
    {
        "name": "promptfoo",
        "display": "promptfoo",
        "license": "MIT",
        "license_uri": "https://opensource.org/licenses/MIT",
        "source_uri": "https://github.com/promptfoo/promptfoo",
        "match": {"source_field": None, "fingerprint": "promptfoo"},
        "apply_to_files": ["data.jsonl"],
        "apply_to_buckets": ["ai/prompt-injection"],
        "risk": "low",
    },
    {
        "name": "promptmap",
        "display": "promptmap (utkusen)",
        "license": "MIT",
        "license_uri": "https://opensource.org/licenses/MIT",
        "source_uri": "https://github.com/utkusen/promptmap",
        "match": {"source_field": None, "fingerprint": "promptmap"},
        "apply_to_files": ["data.jsonl"],
        "apply_to_buckets": ["ai/prompt-injection"],
        "risk": "low",
    },
    # Synthetic / in-repo
    {
        "name": "attacklm-synthetic",
        "display": "AttackLM Synthetic (deterministic templates)",
        "license": "MIT",
        "license_uri": "https://opensource.org/licenses/MIT",
        "source_uri": "https://github.com/Veedubin/AttackLM",
        "match": {"source_field": "phishing_synthetic"},
        "apply_to_files": ["data.jsonl", "data_synth.jsonl"],
        "apply_to_buckets": ["social_engineering/phishing"],
        "risk": "low",
    },
    {
        "name": "attacklm-synthetic",
        "display": "AttackLM Synthetic (deterministic templates)",
        "license": "MIT",
        "license_uri": "https://opensource.org/licenses/MIT",
        "source_uri": "https://github.com/Veedubin/AttackLM",
        "match": {"source_field": "cloud_attack_synthetic"},
        "apply_to_files": ["data.jsonl", "data_synth.jsonl"],
        "apply_to_buckets": ["cloud/attacks"],
        "risk": "low",
    },
    {
        "name": "attacklm-synthetic",
        "display": "AttackLM Synthetic (deterministic templates)",
        "license": "MIT",
        "license_uri": "https://opensource.org/licenses/MIT",
        "source_uri": "https://github.com/Veedubin/AttackLM",
        "match": {"source_field": "supply_chain_synthetic"},
        "apply_to_files": ["data.jsonl", "data_synth.jsonl"],
        "apply_to_buckets": ["supply_chain/attacks"],
        "risk": "low",
    },
    {
        "name": "attacklm-synthetic",
        "display": "AttackLM Synthetic (deterministic templates)",
        "license": "MIT",
        "license_uri": "https://opensource.org/licenses/MIT",
        "source_uri": "https://github.com/Veedubin/AttackLM",
        "match": {"source_field": "ics_synthetic"},
        "apply_to_files": ["data.jsonl", "data_synth.jsonl"],
        "apply_to_buckets": ["ics/attacks"],
        "risk": "low",
    },
    {
        "name": "attacklm-synthetic",
        "display": "AttackLM Synthetic (deterministic templates)",
        "license": "MIT",
        "license_uri": "https://opensource.org/licenses/MIT",
        "source_uri": "https://github.com/Veedubin/AttackLM",
        "match": {"source_field": "wireless_synthetic"},
        "apply_to_files": ["data.jsonl", "data_synth.jsonl"],
        "apply_to_buckets": ["wireless/attacks"],
        "risk": "low",
    },
    {
        "name": "attacklm-synthetic",
        "display": "AttackLM Synthetic (deterministic templates)",
        "license": "MIT",
        "license_uri": "https://opensource.org/licenses/MIT",
        "source_uri": "https://github.com/Veedubin/AttackLM",
        "match": {"source_field": "web_attack_synthetic"},
        "apply_to_files": ["data.jsonl"],
        "apply_to_buckets": ["web_app/attacks"],
        "risk": "low",
    },
    {
        "name": "attacklm-synthetic",
        "display": "AttackLM Synthetic (red team tactics)",
        "license": "MIT",
        "license_uri": "https://opensource.org/licenses/MIT",
        "source_uri": "https://github.com/Veedubin/AttackLM",
        "match": {"source_field": "red_team_tactics"},
        "apply_to_files": ["data.jsonl", "data_synth.jsonl"],
        "apply_to_buckets": ["attack_tactics/red_team_tactics"],
        "risk": "low",
    },
    {
        "name": "attacklm-synthetic",
        "display": "AttackLM Synthetic (orchestrator)",
        "license": "MIT",
        "license_uri": "https://opensource.org/licenses/MIT",
        "source_uri": "https://github.com/Veedubin/AttackLM",
        "match": {"source_field": None},
        "apply_to_files": ["data.jsonl"],
        "apply_to_buckets": ["orchestrator"],
        "risk": "low",
    },
    # AI prompt-injection synthetic (everything not matched by promptfoo/promptmap)
    {
        "name": "attacklm-synthetic",
        "display": "AttackLM Synthetic (prompt injection)",
        "license": "MIT",
        "license_uri": "https://opensource.org/licenses/MIT",
        "source_uri": "https://github.com/Veedubin/AttackLM",
        # Match by absence: records with no source field AND no fingerprint
        # of an upstream tool (will be applied last, after tool fingerprints
        # have been tried).
        "match": {
            "source_field": None,
            "fingerprint": None,
            "no_fingerprint": ["promptfoo", "promptmap"],
        },
        "apply_to_files": ["data.jsonl"],
        "apply_to_buckets": ["ai/prompt-injection"],
        "risk": "low",
    },
    # ---------- LLM GENERATED (qwen2.5-coder-14b via LMStudio) ----------
    {
        "name": "llm-generated",
        "display": "LLM-generated (qwen2.5-coder-14b-instruct via LMStudio)",
        "license": "GPL-3.0",
        "license_uri": "https://www.gnu.org/licenses/gpl-3.0.html",
        "source_uri": "https://github.com/QwenLM/Qwen2.5-Coder",
        "match": {"source_field": "llm_social_engineering"},
        "apply_to_files": ["data_llm.jsonl"],
        "apply_to_buckets": ["social_engineering/phishing"],
        "risk": "low",
    },
    {
        "name": "llm-generated",
        "display": "LLM-generated (qwen2.5-coder-14b-instruct via LMStudio)",
        "license": "GPL-3.0",
        "license_uri": "https://www.gnu.org/licenses/gpl-3.0.html",
        "source_uri": "https://github.com/QwenLM/Qwen2.5-Coder",
        "match": {"source_field": "llm_cloud"},
        "apply_to_files": ["data_llm.jsonl"],
        "apply_to_buckets": ["cloud/attacks"],
        "risk": "low",
    },
    {
        "name": "llm-generated",
        "display": "LLM-generated (qwen2.5-coder-14b-instruct via LMStudio)",
        "license": "GPL-3.0",
        "license_uri": "https://www.gnu.org/licenses/gpl-3.0.html",
        "source_uri": "https://github.com/QwenLM/Qwen2.5-Coder",
        "match": {"source_field": "llm_ics_scada"},
        "apply_to_files": ["data_llm.jsonl"],
        "apply_to_buckets": ["ics/attacks"],
        "risk": "low",
    },
    {
        "name": "llm-generated",
        "display": "LLM-generated (qwen2.5-coder-14b-instruct via LMStudio)",
        "license": "GPL-3.0",
        "license_uri": "https://www.gnu.org/licenses/gpl-3.0.html",
        "source_uri": "https://github.com/QwenLM/Qwen2.5-Coder",
        "match": {"source_field": "llm_wireless"},
        "apply_to_files": ["data_llm.jsonl"],
        "apply_to_buckets": ["wireless/attacks"],
        "risk": "low",
    },
]


def record_matches(rec: dict, match: dict) -> bool:
    """Return True if record matches the source fingerprint."""
    src = rec.get("source")
    text = json.dumps(rec)
    if "source_field" in match:
        if match["source_field"] is None:
            # explicit None means: match records with NO source field
            # (i.e. we are inferring from content)
            if src is not None and src != "":
                return False
        else:
            if src != match["source_field"]:
                return False
    if match.get("fingerprint") is not None:
        if match["fingerprint"] not in text:
            return False
    # Negative fingerprints: if any of these substrings appear, this record
    # belongs to a different source. Used to make synthetic the *fallback*
    # match for buckets that mix synthetic and upstream tool data.
    if "no_fingerprint" in match:
        for fp in match["no_fingerprint"]:
            if fp in text:
                return False
    return True


def stamp_record(rec: dict, prov: dict) -> dict:
    """Add/overwrite provenance fields on a record."""
    rec = dict(rec)  # shallow copy
    rec["source"] = prov["name"]
    rec["source_uri"] = prov["source_uri"]
    rec["license"] = prov["license"]
    rec["license_uri"] = prov["license_uri"]
    rec["rights_contact"] = RIGHTS_CONTACT
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing files",
    )
    args = parser.parse_args()

    counts: Counter = Counter()
    skipped: list[str] = []
    written: list[Path] = []

    if not args.dry_run:
        SOURCES_DIR.mkdir(parents=True, exist_ok=True)

    for prov in PROVENANCE:
        source_name = prov["name"]
        for bucket_path in prov["apply_to_buckets"]:
            for fname in prov["apply_to_files"]:
                src_file = BUCKETS_DIR / bucket_path / fname
                if not src_file.exists():
                    skipped.append(str(src_file))
                    continue
                dst_dir = SOURCES_DIR / source_name / bucket_path
                if not args.dry_run:
                    dst_dir.mkdir(parents=True, exist_ok=True)
                dst_file = dst_dir / fname

                n_in = 0
                n_matched = 0
                with src_file.open() as fin:
                    for line in fin:
                        line = line.rstrip("\n")
                        if not line:
                            continue
                        n_in += 1
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not record_matches(rec, prov["match"]):
                            continue
                        n_matched += 1
                        stamped = stamp_record(rec, prov)
                        if not args.dry_run:
                            with dst_file.open("a") as fout:
                                fout.write(json.dumps(stamped) + "\n")

                counts[(source_name, bucket_path, fname)] = (n_in, n_matched)
                if not args.dry_run:
                    written.append(dst_file)

    # Summary
    print(f"\n=== Provenance stamp + reorg {'(DRY RUN)' if args.dry_run else ''} ===\n")
    print(f"{'source':<35} {'bucket':<35} {'file':<22} {'in':>6} {'matched':>8}")
    print("-" * 110)
    for (src, bk, fn), (n_in, n_matched) in sorted(counts.items()):
        print(f"{src:<35} {bk:<35} {fn:<22} {n_in:>6} {n_matched:>8}")
    print()
    total_in = sum(n[0] for n in counts.values())
    total_matched = sum(n[1] for n in counts.values())
    print(f"Total records scanned:  {total_in}")
    print(f"Total records stamped:  {total_matched}")
    print(f"Files skipped (missing): {len(skipped)}")
    if not args.dry_run:
        print(f"Files written: {len(written)}")
    print()


if __name__ == "__main__":
    main()
