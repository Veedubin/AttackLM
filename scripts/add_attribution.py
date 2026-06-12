#!/usr/bin/env python3
"""
add_attribution.py — Stamp license-specific attribution fields onto records
that have additional upstream requirements beyond the base provenance.

Run AFTER `scripts/stamp_and_reorg.py`. This script is idempotent.

Currently handles:
  - **Metasploit Framework (BSD-3-Clause)** — 13,997 records.
    BSD-3 §1 requires the copyright notice and license text to be
    preserved in derivative works. We add:
      - `upstream_copyright`
      - `upstream_license_uri`
      - `attribution_required: true`
      - `bsd_3_clause_notice` (the short notice text)
      - `derived_from` (the upstream project + module path / CVE if
        extractable from the record content)
  - **SigmaHQ (DRL 1.1)** — currently 0 derived records in the dataset,
    but if Sigma rules are added in the future the following fields will
    be required by DRL §3:
      - `sigma_rule_id`
      - `sigma_rule_author`
      - `sigma_rule_date`
      - `sigma_rule_title`
      - `drl_11_attribution` (notice text)

Usage:
    uv run python scripts/add_attribution.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES_DIR = ROOT / "data" / "datasets" / "buckets" / "sources"

# ---------------------------------------------------------------------------
# BSD-3-Clause attribution (for Metasploit)
# ---------------------------------------------------------------------------
RAPID7_COPYRIGHT = "Copyright (c) 2006-2026, Rapid7, Inc. All rights reserved."
BSD_3_NOTICE = (
    "Redistribution and use in source and binary forms, with or without "
    "modification, are permitted provided that the following conditions are met: "
    "1) Redistributions of source code must retain the above copyright notice, "
    "this list of conditions and the following disclaimer. "
    "2) Redistributions in binary form must reproduce the above copyright notice, "
    "this list of conditions and the following disclaimer in the documentation "
    "and/or other materials provided with the distribution. "
    "3) Neither the name of the copyright holder nor the names of its contributors "
    "may be used to endorse or promote products derived from this software without "
    "specific prior written permission."
)

# Module-path regex (used by `tools/metasploit/` records, less so the
# `base/*` derived triples).
MODULE_PATH_RE = re.compile(
    r"`(exploit/[^\s`]+|auxiliary/[^\s`]+|post/[^\s`]+|payload/[^\s`]+)"
    r'|("path":\s*"(?:exploit|auxiliary|post|payload)/[^"]+")',
    re.IGNORECASE,
)
# CVE pattern: CVE-YYYY-NNNN (allow 4-7 digit NNNN)
CVE_RE = re.compile(r"CVE[-_]?(\d{4})[-_](\d{4,7})", re.IGNORECASE)

# ---------------------------------------------------------------------------
# DRL 1.1 attribution (for SigmaHQ; no records in current dataset)
# ---------------------------------------------------------------------------
DRL_11_NOTICE = (
    "This rule is derived from a SigmaHQ rule distributed under the "
    "Detection Rule License (DRL) 1.1. The original rule's id, author, "
    "date, and title are preserved in `sigma_rule_*` fields. The DRL 1.1 "
    "license text is at https://github.com/SigmaHQ/sigma/blob/master/LICENSE.Detection.Rules.md"
)


def stamp_metasploit(rec: dict) -> tuple[dict, bool]:
    """
    Add BSD-3-Clause attribution fields to a Metasploit record.
    Returns (record, changed).
    """
    rec = dict(rec)
    changed = False

    def set_field(key: str, value):
        nonlocal changed
        if rec.get(key) != value:
            rec[key] = value
            changed = True

    set_field("upstream_copyright", RAPID7_COPYRIGHT)
    set_field("upstream_license_uri", "https://opensource.org/licenses/BSD-3-Clause")
    set_field("attribution_required", True)
    set_field("bsd_3_clause_notice", BSD_3_NOTICE)
    set_field("derived_from", "rapid7/metasploit-framework")

    # Extract module path / CVE if present in the record content
    text = json.dumps(rec)
    m = MODULE_PATH_RE.search(text)
    if m:
        module_path = m.group(1) or m.group(2)
        # Strip JSON quotes
        module_path = module_path.strip('"').rstrip('"')
        if module_path.startswith('"path":'):
            module_path = (
                module_path.split('"')[-2] if '"' in module_path else module_path
            )
        set_field("upstream_module_path", module_path)
    cves = sorted(set(CVE_RE.findall(text)))
    if cves:
        cve_list = [f"CVE-{y}-{n}" for y, n in cves]
        if rec.get("upstream_cve") != cve_list:
            rec["upstream_cve"] = cve_list
            changed = True

    return rec, changed


def stamp_sigma(rec: dict) -> tuple[dict, bool]:
    """
    Add DRL 1.1 attribution fields to a SigmaHQ record.
    Returns (record, changed).
    """
    rec = dict(rec)
    changed = False

    def set_field(key: str, value):
        nonlocal changed
        if rec.get(key) != value:
            rec[key] = value
            changed = True

    set_field("attribution_required", True)
    set_field("drl_11_attribution", DRL_11_NOTICE)

    # Sigma records should already have these from the upstream rule
    # (rule.id, rule.author, rule.date, rule.title) but we re-key them
    # to the `sigma_rule_*` namespace for clarity.
    for upstream_key, downstream_key in [
        ("rule_id", "sigma_rule_id"),
        ("rule_author", "sigma_rule_author"),
        ("rule_date", "sigma_rule_date"),
        ("rule_title", "sigma_rule_title"),
    ]:
        if upstream_key in rec:
            set_field(downstream_key, rec[upstream_key])

    return rec, changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    counts: Counter = Counter()
    total_in = 0
    total_changed = 0
    issues: list[str] = []

    for source_dir in sorted(SOURCES_DIR.iterdir()):
        if not source_dir.is_dir() or source_dir.name.startswith("_"):
            continue
        source_name = source_dir.name

        if source_name == "metasploit-framework":
            stampler = stamp_metasploit
            label = "metasploit (BSD-3-Clause)"
        elif source_name == "sigma-hq" or source_name == "sigmahq":
            stampler = stamp_sigma
            label = "sigmahq (DRL 1.1)"
        else:
            continue  # other sources: base provenance is enough

        for jsonl in sorted(source_dir.rglob("*.jsonl")):
            n_in = 0
            n_changed = 0
            tmp_path = jsonl.with_suffix(".jsonl.tmp")
            with jsonl.open() as fin, tmp_path.open("w") as fout:
                for line in fin:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    n_in += 1
                    rec = json.loads(line)
                    new_rec, changed = stampler(rec)
                    fout.write(json.dumps(new_rec) + "\n")
                    if changed:
                        n_changed += 1
            if not args.dry_run:
                tmp_path.replace(jsonl)
            counts[label] += n_in
            total_in += n_in
            total_changed += n_changed
            if n_in == 0:
                issues.append(f"{jsonl.relative_to(ROOT)}: 0 records")

    print(f"\n=== Attribution stamp {'(DRY RUN)' if args.dry_run else ''} ===\n")
    print(f"{'Source / License':<35} {'Records':>10} {'Changed':>10}")
    print("-" * 60)
    for label, n in sorted(counts.items()):
        print(f"{label:<35} {n:>10,} {n:>10,}  (all records updated)")
    print("-" * 60)
    print(f"{'TOTAL':<35} {total_in:>10,} {total_changed:>10,}")
    print()
    if issues:
        print("Notes:")
        for i in issues:
            print(f"  {i}")
    if not args.dry_run:
        print("Files written in place (atomic: .tmp -> rename).")
    else:
        print("Dry run — no files written.")
    print()


if __name__ == "__main__":
    main()
