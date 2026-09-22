#!/usr/bin/env python3
"""Generate calibration holdout files for eval_calibration.py.

Naive heuristic split of questions.jsonl into 3 buckets:
  - calibration_in.jsonl  (in-distribution: security domain)
  - calibration_near.jsonl (near-OOD: adjacent tech topics)
  - calibration_ood.jsonl  (OOD: off-topic)

This is a v0.18 placeholder. A future version will use proper OOD curation.

Usage:
    python scripts/gen_calibration_holdouts.py [--output-dir data/bench] [--source data/bench/questions.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Heuristic keywords for classifying questions into buckets.
# In-distribution: security/offensive/red-team topics.
SECURITY_KEYWORDS = {
    "exploit",
    "vulnerability",
    "attack",
    "malware",
    "phishing",
    "pentest",
    "red-team",
    "reverse engineering",
    "forensics",
    "incident response",
    "firewall",
    "ids",
    "ips",
    "siem",
    "threat",
    "cve",
    "rce",
    "xss",
    "sql injection",
    "privilege escalation",
    "lateral movement",
    "c2",
    "command and control",
    "payload",
    "shellcode",
    "buffer overflow",
    "mitre",
    "att&ck",
    "technique",
    "tactic",
    "procedure",
    "ttp",
    "reconnaissance",
    "exfiltration",
    "persistence",
    "defense evasion",
    "credential access",
    "initial access",
    "execution",
    "collection",
    "discovery",
    "impact",
    "resource development",
}

# Near-OOD: adjacent tech topics (coding, networking, sysadmin).
NEAR_OOD_KEYWORDS = {
    "python",
    "javascript",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "network",
    "tcp",
    "udp",
    "dns",
    "http",
    "ssl",
    "tls",
    "encryption",
    "database",
    "sql",
    "api",
    "rest",
    "microservice",
    "container",
    "linux",
    "shell",
    "bash",
    "ssh",
    "vpn",
    "proxy",
    "load balancer",
    "monitoring",
    "logging",
    "backup",
    "deployment",
    "ci/cd",
    "git",
    "compiler",
    "debugging",
    "testing",
    "refactor",
    "architecture",
}


def classify_question(question: dict) -> str:
    """Classify a question as 'in', 'near', or 'ood'."""
    # Combine all text fields for classification.
    text = ""
    if "instruction" in question:
        text += " " + str(question["instruction"])
    if "input" in question:
        text += " " + str(question["input"])
    if "output" in question:
        text += " " + str(question["output"])
    if "prompt" in question:
        text += " " + str(question["prompt"])
    if "question" in question:
        text += " " + str(question["question"])
    if "text" in question:
        text += " " + str(question["text"])

    text_lower = text.lower()

    # Check for security keywords (in-distribution).
    security_hits = sum(1 for kw in SECURITY_KEYWORDS if kw in text_lower)
    if security_hits >= 2:
        return "in"

    # Check for near-OOD keywords.
    near_hits = sum(1 for kw in NEAR_OOD_KEYWORDS if kw in text_lower)
    if near_hits >= 2:
        return "near"

    # Check source/bucket metadata if available.
    source = question.get("source", "").lower()
    bucket = question.get("bucket", "").lower()
    if any(kw in source for kw in SECURITY_KEYWORDS):
        return "in"
    if any(kw in bucket for kw in SECURITY_KEYWORDS):
        return "in"
    if any(kw in source for kw in NEAR_OOD_KEYWORDS):
        return "near"

    # Default: OOD.
    return "ood"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate calibration holdout files from questions.jsonl",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/bench/questions.jsonl",
        help="Source questions.jsonl file (default: data/bench/questions.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/bench",
        help="Output directory (default: data/bench)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    output_dir = Path(args.output_dir)

    if not source_path.exists():
        print(f"Error: source file not found: {source_path}", file=sys.stderr)
        return 1

    # Read source questions.
    questions = []
    with open(source_path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    if not questions:
        print(f"Error: no questions found in {source_path}", file=sys.stderr)
        return 1

    # Classify and split.
    in_dist: list[dict] = []
    near_ood: list[dict] = []
    ood: list[dict] = []

    for q in questions:
        bucket = classify_question(q)
        if bucket == "in":
            in_dist.append(q)
        elif bucket == "near":
            near_ood.append(q)
        else:
            ood.append(q)

    # Shuffle with fixed seed for reproducibility.
    import random

    rng = random.Random(args.seed)
    rng.shuffle(in_dist)
    rng.shuffle(near_ood)
    rng.shuffle(ood)

    # Write output files.
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "calibration_in.jsonl": in_dist,
        "calibration_near.jsonl": near_ood,
        "calibration_ood.jsonl": ood,
    }

    for filename, data in files.items():
        out_path = output_dir / filename
        with open(out_path, "w") as f:
            for q in data:
                f.write(json.dumps(q) + "\n")
        print(f"Wrote {len(data)} questions to {out_path}")

    print("\nClassification summary:")
    print(f"  In-distribution (security): {len(in_dist)}")
    print(f"  Near-OOD (adjacent tech):    {len(near_ood)}")
    print(f"  OOD (off-topic):             {len(ood)}")
    print(f"  Total:                       {len(questions)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
