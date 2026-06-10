# Contributing to AttackLM

Thanks for your interest in AttackLM. This is a security tooling project
and contributions are welcome — but please read this first.

## ⚠️ Ethics and legal use

AttackLM is a tool for **authorized security testing, red teaming, and
AI safety research**. By contributing, you agree that:

- Your contributions are released under the MIT License.
- You will not use AttackLM for unauthorized access, attacks on systems
  you don't own or have explicit permission to test, or any illegal
  activity.
- Generated model output is for testing defensive systems. It is not a
  green light to attack any system.

The training data is sourced from openly licensed public security
projects and is intended to train models that can be used by blue teams
to recognize, classify, and respond to attacker tradecraft.

## What to contribute

We welcome:

- **New extractors** for additional open-source security data sources
  (Sigma rules, OWASP rules, MITRE ATLAS, etc.)
- **New bucket types** for new attack categories
- **Bug fixes** in training scripts (especially the 13 OOM fixes)
- **HPO improvements** (better verdict logic, new axes, etc.)
- **Documentation** improvements (typos, unclear instructions)
- **Inference utilities** (Ollama conversion, LM Studio integration, etc.)

We will **not** accept:

- Training data scraped from closed/private sources without a clear
  open license
- Anything that adds obfuscation or evasion specifically intended to
  bypass detection systems (we're a defensive tool — not a malware
  builder)
- Code with no provenance or attribution to upstream sources

## Adding a new data source

To add training data from a new open-source project:

1. **Verify the license** — must be MIT, Apache-2.0, BSD (2/3-clause),
   DRL-1.1, or another permissive license. GPL/AGPL is OK but adds
   redistribution obligations; document the implications in
   `ATTRIBUTION.md`.
2. **Add the clone URL to `scripts/clone_repos.sh`** with a license
   comment. Example:
   ```bash
   "https://github.com/owner/repo.git dirname    -- LICENSE"
   ```
3. **Write an extractor** in `scripts/extract_<source>_to_jsonl.py`.
   It should be deterministic (no LLM calls) and emit OpenAI-style
   message triples. Add a `# CREDITS` block at the top with the
   upstream URL, license, and copyright.
4. **Run the extractor** and verify the output JSONL has the expected
   `messages:[{role, content}]` schema.
5. **Add a bucket entry** in `BUCKET_ATTRIBUTION` in
   `scripts/augment_attribution.py` so per-pair source/license
   attribution is added.
6. **Update `ATTRIBUTION.md`** with the new source, license, and pair
   counts.
7. **Update `README.md`** if the new source warrants a mention in the
   "Data Sources" table.

## Adding a new bucket

Buckets live in `data/datasets/buckets/<bucket_name>/` and contain a
`data.jsonl`. To add one:

1. Create the directory and write `data.jsonl` with your training pairs.
2. Add a `metadata.json` sidecar:
   ```json
   {
     "name": "your_bucket",
     "display_name": "Your Bucket",
     "category": "tactic",
     "mitre_tactic": "TA000X",
     "description": "..."
   }
   ```
3. Add the bucket to `data/datasets/buckets/manifest.json` (manually or
   via a small script).
4. Add the attribution mapping in `scripts/augment_attribution.py`.

## Development setup

```bash
# Install dependencies
uv sync

# Run a sanity check on the training pipeline
uv run python scripts/train_template.py \
  --dataset data/datasets/combined/combined_*.jsonl \
  --output /tmp/test-output --dry-run
```

## Pull request process

1. Fork the repo and create a feature branch.
2. Add tests or a dry-run command that exercises your change.
3. Update `ATTRIBUTION.md` and `README.md` as needed.
4. Run a quick training smoke test (1 epoch, small dataset).
5. Submit the PR with:
   - A clear description of what the change does
   - The motivation / use case
   - Any new dependencies introduced
   - Attribution for any new data sources

## Coding style

- Python: PEP 8 + Black formatter (88-char line length)
- Bash: `shellcheck` clean
- Markdown: 80-char line length in docs, longer is OK in code blocks

## License

By contributing, you agree your contributions are licensed under the
MIT License. See [LICENSE](LICENSE) for the full text.
