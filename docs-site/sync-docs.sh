#!/usr/bin/env bash
# Generates the docs-site content files from the real source docs.
#
# The files under docs-site/docs/ are BUILD ARTIFACTS (gitignored) —
# never edit them directly. Edit the sources in the repo root (or in
# the attacklm-dataset sibling repo) and re-run this script.
#
# Local dev:   docs-site/sync-docs.sh && (cd docs-site && mkdocs serve)
# CI:          docs.yml runs the same copies before `mkdocs build`.
#
# Set ATTACKLM_DATASET_DIR if the dataset repo is not a sibling checkout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DOCS_DIR="$SCRIPT_DIR/docs"
DATASET_DIR="${ATTACKLM_DATASET_DIR:-$REPO_ROOT/../attacklm-dataset}"

mkdir -p "$DOCS_DIR"

# AttackLM docs (this repo)
cp "$REPO_ROOT/README.md"          "$DOCS_DIR/attacklm-readme.md"
cp "$REPO_ROOT/CHANGELOG.md"       "$DOCS_DIR/attacklm-changelog.md"
cp "$REPO_ROOT/docs/RL_RECIPE.md"  "$DOCS_DIR/attacklm-rl-recipe.md"
cp "$REPO_ROOT/EVALUATION.md"      "$DOCS_DIR/attacklm-evaluation.md"
cp "$REPO_ROOT/CONTRIBUTING.md"    "$DOCS_DIR/attacklm-contributing.md"
cp "$REPO_ROOT/ATTRIBUTION.md"     "$DOCS_DIR/attacklm-attribution.md"

# attacklm-dataset docs (sibling repo; internal methodology docs are
# intentionally NOT mirrored — they are local-only per project policy)
cp "$DATASET_DIR/README.md"       "$DOCS_DIR/attacklm-dataset-readme.md"
cp "$DATASET_DIR/CHANGELOG.md"    "$DOCS_DIR/attacklm-dataset-changelog.md"
cp "$DATASET_DIR/ATTRIBUTION.md"  "$DOCS_DIR/attacklm-dataset-attribution.md"
cp "$DATASET_DIR/PROVENANCE.md"   "$DOCS_DIR/attacklm-dataset-provenance.md"
cp "$DATASET_DIR/RIGHTS.md"       "$DOCS_DIR/attacklm-dataset-rights.md"
cp "$DATASET_DIR/SECURITY.md"     "$DOCS_DIR/attacklm-dataset-security.md"

echo "Synced 12 docs into $DOCS_DIR"
