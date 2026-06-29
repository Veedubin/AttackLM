#!/usr/bin/env bash
# ============================================================================
# clone_repos.sh — Clone or update threat-intelligence data repositories
#
# Clones the upstream open-source projects whose data is used to train
# AttackLM. See /ATTRIBUTION.md for the full list and license details.
# If a repo directory already exists, performs a git pull instead.
#
# Usage:
#   ./scripts/clone_repos.sh
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Root directory is one level up from this script (AttackLM/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
DATA_DIR="${ROOT_DIR}/data"

# Each entry: "url destination_dirname  -- license"
# License is informational only — see /ATTRIBUTION.md for full terms.
# This script is a thin wrapper around `git clone` and does NOT enforce
# or modify any upstream license.
REPOS=(
  "https://github.com/redcanaryco/atomic-red-team.git atomic-red-team        -- MIT"
  "https://github.com/mitre/stockpile.git stockpile                       -- Apache-2.0"
  "https://github.com/SigmaHQ/sigma.git sigma                             -- DRL-1.1"
  "https://github.com/rapid7/metasploit-framework.git metasploit-framework -- BSD-3-Clause"
  "https://github.com/elastic/detection-rules.git elastic-detection-rules -- Apache-2.0"
  "https://github.com/splunk/security_content.git splunk-security-content  -- Apache-2.0"
  "https://github.com/OTRF/Security-Datasets.git mordor                   -- Apache-2.0"
  "https://github.com/OTRF/ThreatHunter-Playbook.git threathunter-playbook -- Apache-2.0"
)
# NOTE: nist-sp800-61r3 is a PDF download, not a git repo.
# Download manually from: https://csrc.nist.gov/pubs/sp/800-61/r3/final
# and place at data/nist-sp800-61r3/NIST.SP.800-61r3.pdf

# ---------------------------------------------------------------------------
# Ensure the data directory exists
# ---------------------------------------------------------------------------
mkdir -p "${DATA_DIR}"

# ---------------------------------------------------------------------------
# Clone or pull each repository
# ---------------------------------------------------------------------------
cloned=0
pulled=0
failed=0

for entry in "${REPOS[@]}"; do
  # Split on whitespace. Format: "url dirname -- license"
  # url is the first token, dirname is the second, "-- license" is the rest.
  url="$(echo "${entry}" | awk '{print $1}')"
  dirname="$(echo "${entry}" | awk '{print $2}')"
  # license info is at end of line, drop the leading "--"
  license_info="$(echo "${entry}" | sed -E 's/.*-- //')"
  dest="${DATA_DIR}/${dirname}"

  if [ -d "${dest}/.git" ]; then
    # Repository already cloned — pull latest changes
    echo "==> Pulling ${dirname} (${license_info})"
    if git -C "${dest}" pull; then
      echo "    OK — ${dirname} updated"
      pulled=$((pulled + 1))
    else
      echo "    FAIL — could not pull ${dirname}" >&2
      failed=$((failed + 1))
    fi
  else
    # Fresh clone
    echo "==> Cloning ${dirname} (${license_info})"
    if git -C "${DATA_DIR}" clone "${url}" "${dirname}"; then
      echo "    OK — ${dirname} cloned"
      cloned=$((cloned + 1))
    else
      echo "    FAIL — could not clone ${dirname}" >&2
      failed=$((failed + 1))
    fi
  fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total=$((cloned + pulled + failed))

echo ""
echo "============================================"
echo "  Repository sync complete"
echo "============================================"
echo "  Total repos : ${total}"
echo "  Cloned      : ${cloned}"
echo "  Pulled      : ${pulled}"
echo "  Failed      : ${failed}"
echo "  Data dir    : ${DATA_DIR}"
echo "============================================"

# Exit with failure if any repo failed
if [ "${failed}" -gt 0 ]; then
  exit 1
fi