"""Resolve the AttackLM project root directory.

When running from source (``scripts/train_all.py``), the project root is the
parent of the ``scripts/`` directory.  When installed from PyPI
(``site-packages/attacklm/scripts/train_all.py``), the project root is the
current working directory — the user is expected to be inside the cloned
AttackLM repo.

Usage::

    from attacklm._project_root import BASE_DIR, require_manifest
    require_manifest()  # exits with a helpful message if manifest is missing
"""

from __future__ import annotations

import sys
from pathlib import Path


def _resolve_base_dir() -> Path:
    """Find the AttackLM project root."""
    # Check if we're running from source (scripts/ is at repo root)
    import __main__ as _main

    main_file = Path(getattr(_main, "__file__", "")).resolve()
    if main_file.name and main_file.parent.name == "scripts":
        candidate = main_file.parent.parent
        if (candidate / "data" / "datasets" / "buckets" / "manifest.json").exists():
            return candidate

    # Running from PyPI install — use CWD
    cwd = Path.cwd()
    return cwd


BASE_DIR = _resolve_base_dir()
DATASETS_DIR = BASE_DIR / "data" / "datasets"
BUCKETS_DIR = DATASETS_DIR / "buckets"
SOURCES_DIR = BUCKETS_DIR / "sources"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
PRESETS_DIR = BASE_DIR / "presets"


def require_manifest() -> Path:
    """Return the manifest path, or exit with a helpful message."""
    manifest = BUCKETS_DIR / "manifest.json"
    if not manifest.exists():
        print(
            "ERROR: Manifest not found. Run this command from the AttackLM repo root.\n"
            "  git clone https://github.com/Veedubin/AttackLM.git\n"
            "  cd AttackLM\n"
            "  attacklm train --all ...",
            file=sys.stderr,
        )
        sys.exit(1)
    return manifest
