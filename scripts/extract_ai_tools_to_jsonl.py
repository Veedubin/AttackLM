#!/usr/bin/env python3
# CREDITS — DATA SOURCE ATTRIBUTION
# ----------------------------------
# This script transforms data from six AI-security open-source projects:
#   - promptfoo:         https://github.com/promptfoo/promptfoo   (MIT)
#   - garak (NVIDIA):    https://github.com/NVIDIA/garak            (Apache-2.0)
#   - TheBigPromptLibrary: https://github.com/Resident-Falker/TheBigPromptLibrary  (mixed MIT/MPL)
#   - promptmap:         https://github.com/utkusen/promptmap     (MIT)
#   - PyRIT (Azure):     https://github.com/Azure/PyRIT            (MIT)
#   - FuzzyAI (CyberArk): https://github.com/cyberark/FuzzyAI       (Apache-2.0)
#
# Each upstream repo is cloned into data/ai_tools/<name>/. The output
# JSONL is a *transformation* of upstream prompt/probe/template files
# into chat triples. See /ATTRIBUTION.md for full per-source details.
# ----------------------------------
"""Extract prompt injection/jailbreak data from cloned AI tool repos into AttackLM JSONL training pairs.

Walks the repos cloned into ``data/ai_tools/`` and produces OpenAI-style
message triples (system / user / assistant) for supervised fine-tuning.
Each parser targets a specific repo and yields ``dict`` rows ready for
JSONL serialisation.

Supported repos:
  - promptfoo  — redteam TypeScript plugins
  - garak      — DAN/probe JSON & TXT resources
  - TheBigPromptLibrary — Jailbreak, SystemPrompts, Security markdown
  - promptmap  — YAML rule files
  - PyRIT      — jailbreak templates (if repo populated)
  - FuzzyAI    — adversarial prompts, suffixes, harmful behaviours CSV

This file contains ONLY the parser functions and supporting helpers.
``main()`` and CLI argument handling live in a separate CHUNK added later.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
AI_TOOLS_DIR = BASE_DIR / "data" / "ai_tools"
DATASETS_DIR = BASE_DIR / "data" / "datasets"

# ---------------------------------------------------------------------------
# System message
# ---------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are an authorized Red Team Prompt Injection specialist. "
    "You provide precise prompt injection, jailbreak crafting, and AI system "
    "exploitation techniques for security validation."
)

# ---------------------------------------------------------------------------
# MITRE ATT&CK analogue mapping
# ---------------------------------------------------------------------------
MITRE_AI_MAP: dict[str, tuple[str, str]] = {
    "prompt_injection": (
        "T1566.001",
        "Spearphishing Attachment — social engineering the AI",
    ),
    "jailbreak": (
        "T1548",
        "Abuse Elevation Control — bypassing model restrictions",
    ),
    "system_prompt_extraction": (
        "T1552",
        "Unsecured Credentials — extracting protected information",
    ),
    "token_smuggling": (
        "T1027",
        "Obfuscated Files — encoding to evade detection",
    ),
    "data_poisoning": (
        "T1565.001",
        "Stored Data Manipulation — corrupting training data",
    ),
    "tool_misuse": (
        "T1203",
        "Exploitation for Client Execution — exploiting tool capabilities",
    ),
    "multi_turn": (
        "T1068",
        "Exploitation for Privilege Escalation — escalating access",
    ),
}


# ---------------------------------------------------------------------------
# Helper — build a single training pair
# ---------------------------------------------------------------------------
def _make_pair(
    technique_name: str,
    category: str,
    prompt_text: str,
    description: str,
) -> dict:
    """Return an OpenAI-style message triple for supervised fine-tuning.

    Parameters
    ----------
    technique_name:
        Human-readable name of the attack technique.
    category:
        One of the keys in :data:`MITRE_AI_MAP` (e.g. ``"jailbreak"``).
    prompt_text:
        The raw attack string / prompt payload.
    description:
        Brief context or description of the technique.

    Returns
    -------
    dict
        ``{"messages": [system, user, assistant]}`` ready for JSONL.
    """
    mitre_id, mitre_name = MITRE_AI_MAP.get(
        category, ("T0000", "Unknown — unmapped technique")
    )

    assistant_content = (
        f"**Technique: {technique_name} — MITRE Analogue: {mitre_id} ({mitre_name})**\n\n"
        f"**Attack String:**\n```\n{prompt_text}\n```\n\n"
        f"**Description:** {description}\n\n"
        f"**Detection Artifacts:**\n"
        f"- Input pattern matching for {category} signatures\n"
        f"- Output toxicity/anomaly scoring\n"
        f"- Conversation history analysis\n\n"
        f"**Mitigation:**\n"
        f"- Deploy input guardrails for {category} patterns\n"
        f"- Use constitutional AI classifiers\n"
        f"- Monitor output for policy violations"
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {
                "role": "user",
                "content": f"How do I perform {technique_name}?",
            },
            {"role": "assistant", "content": assistant_content},
        ]
    }


# ---------------------------------------------------------------------------
# Regex helpers for TypeScript extraction
# ---------------------------------------------------------------------------
_TS_STRING_RE = re.compile(
    r"""(?:prompt|content|text|template|attack)\s*[:=]\s*["'`]"""
    r"""([\s\S]{10,}?)["'`]""",
    re.IGNORECASE,
)
_TS_TEMPLATE_RE = re.compile(
    r"""(?:prompt|content|text|template|attack)\s*[:=]\s*`"""
    r"""([\s\S]{20,}?)""" + r"""`""",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Parser: promptfoo
# ---------------------------------------------------------------------------
def parse_promptfoo(max_pairs: int = 50) -> list[dict]:
    """Parse promptfoo redteam TypeScript files for attack prompts.

    Walks ``data/ai_tools/promptfoo/src/redteam`` for ``.ts`` files,
    extracting string literals that look like prompt/injection payloads.

    Parameters
    ----------
    max_pairs:
        Maximum number of training pairs to return.

    Returns
    -------
    list[dict]
        Training pairs from ``_make_pair``.
    """
    redteam_dir = AI_TOOLS_DIR / "promptfoo" / "src" / "redteam"
    if not redteam_dir.exists():
        warnings.warn(
            f"[promptfoo] Directory not found: {redteam_dir}. "
            f"Run scripts/clone_repos.sh first.",
            stacklevel=2,
        )
        return []

    print(f"  [promptfoo] Scanning {redteam_dir} ...")
    pairs: list[dict] = []
    file_count = 0

    for ts_file in sorted(redteam_dir.rglob("*.ts")):
        if len(pairs) >= max_pairs:
            break
        try:
            text = ts_file.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            warnings.warn(f"[promptfoo] Cannot read {ts_file}: {exc}", stacklevel=2)
            continue

        file_count += 1

        # Categorise based on path segments
        path_lower = str(ts_file).lower()
        if any(kw in path_lower for kw in ("injection", "prompt_injection", "inject")):
            category = "prompt_injection"
        elif any(kw in path_lower for kw in ("jailbreak", "jb", "escape")):
            category = "jailbreak"
        else:
            # Default for redteam files — most are injection-related
            category = "prompt_injection"

        # Extract string-like payloads from TS
        for match in _TS_TEMPLATE_RE.finditer(text):
            if len(pairs) >= max_pairs:
                break
            payload = match.group(1).strip()
            if len(payload) < 20:
                continue

            technique = f"promptfoo: {ts_file.stem}"
            description = f"Extracted from promptfoo redteam plugin {ts_file.relative_to(redteam_dir)}"
            pairs.append(_make_pair(technique, category, payload, description))

        # Also check double-quoted single-line strings
        for match in _TS_STRING_RE.finditer(text):
            if len(pairs) >= max_pairs:
                break
            payload = match.group(1).strip()
            if len(payload) < 20:
                continue

            technique = f"promptfoo: {ts_file.stem}"
            description = f"Extracted from promptfoo redteam plugin {ts_file.relative_to(redteam_dir)}"
            pairs.append(_make_pair(technique, category, payload, description))

    # De-duplicate by assistant content
    seen: set[str] = set()
    unique_pairs: list[dict] = []
    for pair in pairs:
        key = pair["messages"][2]["content"]
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)

    print(f"  [promptfoo] Extracted {len(unique_pairs)} pairs from {file_count} files")
    return unique_pairs[:max_pairs]


# ---------------------------------------------------------------------------
# Parser: garak
# ---------------------------------------------------------------------------
def parse_garak(max_pairs: int = 50) -> list[dict]:
    """Parse garak probe data for jailbreak/prompt-attack payloads.

    Walks ``data/ai_tools/garak/garak/data`` for ``.json`` and ``.txt``
    files containing attack prompts.  Also scans ``garak/probes/`` for
    Python files with embedded prompt strings.

    Parameters
    ----------
    max_pairs:
        Maximum number of training pairs to return.

    Returns
    -------
    list[dict]
        Training pairs from ``_make_pair``.
    """
    garak_root = AI_TOOLS_DIR / "garak"
    data_dir = garak_root / "garak" / "data"
    probes_dir = garak_root / "garak" / "probes"

    if not garak_root.exists():
        warnings.warn(
            f"[garak] Directory not found: {garak_root}. "
            f"Run scripts/clone_repos.sh first.",
            stacklevel=2,
        )
        return []

    print(f"  [garak] Scanning {data_dir} ...")
    pairs: list[dict] = []

    # --- Category mapping from directory names ---
    _GARAK_DIR_CATEGORY: dict[str, str] = {
        "dan": "jailbreak",
        "autodan": "jailbreak",
        "gcg": "token_smuggling",
        "beast": "token_smuggling",
        "agent_breaker": "multi_turn",
        "sysprompt_extraction": "system_prompt_extraction",
        "tap": "jailbreak",
        "xss": "prompt_injection",
        "phishing": "prompt_injection",
    }

    # --- Walk data/ for .json files ---
    if data_dir.exists():
        for json_file in sorted(data_dir.rglob("*.json")):
            if len(pairs) >= max_pairs:
                break
            # Determine category from parent directory
            rel_dir = json_file.parent.name
            category = _GARAK_DIR_CATEGORY.get(rel_dir, "jailbreak")

            try:
                with open(json_file, "r", encoding="utf-8", errors="ignore") as fh:
                    content = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                warnings.warn(
                    f"[garak] Cannot parse {json_file.name}: {exc}", stacklevel=2
                )
                continue

            # JSON may be a list of strings or a single string
            prompts: list[str] = []
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, str) and len(item.strip()) >= 20:
                        prompts.append(item.strip())
            elif isinstance(content, str) and len(content.strip()) >= 20:
                prompts.append(content.strip())

            for idx, prompt_text in enumerate(prompts):
                if len(pairs) >= max_pairs:
                    break
                technique = f"garak: {json_file.stem}"
                if len(prompts) > 1:
                    technique += f" ({idx + 1}/{len(prompts)})"
                description = f"Garak probe data from {json_file.relative_to(data_dir)}"
                pairs.append(_make_pair(technique, category, prompt_text, description))

    # --- Walk data/ for .txt files (one prompt per line or whole file) ---
    if data_dir.exists():
        for txt_file in sorted(data_dir.rglob("*.txt")):
            if len(pairs) >= max_pairs:
                break
            rel_dir = txt_file.parent.name
            category = _GARAK_DIR_CATEGORY.get(rel_dir, "jailbreak")

            try:
                text = txt_file.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                warnings.warn(
                    f"[garak] Cannot read {txt_file.name}: {exc}", stacklevel=2
                )
                continue

            lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 20]
            for idx, line in enumerate(lines):
                if len(pairs) >= max_pairs:
                    break
                technique = f"garak: {txt_file.stem} (line {idx + 1})"
                description = f"Garak probe data from {txt_file.relative_to(data_dir)}"
                pairs.append(_make_pair(technique, category, line, description))

    # --- Scan probes/ for Python files with embedded prompt strings ---
    if probes_dir.exists():
        _PY_PROMPT_RE = re.compile(
            r"""(?:prompts|prompt_strings|attack_prompts|strings)\s*=\s*\["""
            r"""([\s\S]{20,}?)"""
            r"""\]""",
        )
        for py_file in sorted(probes_dir.rglob("*.py")):
            if len(pairs) >= max_pairs:
                break
            try:
                text = py_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            # Use directory/category hint from filename
            py_stem = py_file.stem.lower()
            if any(kw in py_stem for kw in ("dan", "jailbreak")):
                category = "jailbreak"
            elif any(kw in py_stem for kw in ("inject", "xss", "sink")):
                category = "prompt_injection"
            elif any(kw in py_stem for kw in ("extract", "leak", "sysprompt")):
                category = "system_prompt_extraction"
            elif any(kw in py_stem for kw in ("token", "gcg", "smuggl", "encode")):
                category = "token_smuggling"
            else:
                category = "prompt_injection"

            for match in _PY_PROMPT_RE.finditer(text):
                if len(pairs) >= max_pairs:
                    break
                payload = match.group(1).strip()
                if len(payload) < 20:
                    continue
                technique = f"garak: {py_file.stem}"
                description = f"Embedded prompt from garak probe {py_file.stem}"
                pairs.append(_make_pair(technique, category, payload, description))

    # De-duplicate
    seen: set[str] = set()
    unique_pairs: list[dict] = []
    for pair in pairs:
        key = pair["messages"][2]["content"]
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)

    print(f"  [garak] Extracted {len(unique_pairs)} pairs")
    return unique_pairs[:max_pairs]


# ---------------------------------------------------------------------------
# Parser: TheBigPromptLibrary
# ---------------------------------------------------------------------------
def parse_big_prompt_library(max_pairs: int = 50) -> list[dict]:
    """Parse TheBigPromptLibrary for jailbreak, system prompt, and security data.

    Walks ``Jailbreak/``, ``SystemPrompts/``, and ``Security/`` folders for
    ``.md`` and ``.txt`` files.  Each file produces one training pair.

    Parameters
    ----------
    max_pairs:
        Maximum number of training pairs to return.

    Returns
    -------
    list[dict]
        Training pairs from ``_make_pair``.
    """
    lib_dir = AI_TOOLS_DIR / "TheBigPromptLibrary"
    if not lib_dir.exists():
        warnings.warn(
            f"[TheBigPromptLibrary] Directory not found: {lib_dir}. "
            f"Run scripts/clone_repos.sh first.",
            stacklevel=2,
        )
        return []

    print(f"  [TheBigPromptLibrary] Scanning {lib_dir} ...")
    pairs: list[dict] = []

    # --- Category mapping from top-level folder ---
    _LIB_FOLDER_CATEGORY: dict[str, str] = {
        "Jailbreak": "jailbreak",
        "SystemPrompts": "system_prompt_extraction",
        "Security": "system_prompt_extraction",
    }

    _scan_dirs = ["Jailbreak", "SystemPrompts", "Security"]

    for folder in _scan_dirs:
        folder_path = lib_dir / folder
        if not folder_path.exists():
            continue

        category = _LIB_FOLDER_CATEGORY.get(folder, "prompt_injection")

        for md_file in sorted(folder_path.rglob("*.md")):
            if len(pairs) >= max_pairs:
                break
            # Skip README and image-only stubs
            if md_file.name.lower() == "readme.md":
                continue

            try:
                content = md_file.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                warnings.warn(
                    f"[TheBigPromptLibrary] Cannot read {md_file}: {exc}",
                    stacklevel=2,
                )
                continue

            # Skip very short files (likely images or stubs)
            if len(content.strip()) < 30:
                continue

            technique = f"{folder}: {md_file.stem}"
            rel_path = md_file.relative_to(folder_path)
            description = f"From TheBigPromptLibrary/{folder}/{rel_path}"

            # For system prompts, extract the prompt content
            if folder == "SystemPrompts":
                # System prompts get special treatment — they ARE the extraction target
                pairs.append(
                    _make_pair(
                        technique,
                        category,
                        content[:2000],  # Truncate very long prompts
                        description + " — system prompt extraction target",
                    )
                )
            elif folder == "Security":
                pairs.append(
                    _make_pair(
                        technique,
                        "system_prompt_extraction",
                        content[:2000],
                        description + " — GPT protection/prompt defense technique",
                    )
                )
            else:
                pairs.append(
                    _make_pair(technique, category, content[:2000], description)
                )

        # Also scan for .txt files
        for txt_file in sorted(folder_path.rglob("*.txt")):
            if len(pairs) >= max_pairs:
                break

            try:
                content = txt_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            if len(content.strip()) < 30:
                continue

            technique = f"{folder}: {txt_file.stem}"
            rel_path = txt_file.relative_to(folder_path)
            description = f"From TheBigPromptLibrary/{folder}/{rel_path}"
            pairs.append(_make_pair(technique, category, content[:2000], description))

    # De-duplicate
    seen: set[str] = set()
    unique_pairs: list[dict] = []
    for pair in pairs:
        key = pair["messages"][2]["content"]
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)

    print(f"  [TheBigPromptLibrary] Extracted {len(unique_pairs)} pairs")
    return unique_pairs[:max_pairs]


# ---------------------------------------------------------------------------
# Parser: promptmap
# ---------------------------------------------------------------------------
def parse_promptmap(max_pairs: int = 30) -> list[dict]:
    """Parse promptmap YAML rules for attack prompts and conditions.

    Walks ``data/ai_tools/promptmap/rules`` for ``.yaml`` files grouped by
    category subdirectory.  Each YAML file defines an attack rule with
    ``name``, ``type``, ``prompt``, ``pass_conditions``, and ``fail_conditions``.

    Parameters
    ----------
    max_pairs:
        Maximum number of training pairs to return.

    Returns
    -------
    list[dict]
        Training pairs from ``_make_pair``.
    """
    rules_dir = AI_TOOLS_DIR / "promptmap" / "rules"
    if not rules_dir.exists():
        warnings.warn(
            f"[promptmap] Directory not found: {rules_dir}. "
            f"Run scripts/clone_repos.sh first.",
            stacklevel=2,
        )
        return []

    print(f"  [promptmap] Scanning {rules_dir} ...")
    pairs: list[dict] = []

    # --- Category mapping from promptmap subdirectory names ---
    _PROMPTMAP_DIR_CATEGORY: dict[str, str] = {
        "jailbreak": "jailbreak",
        "prompt_stealing": "system_prompt_extraction",
        "harmful": "prompt_injection",
        "hate": "prompt_injection",
        "distraction": "multi_turn",
        "social_bias": "prompt_injection",
    }

    for yaml_file in sorted(rules_dir.rglob("*.yaml")):
        if len(pairs) >= max_pairs:
            break

        # Derive category from parent directory
        parent_dir = yaml_file.parent.name
        category = _PROMPTMAP_DIR_CATEGORY.get(parent_dir, "prompt_injection")

        try:
            with open(yaml_file, "r", encoding="utf-8") as fh:
                doc = yaml.safe_load(fh)
        except (yaml.YAMLError, OSError) as exc:
            warnings.warn(
                f"[promptmap] Cannot parse {yaml_file.name}: {exc}", stacklevel=2
            )
            continue

        if not isinstance(doc, dict):
            continue

        # Some YAML files contain multiple documents separated by "---"
        # We handle single-doc case here; multi-doc loop below.

        # --- Handle both single-doc and list-of-docs ---
        docs = [doc]
        # If yaml.safe_load returned None it was caught above; but also check
        # for files that might have been concatenated (no "---" separator)
        # In practice promptmap files are single-YAML-per-file.

        for entry in docs:
            if not isinstance(entry, dict):
                continue

            name = entry.get("name", yaml_file.stem)
            prompt_text = str(entry.get("prompt", "")).strip()
            if not prompt_text or len(prompt_text) < 10:
                continue

            pass_conds = entry.get("pass_conditions", [])
            fail_conds = entry.get("fail_conditions", [])
            severity = entry.get("severity", "medium")

            # Build description from conditions
            desc_parts = [
                f"promptmap rule '{name}' (severity: {severity})",
            ]
            if pass_conds:
                cond_text = "; ".join(str(c) for c in pass_conds[:3])
                desc_parts.append(f"Pass conditions: {cond_text}")
            if fail_conds:
                cond_text = "; ".join(str(c) for c in fail_conds[:3])
                desc_parts.append(f"Fail conditions: {cond_text}")
            description = " — ".join(desc_parts)

            pairs.append(_make_pair(name, category, prompt_text, description))

    # De-duplicate
    seen: set[str] = set()
    unique_pairs: list[dict] = []
    for pair in pairs:
        key = pair["messages"][2]["content"]
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)

    print(f"  [promptmap] Extracted {len(unique_pairs)} pairs")
    return unique_pairs[:max_pairs]


# ---------------------------------------------------------------------------
# Parser: PyRIT
# ---------------------------------------------------------------------------
def parse_pyrit(max_pairs: int = 30) -> list[dict]:
    """Parse PyRIT jailbreak templates and prompt converter data.

    Walks ``data/ai_tools/PyRIT`` for jailbreak template YAML/JSON files and
    prompt converter configurations.

    Parameters
    ----------
    max_pairs:
        Maximum number of training pairs to return.

    Returns
    -------
    list[dict]
        Training pairs from ``_make_pair``.
    """
    pyrit_root = AI_TOOLS_DIR / "PyRIT"
    if not pyrit_root.exists():
        warnings.warn(
            f"[PyRIT] Directory not found: {pyrit_root}. "
            f"Run scripts/clone_repos.sh first.",
            stacklevel=2,
        )
        return []

    print(f"  [PyRIT] Scanning {pyrit_root} ...")
    pairs: list[dict] = []

    # --- Walk for YAML/JSON template files ---
    dataset_dirs = [
        pyrit_root / "datasets",
        pyrit_root / "data",
        pyrit_root / "doc",
        pyrit_root,
    ]

    for search_dir in dataset_dirs:
        if not search_dir.exists():
            continue

        # YAML files (jailbreak templates)
        for yml_file in sorted(search_dir.rglob("*.yaml")) + sorted(
            search_dir.rglob("*.yml")
        ):
            if len(pairs) >= max_pairs:
                break
            try:
                with open(yml_file, "r", encoding="utf-8") as fh:
                    doc = yaml.safe_load(fh)
            except (yaml.YAMLError, OSError):
                continue

            if not isinstance(doc, dict):
                continue

            # Extract prompt-template fields
            prompt_text = ""
            for field in ("prompt", "template", "text", "content"):
                val = doc.get(field)
                if val and isinstance(val, str) and len(val.strip()) >= 10:
                    prompt_text = val.strip()
                    break

            if not prompt_text:
                continue

            name = doc.get("name", yml_file.stem)
            category = "jailbreak"

            # Infer category from tags/keywords
            tags = doc.get("tags", []) or doc.get("categories", [])
            if isinstance(tags, list):
                tag_str = " ".join(str(t).lower() for t in tags)
                if "injection" in tag_str:
                    category = "prompt_injection"
                elif "extraction" in tag_str or "steal" in tag_str:
                    category = "system_prompt_extraction"

            description = f"PyRIT template from {yml_file.relative_to(pyrit_root)}"
            pairs.append(_make_pair(name, category, prompt_text, description))

        # JSON files
        for json_file in sorted(search_dir.rglob("*.json")):
            if len(pairs) >= max_pairs:
                break
            try:
                with open(json_file, "r", encoding="utf-8") as fh:
                    content = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue

            prompts: list[str] = []
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, str) and len(item.strip()) >= 20:
                        prompts.append(item.strip())
                    elif isinstance(item, dict):
                        for field in ("prompt", "text", "content", "template"):
                            val = item.get(field)
                            if val and isinstance(val, str) and len(val.strip()) >= 20:
                                prompts.append(val.strip())

            for idx, prompt_text in enumerate(prompts):
                if len(pairs) >= max_pairs:
                    break
                technique = f"PyRIT: {json_file.stem}"
                if len(prompts) > 1:
                    technique += f" ({idx + 1}/{len(prompts)})"
                description = f"PyRIT data from {json_file.relative_to(pyrit_root)}"
                pairs.append(
                    _make_pair(technique, "jailbreak", prompt_text, description)
                )

    # --- Walk for Python prompt converter files ---
    for py_file in sorted(pyrit_root.rglob("*.py")):
        if len(pairs) >= max_pairs:
            break
        try:
            text = py_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        # Quick heuristic: look for converter class names
        if "converter" not in text.lower() and "template" not in text.lower():
            continue

        # Extract string constants that look like transformative patterns
        _PY_STR_RE = re.compile(r'"""([\s\S]{30,}?)"""')
        for match in _PY_STR_RE.finditer(text):
            if len(pairs) >= max_pairs:
                break
            payload = match.group(1).strip()
            if len(payload) < 20:
                continue
            technique = f"PyRIT converter: {py_file.stem}"
            description = f"PyRIT prompt converter from {py_file.name}"
            category = "token_smuggling"
            pairs.append(_make_pair(technique, category, payload, description))

    # De-duplicate
    seen: set[str] = set()
    unique_pairs: list[dict] = []
    for pair in pairs:
        key = pair["messages"][2]["content"]
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)

    print(f"  [PyRIT] Extracted {len(unique_pairs)} pairs")
    return unique_pairs[:max_pairs]


# ---------------------------------------------------------------------------
# Parser: FuzzyAI
# ---------------------------------------------------------------------------
def parse_fuzzyai(max_pairs: int = 50) -> list[dict]:
    """Parse FuzzyAI adversarial prompt resources.

    Reads text files and CSV from
    ``data/ai_tools/FuzzyAI/src/fuzzyai/resources/`` including
    ``adv_prompts.txt``, ``adv_suffixes.txt``, ``pandoras_prompts.txt``,
    ``harmful_behaviors.csv``, and ``Jailbreaks.md``.

    Parameters
    ----------
    max_pairs:
        Maximum number of training pairs to return.

    Returns
    -------
    list[dict]
        Training pairs from ``_make_pair``.
    """
    resources_dir = AI_TOOLS_DIR / "FuzzyAI" / "src" / "fuzzyai" / "resources"
    if not resources_dir.exists():
        warnings.warn(
            f"[FuzzyAI] Directory not found: {resources_dir}. "
            f"Run scripts/clone_repos.sh first.",
            stacklevel=2,
        )
        return []

    print(f"  [FuzzyAI] Scanning {resources_dir} ...")
    pairs: list[dict] = []

    # --- adv_prompts.txt (one prompt per line) ---
    adv_prompts = resources_dir / "adv_prompts.txt"
    if adv_prompts.exists():
        try:
            text = adv_prompts.read_text(encoding="utf-8", errors="ignore")
            for idx, line in enumerate(text.splitlines()):
                if len(pairs) >= max_pairs:
                    break
                line = line.strip()
                if len(line) < 10:
                    continue
                technique = f"FuzzyAI adv_prompt #{idx + 1}"
                description = f"Adversarial prompt from FuzzyAI adv_prompts.txt"
                pairs.append(
                    _make_pair(technique, "prompt_injection", line, description)
                )
        except OSError as exc:
            warnings.warn(f"[FuzzyAI] Cannot read adv_prompts.txt: {exc}", stacklevel=2)

    # --- adv_suffixes.txt (one suffix per line, these are token-smuggling) ---
    adv_suffixes = resources_dir / "adv_suffixes.txt"
    if adv_suffixes.exists():
        try:
            text = adv_suffixes.read_text(encoding="utf-8", errors="ignore")
            for idx, line in enumerate(text.splitlines()):
                if len(pairs) >= max_pairs:
                    break
                line = line.strip()
                if len(line) < 5:
                    continue
                technique = f"FuzzyAI adv_suffix #{idx + 1}"
                description = f"Adversarial suffix from FuzzyAI adv_suffixes.txt"
                pairs.append(
                    _make_pair(technique, "token_smuggling", line, description)
                )
        except OSError as exc:
            warnings.warn(
                f"[FuzzyAI] Cannot read adv_suffixes.txt: {exc}", stacklevel=2
            )

    # --- pandoras_prompts.txt (one prompt per line) ---
    pandoras = resources_dir / "pandoras_prompts.txt"
    if pandoras.exists():
        try:
            text = pandoras.read_text(encoding="utf-8", errors="ignore")
            for idx, line in enumerate(text.splitlines()):
                if len(pairs) >= max_pairs:
                    break
                line = line.strip()
                if len(line) < 10:
                    continue
                technique = f"FuzzyAI pandora #{idx + 1}"
                description = f"Pandora's prompt from FuzzyAI pandoras_prompts.txt"
                pairs.append(_make_pair(technique, "jailbreak", line, description))
        except OSError as exc:
            warnings.warn(
                f"[FuzzyAI] Cannot read pandoras_prompts.txt: {exc}", stacklevel=2
            )

    # --- harmful_behaviors.csv (goal, target columns) ---
    csv_path = resources_dir / "harmful_behaviors.csv"
    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8", errors="ignore") as fh:
                reader = csv.DictReader(fh)
                for idx, row in enumerate(reader):
                    if len(pairs) >= max_pairs:
                        break
                    goal = row.get("goal", "").strip()
                    target = row.get("target", "").strip()
                    if not goal:
                        continue
                    technique = f"FuzzyAI harmful_behavior #{idx + 1}"
                    description = f"Harmful behavior from FuzzyAI harmful_behaviors.csv"
                    if target:
                        description += f" — target output: {target[:200]}"
                    prompt_text = goal
                    pairs.append(
                        _make_pair(
                            technique, "prompt_injection", prompt_text, description
                        )
                    )
        except (OSError, csv.Error) as exc:
            warnings.warn(
                f"[FuzzyAI] Cannot read harmful_behaviors.csv: {exc}", stacklevel=2
            )

    # --- Jailbreaks.md (documented jailbreak techniques) ---
    jailbreaks_md = resources_dir / "Jailbreaks.md"
    if jailbreaks_md.exists():
        try:
            text = jailbreaks_md.read_text(encoding="utf-8", errors="ignore")
            # Split on ### or ## headers (each is a technique)
            sections = re.split(r"\n#{1,3}\s+", text)
            for section in sections:
                if len(pairs) >= max_pairs:
                    break
                section = section.strip()
                if len(section) < 30:
                    continue

                # Extract header as technique name
                header_end = section.find("\n")
                if header_end == -1:
                    header_end = min(60, len(section))
                technique_name = section[:header_end].strip().rstrip(")")
                # Clean up numbered lists etc.
                technique_name = re.sub(r"^[\d.]+\s*", "", technique_name).strip()
                if not technique_name:
                    technique_name = "FuzzyAI jailbreak technique"

                # Map categories heuristically
                lower = section.lower()
                if any(
                    kw in lower for kw in ("rot", "cipher", "encod", "remap", "map")
                ):
                    category = "token_smuggling"
                elif any(kw in lower for kw in ("system", "prompt extract", "leak")):
                    category = "system_prompt_extraction"
                elif any(kw in lower for kw in ("multi-turn", "conversation", "step")):
                    category = "multi_turn"
                else:
                    category = "jailbreak"

                description = (
                    f"Documented jailbreak technique from FuzzyAI Jailbreaks.md"
                )
                pairs.append(
                    _make_pair(technique_name, category, section[:2000], description)
                )
        except OSError as exc:
            warnings.warn(f"[FuzzyAI] Cannot read Jailbreaks.md: {exc}", stacklevel=2)

    # De-duplicate
    seen: set[str] = set()
    unique_pairs: list[dict] = []
    for pair in pairs:
        key = pair["messages"][2]["content"]
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)

    print(f"  [FuzzyAI] Extracted {len(unique_pairs)} pairs")
    return unique_pairs[:max_pairs]


# ---------------------------------------------------------------------------
# Aggregator: parse all repos
# ---------------------------------------------------------------------------
def parse_all() -> tuple[list[dict], dict[str, int]]:
    """Run all parsers and combine results.

    Returns
    -------
    tuple[list[dict], dict[str, int]]
        A 2-tuple of ``(all_pairs, stats)`` where *stats* maps each repo
        name to the number of pairs extracted.
    """
    parsers: list[tuple[str, callable]] = [
        ("promptfoo", parse_promptfoo),
        ("garak", parse_garak),
        ("TheBigPromptLibrary", parse_big_prompt_library),
        ("promptmap", parse_promptmap),
        ("PyRIT", parse_pyrit),
        ("FuzzyAI", parse_fuzzyai),
    ]

    all_pairs: list[dict] = []
    stats: dict[str, int] = {}

    for name, parser_fn in parsers:
        try:
            pairs = parser_fn()
            count = len(pairs)
            stats[name] = count
            all_pairs.extend(pairs)
            print(f"  [{name}] {count} pairs extracted")
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"[{name}] Parser failed: {exc}", stacklevel=2)
            stats[name] = 0

    total = len(all_pairs)
    print(f"\n  Total training pairs: {total}")
    for name, count in stats.items():
        print(f"    {name}: {count}")

    return all_pairs, stats


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def write_output(pairs: list[dict], output_path: Path) -> int:
    """Write training pairs to JSONL file. Returns count written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    return len(pairs)


def split_by_category(pairs: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Split pairs into injection, jailbreak, and other categories."""
    injection: list[dict] = []
    jailbreak: list[dict] = []
    other: list[dict] = []
    for pair in pairs:
        content = pair["messages"][2]["content"]
        if "injection" in content.lower() or "T1566" in content:
            injection.append(pair)
        elif "jailbreak" in content.lower() or "T1548" in content:
            jailbreak.append(pair)
        else:
            other.append(pair)
    return injection, jailbreak, other


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract prompt injection/jailbreak training data from AI tool repos."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for JSONL files (default: data/datasets/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse repos and print stats without writing files",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Only process one repo (promptfoo, garak, bigprompts, promptmap, pyrit, fuzzyai)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DATASETS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("AttackLM — Extract AI Tool Prompt Data")
    print(f"  AI tools dir: {AI_TOOLS_DIR}")
    print(f"  Output dir:   {output_dir}")
    print()

    # Parse
    if args.repo:
        parsers = {
            "promptfoo": parse_promptfoo,
            "garak": parse_garak,
            "bigprompts": parse_big_prompt_library,
            "promptmap": parse_promptmap,
            "pyrit": parse_pyrit,
            "fuzzyai": parse_fuzzyai,
        }
        if args.repo not in parsers:
            print(f"ERROR: Unknown repo '{args.repo}'. Options: {list(parsers.keys())}")
            sys.exit(1)
        pairs = parsers[args.repo]()
        stats = {args.repo: len(pairs)}
    else:
        pairs, stats = parse_all()

    total = len(pairs)

    if args.dry_run:
        print(f"\n{'=' * 60}")
        print("DRY RUN — Statistics")
        print(f"{'=' * 60}")
        print(f"  Total pairs extracted: {total}")
        for repo, count in sorted(stats.items()):
            print(f"    {repo:25s}: {count:5d}")
        injection, jailbreak, other = split_by_category(pairs)
        print(f"  By category:")
        print(f"    Prompt Injection: {len(injection)}")
        print(f"    Jailbreak:        {len(jailbreak)}")
        print(f"    Other:            {len(other)}")
        return

    # Write combined dataset
    combined_path = output_dir / "prompt_injection_dataset.jsonl"
    written = write_output(pairs, combined_path)
    print(f"\n  Combined dataset: {written} pairs → {combined_path}")

    # Write jailbreak-only dataset
    _, jailbreak, _ = split_by_category(pairs)
    if jailbreak:
        jailbreak_path = output_dir / "jailbreak_dataset.jsonl"
        jb_written = write_output(jailbreak, jailbreak_path)
        print(f"  Jailbreak dataset: {jb_written} pairs → {jailbreak_path}")

    # Stats
    print(f"\n{'=' * 60}")
    print(" Extraction Complete")
    print(f"{'=' * 60}")
    print(f"  Total pairs: {total}")
    for repo, count in sorted(stats.items()):
        pct = count / total * 100 if total > 0 else 0
        print(f"    {repo:25s}: {count:5d} ({pct:.1f}%)")

    injection, jailbreak, other = split_by_category(pairs)
    print(f"\n  By category:")
    print(f"    Prompt Injection: {len(injection)}")
    print(f"    Jailbreak:        {len(jailbreak)}")
    print(f"    Other:            {len(other)}")


if __name__ == "__main__":
    main()
