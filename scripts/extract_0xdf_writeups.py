#!/usr/bin/env python3
# CREDITS — DATA SOURCE ATTRIBUTION
# ----------------------------------
# This script transforms data from: 0xdf's CTF/HackTheBox Writeups
# Website:   https://0xdf.gitlab.io/
# License:   Public blog content (fair use for research/education)
# Author:    0xdf
#
# The output JSONL is a *transformation* of publicly available blog posts
# into OpenAI-style chat triples for security training purposes.
# See /ATTRIBUTION.md for full per-source attribution and re-distribution
# guidance.
# ----------------------------------
"""Extract 0xdf CTF/HackTheBox writeup posts into AttackLM JSONL training pairs.

Scrapes the 0xdf.gitlab.io blog (via sitemap.xml), extracts post content,
chunks long posts by section headers, identifies MITRE ATT&CK technique
IDs, and generates instruction-response pairs for supervised fine-tuning.

Output: ``data/datasets/buckets/sources/0xdf-writeups/ctf_walkthroughs/data.jsonl``

Usage:
    python scripts/extract_0xdf_writeups.py
    python scripts/extract_0xdf_writeups.py --max-posts 3 --output-dir /tmp/test_0xdf
    python scripts/extract_0xdf_writeups.py --dry-run --max-posts 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    print(
        "ERROR: 'requests' package required. Install with: pip install requests",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from bs4 import BeautifulSoup
except ImportError:
    print(
        "ERROR: 'beautifulsoup4' package required. Install with: pip install beautifulsoup4",
        file=sys.stderr,
    )
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from mitre_tactic_lookup import get_tactic_for_technique, get_tactic_name

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "data" / ".cache" / "0xdf-writeups"
DEFAULT_OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "datasets"
    / "buckets"
    / "sources"
    / "0xdf-writeups"
    / "ctf_walkthroughs"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://0xdf.gitlab.io"
SITEMAP_URL = f"{BASE_URL}/sitemap.xml"
SITE_URLS_TO_SKIP = {
    # Skip non-post pages
    "cheatsheets",
    "hackvent",
    "adventofcode",
    "holidayhack",
    "flare-on",
    "flagvent",
}

SYSTEM_MSG = (
    "You are a cybersecurity expert specializing in CTF challenges and "
    "penetration testing. Provide detailed walkthrough explanations including "
    "reconnaissance, exploitation techniques, privilege escalation, and "
    "MITRE ATT&CK mappings."
)

ATTRIBUTION = {
    "source": "0xdf-writeups",
    "source_uri": "https://0xdf.gitlab.io/",
    "license": "Fair-Use-Research",
    "license_uri": "https://0xdf.gitlab.io/",
    "rights_contact": "0xdf",
    "attribution_text": (
        "Content from 0xdf's CTF/HackTheBox writeups (0xdf.gitlab.io). "
        "Used for research and educational purposes under fair use."
    ),
}

# ---------------------------------------------------------------------------
# MITRE ATT&CK technique ID regex
# ---------------------------------------------------------------------------
_MITRE_RE = re.compile(r"\b(T\d{4}(?:\.\d{3})?)\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Post category classification
# ---------------------------------------------------------------------------
_POST_TYPE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"htb[-]", re.IGNORECASE), "HackTheBox"),
    (re.compile(r"sherlock", re.IGNORECASE), "HackTheBox-Sherlock"),
    (re.compile(r"flare[-]on", re.IGNORECASE), "Flare-On"),
    (re.compile(r"hackvent", re.IGNORECASE), "HackVent"),
    (re.compile(r"holidayhack", re.IGNORECASE), "HolidayHack"),
    (re.compile(r"flagvent", re.IGNORECASE), "FlagVent"),
    (re.compile(r"adventofcode", re.IGNORECASE), "AdventOfCode"),
    (re.compile(r"ctf", re.IGNORECASE), "CTF"),
]


def classify_post(url: str, title: str) -> str:
    """Classify the post type from URL and title."""
    text = url.lower() + " " + title.lower()
    for pattern, category in _POST_TYPE_PATTERNS:
        if pattern.search(text):
            return category
    return "CTF"


def extract_machine_name(url: str, title: str) -> str:
    """Extract the machine/challenge name from URL or title.

    For HTB posts like /2024/12/21/htb-sea.html -> 'Sea'
    For CTF posts, use the title.
    """
    # Try to extract from URL pattern: htb-<name>.html
    m = re.search(
        r"htb[-]([a-z0-9]+)(?:[-](?:linux|win|beyond[-]root))?\.html",
        url,
        re.IGNORECASE,
    )
    if m:
        name = m.group(1)
        # Handle two-part names like "htb-codetwo" -> "CodeTwo"
        return re.sub(r"[-]", " ", name).title()

    # For non-HTB URLs, use the last path segment
    path = urlparse(url).path.rstrip("/")
    slug = path.rsplit("/", 1)[-1]
    slug = slug.replace(".html", "")
    return slug.replace("-", " ").title()


# ---------------------------------------------------------------------------
# Sitemap parsing
# ---------------------------------------------------------------------------
def fetch_sitemap(delay: float = 1.0) -> list[str]:
    """Fetch and parse the sitemap.xml to get all post URLs."""
    print(f"  Fetching sitemap: {SITEMAP_URL}", file=sys.stderr)
    try:
        resp = requests.get(SITEMAP_URL, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  ERROR: Failed to fetch sitemap: {exc}", file=sys.stderr)
        return []

    soup = BeautifulSoup(resp.text, "xml")
    urls: list[str] = []
    for loc in soup.find_all("url"):
        loc_tag = loc.find("loc")
        if loc_tag and loc_tag.string:
            url = loc_tag.string.strip()
            # Only include blog post URLs (have date pattern or are challenge pages)
            if _is_post_url(url):
                urls.append(url)

    print(f"  Found {len(urls)} post URLs in sitemap", file=sys.stderr)
    return urls


def _is_post_url(url: str) -> str | None:
    """Check if a URL is a blog post (not a category/index page).

    Returns the URL if it's a post, None otherwise.
    """
    path = urlparse(url).path

    # Skip root index pages
    if path in ("", "/", "/index.html"):
        return None

    # Skip cheatsheet pages (not walkthroughs)
    if path.startswith("/cheatsheets"):
        return None

    # Skip /hackvent and /flagvent difficulty-level pages (easy/medium/hard/leet)
    # but include specific challenge pages
    # e.g. /hackvent2024/easy is an index, /hackvent2024/day01 would be a challenge

    # Include URLs with date pattern: /YYYY/MM/DD/
    if re.search(r"/\d{4}/\d{2}/d{2}/", url):
        return url

    # Include HTB posts
    if "/htb-" in url.lower():
        return url

    # Include specific challenge pages (have a sub-path after the event)
    # e.g. /holidayhack2024/act-i/hardware is a challenge
    # but /holidayhack2024/ is an index
    if re.match(r"https://0xdf\.gitlab\.io/\d{4}/\d{2}/\d{2}/", url):
        return url

    # Include flare-on, holidayhack challenge pages (not just the index)
    for prefix in ("flare-on-", "holidayhack", "hackvent"):
        if f"/{prefix}" in url.lower():
            # Must have something after the prefix/index
            parts = path.strip("/").split("/")
            if len(parts) >= 2:
                return url
            return None

    # Default: include if it has a date pattern
    if re.search(r"\d{4}/\d{2}/\d{2}", url):
        return url

    return None


# ---------------------------------------------------------------------------
# Page fetching with caching
# ---------------------------------------------------------------------------
def _cache_path(url: str) -> Path:
    """Generate a deterministic cache file path for a URL."""
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    # Make filename from URL slug
    path = urlparse(url).path.strip("/").replace("/", "_")
    if len(path) > 80:
        path = path[:80]
    return CACHE_DIR / f"{path}_{url_hash}.html"


def fetch_page(url: str, delay: float = 1.0, use_cache: bool = True) -> str | None:
    """Fetch a page with caching and rate limiting.

    Returns HTML content string, or None on failure.
    """
    cache_file = _cache_path(url)

    # Check cache
    if use_cache and cache_file.exists():
        try:
            return cache_file.read_text(encoding="utf-8")
        except OSError:
            pass

    # Rate limit
    time.sleep(delay)

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        html = resp.text
    except requests.RequestException as exc:
        print(f"  [WARN] Failed to fetch {url}: {exc}", file=sys.stderr)
        return None

    # Save to cache
    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            cache_file.write_text(html, encoding="utf-8")
        except OSError:
            pass

    return html


# ---------------------------------------------------------------------------
# Post content parsing
# ---------------------------------------------------------------------------
def parse_post(html: str, url: str) -> dict[str, Any] | None:
    """Parse an HTML page and extract post content.

    Returns dict with keys: title, date, categories, tags, sections, full_text,
    url. Returns None if not a valid post.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Extract title
    title_tag = soup.find("h1", class_="post-title") or soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else ""
    if not title:
        # Try <title> tag
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else urlparse(url).path

    # Extract date
    date_tag = soup.find("time") or soup.find("span", class_="post-date")
    date_str = ""
    if date_tag:
        date_str = date_tag.get("datetime", "") or date_tag.get_text(strip=True)

    # Extract categories/tags
    categories: list[str] = []
    tags: list[str] = []
    for cat_tag in soup.find_all("a", rel="category tag"):
        cat_text = cat_tag.get_text(strip=True)
        if cat_text:
            categories.append(cat_text)
    for tag_tag in soup.find_all("a", rel="tag"):
        tag_text = tag_tag.get_text(strip=True)
        if tag_text:
            tags.append(tag_text)
    # Also check meta tags for keywords
    meta_tags = soup.find("meta", attrs={"name": "keywords"})
    if meta_tags and meta_tags.get("content"):
        for kw in meta_tags["content"].split(","):
            kw = kw.strip()
            if kw and kw not in tags:
                tags.append(kw)

    # Extract main content
    content_div = (
        soup.find("div", class_="post-content")
        or soup.find("article")
        or soup.find("div", class_="entry-content")
        or soup.find("div", class_="content")
    )

    if not content_div:
        # Fallback: find the main content area
        content_div = soup.find("main") or soup.find("div", role="main")

    if not content_div:
        return None

    # Remove script and style tags
    for tag in content_div.find_all(["script", "style", "nav", "footer"]):
        tag.decompose()

    # Chunk by section headers (h2, h3)
    sections = _chunk_by_headers(content_div, title)

    full_text = content_div.get_text(separator="\n", strip=True)

    # Skip very short posts (likely not walkthroughs)
    if len(full_text) < 200:
        return None

    return {
        "title": title,
        "date": date_str,
        "categories": categories,
        "tags": tags,
        "sections": sections,
        "full_text": full_text,
        "url": url,
    }


def _chunk_by_headers(
    content_div: BeautifulSoup, fallback_title: str
) -> list[dict[str, str]]:
    """Chunk content by h2/h3 headers into sections.

    Returns list of dicts with 'header' and 'content' keys.
    """
    sections: list[dict[str, str]] = []
    current_header = fallback_title
    current_parts: list[str] = []

    for element in content_div.find_all(
        ["h2", "h3", "p", "pre", "ul", "ol", "blockquote", "table"]
    ):
        if element.name in ("h2", "h3"):
            # Save previous section if it has content
            if current_parts:
                content = "\n\n".join(current_parts).strip()
                if len(content) >= 50:
                    sections.append({"header": current_header, "content": content})
            current_header = element.get_text(strip=True)
            current_parts = []
        else:
            text = element.get_text(separator=" ", strip=True)
            if text:
                current_parts.append(text)

    # Don't forget the last section
    if current_parts:
        content = "\n\n".join(current_parts).strip()
        if len(content) >= 50:
            sections.append({"header": current_header, "content": content})

    # If no sections found, use the whole content as one section
    if not sections:
        full_text = content_div.get_text(separator="\n", strip=True)
        if len(full_text) >= 50:
            sections.append({"header": fallback_title, "content": full_text})

    return sections


# ---------------------------------------------------------------------------
# MITRE technique extraction
# ---------------------------------------------------------------------------
def extract_mitre_ids(text: str) -> list[str]:
    """Extract MITRE ATT&CK technique IDs from text."""
    matches = _MITRE_RE.findall(text)
    # Normalize to uppercase
    return sorted(set(m.upper() for m in matches))


# ---------------------------------------------------------------------------
# Training pair generation
# ---------------------------------------------------------------------------
def generate_pairs_from_post(post: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate training pairs from a parsed post."""
    pairs: list[dict[str, Any]] = []
    title = post["title"]
    sections = post["sections"]
    url = post["url"]
    categories = post["categories"]
    tags = post["tags"]
    post_type = classify_post(url, title)
    machine_name = extract_machine_name(url, title)

    # Extract MITRE IDs from full text
    all_mitre_ids = extract_mitre_ids(post["full_text"])

    # Generate one overview pair from the full post
    overview_text = post["full_text"]
    # Truncate if very long (max 4000 chars for overview)
    if len(overview_text) > 4000:
        overview_text = overview_text[:4000] + "\n\n[... content truncated]"

    # Build technique listing for the instruction
    technique_str = ""
    if all_mitre_ids:
        technique_names = []
        for tid in all_mitre_ids[:5]:
            tactic = get_tactic_for_technique(tid)
            tactic_name = get_tactic_name(tactic) if tactic else ""
            technique_names.append(f"{tid} ({tactic_name})" if tactic_name else tid)
        technique_str = f" The following MITRE ATT&CK techniques are relevant: {', '.join(technique_names)}."

    # Pair 1: Full walkthrough overview
    user_instruction = (
        f"Walk through the exploitation of {machine_name} ({post_type}). "
        f"What techniques were used for initial access, privilege escalation, and persistence?{technique_str}"
    )

    assistant_response = f"## {title}\n\n{overview_text}"

    overview_pair = {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_instruction},
            {"role": "assistant", "content": assistant_response},
        ],
        "mitre_ids": all_mitre_ids,
        "post_type": post_type,
        "machine_name": machine_name,
        "url": url,
        **ATTRIBUTION,
    }

    # Add tactic info
    _add_tactic_info(overview_pair, all_mitre_ids)
    pairs.append(overview_pair)

    # Generate section-level pairs for long posts
    if len(sections) > 1:
        for section in sections:
            section_text = section["content"]
            # Skip very short sections
            if len(section_text) < 100:
                continue

            # Truncate long sections
            if len(section_text) > 3000:
                section_text = section_text[:3000] + "\n\n[... truncated]"

            section_mitre = extract_mitre_ids(section_text)
            section_header = section["header"]

            # Determine instruction based on section content
            section_instruction = _instruction_for_section(
                section_header, section_text, machine_name, post_type, section_mitre
            )

            section_pair = {
                "messages": [
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": section_instruction},
                    {
                        "role": "assistant",
                        "content": f"## {title} — {section_header}\n\n{section_text}",
                    },
                ],
                "mitre_ids": section_mitre,
                "post_type": post_type,
                "machine_name": machine_name,
                "url": url,
                **ATTRIBUTION,
            }

            _add_tactic_info(section_pair, section_mitre)
            pairs.append(section_pair)

    return pairs


def _instruction_for_section(
    header: str,
    content: str,
    machine_name: str,
    post_type: str,
    mitre_ids: list[str],
) -> str:
    """Generate a contextually appropriate instruction for a section."""
    header_lower = header.lower()
    content_lower = content.lower()

    # Map section headers to appropriate questions
    if any(kw in header_lower for kw in ("recon", "nmap", "port scan", "enumerat")):
        return (
            f"How would you enumerate {machine_name} ({post_type})? "
            f"What ports, services, and vulnerabilities would you discover?"
        )
    elif any(
        kw in header_lower
        for kw in ("shell", "foothold", "initial access", "web shell", "rce", "exploit")
    ):
        technique_str = f" (MITRE: {', '.join(mitre_ids)})" if mitre_ids else ""
        return (
            f"How do you gain initial access/foothold on {machine_name} ({post_type})?{technique_str} "
            f"Explain the exploitation technique and steps."
        )
    elif any(
        kw in header_lower
        for kw in ("privesc", "privilege escalat", "root", "admin", "escalat")
    ):
        technique_str = f" (MITRE: {', '.join(mitre_ids)})" if mitre_ids else ""
        return (
            f"How do you escalate privileges on {machine_name} ({post_type})?{technique_str} "
            f"Describe the privilege escalation path."
        )
    elif any(kw in header_lower for kw in ("lateral", "pivot", "network", "tunnel")):
        return (
            f"How do you perform lateral movement or pivoting on {machine_name} ({post_type})? "
            f"Describe the network traversal techniques."
        )
    elif any(kw in header_lower for kw in ("beyond root", "alternative", "extra")):
        return (
            f"What additional or alternative exploitation paths exist on {machine_name} ({post_type})? "
            f"Describe beyond-root findings."
        )
    else:
        # Generic section instruction
        technique_str = f" (MITRE: {', '.join(mitre_ids)})" if mitre_ids else ""
        return (
            f"Explain the '{header}' phase of the {machine_name} ({post_type}) "
            f"exploitation.{technique_str}"
        )


def _add_tactic_info(pair: dict[str, Any], mitre_ids: list[str]) -> None:
    """Add tactic information to a pair dict."""
    if mitre_ids:
        for tech_id in mitre_ids:
            tactic_id = get_tactic_for_technique(tech_id)
            if tactic_id:
                pair["mitre_tactic_id"] = tactic_id
                tactic_name = get_tactic_name(tactic_id)
                if tactic_name:
                    pair["tactic"] = tactic_name
                    pair["kill_chain_phase"] = tactic_name
                break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract 0xdf CTF/HackTheBox writeups into AttackLM JSONL training pairs.",
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=0,
        help="Maximum number of posts to process (0 = all). Useful for testing.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between HTTP requests (default: 1.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for JSONL.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and parse posts without writing output.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable local page caching.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_path = output_dir / "data.jsonl"

    print("AttackLM — Extract 0xdf CTF/HackTheBox Writeups")
    print(f"  Source:    {BASE_URL}")
    print(f"  Output:   {output_path}")
    print(f"  Cache:    {CACHE_DIR}")
    print(f"  Delay:    {args.delay}s")
    print(f"  Max posts: {'all' if args.max_posts == 0 else args.max_posts}")
    print()

    # Step 1: Fetch sitemap
    urls = fetch_sitemap(delay=0)
    if not urls:
        print("ERROR: No URLs found in sitemap.", file=sys.stderr)
        return 1

    # Filter to blog posts only (skip cheatsheets, indexes)
    post_urls: list[str] = []
    for url in urls:
        result = _is_post_url(url)
        if result:
            post_urls.append(result)

    # Sort by date extracted from URL (newest first), with HTB posts prioritized
    def _sort_key(url: str) -> tuple:
        """Sort key: date-based URLs first (newest first), then others."""
        m = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", url)
        if m:
            # Negative date so reverse=True gives newest first
            return (
                0,
                -(int(m.group(1)) * 10000 + int(m.group(2)) * 100 + int(m.group(3))),
            )
        # Non-date URLs sort after date-based ones
        return (1, url)

    post_urls.sort(key=_sort_key)

    if args.max_posts > 0:
        post_urls = post_urls[: args.max_posts]

    print(f"  Filtered to {len(post_urls)} blog post URLs", file=sys.stderr)
    print()

    # Step 2: Fetch and parse each post
    all_pairs: list[dict[str, Any]] = []
    post_types: dict[str, int] = {}
    skipped = 0
    errors = 0

    for i, url in enumerate(post_urls, 1):
        print(f"  [{i}/{len(post_urls)}] {url}", file=sys.stderr)

        html = fetch_page(url, delay=args.delay, use_cache=not args.no_cache)
        if html is None:
            errors += 1
            continue

        post = parse_post(html, url)
        if post is None:
            skipped += 1
            print(f"    SKIPPED: Could not parse content", file=sys.stderr)
            continue

        # Generate pairs
        pairs = generate_pairs_from_post(post)
        if not pairs:
            skipped += 1
            continue

        post_type = classify_post(url, post["title"])
        post_types[post_type] = post_types.get(post_type, 0) + 1

        all_pairs.extend(pairs)
        mitre_str = ""
        pair_mitre = set()
        for p in pairs:
            pair_mitre.update(p.get("mitre_ids", []))
        if pair_mitre:
            mitre_str = f" | MITRE: {', '.join(sorted(pair_mitre)[:5])}"

        print(
            f"    {post['title'][:60]} → {len(pairs)} pairs{mitre_str}",
            file=sys.stderr,
        )

    # Step 3: Summary
    all_mitre: set[str] = set()
    for pair in all_pairs:
        all_mitre.update(pair.get("mitre_ids", []))

    print(f"\n{'=' * 60}")
    print(f"  Posts fetched:    {len(post_urls)}")
    print(f"  Posts parsed:     {len(post_urls) - skipped - errors}")
    print(f"  Posts skipped:    {skipped}")
    print(f"  Fetch errors:     {errors}")
    print(f"  Total pairs:      {len(all_pairs)}")
    print(f"  Unique MITRE IDs: {len(all_mitre)}")
    if all_mitre:
        print(
            f"    {', '.join(sorted(all_mitre)[:20])}{'...' if len(all_mitre) > 20 else ''}"
        )
    print(f"\n  By post type:")
    for ptype, count in sorted(post_types.items()):
        print(f"    {ptype:25s}: {count}")

    if args.dry_run:
        print(f"\n{'=' * 60}")
        print("  DRY RUN — No files written")
        print(f"{'=' * 60}")

        # Show sample pair
        if all_pairs:
            print(f"\n  Sample pair (first):\n")
            sample = all_pairs[0]
            sample_json = json.dumps(sample, indent=2, ensure_ascii=False)
            print(sample_json[:3000])
            if len(sample_json) > 3000:
                print("  ... (truncated)")
        return 0

    # Step 4: Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n  Output written: {len(all_pairs)} pairs → {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
