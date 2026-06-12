"""
Blob callback for git-filter-repo to scrub BigPromptLibrary records from
any historical version of `data/datasets/buckets/ai/jailbreaking/data.jsonl`.

Drop JSONL lines whose user message is a "How do I perform Jailbreak: ..."
prompt (these are the 6 BPL records, copyright-laundering risk).

The body is wrapped by filter-repo inside a function `blob_callback(blob)`,
so this file must be valid as a function body. Top-level `import` is OK;
top-level `def` would be discarded, so use plain statements.
"""

import sys

BPL_MARKER = b'"role": "user", "content": "How do I perform Jailbreak: '
SPECIALIST_MARKER = b"Prompt Injection specialist"

data = blob.data
if BPL_MARKER in data and SPECIALIST_MARKER in data:
    lines = data.split(b"\n")
    kept = []
    dropped = 0
    for line in lines:
        if BPL_MARKER in line:
            dropped += 1
            continue
        kept.append(line)
    if dropped > 0:
        new_data = b"\n".join(kept)
        print(
            f"  [BPL scrub] dropped {dropped} records from blob "
            f"({len(data):,} -> {len(new_data):,} bytes)",
            file=sys.stderr,
        )
        blob.data = new_data
