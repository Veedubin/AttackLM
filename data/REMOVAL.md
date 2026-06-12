# Removal Requests

**Last updated:** 2026-06-11

If you are a rights holder for any source used in the AttackLM dataset
and would like records derived from your work removed, this document
describes the process.

**The author will not dispute removal requests. Removal is fast,
unconditional, and free.**

---

## Who can request removal

- **Upstream rights holders** — copyright holders for any source
  listed in [`data/LEGAL.md`](LEGAL.md) or
  `data/datasets/buckets/sources/_index.json`.
- **Designated agents** — anyone with written authorization from a
  rights holder (legal counsel, employer, authorized representative).
- **Affected individuals** — if you believe a record contains
  identifying information about you (a name, a username, a real
  command you authored, etc.), you can also request removal.

---

## What you can request

You can request any of the following:

1. **Full source removal** — remove all records derived from a single
   upstream source (e.g. all 13,997 Metasploit-derived records).
2. **Bucket removal** — remove all records in a specific bucket
   (e.g. all of `ai/jailbreaking`).
3. **Per-record removal** — remove specific records (e.g. records
   that contain your name, your code, or content you authored).
4. **License change** — if the license of an upstream source has
   changed, request a re-review.
5. **Attribution change** — request a different display name, summary,
   or upstream URL.

---

## How to submit a request

Send an email to the address listed in the repository's GitHub profile
(<https://github.com/Veedubin>). Include:

1. **Your name and relationship to the source** (e.g. "I am the
   copyright holder of X" or "I am legal counsel for Y").
2. **The source, bucket, or record(s) you want removed** — be as
   specific as possible. If you can, reference the `source` field,
   bucket path (e.g. `data/datasets/buckets/sources/metasploit-framework/`),
   or record content.
3. **The reason for the request** (briefly).
4. **Verification of your identity / authority** (this protects against
   bad-faith takedowns of others' work). For copyright holders, a link
   to the upstream repo or a published copyright notice is usually
   enough. For per-record requests, mention something only the original
   author would know about the content.

---

## What happens after you submit

| Step | Time | What happens |
|---|---|---|
| 1. Acknowledgement        | within 48 hours        | The author confirms receipt of your request. |
| 2. Verification           | within 7 days          | The author verifies the request is from a legitimate rights holder. |
| 3. Removal                | within 7 days of step 2 | Records are removed from the public dataset, the manifest is regenerated, and `CHANGELOG.md` is updated with a removal-log entry (no specific requester information is logged). |
| 4. Confirmation           | within 7 days of step 3 | You receive a confirmation that the removal is complete. |
| 5. Git history scrub      | on next release         | The removed records are scrubbed from the git history in the next release. (Records removed earlier may persist in older history until the next release; if you need an immediate git-history scrub, say so in the request.) |
| 6. Notification of downstream | within 7 days of step 3 | If AttackLM has been used in any public HuggingFace dataset push or model release, the author will issue a takedown / correction notice. |

---

## What is NOT in scope

- Requests to remove records that are **publicly available and properly
  licensed upstream**. The author will not remove a source just because
  the requester disagrees with the use; the author will, however,
  listen to concerns about license interpretation and re-review.
- Requests to remove records for **offensive content reasons**. The
  entire purpose of AttackLM is to teach offensive techniques. If you
  authored a technique and it's documented in MITRE ATT&CK or a
  security research paper, it is appropriate for inclusion.

  That said, if a record contains a **specific exploit for an
  unpatched CVE in software you maintain** that you would like
  redacted, that is a legitimate request and will be honored.
- Requests from parties who are **not the rights holder** of the
  upstream source. If you are not the copyright holder or an
  authorized agent, the request will be redirected to the actual
  rights holder.

---

## Special handling for the 3 excluded sources

The following sources were already excluded from the public dataset
in 2026-06-11 due to license risk:

- `endgameinc/RTA` (AGPL-3.0)
- `guardicore/infection_monkey` (GPL-3.0)
- `TheBigPromptLibrary` (mixed/unclear)

If you are a rights holder for one of these and want the **archived
copy** at `archive/restricted-sources/<source>/` removed entirely
(not just from the public dataset), the same process applies. The
author will delete the archive directory and scrub git history.

---

## Removal log (anonymized)

A short log of removals is kept in `CHANGELOG.md` (search for the
`[REMOVAL]` tag). The log records the date, the source, and the
reason; it does **not** record the requester's identity.

---

## Why this is fast and unconditional

The author is doing research to see if a model can be trained in
"the art of hacking" — the kind of knowledge that offense-oriented
security researchers have, the kind that helps defenders think like
attackers. The dataset is a research artifact, not a commercial
product. Disputing a rights-holder request would be a poor use of
time and would also be contrary to the research ethos of "it takes
a hacker to catch a hacker" — the very people whose work the
dataset relies on.

If you have any concerns about the dataset, the author wants to
hear from you.
