# Decontamination (--decontam-against)

> **Methodology credit**: This document implements techniques from
> "MAI-Thinking-1: Building a Hill-Climbing Machine" by The Microsoft AI Team
> (June 2026, 109 pages). The specific methodology we adopted is described in
> [section 2.3.1 and 2.4.3 of that paper](paper-citation). We thank the Microsoft AI Team
> for sharing their development methodology in detail.
> 
> **Section reference**: MAI-Thinking-1 §2.3.1 — Public Evaluation Decontamination; §2.4.3 — Deduplication
> **What we took**: The 20-gram fuzzy deduplication methodology at 80% similarity threshold, with optional quarantine output for matched records.
> **What we adapted**: Microsoft's pipeline removes matches from training data; ours defaults to a non-destructive quarantine (opt-in `--quarantine-output` writes matched records to a separate file for review).
> **What we did NOT take**: Microsoft's data-licensing restrictions and `huggingface.co`-domain exclusion (their data is public-domain web + licensed; ours is per-source security corpora with explicit license headers).
>
> *If the paper later gets a public URL, replace `(paper-citation)` in this block
> with the real URL. The section number + title is the canonical link for now.*

## Overview

Decontamination is the process of ensuring that a model's training set does not contain examples from its evaluation sets. In the context of security LLMs, "contamination" often occurs when public benchmarks (e.g., Atomic Red Team tests or MITRE Caldera evaluations) are accidentally ingested into the training corpus via web scrapes or open-source repositories.

If a model is trained on its evaluation data, its performance metrics become artificially inflated—a phenomenon known as "data leakage." This masks the model's actual generalization ability and can lead to a false sense of security regarding the model's capability to handle unseen, real-world adversarial scenarios.

## The 20-gram fuzzy approach

To detect leakage, we employ a fuzzy matching strategy based on character-level n-grams, as described in MAI-Thinking-1 §2.3.1. Unlike exact string matching, which is easily defeated by minor formatting changes or whitespace variations, the n-gram approach identifies semantic and structural overlap.

### Technical implementation
1. **N-gram Extraction**: Both training and evaluation records are decomposed into sets of overlapping character-level sequences of length $N=20$.
2. **MinHash LSH**: To avoid the $O(N^2)$ cost of comparing every training record to every eval record, we use MinHash Locality Sensitive Hashing (LSH). This maps high-similarity records into the same "buckets" with high probability.
3. **Jaccard Similarity**: For all candidate pairs identified by LSH, we compute the exact Jaccard similarity:
   $$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
4. **Threshold**: A similarity threshold of $0.80$ is applied. Records with $\ge 80\%$ overlap in 20-grams are flagged as contaminated.

Microsoft found that 20-grams at 0.80 threshold provide an optimal balance between precision and recall, effectively capturing "near-verbatim" leaks while ignoring common boilerplate or generic security terminology.

## CLI

The decontamination tool is available via `scripts/decontam.py`.

### Argument Specification

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--training-data-root` | `PATH` | `data/datasets/buckets/sources` | Root directory containing per-source training data. |
| `--eval-set-dir` | `PATH` | **Required** | Path to a directory of `.jsonl` eval sets or a single `.jsonl` file. |
| `--threshold` | `FLOAT` | `0.80` | MinHash similarity threshold (Jaccard). |
| `--ngram-size` | `INT` | `20` | Size of character-level n-grams. |
| `--permutations` | `INT` | `128` | Number of MinHash permutations (precision vs speed). |
| `--source-filter` | `STR+` | `None` | Space-separated list of sources to audit (defaults to all non-restricted). |
| `--quarantine-output`| `PATH` | `None` | Path to write matched training records to a separate JSONL file. |
| `--report-output` | `PATH` | `stdout` | Path to write the final contamination JSON report. |
| `--dry-run` | `FLAG` | `False` | Print report stats only; do not write quarantine files. |
| `--num-proc` | `INT` | `4` | Number of parallel processes for MinHash index construction. |

### Example Usage
```bash
python scripts/decontam.py \
    --eval-set-dir data/eval_sets/atomic-red-team-tests.jsonl \
    --quarantine-output data/quarantine_v0.6.0.jsonl \
    --report-output reports/decontam_report.json
```

## Output schema

The tool produces a JSON report detailing contamination per source.

### Report Format
```json
{
  "total_overlaps": 42,
  "sources": {
    "metasploit-framework": {
      "n_training_records": 12400,
      "n_eval_overlaps": 15,
      "n_unique_eval_records_overlapping": 12,
      "examples": [
        {
          "training_id": "metasploit-framework::402",
          "training_source": "metasploit-framework",
          "eval_id": "eval_12",
          "eval_source": "atomic-red-team",
          "similarity": 0.842105
        }
      ]
    }
  }
}
```

## Decision tree

**Which mode should I use?**

1. **I just want to know if there is a problem**:
   - Use `--dry-run`. This generates the report in stdout without touching the disk.
2. **I want to analyze the leaks before deciding what to delete**:
   - Use `--quarantine-output <path>`. This moves potential leaks into a "holding area" for manual review without modifying the original training sets.
3. **I am seeing too many false positives (common security terminology flagged)**:
   - Increase `--threshold` to `0.90` or `0.95`.
4. **I am missing obvious leaks (slight paraphrasing)**:
   - Decrease `--threshold` to `0.70` or decrease `--ngram-size` to `15`.
5. **I have a new public benchmark I need to protect against**:
   - Add the benchmark as a `.jsonl` file in `data/eval_sets/` and run the tool.

## Known limitations

- **False Positive Rate**: At a $0.80$ threshold, there is a $5-10\%$ false-positive rate on technical text due to repeated boilerplate (e.g., common API headers or standardized exploit templates).
- **Language Agnostic**: The tool is character-level and does not understand the semantic meaning of the text.
- **Paraphrasing**: It does not detect high-level semantic paraphrasing where the structure and vocabulary change significantly but the meaning remains the same.

## Adding new eval sets

To add a new evaluation benchmark for decontamination:

1. Create a `.jsonl` file in `data/eval_sets/`.
2. Ensure each line is a JSON object: `{"id": "unique-id", "text": "the content", "source": "benchmark-name"}`.
3. **Mandatory**: Include a license header at the top of the file (prefixed with `#`) specifying the rights to use the data for auditing.
4. Run `scripts/decontam.py --eval-set-dir data/eval_sets/`.

## References

- **MAI-Thinking-1 §2.3.1** (Public Evaluation Decontamination) and **§2.4.3** (Deduplication) by The Microsoft AI Team, June 2026.
- **datasketch**: Python library for MinHash LSH (https://datasketch.readthedocs.io/).
