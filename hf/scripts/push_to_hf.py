#!/usr/bin/env python3
"""Push AttackLM dataset to HuggingFace Hub.

Builds a HuggingFace DatasetDict from the pre-built train/test JSONL files
and pushes it to the HuggingFace Hub. Validates format before upload.

Prerequisites:
    pip install datasets huggingface_hub

Usage:
    # Set HF token via environment variable
    export HF_TOKEN="hf_..."

    # Push to public repo (default)
    python hf/scripts/push_to_hf.py

    # Push to private repo
    python hf/scripts/push_to_hf.py --private

    # Specify token via flag
    python hf/scripts/push_to_hf.py --token hf_...

    # Custom repo name
    python hf/scripts/push_to_hf.py --repo-id myorg/my-attacklm-dataset

    # Custom data directory
    python hf/scripts/push_to_hf.py --data-dir hf/data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {"messages", "mitre_ids", "source", "license", "bucket", "category"}
REQUIRED_MESSAGE_ROLES = {"system", "user", "assistant"}


def validate_record(record: dict, line_num: int, path: str) -> list[str]:
    """Validate a single dataset record. Returns list of error strings."""
    errors: list[str] = []

    # Check required top-level fields
    missing = REQUIRED_FIELDS - set(record.keys())
    if missing:
        errors.append(f"Line {line_num} in {path}: missing fields: {missing}")

    # Check messages structure
    messages = record.get("messages", [])
    if not isinstance(messages, list):
        errors.append(f"Line {line_num} in {path}: 'messages' must be a list")
        return errors

    if len(messages) < 2:
        errors.append(
            f"Line {line_num} in {path}: 'messages' must have at least 2 entries"
        )

    present_roles = {m.get("role") for m in messages if isinstance(m, dict)}
    missing_roles = REQUIRED_MESSAGE_ROLES - present_roles
    if missing_roles:
        errors.append(
            f"Line {line_num} in {path}: 'messages' missing roles: {missing_roles}"
        )

    # Check mitre_ids is a list
    mitre_ids = record.get("mitre_ids")
    if mitre_ids is not None and not isinstance(mitre_ids, list):
        errors.append(
            f"Line {line_num} in {path}: 'mitre_ids' must be a list, got {type(mitre_ids).__name__}"
        )

    return errors


def validate_jsonl(path: Path, max_records: int = 0) -> tuple[int, list[str]]:
    """Validate a JSONL file. Returns (record_count, error_list).

    If max_records > 0, only validate the first max_records records.
    """
    errors: list[str] = []
    count = 0

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num} in {path.name}: invalid JSON: {e}")
                continue

            if max_records > 0 and count >= max_records:
                break

            record_errors = validate_record(record, line_num, path.name)
            errors.extend(record_errors)
            count += 1

    return count, errors


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------


def load_jsonl_as_dicts(path: Path) -> list[dict]:
    """Load a JSONL file as a list of dicts."""
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_dataset_dict(data_dir: Path) -> "DatasetDict":
    """Build a HuggingFace DatasetDict from train/test JSONL files.

    Requires the `datasets` library.
    """
    try:
        from datasets import Dataset, DatasetDict
    except ImportError:
        print(
            "ERROR: The `datasets` library is required. Install with:", file=sys.stderr
        )
        print("  pip install datasets", file=sys.stderr)
        sys.exit(1)

    train_path = data_dir / "attacklm-train.jsonl"
    test_path = data_dir / "attacklm-test.jsonl"

    if not train_path.exists():
        print(f"ERROR: Train file not found: {train_path}", file=sys.stderr)
        print(
            "Run build_hf_dataset.py first to create the data files.", file=sys.stderr
        )
        sys.exit(1)

    if not test_path.exists():
        print(f"ERROR: Test file not found: {test_path}", file=sys.stderr)
        print(
            "Run build_hf_dataset.py first to create the data files.", file=sys.stderr
        )
        sys.exit(1)

    print(f"Loading train data from {train_path}...")
    train_records = load_jsonl_as_dicts(train_path)
    print(f"  {len(train_records)} records")

    print(f"Loading test data from {test_path}...")
    test_records = load_jsonl_as_dicts(test_path)
    print(f"  {len(test_records)} records")

    # Convert to HuggingFace Datasets
    # We need to flatten the messages list into separate columns for HF compatibility
    train_dataset = Dataset.from_list(train_records)
    test_dataset = Dataset.from_list(test_records)

    dataset_dict = DatasetDict(
        {
            "train": train_dataset,
            "test": test_dataset,
        }
    )

    return dataset_dict


# ---------------------------------------------------------------------------
# Push to Hub
# ---------------------------------------------------------------------------


def push_to_hub(
    dataset_dict: "DatasetDict",
    repo_id: str,
    token: str | None,
    private: bool = False,
    commit_message: str = "Upload AttackLM dataset",
) -> None:
    """Push the dataset to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(
            "ERROR: The `huggingface_hub` library is required. Install with:",
            file=sys.stderr,
        )
        print("  pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    print(f"\nPushing dataset to: {repo_id}")
    print(f"  Private: {private}")
    print(f"  Train: {len(dataset_dict['train'])} examples")
    print(f"  Test: {len(dataset_dict['test'])} examples")

    # Push the dataset
    dataset_dict.push_to_hub(
        repo_id=repo_id,
        private=private,
        token=token,
        commit_message=commit_message,
    )

    print(
        f"\n✓ Dataset pushed successfully to https://huggingface.co/datasets/{repo_id}"
    )

    # Also upload the README.md as the dataset card
    readme_path = Path(__file__).resolve().parent.parent / "README.md"
    if readme_path.exists():
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload dataset card (README.md)",
        )
        print(f"✓ Dataset card uploaded from {readme_path}")

    # Upload dataset_infos.json
    infos_path = Path(__file__).resolve().parent.parent / "dataset_infos.json"
    if infos_path.exists():
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(infos_path),
            path_in_repo="dataset_infos.json",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload dataset_infos.json",
        )
        print(f"✓ dataset_infos.json uploaded from {infos_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push AttackLM dataset to HuggingFace Hub."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="neuralgentics/attacklm",
        help="HuggingFace dataset repository ID (default: neuralgentics/attacklm)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("hf/data"),
        help="Directory containing attacklm-train.jsonl and attacklm-test.jsonl (default: hf/data)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset repository private",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate data files without pushing to Hub",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation before pushing (not recommended)",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload AttackLM v0.3.0 dataset",
        help="Commit message for the push (default: Upload AttackLM v0.3.0 dataset)",
    )
    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = (project_root / args.data_dir).resolve()

    # Get token from env or flag
    import os

    token = (
        args.token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )

    if not token and not args.validate_only:
        print(
            "ERROR: HuggingFace token required. Set HF_TOKEN env var or use --token.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate data files
    train_path = data_dir / "attacklm-train.jsonl"
    test_path = data_dir / "attacklm-test.jsonl"

    if not args.skip_validation:
        print("Validating train data...")
        train_count, train_errors = validate_jsonl(train_path)
        if train_errors:
            print(
                f"\n❌ Found {len(train_errors)} validation errors in train data:",
                file=sys.stderr,
            )
            for err in train_errors[:20]:  # Show first 20 errors
                print(f"  {err}", file=sys.stderr)
            if len(train_errors) > 20:
                print(f"  ... and {len(train_errors) - 20} more", file=sys.stderr)
            sys.exit(1)
        print(f"  ✓ {train_count} train records validated")

        print("Validating test data...")
        test_count, test_errors = validate_jsonl(test_path)
        if test_errors:
            print(
                f"\n❌ Found {len(test_errors)} validation errors in test data:",
                file=sys.stderr,
            )
            for err in test_errors[:20]:
                print(f"  {err}", file=sys.stderr)
            if len(test_errors) > 20:
                print(f"  ... and {len(test_errors) - 20} more", file=sys.stderr)
            sys.exit(1)
        print(f"  ✓ {test_count} test records validated")
    else:
        print("⚠ Validation skipped (--skip-validation)")

    if args.validate_only:
        print("\n✓ Validation complete. Use without --validate-only to push to Hub.")
        return

    # Build and push dataset
    dataset_dict = build_dataset_dict(data_dir)

    push_to_hub(
        dataset_dict=dataset_dict,
        repo_id=args.repo_id,
        token=token,
        private=args.private,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
