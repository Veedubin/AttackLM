#!/usr/bin/env python3
"""Migrate AttackLM bucket layout to the v0.2.1 '4 parents' structure.

v0.2.0 and earlier had a mixed layout:
  data/datasets/buckets/
    collection/              <- top-level (10 of these)
    command_and_control/     <- top-level
    ...
    orchestrator/            <- top-level
    ai-models/               <- PARENT (2 sub-buckets)
      jailbreaking/
      prompt-injection/
    tools/                   <- PARENT (3 sub-buckets)
      metasploit/
      infection_monkey/
      rta/

v0.2.1 normalizes to 4 parents:
  data/datasets/buckets/
    base/                    <- NEW parent (10 tactic buckets move here)
      collection/
      command_and_control/
      ...
      privilege_escalation/
    tools/                   <- unchanged
      metasploit/
      infection_monkey/
      rta/
    ai/                      <- RENAMED from ai-models/
      jailbreaking/
      prompt-injection/
    orchestrator/            <- unchanged

Bucket paths in the manifest get updated:
  'collection'                  → 'base/collection'
  'tools/metasploit'            → 'tools/metasploit'   (unchanged)
  'ai-models/prompt-injection'  → 'ai/prompt-injection' (renamed)
  'orchestrator'                → 'orchestrator'        (unchanged)

The bucket spec resolver in bucket_loader.py needs to be updated
to match (base/ prefix, ai/ prefix). This script does the move
AND prints a list of code paths that need updating.

Run:
    python scripts/migrate_buckets_to_v021.py
    python scripts/migrate_buckets_to_v021.py --dry-run   # show what would happen
    python scripts/migrate_buckets_to_v021.py --rollback  # undo a previous migration
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BUCKETS_DIR = BASE_DIR / "data" / "datasets" / "buckets"
BACKUP_ROOT = BASE_DIR / "data" / ".bucket_layout_backup"

# The 10 MITRE tactic bucket names that should move under base/
TACTIC_BUCKETS = [
    "collection",
    "command_and_control",
    "credential_access",
    "defense_evasion",
    "discovery",
    "execution",
    "exfiltration",
    "lateral_movement",
    "persistence",
    "privilege_escalation",
]

# Mapping: old_path → new_path (relative to BUCKETS_DIR)
PATH_MIGRATIONS = {}
for tactic in TACTIC_BUCKETS:
    PATH_MIGRATIONS[tactic] = f"base/{tactic}"
# tools/, orchestrator/ unchanged
# ai-models/ → ai/
PATH_MIGRATIONS["ai-models/jailbreaking"] = "ai/jailbreaking"
PATH_MIGRATIONS["ai-models/prompt-injection"] = "ai/prompt-injection"


def parse_manifest():
    """Read the current manifest. Returns (manifest_dict, source_path)."""
    manifest_path = BUCKETS_DIR / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}")
        sys.exit(1)
    with manifest_path.open() as f:
        return json.load(f), manifest_path


def write_manifest(manifest, manifest_path):
    """Write the manifest atomically."""
    tmp = manifest_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    tmp.replace(manifest_path)


def backup_layout():
    """Snapshot the current bucket layout to data/.bucket_layout_backup/."""
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot = BACKUP_ROOT / f"buckets_{ts}"
    shutil.copytree(BUCKETS_DIR, snapshot)
    print(f"  Backup of BUCKETS_DIR created at: {snapshot}")
    return snapshot


def restore_from_backup(snapshot_path: Path):
    """Restore BUCKETS_DIR from a previous snapshot."""
    if not snapshot_path.exists():
        print(f"ERROR: snapshot not found at {snapshot_path}")
        sys.exit(1)
    # Wipe the current BUCKETS_DIR and replace
    for child in BUCKETS_DIR.iterdir():
        if child.name == ".bucket_layout_backup":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    # Copy from snapshot
    for child in snapshot_path.iterdir():
        dest = BUCKETS_DIR / child.name
        if child.is_dir():
            shutil.copytree(child, dest)
        else:
            shutil.copy2(child, dest)
    print(f"  Restored BUCKETS_DIR from: {snapshot_path}")


def list_backups() -> list[Path]:
    """List available backup snapshots, newest first."""
    if not BACKUP_ROOT.exists():
        return []
    return sorted(
        BACKUP_ROOT.glob("buckets_*"),
        key=lambda p: p.name,
        reverse=True,
    )


def needs_migration(manifest: dict) -> bool:
    """True if any bucket has a path that needs updating."""
    for b in manifest.get("buckets", []):
        if b["path"] in PATH_MIGRATIONS:
            return True
    return False


def is_already_migrated(manifest: dict) -> bool:
    """True if all bucket paths are already in v0.2.1 form."""
    for b in manifest.get("buckets", []):
        if b["path"] in PATH_MIGRATIONS:
            return False
    return True


def migrate_directory_layout(dry_run: bool = False) -> dict:
    """Move directory contents per PATH_MIGRATIONS. Returns a stats dict."""
    stats = {"moved": [], "skipped": [], "errors": [], "emptied_parents": []}

    for old_rel, new_rel in PATH_MIGRATIONS.items():
        old_path = BUCKETS_DIR / old_rel
        new_path = BUCKETS_DIR / new_rel

        if not old_path.exists():
            stats["skipped"].append(f"{old_rel} (not found)")
            continue
        if new_path.exists():
            stats["skipped"].append(f"{old_rel} → {new_rel} (destination exists)")
            continue

        if dry_run:
            stats["moved"].append(f"{old_rel} → {new_rel} (dry-run)")
        else:
            try:
                new_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old_path), str(new_path))
                stats["moved"].append(f"{old_rel} → {new_rel}")
            except Exception as e:
                stats["errors"].append(f"{old_rel} → {new_rel}: {e}")

    # Cleanup pass: rmdir any parent dir that was the *old* container
    # for one of the moves and is now empty. Example: ai-models/ is
    # empty after its 2 sub-buckets moved into ai/. Without this,
    # an empty ai-models/ dir lingers on disk and confuses users.
    if not dry_run:
        for old_rel in PATH_MIGRATIONS:
            old_path = BUCKETS_DIR / old_rel
            # The parent of a nested path (e.g. ai-models/) is its top-level
            parent = old_path.parent
            if parent.exists() and parent.is_dir() and parent != BUCKETS_DIR:
                # Check if parent is empty (ignoring hidden files)
                contents = [c for c in parent.iterdir() if not c.name.startswith(".")]
                if not contents:
                    try:
                        parent.rmdir()
                        stats["emptied_parents"].append(
                            str(parent.relative_to(BUCKETS_DIR))
                        )
                    except OSError:
                        pass

    return stats


def update_manifest(manifest: dict, dry_run: bool = False) -> int:
    """Update manifest bucket paths. Returns number of buckets updated."""
    n_updated = 0
    for b in manifest.get("buckets", []):
        old_path = b["path"]
        if old_path in PATH_MIGRATIONS:
            new_path = PATH_MIGRATIONS[old_path]
            b["path"] = new_path
            n_updated += 1
            if dry_run:
                print(f"  manifest: {old_path} → {new_path}")
    return n_updated


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate bucket layout to v0.2.1 (4 parents: base/, tools/, ai/, orchestrator/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making any changes",
    )
    parser.add_argument(
        "--rollback",
        type=str,
        default=None,
        metavar="SNAPSHOT_NAME",
        help="Roll back a previous migration. Use the snapshot name from "
        "`--list-backups` (e.g. 'buckets_20260610_080000').",
    )
    parser.add_argument(
        "--list-backups",
        action="store_true",
        help="List available rollback snapshots and exit",
    )
    args = parser.parse_args()

    if args.list_backups:
        print("Available backup snapshots (newest first):")
        for snap in list_backups():
            print(f"  {snap.name}")
        return 0

    if args.rollback:
        # Find the snapshot
        candidates = [s for s in list_backups() if s.name == args.rollback]
        if not candidates:
            print(f"ERROR: snapshot '{args.rollback}' not found.")
            print("Run with --list-backups to see available snapshots.")
            return 1
        restore_from_backup(candidates[0])
        # Also restore the manifest
        manifest_backup = candidates[0] / "manifest.json"
        if manifest_backup.exists():
            shutil.copy2(manifest_backup, BUCKETS_DIR / "manifest.json")
            print(f"  Restored manifest.json from snapshot")
        print()
        print("Done. Verify with: attacklm-buckets --list")
        return 0

    print(f"Buckets dir: {BUCKETS_DIR}")
    print(f"Dry-run:     {args.dry_run}")
    print()

    # Read manifest
    manifest, manifest_path = parse_manifest()

    # Check if migration is needed
    if is_already_migrated(manifest):
        print("No migration needed. All bucket paths are already in v0.2.1 form.")
        return 0
    if not needs_migration(manifest):
        print("Unexpected: manifest is not fully v0.2.1 but no known paths to migrate.")
        return 1

    # Backup
    if not args.dry_run:
        print("Step 1/3: Backing up current layout...")
        snapshot = backup_layout()
        print(f"  (rollback with: --rollback {snapshot.name})")
        print()

    # Move directories
    print("Step 2/3: Moving directories...")
    stats = migrate_directory_layout(dry_run=args.dry_run)
    for s in stats["moved"]:
        print(f"  ✓ {s}")
    for s in stats["skipped"]:
        print(f"  ⏭  {s}")
    for s in stats["errors"]:
        print(f"  ✗ {s}")
    for s in stats.get("emptied_parents", []):
        print(f"  🧹 Removed empty parent dir: {s}/")
    if stats["errors"]:
        print()
        print("ERRORS during move. Aborting before manifest update.")
        return 1
    print()

    # Update manifest
    print("Step 3/3: Updating manifest.json...")
    n = update_manifest(manifest, dry_run=args.dry_run)
    if not args.dry_run:
        write_manifest(manifest, manifest_path)
    print(f"  Updated {n} bucket paths in manifest.json")
    print()

    # Print a summary
    print("=" * 60)
    print(" Migration summary")
    print("=" * 60)
    print(f"  Buckets moved:    {len(stats['moved'])}")
    print(f"  Buckets skipped:  {len(stats['skipped'])}")
    print(f"  Errors:           {len(stats['errors'])}")
    print()
    if not args.dry_run:
        print("Done. Next steps:")
        print(
            "  1. Verify: uv run python -c \"from scripts.bucket_loader import list_buckets; [print(b['path'], b['count']) for b in list_buckets()]\""
        )
        print(
            "  2. Re-run training: attacklm-train-all --single-model --dataset all --epochs 5"
        )
        print(
            "  3. The bucket_loader.py needs updating to recognize the new 'base/' and 'ai/' prefixes."
        )
        print("     (See commit ba64186 notes for the resolver changes.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
