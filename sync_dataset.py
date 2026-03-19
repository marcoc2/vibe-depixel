"""
Sync dataset from ComfyUI output to local dataset/train/.
Copies only new matched LR/HR pairs, skips existing files.

Usage:
    python sync_dataset.py
    python sync_dataset.py --dry-run    # preview without copying
"""

import argparse
import shutil
from pathlib import Path

SRC = Path(r"F:\AppsCrucial\ComfyUI_phoenix3\ComfyUI\output\dataset\upscale")
DST_LR = Path("dataset/train/lr")
DST_HR = Path("dataset/train/hr")
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def sync(dry_run: bool = False):
    DST_LR.mkdir(parents=True, exist_ok=True)
    DST_HR.mkdir(parents=True, exist_ok=True)

    existing = set(p.name for p in DST_LR.iterdir())
    copied = 0
    skipped_no_hr = 0

    for batch_dir in sorted(SRC.iterdir()):
        if not batch_dir.is_dir():
            continue
        lr_dir = batch_dir / "lr"
        hr_dir = batch_dir / "hr"
        if not lr_dir.exists() or not hr_dir.exists():
            continue

        hr_names = set(f.name for f in hr_dir.iterdir() if f.suffix.lower() in EXTENSIONS)

        for lr_file in sorted(lr_dir.iterdir()):
            if lr_file.suffix.lower() not in EXTENSIONS:
                continue
            if lr_file.name in existing:
                continue
            if lr_file.name not in hr_names:
                skipped_no_hr += 1
                continue

            if dry_run:
                print(f"  [new] {batch_dir.name}/{lr_file.name}")
            else:
                shutil.copy2(lr_file, DST_LR / lr_file.name)
                shutil.copy2(hr_dir / lr_file.name, DST_HR / lr_file.name)
            existing.add(lr_file.name)
            copied += 1

    total = len(list(DST_LR.iterdir())) if not dry_run else len(existing)
    print(f"\n{'[DRY RUN] ' if dry_run else ''}+{copied} new pairs  |  {skipped_no_hr} skipped (no HR)  |  {total} total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync ComfyUI dataset to local train/")
    parser.add_argument("--dry-run", action="store_true", help="Preview without copying")
    sync(parser.parse_args().dry_run)
