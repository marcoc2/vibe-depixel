#!/usr/bin/env python3
"""
Queue patched prompts to ComfyUI API.

Strategy: within each subfolder, group images by resolution.
Same resolution = same source GIF → one image per group is enough.
Images sorted oldest-first so we process in original order.
"""
import json
import uuid
import time
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from collections import defaultdict

from PIL import Image

COMFY_URL = "http://127.0.0.1:8188"


def get_queue_size(base_url: str) -> int:
    try:
        with urllib.request.urlopen(f"{base_url}/queue") as r:
            data = json.loads(r.read())
            return len(data.get("queue_running", [])) + len(data.get("queue_pending", []))
    except Exception:
        return -1


def queue_prompt(prompt: dict, client_id: str, base_url: str) -> str | None:
    payload = json.dumps({"prompt": prompt, "client_id": client_id}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            resp = json.loads(r.read())
            return resp.get("prompt_id")
    except urllib.error.HTTPError as e:
        print(f"    [HTTP {e.code}] {e.read().decode()[:200]}")
        return None


def collect_jobs(backup_dir: Path) -> list[Path]:
    """
    For each subfolder, group PNGs by (width, height).
    Pick the oldest file from each group.
    Returns list of representative PNGs sorted oldest-first overall.
    """
    jobs = []

    for folder in sorted(backup_dir.rglob("*")):
        if not folder.is_dir():
            continue

        # Skip lr folders — the workflow saves both lr and hr in one run
        if folder.name == "lr":
            continue

        pngs = sorted(
            [p for p in folder.glob("*.png") if not p.name.startswith(".")],
            key=lambda p: p.stat().st_mtime,
        )
        if not pngs:
            continue

        by_size = defaultdict(list)
        for png in pngs:
            try:
                img = Image.open(png)
                if "prompt" not in img.info:
                    continue
                by_size[img.size].append(png)
            except Exception:
                continue

        for size, files in by_size.items():
            # oldest file in this size group
            jobs.append(min(files, key=lambda p: p.stat().st_mtime))

    # Sort all jobs oldest-first
    jobs.sort(key=lambda p: p.stat().st_mtime)
    return jobs


def main():
    parser = argparse.ArgumentParser(description="Queue ComfyUI jobs from patched PNGs")
    parser.add_argument(
        "folder",
        nargs="?",
        default="F:/AppsCrucial/ComfyUI_phoenix3/ComfyUI/output/dataset/upscale_backup",
    )
    parser.add_argument("--url", default=COMFY_URL, help="ComfyUI base URL")
    parser.add_argument("--dry-run", action="store_true", help="List jobs without queuing")
    parser.add_argument(
        "--max-queue",
        type=int,
        default=4,
        help="Wait when queue reaches this size (default: 4)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds between requests (default: 0.5)",
    )
    args = parser.parse_args()

    backup_dir = Path(args.folder)
    jobs = collect_jobs(backup_dir)

    print(f"Found {len(jobs)} unique jobs (one per resolution per folder)")
    if args.dry_run:
        for j in jobs:
            img = Image.open(j)
            print(f"  {img.size}  {j.relative_to(backup_dir)}")
        return

    client_id = str(uuid.uuid4())
    queued = 0
    errors = 0

    for i, png in enumerate(jobs, 1):
        img = Image.open(png)
        prompt = json.loads(img.info["prompt"])
        rel = png.relative_to(backup_dir)

        # Throttle: wait if queue is too full
        while True:
            qs = get_queue_size(args.url)
            if qs == -1:
                print("  [error] cannot reach ComfyUI — is it running?")
                return
            if qs < args.max_queue:
                break
            print(f"  [wait] queue={qs}, sleeping 5s...", end="\r")
            time.sleep(5)

        prompt_id = queue_prompt(prompt, client_id, args.url)
        if prompt_id:
            queued += 1
            print(f"  [{i}/{len(jobs)}] queued {rel}  id={prompt_id[:8]}")
        else:
            errors += 1
            print(f"  [{i}/{len(jobs)}] FAILED {rel}")

        time.sleep(args.delay)

    print(f"\nDone. Queued: {queued}  Errors: {errors}")


if __name__ == "__main__":
    main()
