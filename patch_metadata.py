#!/usr/bin/env python3
"""
Patch ComfyUI workflow metadata in dataset PNGs.

Changes applied to every PNG with 'prompt' metadata:
  - Node 223 (ACN): end_percent 0.867 -> 1.0
  - Node 224 (VAEEncode): removed
  - Node 264 (VAEEncode): pixels connection ['201',0] -> ['302',0]
  - Node 282 (ACN): strength 0.77->0.85, end_percent 0.686->1.0
  - Node 302 (imageColorMatch): method 'mvgd' -> 'reinhard'
  - Node 346/347 (SaveImage): filename_prefix 'dataset/upscale/' -> 'dataset/upscale_fix/'
"""
import json
import struct
import zlib
from pathlib import Path
import argparse


def patch_prompt(prompt: dict) -> dict:
    import copy
    p = copy.deepcopy(prompt)

    # Node 223: ACN — only end_percent (strength is character-specific, don't touch)
    if "223" in p:
        p["223"]["inputs"]["end_percent"] = 1.0

    # Node 224: VAEEncode — remove entirely
    p.pop("224", None)

    # Node 264: fix pixels connection (was going through 224, now direct to 302)
    if "264" in p:
        if p["264"]["inputs"].get("pixels") == ["201", 0]:
            p["264"]["inputs"]["pixels"] = ["302", 0]

    # Node 280: UpscaleModelLoader — use custom trained model
    if "280" in p:
        p["280"]["inputs"]["model_name"] = "my_depixel.pth"

    # Node 282: ACN strength + end_percent
    if "282" in p:
        p["282"]["inputs"]["strength"] = 0.85
        p["282"]["inputs"]["end_percent"] = 1.0

    # Node 302: imageColorMatch method
    if "302" in p:
        p["302"]["inputs"]["method"] = "mvgd"

    # Node 357: StringConcatenate — output path prefix (most images use this)
    if "357" in p:
        val = p["357"]["inputs"].get("string_a", "")
        if isinstance(val, str):
            p["357"]["inputs"]["string_a"] = val.replace(
                "dataset/upscale/", "dataset/upscale_fix/"
            )

    # Nodes 346/347: SaveImage — fallback for images with hardcoded filename_prefix
    for nid in ("346", "347"):
        if nid in p:
            prefix = p[nid]["inputs"].get("filename_prefix", "")
            if isinstance(prefix, str):
                p[nid]["inputs"]["filename_prefix"] = prefix.replace(
                    "dataset/upscale/", "dataset/upscale_fix/"
                )

    return p


def read_png_chunks(data: bytes):
    """Yield (chunk_type, chunk_data) for each PNG chunk."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "Not a PNG file"
    pos = 8
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos+4])[0]
        chunk_type = data[pos+4:pos+8]
        chunk_data = data[pos+8:pos+8+length]
        pos += 12 + length
        yield chunk_type, chunk_data


def build_text_chunk(keyword: str, text: str) -> bytes:
    """Build a PNG tEXt chunk."""
    payload = keyword.encode("latin-1") + b"\x00" + text.encode("latin-1")
    length = struct.pack(">I", len(payload))
    chunk_type = b"tEXt"
    crc = struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    return length + chunk_type + payload + crc


def patch_png(path: Path, dry_run: bool = False) -> bool:
    raw = path.read_bytes()

    # Parse existing chunks
    chunks = list(read_png_chunks(raw))

    # Find and decode text chunks
    text_chunks = {}
    for chunk_type, chunk_data in chunks:
        if chunk_type == b"tEXt":
            sep = chunk_data.index(b"\x00")
            key = chunk_data[:sep].decode("latin-1")
            val = chunk_data[sep+1:].decode("latin-1")
            text_chunks[key] = val

    if "prompt" not in text_chunks:
        return False  # no ComfyUI metadata

    prompt = json.loads(text_chunks["prompt"])
    new_prompt = patch_prompt(prompt)

    if new_prompt == prompt:
        return False  # nothing changed

    if dry_run:
        return True

    # Rebuild PNG: keep all chunks except old tEXt keys we'll replace
    keys_to_replace = {"prompt"}  # only patch prompt, leave workflow as-is
    sig = b"\x89PNG\r\n\x1a\n"
    out = bytearray(sig)

    text_written = set()
    for chunk_type, chunk_data in chunks:
        if chunk_type == b"tEXt":
            sep = chunk_data.index(b"\x00")
            key = chunk_data[:sep].decode("latin-1")
            if key in keys_to_replace:
                if key not in text_written:
                    new_val = json.dumps(new_prompt, ensure_ascii=True, separators=(",", ":"))
                    out += build_text_chunk(key, new_val)
                    text_written.add(key)
                continue  # skip old chunk
        # Write chunk as-is
        length = len(chunk_data)
        out += struct.pack(">I", length)
        out += chunk_type
        out += chunk_data
        crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        out += struct.pack(">I", crc)

    path.write_bytes(bytes(out))
    return True


def main():
    parser = argparse.ArgumentParser(description="Patch ComfyUI metadata in dataset PNGs")
    parser.add_argument(
        "folder",
        nargs="?",
        default="F:/AppsCrucial/ComfyUI_phoenix3/ComfyUI/output/dataset/upscale_backup",
        help="Root folder to scan recursively",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    args = parser.parse_args()

    folder = Path(args.folder)
    pngs = sorted(folder.rglob("*.png"))
    print(f"Scanning {len(pngs)} PNGs in {folder}")

    patched = 0
    skipped = 0
    errors = 0

    for p in pngs:
        try:
            changed = patch_png(p, dry_run=args.dry_run)
            if changed:
                patched += 1
                if args.dry_run:
                    print(f"  [would patch] {p.relative_to(folder)}")
            else:
                skipped += 1
        except Exception as e:
            errors += 1
            print(f"  [error] {p.name}: {e}")

    action = "Would patch" if args.dry_run else "Patched"
    print(f"\n{action}: {patched}  |  skipped (no metadata / no change): {skipped}  |  errors: {errors}")


if __name__ == "__main__":
    main()
