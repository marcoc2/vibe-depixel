"""Re-export swinir1x checkpoints to ComfyUI-compatible format.

Loads old checkpoint (with conv_before_upsample contamination),
rebuilds the model correctly, and saves a clean state_dict.

Usage:
    python reexport_swinir1x.py checkpoints/experiments/swinir1x_lr1e-4_bs2_perc_lum_dists_best.pth
    python reexport_swinir1x.py checkpoints/experiments/swinir1x_*.pth  # glob via shell
"""

import sys
import torch
from pathlib import Path
from core.sr_model import load_swinir1x

SWINIR_PRETRAINED = "4x-PBRify_UpscalerSIR-M_V2.pth"


def reexport(ckpt_path: str):
    p = Path(ckpt_path)
    device = torch.device("cpu")

    raw = torch.load(p, map_location=device, weights_only=True)
    old_state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw

    # Build clean 1x model from pretrained
    model = load_swinir1x(SWINIR_PRETRAINED, device)

    # Load trained weights — strict=False to ignore shape mismatches on conv_last
    missing, unexpected = model.load_state_dict(old_state, strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys (ignored): {len(unexpected)}")

    out_path = p.with_stem(p.stem + "_fixed")
    torch.save({"state_dict": model.state_dict(), "scale": 1, "preset": "swinir1x"}, out_path)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        print("Usage: python reexport_swinir1x.py <checkpoint.pth> [...]")
        sys.exit(1)
    for p in paths:
        print(f"Re-exporting {p}...")
        reexport(p)
    print("Done.")
