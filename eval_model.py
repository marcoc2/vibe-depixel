"""
Model evaluation on the validation split.

Runs a trained SR model on the held-out validation set (same seed=42 split
used during training) and reports per-image and aggregate metrics:
  - psnr:        peak signal-to-noise ratio (dB)
  - ssim:        structural similarity index
  - edge_score:  Sobel-gradient cosine similarity between SR and HR (edge preservation)
  - color_err:   mean absolute error in CIELAB (color fidelity)
  - lpips:       learned perceptual image patch similarity (optional)

Usage:
  python eval_model.py --checkpoint checkpoints/default/sr_model_best.pth
  python eval_model.py --checkpoint checkpoints/experiments/gemini_lr1e-4_bs16_perc_best.pth --preset gemini
  python eval_model.py --checkpoint path/to/model.pth --save-report eval.csv
  python eval_model.py --checkpoint path/to/model.pth --save-grid       # visual sample grid
  python eval_model.py --checkpoint path/to/model.pth --full-val        # all val images, not just 50
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

LR_DIR = "dataset/train/lr"
HR_DIR = "dataset/train/hr"
SCALE = 4
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VAL_SEED = 42


# ── Metrics ───────────────────────────────────────────────────────────────────

def psnr(sr: np.ndarray, hr: np.ndarray) -> float:
    mse = np.mean((sr.astype(np.float32) - hr.astype(np.float32)) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * math.log10(255.0 ** 2 / mse))


def ssim_metric(sr: np.ndarray, hr: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        print("[error] Missing scikit-image: pip install scikit-image")
        sys.exit(1)
    return float(structural_similarity(sr, hr, channel_axis=2, data_range=255))


def edge_score(sr: np.ndarray, hr: np.ndarray) -> float:
    """Cosine similarity between Sobel gradient magnitudes of SR and HR.
    1.0 = identical edges, lower = missing or extra edges."""
    try:
        from skimage.filters import sobel
    except ImportError:
        print("[error] Missing scikit-image: pip install scikit-image")
        sys.exit(1)
    sr_gray = sr.mean(axis=2).astype(np.float32)
    hr_gray = hr.mean(axis=2).astype(np.float32)
    sr_edge = sobel(sr_gray).ravel()
    hr_edge = sobel(hr_gray).ravel()
    denom = np.linalg.norm(sr_edge) * np.linalg.norm(hr_edge)
    if denom < 1e-10:
        return 0.0
    return float(np.dot(sr_edge, hr_edge) / denom)


def color_err(sr: np.ndarray, hr: np.ndarray) -> float:
    """Mean absolute error in CIELAB (perceptually uniform color distance)."""
    try:
        from skimage.color import rgb2lab
    except ImportError:
        print("[error] Missing scikit-image: pip install scikit-image")
        sys.exit(1)
    sr_lab = rgb2lab(sr.astype(np.float32) / 255.0)
    hr_lab = rgb2lab(hr.astype(np.float32) / 255.0)
    return float(np.mean(np.abs(sr_lab - hr_lab)))


# LPIPS: lazy-loaded, optional
_lpips_fn = None

def lpips_metric(sr: np.ndarray, hr: np.ndarray, device: torch.device) -> float | None:
    global _lpips_fn
    try:
        if _lpips_fn is None:
            import lpips
            _lpips_fn = lpips.LPIPS(net="vgg").to(device)
            _lpips_fn.eval()
        def _to_tensor(arr):
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            return t.to(device)
        with torch.no_grad():
            return float(_lpips_fn(_to_tensor(sr), _to_tensor(hr)).item())
    except ImportError:
        return None


# ── Val split (mirrors sr_train.py logic) ─────────────────────────────────────

def _list_images(directory: str) -> list[str]:
    return sorted(
        p.name for p in Path(directory).iterdir()
        if p.suffix.lower() in EXTENSIONS
    )


def get_val_filenames(lr_dir: str, hr_dir: str) -> list[str]:
    lr_names = _list_images(lr_dir)
    hr_names = set(_list_images(hr_dir))
    matched = [n for n in lr_names if n in hr_names]
    if not matched:
        print(f"[error] No matching pairs in {lr_dir} / {hr_dir}")
        sys.exit(1)

    n = len(matched)
    n_val = max(1, n // 10)
    n_train = n - n_val

    # Replicate random_split with seed=42 — uses torch's default sampler
    import torch
    from torch.utils.data import Subset
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(VAL_SEED)).tolist()
    val_indices = indices[n_train:]  # random_split takes first n_train for train
    return [matched[i] for i in sorted(val_indices)]


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model: torch.nn.Module, lr_arr: np.ndarray, device: torch.device) -> np.ndarray:
    """Run model on a full LR image. Returns SR as uint8 HxWxC."""
    t = torch.from_numpy(lr_arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t = t.to(device)
    with torch.no_grad():
        sr = model(t)
    sr = sr.squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return (sr * 255).astype(np.uint8)


# ── Report ────────────────────────────────────────────────────────────────────

def _print_summary(results: list[dict], use_lpips: bool):
    keys = ["psnr", "ssim", "edge_score", "color_err"]
    if use_lpips:
        keys.append("lpips")

    print(f"\n{'=' * 65}")
    print(f"  Validation set: {len(results)} images")
    print(f"{'=' * 65}")
    for key in keys:
        vals = [r[key] for r in results if r.get(key) is not None]
        if not vals:
            continue
        print(
            f"  {key:<14}  mean={np.mean(vals):.4f}  "
            f"std={np.std(vals):.4f}  "
            f"min={np.min(vals):.4f}  "
            f"max={np.max(vals):.4f}"
        )
    print(f"{'=' * 65}")


def _save_csv(results: list[dict], path: str):
    if not results:
        return
    fields = list(results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"Per-image report saved to {path}")


def _save_grid(results: list[dict], lr_dir: str, hr_dir: str,
               model: torch.nn.Module, device: torch.device,
               n: int = 6, out_path: str = "eval_grid.png"):
    """Save a visual grid of n best + n worst samples by PSNR."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[warn] matplotlib not found — skipping grid")
        return

    ranked = sorted(results, key=lambda r: r["psnr"])
    worst = ranked[:n]
    best = ranked[-n:]
    samples = [("worst", worst), ("best", best)]

    n_cols = n
    n_rows = 2 * 3  # 2 groups × 3 rows (LR, SR, HR)

    fig = plt.figure(figsize=(n_cols * 2.5, n_rows * 2.5))
    fig.suptitle("Model Evaluation — Worst vs Best (by PSNR)", fontsize=12, fontweight="bold")
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.05, wspace=0.05)

    for group_idx, (label, group) in enumerate(samples):
        row_base = group_idx * 3
        for col, r in enumerate(group):
            lr_arr = np.array(Image.open(Path(lr_dir) / r["filename"]).convert("RGB"))
            hr_arr = np.array(Image.open(Path(hr_dir) / r["filename"]).convert("RGB"))
            sr_arr = run_inference(model, lr_arr, device)

            # LR — nearest-neighbor upscaled for display
            h, w = hr_arr.shape[:2]
            lr_disp = np.array(Image.fromarray(lr_arr).resize((w, h), Image.NEAREST))

            title = f"PSNR {r['psnr']:.1f}dB"
            for row_offset, (img, row_label) in enumerate([(lr_disp, "LR"), (sr_arr, "SR"), (hr_arr, "HR")]):
                ax = fig.add_subplot(gs[row_base + row_offset, col])
                ax.imshow(img)
                ax.axis("off")
                if col == 0:
                    ax.set_ylabel(f"{label}\n{row_label}", fontsize=7, rotation=0, labelpad=40, va="center")
                if row_offset == 0:
                    ax.set_title(title, fontsize=7)

    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Visual grid saved to {out_path}")


def _save_metric_plots(results: list[dict], use_lpips: bool, out_path: str = "eval_metrics.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not found — skipping plots")
        return

    keys = [("psnr", "PSNR (dB)"), ("ssim", "SSIM"), ("edge_score", "Edge Score"), ("color_err", "Color Error (LAB)")]
    if use_lpips:
        keys.append(("lpips", "LPIPS"))

    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle("Validation Metric Distributions", fontsize=12, fontweight="bold")

    for ax, (key, title) in zip(axes, keys):
        vals = [r[key] for r in results if r.get(key) is not None]
        ax.hist(vals, bins=30, color="#4c8bf5", edgecolor="white", linewidth=0.4)
        ax.axvline(np.mean(vals), color="#e05c5c", linestyle="--",
                   label=f"mean={np.mean(vals):.3f}")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Metric distributions saved to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate SR model on validation set")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--preset", default="default",
                        help="Model preset: default, gemini, esrgan (default: default)")
    parser.add_argument("--pretrained", default=None,
                        help="Pretrained .pth for ESRGAN architecture (required for --preset esrgan)")
    parser.add_argument("--lr-dir", default=LR_DIR)
    parser.add_argument("--hr-dir", default=HR_DIR)
    parser.add_argument("--scale", type=int, default=SCALE)
    parser.add_argument("--full-val", action="store_true",
                        help="Evaluate all val images instead of capping at 50")
    parser.add_argument("--lpips", action="store_true",
                        help="Compute LPIPS (requires: pip install lpips)")
    parser.add_argument("--save-report", default=None,
                        help="Save per-image CSV report")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save metric distribution plots")
    parser.add_argument("--save-grid", action="store_true",
                        help="Save visual grid of best/worst samples")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"[error] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    from core.sr_model import load_model, SPANDREL_PRESETS
    model = load_model(args.preset, device, pretrained_path=args.pretrained, scale=args.scale)
    if args.preset in SPANDREL_PRESETS:
        if Path(args.checkpoint).resolve() != Path(args.pretrained or "").resolve():
            if Path(args.checkpoint).exists():
                model.load_state_dict(
                    torch.load(args.checkpoint, map_location=device, weights_only=True), strict=False
                )
    else:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: preset={args.preset}  params={n_params:,}  checkpoint={args.checkpoint}")

    # Val split
    val_filenames = get_val_filenames(args.lr_dir, args.hr_dir)
    if not args.full_val:
        val_filenames = val_filenames[:50]
    print(f"Val images: {len(val_filenames)}")

    # Evaluate
    results = []
    for i, fname in enumerate(val_filenames, 1):
        if i % 10 == 0 or i == len(val_filenames):
            print(f"  {i}/{len(val_filenames)}", end="\r", flush=True)

        try:
            lr_arr = np.array(Image.open(Path(args.lr_dir) / fname).convert("RGB"))
            hr_arr = np.array(Image.open(Path(args.hr_dir) / fname).convert("RGB"))
            sr_arr = run_inference(model, lr_arr, device)

            # Crop SR/HR to same size if model output is slightly off
            min_h = min(sr_arr.shape[0], hr_arr.shape[0])
            min_w = min(sr_arr.shape[1], hr_arr.shape[1])
            sr_c = sr_arr[:min_h, :min_w]
            hr_c = hr_arr[:min_h, :min_w]

            row = {
                "filename": fname,
                "psnr": round(psnr(sr_c, hr_c), 4),
                "ssim": round(ssim_metric(sr_c, hr_c), 4),
                "edge_score": round(edge_score(sr_c, hr_c), 4),
                "color_err": round(color_err(sr_c, hr_c), 4),
            }
            if args.lpips:
                lp = lpips_metric(sr_c, hr_c, device)
                row["lpips"] = round(lp, 4) if lp is not None else None

            results.append(row)

        except Exception as e:
            print(f"\n  [warn] {fname}: {e}")

    print()  # flush \r

    _print_summary(results, use_lpips=args.lpips)

    if args.save_report:
        _save_csv(results, args.save_report)

    if args.save_plots:
        stem = Path(args.checkpoint).stem
        _save_metric_plots(results, use_lpips=args.lpips,
                           out_path=f"eval_metrics_{stem}.png")

    if args.save_grid:
        stem = Path(args.checkpoint).stem
        _save_grid(results, args.lr_dir, args.hr_dir, model, device,
                   out_path=f"eval_grid_{stem}.png")


if __name__ == "__main__":
    main()
