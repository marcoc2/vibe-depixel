"""
Dataset quality analysis for SR training pairs.

Computes per-pair metrics to detect bad ComfyUI batches:
  - ssim:            structural similarity (NN-upscaled LR vs HR) — low = bad pair
  - color_drift:     mean absolute error in LAB color space — high = color shift
  - sharpness_ratio: Laplacian variance of HR / LR upscaled — <1 means HR is blurrier
  - wrong_scale:     flagged if HR is not ~4x the LR resolution

Usage:
  python analyze_dataset.py
  python analyze_dataset.py --lr-dir dataset/train/lr --hr-dir dataset/train/hr
  python analyze_dataset.py --top-bad 20       # show worst 20 pairs
  python analyze_dataset.py --save-report report.csv
  python analyze_dataset.py --save-plots       # save distribution plots
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

LR_DIR = "dataset/train/lr"
HR_DIR = "dataset/train/hr"
SCALE = 4
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# Thresholds for flagging
# Calibrated for pixel art → SD upscaling datasets:
# - SSIM and sharpness are naturally low (SD smooths pixel edges by design)
# - color_drift is the most reliable indicator of a bad pair
SSIM_MIN = 0.40         # only catch structurally completely wrong pairs
COLOR_DRIFT_MAX = 10.0  # LAB units; above → SD changed the color palette too much
SHARPNESS_MIN = 0.0     # disabled — SD output is intentionally smoother than pixel art


# ── Optional imports ───────────────────────────────────────────────────────────

def _require(module: str, pkg: str):
    try:
        return __import__(module, fromlist=[""])
    except ImportError:
        print(f"[error] Missing dependency: pip install {pkg}")
        sys.exit(1)


# ── Metrics ───────────────────────────────────────────────────────────────────

def _nn_upscale(lr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    return np.array(Image.fromarray(lr).resize((target_w, target_h), Image.NEAREST))


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    skimage = _require("skimage.metrics", "scikit-image")
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=2, data_range=255))


def _color_drift(lr_up: np.ndarray, hr: np.ndarray) -> float:
    """Mean absolute error in CIELAB color space (perceptually uniform)."""
    from skimage.color import rgb2lab
    _require("skimage.color", "scikit-image")
    lr_lab = rgb2lab(lr_up.astype(np.float32) / 255.0)
    hr_lab = rgb2lab(hr.astype(np.float32) / 255.0)
    return float(np.mean(np.abs(lr_lab - hr_lab)))


def _laplacian_var(gray: np.ndarray) -> float:
    """Variance of Laplacian — proxy for sharpness."""
    try:
        from scipy.ndimage import laplace
        lap = laplace(gray.astype(np.float32))
    except ImportError:
        # Manual 3x3 Laplacian kernel fallback
        from numpy.lib.stride_tricks import sliding_window_view
        g = gray.astype(np.float32)
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        pad = np.pad(g, 1, mode="reflect")
        windows = sliding_window_view(pad, (3, 3))
        lap = np.sum(windows * kernel, axis=(-2, -1))
    return float(np.var(lap))


def _sharpness_ratio(lr_up: np.ndarray, hr: np.ndarray) -> float:
    lr_gray = lr_up.mean(axis=2)
    hr_gray = hr.mean(axis=2)
    lr_var = _laplacian_var(lr_gray)
    hr_var = _laplacian_var(hr_gray)
    if lr_var < 1e-6:
        return 0.0
    return hr_var / lr_var


def analyze_pair(lr_path: Path, hr_path: Path) -> dict:
    lr = np.array(Image.open(lr_path).convert("RGB"))
    hr = np.array(Image.open(hr_path).convert("RGB"))

    lh, lw = lr.shape[:2]
    hh, hw = hr.shape[:2]
    scale_x = hw / lw
    scale_y = hh / lh

    lr_up = _nn_upscale(lr, hh, hw)

    ssim_val = _ssim(lr_up, hr)
    drift_val = _color_drift(lr_up, hr)
    sharp_val = _sharpness_ratio(lr_up, hr)

    bad_reasons = []
    if ssim_val < SSIM_MIN:
        bad_reasons.append(f"low_ssim({ssim_val:.2f})")
    if drift_val > COLOR_DRIFT_MAX:
        bad_reasons.append(f"color_drift({drift_val:.1f})")
    if sharp_val < SHARPNESS_MIN:
        bad_reasons.append(f"hr_blurry({sharp_val:.2f})")
    if abs(scale_x - SCALE) > 0.5 or abs(scale_y - SCALE) > 0.5:
        bad_reasons.append(f"wrong_scale({scale_x:.1f}x{scale_y:.1f})")

    return {
        "filename": lr_path.name,
        "lr_size": f"{lw}x{lh}",
        "hr_size": f"{hw}x{hh}",
        "scale_x": round(scale_x, 2),
        "scale_y": round(scale_y, 2),
        "ssim": round(ssim_val, 4),
        "color_drift": round(drift_val, 4),
        "sharpness_ratio": round(sharp_val, 4),
        "bad": bool(bad_reasons),
        "reasons": "; ".join(bad_reasons),
    }


# Top-level wrapper so ProcessPoolExecutor can pickle it
def _analyze_pair_worker(args: tuple) -> dict:
    lr_p, hr_p = args
    try:
        return analyze_pair(Path(lr_p), Path(hr_p))
    except Exception as e:
        return {
            "filename": Path(lr_p).name, "lr_size": "?", "hr_size": "?",
            "scale_x": 0, "scale_y": 0,
            "ssim": 0.0, "color_drift": 999.0, "sharpness_ratio": 0.0,
            "bad": True, "reasons": f"error: {e}",
        }


# ── File listing (same logic as SRDataset) ────────────────────────────────────

def _list_pairs(lr_dir: str, hr_dir: str) -> list[tuple[Path, Path]]:
    lr_names = sorted(
        p.name for p in Path(lr_dir).iterdir() if p.suffix.lower() in EXTENSIONS
    )
    hr_names = set(
        p.name for p in Path(hr_dir).iterdir() if p.suffix.lower() in EXTENSIONS
    )
    matched = [n for n in lr_names if n in hr_names]
    if not matched:
        print(f"[error] No matching pairs found in {lr_dir} / {hr_dir}")
        sys.exit(1)
    return [(Path(lr_dir) / n, Path(hr_dir) / n) for n in matched]


# ── Report ────────────────────────────────────────────────────────────────────

def _print_summary(results: list[dict]):
    n = len(results)
    bad = [r for r in results if r["bad"]]
    print(f"\n{'=' * 65}")
    print(f"  Dataset: {n} pairs  |  Flagged: {len(bad)} ({100*len(bad)/n:.1f}%)")
    print(f"{'=' * 65}")
    for key in ("ssim", "color_drift", "sharpness_ratio"):
        vals = [r[key] for r in results]
        print(
            f"  {key:<18}  mean={np.mean(vals):.3f}  "
            f"std={np.std(vals):.3f}  "
            f"min={np.min(vals):.3f}  "
            f"max={np.max(vals):.3f}"
        )
    print(f"{'=' * 65}")


def _print_bad(results: list[dict], top_k: int):
    bad = [r for r in results if r["bad"]]
    if not bad:
        print("\nNo bad pairs found.")
        return

    # Sort by ssim ascending (worst first)
    bad.sort(key=lambda r: r["ssim"])
    show = bad[:top_k]
    print(f"\n{'─' * 65}")
    print(f"  Top {min(top_k, len(bad))} bad pairs (sorted by SSIM ascending):")
    print(f"{'─' * 65}")
    print(f"  {'FILENAME':<30} {'SSIM':>6} {'DRIFT':>7} {'SHARP':>6}  REASONS")
    print(f"  {'-'*30} {'------':>6} {'-------':>7} {'------':>6}  -------")
    for r in show:
        print(
            f"  {r['filename']:<30} {r['ssim']:>6.3f} "
            f"{r['color_drift']:>7.2f} {r['sharpness_ratio']:>6.2f}  {r['reasons']}"
        )
    print(f"{'─' * 65}")


def _save_csv(results: list[dict], path: str):
    fields = ["filename", "lr_size", "hr_size", "scale_x", "scale_y",
              "ssim", "color_drift", "sharpness_ratio", "bad", "reasons"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"Report saved to {path}")


def _save_plots(results: list[dict], out_dir: str = "."):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not found — skipping plots")
        return

    metrics = [
        ("ssim", "SSIM (NN-LR vs HR)", SSIM_MIN, "min"),
        ("color_drift", "Color Drift (LAB MAE)", COLOR_DRIFT_MAX, "max"),
        ("sharpness_ratio", "Sharpness Ratio (HR/LR)", SHARPNESS_MIN, "min"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Dataset Quality Distribution", fontsize=13, fontweight="bold")

    for ax, (key, title, threshold, direction) in zip(axes, metrics):
        vals = [r[key] for r in results]
        ax.hist(vals, bins=40, color="#4c8bf5", edgecolor="white", linewidth=0.4)
        color = "#e05c5c"
        if direction == "min":
            ax.axvline(threshold, color=color, linestyle="--", label=f"min={threshold}")
        else:
            ax.axvline(threshold, color=color, linestyle="--", label=f"max={threshold}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(key)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(out_dir) / "dataset_quality.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Distribution plot saved to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dataset quality analysis")
    parser.add_argument("--lr-dir", default=LR_DIR)
    parser.add_argument("--hr-dir", default=HR_DIR)
    parser.add_argument("--scale", type=int, default=SCALE)
    parser.add_argument("--top-bad", type=int, default=30,
                        help="How many bad pairs to print (default: 30)")
    parser.add_argument("--save-report", default=None,
                        help="Save full results to CSV (e.g. report.csv)")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save distribution plots to dataset_quality.png")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4),
                        help="Parallel workers (default: min(8, cpu_count))")
    args = parser.parse_args()

    pairs = _list_pairs(args.lr_dir, args.hr_dir)
    print(f"Analyzing {len(pairs)} pairs — {args.workers} workers")

    try:
        from tqdm import tqdm
        _tqdm = tqdm
    except ImportError:
        _tqdm = None

    results = [None] * len(pairs)
    tasks = [(str(lr_p), str(hr_p)) for lr_p, hr_p in pairs]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_analyze_pair_worker, t): i for i, t in enumerate(tasks)}
        if _tqdm:
            it = _tqdm(as_completed(futures), total=len(futures), unit="img", dynamic_ncols=True)
        else:
            it = as_completed(futures)
            done = 0
        for fut in it:
            i = futures[fut]
            results[i] = fut.result()
            if not _tqdm:
                done += 1
                if done % 50 == 0 or done == len(futures):
                    print(f"  {done}/{len(futures)}", end="\r", flush=True)

    if not _tqdm:
        print()  # newline after \r
    _print_summary(results)
    _print_bad(results, args.top_bad)

    if args.save_report:
        _save_csv(results, args.save_report)

    if args.save_plots:
        _save_plots(results)


if __name__ == "__main__":
    main()
