"""
Hyperparameter sweep script.
Runs all experiment combinations sequentially, tracks results in a leaderboard,
and keeps only the top-K best checkpoints globally.

Usage:
    python experiments.py
    python experiments.py --epochs 100   # faster, for quick comparison
    python experiments.py --top-k 4      # keep top 4 (default)
    python experiments.py --dry-run      # print plan without training
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from core.sr_train import train as sr_train
from core.sr_plot import plot_metrics, plot_comparison

# ── Experiment grid ───────────────────────────────────────────────────────────
# Add or remove combinations here freely.
ESRGAN_PRETRAINED = "1x-Focus.pth"
# Download from: github.com/JingyunLiang/SwinIR — Releases — real_sr
SWINIR_PRETRAINED = "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
# SWINIR_PRETRAINED = "4x-PBRify_UpscalerSIR-M_V2.pth"

EXPERIMENTS = [
    # default preset
    {"preset": "default", "lr": 5e-4, "batch_size": 16, "perceptual": False},
    {"preset": "default", "lr": 5e-4, "batch_size": 16, "perceptual": True},
    {"preset": "default", "lr": 2e-4, "batch_size": 16, "perceptual": True},
    {"preset": "default", "lr": 1e-4, "batch_size": 16, "perceptual": True},
    {"preset": "default", "lr": 5e-4, "batch_size": 32, "perceptual": True},
    # gemini preset
    {"preset": "gemini", "lr": 1e-4, "batch_size": 16, "perceptual": False},
    {"preset": "gemini", "lr": 1e-4, "batch_size": 16, "perceptual": True},
    {"preset": "gemini", "lr": 5e-5, "batch_size": 16, "perceptual": True},
    {"preset": "gemini", "lr": 2e-4, "batch_size": 16, "perceptual": True},
    {"preset": "gemini", "lr": 1e-4, "batch_size": 32, "perceptual": True},
    # esrgan fine-tune (lower LRs — model already converged)
    {
        "preset": "esrgan",
        "lr": 1e-4,
        "batch_size": 16,
        "perceptual": True,
        "pretrained": ESRGAN_PRETRAINED,
    },
    {
        "preset": "esrgan",
        "lr": 5e-4,
        "batch_size": 16,
        "perceptual": True,
        "pretrained": ESRGAN_PRETRAINED,
    },
    {
        "preset": "esrgan",
        "lr": 5e-5,
        "batch_size": 16,
        "perceptual": False,
        "pretrained": ESRGAN_PRETRAINED,
    },
    {
        "preset": "esrgan",
        "lr": 1e-5,
        "batch_size": 16,
        "perceptual": True,
        "pretrained": ESRGAN_PRETRAINED,
    },
    {
        "preset": "esrgan",
        "lr": 5e-5,
        "batch_size": 16,
        "perceptual": True,
        "pretrained": ESRGAN_PRETRAINED,
    },
    # lum loss variants — YCbCr-weighted loss, robust to color drift
    {
        "preset": "default",
        "lr": 5e-4,
        "batch_size": 16,
        "perceptual": True,
        "lum_loss": True,
        "perceptual_type": "freq",
    },
    {
        "preset": "default",
        "lr": 2e-4,
        "batch_size": 16,
        "perceptual": True,
        "lum_loss": True,
        "perceptual_type": "freq",
    },
    {
        "preset": "gemini",
        "lr": 1e-4,
        "batch_size": 16,
        "perceptual": True,
        "lum_loss": True,
        "perceptual_type": "freq",
    },
    {
        "preset": "gemini",
        "lr": 5e-5,
        "batch_size": 16,
        "perceptual": True,
        "lum_loss": True,
        "perceptual_type": "freq",
    },
    # esrgan_lum: fine-tuning requer LR baixo (modelo já pré-treinado)
    # esrgan_lum: Prodigy — LR adaptativo
    {
        "preset": "esrgan",
        "lr": 1.0,
        "batch_size": 8,
        "perceptual": True,
        "lum_loss": True,
        "perceptual_type": "freq",
        "use_adamw": "prodigy",
        "pretrained": ESRGAN_PRETRAINED,
    },
    {
        "preset": "esrgan",
        "lr": 1.0,
        "batch_size": 8,
        "perceptual": True,
        "lum_loss": True,
        "perceptual_type": "dists",
        "use_adamw": "prodigy",
        "pretrained": ESRGAN_PRETRAINED,
    },
    # swinir_lum: AdamW + weight_decay (preset defaults: patch=128, clip=0.1, wd=0.01)
    {
        "preset": "swinir",
        "lr": 1.0,  # ignorado pelo Prodigy, mas mantido pro run_id mostrar "prodigy"
        "batch_size": 2,
        "patch_size": 96,
        "perceptual": True,
        "lum_loss": True,
        "perceptual_type": "freq",
        "use_adamw": "prodigy",
        "pretrained": SWINIR_PRETRAINED,
    },
    {
        "preset": "swinir",
        "lr": 5e-5,
        "batch_size": 2,
        "perceptual": True,
        "lum_loss": True,
        "perceptual_type": "freq",
        "pretrained": SWINIR_PRETRAINED,
    },
    {
        "preset": "swinir",
        "lr": 1e-5,
        "batch_size": 2,
        "perceptual": True,
        "lum_loss": True,
        "perceptual_type": "freq",
        "pretrained": SWINIR_PRETRAINED,
    },
    # swinir_lum: Prodigy — LR adaptativo, elimina necessidade de escolher LR
    {
        "preset": "swinir",
        "lr": 1.0,
        "batch_size": 2,
        "perceptual": True,
        "lum_loss": True,
        "perceptual_type": "dists",
        "use_adamw": "prodigy",
        "pretrained": SWINIR_PRETRAINED,
    },
    # swinir 4x revisited — gan_warmup longo, patch 128, dists
    {
        "preset": "swinir",
        "lr": 1.0,
        "batch_size": 2,
        "patch_size": 128,
        "perceptual": True,
        "lum_loss": True,
        "perceptual_type": "dists",
        "use_adamw": "prodigy",
        "pretrained": SWINIR_PRETRAINED,
    },
    {
        "preset": "swinir",
        "lr": 1e-5,
        "batch_size": 2,
        "patch_size": 128,
        "perceptual": True,
        "lum_loss": True,
        "perceptual_type": "dists",
        "pretrained": SWINIR_PRETRAINED,
    },
    # enhance — ESRGAN 1x fine-tune (mesma arquitetura que esrgan, pretrained 1x-Focus.pth)
    {
        "preset": "enhance",
        "scale": 1,
        "lr": 1.0,
        "batch_size": 8,
        "patch_size": 256,
        "perceptual": True,
        "lum_loss": True,
        "perceptual_type": "dists",
        "pretrained": ESRGAN_PRETRAINED,
        "use_adamw": "prodigy",
    },
    {
        "preset": "enhance",
        "scale": 1,
        "lr": 1e-4,
        "batch_size": 8,
        "patch_size": 256,
        "perceptual": True,
        "lum_loss": True,
        "perceptual_type": "freq",
        "pretrained": ESRGAN_PRETRAINED,
    },
    {
        "preset": "enhance",
        "scale": 1,
        "lr": 1e-5,
        "batch_size": 8,
        "patch_size": 256,
        "perceptual": True,
        "lum_loss": True,
        "perceptual_type": "freq",
        "pretrained": ESRGAN_PRETRAINED,
    },
    {
        "preset": "enhance",
        "scale": 1,
        "lr": 1e-4,
        "batch_size": 8,
        "patch_size": 256,
        "perceptual": True,
        "lum_loss": True,
        "perceptual_type": "dists",
        "pretrained": ESRGAN_PRETRAINED,
    },
    # swinir1x — transfer learning parcial: RSTB blocks do SwinIR 4x, head substituído para 1x
    {
        "preset": "swinir1x",
        "scale": 1,
        "lr": 1.0,
        "batch_size": 2,
        "patch_size": 128,
        "perceptual": True,
        "lum_loss": True,
        "perceptual_type": "dists",
        "pretrained": SWINIR_PRETRAINED,
        "use_adamw": "prodigy",
    },
    {
        "preset": "swinir1x",
        "scale": 1,
        "lr": 1e-4,
        "batch_size": 2,
        "patch_size": 128,
        "perceptual": True,
        "lum_loss": True,
        "perceptual_type": "dists",
        "pretrained": SWINIR_PRETRAINED,
    },
    {
        "preset": "swinir1x",
        "scale": 1,
        "lr": 1e-5,
        "batch_size": 2,
        "patch_size": 128,
        "perceptual": True,
        "lum_loss": True,
        "perceptual_type": "dists",
        "pretrained": SWINIR_PRETRAINED,
    },
    # swinir1x_chaos — só conv_last treinável, kaiming init, lr agressivo
    {
        "preset": "swinir1x",
        "scale": 1,
        "lr": 1e-3,
        "batch_size": 4,
        "patch_size": 64,
        "perceptual": True,
        "lum_loss": True,
        "perceptual_type": "dists",
        "pretrained": SWINIR_PRETRAINED,
        "kaiming_init": True,
        "gan_weight": 0.1,
    },
    # enhance_nopx — perceptual-only (sem L1/pixel), força alucinação de textura via GAN+DISTS
    {
        "preset": "enhance",
        "scale": 1,
        "lr": 1e-4,
        "batch_size": 2,
        "patch_size": 198,
        "perceptual": True,
        "lum_loss": True,
        "perceptual_type": "dists_nopx",
        "pretrained": ESRGAN_PRETRAINED,
        "no_pixel_loss": True,
    },
    {
        "preset": "enhance",
        "scale": 1,
        "lr": 1.0,
        "batch_size": 2,
        "patch_size": 198,
        "perceptual": True,
        "lum_loss": True,
        "perceptual_type": "dists_nopx",
        "pretrained": ESRGAN_PRETRAINED,
        "no_pixel_loss": True,
        "use_adamw": "prodigy",
    },
]

LR_DIR = "dataset/train_cc_clean/lr"
HR_DIR = "dataset/train_cc_clean/hr"
CKPT_BASE = Path("checkpoints/experiments")
LEADERBOARD = CKPT_BASE / "leaderboard.json"
# ─────────────────────────────────────────────────────────────────────────────


def _run_id(cfg: dict) -> str:
    percep = "perc" if cfg["perceptual"] else "noperc"
    lr_str = (
        "prodigy"
        if cfg.get("use_adamw") == "prodigy"
        else f"{cfg['lr']:.0e}".replace("-0", "-")
    )
    lum = "_lum" if cfg.get("lum_loss") else ""
    pt = (
        f"_{cfg['perceptual_type']}"
        if cfg.get("lum_loss") and cfg.get("perceptual_type")
        else ""
    )
    return f"{cfg['preset']}_lr{lr_str}_bs{cfg['batch_size']}_{percep}{lum}{pt}"


def _load_leaderboard() -> list[dict]:
    if LEADERBOARD.exists():
        return json.loads(LEADERBOARD.read_text())
    return []


def _save_leaderboard(board: list[dict]):
    LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
    LEADERBOARD.write_text(json.dumps(board, indent=2))


def _prune(board: list[dict], top_k: int):
    """Keep only top_k successful entries by PSNR. Failed entries are always kept."""
    failed = [e for e in board if e.get("failed")]
    successful = [e for e in board if not e.get("failed")]
    successful.sort(key=lambda x: x["psnr"], reverse=True)
    keep = successful[:top_k]
    evict = successful[top_k:]
    for entry in evict:
        if not entry.get("checkpoint"):
            continue
        ckpt = Path(entry["checkpoint"])
        if ckpt.exists():
            ckpt.unlink()
            print(f"  [pruned] {ckpt.name}  (PSNR {entry['psnr']:.2f} dB)")
        run_dir = ckpt.parent
        if run_dir.exists() and not any(run_dir.iterdir()):
            run_dir.rmdir()
    return keep + failed


def _print_leaderboard(board: list[dict]):
    print("\n" + "=" * 70)
    print(f"{'RANK':<5} {'PSNR':>8}  {'RUN ID':<45} {'EPOCHS':>6}")
    print("-" * 70)
    for i, e in enumerate(board, 1):
        print(f"  {i:<3} {e['psnr']:>7.2f} dB  {e['run_id']:<45} {e['epochs']:>6}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only experiments matching this preset (e.g. esrgan)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run experiments even if already in leaderboard",
    )
    parser.add_argument(
        "--show", action="store_true", help="Print leaderboard and exit"
    )
    parser.add_argument("--lr-dir", default=LR_DIR)
    parser.add_argument("--hr-dir", default=HR_DIR)
    parser.add_argument(
        "--constant-lr",
        action="store_true",
        help="Disable CosineAnnealingLR — keep LR fixed for all experiments in this run",
    )
    parser.add_argument(
        "--gan", action="store_true", help="Enable PatchGAN adversarial training"
    )
    parser.add_argument(
        "--gan-weight",
        type=float,
        default=None,
        help="Adversarial loss weight (default: DEFAULT_GAN_WEIGHT in core/sr_train.py)",
    )
    parser.add_argument(
        "--gan-warmup",
        type=int,
        default=0,
        help="Epochs of supervised-only warmup before GAN activates",
    )
    parser.add_argument(
        "--preview-every",
        type=int,
        default=0,
        metavar="N",
        help="Save a preview image every N epochs per experiment (0 = disabled)",
    )
    args = parser.parse_args()

    if args.show:
        board = _load_leaderboard()
        if not board:
            print("No experiments in leaderboard yet.")
            return
        successful = sorted(
            [e for e in board if not e.get("failed")],
            key=lambda x: x["psnr"],
            reverse=True,
        )
        failed = [e for e in board if e.get("failed")]
        print(f"\n{'=' * 80}")
        print(f"  {'RANK':<5} {'PSNR':>8}  {'RUN ID':<48} {'EPOCHS':>6}  DATE")
        print(f"  {'-' * 75}")
        for i, e in enumerate(successful, 1):
            date = e.get("date", "")[:10]
            print(
                f"  {i:<5} {e['psnr']:>7.2f} dB  {e['run_id']:<48} {e['epochs']:>6}  {date}"
            )
        if failed:
            print(f"\n  Failed ({len(failed)}):")
            for e in failed:
                print(f"    {e['run_id']}  — {e.get('failed', '?')[:60]}")
        print(f"{'=' * 80}")
        if successful:
            best = successful[0]
            print(f"\n  Best checkpoint: {best.get('checkpoint', 'N/A')}\n")
        return

    def _matches(cfg: dict) -> bool:
        if not args.only:
            return True
        if args.only == "lum":
            return bool(cfg.get("lum_loss"))
        if args.only == "nopx":
            return bool(cfg.get("no_pixel_loss"))
        if args.only.endswith("_lum"):
            return cfg["preset"] == args.only[:-4] and bool(cfg.get("lum_loss"))
        if args.only.endswith("_nopx"):
            return cfg["preset"] == args.only[:-5] and bool(cfg.get("no_pixel_loss"))
        return cfg["preset"] == args.only

    experiments = [e for e in EXPERIMENTS if _matches(e)]

    print(f"\n{'=' * 70}")
    print(
        f"  Sweep: {len(experiments)} experiments  |  {args.epochs} epochs each  |  keep top {args.top_k}"
    )
    if args.only:
        print(f"  Filter: preset={args.only}")
    print(f"{'=' * 70}\n")

    if args.dry_run:
        for i, cfg in enumerate(experiments, 1):
            print(f"  [{i:02d}] {_run_id(cfg)}")
        return

    board = _load_leaderboard()
    done_ids = {e["run_id"] for e in board}

    for i, cfg in enumerate(experiments, 1):
        run_id = _run_id(cfg)
        print(f"\n{'─' * 70}")
        print(f"  Experiment {i}/{len(experiments)}: {run_id}")
        print(f"{'─' * 70}")

        if run_id in done_ids:
            if not args.force:
                print(f"  [skip] already in leaderboard")
                continue
            board = [e for e in board if e["run_id"] != run_id]
            print(f"  [force] removing previous entry, re-running")

        try:
            best_psnr, best_ckpt = sr_train(
                lr_dir=args.lr_dir,
                hr_dir=args.hr_dir,
                epochs=args.epochs,
                lr=cfg["lr"],
                batch_size=cfg["batch_size"],
                use_perceptual=cfg["perceptual"],
                use_lum_loss=cfg.get("lum_loss", False),
                perceptual_type=cfg.get("perceptual_type", "freq"),
                preset=cfg["preset"],
                checkpoint_dir=str(CKPT_BASE / run_id),
                pretrained_path=cfg.get("pretrained"),
                scale=cfg.get("scale", 4),
                patch_size=cfg.get("patch_size"),
                grad_clip=cfg.get("grad_clip"),
                weight_decay=cfg.get("weight_decay"),
                use_adamw=cfg.get("use_adamw"),
                constant_lr=args.constant_lr,
                use_gan=args.gan,
                **(
                    {"gan_weight": args.gan_weight}
                    if args.gan_weight is not None
                    else {}
                ),
                gan_warmup=args.gan_warmup,
                preview_every=args.preview_every,
                no_pixel_loss=cfg.get("no_pixel_loss", False),
                kaiming_init=cfg.get("kaiming_init", False),
                **({"gan_weight": cfg["gan_weight"]} if "gan_weight" in cfg else {}),
            )
        except RuntimeError as e:
            print(f"  [FAILED] {run_id}: {e}")
            board.append(
                {
                    "run_id": run_id,
                    "psnr": -999.0,
                    "checkpoint": None,
                    "epochs": args.epochs,
                    "config": cfg,
                    "date": datetime.now().isoformat(timespec="seconds"),
                    "failed": str(e),
                }
            )
            _save_leaderboard(board)
            continue

        # Move best checkpoint to a flat, clearly named file
        final_ckpt = CKPT_BASE / f"{run_id}_best.pth"
        if not Path(best_ckpt).exists():
            print(
                f"  [FAILED] {run_id}: best checkpoint not found (PSNR never improved)"
            )
            board.append(
                {
                    "run_id": run_id,
                    "psnr": -999.0,
                    "checkpoint": None,
                    "epochs": args.epochs,
                    "config": cfg,
                    "date": datetime.now().isoformat(timespec="seconds"),
                    "failed": "best checkpoint not found",
                }
            )
            _save_leaderboard(board)
            continue
        shutil.copy2(best_ckpt, final_ckpt)

        board.append(
            {
                "run_id": run_id,
                "psnr": round(best_psnr, 4),
                "checkpoint": str(final_ckpt),
                "epochs": args.epochs,
                "config": cfg,
                "date": datetime.now().isoformat(timespec="seconds"),
            }
        )

        # Plot individual experiment
        metrics_path = CKPT_BASE / run_id / cfg["preset"] / "metrics.json"
        if metrics_path.exists():
            plot_metrics(
                metrics_path, title=run_id, save_path=CKPT_BASE / f"{run_id}_plot.png"
            )

        board = _prune(board, args.top_k)
        _save_leaderboard(board)
        _print_leaderboard(board)

    print("\nSweep complete.")
    final_board = _load_leaderboard()
    _print_leaderboard(final_board)

    # Comparison plot of all successful experiments
    pairs = [
        (e["run_id"], CKPT_BASE / e["run_id"] / e["config"]["preset"] / "metrics.json")
        for e in final_board
        if not e.get("failed") and e.get("config")
    ]
    if len(pairs) > 1:
        plot_comparison(pairs, save_path=CKPT_BASE / "comparison.png")


if __name__ == "__main__":
    main()
