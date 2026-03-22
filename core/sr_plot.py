"""Plotting utilities for SR training metrics."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_metrics(metrics_path: str | Path, title: str = "", save_path: str | Path | None = None):
    """
    Plot loss, PSNR and LR from a metrics.json file.
    Saves to save_path (PNG) or shows interactively if None.
    """
    data = json.loads(Path(metrics_path).read_text())
    epochs = list(range(1, len(data["loss"]) + 1))

    has_psnr_y = "psnr_y" in data and data["psnr_y"]
    n_rows = 4 if has_psnr_y else 3

    fig = plt.figure(figsize=(14, 4 * n_rows))
    fig.suptitle(title or Path(metrics_path).parent.name, fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(n_rows, 1, hspace=0.45)

    # Loss
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(epochs, data["loss"], color="#e05c5c", linewidth=1.2)
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("symlog")

    # PSNR (RGB)
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(epochs, data["psnr"], color="#4c8bf5", linewidth=1.2, label="PSNR (RGB)")
    if has_psnr_y:
        ax2.plot(epochs, data["psnr_y"], color="#a78bfa", linewidth=1.0,
                 linestyle="--", label="PSNR-Y")
        ax2.legend(fontsize=8)
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_title("Validation PSNR")
    ax2.grid(True, alpha=0.3)
    best_epoch = data["psnr"].index(max(data["psnr"])) + 1
    best_val = max(data["psnr"])
    ax2.axvline(best_epoch, color="#4c8bf5", linestyle="--", alpha=0.5)
    ax2.annotate(f"best: {best_val:.2f} dB @ ep{best_epoch}",
                 xy=(best_epoch, best_val), xytext=(10, -15),
                 textcoords="offset points", fontsize=8, color="#4c8bf5")

    # LR
    ax3 = fig.add_subplot(gs[n_rows - 1])
    ax3.plot(epochs, data["lr"], color="#5cb85c", linewidth=1.2)
    ax3.set_ylabel("Learning Rate")
    ax3.set_xlabel("Epoch")
    ax3.set_title("Learning Rate Schedule")
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_comparison(metrics_paths: list[tuple[str, str | Path]], save_path: str | Path | None = None):
    """
    Overlay multiple experiments on the same Loss and PSNR axes for comparison.
    metrics_paths: list of (label, path_to_metrics.json)
    """
    fig, (ax_loss, ax_psnr) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("Experiment Comparison", fontsize=13, fontweight="bold")

    for label, path in metrics_paths:
        if not Path(path).exists():
            continue
        data = json.loads(Path(path).read_text())
        epochs = list(range(1, len(data["loss"]) + 1))
        ax_loss.plot(epochs, data["loss"], linewidth=1.0, label=label)
        ax_psnr.plot(epochs, data["psnr"], linewidth=1.0, label=label)

    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training Loss")
    ax_loss.set_yscale("symlog")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=7)

    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.set_xlabel("Epoch")
    ax_psnr.set_title("Validation PSNR")
    ax_psnr.grid(True, alpha=0.3)
    ax_psnr.legend(fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()
