import os
import math
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from PIL import Image

from core.sr_model import EDSRLite, load_model, ESRGAN_PRESET, get_training_defaults
from core.sr_dataset import SRDataset


def psnr(sr: torch.Tensor, hr: torch.Tensor) -> float:
    mse = torch.mean((sr - hr) ** 2).item()
    if mse == 0:
        return float('inf')
    if mse < 0 or math.isnan(mse) or math.isinf(mse):
        return 0.0
    return 10 * math.log10(1.0 / mse)


def psnr_y(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """PSNR on luminance channel only (Y of YCbCr). Fair comparison across loss types."""
    sr_y = 0.299 * sr[:, 0:1] + 0.587 * sr[:, 1:2] + 0.114 * sr[:, 2:3]
    hr_y = 0.299 * hr[:, 0:1] + 0.587 * hr[:, 1:2] + 0.114 * hr[:, 2:3]
    return psnr(sr_y, hr_y)


def train(
    lr_dir: str,
    hr_dir: str,
    val_lr_dir: str | None = None,
    val_hr_dir: str | None = None,
    patch_size: int | None = None,
    scale: int = 4,
    batch_size: int = 16,
    epochs: int = 200,
    lr: float = 5e-4,
    checkpoint_dir: str = 'checkpoints',
    use_perceptual: bool = False,
    use_lum_loss: bool = False,
    perceptual_type: str = 'freq',
    preset: str = 'default',
    resume: bool = False,
    pretrained_path: str | None = None,
    grad_clip: float | None = None,
    weight_decay: float | None = None,
    use_adamw: bool | None = None,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Apply per-preset training defaults for any value not explicitly set
    td = get_training_defaults(preset)
    if patch_size  is None: patch_size  = td["patch_size"]
    if grad_clip   is None: grad_clip   = td["grad_clip"]
    if weight_decay is None: weight_decay = td["weight_decay"]
    if use_adamw   is None: use_adamw   = td["use_adamw"]
    print(f"Training: patch={patch_size}  grad_clip={grad_clip}  "
          f"wd={weight_decay}  optimizer={'AdamW' if use_adamw else 'Adam'}")

    # Datasets
    full_dataset = SRDataset(lr_dir, hr_dir, patch_size=patch_size, scale=scale, augment=True)

    if val_lr_dir and val_hr_dir:
        train_dataset = full_dataset
        val_dataset = SRDataset(val_lr_dir, val_hr_dir, patch_size=patch_size, scale=scale, augment=False)
    else:
        n_val = max(1, len(full_dataset) // 10)
        n_train = len(full_dataset) - n_val
        train_dataset, val_dataset = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"Split: {n_train} train, {n_val} val")

    num_workers = min(8, os.cpu_count() or 4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)
    print(f"DataLoader: {num_workers} workers")

    # Model
    model = load_model(preset, device, pretrained_path=pretrained_path, scale=scale)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Loss
    if use_lum_loss:
        criterion = LumSRLoss(device, use_perceptual=use_perceptual, perceptual_type=perceptual_type)
        percep_label = f" + {perceptual_type}" if use_perceptual else " + freq"
        print(f"Loss: YCbCr-weighted (Y=1.0, CbCr=0.1){percep_label}")
    else:
        criterion = SRLoss(device, use_perceptual=use_perceptual)
        print(f"Loss: L1 + gradient{' + perceptual (VGG19)' if use_perceptual else ''}")

    if use_adamw == "prodigy":
        try:
            from prodigyopt import Prodigy
        except ImportError:
            raise ImportError("Prodigy requires: pip install prodigyopt")
        optimizer = Prodigy(model.parameters(), weight_decay=weight_decay, safeguard_warmup=True)
        # Prodigy adapts LR internally — cosine still works as a shape multiplier
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.1)
        print(f"Optimizer: Prodigy (adaptive LR)  wd={weight_decay}")
    elif use_adamw:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Checkpoints separated by preset
    preset_dir = os.path.join(checkpoint_dir, preset)
    os.makedirs(preset_dir, exist_ok=True)
    best_psnr = 0.0
    start_epoch = 1
    metrics = {"loss": [], "psnr": [], "psnr_y": [], "lr": []}
    metrics_path = os.path.join(preset_dir, 'metrics.json')

    # Resume from checkpoint
    resume_path = os.path.join(preset_dir, 'sr_training_state.pth')
    if resume and os.path.exists(resume_path):
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state['model'], strict=False)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch = state['epoch'] + 1
        best_psnr = state['best_psnr']
        print(f"Resumed from epoch {state['epoch']} (best PSNR: {best_psnr:.2f} dB)")
        if os.path.exists(metrics_path):
            import json as _json
            metrics = _json.loads(Path(metrics_path).read_text())

    for epoch in range(start_epoch, epochs + 1):
        # Train
        model.train()
        epoch_loss = 0.0
        for lr_batch, hr_batch in train_loader:
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
            sr = model(lr_batch)
            loss = criterion(sr, hr_batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        if math.isnan(avg_loss) or math.isinf(avg_loss) or avg_loss > 1e6:
            print(f"  [abort] Loss diverged ({avg_loss:.2f}) at epoch {epoch} — stopping early.")
            raise RuntimeError(f"Loss diverged at epoch {epoch}: {avg_loss}")

        # Validate
        model.eval()
        val_psnr = 0.0
        val_psnr_y = 0.0
        with torch.no_grad():
            for lr_batch, hr_batch in val_loader:
                lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
                sr = model(lr_batch)
                val_psnr   += psnr(sr, hr_batch)
                val_psnr_y += psnr_y(sr, hr_batch)
        val_psnr   /= len(val_loader)
        val_psnr_y /= len(val_loader)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  PSNR={val_psnr:.2f} dB  PSNR-Y={val_psnr_y:.2f} dB  lr={current_lr:.1e}")

        metrics["loss"].append(round(avg_loss, 6))
        metrics["psnr"].append(round(val_psnr, 4))
        metrics["psnr_y"].append(round(val_psnr_y, 4))
        metrics["lr"].append(current_lr)
        Path(metrics_path).write_text(json.dumps(metrics))

        # Checkpointing: lum_loss experiments use PSNR-Y as save criterion
        # (model optimizes Y — RGB PSNR is unfairly penalized by color drift)
        save_metric = val_psnr_y if use_lum_loss else val_psnr
        if save_metric > best_psnr:
            best_psnr = save_metric
            torch.save(model.state_dict(), os.path.join(preset_dir, 'sr_model_best.pth'))
        if epoch % 10 == 0:
            ckpt_path = os.path.join(preset_dir, f'sr_model_epoch{epoch}.pth')
            torch.save(model.state_dict(), ckpt_path)
            # Keep only last 4 epoch checkpoints
            epoch_ckpts = sorted(Path(preset_dir).glob('sr_model_epoch*.pth'), key=lambda p: p.stat().st_mtime)
            for old in epoch_ckpts[:-4]:
                old.unlink()

        # Save training state for resume
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_psnr': best_psnr,
        }, resume_path)

    print(f"\nTraining complete. Best PSNR: {best_psnr:.2f} dB")
    print(f"Best model saved to {os.path.join(preset_dir, 'sr_model_best.pth')}")
    return best_psnr, os.path.join(preset_dir, 'sr_model_best.pth')


def infer(image_path: str, checkpoint: str, output_path: str | None = None, scale: int = 4,
          preset: str = 'default', pretrained_path: str | None = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from core.sr_model import SPANDREL_PRESETS
    if preset in SPANDREL_PRESETS:
        # Spandrel models: load architecture from pretrained, then apply fine-tuned weights if different
        if not pretrained_path:
            raise ValueError(f"--pretrained is required for --preset {preset} inference")
        model = load_model(preset, device, pretrained_path=pretrained_path)
        if os.path.abspath(checkpoint) != os.path.abspath(pretrained_path):
            if os.path.exists(checkpoint):
                model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=False)
                print(f"Applied fine-tuned weights from {checkpoint}")
    else:
        model = EDSRLite.from_preset(preset, scale=scale).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    img = Image.open(image_path).convert('RGB')
    lr = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    lr = lr.to(device)

    with torch.no_grad():
        sr = model(lr)

    sr = sr.squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    sr_img = Image.fromarray((sr * 255).astype(np.uint8))

    if output_path is None:
        os.makedirs("output", exist_ok=True)
        name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join("output", f"{name}_sr4x.png")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sr_img.save(output_path)
    print(f"SR output saved to {output_path}")
    return output_path


class _GradientLoss(nn.Module):
    """Penalizes differences in spatial gradients (edges) between SR and HR."""

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        # Horizontal and vertical gradients via finite differences
        sr_dx = sr[:, :, :, 1:] - sr[:, :, :, :-1]
        sr_dy = sr[:, :, 1:, :] - sr[:, :, :-1, :]
        hr_dx = hr[:, :, :, 1:] - hr[:, :, :, :-1]
        hr_dy = hr[:, :, 1:, :] - hr[:, :, :-1, :]
        return torch.mean(torch.abs(sr_dx - hr_dx)) + torch.mean(torch.abs(sr_dy - hr_dy))


class _VGGPerceptualLoss(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:36].eval().to(device)
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        return self.criterion(self.vgg(sr), self.vgg(hr))


class SRLoss(nn.Module):
    """Combined loss: L1 + gradient + optional perceptual (VGG19)."""

    def __init__(self, device: torch.device, use_perceptual: bool = True,
                 w_l1: float = 1.0, w_grad: float = 0.5, w_percep: float = 0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.grad = _GradientLoss()
        self.percep = _VGGPerceptualLoss(device) if use_perceptual else None
        self.w_l1 = w_l1
        self.w_grad = w_grad
        self.w_percep = w_percep

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        loss = self.w_l1 * self.l1(sr, hr) + self.w_grad * self.grad(sr, hr)
        if self.percep is not None:
            loss = loss + self.w_percep * self.percep(sr, hr)
        return loss


def _rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    """Convert (B, 3, H, W) RGB [0,1] to YCbCr. BT.601 coefficients."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y  =  0.299  * r + 0.587  * g + 0.114  * b
    cb = -0.16874 * r - 0.33126 * g + 0.5    * b + 0.5
    cr =  0.5     * r - 0.41869 * g - 0.08131 * b + 0.5
    return torch.cat([y, cb, cr], dim=1)


class _FrequencyLoss(nn.Module):
    """FFT-based loss on luminance channel.

    Compares magnitude spectra of SR_Y and HR_Y. Directly penalizes missing
    high-frequency content (sharpness, fine edges) without any color influence.
    No pretrained weights needed — purely mathematical.
    """

    def forward(self, sr_y: torch.Tensor, hr_y: torch.Tensor) -> torch.Tensor:
        # sr_y, hr_y: (B, 1, H, W)
        sr_fft = torch.fft.rfft2(sr_y, norm="ortho")
        hr_fft = torch.fft.rfft2(hr_y, norm="ortho")
        # L1 on magnitude spectrum — penalizes missing frequencies equally
        return torch.mean(torch.abs(sr_fft.abs() - hr_fft.abs()))


class _DISTSLoss(nn.Module):
    """DISTS: Deep Image Structure and Texture Similarity (Ding et al. 2020).

    Designed to be invariant to texture resampling and mild color shifts while
    preserving sensitivity to structural differences. Better than LPIPS/VGG for
    SR because it doesn't penalize perceptually equivalent variations.
    Requires: pip install piq
    """

    def __init__(self, device: torch.device):
        super().__init__()
        try:
            from piq import DISTS
            self.dists = DISTS().to(device)
            self.dists.eval()
            for p in self.dists.parameters():
                p.requires_grad = False
        except ImportError:
            raise ImportError("DISTS requires piq: pip install piq")

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        return self.dists(sr.clamp(0, 1), hr.clamp(0, 1))


class LumSRLoss(nn.Module):
    """YCbCr-weighted loss: heavy on luminance (Y), light on chrominance (Cb/Cr).

    perceptual_type options:
      'vgg'   — VGG19 features on RGB (standard, color-aware)
      'dists' — DISTS on RGB (color-invariant, structure/texture focused)
      'freq'  — FFT magnitude on Y only (no pretrained weights, pure frequency)
      'none'  — no perceptual term
    """

    def __init__(self, device: torch.device, use_perceptual: bool = True,
                 perceptual_type: str = "freq",
                 w_y: float = 1.0, w_cbcr: float = 0.1,
                 w_freq: float = 0.05, w_percep: float = 0.1):
        super().__init__()
        self.l1   = nn.L1Loss()
        self.grad = _GradientLoss()
        self.freq = _FrequencyLoss()
        self.w_y      = w_y
        self.w_cbcr   = w_cbcr
        self.w_freq   = w_freq
        self.w_percep = w_percep

        self.percep = None
        self.percep_type = perceptual_type if use_perceptual else "none"
        if use_perceptual:
            if perceptual_type == "vgg":
                self.percep = _VGGPerceptualLoss(device)
            elif perceptual_type == "dists":
                self.percep = _DISTSLoss(device)
            elif perceptual_type == "freq":
                self.percep = None  # freq is always applied separately below
            elif perceptual_type != "none":
                raise ValueError(f"Unknown perceptual_type '{perceptual_type}'. "
                                 "Choose: vgg, dists, freq, none")

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        sr_yuv = _rgb_to_ycbcr(sr)
        hr_yuv = _rgb_to_ycbcr(hr)
        sr_y, hr_y       = sr_yuv[:, 0:1], hr_yuv[:, 0:1]
        sr_cbcr, hr_cbcr = sr_yuv[:, 1:],  hr_yuv[:, 1:]

        # Luminance: L1 + spatial gradient
        loss = self.w_y * (self.l1(sr_y, hr_y) + 0.5 * self.grad(sr_y, hr_y))
        # Chrominance: L1 only, low weight
        loss = loss + self.w_cbcr * self.l1(sr_cbcr, hr_cbcr)
        # Frequency loss on Y (always included when percep_type == 'freq' or as extra)
        if self.percep_type in ("freq", "none"):
            loss = loss + self.w_freq * self.freq(sr_y, hr_y)
        # Perceptual: VGG or DISTS on RGB
        if self.percep is not None:
            loss = loss + self.w_percep * self.percep(sr, hr)
        return loss
