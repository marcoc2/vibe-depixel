import os
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from PIL import Image

from core.sr_model import EDSRLite
from core.sr_dataset import SRDataset


def psnr(sr: torch.Tensor, hr: torch.Tensor) -> float:
    mse = torch.mean((sr - hr) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10 * math.log10(1.0 / mse)


def train(
    lr_dir: str,
    hr_dir: str,
    val_lr_dir: str | None = None,
    val_hr_dir: str | None = None,
    patch_size: int = 64,
    scale: int = 4,
    batch_size: int = 16,
    epochs: int = 200,
    lr: float = 5e-4,
    checkpoint_dir: str = 'checkpoints',
    use_perceptual: bool = False,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Datasets
    full_dataset = SRDataset(lr_dir, hr_dir, patch_size=patch_size, scale=scale, augment=True)

    if val_lr_dir and val_hr_dir:
        train_dataset = full_dataset
        val_dataset = SRDataset(val_lr_dir, val_hr_dir, patch_size=patch_size, scale=scale, augment=False)
    else:
        n_val = max(1, len(full_dataset) // 10)
        n_train = len(full_dataset) - n_val
        train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
        print(f"Split: {n_train} train, {n_val} val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = EDSRLite(scale=scale).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Loss
    criterion = SRLoss(device, use_perceptual=use_perceptual)
    print(f"Loss: L1 + gradient{' + perceptual (VGG19)' if use_perceptual else ''}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_psnr = 0.0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        epoch_loss = 0.0
        for lr_batch, hr_batch in train_loader:
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
            sr = model(lr_batch)
            loss = criterion(sr, hr_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        # Validate
        model.eval()
        val_psnr = 0.0
        with torch.no_grad():
            for lr_batch, hr_batch in val_loader:
                lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
                sr = model(lr_batch)
                val_psnr += psnr(sr, hr_batch)
        val_psnr /= len(val_loader)

        print(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  PSNR={val_psnr:.2f} dB  lr={scheduler.get_last_lr()[0]:.1e}")

        # Checkpointing
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'sr_model_best.pth'))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'sr_model_epoch{epoch}.pth'))

    print(f"\nTraining complete. Best PSNR: {best_psnr:.2f} dB")
    print(f"Best model saved to {os.path.join(checkpoint_dir, 'sr_model_best.pth')}")


def infer(image_path: str, checkpoint: str, output_path: str | None = None, scale: int = 4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EDSRLite(scale=scale).to(device)
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
