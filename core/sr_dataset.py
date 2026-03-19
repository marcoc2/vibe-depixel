import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def _list_images(directory: str) -> list[str]:
    return sorted(
        p.name for p in Path(directory).iterdir()
        if p.suffix.lower() in EXTENSIONS
    )


class SRDataset(Dataset):
    def __init__(
        self,
        lr_dir: str,
        hr_dir: str,
        patch_size: int = 64,
        scale: int = 4,
        augment: bool = True,
    ):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.patch_size = patch_size
        self.scale = scale
        self.augment = augment

        self.filenames = _list_images(lr_dir)
        hr_names = set(_list_images(hr_dir))
        self.filenames = [f for f in self.filenames if f in hr_names]

        if not self.filenames:
            raise RuntimeError(
                f"No matching image pairs found in {lr_dir} and {hr_dir}. "
                "Filenames must match between lr/ and hr/ directories."
            )
        print(f"Found {len(self.filenames)} image pairs.")

    def __len__(self) -> int:
        return len(self.filenames)

    def _load_pair(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        name = self.filenames[idx]
        lr = np.array(Image.open(os.path.join(self.lr_dir, name)).convert('RGB'))
        hr = np.array(Image.open(os.path.join(self.hr_dir, name)).convert('RGB'))
        return lr, hr

    def _random_crop(self, lr: np.ndarray, hr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lr_h, lr_w = lr.shape[:2]
        hr_h, hr_w = hr.shape[:2]
        ps = self.patch_size
        hr_ps = ps * self.scale

        # Clamp LR dims to what the HR can actually support
        usable_h = min(lr_h, hr_h // self.scale)
        usable_w = min(lr_w, hr_w // self.scale)

        if usable_h < ps or usable_w < ps:
            # Reflection-pad small images
            pad_h = max(ps - usable_h, 0)
            pad_w = max(ps - usable_w, 0)
            lr = np.pad(lr, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            hr = np.pad(hr, ((0, pad_h * self.scale), (0, pad_w * self.scale), (0, 0)), mode='reflect')
            usable_h = max(usable_h, ps)
            usable_w = max(usable_w, ps)

        top = random.randint(0, usable_h - ps)
        left = random.randint(0, usable_w - ps)
        lr_patch = lr[top:top + ps, left:left + ps]
        hr_patch = hr[top * self.scale:top * self.scale + hr_ps, left * self.scale:left * self.scale + hr_ps]
        return lr_patch, hr_patch

    @staticmethod
    def _augment(lr: np.ndarray, hr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Random horizontal flip
        if random.random() > 0.5:
            lr = np.flip(lr, axis=1)
            hr = np.flip(hr, axis=1)
        # Random vertical flip
        if random.random() > 0.5:
            lr = np.flip(lr, axis=0)
            hr = np.flip(hr, axis=0)
        # Random 90-degree rotation
        if random.random() > 0.5:
            lr = np.rot90(lr, axes=(0, 1))
            hr = np.rot90(hr, axes=(0, 1))
        return lr.copy(), hr.copy()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        lr, hr = self._load_pair(idx)
        lr, hr = self._random_crop(lr, hr)
        if self.augment:
            lr, hr = self._augment(lr, hr)

        # HWC uint8 → CHW float32 [0, 1]
        lr_t = torch.from_numpy(lr).permute(2, 0, 1).float() / 255.0
        hr_t = torch.from_numpy(hr).permute(2, 0, 1).float() / 255.0
        return lr_t, hr_t
