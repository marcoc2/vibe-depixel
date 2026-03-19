import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, n_feats: int, res_scale: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class Upsampler(nn.Sequential):
    def __init__(self, n_feats: int, scale: int):
        layers = []
        for _ in range(scale // 2):
            layers.append(nn.Conv2d(n_feats, n_feats * 4, 3, padding=1))
            layers.append(nn.PixelShuffle(2))
        super().__init__(*layers)


class EDSRLite(nn.Module):
    def __init__(self, n_feats: int = 64, n_resblocks: int = 16, res_scale: float = 0.1, scale: int = 4):
        super().__init__()
        self.head = nn.Conv2d(3, n_feats, 3, padding=1)
        self.body = nn.Sequential(
            *[ResBlock(n_feats, res_scale) for _ in range(n_resblocks)],
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )
        self.upsampler = Upsampler(n_feats, scale)
        self.tail = nn.Conv2d(n_feats, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head = self.head(x)
        body = self.body(head) + head
        up = self.upsampler(body)
        return self.tail(up)
