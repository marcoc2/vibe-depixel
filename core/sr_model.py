import torch
import torch.nn as nn


PRESETS = {
    "default": {
        "n_feats": 64,
        "n_resblocks": 16,
        "res_scale": 0.1,
        "activation": "relu",
    },
    "gemini": {
        "n_feats": 128,
        "n_resblocks": 32,
        "res_scale": 0.1,
        "activation": "leakyrelu",
    },
}

ESRGAN_PRESET = "esrgan"  # kept for backwards compat

# Any preset listed here is loaded via spandrel from a pretrained .pth
# Add new transformer/GAN architectures here without touching load_model()
SPANDREL_PRESETS = {"esrgan", "swinir", "enhance"}

# Presets that use partial transfer learning (load 4x pretrained, adapt head for 1x)
PARTIAL_TRANSFER_PRESETS = {"swinir1x"}

# Per-preset training hyperparameter defaults.
# Spandrel presets (esrgan, swinir, enhance) are fine-tuning scenarios — use conservative settings.
# SwinIR specifically benefits from larger patches (more attention context) and AdamW.
TRAINING_DEFAULTS = {
    "default":   {"patch_size": 64,  "grad_clip": 1.0,  "weight_decay": 0.0,  "use_adamw": False},
    "gemini":    {"patch_size": 64,  "grad_clip": 1.0,  "weight_decay": 0.0,  "use_adamw": False},
    "esrgan":    {"patch_size": 64,  "grad_clip": 0.1,  "weight_decay": 0.01, "use_adamw": True},
    "swinir":    {"patch_size": 128, "grad_clip": 0.1,  "weight_decay": 0.01, "use_adamw": True},
    "enhance":   {"patch_size": 64,  "grad_clip": 0.1,  "weight_decay": 0.01, "use_adamw": True},
    "swinir1x":  {"patch_size": 256, "grad_clip": 0.1,  "weight_decay": 0.01, "use_adamw": True},
}

def get_training_defaults(preset: str) -> dict:
    return TRAINING_DEFAULTS.get(preset, TRAINING_DEFAULTS["default"])


def _make_activation(name: str) -> nn.Module:
    if name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    return nn.ReLU(inplace=True)


class ResBlock(nn.Module):
    def __init__(self, n_feats: int, res_scale: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            _make_activation(activation),
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
    def __init__(self, n_feats: int = 64, n_resblocks: int = 16,
                 res_scale: float = 0.1, scale: int = 4, activation: str = "relu"):
        super().__init__()
        self.head = nn.Conv2d(3, n_feats, 3, padding=1)
        self.body = nn.Sequential(
            *[ResBlock(n_feats, res_scale, activation) for _ in range(n_resblocks)],
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )
        self.upsampler = Upsampler(n_feats, scale)
        self.tail = nn.Conv2d(n_feats, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head = self.head(x)
        body = self.body(head) + head
        up = self.upsampler(body)
        return self.tail(up)

    @classmethod
    def from_preset(cls, preset: str, scale: int = 4) -> "EDSRLite":
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}")
        cfg = PRESETS[preset]
        print(f"Preset '{preset}': {cfg['n_feats']}ch, {cfg['n_resblocks']} blocks, {cfg['activation']}")
        return cls(
            n_feats=cfg["n_feats"],
            n_resblocks=cfg["n_resblocks"],
            res_scale=cfg["res_scale"],
            scale=scale,
            activation=cfg["activation"],
        )


class PatchGAN(nn.Module):
    """PatchGAN discriminator with Spectral Normalization.

    Outputs a spatial map of real/fake scores (no sigmoid — use LSGAN MSE loss).
    Receptive field covers 70x70 patches at default n_layers=3.
    """

    def __init__(self, in_ch: int = 3, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        from torch.nn.utils import spectral_norm
        layers = [
            nn.Conv2d(in_ch, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        ]
        nf = ndf
        for _ in range(1, n_layers):
            nf_prev, nf = nf, min(nf * 2, 512)
            layers += [
                spectral_norm(nn.Conv2d(nf_prev, nf, 4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, True),
            ]
        nf_prev, nf = nf, min(nf * 2, 512)
        layers += [
            spectral_norm(nn.Conv2d(nf_prev, nf, 4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(nf, 1, 4, stride=1, padding=1)),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_esrgan(pretrained_path: str, device: torch.device) -> nn.Module:
    """Load ESRGAN/SwinIR model via spandrel. Returns a plain nn.Module."""
    from spandrel import ModelLoader
    descriptor = ModelLoader(device=device).load_from_file(pretrained_path)
    model = descriptor.model
    print(f"Loaded ESRGAN: {descriptor.architecture} | scale={descriptor.scale} | {sum(p.numel() for p in model.parameters()):,} params")
    return model


def load_swinir1x(pretrained_path: str, device: torch.device, kaiming_init: bool = False) -> nn.Module:
    """Partial transfer learning: load SwinIR 4x pretrained, replace upsampling head for 1x.

    RSTB blocks (the 'intelligence') are kept from 4x pretrained.
    Only conv_last is replaced with a fresh Conv2d(embed_dim → 3) for 1x restoration.
    Uses the SwinIR denoising forward path: out = input + conv_last(rstb_features)
    """
    from spandrel import ModelLoader
    descriptor = ModelLoader(device=device).load_from_file(pretrained_path)
    model = descriptor.model

    # embed_dim = output channels of conv_after_body (same as RSTB output dim)
    embed_dim = model.conv_after_body.out_channels

    # Remove upsampling head — replace with Identity so keys vanish from state_dict
    # (spandrel detects upsampler type from key presence: conv_before_upsample → pixelshuffle)
    model.upsample = nn.Identity()
    if hasattr(model, "conv_before_upsample"):
        model.conv_before_upsample = nn.Identity()

    # New 1x restoration head: Conv2d(embed_dim → 3)
    model.conv_last = nn.Conv2d(embed_dim, 3, 3, 1, 1).to(device)
    if kaiming_init:
        nn.init.kaiming_normal_(model.conv_last.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(model.conv_last.bias)
    else:
        nn.init.zeros_(model.conv_last.weight)
        nn.init.zeros_(model.conv_last.bias)

    # Switch to 1x restoration forward path
    model.upsampler = ""
    model.upscale = 1

    # Freeze everything except conv_last — only the new head is trained
    for name, param in model.named_parameters():
        if not name.startswith("conv_last"):
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Loaded SwinIR1x (partial transfer from 4x) | embed_dim={embed_dim} | {total:,} params total | {trainable:,} trainable (conv_last only)")
    return model


def load_model(preset: str, device: torch.device, pretrained_path: str | None = None, scale: int = 4, kaiming_init: bool = False) -> nn.Module:
    """Unified model loader for all presets.
    - Presets in SPANDREL_PRESETS (esrgan, swinir, enhance): loaded via spandrel from pretrained_path
    - Presets in PARTIAL_TRANSFER_PRESETS (swinir1x): 4x pretrained with 1x head replacement
    - Presets in PRESETS (default, gemini): EDSRLite trained from scratch
    """
    if preset in PARTIAL_TRANSFER_PRESETS:
        if not pretrained_path:
            raise ValueError(f"--pretrained is required for --preset {preset}")
        return load_swinir1x(pretrained_path, device, kaiming_init=kaiming_init)
    if preset in SPANDREL_PRESETS:
        if not pretrained_path:
            raise ValueError(f"--pretrained is required for --preset {preset}")
        return load_esrgan(pretrained_path, device)
    return EDSRLite.from_preset(preset, scale=scale).to(device)
