"""CycleGAN model components.

Generator:  ResNet9 (encoder → 9 residual blocks → decoder), InstanceNorm2d, tanh.
Discriminator: PatchGAN 70×70 (4 strided convs + 1 output conv), InstanceNorm2d, LeakyReLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

def init_weights(module: nn.Module, gain: float = 0.02) -> None:
    """Gaussian init for conv/linear, constant for norm layers."""
    classname = module.__class__.__name__
    if hasattr(module, "weight") and ("Conv" in classname or "Linear" in classname):
        nn.init.normal_(module.weight.data, mean=0.0, std=gain)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif "InstanceNorm" in classname or "BatchNorm" in classname:
        if module.weight is not None:
            nn.init.normal_(module.weight.data, mean=1.0, std=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ---------------------------------------------------------------------------
# Generator  (ResNet9)
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """ResNet-based generator for 256×256 inputs (9 residual blocks).

    Architecture:
        ReflectionPad + Conv 7×7 (in_ch → ngf)
        2× strided Conv 3×3 downsampling (→ 4×ngf)
        9× ResidualBlock (4×ngf)
        2× ConvTranspose 3×3 upsampling (→ ngf)
        ReflectionPad + Conv 7×7 (ngf → out_ch) + tanh
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, ngf: int = 64) -> None:
        super().__init__()

        # Initial convolution
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        ch = ngf
        for _ in range(2):
            layers += [
                nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ch * 2),
                nn.ReLU(inplace=True),
            ]
            ch *= 2

        # Residual blocks
        for _ in range(9):
            layers.append(ResidualBlock(ch))

        # Upsampling
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(ch, ch // 2, kernel_size=3, stride=2,
                                   padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(ch // 2),
                nn.ReLU(inplace=True),
            ]
            ch //= 2

        # Output convolution
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ch, out_channels, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Discriminator  (PatchGAN 70×70)
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    """PatchGAN discriminator with receptive field ≈ 70×70.

    4 Conv-InstanceNorm-LeakyReLU layers + 1 output conv (no sigmoid —
    compatible with LSGAN loss).
    """

    def __init__(self, in_channels: int = 3, ndf: int = 64) -> None:
        super().__init__()

        def _block(in_ch: int, out_ch: int, stride: int = 2,
                   norm: bool = True) -> list[nn.Module]:
            mods: list[nn.Module] = [
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride,
                          padding=1, bias=not norm),
            ]
            if norm:
                mods.append(nn.InstanceNorm2d(out_ch))
            mods.append(nn.LeakyReLU(0.2, inplace=True))
            return mods

        layers: list[nn.Module] = []
        # Layer 1 — no norm
        layers += _block(in_channels, ndf, stride=2, norm=False)
        # Layers 2–4
        layers += _block(ndf,     ndf * 2, stride=2)
        layers += _block(ndf * 2, ndf * 4, stride=2)
        layers += _block(ndf * 4, ndf * 8, stride=1)
        # Output conv
        layers.append(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
        )

        self.model = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
