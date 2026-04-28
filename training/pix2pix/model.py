"""Pix2Pix model components.

Generator:     U-Net with 8 encoder/decoder blocks + skip connections (256x256 input).
Discriminator: PatchGAN 70x70 — identical to CycleGAN discriminator.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def init_weights(module: nn.Module, gain: float = 0.02) -> None:
    classname = module.__class__.__name__
    if hasattr(module, "weight") and ("Conv" in classname or "Linear" in classname):
        nn.init.normal_(module.weight.data, mean=0.0, std=gain)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif "BatchNorm" in classname or "InstanceNorm" in classname:
        if module.weight is not None:
            nn.init.normal_(module.weight.data, mean=1.0, std=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)


# ---------------------------------------------------------------------------
# U-Net blocks
# ---------------------------------------------------------------------------

class UNetDown(nn.Module):
    """Encoder block: Conv → [Norm] → LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int, normalize: bool = True,
                 dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetUp(nn.Module):
    """Decoder block: ConvTranspose → Norm → ReLU [→ Dropout], then cat skip."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.block(x), skip], dim=1)


# ---------------------------------------------------------------------------
# Generator — U-Net 256
# ---------------------------------------------------------------------------

class UNetGenerator(nn.Module):
    """U-Net generator for 256×256 images (8 encoder + 8 decoder blocks).

    Input/output: (B, 3, 256, 256), values in [-1, 1].
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3) -> None:
        super().__init__()

        # Encoder — no norm on first block
        self.d1 = UNetDown(in_channels,  64,  normalize=False)   # 128
        self.d2 = UNetDown(64,  128)                              # 64
        self.d3 = UNetDown(128, 256)                              # 32
        self.d4 = UNetDown(256, 512, dropout=0.0)                 # 16
        self.d5 = UNetDown(512, 512, dropout=0.0)                 # 8
        self.d6 = UNetDown(512, 512, dropout=0.0)                 # 4
        self.d7 = UNetDown(512, 512, dropout=0.0)                 # 2
        self.d8 = UNetDown(512, 512, normalize=False)             # 1 (bottleneck)

        # Decoder — skip connections double input channels
        self.u1 = UNetUp(512,  512, dropout=0.5)   # in=512, out: 512+512=1024 after cat
        self.u2 = UNetUp(1024, 512, dropout=0.5)   # 512+512=1024
        self.u3 = UNetUp(1024, 512, dropout=0.5)   # 512+512=1024
        self.u4 = UNetUp(1024, 512)                # 512+512=1024
        self.u5 = UNetUp(1024, 256)                # 256+256=512
        self.u6 = UNetUp(512,  128)                # 128+128=256
        self.u7 = UNetUp(256,   64)                # 64+64=128

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        u1 = self.u1(d8, d7)
        u2 = self.u2(u1, d6)
        u3 = self.u3(u2, d5)
        u4 = self.u4(u3, d4)
        u5 = self.u5(u4, d3)
        u6 = self.u6(u5, d2)
        u7 = self.u7(u6, d1)

        return self.final(u7)


# ---------------------------------------------------------------------------
# Discriminator — PatchGAN 70x70  (same as CycleGAN)
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    """PatchGAN discriminator, conditioned on both input and target (cGAN).

    Receives concatenated [input | target] as a 6-channel tensor.
    """

    def __init__(self, in_channels: int = 6, ndf: int = 64) -> None:
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
        layers += _block(in_channels, ndf,     stride=2, norm=False)
        layers += _block(ndf,         ndf * 2, stride=2)
        layers += _block(ndf * 2,     ndf * 4, stride=2)
        layers += _block(ndf * 4,     ndf * 8, stride=1)
        layers.append(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, real_or_fake: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([condition, real_or_fake], dim=1))
