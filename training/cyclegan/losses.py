"""CycleGAN loss functions.

LSGAN adversarial loss (Mao et al. 2017) — uses MSE rather than BCE,
giving more stable gradients early in training.

Cycle-consistency loss  λ=10  (default from the original paper).
Identity loss           λ_id=5.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSGANLoss(nn.Module):
    """Least-squares GAN adversarial loss.

    ``target_real_label=1`` and ``target_fake_label=0`` follow the standard
    convention.  The discriminator tries to push real→1, fake→0; the
    generator tries to push its fakes→1.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:  # type: ignore[override]
        target = torch.ones_like(prediction) if is_real else torch.zeros_like(prediction)
        return F.mse_loss(prediction, target)


def cycle_loss(
    real: torch.Tensor,
    reconstructed: torch.Tensor,
    lambda_cycle: float = 10.0,
) -> torch.Tensor:
    """||G_BA(G_AB(A)) - A||_1  (or the symmetric term for B)."""
    return lambda_cycle * F.l1_loss(reconstructed, real)


def identity_loss(
    real: torch.Tensor,
    same: torch.Tensor,
    lambda_identity: float = 5.0,
) -> torch.Tensor:
    """||G_AB(B) - B||_1  — penalises unnecessary colour shifts."""
    return lambda_identity * F.l1_loss(same, real)
