"""Paired dataset for Pix2Pix training.

NpyPairedDataset loads pixel_art.npy and photorealistic.npy and returns
index-matched (A, B) pairs. Since the dataset is inherently unpaired,
pairing by index is the best available approximation for the Pix2Pix baseline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NpyPairedDataset(Dataset):
    """Paired dataset backed by pre-processed .npy files.

    Returns index-matched (domain_a[i], domain_b[i]) pairs.
    If domains have different lengths, the shorter one is truncated.

    Args:
        data_dir:  directory containing pixel_art.npy and photorealistic.npy.
        domain_a:  filename stem for domain A (default pixel_art).
        domain_b:  filename stem for domain B (default photorealistic).
        augment:   if True, apply random horizontal flip (same flip to both).
    """

    def __init__(
        self,
        data_dir: str | Path,
        domain_a: str = "pixel_art",
        domain_b: str = "photorealistic",
        augment: bool = True,
    ) -> None:
        data_dir = Path(data_dir)
        self.augment = augment

        arr_a: np.ndarray = np.load(data_dir / f"{domain_a}.npy")
        arr_b: np.ndarray = np.load(data_dir / f"{domain_b}.npy")

        # Truncate to the shorter domain so indices always align
        n = min(len(arr_a), len(arr_b))
        self.data_a = arr_a[:n]
        self.data_b = arr_b[:n]

    def __len__(self) -> int:
        return len(self.data_a)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.data_a[idx]   # (H, W, 3) float32 [-1, 1]
        b = self.data_b[idx]

        a_t = torch.from_numpy(a).permute(2, 0, 1)
        b_t = torch.from_numpy(b).permute(2, 0, 1)

        if self.augment and torch.rand(1).item() > 0.5:
            # Same flip applied to both to preserve pairing geometry
            a_t = torch.flip(a_t, dims=[2])
            b_t = torch.flip(b_t, dims=[2])

        return a_t, b_t
