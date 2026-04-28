"""Dataset for CycleGAN unpaired training.

NpyUnpairedDataset loads the pixel_art.npy and photorealistic.npy shards
written by the preprocessing pipeline (shape N×256×256×3, float32, [-1,1]).
Domain A and Domain B are shuffled independently so pairs are unpaired by
design, matching the CycleGAN assumption.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NpyUnpairedDataset(Dataset):
    """Unpaired dataset backed by pre-processed .npy files.

    Args:
        data_dir:   directory containing ``pixel_art.npy`` and
                    ``photorealistic.npy`` (e.g. the ``train/`` split).
        domain_a:   filename stem for domain A (default ``pixel_art``).
        domain_b:   filename stem for domain B (default ``photorealistic``).
        augment:    if True, apply random horizontal flip (p=0.5).
        seed:       RNG seed used for independent domain shuffles.
    """

    def __init__(
        self,
        data_dir: str | Path,
        domain_a: str = "pixel_art",
        domain_b: str = "photorealistic",
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        data_dir = Path(data_dir)
        self.augment = augment

        arr_a: np.ndarray = np.load(data_dir / f"{domain_a}.npy")  # (N, H, W, 3)
        arr_b: np.ndarray = np.load(data_dir / f"{domain_b}.npy")

        rng = np.random.default_rng(seed)
        idx_a = rng.permutation(len(arr_a))
        idx_b = rng.permutation(len(arr_b))

        self.data_a: np.ndarray = arr_a[idx_a]
        self.data_b: np.ndarray = arr_b[idx_b]

        # Dataset length = max of the two domains; shorter domain wraps around
        self.len_a = len(self.data_a)
        self.len_b = len(self.data_b)
        self._len = max(self.len_a, self.len_b)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.data_a[idx % self.len_a]   # (H, W, 3) float32 [-1,1]
        b = self.data_b[idx % self.len_b]

        # HWC → CHW
        a_t = torch.from_numpy(a).permute(2, 0, 1)
        b_t = torch.from_numpy(b).permute(2, 0, 1)

        if self.augment and torch.rand(1).item() > 0.5:
            a_t = torch.flip(a_t, dims=[2])
            b_t = torch.flip(b_t, dims=[2])

        return a_t, b_t
