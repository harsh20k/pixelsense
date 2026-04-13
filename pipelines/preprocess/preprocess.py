"""SageMaker Processing script — image preprocessing for PixelSense.

Runs inside a SageMaker Processing container. Reads raw images from the two
input channels, applies resize + normalisation (CycleGAN [-1, 1] convention),
performs a deterministic 80/10/10 train/val/test split, and writes NumPy shards
to the output channel.

Input channels (mounted by SageMaker):
    /opt/ml/processing/input/pixel_art/
    /opt/ml/processing/input/photorealistic/

Output channel:
    /opt/ml/processing/output/  →  train/, val/, test/
        Each split contains:
            pixel_art.npy       shape (N, 256, 256, 3)  float32  [-1, 1]
            photorealistic.npy  shape (N, 256, 256, 3)  float32  [-1, 1]
            manifest.json       list of source filenames
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

INPUT_ROOT = Path("/opt/ml/processing/input")
OUTPUT_ROOT = Path("/opt/ml/processing/output")

IMAGE_SIZE = (256, 256)
SEED = 42
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def load_images(folder: Path) -> tuple[list[np.ndarray], list[str]]:
    """Load all images from *folder*, return arrays and filenames."""
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")

    arrays, names = [], []
    for p in paths:
        img = Image.open(p).convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0  # [0,255] → [-1, 1]
        arrays.append(arr)
        names.append(p.name)

    log.info("Loaded %d images from %s", len(arrays), folder)
    return arrays, names


def split_data(
    arrays: list[np.ndarray],
    names: list[str],
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Split into train / val / test deterministically."""
    idx = list(range(len(arrays)))

    train_idx, rest_idx = train_test_split(idx, test_size=1 - SPLITS["train"], random_state=SEED)
    val_size_relative = SPLITS["val"] / (SPLITS["val"] + SPLITS["test"])
    val_idx, test_idx = train_test_split(rest_idx, test_size=1 - val_size_relative, random_state=SEED)

    result: dict[str, tuple[np.ndarray, list[str]]] = {}
    for split_name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        result[split_name] = (
            np.stack([arrays[i] for i in indices]),
            [names[i] for i in indices],
        )
        log.info("Split %s: %d images", split_name, len(indices))

    return result


def save_split(
    split_name: str,
    domain: str,
    arrays: np.ndarray,
    names: list[str],
    output_root: Path,
) -> None:
    split_dir = output_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    np.save(split_dir / f"{domain}.npy", arrays)
    (split_dir / f"{domain}_manifest.json").write_text(json.dumps(names, indent=2))


def main(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    domains = {
        "pixel_art": input_root / "pixel_art",
        "photorealistic": input_root / "photorealistic",
    }

    split_data_map: dict[str, dict[str, tuple[np.ndarray, list[str]]]] = {}
    for domain, folder in domains.items():
        arrays, names = load_images(folder)
        split_data_map[domain] = split_data(arrays, names)

    for domain, splits in split_data_map.items():
        for split_name, (arr, names) in splits.items():
            save_split(split_name, domain, arr, names, output_root)
            log.info("Saved %s/%s.npy  shape=%s", split_name, domain, arr.shape)

    log.info("Preprocessing complete. Output at %s", output_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PixelSense image preprocessing")
    parser.add_argument(
        "--input-root",
        default=str(INPUT_ROOT),
        help="Root dir containing pixel_art/ and photorealistic/ sub-dirs.",
    )
    parser.add_argument(
        "--output-root",
        default=str(OUTPUT_ROOT),
        help="Root dir to write train/val/test splits.",
    )
    main(parser.parse_args())
