"""Validate a preprocessed batch of .npy files with Great Expectations.

Called post-SageMaker Processing Job (or locally after dvc repro):
    python validation/validate_batch.py \\
        --processed-dir /path/to/processed/train

Exits non-zero if any expectation fails — suitable for CI/CD gates and
SageMaker Pipeline condition steps.

The script converts each .npy array into a flat pandas DataFrame that GE can
ingest, with columns:
    pixel_value  — every pixel channel value across all images
    height       — image height (256 for all after preprocessing)
    width        — image width
    channels     — number of channels (3 = RGB)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import great_expectations as gx
import numpy as np
import pandas as pd
from great_expectations.core.batch import RuntimeBatchRequest

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

GX_ROOT = Path(__file__).parent / "gx"
DOMAINS = ["pixel_art", "photorealistic"]


def npy_to_dataframe(npy_path: Path) -> pd.DataFrame:
    """Flatten a (N, H, W, C) array into a validation-friendly DataFrame."""
    arr = np.load(npy_path)

    if arr.ndim != 4:
        raise ValueError(f"Expected 4-D array (N,H,W,C), got shape {arr.shape}")

    n, h, w, c = arr.shape
    pixel_values = arr.flatten()

    return pd.DataFrame({
        "pixel_value": pixel_values,
        "height": np.full(len(pixel_values), h),
        "width": np.full(len(pixel_values), w),
        "channels": np.full(len(pixel_values), c),
    })


def validate(processed_dir: Path, split: str) -> bool:
    split_dir = processed_dir / split
    context = gx.get_context(context_root_dir=str(GX_ROOT))
    checkpoint = context.get_checkpoint("pixelsense_checkpoint")
    validations = []

    for domain in DOMAINS:
        npy_path = split_dir / f"{domain}.npy"
        if not npy_path.exists():
            log.warning("Missing: %s — skipping", npy_path)
            continue

        log.info("Preparing checkpoint batch for %s/%s ...", split, domain)
        df = npy_to_dataframe(npy_path)

        batch_request = RuntimeBatchRequest(
            datasource_name="pixelsense_runtime",
            data_connector_name="runtime_connector",
            data_asset_name=f"{domain}_{split}",
            runtime_parameters={"batch_data": df},
            batch_identifiers={"domain": domain, "split": split},
        )
        validations.append(
            {
                "batch_request": batch_request,
                "expectation_suite_name": f"{domain}_suite",
            }
        )

    if not validations:
        log.warning("No .npy files found for split=%s", split)
        return True

    checkpoint_result = checkpoint.run(validations=validations)
    passed = bool(checkpoint_result["success"])
    status = "PASSED" if passed else "FAILED"
    log.info("Checkpoint %s for split=%s", status, split)
    return passed


def main(args: argparse.Namespace) -> None:
    processed_dir = Path(args.processed_dir)
    splits = args.splits if args.splits else ["train", "val", "test"]

    all_passed = True
    for split in splits:
        ok = validate(processed_dir, split)
        if not ok:
            all_passed = False

    if all_passed:
        log.info("All validations passed.")
        sys.exit(0)
    else:
        log.error("One or more validations FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate preprocessed PixelSense batches")
    parser.add_argument(
        "--processed-dir",
        required=True,
        help="Root directory containing train/, val/, test/ split folders.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=None,
        help="Which splits to validate (default: all three).",
    )
    main(parser.parse_args())
