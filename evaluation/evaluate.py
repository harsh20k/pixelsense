"""Evaluate a single model's outputs: FID, SSIM, and inference latency.

Computes three metrics against the test split ground truth:
  - FID   (Frechet Inception Distance) — lower is better
  - SSIM  (Structural Similarity Index) — higher is better
  - Inference latency (mean / p50 / p95 ms) — lower is better

Outputs a JSON file that compare.py aggregates across models.

Usage:
    # CycleGAN
    python evaluation/evaluate.py \
        --model-name cyclegan \
        --model-type cyclegan \
        --model-path /tmp/cyclegan_out/G_AB_final.pt \
        --test-dir   data/processed/test \
        --output     results/cyclegan_metrics.json

    # Pix2Pix
    python evaluation/evaluate.py \
        --model-name pix2pix \
        --model-type pix2pix \
        --model-path /tmp/pix2pix_out/G_final.pt \
        --test-dir   data/processed/test \
        --output     results/pix2pix_metrics.json

    # SD img2img (no model-path needed — reads pre-translated .npy)
    python evaluation/evaluate.py \
        --model-name sd_img2img \
        --model-type sd_img2img \
        --translated-npy results/sd_img2img/translated.npy \
        --test-dir   data/processed/test \
        --output     results/sd_img2img_metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    log.warning("torchmetrics not installed — install with: pip install torchmetrics[image]>=1.3")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_generator(model_type: str, model_path: str, device: torch.device) -> torch.nn.Module:
    """Load a trained generator for inference."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if model_type == "cyclegan":
        sys.path.insert(0, "training/cyclegan")
        from model import Generator  # type: ignore
        G = Generator()
        G.load_state_dict(torch.load(model_path, map_location=device))

    elif model_type == "pix2pix":
        sys.path.insert(0, "training/pix2pix")
        from model import UNetGenerator  # type: ignore
        G = UNetGenerator()
        G.load_state_dict(torch.load(model_path, map_location=device))

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use cyclegan or pix2pix.")

    G.eval().to(device)
    return G


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def run_inference(
    G: torch.nn.Module,
    inputs: np.ndarray,
    device: torch.device,
    n_latency_runs: int = 20,
) -> tuple[np.ndarray, dict[str, float]]:
    """Run generator on all inputs; also measure inference latency.

    Returns:
        outputs: (N, 256, 256, 3) float32 [-1, 1]
        latency: dict with mean/p50/p95 in ms
    """
    outputs: list[np.ndarray] = []
    latencies: list[float] = []

    with torch.inference_mode():
        for i, arr in enumerate(inputs):
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

            # Time the first n_latency_runs images
            if i < n_latency_runs:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = G(t)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)
            else:
                out = G(t)

            out_arr = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
            outputs.append(out_arr)

    latencies_arr = np.array(latencies)
    latency_stats = {
        "latency_ms_mean": float(np.mean(latencies_arr)),
        "latency_ms_p50":  float(np.percentile(latencies_arr, 50)),
        "latency_ms_p95":  float(np.percentile(latencies_arr, 95)),
    }
    return np.stack(outputs, axis=0), latency_stats


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _to_uint8_tensor(arr: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) float32 [-1,1] -> (N, 3, H, W) uint8 [0,255]."""
    uint8 = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return torch.from_numpy(uint8).permute(0, 3, 1, 2)   # (N, 3, H, W)


def _to_float_tensor(arr: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) float32 [-1,1] -> (N, 3, H, W) float32 [-1,1]."""
    return torch.from_numpy(arr).permute(0, 3, 1, 2)


def compute_fid(real: np.ndarray, fake: np.ndarray, device: torch.device) -> float:
    """Compute FID between real and fake image arrays."""
    if not TORCHMETRICS_AVAILABLE:
        log.warning("torchmetrics unavailable — FID skipped, returning -1")
        return -1.0

    fid_metric = FrechetInceptionDistance(normalize=False).to(device)
    real_t = _to_uint8_tensor(real).to(device)
    fake_t = _to_uint8_tensor(fake).to(device)

    fid_metric.update(real_t, real=True)
    fid_metric.update(fake_t, real=False)
    return float(fid_metric.compute())


def compute_ssim(real: np.ndarray, fake: np.ndarray, device: torch.device) -> float:
    """Compute mean SSIM between real and fake image arrays."""
    if not TORCHMETRICS_AVAILABLE:
        log.warning("torchmetrics unavailable — SSIM skipped, returning -1")
        return -1.0

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    real_t = _to_float_tensor(real).to(device)
    fake_t = _to_float_tensor(fake).to(device)
    return float(ssim_metric(fake_t, real_t))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    test_dir = Path(args.test_dir)
    real_arr: np.ndarray = np.load(test_dir / "photorealistic.npy")
    input_arr: np.ndarray = np.load(test_dir / "pixel_art.npy")
    log.info("Test set: %d real images, %d input images", len(real_arr), len(input_arr))

    # --- Get translated outputs ---
    latency_stats: dict[str, float]

    if args.model_type == "sd_img2img":
        if not args.translated_npy:
            raise ValueError("--translated-npy required for model-type=sd_img2img")
        fake_arr = np.load(args.translated_npy)
        log.info("Loaded pre-translated SD outputs: shape=%s", fake_arr.shape)
        # Latency from the stats file written by translate.py
        stats_file = Path(args.translated_npy).parent / "latency_stats.json"
        if stats_file.exists():
            saved = json.loads(stats_file.read_text())
            latency_stats = {k: saved[k] for k in ("latency_ms_mean", "latency_ms_p50", "latency_ms_p95")}
        else:
            latency_stats = {"latency_ms_mean": -1.0, "latency_ms_p50": -1.0, "latency_ms_p95": -1.0}
    else:
        G = load_generator(args.model_type, args.model_path, device)
        fake_arr, latency_stats = run_inference(G, input_arr, device)

    # Align lengths (SD may have translated fewer images)
    n = min(len(real_arr), len(fake_arr))
    real_arr  = real_arr[:n]
    fake_arr  = fake_arr[:n]

    # --- Compute FID and SSIM ---
    log.info("Computing FID...")
    fid = compute_fid(real_arr, fake_arr, device)
    log.info("FID = %.4f", fid)

    log.info("Computing SSIM...")
    ssim = compute_ssim(real_arr, fake_arr, device)
    log.info("SSIM = %.4f", ssim)

    results = {
        "model":           args.model_name,
        "model_type":      args.model_type,
        "n_test_images":   n,
        "fid":             fid,
        "ssim":            ssim,
        **latency_stats,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    log.info("Results saved to %s", output_path)
    log.info("  FID=%.4f  SSIM=%.4f  latency_mean=%.0f ms",
             fid, ssim, latency_stats["latency_ms_mean"])

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PixelSense model metrics")
    parser.add_argument("--model-name",  required=True, help="Label for this model in results")
    parser.add_argument("--model-type",  required=True, choices=["cyclegan", "pix2pix", "sd_img2img"])
    parser.add_argument("--model-path",  default=None,  help="Path to .pt weights file (not needed for sd_img2img)")
    parser.add_argument("--translated-npy", default=None, help="Path to pre-translated .npy (sd_img2img only)")
    parser.add_argument("--test-dir",    required=True, help="Path to processed test split dir")
    parser.add_argument("--output",      default="results/metrics.json")
    evaluate(parser.parse_args())
