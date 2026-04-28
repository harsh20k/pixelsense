"""Stable Diffusion img2img baseline for PixelSense.

Uses a pretrained SD 2.1 pipeline (no fine-tuning) to translate the test-set
pixel art images to photorealistic style. Outputs are saved as .npy arrays
(N, 256, 256, 3) float32 [-1, 1] — same format as the preprocessing pipeline
— so they can be fed directly into evaluation/evaluate.py.

Usage:
    python training/sd_img2img/translate.py \
        --test-dir   data/processed/test \
        --output-dir results/sd_img2img \
        --strength   0.6 \
        --guidance-scale 7.5 \
        --prompt "a high quality photorealistic photograph"

    # With MLflow:
    python training/sd_img2img/translate.py \
        --test-dir   data/processed/test \
        --output-dir results/sd_img2img \
        --mlflow-tracking-uri http://localhost:5000

Dependencies:
    pip install diffusers>=0.28 accelerate>=0.30 transformers
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from diffusers import StableDiffusionImg2ImgPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    log.warning("diffusers not installed — install with: pip install diffusers accelerate")


MODEL_ID = "stabilityai/stable-diffusion-2-1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def npy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert [-1, 1] float32 (H, W, 3) array to uint8 PIL image."""
    uint8 = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(uint8, mode="RGB")


def pil_to_npy(img: Image.Image, size: int = 256) -> np.ndarray:
    """Convert PIL image to [-1, 1] float32 (H, W, 3) numpy array."""
    img = img.convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    return arr


# ---------------------------------------------------------------------------
# Main translation function
# ---------------------------------------------------------------------------

def translate(args: argparse.Namespace) -> None:
    if not DIFFUSERS_AVAILABLE:
        raise ImportError(
            "diffusers is required. Install with: pip install diffusers>=0.28 accelerate>=0.30"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    use_mlflow = MLFLOW_AVAILABLE and bool(args.mlflow_tracking_uri)
    if use_mlflow:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment("pixelsense-sd-img2img")
        mlflow.start_run(run_name="sd-img2img-translation")
        mlflow.log_params({
            "model_id":       MODEL_ID,
            "prompt":         args.prompt,
            "strength":       args.strength,
            "guidance_scale": args.guidance_scale,
            "num_steps":      args.num_inference_steps,
        })

    # Load pipeline
    log.info("Loading SD pipeline: %s", MODEL_ID)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # Load test pixel art images
    test_dir = Path(args.test_dir)
    pixel_art_path = test_dir / "pixel_art.npy"
    if not pixel_art_path.exists():
        raise FileNotFoundError(f"No pixel_art.npy found in {test_dir}")

    pixel_art_arr: np.ndarray = np.load(pixel_art_path)   # (N, 256, 256, 3) [-1, 1]
    log.info("Loaded %d test images from %s", len(pixel_art_arr), pixel_art_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    translated: list[np.ndarray] = []
    latencies: list[float] = []

    generator = torch.Generator(device=device).manual_seed(args.seed)

    for i, raw in enumerate(pixel_art_arr):
        pil_in = npy_to_pil(raw)

        t0 = time.perf_counter()
        result = pipe(
            prompt=args.prompt,
            image=pil_in,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        pil_out = result.images[0]
        npy_out = pil_to_npy(pil_out, size=256)
        translated.append(npy_out)

        if (i + 1) % 5 == 0 or i == len(pixel_art_arr) - 1:
            log.info("  [%d/%d] %.0f ms", i + 1, len(pixel_art_arr), elapsed_ms)

    # Save as .npy (same format as other models)
    out_arr = np.stack(translated, axis=0).astype(np.float32)
    npy_out_path = output_dir / "translated.npy"
    np.save(npy_out_path, out_arr)
    log.info("Saved translated.npy  shape=%s  to %s", out_arr.shape, output_dir)

    # Save latency stats
    latencies_arr = np.array(latencies)
    stats = {
        "model":               "sd_img2img",
        "model_id":            MODEL_ID,
        "prompt":              args.prompt,
        "strength":            args.strength,
        "guidance_scale":      args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "n_images":            len(translated),
        "latency_ms_mean":     float(np.mean(latencies_arr)),
        "latency_ms_p50":      float(np.percentile(latencies_arr, 50)),
        "latency_ms_p95":      float(np.percentile(latencies_arr, 95)),
    }
    stats_path = output_dir / "latency_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    log.info("Latency: mean=%.0f ms  p95=%.0f ms", stats["latency_ms_mean"], stats["latency_ms_p95"])

    if use_mlflow:
        mlflow.log_metrics({
            "latency_ms_mean": stats["latency_ms_mean"],
            "latency_ms_p50":  stats["latency_ms_p50"],
            "latency_ms_p95":  stats["latency_ms_p95"],
        })
        mlflow.log_artifact(str(stats_path))
        mlflow.log_artifact(str(npy_out_path))
        mlflow.end_run()
        log.info("MLflow run ended")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SD img2img inference for PixelSense")
    parser.add_argument("--test-dir",   required=True, help="Path to processed test split dir")
    parser.add_argument("--output-dir", default="results/sd_img2img")
    parser.add_argument("--prompt",     default="a high quality photorealistic photograph, detailed, natural lighting")
    parser.add_argument("--strength",          type=float, default=0.6,
                        help="How much to transform (0=no change, 1=ignore input)")
    parser.add_argument("--guidance-scale",    type=float, default=7.5)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--mlflow-tracking-uri", default="")
    translate(parser.parse_args())
