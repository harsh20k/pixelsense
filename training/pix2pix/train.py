"""Pix2Pix SageMaker training entrypoint.

Trains a conditional GAN (cGAN) with LSGAN adversarial loss + L1 pixel
reconstruction loss (lambda_L1=100) in one direction: A (pixel_art) -> B (photorealistic).

SageMaker channels:
    $SM_CHANNEL_TRAIN   — train split (pixel_art.npy + photorealistic.npy)
    $SM_MODEL_DIR       — checkpoint + final model output

Usage (local smoke-test):
    python training/pix2pix/train.py \
        --data-dir data/processed/train \
        --model-dir /tmp/pix2pix_out \
        --epochs 2 \
        --batch-size 1
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import NpyPairedDataset
from model import Discriminator, UNetGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


# ---------------------------------------------------------------------------
# LSGAN helper
# ---------------------------------------------------------------------------

def lsgan_loss(pred: torch.Tensor, is_real: bool) -> torch.Tensor:
    target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
    return F.mse_loss(pred, target)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    use_mlflow = MLFLOW_AVAILABLE and bool(args.mlflow_tracking_uri)
    if use_mlflow:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment("pixelsense-pix2pix")
        mlflow.start_run(run_name=f"pix2pix-e{args.epochs}-b{args.batch_size}")
        mlflow.log_params(vars(args))
        log.info("MLflow run started at %s", args.mlflow_tracking_uri)
    elif not MLFLOW_AVAILABLE:
        log.warning("mlflow not installed — skipping experiment tracking")

    dataset = NpyPairedDataset(args.data_dir, augment=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    log.info("Dataset size: %d  |  batches/epoch: %d", len(dataset), len(loader))

    G = UNetGenerator().to(device)
    D = Discriminator().to(device)   # conditioned: in_channels=6

    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    def _lr_lambda(epoch: int) -> float:
        decay_start = args.epochs // 2
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / (args.epochs - decay_start))

    schedulers = [
        torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
        for opt in (opt_G, opt_D)
    ]

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()

        epoch_losses: dict[str, float] = {"G_total": 0.0, "G_adv": 0.0, "G_l1": 0.0, "D": 0.0}

        for real_A, real_B in loader:
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            fake_B = G(real_A)

            # ---- Discriminator update ----
            for p in D.parameters(): p.requires_grad_(True)
            opt_D.zero_grad()

            loss_D_real = lsgan_loss(D(real_B, real_A), is_real=True)
            loss_D_fake = lsgan_loss(D(fake_B.detach(), real_A), is_real=False)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            opt_D.step()

            # ---- Generator update ----
            for p in D.parameters(): p.requires_grad_(False)
            opt_G.zero_grad()

            loss_G_adv = lsgan_loss(D(fake_B, real_A), is_real=True)
            loss_G_l1  = args.lambda_l1 * F.l1_loss(fake_B, real_B)
            loss_G     = loss_G_adv + loss_G_l1
            loss_G.backward()
            opt_G.step()

            epoch_losses["G_total"] += loss_G.item()
            epoch_losses["G_adv"]   += loss_G_adv.item()
            epoch_losses["G_l1"]    += loss_G_l1.item()
            epoch_losses["D"]       += loss_D.item()

        n = len(loader)
        mean = {k: v / n for k, v in epoch_losses.items()}
        for sched in schedulers:
            sched.step()

        log.info(
            "Epoch %d/%d  G=%.4f  adv=%.4f  L1=%.4f  D=%.4f",
            epoch, args.epochs, mean["G_total"], mean["G_adv"], mean["G_l1"], mean["D"],
        )

        if use_mlflow:
            mlflow.log_metrics({f"train/{k}": v for k, v in mean.items()}, step=epoch)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = {
                "epoch": epoch,
                "G": G.state_dict(),
                "D": D.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
            }
            ckpt_path = model_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            torch.save(ckpt, ckpt_path)
            log.info("Saved checkpoint: %s", ckpt_path)

    torch.save(G.state_dict(), model_dir / "G_final.pt")
    log.info("Saved final generator to %s", model_dir)

    if use_mlflow:
        mlflow.log_artifacts(str(model_dir), artifact_path="models")
        mlflow.end_run()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Pix2Pix on PixelSense data")
    parser.add_argument("--data-dir",  default=os.environ.get("SM_CHANNEL_TRAIN", "data/processed/train"))
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/tmp/pix2pix_out"))
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--batch-size", type=int,   default=1)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--lambda-l1",  type=float, default=100.0)
    parser.add_argument("--num-workers", type=int,  default=4)
    parser.add_argument("--save-every",  type=int,  default=10)
    parser.add_argument("--mlflow-tracking-uri", default="")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
