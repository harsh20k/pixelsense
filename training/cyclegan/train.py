"""CycleGAN SageMaker training entrypoint.

SageMaker injects hyperparameters as CLI args and mounts data channels at:
    $SM_CHANNEL_TRAIN   — train split (pixel_art.npy + photorealistic.npy)
    $SM_MODEL_DIR       — where to write model artifacts

MLflow tracking URI is passed as a hyperparameter (--mlflow-tracking-uri).
Pass an empty string or omit it to skip MLflow logging (e.g. local runs).

Usage (local smoke-test):
    python training/cyclegan/train.py \
        --data-dir data/processed/train \
        --model-dir /tmp/cyclegan_out \
        --epochs 2 \
        --batch-size 1
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import NpyUnpairedDataset
from losses import LSGANLoss, cycle_loss, identity_loss
from model import Discriminator, Generator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Try importing MLflow; gracefully skip if unavailable
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


# ---------------------------------------------------------------------------
# Image buffer — reduces model oscillation by replaying past generated images
# ---------------------------------------------------------------------------

class ImageBuffer:
    """Stores up to ``capacity`` generated images; returns a mix of stored
    and fresh images to update discriminators (Shrivastava et al. 2017)."""

    def __init__(self, capacity: int = 50) -> None:
        self.capacity = capacity
        self.buffer: list[torch.Tensor] = []

    def push_and_pop(self, images: torch.Tensor) -> torch.Tensor:
        if self.capacity == 0:
            return images
        out: list[torch.Tensor] = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.buffer) < self.capacity:
                self.buffer.append(img)
                out.append(img)
            elif torch.rand(1).item() > 0.5:
                idx = int(torch.randint(0, len(self.buffer), (1,)))
                swapped = self.buffer[idx].clone()
                self.buffer[idx] = img
                out.append(swapped)
            else:
                out.append(img)
        return torch.cat(out, dim=0)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # --- MLflow setup ---
    use_mlflow = MLFLOW_AVAILABLE and bool(args.mlflow_tracking_uri)
    if use_mlflow:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment("pixelsense-cyclegan")
        mlflow.start_run(run_name=f"cyclegan-e{args.epochs}-b{args.batch_size}")
        mlflow.log_params(vars(args))
        log.info("MLflow run started at %s", args.mlflow_tracking_uri)
    elif not MLFLOW_AVAILABLE:
        log.warning("mlflow not installed — skipping experiment tracking")

    # --- Data ---
    dataset = NpyUnpairedDataset(args.data_dir, augment=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    log.info("Dataset size: %d  |  batches/epoch: %d", len(dataset), len(loader))

    # --- Models ---
    G_AB = Generator().to(device)   # pixel_art → photorealistic
    G_BA = Generator().to(device)   # photorealistic → pixel_art
    D_A  = Discriminator().to(device)
    D_B  = Discriminator().to(device)

    # --- Optimisers (Adam with linear LR decay after half-way) ---
    opt_G = torch.optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()),
        lr=args.lr, betas=(0.5, 0.999),
    )
    opt_D_A = torch.optim.Adam(D_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D_B = torch.optim.Adam(D_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

    def _lr_lambda(epoch: int) -> float:
        decay_start = args.epochs // 2
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / (args.epochs - decay_start))

    schedulers = [
        torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
        for opt in (opt_G, opt_D_A, opt_D_B)
    ]

    # --- Loss & buffers ---
    gan_loss = LSGANLoss()
    buf_A = ImageBuffer(50)
    buf_B = ImageBuffer(50)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # --- Epoch loop ---
    for epoch in range(1, args.epochs + 1):
        G_AB.train(); G_BA.train(); D_A.train(); D_B.train()

        epoch_losses: dict[str, float] = {
            "G_total": 0.0, "G_adv": 0.0,
            "cycle": 0.0, "identity": 0.0,
            "D_A": 0.0, "D_B": 0.0,
        }

        for i, (real_A, real_B) in enumerate(loader):
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # ---- Generator update ----------------------------------------
            # Disable D grads to save compute during G step
            for p in D_A.parameters(): p.requires_grad_(False)
            for p in D_B.parameters(): p.requires_grad_(False)

            opt_G.zero_grad()

            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)
            rec_A  = G_BA(fake_B)
            rec_B  = G_AB(fake_A)
            idt_A  = G_BA(real_A)
            idt_B  = G_AB(real_B)

            loss_adv_G = (
                gan_loss(D_B(fake_B), is_real=True)
                + gan_loss(D_A(fake_A), is_real=True)
            )
            loss_cycle = cycle_loss(real_A, rec_A) + cycle_loss(real_B, rec_B)
            loss_idt   = identity_loss(real_A, idt_A) + identity_loss(real_B, idt_B)
            loss_G     = loss_adv_G + loss_cycle + loss_idt

            loss_G.backward()
            opt_G.step()

            # ---- Discriminator A update -----------------------------------
            for p in D_A.parameters(): p.requires_grad_(True)
            opt_D_A.zero_grad()

            fake_A_buf = buf_A.push_and_pop(fake_A.detach())
            loss_D_A = 0.5 * (
                gan_loss(D_A(real_A), is_real=True)
                + gan_loss(D_A(fake_A_buf), is_real=False)
            )
            loss_D_A.backward()
            opt_D_A.step()

            # ---- Discriminator B update -----------------------------------
            for p in D_B.parameters(): p.requires_grad_(True)
            opt_D_B.zero_grad()

            fake_B_buf = buf_B.push_and_pop(fake_B.detach())
            loss_D_B = 0.5 * (
                gan_loss(D_B(real_B), is_real=True)
                + gan_loss(D_B(fake_B_buf), is_real=False)
            )
            loss_D_B.backward()
            opt_D_B.step()

            # Accumulate for epoch mean
            epoch_losses["G_total"]   += loss_G.item()
            epoch_losses["G_adv"]     += loss_adv_G.item()
            epoch_losses["cycle"]     += loss_cycle.item()
            epoch_losses["identity"]  += loss_idt.item()
            epoch_losses["D_A"]       += loss_D_A.item()
            epoch_losses["D_B"]       += loss_D_B.item()

        # --- End of epoch ---
        n = len(loader)
        mean = {k: v / n for k, v in epoch_losses.items()}
        for sched in schedulers:
            sched.step()

        log.info(
            "Epoch %d/%d  G=%.4f  cyc=%.4f  idt=%.4f  D_A=%.4f  D_B=%.4f",
            epoch, args.epochs,
            mean["G_total"], mean["cycle"], mean["identity"],
            mean["D_A"], mean["D_B"],
        )

        if use_mlflow:
            mlflow.log_metrics({f"train/{k}": v for k, v in mean.items()}, step=epoch)

        # Checkpoint every N epochs and at the end
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = {
                "epoch": epoch,
                "G_AB": G_AB.state_dict(),
                "G_BA": G_BA.state_dict(),
                "D_A": D_A.state_dict(),
                "D_B": D_B.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D_A": opt_D_A.state_dict(),
                "opt_D_B": opt_D_B.state_dict(),
            }
            ckpt_path = model_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            torch.save(ckpt, ckpt_path)
            log.info("Saved checkpoint: %s", ckpt_path)

    # --- Final: save inference-ready generators separately ---
    torch.save(G_AB.state_dict(), model_dir / "G_AB_final.pt")
    torch.save(G_BA.state_dict(), model_dir / "G_BA_final.pt")
    log.info("Saved final generators to %s", model_dir)

    if use_mlflow:
        mlflow.log_artifacts(str(model_dir), artifact_path="models")
        mlflow.end_run()
        log.info("MLflow run ended")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    # SageMaker sets SM_CHANNEL_TRAIN and SM_MODEL_DIR as env vars
    parser = argparse.ArgumentParser(description="Train CycleGAN on PixelSense data")

    parser.add_argument(
        "--data-dir",
        default=os.environ.get("SM_CHANNEL_TRAIN", "data/processed/train"),
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("SM_MODEL_DIR", "/tmp/cyclegan_out"),
    )
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--batch-size",   type=int,   default=1)
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--num-workers",  type=int,   default=4)
    parser.add_argument("--save-every",   type=int,   default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--mlflow-tracking-uri", default="",
                        help="MLflow tracking URI; leave empty to disable")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
