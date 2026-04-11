# PixelSense

Production-grade **bidirectional pixel art ↔ photorealistic image translation** with model evaluation, edge deployment, and full MLOps on [AWS SageMaker](https://aws.amazon.com/sagemaker/).

## Overview

PixelSense is an ML system that translates images in both directions between pixel art and photorealistic styles. The goal is a production-ready pipeline: rigorous evaluation, reproducible data and training, monitored serving, and paths to cloud and edge inference.

## Stack

| Area | Choice |
|------|--------|
| **Model** | CycleGAN (PyTorch), selected via comparison with Pix2Pix and Stable Diffusion img2img |
| **Evaluation** | FID, SSIM, inference latency benchmarks |
| **Training** | AWS SageMaker (`ml.g4dn.xlarge`) |
| **MLOps** | SageMaker Pipelines, MLflow, SageMaker Model Monitor |
| **Data** | DVC + versioned S3 buckets |
| **Infrastructure** | Terraform |
| **Application** | React frontend + FastAPI backend on EC2 |
| **CI/CD** | GitHub Actions |
| **Edge** | ONNX + CoreML (iOS / Apple Silicon) · TensorRT (NVIDIA Jetson) |

## Project tracking

Requirements, milestones, and delivery are tracked in Linear: [PixelSense project](https://linear.app/masterchief-life/project/pixelsense-cdb3a72a8762).

## Repository status

This repository is the home for PixelSense application code, infrastructure, and ML assets as they land. Clone and follow Linear / forthcoming docs for setup once modules are added.
