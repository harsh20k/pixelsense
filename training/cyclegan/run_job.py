"""Launch CycleGAN training as a SageMaker PyTorch Training Job.

Usage:
    python training/cyclegan/run_job.py \
        --env dev \
        --epochs 200 \
        --mlflow-tracking-uri http://my-mlflow-server:5000

Requires:
    - AWS credentials with SageMaker + S3 permissions
    - Terraform-provisioned buckets (pixelsense-data, pixelsense-artifacts)
    - IAM execution role with SageMaker + S3 access
"""

from __future__ import annotations

import argparse
import logging

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

PYTORCH_VERSION = "2.2"
PYTHON_VERSION  = "py311"
SCRIPT_PATH     = "training/cyclegan/train.py"
SOURCE_DIR      = "training/cyclegan"

DATA_BUCKET      = "pixelsense-data"
ARTIFACTS_BUCKET = "pixelsense-artifacts"


def get_execution_role(role_arn: str | None, sm_session: sagemaker.Session) -> str:
    if role_arn:
        return role_arn
    try:
        return sagemaker.get_execution_role(sm_session)
    except Exception:
        raise RuntimeError(
            "Could not determine SageMaker execution role. "
            "Pass --role-arn explicitly when running outside SageMaker."
        )


def run(args: argparse.Namespace) -> None:
    boto_session = boto3.Session(region_name=args.region)
    sm_session   = sagemaker.Session(boto_session=boto_session)

    role = get_execution_role(args.role_arn, sm_session)

    processed_s3 = f"s3://{ARTIFACTS_BUCKET}/processed/{args.env}/train"
    output_s3    = f"s3://{ARTIFACTS_BUCKET}/models/cyclegan/{args.env}/"

    hyperparameters: dict[str, str | int | float] = {
        "epochs":              args.epochs,
        "batch-size":          args.batch_size,
        "lr":                  args.lr,
        "save-every":          args.save_every,
        "num-workers":         2,
        "mlflow-tracking-uri": args.mlflow_tracking_uri,
    }

    estimator = PyTorch(
        entry_point="train.py",
        source_dir=SOURCE_DIR,
        role=role,
        instance_type=args.instance_type,
        instance_count=1,
        framework_version=PYTORCH_VERSION,
        py_version=PYTHON_VERSION,
        hyperparameters=hyperparameters,
        output_path=output_s3,
        base_job_name="pixelsense-cyclegan",
        sagemaker_session=sm_session,
        # Keep training artifacts (checkpoints) in S3 even on failure
        keep_alive_period_in_seconds=0,
    )

    log.info("Launching SageMaker PyTorch Training Job")
    log.info("  instance type  : %s", args.instance_type)
    log.info("  processed data : %s", processed_s3)
    log.info("  output path    : %s", output_s3)
    log.info("  epochs         : %d", args.epochs)

    estimator.fit(
        inputs={"train": processed_s3},
        wait=args.wait,
        logs="All" if args.wait else None,
    )

    if args.wait:
        log.info("Training complete. Artifacts at: %s", output_s3)
    else:
        log.info("Job submitted (async). Check SageMaker console for status.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch CycleGAN SageMaker Training Job")

    parser.add_argument("--env",           default="dev", choices=["dev", "staging", "prod"])
    parser.add_argument("--region",        default="us-east-1")
    parser.add_argument("--instance-type", default="ml.g4dn.xlarge")
    parser.add_argument("--role-arn",      default=None)
    parser.add_argument("--epochs",        type=int,   default=200)
    parser.add_argument("--batch-size",    type=int,   default=1)
    parser.add_argument("--lr",            type=float, default=2e-4)
    parser.add_argument("--save-every",    type=int,   default=10)
    parser.add_argument("--mlflow-tracking-uri", default="")
    parser.add_argument(
        "--no-wait",
        dest="wait",
        action="store_false",
        help="Submit job and return without waiting.",
    )
    parser.set_defaults(wait=True)

    run(parser.parse_args())
