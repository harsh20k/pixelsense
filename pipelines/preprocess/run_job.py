"""Launch PixelSense preprocessing as a SageMaker Processing Job.

Usage:
    python pipelines/preprocess/run_job.py \\
        --env dev \\
        --instance-type ml.m5.large

Requires:
    - AWS credentials with SageMaker + S3 permissions
    - Terraform-provisioned buckets (pixelsense-data, pixelsense-artifacts)
"""

from __future__ import annotations

import argparse
import logging

import boto3
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

SKLEARN_VERSION = "1.2-1"
SCRIPT_PATH = "pipelines/preprocess/preprocess.py"


def get_execution_role(role_arn: str | None) -> str:
    if role_arn:
        return role_arn
    try:
        return sagemaker.get_execution_role()
    except Exception:
        raise RuntimeError(
            "Could not determine SageMaker execution role. "
            "Pass --role-arn explicitly when running outside SageMaker."
        )


def build_processor(
    role: str,
    instance_type: str,
    instance_count: int,
    sagemaker_session: sagemaker.Session,
) -> SKLearnProcessor:
    return SKLearnProcessor(
        framework_version=SKLEARN_VERSION,
        role=role,
        instance_type=instance_type,
        instance_count=instance_count,
        sagemaker_session=sagemaker_session,
        base_job_name="pixelsense-preprocess",
    )


def run(args: argparse.Namespace) -> None:
    boto_session = boto3.Session(region_name=args.region)
    sm_session = sagemaker.Session(boto_session=boto_session)

    role = get_execution_role(args.role_arn)

    data_bucket = "pixelsense-data"
    artifacts_bucket = "pixelsense-artifacts"

    raw_pixel_art_s3 = f"s3://{data_bucket}/raw/pixel_art/"
    raw_photo_s3 = f"s3://{data_bucket}/raw/photorealistic/"
    processed_s3 = f"s3://{artifacts_bucket}/processed/{args.env}/"

    processor = build_processor(
        role=role,
        instance_type=args.instance_type,
        instance_count=1,
        sagemaker_session=sm_session,
    )

    log.info("Launching SageMaker Processing Job…")
    log.info("  data bucket     : %s", data_bucket)
    log.info("  artifacts bucket: %s", artifacts_bucket)
    log.info("  instance type   : %s", args.instance_type)

    processor.run(
        code=SCRIPT_PATH,
        inputs=[
            ProcessingInput(
                source=raw_pixel_art_s3,
                destination="/opt/ml/processing/input/pixel_art",
                input_name="pixel_art",
            ),
            ProcessingInput(
                source=raw_photo_s3,
                destination="/opt/ml/processing/input/photorealistic",
                input_name="photorealistic",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=processed_s3,
                output_name="processed",
            ),
        ],
        arguments=[
            "--input-root", "/opt/ml/processing/input",
            "--output-root", "/opt/ml/processing/output",
        ],
        wait=args.wait,
        logs=args.wait,
    )

    if args.wait:
        log.info("Job complete. Output at: %s", processed_s3)
    else:
        log.info("Job submitted (async). Check SageMaker console for status.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch PixelSense SageMaker Processing Job")
    parser.add_argument("--env", default="dev", choices=["dev", "staging", "prod"])
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--instance-type", default="ml.m5.large")
    parser.add_argument("--role-arn", default=None, help="SageMaker execution role ARN.")
    parser.add_argument(
        "--no-wait",
        dest="wait",
        action="store_false",
        help="Submit job and return without waiting for completion.",
    )
    parser.set_defaults(wait=True)
    run(parser.parse_args())
