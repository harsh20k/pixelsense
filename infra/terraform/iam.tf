# ── IAM Role: SageMaker Execution ─────────────────────────────────────────────

data "aws_iam_policy_document" "sagemaker_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker_execution" {
  name               = "${var.project}-sagemaker-execution-role-${var.environment}"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume_role.json
}

data "aws_iam_policy_document" "sagemaker_permissions" {
  # S3 bucket access
  statement {
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket"
    ]
    resources = [
      aws_s3_bucket.data.arn,
      "${aws_s3_bucket.data.arn}/*",
      aws_s3_bucket.artifacts.arn,
      "${aws_s3_bucket.artifacts.arn}/*"
    ]
  }

  # CloudWatch Logs
  statement {
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams"
    ]
    resources = ["arn:aws:logs:*:*:log-group:/aws/sagemaker/*"]
  }

  # ECR access (for pulling SageMaker built-in algorithm containers)
  statement {
    actions = [
      "ecr:GetAuthorizationToken",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage"
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "sagemaker_permissions" {
  name   = "${var.project}-sagemaker-permissions-${var.environment}"
  role   = aws_iam_role.sagemaker_execution.id
  policy = data.aws_iam_policy_document.sagemaker_permissions.json
}