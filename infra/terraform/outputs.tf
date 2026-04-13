output "data_bucket_name" {
  description = "Name of the S3 bucket used for raw data and DVC storage."
  value       = aws_s3_bucket.data.bucket
}

output "data_bucket_arn" {
  description = "ARN of the data S3 bucket."
  value       = aws_s3_bucket.data.arn
}

output "artifacts_bucket_name" {
  description = "Name of the S3 bucket used for model artifacts and SageMaker outputs."
  value       = aws_s3_bucket.artifacts.bucket
}

output "artifacts_bucket_arn" {
  description = "ARN of the artifacts S3 bucket."
  value       = aws_s3_bucket.artifacts.arn
}

output "dvc_remote_url" {
  description = "DVC remote URL — pass to: dvc remote add -d s3remote <value>"
  value       = "s3://${aws_s3_bucket.data.bucket}/dvc-store"
}
