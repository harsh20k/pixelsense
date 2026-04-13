variable "aws_region" {
  description = "AWS region to deploy resources into."
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Project name — used in resource names and tags."
  type        = string
  default     = "pixelsense"
}

variable "environment" {
  description = "Deployment environment (dev / staging / prod)."
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be one of: dev, staging, prod."
  }
}

variable "data_lifecycle_glacier_days" {
  description = "Days before non-current object versions in the data bucket are transitioned to Glacier."
  type        = number
  default     = 90
}
