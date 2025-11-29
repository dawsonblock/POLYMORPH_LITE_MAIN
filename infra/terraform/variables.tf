variable "region" {
  description = "AWS Region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "polymorph-lite"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "eks_instance_types" {
  description = "EC2 instance types for EKS nodes"
  type        = list(string)
  default     = ["t3.medium"]
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "polymorph"
}

variable "db_username" {
  description = "Database master username"
  type        = string
  default     = "polymorph_admin"
}

variable "db_password" {
  description = "Database master password (pass via env var or secrets manager)"
  type        = string
  sensitive   = true
  default     = "ChangeMeInProd123!"
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}
