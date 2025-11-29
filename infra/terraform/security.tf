# Terraform configuration for enhanced security
# KMS, secrets encryption, and hardened networking

resource "aws_kms_key" "polymorph" {
  description             = "POLYMORPH_LITE v4.0 encryption key"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  
  tags = {
    Name        = "polymorph-lite-kms"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

resource "aws_kms_alias" "polymorph" {
  name          = "alias/polymorph-lite"
  target_key_id = aws_kms_key.polymorph.key_id
}

# Secrets Manager for application secrets
resource "aws_secretsmanager_secret" "app_secrets" {
  name                    = "polymorph-lite/${var.environment}/app-secrets"
  description             = "Application secrets for POLYMORPH_LITE"
  kms_key_id              = aws_kms_key.polymorph.arn
  recovery_window_in_days = 30

  tags = {
    Name        = "polymorph-app-secrets"
    Environment = var.environment
  }
}

resource "aws_secretsmanager_secret_version" "app_secrets" {
  secret_id = aws_secretsmanager_secret.app_secrets.id
  secret_string = jsonencode({
    jwt_secret_key     = var.jwt_secret_key
    postgres_password  = var.postgres_password
    redis_password     = var.redis_password
    admin_password     = var.admin_password
  })
}

# IAM role for EKS pods to access secrets
resource "aws_iam_role" "pod_secrets_access" {
  name = "polymorph-pod-secrets-access"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRoleWithWebIdentity"
      Effect = "Allow"
      Principal = {
        Federated = aws_iam_openid_connect_provider.eks.arn
      }
      Condition = {
        StringEquals = {
          "${replace(aws_iam_openid_connect_provider.eks.url, "https://", "")}:sub" = "system:serviceaccount:polymorph-lite:polymorph-sa"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "pod_secrets_policy" {
  name = "pod-secrets-policy"
  role = aws_iam_role.pod_secrets_access.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ]
      Resource = aws_secretsmanager_secret.app_secrets.arn
    },
    {
      Effect = "Allow"
      Action = [
        "kms:Decrypt",
        "kms:DescribeKey"
      ]
      Resource = aws_kms_key.polymorph.arn
    }]
  })
}

# Security group for RDS with restricted access
resource "aws_security_group" "rds" {
  name_description = "Security group for POLYMORPH RDS instance"
  vpc_id          = aws_vpc.main.id

  # Only allow ingress from EKS nodes
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_nodes.id]
    description     = "PostgreSQL from EKS nodes only"
  }

  # No public ingress allowed
  # No egress rules (database doesn't need outbound)
  
  tags = {
    Name = "polymorph-rds-sg"
  }
}

# Network ACL for database subnet
resource "aws_network_acl" "database" {
  vpc_id     = aws_vpc.main.id
  subnet_ids = aws_subnet.database[*].id

  # Ingress from private subnets only
  ingress {
    protocol   = "tcp"
    rule_no    = 100
    action     = "allow"
    cidr_block = "10.0.0.0/16"  # VPC CIDR
    from_port  = 5432
    to_port    = 5432
  }

  # Egress to private subnets only
  egress {
    protocol   = "tcp"
    rule_no    = 100
    action     = "allow"
    cidr_block = "10.0.0.0/16"
    from_port  = 1024
    to_port    = 65535
  }

  tags = {
    Name = "polymorph-database-nacl"
  }
}

# S3 bucket for backups with encryption
resource "aws_s3_bucket" "backups" {
  bucket = "polymorph-lite-backups-${var.environment}"

  tags = {
    Name        = "polymorph-backups"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.polymorph.arn
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "backups" {
  bucket = aws_s3_bucket.backups.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "backups" {
  bucket = aws_s3_bucket.backups.id

  versioning_configuration {
    status = "Enabled"
  }
}

# VPC Flow Logs for network monitoring
resource "aws_flow_log" "main" {
  iam_role_arn    = aws_iam_role.flow_logs.arn
  log_destination = aws_cloudwatch_log_group.flow_logs.arn
  traffic_type    = "ALL"
  vpc_id          = aws_vpc.main.id

  tags = {
    Name = "polymorph-vpc-flow-logs"
  }
}

resource "aws_cloudwatch_log_group" "flow_logs" {
  name              = "/aws/vpc/polymorph-lite"
  retention_in_days = 30
  kms_key_id        = aws_kms_key.polymorph.arn
}

resource "aws_iam_role" "flow_logs" {
  name = "polymorph-vpc-flow-logs"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "vpc-flow-logs.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "flow_logs" {
  name = "flow-logs-policy"
  role = aws_iam_role.flow_logs.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogGroups",
        "logs:DescribeLogStreams"
      ]
      Resource = "*"
    }]
  })
}
